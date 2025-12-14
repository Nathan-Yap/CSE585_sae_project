

import math
import torch
import triton
import triton.language as tl

# Maybe a circular import
from utils import measure


@triton.jit
def spmm_coo_kernel(
    batch_idx_ptr,      # int32 [nnz]
    row_idx_ptr,      # int32 [nnz]
    col_idx_ptr,      # int32 [nnz]
    vals_ptr,         # float [nnz]
    base_W_ptr,            # float [K, N]
    base_Y_ptr,            # float [B, M, N]
    nnz, K, N,
    stride_w_k, stride_w_n,
    stride_y_b, stride_y_m, stride_y_n,
    BLOCK_E: tl.constexpr,   # number of nonzeros handled per program
    BLOCK_N: tl.constexpr    # width of vectorized N-dimension per load/store
):
    pid = tl.program_id(0)                    # program id (over groups of nonzeros)
    start = pid * BLOCK_E                     # index of first nonzero handled by this program

    # vector over N dimension for this program
    n_off = tl.arange(0, BLOCK_N) # For DEBUG -> 0, 1
    n_stride = tl.cdiv(N, BLOCK_N)
    # mask_n = n_off * n_stride < N   # vector mask for N

    # loop unrolled at compile time
    for i in tl.static_range(0, BLOCK_E):
        e = start + i                # scalar index for the i-th nonzero
        valid = e < nnz              # scalar boolean (will broadcast when combined with mask_n)
        # load scalar index values (use pointer arithmetic with scalar offset)
        batch = tl.load(batch_idx_ptr + e, mask=valid, other=0)   # scalar int32
        row   = tl.load(row_idx_ptr   + e, mask=valid, other=0)
        col   = tl.load(col_idx_ptr   + e, mask=valid, other=0)
        val   = tl.load(vals_ptr   + e, mask=valid, other=0.0)

        for n in range(0, n_stride):
            w_ptr = base_W_ptr + col * stride_w_k + n_off * n_stride + n
            y_ptr = base_Y_ptr + batch * stride_y_b + row * stride_y_m + n_off * n_stride + n

            w_val = tl.load(w_ptr, mask=n_off*n_stride+n < N, other=0.0)

            upd = w_val * val

            # atomic add into Y; combine masks so we don't update when this e is invalid
            # active_mask = mask_n & valid
            tl.atomic_add(y_ptr, upd)


def spmm_coo_triton(sparse_tensor, W, BLOCK_E=64, BLOCK_N=64):
    """
    FIXME: Do we need to track the num of warps?
    Simple driver for the Triton COO SpMM kernel.

    Inputs:
      - sparse_tensor: [B, M, K]
      - W: [K, N] dense matrix (torch.Tensor, contiguous or with strides provided)

    Returns:
      - Y: [B, M, N] dense result (torch.zeros + atomics)
    """
    device = W.device
    dtype = W.dtype

    # Sparse tensor shape
    # torch.Size([16, 128, 24576])
    # Dense tensor shape
    # torch.Size([24576, 768])
    # Output shape
    # torch.Size([16, 128, 768])

    # These are the dimensions for the SAE
    # N = 768
    # M = 128
    # K = 24576
    N = W.shape[1]
    M = sparse_tensor.shape[1]
    K = sparse_tensor.shape[2]
    batch_size = sparse_tensor.shape[0]


    batch_idx = sparse_tensor.indices()[0]
    row_idx = sparse_tensor.indices()[1]
    col_idx = sparse_tensor.indices()[2]

    vals = sparse_tensor.values()

    nnz = batch_idx.shape[0]
    Y = torch.zeros((batch_size, M, N), device=device, dtype=dtype)

    grid = (math.ceil(nnz / BLOCK_E),)

    spmm_coo_kernel[grid](
        batch_idx,
        row_idx,
        col_idx,
        vals,
        W,
        Y,
        nnz, W.shape[0], N,
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_E=BLOCK_E,
        BLOCK_N=BLOCK_N,
    )
    return Y


if __name__ == "__main__":
    # This is code for doing some simple test cases to make sure the kernel is working. Then checking against a normal BMM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SAE Tensor Fakes
    # normal_sparse_tensor = torch.zeros((16, 128, 24576))
    # weight_tensor = torch.ones((24576, 768))

    for sparse_shape, dense_shape, use_sparse, block_e, block_n in [
        ((3,2,5), (5,6), False, 8, 2),
        ((3,2,5), (5,40), False, 8, 8),
        ((8, 128, 3072), (3072, 512), True, 20, 512),
        ((1, 128, 24576), (24576, 768), True, 20, 32)
    ]:
        print(f"Testing sparse tensor of shape: {str(sparse_shape)}, with dense of shape: {str(dense_shape)}, with sparsity: {use_sparse}")
        normal_sparse_tensor = torch.rand(sparse_shape)

        # This is for making the matrices sparse
        if use_sparse:
            mask = torch.rand(normal_sparse_tensor.shape) < 0.001
            normal_sparse_tensor[mask] = 1

        weight_tensor = torch.ones(dense_shape) # Normally 24576

        # print(normal_sparse_tensor)
        # print(weight_tensor)

        output, bmm_time = measure(torch.matmul, normal_sparse_tensor, weight_tensor)

        sparse_tensor = normal_sparse_tensor.to_sparse().to(device)

        weight_tensor = weight_tensor.to(device)

        # Different blocking along the non-zero elements (idxs) and the hidden dim of W matrix

        Y, kernel_time = measure(spmm_coo_triton, sparse_tensor, weight_tensor, block_e, block_n)

        # print("Y")
        # print(Y)
        # print(Y.shape)

        # print("CORRECT")
        # print(output)

        print(f"Output matches normal BMM: {torch.allclose(Y, output.to(device))}")
        print(f"Normal BMM time: {bmm_time}")
        print(f"Triton Kernel time: {kernel_time}")