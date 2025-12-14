# profile_sae.py

import time
import torch
from sae_lens import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------ LOAD SAE ------------------
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device,
)
sae.eval()
print("Loaded SAE")

print(sae)

# ============================================================
#          cuSPARSE DECODER (Sparse × Dense SpMM)
# ============================================================

def cusparse_decode(sparse_tensor, W_dec):
    """Perform sparse (CSR) × dense decode using cuSPARSE (torch.sparse.mm)."""
    result_list = []
    for i in range(32):
        # sparse @ dense
        result_list.append(torch.sparse.mm(sparse_tensor[i], W_dec))  # [128, 768]

    # Stack results to get [32, 128, 768]
    return torch.stack(result_list, dim=0)


# ------------------ TIMING HELPER ------------------
def measure(fn, *args):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return out, end - start



def generate_sparse_tensor_uniform_batch(
    shape: tuple[int, int, int], sparsity: float, device='cpu'
) -> torch.Tensor:
    """
    Generates a 3D sparse tensor in COO format with uniform sparsity across batches.

    Args:
        shape (tuple[int,int,int]): (batch, seq_len, hidden_size)
        sparsity (float): Fraction of elements that should be zero (0.0 to 1.0)
        device (str): Device for the tensor

    Returns:
        torch.Tensor: Sparse COO tensor of shape `shape`
    """
    assert 0 <= sparsity <= 1, "Sparsity must be between 0 and 1"

    batch, seq_len, hidden_size = shape
    nnz_per_batch = int(seq_len * hidden_size * (1 - sparsity))  # non-zero per batch

    all_indices = []
    all_values = []

    for b in range(batch):
        # Random indices within this batch
        row_indices = torch.randint(0, seq_len, (nnz_per_batch,), device=device)
        col_indices = torch.randint(0, hidden_size, (nnz_per_batch,), device=device)
        batch_indices = torch.full((nnz_per_batch,), b, device=device, dtype=torch.long)

        # Stack as [3, nnz]
        batch_indices_stack = torch.stack([batch_indices, row_indices, col_indices], dim=0)
        all_indices.append(batch_indices_stack)

        # Random values
        all_values.append(torch.randn(nnz_per_batch, device=device))

    # Combine all batches
    indices = torch.cat(all_indices, dim=1)  # [3, total_nnz]
    values = torch.cat(all_values)  # [total_nnz]

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=shape, device=device)
    return sparse_tensor


# Test range: e.g. 0.90 to 0.995
sparsities = [0.85 + i * 0.005 for i in range(30)]
actual_sparsity = []

times = []
for sparsity in sparsities:
    # Example usage
    features_sparse = generate_sparse_tensor_uniform_batch((32, 128, 24576), sparsity=sparsity)

    print("Fake Shape:", features_sparse.shape)
    print("Fake NNZ:", features_sparse._nnz())
    print("Sparsity:", features_sparse._nnz() / torch.numel(features_sparse))
    actual_sparsity.append(features_sparse._nnz() / torch.numel(features_sparse))

    # ---- PROFILE cuSPARSE DECODER ----
    decoded_sparse, dec_sparse_time = measure(cusparse_decode, features_sparse.to(device), sae.W_dec.to(device))
    print(f"Decoder time (cuSPARSE): {dec_sparse_time * 1000:.3f} ms")
    times.append(dec_sparse_time)


import matplotlib.pyplot as plt

def plot_sparsity_times(sparsities, times, filename="plot.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(sparsities, times, marker="o")   # no custom colors per your constraints
    plt.xlabel("Sparsity (percent non-zero)")
    plt.ylabel("Decode Time (ms)")
    plt.title("Decode Time vs Sparsity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_sparsity_times(actual_sparsity[1:], times[1:])

# # ---- OPTIONAL: Compare dense vs cuSPARSE decoder results ----
# decoded_dense = features @ sae.W_dec
# max_diff = (decoded_dense - decoded_sparse).abs().max().item()
# print(f"Max difference (dense vs sparse decode): {max_diff:.6f}")