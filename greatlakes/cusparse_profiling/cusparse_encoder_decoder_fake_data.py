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

import time
import torch
from sae_lens import SAE
from utils import monkeypatch_sae_decode_with_kernel

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

# USing fake data for now
# # ------------------ LOAD RESIDUAL STREAM ------------------
# og_residual_stream = torch.load("/scratch/cse585f25_class_root/cse585f25_class/nyap/residual_stream_2056_tokens.pt").to(device)
# print("Loaded residual stream:", og_residual_stream.shape)
# print("")


# ------------------ TIMING HELPER ------------------
def measure(fn, *args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = fn(*args)
    end.record()
    torch.cuda.synchronize()

    return out, start.elapsed_time(end)

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
    return sparse_tensor.to_dense()

# Largest possible batch size on 16GB VRAM (256)
batch_size = 1
sparsities = [0.99 + i * 0.0005 for i in range(20)]
real_sparsity = []
# Making these global so that they can be modified in the monkeypatch?

for sparsity in sparsities: # First iteration is a warm-up cycle

    # residual_stream = og_residual_stream[0:batch_size]
    features = generate_sparse_tensor_uniform_batch((batch_size, 128, 24576), sparsity=sparsity, device=device)

    print("")
    print(f"NEW ITERATION: using sparsity of shape: {sparsity}")
    print("")
    # # ------------------ PROFILE ENCODER ------------------
    # features, enc_time = measure(sae.encode, residual_stream)
    # print(f"Full Encoder time: {enc_time * 1000:.3f} ms")


    # ------------------ FEATURE SPARSITY STATS ------------------
    active_per_token = (features > 0).sum(dim=-1)
    avg_active = active_per_token.float().mean().item()
    total_feats = features.numel()
    active_feats = (features > 0).sum().item()
    sparsity = 1 - (active_feats / total_feats)

    print("")
    print(f"Average active features/token: {avg_active:.2f}")
    print(f"Total number of features {total_feats}")
    print(f"Sparsity: {sparsity*100:.4f}%")
    real_sparsity.append(sparsity)
    print("")


    # ============================================================
    #          cuSPARSE DECODER (Sparse × Dense SpMM)
    # ============================================================

    def cusparse_decode(sparse_tensor, W_dec):
        return torch.vmap(lambda x: torch.sparse.mm(x, W_dec))(sparse_tensor)

    def normal_decode(feature_acts, W_dec):
        # Same as torch.bmm
        return feature_acts @ W_dec

    # Normal decoder timing
    print("Normal decode timing:")
    timed_sae = monkeypatch_sae_decode_with_kernel(sae, normal_decode)
    decoded, dec_dense_time = measure(timed_sae.decode, features)
    print(f"Normal time (dense): {dec_dense_time * 1000:.3f} ms")
    print("")

    # ---- Convert dense activations to CSR sparse format ----
    # PyTorch will call cuSPARSE for this kernel
    start = time.perf_counter()

    # features_2d = features.reshape(-1, features.shape[-1])  # [128, n_feats]
    features_sparse = features.to_sparse() # COO format

    end = time.perf_counter()
    print(f"Time to convert to COO {end - start}")
    print("")

    print(f"Sparse COO features: nnz = {features_sparse._nnz()} "
        f"({100 * features_sparse._nnz() / features.numel():.4f}% nonzero)")
    print("")

    # ---- PROFILE cuSPARSE DECODER ----
    sae_sparse = monkeypatch_sae_decode_with_kernel(sae, cusparse_decode)
    decoded_sparse, dec_sparse_time = measure(sae_sparse.decode, features_sparse)
    print(f"Decoder time (cuSPARSE): {dec_sparse_time * 1000:.3f} ms")

print("SPARSITIES:")
print(real_sparsity)
print("CACHED TIMINGS:")
for k, v in PROFILING_CACHE.items():
    print(f"Func: {k}")
    print(v)
    print("\n")


# # Test range: e.g. 0.90 to 0.995
# sparsities = [0.85 + i * 0.005 for i in range(30)]
# actual_sparsity = []

# times = []
# for sparsity in sparsities:
#     # Example usage
#     features_sparse = generate_sparse_tensor_uniform_batch((256, 128, 24576), sparsity=sparsity)

#     print("Fake Shape:", features_sparse.shape)
#     print("Fake NNZ:", features_sparse._nnz())
#     print("Sparsity:", features_sparse._nnz() / torch.numel(features_sparse))
#     actual_sparsity.append(features_sparse._nnz() / torch.numel(features_sparse))

#     # ---- PROFILE cuSPARSE DECODER ----
#     decoded_sparse, dec_sparse_time = measure(cusparse_decode, features_sparse.to(device), sae.W_dec.to(device))
#     print(f"Decoder time (cuSPARSE): {dec_sparse_time * 1000:.3f} ms")
#     times.append(dec_sparse_time)


# import matplotlib.pyplot as plt

# def plot_sparsity_times(sparsities, times, filename="plot.png"):
#     plt.figure(figsize=(8, 5))
#     plt.plot(sparsities, times, marker="o")   # no custom colors per your constraints
#     plt.xlabel("Sparsity (percent non-zero)")
#     plt.ylabel("Decode Time (ms)")
#     plt.title("Decode Time vs Sparsity")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close()

# plot_sparsity_times(actual_sparsity[1:], times[1:])

# # ---- OPTIONAL: Compare dense vs cuSPARSE decoder results ----
# decoded_dense = features @ sae.W_dec
# max_diff = (decoded_dense - decoded_sparse).abs().max().item()
# print(f"Max difference (dense vs sparse decode): {max_diff:.6f}")