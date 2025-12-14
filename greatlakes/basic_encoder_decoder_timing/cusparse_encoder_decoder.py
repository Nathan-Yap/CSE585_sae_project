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

# ------------------ LOAD RESIDUAL STREAM ------------------
residual_stream = torch.load("/scratch/cse585f25_class_root/cse585f25_class/nyap/residual_stream.pt").to(device)
print("Loaded residual stream:", residual_stream.shape)

# residual_stream = residual_stream[0:1]
# print("Residual stream only first token in batch:", residual_stream.shape)


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


# ------------------ PROFILE ENCODER ------------------
features, enc_time = measure(sae.encode, residual_stream)
print(f"Encoder time: {enc_time * 1000:.3f} ms")


# ------------------ FEATURE SPARSITY STATS ------------------
active_per_token = (features > 0).sum(dim=-1)
avg_active = active_per_token.float().mean().item()
total_feats = features.numel()
active_feats = (features > 0).sum().item()
sparsity = 1 - (active_feats / total_feats)

print(f"Average active features/token: {avg_active:.2f}")
print(f"Total number of features {total_feats}")
print(f"Sparsity: {sparsity*100:.4f}%")


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


# ---- Convert dense activations to CSR sparse format ----
# PyTorch will call cuSPARSE for this kernel
start = time.perf_counter()

# features_2d = features.reshape(-1, features.shape[-1])  # [128, n_feats]
features_sparse = features.to_sparse()

end = time.perf_counter()
print(f"Time to convert to COO {end - start}")


print(f"Sparse COO features: nnz = {features_sparse._nnz()} "
      f"({100 * features_sparse._nnz() / features.numel():.4f}% nonzero)")


# ---- PROFILE cuSPARSE DECODER ----
decoded_sparse, dec_sparse_time = measure(cusparse_decode, features_sparse, sae.W_dec.to(device))
print(f"Decoder time (cuSPARSE): {dec_sparse_time * 1000:.3f} ms")


# ---- OPTIONAL: Compare dense vs cuSPARSE decoder results ----
decoded_dense = features @ sae.W_dec
max_diff = (decoded_dense - decoded_sparse).abs().max().item()
print(f"Max difference (dense vs sparse decode): {max_diff:.6f}")