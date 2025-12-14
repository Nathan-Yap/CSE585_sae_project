# profile_sae.py

import time
import torch
from sae_lens import SAE


# ------------------ DEVICE ------------------
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


# ------------------ LOAD PREGENERATED RESIDUAL STREAM ------------------
residual_stream = torch.load("../residual_stream/residual_stream.pt").to(device)

print("Loaded residual stream with dimensions")
print(residual_stream.shape)


# ------------------ TIMING FUNCTION ------------------
def measure(fn, *args):
    """Measure wall-clock time."""
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

try:
    # Count how many features are active per token (activation > 0)
    active_per_token = (features > 0).sum(dim=-1)

    # Average active features
    avg_active = active_per_token.float().mean().item()

    # Global sparsity %
    total_feats = features.numel()
    active_feats = (features > 0).sum().item()
    sparsity = 1 - (active_feats / total_feats)

    print(f"Average active features per token: {avg_active:.2f}")
    print(f"Total number of features {total_feats}")
    print(f"Sparsity: {sparsity*100:.4f}%")
except:
    pass

# ------------------ PROFILE DECODER ------------------
decoded, dec_time = measure(sae.decode, features)
print(f"Decoder time: {dec_time * 1000:.3f} ms")
