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

# ------------------ LOAD RESIDUAL STREAM ------------------
og_residual_stream = torch.load("/scratch/cse585f25_class_root/cse585f25_class/nyap/residual_stream_2056_tokens.pt").to(device)
print("Loaded residual stream:", og_residual_stream.shape)
print("")


# ------------------ TIMING HELPER ------------------
def measure(fn, *args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = fn(*args)
    end.record()
    torch.cuda.synchronize()

    return out, start.elapsed_time(end)

for batch_size in [16, 16, 32, 64, 128, 256, 512, 1028, 2056]: # Added extra 16 for warm-up cycle
    residual_stream = og_residual_stream[0:batch_size]
    print("NEW ITERATION: using batch of shape: ")
    print(residual_stream.shape)
    print("")
    # ------------------ PROFILE ENCODER ------------------
    features, enc_time = measure(sae.encode, residual_stream)
    print(f"Full Encoder time: {enc_time * 1000:.3f} ms")


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
    print("")


    # ============================================================
    #          cuSPARSE DECODER (Sparse × Dense SpMM)
    # ============================================================

    def cusparse_decode(sparse_tensor, W_dec):
        """Perform sparse (CSR) × dense decode using cuSPARSE (torch.sparse.mm)."""
        # result_list = []
        # for i in range(batch_size):
        #     # sparse @ dense
        #     result_list.append(torch.sparse.mm(sparse_tensor[i], W_dec))  # [128, 768]

        # # Stack results to get [32, 128, 768]
        # return torch.stack(result_list, dim=0)
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
