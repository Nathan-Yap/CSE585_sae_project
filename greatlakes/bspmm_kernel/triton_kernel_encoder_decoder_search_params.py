import time
import argparse
import torch
from sparse_triton_kernel import spmm_coo_triton
from sae_lens import SAE
from utils import monkeypatch_sae_decode_with_kernel, measure

# ============================================================
#                   ARGUMENT PARSING
# ============================================================

# Passing in the custom kernel parameters since this process often fails
parser = argparse.ArgumentParser()
parser.add_argument("--block-e", type=int, default=4, help="BLOCK_E size for Triton kernel")
parser.add_argument("--block-n", type=int, default=128, help="BLOCK_N size for Triton kernel")
args = parser.parse_args()

BLOCK_E = args.block_e
BLOCK_N = args.block_n

print(f"Using Triton kernel params: BLOCK_E={BLOCK_E}, BLOCK_N={BLOCK_N}")

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

# ------------------ LOAD RESIDUAL STREAM ------------------
og_residual_stream = torch.load("/scratch/cse585f25_class_root/cse585f25_class/nyap/residual_stream_2056_tokens.pt")
print("Loaded residual stream:", og_residual_stream.shape)
print("")

with open(f"grid_search_output/BLOCK_E_{BLOCK_E}_BLOCK_N_{BLOCK_N}.txt", 'w') as f:
    for batch_size in [1, 4, 8, 16, 32, 64, 128, 256, 512, 1028, 2056]: # Added extra 1 for warm-up cycle
        residual_stream = og_residual_stream[0:batch_size].to(device)
        print("NEW ITERATION: using batch of shape: ")
        print(residual_stream.shape)
        f.write("Batch Shape: ")
        f.write(str(residual_stream.shape))
        f.write("\n")
        # print("")
        # ------------------ PROFILE ENCODER ------------------
        features, enc_time = measure(sae.encode, residual_stream)
        # print(f"Full Encoder time: {enc_time * 1000:.3f} ms")


        # # ------------------ FEATURE SPARSITY STATS ------------------
        # THIS IS ONLY FOR PROFILING
        # active_per_token = (features > 0).sum(dim=-1)
        # avg_active = active_per_token.float().mean().item()
        # total_feats = features.numel()
        # active_feats = (features > 0).sum().item()
        # sparsity = 1 - (active_feats / total_feats)

        # print("")
        # print(f"Average active features/token: {avg_active:.2f}")
        # print(f"Total number of features {total_feats}")
        # print(f"Sparsity: {sparsity*100:.4f}%")
        # print("")


        # ============================================================
        #          cuSPARSE DECODER (Sparse × Dense SpMM)
        # ============================================================

        def custom_triton_decode(sparse_tensor, W_dec):
            return spmm_coo_triton(sparse_tensor, W_dec, BLOCK_E=BLOCK_E, BLOCK_N=BLOCK_N)

        def naive_cusparse_decode(sparse_tensor, W_dec):
            return torch.vmap(lambda x: torch.sparse.mm(x, W_dec))(sparse_tensor)

        def normal_decode(feature_acts, W_dec):
            return feature_acts @ W_dec

        # Normal decoder timing
        timed_sae = monkeypatch_sae_decode_with_kernel(sae, normal_decode)
        decoded, dec_dense_time = measure(timed_sae.decode, features)
        print(f"\tNormal BMM (dense): {dec_dense_time * 1000:.3f} ms")
        f.write(f"\tNormal BMM (dense): {dec_dense_time * 1000:.3f} ms\n")


        # # ---- Convert dense activations to CSR sparse format ----
        # # PyTorch will call cuSPARSE for this kernel
        # start = time.perf_counter()

        # # features_2d = features.reshape(-1, features.shape[-1])  # [128, n_feats]
        features_sparse = features.to_sparse() # COO format

        # end = time.perf_counter()
        # print(f"Time to convert to COO {end - start}")
        # print("")

        # print(f"Sparse COO features: nnz = {features_sparse._nnz()} "
        #     f"({100 * features_sparse._nnz() / features.numel():.4f}% nonzero)")
        # print("")

        # ---- PROFILE cuSPARSE DECODER ----
        naive_sae_sparse = monkeypatch_sae_decode_with_kernel(sae, naive_cusparse_decode)
        decoded_sparse, dec_sparse_time = measure(naive_sae_sparse.decode, features_sparse)
        print(f"\tNaive cusparse decoder time: {dec_sparse_time * 1000:.3f} ms")
        f.write(f"\tNaive cusparse decoder time: {dec_sparse_time * 1000:.3f} ms\n")

        # ---- PROFILE cuSPARSE DECODER ----
        custom_sae_sparse = monkeypatch_sae_decode_with_kernel(sae, custom_triton_decode)
        decoded_triton_sparse, dec_triton_sparse_time = measure(custom_sae_sparse.decode, features_sparse)
        print(f"\tCustom triton kernel time: {dec_triton_sparse_time * 1000:.3f} ms")
        f.write(f"\tCustom triton kernel time: {dec_triton_sparse_time * 1000:.3f} ms\n")
