# generate_residual_stream.py

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from transformer_lens.utils import tokenize_and_concatenate


# ------------------ DEVICE ------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ------------------ LOAD MODEL + SAE ------------------
print("Loading model + SAE...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device,
)

# ------------------ LOAD DATA ------------------
print("Loading dataset...")
dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,
    tokenizer=model.tokenizer,
    streaming=True,
    max_length=sae.cfg.metadata.context_size,
    add_bos_token=sae.cfg.metadata.prepend_bos,
)

print("Dataset tokenized.")

# ------------------ PARAMETERS ------------------
TARGET_BATCH = 2056        # collect this many sequences
CHUNK_SIZE = 32            # process dataset in chunks of 32
hook_name = sae.cfg.metadata.hook_name


# ------------------ COLLECT RESIDUAL STREAMS ------------------
print("Generating residual stream...")

all_residuals = []
collected = 0

with torch.no_grad():
    for i in range(0, TARGET_BATCH, CHUNK_SIZE):
        if collected >= TARGET_BATCH:
            break

        # Slice chunk
        chunk = token_dataset[i : i + CHUNK_SIZE]["tokens"]
        chunk = chunk.to(device)

        # Forward + cache
        _, cache = model.run_with_cache(chunk, prepend_bos=True)

        residual_chunk = cache[hook_name].to("cpu")   # shape: [32, seq_len, d_model]
        del cache

        all_residuals.append(residual_chunk)
        collected += residual_chunk.size(0)

        print(f"Collected {collected}/{TARGET_BATCH} sequences...")

        if collected >= TARGET_BATCH:
            break


# ------------------ CONCAT + TRIM ------------------
residual_stream = torch.cat(all_residuals, dim=0)

# Trim to exactly 2056 sequences in case we overshot
residual_stream = residual_stream[:TARGET_BATCH]

print(f"Final residual stream shape: {residual_stream.shape}")


# ------------------ SAVE TO DISK ------------------
save_path = "residual_stream.pt"
torch.save(residual_stream, save_path)

print(f"Saved residual stream to {save_path}")