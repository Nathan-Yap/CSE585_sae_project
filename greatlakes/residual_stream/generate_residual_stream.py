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

# ------------------ PICK A BATCH ------------------
batch_tokens = token_dataset[:32]["tokens"]


# ------------------ GET RESIDUAL STREAM ------------------
print("Generating residual stream...")
with torch.no_grad():
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

residual_stream = cache[sae.cfg.metadata.hook_name].to("cpu")
del cache

# ------------------ SAVE TO DISK ------------------
save_path = "residual_stream.pt"
torch.save(residual_stream, save_path)

print(f"Saved residual stream to {save_path}")
