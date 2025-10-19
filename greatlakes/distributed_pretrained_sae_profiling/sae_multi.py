import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
import pickle
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

from transformer_lens.utils import tokenize_and_concatenate


if torch.cuda.is_available():
    device = "cuda"
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
elif torch.backends.mps.is_available():
    device = "mps"
    num_gpus = 1
    print("Using Apple Silicon MPS")
else:
    device = "cpu"
    num_gpus = 1
    print("Using CPU")


model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# Use DataParallel for multiple GPUs
if torch.cuda.is_available() and num_gpus > 1:
    model = torch.nn.DataParallel(model)
    print(f"Using DataParallel across {num_gpus} GPUs")

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
    device=device,
)

# Use DataParallel for SAE if multiple GPUs available
if torch.cuda.is_available() and num_gpus > 1:
    sae = torch.nn.DataParallel(sae)
    print(f"Using DataParallel for SAE across {num_gpus} GPUs")

print("LOADED MODELS")
print(model)


dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=sae.cfg.metadata.context_size,
    add_bos_token=sae.cfg.metadata.prepend_bos,
)

print("TOKENIZED DATASET")

sae.eval()  # ensure deterministic / no grads

# Increase batch size for multi-GPU setups
batch_size = 32 * num_gpus if torch.cuda.is_available() and num_gpus > 1 else 32
batch_tokens = token_dataset[:batch_size]["tokens"]
print(f"Using batch size: {batch_size}")

with torch.no_grad():
    # Run model to get cache (outside profiling)
    if torch.cuda.is_available() and num_gpus > 1:
        # For DataParallel, we need to access the module attribute
        _, cache = model.module.run_with_cache(batch_tokens, prepend_bos=True)
    else:
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Profile only SAE encode/decode
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with record_function("SAE_encode_decode"):
            # Get the hook name from the appropriate SAE object
            if torch.cuda.is_available() and num_gpus > 1:
                hook_name = sae.module.cfg.metadata.hook_name
                feature_acts = sae.encode(cache[hook_name])
                sae_out = sae.decode(feature_acts)
            else:
                feature_acts = sae.encode(cache[sae.cfg.metadata.hook_name])
                sae_out = sae.decode(feature_acts)

    # Optional: delete cache to save memory
    del cache

# Print a table of top ops sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export a Chrome trace for visualization
prof.export_chrome_trace("sae_encode_decode_trace.json")