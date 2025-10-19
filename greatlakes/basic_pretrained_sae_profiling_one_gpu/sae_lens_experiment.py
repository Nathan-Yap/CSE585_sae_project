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
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
    device=device,
)

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

batch_tokens = token_dataset[:32]["tokens"]

with torch.no_grad():
    # Run model to get cache (outside profiling)
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
            feature_acts = sae.encode(cache[sae.cfg.metadata.hook_name])
            sae_out = sae.decode(feature_acts)

    # Optional: delete cache to save memory
    del cache

# Print a table of top ops sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export a Chrome trace for visualization
prof.export_chrome_trace("sae_encode_decode_trace.json")