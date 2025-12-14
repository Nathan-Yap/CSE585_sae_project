import torch
from torch.profiler import profile, record_function, ProfilerActivity
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

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
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
    dataset=dataset,
    tokenizer=model.tokenizer,
    streaming=True,
    max_length=sae.cfg.metadata.context_size,
    add_bos_token=sae.cfg.metadata.prepend_bos,
)

print("TOKENIZED DATASET")

sae.eval()

batch_tokens = token_dataset[:32]["tokens"]

with torch.no_grad():
    # Run model to get cache
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
    residual_stream = cache[sae.cfg.metadata.hook_name]

    ############################################
    #          PROFILE ENCODER ONLY            #
    ############################################
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as encoder_prof:
        with record_function("SAE_encode"):
            feature_acts = sae.encode(residual_stream)

    print("\n====== SAE ENCODER PROFILE ======")
    print(encoder_prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    encoder_prof.export_chrome_trace("sae_encode_trace.json")

    ############################################
    #          PROFILE DECODER ONLY            #
    ############################################
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as decoder_prof:
        with record_function("SAE_decode"):
            sae_out = sae.decode(feature_acts)

    print("\n====== SAE DECODER PROFILE ======")
    print(decoder_prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    decoder_prof.export_chrome_trace("sae_decode_trace.json")

del cache
