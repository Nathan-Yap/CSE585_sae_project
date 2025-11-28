'''
Data Parallel profiling

python profile_sae_dp.py --gpus 1 --batch_size 32
python profile_sae_dp.py --gpus 2 --batch_size 32
python profile_sae_dp.py --gpus 4 --batch_size 32

python profile_sae_dp.py --gpus 2 --batch_size 32 --profile_rank0

--profile_rank0:
    Profiles only the first SAE encode on GPU 0,
    Saves a Chrome trace and prints a nice op table, (chrome://tracing)
    Then runs the rest normally and still reports average ms/iter across GPUs.
'''

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity

from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs / processes to use (1–4)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Global batch size (must be divisible by gpus)."
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Number of timed SAE.encode iterations."
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations (not timed)."
    )
    parser.add_argument(
        "--profile_rank0", action="store_true",
        help="If set, run torch.profiler around SAE.encode on rank 0 (first timed iter)."
    )
    parser.add_argument(
        "--trace_file", type=str, default="sae_encode_trace_rank0.json",
        help="Chrome trace file for rank 0 profiler."
    )
    return parser.parse_args()


def setup_process(rank: int, world_size: int):
    """Initialize distributed backend for this rank."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_process():
    dist.destroy_process_group()


def main_worker(rank: int, world_size: int, args):
    setup_process(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"[Rank {rank}] World size = {world_size}, global batch = {args.batch_size}")
        print(f"[Rank {rank}] iters = {args.iters}, warmup = {args.warmup}")

    # --- Load model + SAE (one full copy per rank = data parallel) ---
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device,
    )
    sae.eval()

    sae_cfg = sae.cfg
    context_size = sae_cfg.metadata.context_size
    prepend_bos = sae_cfg.metadata.prepend_bos
    hook_name = sae_cfg.metadata.hook_name

    # --- Load & tokenize dataset (per rank, for simplicity) ---
    dataset = load_dataset(
        path="NeelNanda/pile-10k",
        split="train",
        streaming=False,
    )

    tokenizer = model.tokenizer

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=tokenizer,
        streaming=False,           # easier indexing
        max_length=context_size,
        add_bos_token=prepend_bos,
    )

    # --- Split global batch across ranks ---
    assert args.batch_size % world_size == 0, \
        f"batch_size ({args.batch_size}) must be divisible by gpus ({world_size})"

    local_batch = args.batch_size // world_size
    start_idx = rank * local_batch
    end_idx = start_idx + local_batch

    sample = token_dataset[start_idx:end_idx]
    batch_tokens = sample["tokens"]  # shape [local_batch, context_size]
    batch_tokens = torch.tensor(batch_tokens, dtype=torch.long, device=device)

    if rank == 0:
        print(f"[Rank {rank}] Local batch = {local_batch}, tokens shape = {batch_tokens.shape}")

    # --- Get activations from GPT-2 (outside profiling region) ---
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        if rank == 0:
            print(f"[Rank {rank}] Running GPT-2 to get cache...")

        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
        acts = cache[hook_name]  # e.g. [local_batch, seq, d_model]

        # Optional: free cache to save memory
        del cache
        torch.cuda.empty_cache()

    # --- Barrier: ensure all ranks have activations ready ---
    dist.barrier()

    # --- Benchmark SAE.encode on each rank ---
    with torch.no_grad():
        # Warmup iterations (not timed)
        for _ in range(args.warmup):
            _ = sae.encode(acts)
        torch.cuda.synchronize(device)

        local_times = []

        if args.profile_rank0 and rank == 0:
            # Profile only the FIRST timed iteration on rank 0
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
            ) as prof:
                with record_function("SAE_encode"):
                    start = time.perf_counter()
                    _ = sae.encode(acts)
                    torch.cuda.synchronize(device)
                    end = time.perf_counter()
                    local_times.append(end - start)

            # Remaining timed iterations without profiler
            for _ in range(args.iters - 1):
                start = time.perf_counter()
                _ = sae.encode(acts)
                torch.cuda.synchronize(device)
                end = time.perf_counter()
                local_times.append(end - start)

            # Save trace & table
            prof.export_chrome_trace(args.trace_file)
            print(f"[Rank 0] Exported profiler trace to {args.trace_file}")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        else:
            # No profiler on this rank
            for _ in range(args.iters):
                start = time.perf_counter()
                _ = sae.encode(acts)
                torch.cuda.synchronize(device)
                end = time.perf_counter()
                local_times.append(end - start)

        # Average time per iteration on this rank
        local_time = sum(local_times) / len(local_times)

    # --- Gather timing results ---
    times_list = [None for _ in range(world_size)]
    dist.all_gather_object(times_list, local_time)

    if rank == 0:
        step_time = max(times_list)
        print(f"\n=== SAE.encode DP timing (world_size={world_size}) ===")
        for r, t in enumerate(times_list):
            print(f"  Rank {r}: {t*1000:.3f} ms (avg over {args.iters} iters)")
        print(f"  -> Effective step time (max over ranks): {step_time*1000:.3f} ms\n")

    cleanup_process()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    available_gpus = torch.cuda.device_count()
    if args.gpus > available_gpus:
        raise RuntimeError(
            f"Requested {args.gpus} GPUs but only {available_gpus} available."
        )

    world_size = args.gpus

    # Spawn one process per GPU
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
