#!/usr/bin/env python

import argparse
import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.profiler import profile, ProfilerActivity
from sae_lens import SAE


# ----------------- CLI ----------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile SAE-like encoder/decoder with tensor parallelism"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (1–4, etc.)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for synthetic activations (number of token positions)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup iterations before timing / profiling",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="Number of iterations to time (and profile)",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Optional Chrome trace path (rank 0 only), e.g. sae_tp_trace.json",
    )
    parser.add_argument(
        "--release",
        type=str,
        default="gpt2-small-res-jb",
        help="SAE release name (same as your DP script)",
    )
    parser.add_argument(
        "--sae-id",
        type=str,
        default="blocks.8.hook_resid_pre",
        help="SAE id / hook name (same as your DP script)",
    )
    return parser.parse_args()


# ------------- Helper: profile only on rank 0 ------------- #

@contextmanager
def profile_rank0(rank: int, **profile_kwargs):
    """
    Small helper so only rank 0 runs torch.profiler.profile.

    On non-zero ranks this becomes a no-op context manager.
    """
    if rank == 0:
        with profile(**profile_kwargs) as prof:
            yield prof
    else:
        # Dummy context: yields None and does nothing
        yield None


# ------------- Tensor-parallel SAE-like module ------------- #

class TensorParallelSAE(torch.nn.Module):
    """
    Simple SAE-style module:

        x (B, D_in) -> ReLU( x @ W_enc_shard ) -> h_local (B, F_local)
                    -> y_local = h_local @ W_dec_shard (B, D_in)
                    -> all_reduce over ranks to sum y_local

    We shard the *feature* dimension F across GPUs. Each rank holds a slice
    of features [start:end] and its own encoder/decoder weights for that slice.
    """

    def __init__(self, d_in: int, d_sae: int, world_size: int, rank: int, device):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.world_size = world_size
        self.rank = rank

        # ---- Uneven sharding of feature dimension ----
        base = d_sae // world_size
        remainder = d_sae % world_size

        if rank < remainder:
            start = rank * (base + 1)
            end = start + (base + 1)
        else:
            start = remainder * (base + 1) + (rank - remainder) * base
            end = start + base

        self.start = start
        self.end = end
        local_f = end - start

        # Local encoder/decoder for this feature slice
        self.encoder = torch.nn.Linear(d_in, local_f, bias=True, device=device)
        self.decoder = torch.nn.Linear(local_f, d_in, bias=False, device=device)

    def forward(self, x: torch.Tensor, world_group):
        """
        x: [batch_size, d_in] identical on all ranks

        Returns:
            y: [batch_size, d_in] (full decoded activations, via all_reduce)
            h_local: [batch_size, local_f] (this rank's feature activations)
        """
        # Local encoder: x @ W_enc_shard
        h_local = torch.relu(self.encoder(x))  # [B, F_local]

        # Local decode: h_local @ W_dec_shard
        y_local = self.decoder(h_local)        # [B, D_in]

        # Aggregate partial decodes across all ranks
        if self.world_size > 1:
            dist.all_reduce(y_local, op=dist.ReduceOp.SUM, group=world_group)

        return y_local, h_local


# ------------- Distributed init / cleanup ------------- #

def setup_process_group(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def cleanup_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ------------- Worker (one per GPU) ------------- #

def worker(rank: int, world_size: int, d_in: int, d_sae: int, args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this tensor-parallel profiling script.")

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    if world_size > 1:
        setup_process_group(rank, world_size)
        world_group = dist.group.WORLD
    else:
        world_group = None

    # Build TP SAE-like module on this rank
    model = TensorParallelSAE(
        d_in=d_in,
        d_sae=d_sae,
        world_size=world_size,
        rank=rank,
        device=device,
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size = args.batch_size
    warmup_steps = args.warmup_steps
    profile_steps = args.profile_steps
    total_steps = warmup_steps + profile_steps

    # Synthetic activations: shapes match the SAE input dimension
    x = torch.randn(batch_size, d_in, device=device)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    timings = []

    with profile_rank0(
        rank,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            start = time.perf_counter()

            y, h_local = model(x, world_group)

            torch.cuda.synchronize(device)
            end = time.perf_counter()

            if step >= warmup_steps:
                timings.append(end - start)

            if prof is not None:
                prof.step()

    avg_ms = (sum(timings) / len(timings)) * 1000.0

    if rank == 0:
        print("=" * 80)
        print(f"Tensor-parallel SAE-like encode+decode with {world_size} GPU(s)")
        print(f"  d_in      = {d_in}")
        print(f"  d_sae     = {d_sae}")
        print(f"  batch     = {batch_size}")
        print(f"  warmup    = {warmup_steps}")
        print(f"  measured  = {profile_steps}")
        print(f"  mean step = {avg_ms:.3f} ms")
        print("=" * 80)

        if prof is not None:
            print("\nTop CUDA-time ops (rank 0):")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

            if args.trace_file:
                prof.export_chrome_trace(args.trace_file)
                print(f"\nChrome trace written to: {args.trace_file}")

    cleanup_process_group()


# ------------- Main ------------- #

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPUs are required to run this script.")

    available_gpus = torch.cuda.device_count()
    if args.gpus < 1:
        raise SystemExit("You must request at least 1 GPU.")
    if args.gpus > available_gpus:
        raise SystemExit(
            f"Requested {args.gpus} GPU(s), but only {available_gpus} are visible."
        )

    # Load SAE once on CPU just to get dimensions (same release / sae_id as your DP script)
    print("Loading SAE on CPU to get dimensions...")
    sae = SAE.from_pretrained(
        release=args.release,
        sae_id=args.sae_id,
        device="cpu",
    )
    # These names match typical SAELens configs and your earlier scripts
    d_in = sae.cfg.d_in
    d_sae = sae.cfg.d_sae
    print(f"SAE dims: d_in={d_in}, d_sae={d_sae}")
    del sae

    world_size = args.gpus

    if world_size == 1:
        # Single-process path (no actual communication, but same code path)
        worker(rank=0, world_size=1, d_in=d_in, d_sae=d_sae, args=args)
    else:
        # Multi-process tensor parallel
        mp.spawn(
            worker,
            args=(world_size, d_in, d_sae, args),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
