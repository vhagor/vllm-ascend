# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU/Gloo or NPU/HCCL equivalence checks for DSpark argmax.

Run directly to avoid importing the full vLLM/NPU unit-test fixture stack.
"""

import argparse
import importlib.util
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load_sampler(path):
    spec = importlib.util.spec_from_file_location("dspark_local_argmax_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_local_draft_tokens


def check_rank(rank, world_size, backend, rendezvous, implementation):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    if backend == "hccl":
        import torch_npu  # noqa: F401

        device = torch.device(f"npu:{rank}")
        torch.npu.set_device(device)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=world_size)
    sample = load_sampler(implementation)
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for vocab_size, mode in ((33, "random"), (33, "ties"), (131077, "markov")):
                torch.manual_seed(41)
                batch, steps = 2, 7
                full = torch.randn(batch, steps, vocab_size, dtype=dtype).to(device)
                if mode == "ties":
                    full.zero_()
                if mode == "markov":
                    full.zero_()
                    full[:, :, 5] = 9
                seed = torch.tensor([0, vocab_size - 4], dtype=torch.int64, device=device)

                def bias(tokens, batch=batch, vocab_size=vocab_size, dtype=dtype, mode=mode):
                    values = torch.zeros(batch, vocab_size, dtype=dtype, device=device)
                    if mode != "ties":
                        indices = ((tokens + 3) % vocab_size).unsqueeze(-1)
                        values.scatter_(1, indices, 20)
                    return values

                reference = torch.empty(batch, steps, dtype=torch.int64, device=device)
                previous = seed
                for position in range(steps):
                    corrected = full[:, position].clone()
                    corrected.add_(bias(previous))
                    reference[:, position] = corrected.argmax(-1)
                    previous = reference[:, position]
                shard_size = (vocab_size + world_size - 1) // world_size
                start = rank * shard_size
                valid = max(0, min(shard_size, vocab_size - start))
                local = torch.full((batch, steps, shard_size), 1000, dtype=dtype, device=device)
                local[:, :, :valid] = full[:, :, start : start + valid]
                calls = []

                def gather(pairs, calls=calls):
                    calls.append(tuple(pairs.shape))
                    outputs = [torch.empty_like(pairs) for _ in range(world_size)]
                    dist.all_gather(outputs, pairs)
                    return torch.cat(outputs, dim=-1)

                actual = sample(local, seed, bias, start, vocab_size, world_size, gather)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert calls == ([(batch, 2)] * steps if world_size > 1 else [])
                if rank == 0:
                    print(
                        f"PASS backend={backend} tp={world_size} dtype={dtype} mode={mode} vocab={vocab_size}",
                        flush=True,
                    )
        try:
            sample(torch.zeros(1, 1, 1, device=device), seed[:1], bias, 0, (1 << 24) + 1, world_size, gather)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected rejection of inexact FP32 vocabulary IDs")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    parser.add_argument("--world-sizes", nargs="+", type=int, default=[1, 2, 8])
    parser.add_argument("--implementation", required=True)
    args = parser.parse_args()
    for size in args.world_sizes:
        with tempfile.TemporaryDirectory(prefix="dspark-argmax-") as directory:
            rendezvous = "file://" + str(Path(directory) / "store")
            mp.spawn(check_rank, args=(size, args.backend, rendezvous, args.implementation), nprocs=size, join=True)


if __name__ == "__main__":
    main()
