#!/usr/bin/env python3
"""Benchmark Chronicle: batched engine.batch_generate."""

import argparse
import json
import time

import torch

from chronicle_runtime.runtime.engine import batch_generate
from chronicle_runtime.bench.prompts import make_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-prompts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--fixed-length", type=int, default=64)
    parser.add_argument("--varied", action="store_true", help="Use varied prompt lengths")
    parser.add_argument("--batch-size", type=int, default=None, help="Chunk size (default: all)")
    parser.add_argument("-o", "--output", default="-", help="Output JSON file (- for stdout)")
    args = parser.parse_args()

    prompts = make_prompts(args.num_prompts, args.fixed_length, args.varied)
    max_new_tokens_list = [args.max_new_tokens] * len(prompts)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    batch_size = args.batch_size or len(prompts)
    latencies_ms: list[float] = []
    tokens_per_request: list[int] = []
    batch_sizes: list[int] = []

    start = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        chunk_p = prompts[i : i + batch_size]
        chunk_m = max_new_tokens_list[i : i + batch_size]
        t0 = time.perf_counter()
        texts, tokens, prefill_ms, decode_ms = batch_generate(chunk_p, chunk_m, seed=42)
        batch_time_ms = (time.perf_counter() - t0) * 1000
        batch_sizes.append(len(chunk_p))
        tokens_per_request.extend(tokens)
        # Per-request latency: amortized batch time
        for _ in chunk_p:
            latencies_ms.append(batch_time_ms / len(chunk_p))
    total_s = time.perf_counter() - start

    total_tokens = sum(tokens_per_request)
    tokens_per_sec = total_tokens / total_s if total_s > 0 else 0

    gpu_mem_mb = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    result = {
        "backend": "chronicle",
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "fixed_length": args.fixed_length,
        "varied": args.varied,
        "total_tokens": total_tokens,
        "total_s": total_s,
        "tokens_per_sec": tokens_per_sec,
        "latencies_ms": latencies_ms,
        "batch_sizes": batch_sizes,
    }
    if gpu_mem_mb is not None:
        result["gpu_mem_mb"] = round(gpu_mem_mb, 2)

    out = json.dumps(result, indent=2)
    if args.output == "-":
        print(out)
    else:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
