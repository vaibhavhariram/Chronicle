#!/usr/bin/env python3
"""Print benchmark summary: tokens/sec, p50/p95 latency, batch_size stats."""

import argparse
import json
import sys


def _percentile(sorted_arr: list[float], p: float) -> float:
    if not sorted_arr:
        return 0.0
    k = (len(sorted_arr) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_arr) - 1)
    return sorted_arr[f] + (k - f) * (sorted_arr[c] - sorted_arr[f])


def report(data: dict) -> str:
    lines = []
    backend = data.get("backend", "?")
    lines.append(f"=== {backend.upper()} ===")
    lines.append(f"  num_prompts:    {data.get('num_prompts', 0)}")
    lines.append(f"  total_tokens:   {data.get('total_tokens', 0)}")
    lines.append(f"  total_s:        {data.get('total_s', 0):.2f}")
    lines.append(f"  tokens/sec:     {data.get('tokens_per_sec', 0):.2f}")
    if "gpu_mem_mb" in data:
        lines.append(f"  gpu_mem_mb:     {data['gpu_mem_mb']:.2f}")

    latencies = data.get("latencies_ms", [])
    if latencies:
        sorted_lat = sorted(latencies)
        p50 = _percentile(sorted_lat, 50)
        p95 = _percentile(sorted_lat, 95)
        lines.append(f"  latency p50:    {p50:.2f} ms")
        lines.append(f"  latency p95:    {p95:.2f} ms")

    batch_sizes = data.get("batch_sizes", [])
    if batch_sizes:
        avg = sum(batch_sizes) / len(batch_sizes)
        lines.append(f"  batch_sizes:    min={min(batch_sizes)}, max={max(batch_sizes)}, avg={avg:.1f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSON result files (or stdin)")
    args = parser.parse_args()

    if args.files:
        for path in args.files:
            with open(path) as f:
                data = json.load(f)
            print(report(data))
            print()
    else:
        data = json.load(sys.stdin)
        print(report(data))


if __name__ == "__main__":
    main()
