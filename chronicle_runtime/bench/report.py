#!/usr/bin/env python3
"""Print benchmark summary and save results to bench/results.json."""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone


def _percentile(sorted_arr: list[float], p: float) -> float:
    if not sorted_arr:
        return 0.0
    k = (len(sorted_arr) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_arr) - 1)
    return sorted_arr[f] + (k - f) * (sorted_arr[c] - sorted_arr[f])


def _batch_size_distribution(batch_sizes: list[int]) -> dict:
    if not batch_sizes:
        return {}
    c = Counter(batch_sizes)
    return dict(sorted(c.items()))


def report(data: dict) -> str:
    """Format a single benchmark result for display."""
    lines = []
    backend = data.get("backend", "?")
    lines.append(f"=== {backend.upper()} ===")
    lines.append(f"  model_name:           {data.get('model_name', 'N/A')}")
    lines.append(f"  device:               {data.get('device', 'N/A')}")
    lines.append(f"  prompt_token_length:  {data.get('prompt_token_length', 'N/A')}")
    lines.append(f"  max_new_tokens:       {data.get('max_new_tokens', 'N/A')}")
    lines.append(f"  num_prompts:          {data.get('num_prompts', 0)}")
    lines.append(f"  total_tokens:         {data.get('total_tokens', 0)}")
    lines.append(f"  total_s:              {data.get('total_s', 0):.2f}")
    lines.append(f"  tokens/sec:           {data.get('tokens_per_sec', 0):.2f}")

    latencies = [x for x in data.get("latencies_ms", []) if x >= 0]
    if latencies:
        sorted_lat = sorted(latencies)
        p50 = _percentile(sorted_lat, 50)
        p95 = _percentile(sorted_lat, 95)
        p99 = _percentile(sorted_lat, 99)
        lines.append(f"  latency p50 (ms):     {p50:.2f}")
        lines.append(f"  latency p95 (ms):     {p95:.2f}")
        lines.append(f"  latency p99 (ms):     {p99:.2f}")

    batch_sizes = data.get("batch_sizes", [])
    if batch_sizes:
        dist = _batch_size_distribution(batch_sizes)
        lines.append(f"  batch_size_dist:      {dist}")
        avg = sum(batch_sizes) / len(batch_sizes)
        lines.append(f"  batch_sizes:          min={min(batch_sizes)}, max={max(batch_sizes)}, avg={avg:.1f}")

    if "gpu_mem_mb" in data:
        lines.append(f"  gpu_mem_mb:           {data['gpu_mem_mb']:.2f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSON result files (or stdin)")
    parser.add_argument(
        "-o", "--output",
        default=".bench/results.json",
        help="Save combined results to this file",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save to results.json")
    args = parser.parse_args()

    all_data = []
    if args.files:
        for path in args.files:
            with open(path) as f:
                all_data.append(json.load(f))
    else:
        all_data.append(json.load(sys.stdin))

    for data in all_data:
        print(report(data))
        print()

    if not args.no_save and all_data:
        import platform
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "runs": all_data,
        }
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
