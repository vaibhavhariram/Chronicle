#!/usr/bin/env python3
"""HTTP load test: concurrent requests to /generate with latency percentiles."""

import argparse
import asyncio
import json
import os
import time

import httpx

SWEEP_LEVELS = [10, 25, 50, 100, 250]


def _percentile(sorted_arr: list[float], p: float) -> float:
    if not sorted_arr:
        return 0.0
    k = (len(sorted_arr) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_arr) - 1)
    return sorted_arr[f] + (k - f) * (sorted_arr[c] - sorted_arr[f])


async def worker(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_new_tokens: int,
    n: int,
    latencies: list[float],
    errors: list[Exception],
) -> None:
    """Send n requests and record latencies."""
    for _ in range(n):
        start = time.perf_counter()
        try:
            resp = await client.post(
                url,
                json={"prompt": prompt, "max_new_tokens": max_new_tokens},
                timeout=120.0,
            )
            resp.raise_for_status()
            latencies.append((time.perf_counter() - start) * 1000)
        except Exception as e:
            errors.append(e)
            latencies.append(-1)  # sentinel for failed request


async def run_load(
    url: str,
    concurrency: int,
    requests_per_worker: int,
    max_new_tokens: int,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
) -> tuple[list[float], list[Exception]]:
    """Run load test and return (latencies_ms, errors)."""
    latencies: list[float] = []
    errors: list[Exception] = []

    async with httpx.AsyncClient() as client:
        workers = [
            worker(
                client,
                url,
                prompt,
                max_new_tokens,
                requests_per_worker,
                latencies,
                errors,
            )
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)

    return latencies, errors


def _print_level_results(
    url: str,
    concurrency: int,
    total_requests: int,
    max_new_tokens: int,
    elapsed_s: float,
    latencies: list[float],
    errors: list[Exception],
) -> dict:
    ok_latencies = [l for l in latencies if l >= 0]
    error_count = len(errors)
    ok_count = len(ok_latencies)
    sorted_lat = sorted(ok_latencies) if ok_latencies else []

    p50 = _percentile(sorted_lat, 50) if sorted_lat else 0.0
    p95 = _percentile(sorted_lat, 95) if sorted_lat else 0.0
    p99 = _percentile(sorted_lat, 99) if sorted_lat else 0.0
    throughput = ok_count / elapsed_s if elapsed_s > 0 else 0.0

    print("=== Load test results ===")
    print(f"  url:              {url}")
    print(f"  concurrency:      {concurrency}")
    print(f"  total_requests:   {total_requests}")
    print(f"  max_new_tokens:   {max_new_tokens}")
    print(f"  elapsed_s:        {elapsed_s:.2f}")
    print(f"  throughput:       {throughput:.2f} req/s")
    print(f"  errors:           {error_count} ({100 * error_count / total_requests:.1f}%)")
    if sorted_lat:
        print(f"  latency p50:      {p50:.2f} ms")
        print(f"  latency p95:      {p95:.2f} ms")
        print(f"  latency p99:      {p99:.2f} ms")
    else:
        print("  (no successful requests)")

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "elapsed_s": round(elapsed_s, 3),
        "throughput_req_s": round(throughput, 2),
        "error_count": error_count,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--url",
        default="http://localhost:8000/generate",
        help="Base URL for /generate (default: http://localhost:8000/generate)",
    )
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("-n", "--requests-per-worker", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=f"Sweep concurrency levels {SWEEP_LEVELS} and save to .bench/load_sweep.json",
    )
    parser.add_argument(
        "--sweep-requests", type=int, default=5,
        help="Requests per worker per concurrency level during sweep (default: 5)",
    )
    args = parser.parse_args()

    if args.sweep:
        sweep_results = []
        for level in SWEEP_LEVELS:
            print(f"\n--- concurrency={level} ---")
            total_requests = level * args.sweep_requests
            start = time.perf_counter()
            latencies, errors = asyncio.run(
                run_load(
                    url=args.url,
                    concurrency=level,
                    requests_per_worker=args.sweep_requests,
                    max_new_tokens=args.max_new_tokens,
                    prompt=args.prompt,
                )
            )
            elapsed_s = time.perf_counter() - start
            row = _print_level_results(
                args.url, level, total_requests, args.max_new_tokens,
                elapsed_s, latencies, errors,
            )
            sweep_results.append(row)

        print("\n=== Sweep Summary ===")
        print(f"{'concurrency':>12}  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}  {'req/s':>8}  {'errors':>6}")
        for row in sweep_results:
            print(
                f"{row['concurrency']:>12}  {row['p50_ms']:>8.1f}  {row['p95_ms']:>8.1f}"
                f"  {row['p99_ms']:>8.1f}  {row['throughput_req_s']:>8.2f}"
                f"  {row['error_count']:>6}"
            )

        out_dir = os.path.join(os.getcwd(), ".bench")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "load_sweep.json")
        with open(out_path, "w") as f:
            json.dump({"url": args.url, "max_new_tokens": args.max_new_tokens, "levels": sweep_results}, f, indent=2)
        print(f"\nSaved to {out_path}")
        return

    total_requests = args.concurrency * args.requests_per_worker
    start = time.perf_counter()
    latencies, errors = asyncio.run(
        run_load(
            url=args.url,
            concurrency=args.concurrency,
            requests_per_worker=args.requests_per_worker,
            max_new_tokens=args.max_new_tokens,
            prompt=args.prompt,
        )
    )
    elapsed_s = time.perf_counter() - start
    _print_level_results(
        args.url, args.concurrency, total_requests, args.max_new_tokens,
        elapsed_s, latencies, errors,
    )


if __name__ == "__main__":
    main()
