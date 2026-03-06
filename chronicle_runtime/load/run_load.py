#!/usr/bin/env python3
"""HTTP load test: concurrent requests to /generate with latency percentiles."""

import argparse
import asyncio
import time

import httpx


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
    args = parser.parse_args()

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

    ok_latencies = [l for l in latencies if l >= 0]
    error_count = len(errors)
    ok_count = len(ok_latencies)

    print("=== Load test results ===")
    print(f"  url:              {args.url}")
    print(f"  concurrency:      {args.concurrency}")
    print(f"  total_requests:   {total_requests}")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    print(f"  elapsed_s:        {elapsed_s:.2f}")
    print(f"  throughput:       {ok_count / elapsed_s:.2f} req/s")
    print(f"  errors:           {error_count} ({100 * error_count / total_requests:.1f}%)")

    if ok_latencies:
        sorted_lat = sorted(ok_latencies)
        print(f"  latency p50:      {_percentile(sorted_lat, 50):.2f} ms")
        print(f"  latency p95:      {_percentile(sorted_lat, 95):.2f} ms")
        print(f"  latency p99:      {_percentile(sorted_lat, 99):.2f} ms")
    else:
        print("  (no successful requests)")


if __name__ == "__main__":
    main()
