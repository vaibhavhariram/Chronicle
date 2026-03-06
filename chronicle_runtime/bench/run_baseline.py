#!/usr/bin/env python3
"""Benchmark HF baseline: sequential model.generate per request."""

import argparse
import json
import sys
import time

from chronicle_runtime.runtime.model import generate as model_generate, load_model
from chronicle_runtime.bench.prompts import make_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-prompts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--fixed-length", type=int, default=64)
    parser.add_argument("--varied", action="store_true", help="Use varied prompt lengths")
    parser.add_argument("-o", "--output", default="-", help="Output JSON file (- for stdout)")
    args = parser.parse_args()

    prompts = make_prompts(args.num_prompts, args.fixed_length, args.varied)
    max_new_tokens_list = [args.max_new_tokens] * len(prompts)
    tokenizer, _ = load_model()

    latencies_ms: list[float] = []
    tokens_per_request: list[int] = []

    start = time.perf_counter()
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        text = model_generate(prompt, max_new_tokens=max_new_tokens_list[i], do_sample=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        prompt_len = len(tokenizer.encode(prompt))
        output_len = len(tokenizer.encode(text))
        tokens_per_request.append(max(0, output_len - prompt_len))
    total_s = time.perf_counter() - start

    total_tokens = sum(tokens_per_request)
    tokens_per_sec = total_tokens / total_s if total_s > 0 else 0

    result = {
        "backend": "baseline",
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "fixed_length": args.fixed_length,
        "varied": args.varied,
        "total_tokens": total_tokens,
        "total_s": total_s,
        "tokens_per_sec": tokens_per_sec,
        "latencies_ms": latencies_ms,
        "batch_sizes": [1] * len(prompts),
    }

    out = json.dumps(result, indent=2)
    if args.output == "-":
        print(out)
    else:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
