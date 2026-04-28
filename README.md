# Chronicle

LLM inference runtime. Timeout-based micro-batching, KV-cache reuse across decode steps.

**Stack:** Python, PyTorch, CUDA, FastAPI

## What it does

Continuous batching server for transformer inference. Requests land in a queue, get flushed to GPU on either a configurable time window or a max-batch-size trigger — whichever fires first. Starvation prevention via a fairness flush so old requests don't sit behind a never-filling batch.

KV-cache persists across decode steps within a request and is reused across the prefill→decode boundary.

## Scheduler

Three knobs:

- `window_ms` — max wait before flush
- `max_batch` — hard cap on batch size
- `fairness_ms` — oldest-request-age trigger, overrides `window_ms`

Flush condition: `len(queue) >= max_batch` OR `now - batch_start >= window_ms` OR `now - oldest_request >= fairness_ms`.

## Benchmarks

Mistral-7B, fp16, single A10 GPU.

| Setup | Throughput vs HF baseline |
|---|---|
| HuggingFace Transformers, batch 8 | 1.0x |
| Chronicle, batch 8 | **1.32x** |

Load test — async `httpx` harness, 100 concurrent requests:

| Metric | Value |
|---|---|
| p95 latency | <200ms |
| p99 latency | <350ms |

Adaptive batch scheduling holds p99 under load — no tail blowup at the configured concurrency.

## Run

```bash
pip install -r requirements.txt
python -m chronicle.server --model mistralai/Mistral-7B-v0.1 --window-ms 10 --max-batch 16
```

Server exposes `/generate` (POST, JSON: `prompt`, `max_tokens`).

## Load test

```bash
python bench/load.py --concurrency 100 --duration 60
```

Reports p50/p95/p99, throughput, and per-batch occupancy.

## Layout

```
chronicle/
  server.py        # FastAPI entrypoint
  scheduler.py     # micro-batching loop
  cache.py         # KV-cache manager
  model.py         # PyTorch model wrapper
bench/
  load.py          # async httpx harness
  compare_hf.py    # HuggingFace baseline
```
