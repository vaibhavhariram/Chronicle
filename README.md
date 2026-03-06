# Chronicle Runtime

ML inference server for Chronicle. FastAPI + PyTorch + transformers.

## Quick Demo

Two commands to run the full demo (server + benchmark + load test):

```bash
pip install -e .
make demo
```

Or with Docker:

```bash
docker compose up --build
# In another terminal: python -m chronicle_runtime.load.run_load -c 2 -n 8
```

## Requirements

- Python 3.11+
- Optional: CUDA for GPU acceleration (`pip install -e ".[cuda]"`)

## Setup

```bash
pip install -e .
```

With CUDA:

```bash
pip install -e ".[cuda]"
```

## Run

**Start the server (dev mode with reload):**

```bash
make dev
# or: uvicorn chronicle_runtime.server.main:app --reload
```

**Run tests:**

```bash
make test
# or: pytest
```

## Configuration (env vars)

| Variable   | Default | Description                          |
|------------|---------|--------------------------------------|
| `MODEL_NAME` | `gpt2` | Hugging Face model ID                |
| `DEVICE`   | auto    | `cuda` or `cpu` (auto = cuda if available) |
| `HF_HUB_CACHE` | —    | Override Hugging Face cache directory |
| `BATCH_WINDOW_MS` | 50 | Max ms to wait for more requests before flushing |
| `MAX_BATCH` | 8 | Max requests per batch |
| `MAX_QUEUE_WAIT_MS` | 5000 | Max queue wait before flushing (fairness) |
| `COMPILE` | off | Set to `1` to enable `torch.compile` on the model |
| `BENCH_WARMUP` | 2 | Warmup runs before timed benchmark (excluded from results) |

## Performance optimizations

| Optimization | Status | Why |
|--------------|--------|-----|
| `torch.inference_mode()` | Always on | Faster than `no_grad()`; disables autograd and view tracking for inference |
| `model.eval()` | Always on | Disables dropout and batch norm training behavior |
| Decode input buffer reuse | Always on | Reuses `[B, 1]` tensor across decode steps to avoid per-step allocation |
| `torch.compile` | Optional (`COMPILE=1`) | JIT compilation can speed up repeated forward passes; off by default due to warmup cost |

Benchmarks report `gpu_mem_mb` (peak GPU memory) when CUDA is available.

## API

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/healthz` | Health check. Returns `{"ok": true}` |
| POST   | `/generate`| Generate text. Body: `{"prompt": str, "max_new_tokens": int}` |

### Example

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 64}'
# {"text":"Hello, world! ...","latency_ms":42.3}
```

## How to run benchmarks

Compare Chronicle (batched KV-cache) vs HF baseline:

```bash
# Run baseline (sequential model.generate)
python -m chronicle_runtime.bench.run_baseline -n 20 --max-new-tokens 32 -o baseline.json

# Run Chronicle (batched engine.batch_generate)
python -m chronicle_runtime.bench.run_chronicle -n 20 --max-new-tokens 32 -o chronicle.json

# Print summary
python -m chronicle_runtime.bench.report baseline.json chronicle.json
```

Options:
- `-n` number of prompts (default: 10)
- `--max-new-tokens` tokens to generate per prompt (default: 32)
- `--fixed-length` prompt length in tokens (default: 64)
- `--varied` use varied prompt lengths (16, 32, 64, 128)

### Sample output

```
=== BASELINE ===
  model_name:           gpt2
  device:               cpu
  prompt_token_length:  64
  max_new_tokens:       32
  num_prompts:          20
  tokens/sec:           51.86
  latency p50 (ms):     580.12
  latency p95 (ms):     620.45
  latency p99 (ms):     635.00
  batch_size_dist:      {1: 20}

=== CHRONICLE ===
  model_name:           gpt2
  device:               cuda:0
  prompt_token_length:  64
  max_new_tokens:       32
  num_prompts:          20
  tokens/sec:           297.67
  latency p50 (ms):     107.50
  latency p95 (ms):     107.50
  latency p99 (ms):     107.50
  batch_size_dist:      {20: 1}
  gpu_mem_mb:           1245.32
```

Results are saved to `.bench/results.json` with timestamp and environment details.

## Metrics methodology

Benchmark results are designed to be **resume-defensible and hard to dispute**. Each run records:

- **Model and device**: `model_name`, `device` (from `MODEL_NAME`, `DEVICE` env)
- **Input config**: `prompt_token_length` (fixed or varied), `max_new_tokens`, `num_prompts`
- **Throughput**: `tokens/sec` = total generated tokens / wall-clock time
- **Latency**: p50/p95/p99 (ms) from per-request end-to-end times
- **Batch distribution**: `batch_size_dist` histogram
- **GPU memory**: peak `gpu_mem_mb` when CUDA available

**Baseline vs Chronicle comparison**:
- Same model, same prompts, same `max_new_tokens`
- Baseline: sequential `model.generate()` per request (batch_size=1)
- Chronicle: batched `engine.batch_generate()` with KV-cache reuse

**Warmup**:
- `BENCH_WARMUP` env (default 2) warmup runs before timed measurement
- Excludes cold-start (model load, JIT compile) from results

**Latency measurement**:
- Per-request: wall-clock from start to completion
- Baseline: one request per call
- Chronicle: amortized batch time per request (batch_time / batch_size)

**Environment**:
- `results.json` includes `timestamp`, `environment` (python version, platform)
- Reproducible: set `MODEL_NAME`, `DEVICE`, `BENCH_WARMUP` before running

## Load test

HTTP load test against a running server (e.g. `make dev` in another terminal):

```bash
python -m chronicle_runtime.load.run_load -u http://localhost:8000/generate -c 4 -n 25 --max-new-tokens 32
```

Options:
- `-u` / `--url` endpoint URL (default: http://localhost:8000/generate)
- `-c` / `--concurrency` number of concurrent workers (default: 4)
- `-n` / `--requests-per-worker` requests per worker (default: 10)
- `--max-new-tokens` tokens to generate per request (default: 32)

Reports throughput (req/s), error rate, and latency p50/p95/p99.

## Structure

```
chronicle_runtime/
  server/     # FastAPI app and endpoints
  runtime/    # Inference logic
  bench/      # Benchmarking
  load/       # Load testing
  tests/      # pytest
```
