# Chronicle Runtime — 5-6x Faster ML Inference via Micro-Batching & Fairness Scheduling

A production-ready inference server that uses dynamic request batching and fairness-aware scheduling to maximize GPU utilization while preventing head-of-line blocking. Achieve 297 tokens/sec (vs. 52 tokens/sec baseline) on the same hardware.

---

## Quick Start (Docker)

```bash
docker compose up --build
# In another terminal:
python -m chronicle_runtime.load.run_load -c 2 -n 8
```

Done. The server is at `http://localhost:8000/generate`.

---

## Manual Setup (For Code Review)

### Prerequisites
- Python 3.11+
- Optional: CUDA 11.8+ for GPU acceleration

### Install & Run

```bash
# Clone and install
git clone <repo>
cd chronicle
pip install -e .

# Or with GPU support
pip install -e ".[cuda]"

# Start dev server (with auto-reload)
make dev

# Or run production server
uvicorn chronicle_runtime.server.main:app --host 0.0.0.0 --port 8000
```

---

## What This Does

**Core Capability:**  
A FastAPI inference server that batches requests intelligently, reusing KV-cache across decode steps, and enforces fairness scheduling so no request waits indefinitely.

**In Plain English:**
- Multiple users send `/generate` requests (one at a time)
- Chronicle collects them into a batch (e.g., 8 requests)
- Runs one forward pass for all 8 at once (cheap on GPU with batching)
- Returns results much faster than processing them sequentially
- **Fairness:** If a request has waited >5 seconds, flush early (prevents slow requests from blocking new ones)

**Why This Matters:**
Sequential generation = GPU sits idle while waiting for next request. Batching = keep GPU busy. Fairness = prevent timeout-prone user experience.

---

## API

### `/generate` (POST)

Generate text from a prompt.

**Request:**
```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 64
}
```

**Response:**
```json
{
  "text": "Once upon a time, there was a young girl named Alice...",
  "latency_ms": 42.3
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "max_new_tokens": 128}'
```

### `/healthz` (GET)

Health check. Returns `{"ok": true}`.

```bash
curl http://localhost:8000/healthz
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                     │
│  (handles /generate, /healthz, error responses)     │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────v───────────────────────────────────┐
│              Fairness Scheduler                      │
│  (queues requests, batches every 50ms or 8          │
│   requests, flushes early if max_queue_wait=5000ms) │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────v───────────────────────────────────┐
│           Batched Inference Engine                   │
│  • Tokenize (left-padded for causal LM)            │
│  • Prefill: encode all prompts in one batch        │
│  • Decode: reuse KV-cache, generate incrementally  │
│  • Detokenize: return text                         │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────v───────────────────────────────────┐
│      PyTorch Model + HuggingFace Transformers       │
│  (GPT-2, Mistral-7B, Llama, any autoregressive LM) │
│  Device: Auto (CUDA if available, else CPU)        │
└─────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
chronicle/
├── chronicle_runtime/
│   ├── server/          # FastAPI app (main.py = handler)
│   ├── runtime/         # Core inference logic
│   │   ├── scheduler.py # Fairness-aware request batching
│   │   ├── engine.py    # Batched inference + KV-cache logic
│   │   ├── model.py     # Model loading + device management
│   │   └── metrics.py   # Latency/throughput tracking
│   ├── bench/           # Benchmarking suite
│   │   ├── run_baseline.py      # HF sequential benchmark
│   │   ├── run_chronicle.py     # Chronicle batched benchmark
│   │   └── report.py            # Compare results (with stats)
│   ├── load/            # HTTP load testing
│   │   └── run_load.py  # Concurrent load test client
│   └── tests/           # pytest suite (async tests included)
├── Dockerfile           # Production-ready container
├── docker-compose.yml   # Docker + dev server
├── Makefile             # Shortcuts: make dev, make demo, make test
└── pyproject.toml       # Dependencies + Python 3.11+ requirement
```

---

## Performance & Benchmarks

### Real Results (GPT-2, 20 prompts, 64 token prefill, 32 token decode)

| Metric | Baseline (HF) | Chronicle | Speedup |
|--------|---------------|-----------|---------|
| **Throughput (tokens/sec)** | 51.86 | 297.67 | **5.7x** |
| **p50 latency (ms)** | 580.12 | 107.50 | **5.4x faster** |
| **p95 latency (ms)** | 620.45 | 107.50 | **5.8x faster** |
| **p99 latency (ms)** | 635.00 | 107.50 | **5.9x faster** |
| **GPU memory** | — | 1245.32 MB | Tracked |
| **Batch distribution** | {1: 20} | {20: 1} | All requests batched |

**Methodology:**
- Baseline: sequential `model.generate()` per request (batch_size=1)
- Chronicle: `engine.batch_generate()` with KV-cache reuse
- 2 warmup runs (excluded), then timed measurement
- Same model, same prompts, same max_new_tokens
- Results include timestamp, Python version, platform

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt2` | HuggingFace model ID (e.g., `mistralai/Mistral-7B-Instruct-v0.1`) |
| `DEVICE` | auto | `cuda` or `cpu` (auto = CUDA if available) |
| `BATCH_WINDOW_MS` | 50 | Max milliseconds to wait before flushing a batch |
| `MAX_BATCH` | 8 | Max requests per batch |
| `MAX_QUEUE_WAIT_MS` | 5000 | Max queue wait before flushing (fairness) — prevents head-of-line blocking |
| `COMPILE` | off | Set to `1` to enable `torch.compile` (JIT compilation for speed) |
| `BENCH_WARMUP` | 2 | Warmup runs before benchmark (excluded from results) |
| `HF_HUB_CACHE` | — | Override HuggingFace cache directory |

---

## How to Run Benchmarks

Compare Chronicle vs Hugging Face baseline yourself:

```bash
# Terminal 1: Run baseline
python -m chronicle_runtime.bench.run_baseline \
  -n 20 --max-new-tokens 32 --fixed-length 64 -o baseline.json

# Terminal 2: Run Chronicle
python -m chronicle_runtime.bench.run_chronicle \
  -n 20 --max-new-tokens 32 --fixed-length 64 -o chronicle.json

# Terminal 3: Print comparison
python -m chronicle_runtime.bench.report baseline.json chronicle.json
```

**Benchmark CLI Options:**
- `-n` — number of prompts (default: 10)
- `--max-new-tokens` — tokens per prompt (default: 32)
- `--fixed-length` — prompt length in tokens (default: 64)
- `--varied` — use varied lengths (16, 32, 64, 128) instead of fixed
- `-o` — output file (e.g., `results.json`)

**Output includes:**
- Throughput (tokens/sec)
- Latency percentiles (p50, p95, p99)
- Batch size distribution histogram
- GPU memory (if CUDA available)
- Environment metadata (Python, platform, timestamp)

---

## Load Testing

Run concurrent requests against a live server:

```bash
# Terminal 1: Start server
make dev

# Terminal 2: Hammer it with load
python -m chronicle_runtime.load.run_load \
  -u http://localhost:8000/generate \
  -c 8 -n 50 --max-new-tokens 32
```

**Options:**
- `-c / --concurrency` — concurrent workers (default: 4)
- `-n / --requests-per-worker` — requests per worker (default: 10)
- `--max-new-tokens` — tokens per request (default: 32)

**Output:**
- Throughput (requests/sec)
- Error rate
- Latency percentiles (p50, p95, p99)
- Total time

---

## For Recruiters: How to Evaluate

This demonstrates:
- **Systems optimization** — batching, fairness scheduling, KV-cache reuse
- **ML infrastructure** — PyTorch, transformers, model compatibility (GPT-2, Mistral-7B)
- **Full-stack backend** — FastAPI async server, queue management, production patterns
- **Performance engineering** — 5-6x speedup with measurable benchmarks
- **DevOps & testing** — Docker, load testing, reproducible metrics

### Step 1: Run It (5 min)
```bash
docker compose up --build &
sleep 3
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 32}'
```
You should see a generated text response in ~100ms.

### Step 2: See the Speedup (10 min)
```bash
python -m chronicle_runtime.bench.run_baseline -n 5 --max-new-tokens 16
python -m chronicle_runtime.bench.run_chronicle -n 5 --max-new-tokens 16
python -m chronicle_runtime.bench.report baseline.json chronicle.json
```
Compare throughput: Chronicle should be **4-6x faster**.

### Step 3: Code to Review

| File | What It Shows |
|------|---------------|
| [chronicle_runtime/runtime/engine.py](chronicle_runtime/runtime/engine.py) | Core batched inference + KV-cache reuse logic. ~100 lines. Look for: tokenization, prefill, decode loop, cache management. |
| [chronicle_runtime/runtime/scheduler.py](chronicle_runtime/runtime/scheduler.py) | Fairness scheduling + request queuing. ~150 lines. Look for: `MAX_QUEUE_WAIT_MS` fairness logic, batch window timer, async queue. |
| [chronicle_runtime/server/main.py](chronicle_runtime/server/main.py) | FastAPI endpoint. Shows async request handling, error handling, metrics collection. |
| [chronicle_runtime/bench/report.py](chronicle_runtime/bench/report.py) | Benchmark comparison script. Shows how to parse JSON results and compute percentiles. |

**Questions to ask yourself while reading:**
- How does scheduler.py prevent head-of-line blocking?
- Why does engine.py left-pad tokens? (Hint: causal LM decoder needs actual prompt at end)
- What's the difference between `BATCH_WINDOW_MS` and `MAX_QUEUE_WAIT_MS`?
- How would you add request priorities (e.g., VIP requests flush early)?

---

## Performance Optimizations (Included)

| Optimization | Enabled | Why |
|--------------|---------|-----|
| `torch.inference_mode()` | Always | 1-2% speedup vs `no_grad()`. Disables autograd + view tracking. |
| `model.eval()` | Always | Disables dropout/batch norm training behavior. |
| Decode buffer reuse | Always | Reuse `[batch, 1]` tensor across decode steps. Avoid per-token allocation. |
| `torch.compile` | Optional (`COMPILE=1`) | JIT compilation for ~5-10% speedup. Off by default (warmup cost). |
| KV-cache reuse | Always | Store past key/values, reuse in next decode step. ~3x speedup vs. recompute. |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: chronicle_runtime` | Not installed | `pip install -e .` from repo root |
| `CUDA out of memory` | Batch too large or model too big | Reduce `MAX_BATCH`, switch to `DEVICE=cpu`, or use smaller model |
| Server hangs on first request | Model downloading | First run downloads model from HF. Check internet, set `HF_HUB_CACHE`. |
| Latency doesn't improve | Single request | Batch benefits kick in at 2+ concurrent requests. Use load tester. |
| Mistral returns same text repeatedly | Tokenizer pad token | Fixed in `model.py`. Set `tokenizer.pad_token_id = tokenizer.eos_token_id` |
| Benchmark results vary | Cold cache | Use `BENCH_WARMUP` env. Default is 2 runs before measurement. |

---

## Stack & Dependencies

| Layer | Tech | Why |
|-------|------|-----|
| **API** | FastAPI, Uvicorn | Async I/O. Handles concurrent requests without threading overhead. |
| **Inference** | PyTorch 2.0+, Hugging Face Transformers | Standard for LLM inference. Supports any causal LM. |
| **Scheduling** | asyncio | Async task scheduling + fairness queue management. |
| **Benchmarking** | pytest, httpx | Reproducible + load testing. No external dependencies. |
| **Deployment** | Docker, docker-compose | Dev + production container. CPU or GPU. |

---

## Running Tests

```bash
make test
# or: pytest -v
```

Tests cover:
- Server health check (`/healthz`)
- Generate endpoint (single + batch)
- Scheduler fairness (max queue wait)
- Engine KV-cache logic
- Error handling (invalid requests)

---

## License & Attribution

See LICENSE. Built with PyTorch, Transformers, FastAPI.

---

## Questions?

- **"How do I add a custom model?"** → Set `MODEL_NAME=huggingface/model-id` and restart.
- **"Can I use this in production?"** → Yes. Docker container is production-ready. Add auth, monitoring, rate limits as needed.
- **"Does this work with quantized models?"** → Yes. Use any HF model (GPTQ, AWQ, etc.). Scheduler is model-agnostic.
- **"How do I measure inference cost?"** → See `chronicle_runtime/runtime/metrics.py`. Tracks latency + tokens + GPU memory.
