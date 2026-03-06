# Chronicle Runtime

ML inference server for Chronicle. FastAPI + PyTorch + transformers.

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

## Structure

```
chronicle_runtime/
  server/     # FastAPI app and endpoints
  runtime/    # Inference logic
  bench/      # Benchmarking
  load/       # Load testing
  tests/      # pytest
```
