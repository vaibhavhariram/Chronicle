#!/usr/bin/env bash
# Chronicle Runtime demo: server + benchmark + load test
set -e

cd "$(dirname "$0")/.."
export DEVICE=cpu
export MODEL_NAME=gpt2

# Use venv if present
if [ -d .venv ]; then
  PYTHON=".venv/bin/python"
  UVICORN=".venv/bin/uvicorn"
else
  PYTHON="python"
  UVICORN="uvicorn"
fi

echo "=== Chronicle Runtime Demo ==="
echo ""

# 1. Start server in background
echo "[1/4] Starting server..."
$UVICORN chronicle_runtime.server.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

# Wait for server ready
for i in $(seq 1 60); do
  if curl -s http://localhost:8000/healthz 2>/dev/null | grep -q '"ok":true'; then
    break
  fi
  sleep 1
  if [ $i -eq 60 ]; then
    echo "Server failed to start"
    exit 1
  fi
done
echo "    Server ready"
echo ""

# 2. Run benchmark
echo "[2/4] Running benchmark (Chronicle vs baseline)..."
mkdir -p .bench
$PYTHON -m chronicle_runtime.bench.run_chronicle -n 4 --max-new-tokens 8 -o .bench/chronicle.json 2>/dev/null
$PYTHON -m chronicle_runtime.bench.run_baseline -n 4 --max-new-tokens 8 -o .bench/baseline.json 2>/dev/null
$PYTHON -m chronicle_runtime.bench.report .bench/chronicle.json .bench/baseline.json 2>/dev/null
echo ""

# 3. Warmup + load test
echo "[3/4] Running load test..."
curl -s -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt":"Hi","max_new_tokens":2}' >/dev/null || true
LOAD_OUT=$($PYTHON -m chronicle_runtime.load.run_load -c 2 -n 4 --max-new-tokens 8 2>/dev/null)
echo "$LOAD_OUT"
echo ""

# 4. Headline numbers
echo "[4/4] Headline numbers"
echo "----------------------"
CHRONICLE_TOKENS=$($PYTHON -c "import json; d=json.load(open('.bench/chronicle.json')); print(round(d.get('tokens_per_sec',0),1))" 2>/dev/null || echo "0")
CHRONICLE_P95=$($PYTHON -c "
import json
d=json.load(open('.bench/chronicle.json'))
lat=sorted([x for x in d.get('latencies_ms',[]) if x>=0])
print(round(lat[int((len(lat)-1)*0.95)] if lat else 0,1))
" 2>/dev/null || echo "0")
LOAD_THROUGHPUT=$(echo "$LOAD_OUT" | grep "throughput:" | awk '{print $2}')
LOAD_P95=$(echo "$LOAD_OUT" | grep "latency p95:" | awk '{print $3}')
LOAD_CONCURRENCY=$(echo "$LOAD_OUT" | grep "concurrency:" | awk '{print $2}')

echo "  Chronicle tokens/sec:  $CHRONICLE_TOKENS"
echo "  Chronicle p95 (ms):    $CHRONICLE_P95"
echo "  Load test req/s:       ${LOAD_THROUGHPUT:-N/A}"
echo "  Load test p95 (ms):    ${LOAD_P95:-N/A}"
echo "  Load concurrency:      ${LOAD_CONCURRENCY:-N/A}"
echo ""
echo "Demo complete."
