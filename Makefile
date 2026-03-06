.PHONY: dev test install bench load

install:
	pip install -e .

dev:
	uvicorn chronicle_runtime.server.main:app --reload

test:
	pytest

bench:
	$(MAKE) bench-baseline bench-chronicle bench-report

bench-baseline:
	mkdir -p .bench && python -m chronicle_runtime.bench.run_baseline -n 10 --max-new-tokens 16 -o .bench/baseline.json

bench-chronicle:
	python -m chronicle_runtime.bench.run_chronicle -n 10 --max-new-tokens 16 -o .bench/chronicle.json

bench-report:
	python -m chronicle_runtime.bench.report .bench/baseline.json .bench/chronicle.json

load:
	python -m chronicle_runtime.load.run_load -c 4 -n 25 --max-new-tokens 32
