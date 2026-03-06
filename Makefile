.PHONY: dev test install

install:
	pip install -e .

dev:
	uvicorn chronicle_runtime.server.main:app --reload

test:
	pytest
