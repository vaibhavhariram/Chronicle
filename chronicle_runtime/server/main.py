"""FastAPI server for Chronicle runtime inference."""

import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Chronicle Runtime", version="0.1.0")


class GenerateRequest(BaseModel):
    """Request body for /generate."""

    prompt: str
    max_new_tokens: int = 128


class GenerateResponse(BaseModel):
    """Response from /generate."""

    text: str
    latency_ms: float


def _generate_stub(prompt: str, max_new_tokens: int) -> str:
    """Placeholder generate function. Returns prompt + ' [stub]'."""
    return f"{prompt} [stub]"


@app.get("/healthz")
def healthz() -> dict:
    """Health check endpoint."""
    return {"ok": True}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text from prompt. Currently a stub implementation."""
    start = time.perf_counter()
    text = _generate_stub(req.prompt, req.max_new_tokens)
    latency_ms = (time.perf_counter() - start) * 1000
    return GenerateResponse(text=text, latency_ms=latency_ms)
