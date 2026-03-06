"""FastAPI server for Chronicle runtime inference."""

import time
from fastapi import FastAPI
from pydantic import BaseModel

from chronicle_runtime.runtime.model import generate as model_generate

app = FastAPI(title="Chronicle Runtime", version="0.1.0")


class GenerateRequest(BaseModel):
    """Request body for /generate."""

    prompt: str
    max_new_tokens: int = 128


class GenerateResponse(BaseModel):
    """Response from /generate."""

    text: str
    latency_ms: float


@app.get("/healthz")
def healthz() -> dict:
    """Health check endpoint."""
    return {"ok": True}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text from prompt using Hugging Face model."""
    start = time.perf_counter()
    text = model_generate(prompt=req.prompt, max_new_tokens=req.max_new_tokens, do_sample=False)
    latency_ms = (time.perf_counter() - start) * 1000
    return GenerateResponse(text=text, latency_ms=latency_ms)
