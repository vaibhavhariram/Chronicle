"""FastAPI server for Chronicle runtime inference."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from chronicle_runtime.runtime.scheduler import get_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, stop on shutdown."""
    scheduler = get_scheduler()
    scheduler.start()
    yield
    await scheduler.stop()


app = FastAPI(title="Chronicle Runtime", version="0.1.0", lifespan=lifespan)


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
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text from prompt via micro-batch scheduler."""
    start = time.perf_counter()
    scheduler = get_scheduler()
    text = await scheduler.enqueue(prompt=req.prompt, max_new_tokens=req.max_new_tokens)
    latency_ms = (time.perf_counter() - start) * 1000
    return GenerateResponse(text=text, latency_ms=latency_ms)
