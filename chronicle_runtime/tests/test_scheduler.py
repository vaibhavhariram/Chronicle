"""Unit tests for the micro-batch scheduler."""

import asyncio

import pytest

from chronicle_runtime.runtime.scheduler import Scheduler


def _stub_inference(prompt: str, max_new_tokens: int) -> str:
    """Stub inference for fast tests."""
    return f"{prompt} [stub]"


@pytest.mark.asyncio
async def test_scheduler_batches_concurrent_requests():
    """Concurrent requests get collected into one batch."""
    batch_sizes: list[int] = []
    scheduler = Scheduler(
        batch_window_ms=100,
        max_batch=8,
        batch_sizes=batch_sizes,
        inference_fn=_stub_inference,
    )
    scheduler.start()
    try:
        # Fire 3 requests concurrently
        results = await asyncio.gather(
            scheduler.enqueue("a", 10),
            scheduler.enqueue("b", 10),
            scheduler.enqueue("c", 10),
        )
        assert results == ["a [stub]", "b [stub]", "c [stub]"]
        # At least one batch should have 2+ requests
        assert any(sz >= 2 for sz in batch_sizes), f"expected batch size >= 2, got {batch_sizes}"
    finally:
        await scheduler.stop()
