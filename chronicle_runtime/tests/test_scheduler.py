"""Unit tests for the micro-batch scheduler."""

import asyncio

import pytest

from chronicle_runtime.runtime.scheduler import Scheduler


def _stub_batch_inference(
    prompts: list[str], max_new_tokens_list: list[int]
) -> tuple[list[str], list[int], float, float]:
    """Stub batch inference for fast tests."""
    texts = [f"{p} [stub]" for p in prompts]
    tokens = [1] * len(prompts)
    return texts, tokens, 0.0, 0.0


@pytest.mark.asyncio
async def test_scheduler_batches_concurrent_requests():
    """Concurrent requests get collected into one batch."""
    batch_sizes: list[int] = []
    scheduler = Scheduler(
        batch_window_ms=100,
        max_batch=8,
        batch_sizes=batch_sizes,
        batch_inference_fn=_stub_batch_inference,
    )
    scheduler.start()
    try:
        # Fire 3 requests concurrently
        results = await asyncio.gather(
            scheduler.enqueue("a", 10),
            scheduler.enqueue("b", 10),
            scheduler.enqueue("c", 10),
        )
        assert [r["text"] for r in results] == ["a [stub]", "b [stub]", "c [stub]"]
        # At least one batch should have 2+ requests
        assert any(sz >= 2 for sz in batch_sizes), f"expected batch size >= 2, got {batch_sizes}"
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_respects_max_queue_wait():
    """Request is not stuck beyond MAX_QUEUE_WAIT_MS."""
    max_queue_wait_ms = 80
    batch_window_ms = 500
    scheduler = Scheduler(
        batch_window_ms=batch_window_ms,
        max_batch=8,
        max_queue_wait_ms=max_queue_wait_ms,
        batch_inference_fn=_stub_batch_inference,
    )
    scheduler.start()
    try:
        result = await scheduler.enqueue("single", 5)
        assert result["text"] == "single [stub]"
        assert result["queue_wait_ms"] <= max_queue_wait_ms + 50
    finally:
        await scheduler.stop()
