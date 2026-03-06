"""Micro-batch scheduler: queues requests and runs inference in batches."""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from chronicle_runtime.runtime.engine import batch_generate as engine_batch_generate
from chronicle_runtime.runtime.metrics import get_metrics

BATCH_WINDOW_MS = int(os.environ.get("BATCH_WINDOW_MS", "50"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))
MAX_QUEUE_WAIT_MS = int(os.environ.get("MAX_QUEUE_WAIT_MS", "5000"))

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single generate request queued for batching."""

    prompt: str
    max_new_tokens: int
    future: asyncio.Future
    enqueue_time: float = 0.0


class Scheduler:
    """Queues requests and runs batch inference on a timer."""

    def __init__(
        self,
        batch_window_ms: Optional[int] = None,
        max_batch: Optional[int] = None,
        max_queue_wait_ms: Optional[int] = None,
        batch_sizes: Optional[list[int]] = None,
        batch_inference_fn: Optional[
            Callable[[list[str], list[int]], tuple[list[str], list[int], float, float]]
        ] = None,
    ):
        self.batch_window_ms = batch_window_ms or BATCH_WINDOW_MS
        self.max_batch = max_batch or MAX_BATCH
        self.max_queue_wait_ms = max_queue_wait_ms or MAX_QUEUE_WAIT_MS
        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._batch_sizes: list[int] = batch_sizes if batch_sizes is not None else []
        self._batch_inference_fn = batch_inference_fn or engine_batch_generate
        self._task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def batch_sizes(self) -> list[int]:
        """Recorded batch sizes (for testing)."""
        return self._batch_sizes

    async def _run_batch_inference(self, batch: list[BatchRequest]) -> None:
        """Run batched inference and resolve futures with metrics."""
        # Sort by max_new_tokens to reduce tail effects (group similar requests)
        batch = sorted(batch, key=lambda r: r.max_new_tokens)
        prompts = [r.prompt for r in batch]
        max_new_tokens_list = [r.max_new_tokens for r in batch]
        compute_start = time.perf_counter()
        loop = asyncio.get_event_loop()
        fn = self._batch_inference_fn
        try:
            result = await loop.run_in_executor(
                None,
                lambda: fn(prompts, max_new_tokens_list),
            )
            texts, tokens_per_request, prefill_ms, decode_ms = result
            compute_ms = (time.perf_counter() - compute_start) * 1000
            total_tokens = sum(tokens_per_request)
            tok_per_sec = total_tokens / (compute_ms / 1000) if compute_ms > 0 else 0.0

            # Structured JSON logging
            log_event = {
                "batch_size": len(batch),
                "total_tokens": total_tokens,
                "prefill_ms": round(prefill_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "tok_per_sec": round(tok_per_sec, 2),
            }
            logger.info(json.dumps(log_event))

            # Record metrics
            per_request = []
            for i, req in enumerate(batch):
                queue_wait_ms = (compute_start - req.enqueue_time) * 1000
                per_request.append({
                    "queue_wait_ms": queue_wait_ms,
                    "compute_ms": compute_ms,
                    "total_ms": queue_wait_ms + compute_ms,
                })
            get_metrics().record_batch(
                batch_size=len(batch),
                total_tokens=total_tokens,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                tok_per_sec=tok_per_sec,
                per_request=per_request,
            )

            for i, req in enumerate(batch):
                queue_wait_ms = (compute_start - req.enqueue_time) * 1000
                req.future.set_result({
                    "text": texts[i],
                    "queue_wait_ms": queue_wait_ms,
                    "compute_ms": compute_ms,
                    "total_latency_ms": queue_wait_ms + compute_ms,
                    "batch_size": len(batch),
                    "tokens_generated": tokens_per_request[i],
                })
        except Exception as e:
            for req in batch:
                req.future.set_exception(e)

    async def _loop(self) -> None:
        """Background loop: gather requests and run batches."""
        while self._running:
            batch: list[BatchRequest] = []
            try:
                first = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.batch_window_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                continue
            batch.append(first)

            # Flush immediately if oldest request already exceeded max queue wait
            now = asyncio.get_event_loop().time()
            if (now - batch[0].enqueue_time) * 1000 < self.max_queue_wait_ms:
                deadline = now + (self.batch_window_ms / 1000.0)
                while len(batch) < self.max_batch:
                    now = asyncio.get_event_loop().time()
                    oldest_wait_s = now - batch[0].enqueue_time
                    remaining_until_flush = (self.max_queue_wait_ms / 1000.0) - oldest_wait_s
                    if remaining_until_flush <= 0:
                        break
                    remaining_deadline = deadline - now
                    timeout = min(remaining_deadline, remaining_until_flush)
                    if timeout <= 0:
                        break
                    try:
                        req = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=timeout,
                        )
                        batch.append(req)
                    except asyncio.TimeoutError:
                        break

            self._batch_sizes.append(len(batch))
            await self._run_batch_inference(batch)

    def start(self) -> None:
        """Start the background scheduler task."""
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def enqueue(
        self, prompt: str, max_new_tokens: int
    ) -> dict:
        """Enqueue a request and return result dict with text and metrics."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(
            BatchRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                future=future,
                enqueue_time=time.perf_counter(),
            )
        )
        return await future


# Global scheduler instance
_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Get or create the global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler


def set_scheduler(scheduler: Scheduler) -> None:
    """Set the global scheduler (for testing)."""
    global _scheduler
    _scheduler = scheduler
