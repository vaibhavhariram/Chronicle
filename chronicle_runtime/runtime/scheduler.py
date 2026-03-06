"""Micro-batch scheduler: queues requests and runs inference in batches."""

import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Optional

from chronicle_runtime.runtime.engine import batch_generate as engine_batch_generate

BATCH_WINDOW_MS = int(os.environ.get("BATCH_WINDOW_MS", "50"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))


@dataclass
class BatchRequest:
    """A single generate request queued for batching."""

    prompt: str
    max_new_tokens: int
    future: asyncio.Future


class Scheduler:
    """Queues requests and runs batch inference on a timer."""

    def __init__(
        self,
        batch_window_ms: Optional[int] = None,
        max_batch: Optional[int] = None,
        batch_sizes: Optional[list[int]] = None,
        batch_inference_fn: Optional[Callable[[list[str], list[int]], list[str]]] = None,
    ):
        self.batch_window_ms = batch_window_ms or BATCH_WINDOW_MS
        self.max_batch = max_batch or MAX_BATCH
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
        """Run batched inference and resolve futures."""
        prompts = [r.prompt for r in batch]
        max_new_tokens_list = [r.max_new_tokens for r in batch]
        loop = asyncio.get_event_loop()
        fn = self._batch_inference_fn
        try:
            texts = await loop.run_in_executor(
                None,
                lambda: fn(prompts, max_new_tokens_list),
            )
            for req, text in zip(batch, texts):
                req.future.set_result(text)
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

            deadline = asyncio.get_event_loop().time() + (
                self.batch_window_ms / 1000.0
            )
            while len(batch) < self.max_batch:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=remaining,
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

    async def enqueue(self, prompt: str, max_new_tokens: int) -> str:
        """Enqueue a request and return the generated text."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(
            BatchRequest(prompt=prompt, max_new_tokens=max_new_tokens, future=future)
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
