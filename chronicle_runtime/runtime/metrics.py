"""Instrumentation metrics for inference."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Metrics:
    """Aggregated metrics for reporting."""

    tokens_generated_total: int = 0
    requests_total: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    decode_tokens_per_sec: list[float] = field(default_factory=list)
    request_latencies: list[dict] = field(default_factory=list)
    queue_wait_ms: list[float] = field(default_factory=list)

    def record_batch(
        self,
        batch_size: int,
        total_tokens: int,
        prefill_ms: float,
        decode_ms: float,
        tok_per_sec: float,
        per_request: list[dict],
    ) -> None:
        """Record a completed batch and its per-request latencies."""
        self.tokens_generated_total += total_tokens
        self.requests_total += batch_size
        self.batch_sizes.append(batch_size)
        self.decode_tokens_per_sec.append(tok_per_sec)
        self.request_latencies.extend(per_request)
        for pr in per_request:
            if "queue_wait_ms" in pr:
                self.queue_wait_ms.append(pr["queue_wait_ms"])


_metrics: Optional[Metrics] = None


def get_metrics() -> Metrics:
    """Get or create the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


def reset_metrics() -> None:
    """Reset metrics (for testing)."""
    global _metrics
    _metrics = Metrics()
