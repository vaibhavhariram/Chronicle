"""Tests for /generate endpoint."""

from fastapi.testclient import TestClient

from chronicle_runtime.server.main import app

client = TestClient(app)


def test_generate_returns_text_and_metrics():
    """POST /generate returns non-empty text and metrics fields."""
    resp = client.post(
        "/generate",
        json={"prompt": "Hello world", "max_new_tokens": 16},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert len(data["text"]) > 0
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0
    assert "queue_wait_ms" in data
    assert "compute_ms" in data
    assert "total_latency_ms" in data
    assert "batch_size" in data
    assert "tokens_generated" in data
