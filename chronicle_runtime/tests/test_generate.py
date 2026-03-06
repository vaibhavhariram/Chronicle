"""Tests for /generate endpoint."""

from fastapi.testclient import TestClient

from chronicle_runtime.server.main import app

client = TestClient(app)


def test_generate_returns_stub():
    """POST /generate returns prompt + ' [stub]' and latency_ms."""
    resp = client.post(
        "/generate",
        json={"prompt": "Hello world", "max_new_tokens": 64},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello world [stub]"
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0
