"""Smoke tests for the server with real model inference."""

from fastapi.testclient import TestClient

from chronicle_runtime.server.main import app

client = TestClient(app)


def test_generate_returns_200_and_non_empty_text():
    """POST /generate returns status 200 and non-empty text."""
    resp = client.post(
        "/generate",
        json={"prompt": "Hello", "max_new_tokens": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert len(data["text"]) > 0
    assert "latency_ms" in data
