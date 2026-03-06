"""Tests for /healthz endpoint."""

from fastapi.testclient import TestClient

from chronicle_runtime.server.main import app

client = TestClient(app)


def test_healthz_returns_ok():
    """GET /healthz returns {"ok": true}."""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
