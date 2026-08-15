"""Liveness, readiness and the error contract."""
from .conftest import API


def test_health_is_ok(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_all_tables(client):
    r = client.get(f"{API}/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    tables = body["detail"]["tables"]
    assert tables["series"] == 30_490
    assert tables["forecast"] == 30_490 * 28
    assert tables["backtest"] > 6_000_000


def test_every_response_carries_a_request_id(client):
    r = client.get(f"{API}/health")
    assert r.headers.get("X-Request-ID")
    assert float(r.headers["X-Response-Time-ms"]) >= 0


def test_root_advertises_the_frozen_model(client):
    body = client.get("/").json()
    assert "FROZEN" in body["model"]
