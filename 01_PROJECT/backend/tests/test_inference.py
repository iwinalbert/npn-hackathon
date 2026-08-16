"""
The live-inference boundary.

The verification run itself takes ~48 s, so it is marked `slow` and excluded
from the default suite. Everything around it — availability reporting, the
concurrency guard, job lifecycle, and the refusals — is always tested.
"""
import time

import pytest

from .conftest import API


def test_status_reports_availability_and_refusals(client):
    s = client.get(f"{API}/inference/status").json()
    assert "available" in s and "enabled" in s
    assert s["frozen_origin"] == "d_1941"
    assert s["supported_operations"] == ["verify"]
    refused = s["refused_operations"]
    assert "earlier_origin_inference" in refused
    assert "leak" in refused["earlier_origin_inference"].lower()
    assert "retraining" in refused
    assert "price_scenarios" in refused


def test_jobs_endpoint_is_listable(client):
    r = client.get(f"{API}/inference/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_unknown_job_returns_404_and_explains_volatility(client):
    r = client.get(f"{API}/inference/jobs/does_not_exist")
    assert r.status_code == 404
    assert "do not survive a restart" in r.json()["context"]["hint"]


def test_job_runner_reports_its_own_limits(client):
    s = client.get(f"{API}/inference/status").json()
    jobs = s["jobs"]
    assert jobs["durable"] is False
    assert jobs["max_concurrent"] >= 1
    assert "not shared across replicas" in jobs["note"]


@pytest.mark.slow
def test_verification_reproduces_the_frozen_forecast(client):
    """
    THE MODEL-SERVING PROOF.

    Loads the frozen boosters, rebuilds features from the raw panel, re-runs
    both members, blends at w=0.60 and compares against the shipped artefact.
    Expected result: an exact match.
    """
    status = client.get(f"{API}/inference/status").json()
    if not status["available"]:
        pytest.skip(f"inference unavailable: {status['reasons']}")

    r = client.post(f"{API}/inference/verify")
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # a second submission while one is running must be refused, not queued
    assert client.post(f"{API}/inference/verify").status_code == 409

    deadline = time.time() + 300
    job = None
    while time.time() < deadline:
        job = client.get(f"{API}/inference/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(2)

    assert job is not None and job["status"] == "succeeded", job
    res = job["result"]
    assert res["verdict"] == "MATCH", res
    assert res["max_abs_diff"] <= res["tolerance"]
    assert res["n_predictions"] == 853_720
    assert res["blend_weight_direct"] == 0.60
    assert res["leakage_checks"]["passed"] is True
    assert res["models"]["direct"]["features"] == 38
    assert res["models"]["recursive"]["features"] == 32
