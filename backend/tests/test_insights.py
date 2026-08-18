import pytest

from .conftest import API


def test_portfolio_summary_totals_are_consistent(client):
    d = client.get(f"{API}/insights/summary",
                   params={"level": "store", "node_id": "CA_3"}).json()
    assert d["n_series"] == 3_049
    assert d["forecast_total_28d"] > 0
    assert d["forecast_daily_avg"] == pytest.approx(
        d["forecast_total_28d"] / 28, rel=1e-3)
    assert d["expected_accuracy"]["accuracy_pct"] > 90
    assert sum(r["n_series"] for r in d["regime_mix"]) == d["n_series"]


def test_summary_forecast_total_matches_the_aggregate_endpoint(client):
    a = client.get(f"{API}/hierarchy/aggregate",
                   params={"level": "store", "node_id": "CA_2"}).json()
    b = client.get(f"{API}/insights/summary",
                   params={"level": "store", "node_id": "CA_2"}).json()
    assert b["forecast_total_28d"] == pytest.approx(a["total_28d"], rel=1e-4)


def test_top_movers_are_ranked_by_absolute_change(client):
    d = client.get(f"{API}/insights/top-movers",
                   params={"level": "store", "node_id": "CA_1",
                           "limit": 10}).json()
    assert len(d["rising"]) <= 10 and len(d["falling"]) <= 10
    ups = [r["delta_daily"] for r in d["rising"]]
    downs = [r["delta_daily"] for r in d["falling"]]
    assert ups == sorted(ups, reverse=True)
    assert downs == sorted(downs)
    assert all(u > 0 for u in ups)
    assert all(x < 0 for x in downs)
    assert "absolute unit change" in d["basis"]


def test_top_movers_direction_filter(client):
    d = client.get(f"{API}/insights/top-movers",
                   params={"direction": "up", "limit": 5}).json()
    assert "rising" in d and "falling" not in d


def test_planning_summary_range_brackets_the_expectation(client):
    d = client.get(f"{API}/insights/planning/CA_3/FOODS_3_090").json()
    assert d["horizon_days"] == 28
    assert d["planning_range"]["low"] <= d["expected_total"] <= d["planning_range"]["high"]
    assert len(d["weekly_breakdown"]) == 4
    assert sum(w["expected"] for w in d["weekly_breakdown"]) == pytest.approx(
        d["expected_total"], rel=1e-3)


def test_planning_range_is_labelled_as_measured_error(client):
    d = client.get(f"{API}/insights/planning/CA_3/FOODS_3_090").json()
    basis = d["planning_range"]["basis"]
    assert "NOT a model-produced prediction interval" in basis
    assert "NOT a service-" in basis


def test_planning_summary_states_its_caveats(client):
    d = client.get(f"{API}/insights/planning/CA_3/FOODS_3_090").json()
    joined = " ".join(d["caveats"]).lower()
    assert "no ground truth" in joined
    assert "out of stock" in joined


def test_planning_matches_the_series_forecast_total(client):
    f = client.get(f"{API}/series/CA_1/HOBBIES_1_001/forecast").json()
    p = client.get(f"{API}/insights/planning/CA_1/HOBBIES_1_001").json()
    assert p["expected_total"] == pytest.approx(f["total_28d"], abs=0.01)


def test_planning_unknown_series_404(client):
    assert client.get(f"{API}/insights/planning/ZZ_9/NOPE").status_code == 404
