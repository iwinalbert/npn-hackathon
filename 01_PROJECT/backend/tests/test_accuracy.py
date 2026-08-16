"""
Model-performance endpoints.

These tests guard the two honesty rules the accuracy layer exists to enforce:
no accuracy is ever reported for a window without ground truth, and accuracy is
always qualified by aggregation level.
"""
import pytest

from .conftest import API

PRIMARY = 1912


def test_eight_backtest_windows_are_available(client):
    w = client.get(f"{API}/accuracy/windows").json()
    assert len(w) == 8
    for x in w:
        assert x["n_predictions"] == 853_720
        assert 0 < x["rmse"] < 5
        assert x["window_start"] < x["window_end"]


def test_primary_window_reports_the_published_metrics(client):
    w = client.get(f"{API}/accuracy/windows").json()
    primary = [x for x in w if x["is_primary_validation_window"]]
    assert len(primary) == 1
    p = primary[0]
    assert p["rmse"] == pytest.approx(2.0929, abs=5e-4)
    assert p["mae"] == pytest.approx(1.0395, abs=5e-4)
    assert p["origin_day"] == "d_1913"


def test_no_window_covers_the_delivered_forecast_period(client):
    """
    The delivered forecast (d_1942-d_1969) has no ground truth, so it must never
    appear as a scorable window.
    """
    w = client.get(f"{API}/accuracy/windows").json()
    for x in w:
        assert x["window_end"] <= "2016-05-22", (
            f"{x['origin_day']} extends past the last day with known sales")


def test_accuracy_by_level_spans_bottom_to_chain(client):
    body = client.get(f"{API}/accuracy/levels").json()
    levels = body["levels"]
    assert len(levels) >= 12
    by = {lv["level"]: lv["accuracy_pct"] for lv in levels}
    assert by["L1_total"] > 95
    assert by["L12_store_item"] < 35
    assert by["L1_total"] > by["L10_item"] > by["L12_store_item"]
    assert "level that matches the decision" in body["note"]


def test_horizon_error_is_reported_for_all_28_days(client):
    h = client.get(f"{API}/accuracy/horizon").json()
    assert len(h) == 28
    assert [x["horizon"] for x in h] == list(range(1, 29))
    assert all(x["rmse"] > 0 for x in h)


def test_regime_accuracy_covers_the_syntetos_boylan_classes(client):
    body = client.get(f"{API}/accuracy/regimes").json()
    regimes = {r["regime"] for r in body["regimes"]}
    assert {"smooth", "erratic", "intermittent", "lumpy"} <= regimes
    total_share = sum(r["share_of_squared_error_pct"] for r in body["regimes"])
    assert total_share == pytest.approx(100.0, abs=0.5)


def test_member_decomposition_matches_the_research_record(client):
    """
    The blend must still beat both members, with the residual correlation the
    research measured (~0.95). This is the evidence that the ensemble works for
    the stated reason.
    """
    m = client.get(f"{API}/accuracy/members").json()
    direct = next(x for x in m["members"] if x["name"].startswith("Direct"))
    recursive = next(x for x in m["members"] if x["name"].startswith("Recursive"))
    assert direct["weight"] == 0.60
    assert recursive["weight"] == 0.40
    assert direct["rmse"] == pytest.approx(2.1211, abs=1e-3)
    assert recursive["rmse"] == pytest.approx(2.1185, abs=1e-3)
    assert m["blend"]["rmse"] == pytest.approx(2.0929, abs=1e-3)
    assert m["blend"]["rmse"] < min(direct["rmse"], recursive["rmse"])
    assert 0.90 < m["residual_correlation"] < 0.99
    assert m["gain_vs_best_member"] < 0


def test_series_backtest_has_actuals_and_member_split(client):
    d = client.get(f"{API}/accuracy/backtest/CA_3/FOODS_3_090").json()
    assert len(d["points"]) == 28
    for p in d["points"]:
        assert p["actual"] >= 0
        assert p["predicted"] >= 0
        assert "predicted_direct" in p and "predicted_recursive" in p
    assert d["rmse"] > 0
    assert "never saw the days they are scored against" in d["basis"]


def test_aggregate_backtest_reports_level_accuracy(client):
    d = client.get(f"{API}/accuracy/backtest",
                   params={"level": "store", "node_id": "CA_1"}).json()
    assert len(d["points"]) == 28
    assert d["n_series"] == 3_049
    assert 0 < d["accuracy_pct"] < 100
    # store-level aggregation must beat bottom-level accuracy
    assert d["accuracy_pct"] > 50


def test_unknown_backtest_origin_is_rejected_with_valid_options(client):
    r = client.get(f"{API}/accuracy/horizon", params={"origin_idx": 1940})
    assert r.status_code == 404
    body = r.json()
    assert "valid_origins" in body["context"]
    assert "ground truth exists" in body["context"]["hint"]


def test_error_band_table_is_exposed_with_its_disclaimer(client):
    body = client.get(f"{API}/accuracy/error-bands",
                      params={"regime": "smooth"}).json()
    assert len(body["bands"]) == 28
    assert body["measured_coverage"] == 0.90
    assert "NOT a model-produced prediction interval" in body["disclaimer"]
    for b in body["bands"]:
        assert b["q05"] < b["q50"] < b["q95"]


def test_unknown_regime_is_rejected(client):
    r = client.get(f"{API}/accuracy/error-bands", params={"regime": "nonsense"})
    assert r.status_code == 400
    assert "valid_regimes" in r.json()["context"]
