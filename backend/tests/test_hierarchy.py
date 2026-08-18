"""
Hierarchy navigation and — most importantly — COHERENCE.

The coherence tests are the scientific invariant of this product: every
aggregate must be an exact sum of the bottom-level forecasts. If one of them
ever fails, the hierarchy is lying to the user about what the model said.
"""
import pytest

from .conftest import API


def test_twelve_levels_exposed(client):
    levels = client.get(f"{API}/hierarchy/levels").json()
    by = {lv["level"]: lv["node_count"] for lv in levels}
    assert {"total", "state", "store", "item", "series"} <= set(by)
    assert by["total"] == 1
    assert by["state"] == 3
    assert by["store"] == 10
    assert by["category"] == 3
    assert by["department"] == 7
    assert by["item"] == 3_049
    assert by["series"] == 30_490


def test_store_nodes_are_the_ten_real_stores(client):
    nodes = client.get(f"{API}/hierarchy/nodes", params={"level": "store"}).json()
    ids = sorted(n["node_id"] for n in nodes)
    assert ids == ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3",
                   "WI_1", "WI_2", "WI_3"]


@pytest.mark.parametrize("level,node", [("store", "CA_3"), ("state", "TX"),
                                        ("category", "FOODS")])
def test_aggregate_returns_28_days(client, level, node):
    d = client.get(f"{API}/hierarchy/aggregate",
                   params={"level": level, "node_id": node}).json()
    assert len(d["forecast"]) == 28
    assert d["forecast"][0]["horizon"] == 1
    assert d["forecast"][-1]["horizon"] == 28
    assert d["forecast"][0]["date"] == "2016-05-23"
    assert d["forecast"][-1]["date"] == "2016-06-19"
    assert d["total_28d"] > 0


def test_coherence_states_sum_to_total(client):
    """Sum of the three states must equal the chain total, exactly."""
    total = client.get(f"{API}/hierarchy/aggregate",
                       params={"level": "total", "node_id": "ALL"}).json()
    parts = sum(
        client.get(f"{API}/hierarchy/aggregate",
                   params={"level": "state", "node_id": s}).json()["total_28d"]
        for s in ("CA", "TX", "WI")
    )
    assert total["total_28d"] == pytest.approx(parts, rel=1e-6)


def test_coherence_stores_sum_to_state(client):
    ca = client.get(f"{API}/hierarchy/aggregate",
                    params={"level": "state", "node_id": "CA"}).json()
    parts = sum(
        client.get(f"{API}/hierarchy/aggregate",
                   params={"level": "store", "node_id": s}).json()["total_28d"]
        for s in ("CA_1", "CA_2", "CA_3", "CA_4")
    )
    assert ca["total_28d"] == pytest.approx(parts, rel=1e-6)


def test_coherence_categories_sum_to_total(client):
    total = client.get(f"{API}/hierarchy/aggregate",
                       params={"level": "total", "node_id": "ALL"}).json()
    parts = sum(
        client.get(f"{API}/hierarchy/aggregate",
                   params={"level": "category", "node_id": c}).json()["total_28d"]
        for c in ("FOODS", "HOBBIES", "HOUSEHOLD")
    )
    assert total["total_28d"] == pytest.approx(parts, rel=1e-6)


def test_total_covers_every_series(client):
    total = client.get(f"{API}/hierarchy/aggregate",
                       params={"level": "total", "node_id": "ALL"}).json()
    assert total["n_series"] == 30_490


def test_accuracy_is_level_matched_and_rises_with_aggregation(client):
    """
    The same forecast is ~28% accurate per store-item and ~97% chain-wide.
    The API must report the figure for the level being viewed, and those figures
    must increase as aggregation rises. Showing a single global number is the
    most likely way a demand-forecasting UI misleads its user.
    """
    acc = {}
    for level, node in (("total", "ALL"), ("store", "CA_3"),
                        ("item", "FOODS_3_090")):
        d = client.get(f"{API}/hierarchy/aggregate",
                       params={"level": level, "node_id": node}).json()
        assert d["expected_accuracy"] is not None, f"no accuracy for {level}"
        acc[level] = d["expected_accuracy"]["accuracy_pct"]
    assert acc["total"] > acc["store"] > acc["item"]
    assert 95 < acc["total"] < 99
    assert 60 < acc["item"] < 80


def test_history_can_be_attached_to_an_aggregate(client):
    d = client.get(f"{API}/hierarchy/aggregate",
                   params={"level": "store", "node_id": "CA_1",
                           "history_days": 14}).json()
    assert len(d["history"]) == 14
    dates = [h["date"] for h in d["history"]]
    assert dates == sorted(dates)
    assert dates[-1] <= "2016-05-22", "history must not overlap the forecast"


def test_unknown_level_is_rejected_not_executed(client):
    """Level names are whitelisted, so injection attempts cannot reach SQL."""
    r = client.get(f"{API}/hierarchy/aggregate",
                   params={"level": "x'; DROP TABLE series; --", "node_id": "X"})
    assert r.status_code == 400
    # and the table is still there
    assert client.get(f"{API}/ready").json()["detail"]["tables"]["series"] == 30_490


def test_malformed_node_id_is_rejected(client):
    r = client.get(f"{API}/hierarchy/aggregate",
                   params={"level": "store_department", "node_id": "CA_1"})
    assert r.status_code == 400


def test_unknown_node_returns_404(client):
    r = client.get(f"{API}/hierarchy/aggregate",
                   params={"level": "store", "node_id": "NOPE_9"})
    assert r.status_code == 404


def test_search_requires_two_characters(client):
    assert client.get(f"{API}/hierarchy/search",
                      params={"q": "a"}).status_code == 422


def test_search_finds_a_known_item(client):
    hits = client.get(f"{API}/hierarchy/search",
                      params={"q": "FOODS_3_090"}).json()
    assert hits
    assert all("FOODS_3_090" in h["item_id"] for h in hits)
