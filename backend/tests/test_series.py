from .conftest import API


def test_series_detail_carries_regime_and_tier(client, sample_series):
    s = sample_series
    d = client.get(f"{API}/series/{s['store_id']}/{s['item_id']}").json()
    assert d["regime"] in {"smooth", "erratic", "intermittent", "lumpy",
                           "never sold"}
    assert d["volume_tier"] in {"very low", "low", "medium", "high"}
    assert d["regime_explanation"]


def test_history_is_chronological_and_stops_at_the_origin(client, sample_series):
    s = sample_series
    d = client.get(f"{API}/series/{s['store_id']}/{s['item_id']}/history",
                   params={"days": 45}).json()
    hist = d["history"]
    assert len(hist) == 45
    dates = [h["date"] for h in hist]
    assert dates == sorted(dates), "history must be chronological"
    assert all(h["sales"] >= 0 for h in hist)
    assert dates[-1] <= "2016-05-22", "history must not overlap the forecast"


def test_forecast_starts_the_day_after_the_origin(client, sample_series):
    s = sample_series
    d = client.get(f"{API}/series/{s['store_id']}/{s['item_id']}/forecast").json()
    assert len(d["forecast"]) == 28
    assert d["origin_day"] == "d_1941"
    assert d["origin_date"] == "2016-05-22"
    assert d["forecast"][0]["date"] == "2016-05-23"
    assert d["forecast"][-1]["date"] == "2016-06-19"
    assert all(p["yhat"] >= 0 for p in d["forecast"])
    assert d["total_28d"] > 0


def test_bands_bracket_the_point_forecast(client, sample_series):
    s = sample_series
    d = client.get(f"{API}/series/{s['store_id']}/{s['item_id']}/forecast").json()
    for p in d["forecast"]:
        assert p["lower"] <= p["yhat"] <= p["upper"], p
        assert p["lower"] >= 0, "demand cannot be negative"


def test_band_basis_denies_being_a_model_interval(client, sample_series):
    s = sample_series
    d = client.get(f"{API}/series/{s['store_id']}/{s['item_id']}/forecast").json()
    basis = d["band_basis"].lower()
    assert "not a model-produced prediction interval" in basis
    assert "observed model error" in basis
    assert d["band_regime"] == d["series"]["regime"]


def test_bands_can_be_disabled(client, sample_series):
    s = sample_series
    d = client.get(f"{API}/series/{s['store_id']}/{s['item_id']}/forecast",
                   params={"bands": False}).json()
    assert all(p["lower"] is None for p in d["forecast"])
    assert d["band_basis"] is None


def test_bands_scale_with_series_magnitude(client):
    big = client.get(f"{API}/series/CA_3/FOODS_3_090/forecast").json()
    small = client.get(f"{API}/series/CA_1/HOBBIES_1_001/forecast").json()
    bw = big["forecast"][0]["upper"] - big["forecast"][0]["lower"]
    sw = small["forecast"][0]["upper"] - small["forecast"][0]["lower"]
    assert big["forecast"][0]["yhat"] > small["forecast"][0]["yhat"]
    assert bw > sw, "a larger forecast must carry a wider absolute band"


def test_unknown_series_returns_404_with_a_hint(client):
    r = client.get(f"{API}/series/ZZ_9/NOT_AN_ITEM")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert "hint" in body["context"]


def test_series_listing_filters_are_applied(client):
    rows = client.get(f"{API}/series",
                      params={"store_id": "CA_1", "cat_id": "HOBBIES",
                              "limit": 20}).json()
    assert rows
    assert all(r["store_id"] == "CA_1" and r["cat_id"] == "HOBBIES" for r in rows)


def test_listing_limit_is_enforced(client):
    assert client.get(f"{API}/series",
                      params={"limit": 99999}).status_code == 422
