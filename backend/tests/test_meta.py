from .conftest import API


def test_model_card_matches_the_frozen_champion(client):
    card = client.get(f"{API}/meta/model").json()
    assert card["validation_rmse"] == 2.0929
    assert card["validation_mae"] == 1.0395
    assert card["blend_weight_direct"] == 0.60
    assert card["blend_weight_recursive"] == 0.40
    assert card["status"] == "FROZEN"
    assert card["n_series"] == 30_490
    assert card["horizon_days"] == 28
    assert len(card["model_direct_sha256"]) == 64


def test_capability_matrix_declares_all_three_categories(client):
    caps = client.get(f"{API}/meta/capabilities").json()
    assert caps["implemented"] and caps["rejected"] and caps["not_supported"]
    for group in caps.values():
        for c in group:
            assert c["name"] and c["detail"]


def test_price_whatif_is_declared_unsupported(client):
    caps = client.get(f"{API}/meta/capabilities").json()
    names = " ".join(c["name"].lower() for c in caps["not_supported"])
    assert "price what-if" in names or "elasticity" in names


def test_prediction_intervals_are_declared_unsupported(client):
    caps = client.get(f"{API}/meta/capabilities").json()
    joined = " ".join(c["name"].lower() for c in caps["not_supported"])
    assert "prediction interval" in joined


def test_provenance_lists_sources_and_hashes(client):
    p = client.get(f"{API}/meta/provenance").json()
    assert len(p["backtest_origins"]) == 8
    assert p["row_counts"]["forecast"] == 30_490 * 28
    assert p["sources"]["model"].startswith("models/champion/")
