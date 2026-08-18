import csv
import hashlib
from pathlib import Path

import pytest

from app.config import settings
from .conftest import API

ROOT = Path(settings.project_root)
MANIFEST = (ROOT.parent / "docs" / "02_MODEL" / "FROZEN_CHAMPION"
            / "CHAMPION_MANIFEST.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_model_files_exist():
    if not settings.model_direct.exists() or not settings.model_recursive.exists():
        pytest.skip("frozen model binaries not present — gitignored, and not part "
                    "of the deployable unit; see infra/docs/ci-cd.md")
    assert settings.model_direct.exists(), "frozen direct member is missing"
    assert settings.model_recursive.exists(), "frozen recursive member is missing"


def test_champion_manifest_hashes_still_match_the_model_files():
    import json
    if not MANIFEST.exists():
        pytest.skip("champion manifest not present")
    if not settings.model_direct.exists() or not settings.model_recursive.exists():
        pytest.skip("frozen model binaries not present — gitignored, and not part "
                    "of the deployable unit; see infra/docs/ci-cd.md")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_source = {Path(f["canonical_source"]).name: f["sha256"]
                 for f in man["files"]}
    assert _sha256(settings.model_direct) == by_source[settings.model_direct.name]
    assert (_sha256(settings.model_recursive)
            == by_source[settings.model_recursive.name])


def test_served_model_card_hashes_match_the_files_on_disk(client):
    card = client.get(f"{API}/meta/model").json()
    assert card["model_direct_sha256"] == _sha256(settings.model_direct)
    assert card["model_recursive_sha256"] == _sha256(settings.model_recursive)
    assert card["forecast_sha256"] == _sha256(settings.forecast_csv)


def test_served_forecast_matches_the_frozen_csv_row_for_row(client):
    with settings.forecast_csv.open(newline="", encoding="utf-8") as fh:
        rows = {r["id"]: r for _, r in zip(range(400), csv.DictReader(fh))}

    checked = 0
    for series_id, row in list(rows.items())[:5]:
        base = series_id.replace("_evaluation", "")
        parts = base.split("_")
        store_id = "_".join(parts[-2:])
        item_id = "_".join(parts[:-2])
        r = client.get(f"{API}/series/{store_id}/{item_id}/forecast")
        assert r.status_code == 200, f"{store_id}/{item_id} -> {r.status_code}"
        served = r.json()["forecast"]
        for h in range(1, 29):
            expected = float(row[f"F{h}"])
            assert served[h - 1]["yhat"] == pytest.approx(expected, abs=1e-3), (
                f"{item_id}/{store_id} h{h}: served {served[h-1]['yhat']} "
                f"!= frozen {expected}"
            )
        checked += 1
    assert checked == 5


def test_forecast_totals_match_the_frozen_artefact(client):
    total = 0.0
    with settings.forecast_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += sum(float(row[f"F{h}"]) for h in range(1, 29))
    served = client.get(f"{API}/hierarchy/aggregate",
                        params={"level": "total", "node_id": "ALL"}).json()
    assert served["total_28d"] == pytest.approx(total, rel=1e-6)


def test_no_negative_or_missing_forecasts(client):
    from app import db
    row = db.query_one(
        "SELECT count(*) AS bad FROM forecast "
        "WHERE yhat IS NULL OR yhat < 0 OR isnan(yhat)")
    assert row["bad"] == 0


def test_every_series_has_exactly_28_forecast_days(client):
    from app import db
    row = db.query_one(
        "SELECT count(*) AS bad FROM ("
        "  SELECT series_idx, count(*) c FROM forecast GROUP BY 1"
        ") WHERE c <> 28")
    assert row["bad"] == 0


def test_backtest_reproduces_the_published_champion_metrics(client):
    from app import db
    row = db.query_one(
        "SELECT sqrt(avg((y_true - p_blend) * (y_true - p_blend))) AS rmse, "
        "       avg(abs(y_true - p_blend)) AS mae, count(*) AS n "
        f"FROM {db.backtest_source()} WHERE origin_idx = 1912")
    assert row["n"] == 853_720
    assert row["rmse"] == pytest.approx(2.0929, abs=5e-4)
    assert row["mae"] == pytest.approx(1.0395, abs=5e-4)


def test_blend_weight_is_still_0_60(client):
    from app import db
    row = db.query_one(
        "SELECT max(abs(p_blend - (0.60 * p_direct + 0.40 * p_recursive))) AS d "
        f"FROM {db.backtest_source()} WHERE origin_idx = 1912")
    assert row["d"] < 1e-4, f"blend weight drift detected: max diff {row['d']}"


def test_error_bands_are_calibrated(client):
    from app import db
    row = db.query_one(f"""
        SELECT avg(CASE WHEN b.y_true
                          BETWEEN greatest(0, b.p_blend + e.q05 * sqrt(greatest(b.p_blend, 1.0)))
                              AND b.p_blend + e.q95 * sqrt(greatest(b.p_blend, 1.0))
                        THEN 1.0 ELSE 0.0 END) AS coverage
        FROM {db.backtest_source()} b
        JOIN series s USING (series_idx)
        JOIN error_bands e ON e.regime = s.regime AND e.horizon = b.horizon
        USING SAMPLE 300000 ROWS
    """)
    assert 0.86 <= row["coverage"] <= 0.94, f"band coverage {row['coverage']:.3f}"
