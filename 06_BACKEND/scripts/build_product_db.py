"""
BUILD THE PRODUCT DATABASE.

Reads the protected research artefacts READ-ONLY and materialises everything the
API needs into 06_BACKEND/data/product.duckdb.

    python 06_BACKEND/scripts/build_product_db.py

PROTECTION CONTRACT
-------------------
This script opens every research artefact read-only and writes to exactly one
place: 06_BACKEND/data/. It never imports `pipeline` (see the architecture plan
§1.3 — importing pipeline.config has a mkdir side effect), and it never touches
data/, models/, predictions/, experiments/ or reports/ in write mode.

WHY A SEPARATE DATABASE AT ALL
------------------------------
The 59.2M-row panel is already queryable in place: DuckDB reads
data/processed/sales_long_full.parquet at 0.10-0.16 s with predicate pushdown, so
duplicating history would be pure waste. What is NOT already queryable is
everything else the product needs:

  * the forecast lives in a WIDE csv (id, F1..F28) — useless for time queries
  * the backtest cache is 8 separate files keyed by integer series_idx
  * series metadata, volume tiers and intermittency regimes exist only as
    numbers inside research scripts
  * measured per-level accuracy lives in a Stage 7 artefact CSV

So this builds the narrow, indexed, joinable versions of those — and leaves the
history where it is.

TABLES BUILT
------------
  series          30,490  id, item/dept/cat/store/state, volume tier, regime
  forecast       853,720  (series, horizon, date, yhat) — the frozen forecast
  backtest      ~6.8M     8 cached windows x members x truth
  error_bands      ~280   empirical residual quantiles by (tier, horizon)
  level_accuracy    12    measured accuracy per M5 aggregation level
  meta               n/a  model card, hashes, provenance, build timestamp
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "06_BACKEND"
OUT_DIR = BACKEND / "data"
DB_PATH = OUT_DIR / "product.duckdb"

# --- protected inputs, all opened read-only --------------------------------
SALES_EVAL = ROOT / "data" / "raw" / "sales_train_evaluation.csv"
CALENDAR = ROOT / "data" / "raw" / "calendar.csv"
PANEL_PARQUET = ROOT / "data" / "processed" / "sales_long_full.parquet"
FORECAST_CSV = (ROOT / "predictions" / "final_forecast"
                / "final_forecast_28day_v3_diversity_blend.csv")
BACKTEST_DIR = ROOT / "predictions" / "uc11_cache"
LEVEL_ACC = ROOT / "experiments" / "artifacts" / "uc11_hierarchy_levels.csv"
CHAMPION_MANIFEST = ROOT / "02_MODEL" / "FROZEN_CHAMPION" / "CHAMPION_MANIFEST.json"
MODEL_DIRECT = ROOT / "models" / "champion" / "model_11_blend_direct_final_forecast.txt"
MODEL_RECURSIVE = ROOT / "models" / "champion" / "model_12_blend_recursive_shape_final.txt"

# Day-index convention, identical to pipeline/config.py (duplicated rather than
# imported so this script never triggers the pipeline's import side effects).
#
# IMPORTANT: day_idx is ZERO-BASED everywhere in this project.
#     idx 0    == "d_1"    == 2011-01-29
#     idx 1940 == "d_1941" == 2016-05-22   (forecast origin, last known sales)
#     idx 1941 == "d_1942" == 2016-05-23   (first forecast day)
# The backtest artefacts produced by the research pipeline use this convention
# for `target_day_idx`, so the forecast table must match it or the two cannot be
# joined or charted on a common axis.
N_HISTORY_DAYS = 1941               # d_1 .. d_1941
FORECAST_ORIGIN_IDX = 1940          # zero-based index of d_1941
HORIZON = 28
N_SERIES = 30_490

# Syntetos-Boylan cut points, matching scripts/07_usecase11/58_intermittency_audit.py
ADI_CUT, CV2_CUT = 1.32, 0.49
REGIME_HISTORY_DAYS = 728

# Volume tiers, matching pipeline/optimize.py Setup.tier
TIER_EDGES = [-0.001, 0.2, 1.0, 3.0, np.inf]
TIER_LABELS = ["very low", "low", "medium", "high"]


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
def build_series_table() -> pd.DataFrame:
    """Hierarchy + volume tier + intermittency regime, one row per series."""
    log("  reading sales matrix (read-only)...")
    df = pd.read_csv(SALES_EVAL)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]
    if len(df) != N_SERIES or len(day_cols) != N_HISTORY_DAYS:
        raise SystemExit(f"unexpected sales shape: {len(df)} x {len(day_cols)}")

    meta = df[id_cols].copy()
    meta.insert(0, "series_idx", np.arange(len(meta), dtype=np.int32))
    sales = df[day_cols].to_numpy(dtype=np.int16)
    del df

    # --- volume tier, from full pre-forecast history ---------------------
    mean_daily = sales.mean(axis=1)
    meta["mean_daily_sales"] = mean_daily.astype(np.float32)
    meta["total_units"] = sales.sum(axis=1).astype(np.int64)
    meta["volume_tier"] = pd.cut(mean_daily, TIER_EDGES,
                                 labels=TIER_LABELS).astype(str)

    # --- Syntetos-Boylan regime, from the trailing 728 days ---------------
    log("  classifying intermittency regimes (Syntetos-Boylan)...")
    hist = sales[:, -REGIME_HISTORY_DAYS:].astype(np.float64)
    nz = hist > 0
    counts = nz.sum(axis=1)
    adi = np.where(counts > 0, hist.shape[1] / np.maximum(counts, 1), np.inf)

    cv2 = np.zeros(len(hist))
    for i in range(len(hist)):
        v = hist[i][nz[i]]
        if v.size > 1 and v.mean() > 0:
            cv2[i] = (v.std() / v.mean()) ** 2

    regime = np.full(len(hist), "never sold", dtype=object)
    smooth = (adi < ADI_CUT) & (cv2 < CV2_CUT)
    erratic = (adi < ADI_CUT) & (cv2 >= CV2_CUT)
    intermittent = (adi >= ADI_CUT) & (cv2 < CV2_CUT)
    lumpy = (adi >= ADI_CUT) & (cv2 >= CV2_CUT)
    regime[smooth] = "smooth"
    regime[erratic] = "erratic"
    regime[intermittent] = "intermittent"
    regime[lumpy] = "lumpy"
    regime[counts == 0] = "never sold"

    meta["adi"] = adi.astype(np.float32)
    meta["cv2"] = cv2.astype(np.float32)
    meta["regime"] = regime
    meta["zero_pct"] = (1 - nz.mean(axis=1)).astype(np.float32) * 100

    log(f"    regimes: {meta.regime.value_counts().to_dict()}")
    return meta


def build_forecast_table(meta: pd.DataFrame) -> pd.DataFrame:
    """The frozen forecast, unpivoted from wide (id, F1..F28) to long."""
    log("  reading frozen forecast (read-only)...")
    wide = pd.read_csv(FORECAST_CSV)
    if len(wide) != N_SERIES:
        raise SystemExit(f"forecast has {len(wide)} rows, expected {N_SERIES}")

    fcols = [f"F{i}" for i in range(1, HORIZON + 1)]
    long = wide.melt(id_vars="id", value_vars=fcols,
                     var_name="f", value_name="yhat")
    long["horizon"] = long["f"].str.slice(1).astype(np.int16)
    long = long.drop(columns="f")

    # The forecast id carries an "_evaluation" suffix; series ids do not.
    long["id"] = long["id"].str.replace("_evaluation", "", regex=False)
    key = meta[["series_idx", "id"]].copy()
    key["id"] = key["id"].str.replace("_evaluation", "", regex=False)
    long = long.merge(key, on="id", how="inner")
    if len(long) != N_SERIES * HORIZON:
        raise SystemExit(f"forecast join produced {len(long)} rows, "
                         f"expected {N_SERIES * HORIZON}")

    long["day_idx"] = (FORECAST_ORIGIN_IDX + long["horizon"]).astype(np.int32)
    long["yhat"] = long["yhat"].astype(np.float32)
    return long[["series_idx", "horizon", "day_idx", "yhat"]]


def build_backtest_table() -> pd.DataFrame:
    """The 8 cached champion reproductions, with members and truth."""
    frames = []
    for p in sorted(BACKTEST_DIR.glob("champion_blend_origin*_seed42.csv")):
        origin = int(p.stem.split("origin")[1].split("_")[0])
        d = pd.read_csv(p)
        d["origin_idx"] = np.int32(origin)
        frames.append(d)
        log(f"    {p.name:<48} {len(d):>8,} rows  origin d_{origin + 1}")
    if not frames:
        raise SystemExit(f"no backtest artefacts found in {BACKTEST_DIR}")
    bt = pd.concat(frames, ignore_index=True)
    for c in ("y_true", "p_direct", "p_recursive", "p_blend"):
        bt[c] = bt[c].astype(np.float32)
    for c in ("series_idx", "target_day_idx", "origin_idx"):
        bt[c] = bt[c].astype(np.int32)
    bt["horizon"] = bt["horizon"].astype(np.int16)
    return bt


#: Residuals are normalised by sqrt(max(yhat, 1)) before quantiles are taken.
#: See build_error_bands() for why.
BAND_SCALE_FLOOR = 1.0


def build_error_bands(bt: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Empirical residual quantiles by (demand regime x horizon), on a
    variance-stabilised scale.

    THIS IS NOT A MODEL-PRODUCED PREDICTION INTERVAL. The frozen model emits a
    point forecast and nothing else. What is computed here is the observed
    distribution of (actual - predicted) on held-out backtest windows, which is
    a measurement, and the API and UI label it as such everywhere it surfaces.

    WHY NORMALISE BY sqrt(yhat)
    ---------------------------
    A first attempt pooled raw residuals by volume tier. Measurement showed that
    is invalid: inside the single "high" tier the residual standard deviation
    ranges from 3.3 (series predicting <2 units/day) to 21.6 (40+ units/day), a
    6.5x spread. A band built from that pooling is far too wide for small series
    and far too narrow for large ones — exactly the rows where a planner would
    act on it.

    Dividing the residual by sqrt(max(yhat, 1)) collapses that spread to ~1.4x.
    The exponent is not arbitrary: the model was fitted with a Tweedie variance
    power of 1.1, which implies Var(y) ∝ mu^1.1 and therefore sd ∝ mu^0.55.
    Measured exponents of 0.50 and 0.55 both stabilise the spread; 0.50 tested
    marginally better and is the classic count variance-stabilising transform, so
    it is used. The floor of 1.0 stops the divisor collapsing for near-zero
    forecasts, where the raw residual scale is already the right one.

    WHY GROUP BY REGIME AND HORIZON
    -------------------------------
    After normalisation, measured spread by Syntetos-Boylan regime is
    erratic 2.20, lumpy 1.70, smooth 1.64, intermittent 0.89 — regime is by far
    the strongest remaining driver, which makes sense: it is a classification of
    exactly how erratic a series is. Horizon contributes a further ~8% growth
    from h1 to h28. Volume tier adds nothing once normalised, so it is dropped.

    RECONSTRUCTION
    --------------
        scale = sqrt(max(yhat, 1))
        lower = max(0, yhat + q05 * scale)
        upper =        yhat + q95 * scale
    """
    log("  computing empirical error bands (sqrt-normalised, regime x horizon)...")
    d = bt.merge(meta[["series_idx", "regime"]], on="series_idx", how="left")
    scale = np.sqrt(np.maximum(d["p_blend"].to_numpy(np.float64), BAND_SCALE_FLOOR))
    d["norm_resid"] = (d["y_true"] - d["p_blend"]) / scale

    g = d.groupby(["regime", "horizon"], observed=True)["norm_resid"]
    bands = g.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    bands.columns = ["q05", "q25", "q50", "q75", "q95"]
    bands = bands.reset_index()

    agg = d.groupby(["regime", "horizon"], observed=True).agg(
        n=("norm_resid", "size"),
        norm_sd=("norm_resid", "std"),
        raw_mae=("norm_resid", lambda s: float(np.abs(s).mean())),
    ).reset_index()
    bands = bands.merge(agg, on=["regime", "horizon"])
    bands["scale_floor"] = np.float32(BAND_SCALE_FLOOR)

    for c in ("q05", "q25", "q50", "q75", "q95", "norm_sd", "raw_mae"):
        bands[c] = bands[c].astype(np.float32)
    log(f"    {len(bands)} (regime x horizon) cells; "
        f"normalised sd by regime: "
        f"{bands.groupby('regime', observed=True).norm_sd.mean().round(2).to_dict()}")
    return bands


def build_meta_table() -> pd.DataFrame:
    """Model card + provenance. Every number here is read from an artefact."""
    cm = json.loads(CHAMPION_MANIFEST.read_text(encoding="utf-8"))
    fc = cm["frozen_champion"]
    rows = [
        ("model_name", "Direct+Recursive Tweedie Blend"),
        ("blend_formula", fc["blend"]),
        ("blend_weight_direct", str(fc["blend_weight"])),
        ("blend_weight_recursive", str(round(1 - fc["blend_weight"], 2))),
        ("objective", fc["objective"]),
        ("n_estimators", str(fc["n_estimators"])),
        ("seed", str(fc["seed"])),
        ("status", fc["status"]),
        ("validation_rmse", f"{fc['primary_window_RMSE']:.4f}"),
        ("validation_mae", f"{fc['primary_window_MAE']:.4f}"),
        ("validation_window", "d_1914-d_1941 (2016-04-25 to 2016-05-22)"),
        ("validation_n", str(N_SERIES * HORIZON)),
        ("forecast_origin", fc["forecast_origin"]),
        ("forecast_origin_idx", str(FORECAST_ORIGIN_IDX)),
        ("forecast_dates", fc["forecast_dates"]),
        ("horizon_days", str(HORIZON)),
        ("n_series", str(N_SERIES)),
        ("model_direct_sha256", sha256(MODEL_DIRECT)),
        ("model_recursive_sha256", sha256(MODEL_RECURSIVE)),
        ("forecast_sha256", sha256(FORECAST_CSV)),
        ("db_built_at", pd.Timestamp.now(tz="UTC").isoformat()),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    banner("BUILDING PRODUCT DATABASE")
    log(f"  project root : {ROOT}")
    log(f"  output       : {DB_PATH}")

    for p in (SALES_EVAL, CALENDAR, PANEL_PARQUET, FORECAST_CSV, LEVEL_ACC,
              CHAMPION_MANIFEST, MODEL_DIRECT, MODEL_RECURSIVE):
        if not p.exists():
            raise SystemExit(f"MISSING required artefact: {p}")
    if not BACKTEST_DIR.exists():
        raise SystemExit(f"MISSING backtest cache: {BACKTEST_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()          # product-owned file only; rebuildable

    banner("1/6  series metadata")
    meta = build_series_table()

    banner("2/6  frozen forecast")
    fc = build_forecast_table(meta)
    log(f"    {len(fc):,} rows, mean yhat {fc.yhat.mean():.4f}")

    banner("3/6  backtest cache")
    bt = build_backtest_table()
    log(f"    {len(bt):,} rows across {bt.origin_idx.nunique()} windows")

    banner("4/6  empirical error bands")
    bands = build_error_bands(bt, meta)

    banner("5/6  measured level accuracy")
    lvl = pd.read_csv(LEVEL_ACC)
    log(f"    {len(lvl)} hierarchy levels")

    banner("6/6  model card")
    card = build_meta_table()

    banner("WRITING DUCKDB")
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET preserve_insertion_order = false")
    for name, frame in [("series", meta), ("forecast", fc), ("backtest", bt),
                        ("error_bands", bands), ("level_accuracy", lvl),
                        ("model_card", card)]:
        con.register("_t", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        log(f"  {name:<16} {n:>10,} rows")

    # Indexes for the access patterns the API actually uses
    con.execute("CREATE INDEX idx_fc_series ON forecast(series_idx)")
    con.execute("CREATE INDEX idx_bt_series ON backtest(series_idx)")
    con.execute("CREATE INDEX idx_bt_origin ON backtest(origin_idx)")
    con.execute("CREATE INDEX idx_series_store ON series(store_id)")
    con.execute("CREATE INDEX idx_series_item ON series(item_id)")

    # A view joining the panel parquet, so history queries need no ETL.
    con.execute(f"""
        CREATE VIEW panel AS
        SELECT * FROM read_parquet('{PANEL_PARQUET.as_posix()}')
    """)
    n_panel = con.execute("SELECT count(*) FROM panel").fetchone()[0]
    log(f"  panel (view)     {n_panel:>10,} rows  [read-only parquet]")

    con.close()

    size_mb = DB_PATH.stat().st_size / 1e6
    banner("DONE")
    log(f"  {DB_PATH.name}  {size_mb:.1f} MB   built in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
