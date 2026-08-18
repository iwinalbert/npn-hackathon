"""
BUILD THE PRODUCT DATA LAYER.

Reads the protected research artefacts READ-ONLY and materialises everything the
API needs into backend/data/.

    python tasks.py build-db

PROTECTION CONTRACT
-------------------
Every research artefact is opened read-only. The only directory written is
backend/data/. This script never imports `pipeline` — importing
pipeline.config calls mkdir() at import time, which is a filesystem side effect
on the protected tree — so the day-index and tier constants it needs are
restated here and asserted against the artefacts.

WHAT IS PRODUCED
----------------
  data/product.duckdb    small relational tables (~5 MB)
  data/history.parquet   59.2M rows of actuals, sorted    (~32 MB)
  data/backtest.parquet  6.8M rows of backtest results    (~78 MB)

WHY PARQUET SIDECARS RATHER THAN QUERYING THE RESEARCH PARQUET IN PLACE
-----------------------------------------------------------------------
The architecture plan originally proposed querying
data/processed/sales_long_full.parquet directly, reasoning that duplicating
287 MB of history would be waste. Measurement during Phase 2 showed that is both
slower and not deployable:

  * DuckDB bakes an ABSOLUTE path into any view over an external file. The
    generated view contained a Windows path, which is a hard failure in a Linux
    container. That alone rules the approach out.
  * The research parquet is denormalised: item_id, store_id, event names and
    SNAP flags are repeated on every one of 59.2M rows. Keeping only
    (series_idx, day_idx, sales, sell_price) and sorting by series gives 31.7 MB
    instead of 287 MB.
  * Sorted + row-grouped, a single series' full history reads in 8-13 ms against
    the sidecar versus ~320 ms against the research parquet, because zone maps
    prune almost everything.

So the sidecar is smaller, ~25x faster, and portable. A DuckDB table with an
index was also measured and rejected: 1,066 MB and 21 s to build.

The API therefore needs NO access to the research tree at runtime. That is what
makes the container deployable with the research artefacts mounted read-only —
or not mounted at all.
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

# scripts/ -> backend -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
# Every input below is a PROTECTED research artefact, opened read-only. They live
# under research/ as one unit because the research pipeline derives its own
# root from `pipeline/`'s parent and needs these as siblings.
ROOT = REPO_ROOT / "research"
DOCS = REPO_ROOT / "docs"
OUT_DIR = BACKEND / "data"
DB_PATH = OUT_DIR / "product.duckdb"
HISTORY_PARQUET = OUT_DIR / "history.parquet"
BACKTEST_PARQUET = OUT_DIR / "backtest.parquet"

# --- protected inputs, all opened read-only --------------------------------
SALES_EVAL = ROOT / "data" / "raw" / "sales_train_evaluation.csv"
CALENDAR = ROOT / "data" / "raw" / "calendar.csv"
PANEL_PARQUET = ROOT / "data" / "processed" / "sales_long_full.parquet"
FORECAST_CSV = (ROOT / "predictions" / "final_forecast"
                / "final_forecast_28day_v3_diversity_blend.csv")
BACKTEST_DIR = ROOT / "predictions" / "uc11_cache"
LEVEL_ACC = ROOT / "experiments" / "artifacts" / "uc11_hierarchy_levels.csv"
CHAMPION_MANIFEST = (DOCS / "02_MODEL" / "FROZEN_CHAMPION"
                     / "CHAMPION_MANIFEST.json")
MODEL_DIRECT = ROOT / "models" / "champion" / "model_11_blend_direct_final_forecast.txt"
MODEL_RECURSIVE = ROOT / "models" / "champion" / "model_12_blend_recursive_shape_final.txt"

# Day-index convention, identical to pipeline/config.py.
#     idx 0    == "d_1"    == 2011-01-29
#     idx 1940 == "d_1941" == 2016-05-22   (forecast origin, last known sales)
#     idx 1941 == "d_1942" == 2016-05-23   (first forecast day)
# The research backtest artefacts use this convention for `target_day_idx`, so
# the forecast table must match it or the two cannot share a time axis.
N_HISTORY_DAYS = 1941
FORECAST_ORIGIN_IDX = 1940
HORIZON = 28
N_SERIES = 30_490

# Syntetos-Boylan cuts, matching scripts/07_usecase11/58_intermittency_audit.py
ADI_CUT, CV2_CUT = 1.32, 0.49
REGIME_HISTORY_DAYS = 728

# Volume tiers, matching pipeline/optimize.py Setup.tier
TIER_EDGES = [-0.001, 0.2, 1.0, 3.0, np.inf]
TIER_LABELS = ["very low", "low", "medium", "high"]

#: Residuals are normalised by sqrt(max(yhat, FLOOR)) before quantiles.
BAND_SCALE_FLOOR = 1.0

ROW_GROUP = 122_880


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

    mean_daily = sales.mean(axis=1)
    meta["mean_daily_sales"] = mean_daily.astype(np.float32)
    meta["total_units"] = sales.sum(axis=1).astype(np.int64)
    meta["volume_tier"] = pd.cut(mean_daily, TIER_EDGES,
                                 labels=TIER_LABELS).astype(str)

    log("  classifying intermittency regimes (Syntetos-Boylan)...")
    hist = sales[:, -REGIME_HISTORY_DAYS:].astype(np.float64)
    nz = hist > 0
    counts = nz.sum(axis=1)
    adi = np.where(counts > 0, hist.shape[1] / np.maximum(counts, 1), 9999.0)

    cv2 = np.zeros(len(hist))
    for i in range(len(hist)):
        v = hist[i][nz[i]]
        if v.size > 1 and v.mean() > 0:
            cv2[i] = (v.std() / v.mean()) ** 2

    regime = np.full(len(hist), "never sold", dtype=object)
    regime[(adi < ADI_CUT) & (cv2 < CV2_CUT)] = "smooth"
    regime[(adi < ADI_CUT) & (cv2 >= CV2_CUT)] = "erratic"
    regime[(adi >= ADI_CUT) & (cv2 < CV2_CUT)] = "intermittent"
    regime[(adi >= ADI_CUT) & (cv2 >= CV2_CUT)] = "lumpy"
    regime[counts == 0] = "never sold"

    meta["adi"] = np.minimum(adi, 9999.0).astype(np.float32)
    meta["cv2"] = cv2.astype(np.float32)
    meta["regime"] = regime
    meta["zero_pct"] = ((1 - nz.mean(axis=1)) * 100).astype(np.float32)

    # Recent demand level, used by the planning endpoints as a reference point.
    meta["mean_daily_28d"] = sales[:, -28:].mean(axis=1).astype(np.float32)
    meta["mean_daily_91d"] = sales[:, -91:].mean(axis=1).astype(np.float32)

    log(f"    regimes: {meta.regime.value_counts().to_dict()}")
    return meta


def build_calendar_table() -> pd.DataFrame:
    """
    Calendar for all 1,969 days, so the API needs no access to data/raw/.

    calendar.csv runs 28 days past the sales history, which is exactly why the
    forecast window has known covariates.
    """
    cal = pd.read_csv(CALENDAR)
    cal["day_idx"] = cal["d"].str.slice(2).astype(np.int32) - 1
    out = cal[["day_idx", "date", "wday", "month", "year", "event_name_1",
               "event_type_1", "event_name_2", "event_type_2",
               "snap_CA", "snap_TX", "snap_WI"]].copy()
    out["date"] = out["date"].astype(str).str.slice(0, 10)
    out["is_weekend"] = out["wday"].isin([1, 2]).astype(np.int8)
    for c in ("snap_CA", "snap_TX", "snap_WI"):
        out[c] = out[c].astype(np.int8)
    return out.sort_values("day_idx").reset_index(drop=True)


def build_history_parquet(meta: pd.DataFrame) -> int:
    """
    Sorted, minimal history sidecar. See the module docstring for why this is a
    product-owned file rather than a view over the research parquet.
    """
    log(f"  materialising history sidecar from {PANEL_PARQUET.name}...")
    con = duckdb.connect()
    con.register("series_meta", meta[["series_idx", "item_id", "store_id"]])
    con.execute(f"""
        COPY (
            SELECT s.series_idx,
                   CAST(regexp_extract(p.d, 'd_([0-9]+)', 1) AS INTEGER) - 1
                       AS day_idx,
                   CAST(p.sales AS SMALLINT) AS sales,
                   p.sell_price
            FROM read_parquet('{PANEL_PARQUET.as_posix()}') p
            JOIN series_meta s
              ON s.item_id = p.item_id AND s.store_id = p.store_id
            ORDER BY s.series_idx, day_idx
        ) TO '{HISTORY_PARQUET.as_posix()}'
        (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ROW_GROUP})
    """)
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{HISTORY_PARQUET.as_posix()}')"
    ).fetchone()[0]
    con.close()
    if n != N_SERIES * N_HISTORY_DAYS:
        raise SystemExit(f"history has {n} rows, expected "
                         f"{N_SERIES * N_HISTORY_DAYS}")
    return int(n)


def build_forecast_table(meta: pd.DataFrame) -> pd.DataFrame:
    """The frozen forecast, unpivoted from wide (id, F1..F28) to long."""
    log("  reading frozen forecast (read-only)...")
    wide = pd.read_csv(FORECAST_CSV)
    if len(wide) != N_SERIES:
        raise SystemExit(f"forecast has {len(wide)} rows, expected {N_SERIES}")

    fcols = [f"F{i}" for i in range(1, HORIZON + 1)]
    long = wide.melt(id_vars="id", value_vars=fcols, var_name="f",
                     value_name="yhat")
    long["horizon"] = long["f"].str.slice(1).astype(np.int16)
    long = long.drop(columns="f")
    long["id"] = long["id"].str.replace("_evaluation", "", regex=False)

    key = meta[["series_idx", "id"]].copy()
    key["id"] = key["id"].str.replace("_evaluation", "", regex=False)
    long = long.merge(key, on="id", how="inner")
    if len(long) != N_SERIES * HORIZON:
        raise SystemExit(f"forecast join produced {len(long)} rows")

    long["day_idx"] = (FORECAST_ORIGIN_IDX + long["horizon"]).astype(np.int32)
    long["yhat"] = long["yhat"].astype(np.float32)
    if long["yhat"].isna().any() or (long["yhat"] < 0).any():
        raise SystemExit("frozen forecast contains NaN or negative values")
    return long[["series_idx", "horizon", "day_idx", "yhat"]].sort_values(
        ["series_idx", "horizon"]).reset_index(drop=True)


def build_backtest_parquet() -> tuple[int, list[int]]:
    """The 8 cached champion reproductions, concatenated and sorted."""
    frames = []
    for p in sorted(BACKTEST_DIR.glob("champion_blend_origin*_seed42.csv")):
        origin = int(p.stem.split("origin")[1].split("_")[0])
        d = pd.read_csv(p)
        d["origin_idx"] = np.int32(origin)
        frames.append(d)
        log(f"    {p.name:<46} {len(d):>9,} rows  origin d_{origin + 1}")
    if not frames:
        raise SystemExit(f"no backtest artefacts in {BACKTEST_DIR}")

    bt = pd.concat(frames, ignore_index=True)
    for c in ("y_true", "p_direct", "p_recursive", "p_blend"):
        bt[c] = bt[c].astype(np.float32)
    for c in ("series_idx", "target_day_idx", "origin_idx"):
        bt[c] = bt[c].astype(np.int32)
    bt["horizon"] = bt["horizon"].astype(np.int16)
    bt = bt.sort_values(["origin_idx", "series_idx", "horizon"])

    con = duckdb.connect()
    con.register("bt", bt)
    con.execute(f"""
        COPY (SELECT * FROM bt) TO '{BACKTEST_PARQUET.as_posix()}'
        (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ROW_GROUP})
    """)
    con.close()
    return len(bt), sorted(bt.origin_idx.unique().tolist())


def build_error_bands(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Empirical residual quantiles by (demand regime x horizon), on a
    variance-stabilised scale.

    NOT A MODEL-PRODUCED PREDICTION INTERVAL. The frozen model emits a point
    forecast and nothing else. This is the observed distribution of
    (actual - predicted) on held-out backtest windows, and the API labels it so.

    WHY NORMALISE BY sqrt(yhat)
    ---------------------------
    Pooling raw residuals by volume tier was measured to be invalid: inside the
    single "high" tier the residual standard deviation ranges from 3.3 (series
    predicting <2 units/day) to 21.6 (40+ units/day), a 6.5x spread, so a band
    built that way is far too narrow for large series. Dividing by
    sqrt(max(yhat, 1)) collapses that to ~1.4x. The exponent is not arbitrary:
    the model was fitted with Tweedie variance power 1.1, implying sd proportional
    to mu^0.55; 0.50 measured marginally better and is the classic count
    variance-stabilising transform.

    Grouping is by regime (measured normalised sd: erratic 2.28, lumpy 1.71,
    smooth 1.64, intermittent 0.89) and horizon (~8% growth h1 to h28). Volume
    tier adds nothing once normalised.

    Reconstruction:  lower = max(0, yhat + q05 * sqrt(max(yhat, 1)))
    """
    log("  computing empirical error bands (sqrt-normalised, regime x horizon)...")
    con = duckdb.connect()
    con.register("series_meta", meta[["series_idx", "regime"]])
    bands = con.execute(f"""
        WITH r AS (
            SELECT s.regime, b.horizon,
                   (b.y_true - b.p_blend)
                       / sqrt(greatest(b.p_blend, {BAND_SCALE_FLOOR}))
                       AS norm_resid
            FROM read_parquet('{BACKTEST_PARQUET.as_posix()}') b
            JOIN series_meta s USING (series_idx)
        )
        SELECT regime, horizon,
               quantile_cont(norm_resid, 0.05) AS q05,
               quantile_cont(norm_resid, 0.25) AS q25,
               quantile_cont(norm_resid, 0.50) AS q50,
               quantile_cont(norm_resid, 0.75) AS q75,
               quantile_cont(norm_resid, 0.95) AS q95,
               count(*) AS n,
               stddev_samp(norm_resid) AS norm_sd
        FROM r GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchdf()
    con.close()
    bands["scale_floor"] = np.float32(BAND_SCALE_FLOOR)
    for c in ("q05", "q25", "q50", "q75", "q95", "norm_sd"):
        bands[c] = bands[c].astype(np.float32)
    log(f"    {len(bands)} (regime x horizon) cells; normalised sd: "
        f"{bands.groupby('regime').norm_sd.mean().round(2).to_dict()}")
    return bands


def build_window_metrics() -> pd.DataFrame:
    """Headline metrics per backtest window, precomputed for the accuracy views."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT origin_idx,
               count(*)                                    AS n,
               sqrt(avg((y_true - p_blend) ^ 2))           AS rmse,
               avg(abs(y_true - p_blend))                  AS mae,
               sum(abs(y_true - p_blend)) / nullif(sum(y_true), 0) AS wape,
               avg(p_blend - y_true)                       AS bias,
               sqrt(avg((y_true - p_direct) ^ 2))          AS rmse_direct,
               sqrt(avg((y_true - p_recursive) ^ 2))       AS rmse_recursive,
               corr(y_true - p_direct, y_true - p_recursive) AS member_resid_corr
        FROM read_parquet('{BACKTEST_PARQUET.as_posix()}')
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    con.close()
    return df


def build_meta_table(counts: dict) -> pd.DataFrame:
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
        ("schema_version", "2"),
        ("db_built_at", pd.Timestamp.now(tz="UTC").isoformat()),
    ] + [(f"rows_{k}", str(v)) for k, v in counts.items()]
    return pd.DataFrame(rows, columns=["key", "value"])


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    banner("BUILDING PRODUCT DATA LAYER")
    log(f"  project root : {ROOT}")
    log(f"  output       : {OUT_DIR}")

    required = [SALES_EVAL, CALENDAR, PANEL_PARQUET, FORECAST_CSV, LEVEL_ACC,
                CHAMPION_MANIFEST, MODEL_DIRECT, MODEL_RECURSIVE]
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            log(f"  MISSING: {p}")
        raise SystemExit("cannot build: required research artefacts are absent")
    if not BACKTEST_DIR.exists():
        raise SystemExit(f"MISSING backtest cache: {BACKTEST_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (DB_PATH, HISTORY_PARQUET, BACKTEST_PARQUET):
        if p.exists():
            p.unlink()          # product-owned files only; always rebuildable

    banner("1/7  series metadata")
    meta = build_series_table()

    banner("2/7  calendar")
    cal = build_calendar_table()
    log(f"    {len(cal)} days, {cal.day_idx.min()}..{cal.day_idx.max()}")

    banner("3/7  history sidecar")
    n_hist = build_history_parquet(meta)
    log(f"    {n_hist:,} rows -> {HISTORY_PARQUET.name} "
        f"({HISTORY_PARQUET.stat().st_size / 1e6:.1f} MB)")

    banner("4/7  frozen forecast")
    fc = build_forecast_table(meta)
    log(f"    {len(fc):,} rows, mean yhat {fc.yhat.mean():.4f}")

    banner("5/7  backtest sidecar")
    n_bt, origins = build_backtest_parquet()
    log(f"    {n_bt:,} rows across {len(origins)} windows -> "
        f"{BACKTEST_PARQUET.name} "
        f"({BACKTEST_PARQUET.stat().st_size / 1e6:.1f} MB)")

    banner("6/7  derived analytics")
    bands = build_error_bands(meta)
    windows = build_window_metrics()
    log(f"    window metrics: primary RMSE "
        f"{windows[windows.origin_idx == 1912].rmse.iloc[0]:.4f}")
    lvl = pd.read_csv(LEVEL_ACC)
    log(f"    {len(lvl)} measured hierarchy levels")

    banner("7/7  writing product.duckdb")
    counts = {"series": len(meta), "forecast": len(fc), "history": n_hist,
              "backtest": n_bt, "calendar": len(cal)}
    card = build_meta_table(counts)

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET preserve_insertion_order = false")
    for name, frame in [("series", meta), ("forecast", fc), ("calendar", cal),
                        ("error_bands", bands), ("level_accuracy", lvl),
                        ("window_metrics", windows), ("model_card", card)]:
        con.register("_t", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        log(f"  {name:<16} {n:>10,} rows")

    con.execute("CREATE INDEX idx_fc_series ON forecast(series_idx)")
    con.execute("CREATE INDEX idx_series_store ON series(store_id)")
    con.execute("CREATE INDEX idx_series_item ON series(item_id)")
    # NO views over external files: DuckDB bakes absolute paths into view SQL,
    # which breaks the moment the database is opened on another machine or
    # inside a container. Sidecar paths are resolved at runtime from settings.
    con.close()

    total = sum(p.stat().st_size for p in
                (DB_PATH, HISTORY_PARQUET, BACKTEST_PARQUET))
    banner("DONE")
    log(f"  product.duckdb    {DB_PATH.stat().st_size / 1e6:8.1f} MB")
    log(f"  history.parquet   {HISTORY_PARQUET.stat().st_size / 1e6:8.1f} MB")
    log(f"  backtest.parquet  {BACKTEST_PARQUET.stat().st_size / 1e6:8.1f} MB")
    log(f"  TOTAL             {total / 1e6:8.1f} MB   "
        f"built in {time.time() - t0:.1f}s")
    log("\n  The API needs only these three files — no research tree at runtime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
