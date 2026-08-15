"""
Stage 1 — ML pipeline foundation verification.

Runs the whole foundation end to end and writes a machine-readable result file
that the PDF report is then generated from. Nothing here trains a model.

    python scripts/01_foundation_check.py

Outputs:
    artifacts/foundation_checks.json     every check, pass/fail, with detail
    artifacts/sample_features.csv        a small human-inspectable feature sample
    artifacts/feature_summary.csv        per-feature dtype / missingness / range
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics, validation_checks as vc
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.features import (
    FeatureBuilder, FEATURE_GROUPS, CATEGORICAL_FEATURES, all_feature_columns,
)

RESULTS: dict = {}
ALL_CHECKS: list[vc.CheckResult] = []


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run(checks: list[vc.CheckResult]) -> None:
    ALL_CHECKS.extend(checks)
    for c in checks:
        print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}")


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    section("1. LOAD SOURCE DATA (read-only)")
    # ------------------------------------------------------------------
    t0 = time.time()
    data = M5Data()
    load_secs = time.time() - t0
    print(json.dumps(data.describe(), indent=2))
    print(f"  loaded in {load_secs:.1f}s")
    RESULTS["data"] = data.describe()
    RESULTS["data"]["load_seconds"] = round(load_secs, 1)

    # ------------------------------------------------------------------
    section("2. SOURCE DATA INTEGRITY")
    # ------------------------------------------------------------------
    run(vc.check_data_integrity(data))
    run(vc.check_calendar_alignment(data))

    # ------------------------------------------------------------------
    section("3. BACKTEST WINDOW DEFINITION")
    # ------------------------------------------------------------------
    bt = Backtester(data)
    fb = bt.fb
    window = bt.make_window(config.VALIDATION_ORIGIN_IDX)
    wdesc = window.describe()
    print(json.dumps(wdesc, indent=2))
    RESULTS["validation_window"] = wdesc

    future = bt.make_window(config.FINAL_FORECAST_ORIGIN_IDX)
    fdesc = future.describe()
    print("\n  Real (unknown) forecast window:")
    print(json.dumps(fdesc, indent=2))
    RESULTS["final_forecast_window"] = fdesc

    run([vc.CheckResult(
        "validation_window_has_ground_truth",
        wdesc["validation_has_known_sales"],
        f"{wdesc['validation_days']} ({wdesc['validation_dates']}) are real observed sales",
    )])

    # ------------------------------------------------------------------
    section("4. BUILD VALIDATION FRAME (all 30,490 series x 28 days)")
    # ------------------------------------------------------------------
    t0 = time.time()
    valid = bt.build_validation_frame(config.VALIDATION_ORIGIN_IDX)
    build_secs = time.time() - t0
    mem_mb = valid.memory_usage(deep=True).sum() / 1e6
    print(f"  shape {valid.shape}, {mem_mb:.0f} MB, built in {build_secs:.1f}s")

    RESULTS["validation_frame"] = {
        "rows": int(len(valid)),
        "columns": int(valid.shape[1]),
        "memory_mb": round(mem_mb, 1),
        "build_seconds": round(build_secs, 1),
    }

    run(vc.check_frame_structure(valid, config.N_SERIES, config.HORIZON))
    run(vc.check_feature_sanity(valid, data))
    run(vc.check_target_matches_source(valid, data, n_spot=500))

    # ------------------------------------------------------------------
    section("5. LEAKAGE TEST — corrupt every post-origin sales value")
    # ------------------------------------------------------------------
    print("  Overwriting all sales after the origin with 9999 and rebuilding...")
    t0 = time.time()
    run(vc.check_no_future_sales_leakage(data, config.VALIDATION_ORIGIN_IDX))
    print(f"  ({time.time() - t0:.1f}s)")

    section("6. FUTURE-COVARIATE TEST — prices/calendar SHOULD be usable")
    run(vc.check_future_covariates_are_used(data, config.VALIDATION_ORIGIN_IDX))

    # ------------------------------------------------------------------
    section("7. TRAINING FRAME + TRAIN/VALIDATION SEPARATION")
    # ------------------------------------------------------------------
    # A modest series sample keeps this verification fast; the full training build
    # happens in the modelling stage. The separation logic being tested is
    # identical either way.
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_series = np.sort(rng.choice(config.N_SERIES, size=2000, replace=False))

    origins = bt.training_origins(config.VALIDATION_ORIGIN_IDX, n_origins=6)
    print(f"  training origins: {origins}")
    print("  as days: " + ", ".join(
        f"d_{o + 1} ({data.date_of(o).date()})" for o in origins))
    RESULTS["training_origins"] = {
        "n_origins": len(origins),
        "origin_day_idxs": origins,
        "origin_days": [f"d_{o + 1}" for o in origins],
        "origin_dates": [str(data.date_of(o).date()) for o in origins],
        "stride_days": config.HORIZON,
        "series_sampled_for_check": int(len(sample_series)),
    }

    t0 = time.time()
    train = bt.build_training_frame(
        origins, series_idx=sample_series,
        validation_origin=config.VALIDATION_ORIGIN_IDX,
    )
    print(f"  training frame shape {train.shape} in {time.time() - t0:.1f}s")
    RESULTS["training_frame_sample"] = {
        "rows": int(len(train)),
        "columns": int(train.shape[1]),
        "note": "built on a 2,000-series sample for verification speed only",
    }

    run(vc.check_train_validation_separation(train, config.VALIDATION_ORIGIN_IDX))

    dup = train.duplicated(subset=["series_idx", "origin_idx", "target_day_idx"]).sum()
    run([vc.CheckResult(
        "no_duplicate_training_rows", dup == 0,
        f"{dup} duplicate (series, origin, target_day) rows",
    )])

    # ------------------------------------------------------------------
    section("7b. LISTING-AWARE FEATURE BEHAVIOUR ACROSS ORIGINS")
    # ------------------------------------------------------------------
    probe_origins = [200, 700, 1400, config.VALIDATION_ORIGIN_IDX]
    listing_checks, listing_rows = vc.check_listing_feature_behaviour(data, probe_origins)
    print(pd.DataFrame(listing_rows).to_string(index=False))
    print()
    run(listing_checks)
    RESULTS["listing_feature_probe"] = listing_rows

    # ------------------------------------------------------------------
    section("8. FUTURE-HORIZON FRAME (d_1942..d_1969) — no target expected")
    # ------------------------------------------------------------------
    fut = bt.build_future_frame()
    print(f"  shape {fut.shape}")
    covered = {
        "has_calendar": bool(fut["wday"].notna().all()),
        "has_snap": bool(fut["snap"].notna().all()),
        "has_event_fields": bool(fut["event_name_1"].notna().all()),
        "price_present_pct": round(float(fut["sell_price"].notna().mean() * 100), 2),
        "target_column_absent": "sales" not in fut.columns,
    }
    print(json.dumps(covered, indent=2))
    RESULTS["future_frame"] = {"rows": int(len(fut)), "columns": int(fut.shape[1]), **covered}

    run([
        vc.CheckResult("future_frame_row_count",
                       len(fut) == config.N_SERIES * config.HORIZON,
                       f"{len(fut):,} rows = {config.N_SERIES:,} x {config.HORIZON}"),
        vc.CheckResult("future_frame_has_no_target",
                       "sales" not in fut.columns,
                       "no target column attached (correct: those sales do not exist)"),
        vc.CheckResult("future_frame_has_calendar_and_snap",
                       covered["has_calendar"] and covered["has_snap"],
                       "calendar + state-matched SNAP available for all 28 future days"),
        vc.CheckResult("future_frame_price_coverage",
                       covered["price_present_pct"] > 50,
                       f"{covered['price_present_pct']}% of future rows have a known sell_price"),
    ])

    # ------------------------------------------------------------------
    section("9. METRIC PIPELINE SMOKE TEST  (NOT a model result)")
    # ------------------------------------------------------------------
    # These are trivial arithmetic rules with no fitting of any kind. Their only
    # purpose is to prove the metric code runs on a real 853,720-row window and
    # returns sane numbers. They are NOT baselines, NOT models, and MUST NOT be
    # compared with the team's LightGBM+Tweedie benchmark.
    y_true = valid["sales"].to_numpy()

    smoke = {
        "predict_all_zeros": metrics.evaluate(y_true, np.zeros_like(y_true)),
        "predict_rolling_mean_28 (persistence rule, no fitting)":
            metrics.evaluate(y_true, valid["rolling_mean_28"].to_numpy()),
    }
    for name, m in smoke.items():
        print(f"  {name}:")
        print(f"      RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"WAPE={m['WAPE']:.4f}  bias={m['bias']:+.4f}  n={m['n']:,}")
    RESULTS["metric_smoke_test"] = smoke
    RESULTS["metric_smoke_test_disclaimer"] = (
        "Arithmetic reference points used only to prove the metric code works on a "
        "real 853,720-row validation window. No model was trained. These are NOT "
        "comparable to the team's LightGBM+Tweedie benchmark (RMSE 2.0324 / "
        "MAE 1.0869) and no such comparison is claimed."
    )

    run([vc.CheckResult(
        "metrics_run_on_full_validation_window",
        smoke["predict_all_zeros"]["n"] == config.N_SERIES * config.HORIZON,
        f"metrics computed over {smoke['predict_all_zeros']['n']:,} predictions",
    )])

    # ------------------------------------------------------------------
    section("10. FEATURE INVENTORY")
    # ------------------------------------------------------------------
    feat_cols = [c for c in all_feature_columns() if c in valid.columns]
    rows = []
    for grp, cols in FEATURE_GROUPS.items():
        for c in cols:
            if c not in valid.columns:
                continue
            v = valid[c].to_numpy()
            finite = v[np.isfinite(v)] if np.issubdtype(v.dtype, np.floating) else v
            rows.append({
                "group": grp,
                "feature": c,
                "dtype": str(valid[c].dtype),
                "categorical": c in CATEGORICAL_FEATURES,
                "missing_pct": round(float(pd.isna(valid[c]).mean() * 100), 3),
                "min": round(float(finite.min()), 4) if len(finite) else None,
                "max": round(float(finite.max()), 4) if len(finite) else None,
                "mean": round(float(finite.mean()), 4) if len(finite) else None,
            })
    feat_summary = pd.DataFrame(rows)
    print(feat_summary.to_string(index=False))

    RESULTS["features"] = {
        "n_features": len(feat_cols),
        "n_groups": len(FEATURE_GROUPS),
        "by_group": {g: [c for c in cols if c in valid.columns]
                     for g, cols in FEATURE_GROUPS.items()},
        "categorical": [c for c in CATEGORICAL_FEATURES if c in valid.columns],
        "summary": rows,
    }

    # ------------------------------------------------------------------
    section("11. WRITE ARTIFACTS")
    # ------------------------------------------------------------------
    feat_summary.to_csv(config.ARTIFACTS_DIR / "feature_summary.csv", index=False)

    # Human-inspectable sample: 3 series x all 28 horizon days, with real labels
    show_series = [0, 1500, 20000]
    samp = valid[valid["series_idx"].isin(show_series)].copy()
    samp.insert(1, "series_id", data.series_meta["id"].to_numpy()[samp["series_idx"]])
    samp.insert(2, "target_date",
                pd.to_datetime(data.dates[samp["target_day_idx"].to_numpy()]))
    samp = samp.sort_values(["series_idx", "horizon"])
    samp.to_csv(config.ARTIFACTS_DIR / "sample_features.csv", index=False)
    print(f"  wrote sample_features.csv ({len(samp)} rows)")
    print(f"  wrote feature_summary.csv ({len(feat_summary)} rows)")

    summary = vc.summarize(ALL_CHECKS)
    RESULTS["checks"] = summary
    RESULTS["runtime_seconds"] = round(time.time() - t_start, 1)
    RESULTS["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    out_path = config.ARTIFACTS_DIR / "foundation_checks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"  wrote {out_path.name}")

    # ------------------------------------------------------------------
    section("RESULT")
    # ------------------------------------------------------------------
    print(f"  {summary['passed']}/{summary['total_checks']} checks passed "
          f"in {RESULTS['runtime_seconds']}s")
    if summary["failed"]:
        print("\n  FAILURES:")
        for c in ALL_CHECKS:
            if not c.passed:
                print(f"    - {c.name}: {c.detail}")
        sys.exit(1)
    print("  All foundation checks passed.")


if __name__ == "__main__":
    main()
