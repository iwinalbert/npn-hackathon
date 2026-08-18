
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, experiment, feature_sets, models
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment

PRIMARY_DAYS = "d_1914 .. d_1941"
TWEEDIE = {"objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse"}


def select_final() -> dict:
    cands = []
    for r in experiment.load_all():
        if r.get("status") != "completed":
            continue
        if r.get("tuning_window") == "INNER":
            continue
        if r["experiment_name"].startswith("ablation_"):
            continue
        if r.get("validation_days") != PRIMARY_DAYS:
            continue
        if "RMSE" not in r.get("metrics", {}):
            continue
        cands.append(r)
    if not cands:
        raise SystemExit("no completed primary-window experiments found")
    cands.sort(key=lambda r: r["metrics"]["RMSE"])
    return cands[0]


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("PHASE 11 — FINAL MODEL SELECTION (by measured RMSE, mechanically)")
    print("=" * 78)

    best = select_final()
    name = best["experiment_name"]
    print(f"  selected: {name}")
    print(f"    RMSE={best['metrics']['RMSE']:.4f}  MAE={best['metrics']['MAE']:.4f}")
    print(f"    objective: {best.get('objective')}")
    print(f"    feature set: {best.get('feature_set')} ({best.get('n_features')} features)")

    hp = best.get("hyperparameters", {})
    if "stage1" in hp:
        raise SystemExit(
            "The best model is the two-stage hurdle. This script currently "
            "implements the single-booster path only — stopping rather than "
            "silently forecasting with a different model than the one selected."
        )

    n_estimators = best.get("n_estimators", 400)
    feature_set = best.get("feature_set", "base_recency_listing")
    params = dict(TWEEDIE)
    for k in ("num_leaves", "learning_rate", "objective", "tweedie_variance_power"):
        if k in hp:
            params[k] = hp[k]

    n_origins = len(best.get("training_origins", [])) or 15
    print(f"    rounds={n_estimators} leaves={params.get('num_leaves')} "
          f"lr={params.get('learning_rate')} origins={n_origins}")

    print()
    print("=" * 78)
    print("PHASE 12 — RETRAIN ON ALL USABLE HISTORY (targets through d_1941)")
    print("=" * 78)

    data = M5Data()
    bt = Backtester(data)
    cols = feature_sets.get(feature_set)

    FO = config.FINAL_FORECAST_ORIGIN_IDX
    fw = bt.make_window(FO).describe()
    print(f"  forecast origin : {fw['forecast_origin_day']} ({fw['forecast_origin_date']})")
    print(f"  forecast window : {fw['validation_days']} ({fw['validation_dates']})")
    print(f"  ground truth exists for this window: {fw['validation_has_known_sales']}")

    origins = bt.training_origins(FO, n_origins=n_origins)
    print(f"  training origins: {len(origins)} "
          f"(d_{origins[0] + 1} .. d_{origins[-1] + 1})")
    assert max(origins) + config.HORIZON <= config.LAST_KNOWN_DAY_IDX, \
        "a training target would fall beyond d_1941"

    exp = Experiment(
        "model_07_final_forecast",
        model_type="LightGBM",
        objective=params["objective"],
        feature_set=feature_set,
        feature_groups=feature_sets.groups_in(feature_set),
        n_features=len(cols),
        selected_from=name,
        selection_basis=f"lowest measured RMSE on {PRIMARY_DAYS}",
        forecast_origin_day=fw["forecast_origin_day"],
        forecast_dates=fw["validation_dates"],
        forecast_days=fw["validation_days"],
        horizon=config.HORIZON,
        n_series=config.N_SERIES,
    )
    exp.note(
        "Final model. Retrained from scratch with the configuration that achieved "
        "the lowest measured RMSE on the primary validation window. Forecast "
        "targets d_1942..d_1969 have no ground truth anywhere, so NO accuracy "
        "figure can be quoted for this forecast — only the validation-window "
        "estimate."
    )

    X, y, binfo = models.build_training_matrix(
        bt, origins, cols, validation_origin=config.LAST_KNOWN_DAY_IDX, verbose=False)
    print(f"  training rows: {binfo['rows']:,} ({binfo['memory_mb']} MB)")
    exp.set(training_data=binfo, training_rows=binfo["rows"],
            training_origins=binfo["origin_days"])

    booster, minfo = models.train_lightgbm(
        X, y, cols, params=params, n_estimators=n_estimators, verbose=True)
    exp.set(**{k: v for k, v in minfo.items() if k != "params"})
    exp.set(hyperparameters=minfo["params"])
    del X, y

    mpath = config.MODELS_DIR / "model_07_final_forecast.txt"
    booster.save_model(str(mpath))
    exp.set(model_path=str(mpath.relative_to(config.PROJECT_ROOT)))

    print()
    print("=" * 78)
    print("PHASE 13 — FORECAST d_1942 .. d_1969")
    print("=" * 78)

    future = bt.build_future_frame(FO)
    assert "sales" not in future.columns, "future frame must not carry a target"
    print(f"  future frame: {len(future):,} rows "
          f"({config.N_SERIES:,} series x {config.HORIZON} days), no target attached")

    t = time.time()
    preds = models.predict_nonneg(booster, future[cols].to_numpy(np.float32))
    exp.set(prediction_seconds=round(time.time() - t, 1))
    print(f"  predicted in {time.time() - t:.1f}s")
    print(f"  prediction stats: mean={preds.mean():.4f}  min={preds.min():.4f}  "
          f"max={preds.max():.4f}")
    exp.set(forecast_mean=round(float(preds.mean()), 6),
            forecast_min=round(float(preds.min()), 6),
            forecast_max=round(float(preds.max()), 6))

    wide = preds.reshape(config.HORIZON, config.N_SERIES).T
    series_ids = data.series_meta["id"].to_numpy()

    fc = pd.DataFrame(wide, columns=[f"F{i}" for i in range(1, config.HORIZON + 1)])
    fc.insert(0, "id", series_ids)

    print()
    print("=" * 78)
    print("PHASE 14 — VALIDATE FORECAST STRUCTURE")
    print("=" * 78)

    checks = []

    def chk(nm, ok, detail):
        checks.append({"check": nm, "passed": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}: {detail}")

    chk("row_count_30490", len(fc) == config.N_SERIES,
        f"{len(fc):,} rows (expected {config.N_SERIES:,})")
    chk("forecast_columns_F1_F28",
        list(fc.columns[1:]) == [f"F{i}" for i in range(1, 29)],
        f"{len(fc.columns) - 1} forecast columns, F1..F28")
    chk("no_duplicate_ids", fc["id"].duplicated().sum() == 0,
        f"{int(fc['id'].duplicated().sum())} duplicates")
    chk("no_missing_ids", fc["id"].notna().all(), "all ids present")
    chk("no_nan_values", not fc.iloc[:, 1:].isna().to_numpy().any(),
        f"{int(fc.iloc[:, 1:].isna().to_numpy().sum())} NaN values")
    vals = fc.iloc[:, 1:].to_numpy()
    chk("no_negative_predictions", vals.min() >= 0, f"minimum value {vals.min():.6f}")
    chk("all_finite", np.isfinite(vals).all(), "all values finite")

    sub = pd.read_csv(config.SAMPLE_SUBMISSION_CSV, usecols=["id"])
    eval_ids = sub.loc[sub["id"].str.endswith("_evaluation"), "id"]
    chk("ids_match_sample_submission_evaluation_block",
        set(fc["id"]) == set(eval_ids),
        f"{len(eval_ids):,} evaluation-block ids in the template, "
        f"{len(set(fc['id']) & set(eval_ids)):,} matched")

    chk("id_order_matches_template",
        list(fc["id"]) == list(eval_ids),
        "row order identical to sample_submission.csv's evaluation block")

    out_core = config.FINAL_FORECAST_DIR / "final_forecast_28day.csv"
    fc.to_csv(out_core, index=False)
    print(f"\n  wrote {out_core.relative_to(config.PROJECT_ROOT)} "
          f"({len(fc):,} rows x {len(fc.columns)} cols)")

    full = None
    vpath = config.PROJECT_ROOT / "predictions" / "model_06_tuned_primary_validation.csv"
    if vpath.exists():
        VP = pd.read_csv(vpath)
        vw = VP.pivot(index="series_idx", columns="horizon", values="y_pred")
        vw = vw.reindex(range(config.N_SERIES)).sort_index()
        vw.columns = [f"F{i}" for i in vw.columns]
        vblock = pd.DataFrame({"id": [i.replace("_evaluation", "_validation")
                                      for i in series_ids]})
        for c in [f"F{i}" for i in range(1, 29)]:
            vblock[c] = vw[c].to_numpy()

        full = pd.concat([vblock, fc], ignore_index=True)
        full = full.set_index("id").reindex(sub["id"]).reset_index()
        ok = (len(full) == 60980 and not full.iloc[:, 1:].isna().to_numpy().any())
        chk("full_m5_submission_60980_rows", ok,
            f"{len(full):,} rows, "
            f"{int(full.iloc[:, 1:].isna().to_numpy().sum())} NaN")
        out_full = config.FINAL_FORECAST_DIR / "submission_m5_format.csv"
        full.to_csv(out_full, index=False)
        print(f"  wrote {out_full.relative_to(config.PROJECT_ROOT)} "
              f"({len(full):,} rows)")

    exp.set(structure_checks=checks,
            all_structure_checks_passed=all(c["passed"] for c in checks),
            prediction_path=str(out_core.relative_to(config.PROJECT_ROOT)))

    if not all(c["passed"] for c in checks):
        exp.error("one or more structure checks failed")
        exp.save()
        raise SystemExit("STOP: forecast structure invalid")

    exp.save()

    summary = {
        "final_model": name,
        "validation_rmse": best["metrics"]["RMSE"],
        "validation_mae": best["metrics"]["MAE"],
        "forecast_window": fw["validation_dates"],
        "forecast_days": fw["validation_days"],
        "rows": len(fc),
        "structure_checks_passed": len(checks),
        "files": [str(out_core.relative_to(config.PROJECT_ROOT))]
                 + ([str((config.PREDICTIONS_DIR / 'submission_m5_format.csv')
                         .relative_to(config.PROJECT_ROOT))] if full is not None else []),
    }
    (config.ARTIFACTS_DIR / "final_forecast_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  All {len(checks)} structure checks passed.")
    print(f"  total wall time: {time.time() - t0:.0f}s")
    print("\n  Reminder: d_1942..d_1969 has no ground truth in any file, so no")
    print("  accuracy number can be quoted for this forecast itself. The only")
    print("  honest estimate of its quality is the validation-window result above.")


if __name__ == "__main__":
    main()
