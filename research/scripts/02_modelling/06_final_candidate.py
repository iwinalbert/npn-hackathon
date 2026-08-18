"""
Phases 8-9: apply the inner-window-selected configuration to the PRIMARY
validation window, then stress-test it on additional historical windows.

Two separate things happen here, and it matters that they are separate:

  1. The configuration chosen in script 05 (using only d_1886..d_1913) is run
     ONCE against d_1914..d_1941. Because the primary window played no part in
     choosing that configuration, this is an honest held-out estimate.

  2. That same configuration is retrained and evaluated on other 28-day windows
     from different parts of the calendar — including one containing Christmas,
     which the EDA showed is the single most extreme day in the dataset
     (-99.95% vs local baseline). A model that only looks good on one lucky
     spring window would be exposed here.

    python scripts/06_final_candidate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, feature_sets, metrics, models
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment

FEATURE_SET = "base_recency_listing"
TWEEDIE = {"objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse"}
TUNING_CSV = config.ARTIFACTS_DIR / "inner_window_tuning.csv"


def train_and_eval(bt, data, origin_idx, cfg, cols, tag, exp_name, extra_note=""):
    """Train with `cfg` on origins before `origin_idx`, evaluate on its 28 days."""
    wd = bt.make_window(origin_idx).describe()
    valid = bt.build_validation_frame(origin_idx)
    y = valid["sales"].to_numpy()

    origins = bt.training_origins(origin_idx, n_origins=int(cfg["n_origins"]))
    params = dict(TWEEDIE)
    params.update({"num_leaves": int(cfg["num_leaves"]),
                   "learning_rate": float(cfg["learning_rate"])})

    exp = Experiment(
        exp_name,
        model_type="LightGBM",
        objective="tweedie (variance_power=1.1)",
        feature_set=FEATURE_SET,
        feature_groups=feature_sets.groups_in(FEATURE_SET),
        n_features=len(cols),
        window_tag=tag,
        validation_origin_day=wd["forecast_origin_day"],
        validation_dates=wd["validation_dates"],
        validation_days=wd["validation_days"],
        horizon=config.HORIZON,
        n_series=config.N_SERIES,
        validation_rows=int(len(valid)),
        selected_config=dict(cfg),
    )
    if extra_note:
        exp.note(extra_note)

    X, yt, binfo = models.build_training_matrix(
        bt, origins, cols, validation_origin=origin_idx, verbose=False)
    exp.set(training_data=binfo, training_rows=binfo["rows"],
            training_origins=binfo["origin_days"])
    print(f"   training rows: {binfo['rows']:,}")

    booster, minfo = models.train_lightgbm(
        X, yt, cols, params=params, n_estimators=int(cfg["n_estimators"]), verbose=True)
    exp.set(**{k: v for k, v in minfo.items() if k != "params"})
    exp.set(hyperparameters=minfo["params"])
    del X, yt

    preds = models.predict_nonneg(booster, valid[cols].to_numpy(np.float32))
    if np.isnan(preds).any():
        exp.error("NaN predictions"); exp.save()
        raise SystemExit("STOP: NaN predictions")

    m = metrics.evaluate(y, preds)
    exp.set_metrics(**m)
    return exp, booster, valid, preds, m, wd


def main() -> None:
    t_start = time.time()
    if not TUNING_CSV.exists():
        raise SystemExit(f"{TUNING_CSV} missing — run scripts/05_tune_inner_window.py first")

    tune = pd.read_csv(TUNING_CSV).sort_values("inner_RMSE")
    best_cfg = tune.iloc[0].to_dict()

    print("=" * 78)
    print("PHASE 8 — SELECTED CONFIGURATION APPLIED TO THE PRIMARY WINDOW")
    print("=" * 78)
    print(f"  configuration chosen on the inner window: {best_cfg['tag']}")
    print(f"    rounds={int(best_cfg['n_estimators'])} "
          f"leaves={int(best_cfg['num_leaves'])} lr={best_cfg['learning_rate']} "
          f"origins={int(best_cfg['n_origins'])}")
    print(f"    inner RMSE was {best_cfg['inner_RMSE']:.4f}\n")

    data = M5Data()
    bt = Backtester(data)
    cols = feature_sets.get(FEATURE_SET)
    VO = config.VALIDATION_ORIGIN_IDX

    exp, booster, valid, preds, m, wd = train_and_eval(
        bt, data, VO, best_cfg, cols, "primary", "model_06_tuned_primary",
        extra_note=(
            "Hyperparameters were selected on the inner window d_1886..d_1913 "
            "(script 05) and applied here unchanged. The primary window played no "
            "part in the selection, so this is an unbiased held-out estimate."
        ),
    )

    mpath = config.MODELS_DIR / "model_06_tuned_primary.txt"
    booster.save_model(str(mpath))
    exp.set(model_path=str(mpath.relative_to(config.PROJECT_ROOT)))

    pred_df = pd.DataFrame({
        "series_idx": valid["series_idx"].to_numpy(),
        "target_day_idx": valid["target_day_idx"].to_numpy(),
        "horizon": valid["horizon"].to_numpy(),
        "y_true": valid["sales"].to_numpy(),
        "y_pred": np.round(preds, 5),
    })
    ppath = config.PREDICTIONS_DIR / "model_06_tuned_primary_validation.csv"
    pred_df.to_csv(ppath, index=False)
    exp.set(prediction_path=str(ppath.relative_to(config.PROJECT_ROOT)))
    exp.save()

    print(f"\n   PRIMARY  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
          f"WAPE={m['WAPE']:.4f}  bias={m['bias']:+.4f}")

    prev_best = 2.1210  # model_04, measured earlier in this project
    print(f"   previous best (Model 4, untuned): RMSE 2.1210")
    print(f"   change: {m['RMSE'] - prev_best:+.4f} RMSE")

    # ==================================================================
    print()
    print("=" * 78)
    print("PHASE 9 — ADDITIONAL VALIDATION WINDOWS")
    print("=" * 78)
    print("  Same configuration, retrained per window using only data before that")
    print("  window. Purpose: check the result is not specific to one lucky period.\n")

    dates = pd.to_datetime(data.calendar["date"])

    def idx_of(datestr: str) -> int:
        hit = data.calendar.index[dates == pd.Timestamp(datestr)]
        return int(hit[0])

    # A window centred on Christmas 2015 — the most extreme calendar effect in
    # the dataset, and the hardest test of whether event handling works.
    xmas = idx_of("2015-12-25")
    xmas_origin = xmas - 14
    # An ordinary summer window with no major holidays.
    summer_origin = idx_of("2015-07-15")

    windows = [
        ("primary_spring_2016", VO, m),
        ("christmas_2015", xmas_origin, None),
        ("summer_2015", summer_origin, None),
    ]

    rows = [{
        "window": "primary_spring_2016",
        "origin_day": wd["forecast_origin_day"],
        "dates": wd["validation_dates"],
        "RMSE": m["RMSE"], "MAE": m["MAE"], "WAPE": m["WAPE"],
    }]

    for tag, origin, done in windows:
        if done is not None:
            continue
        w = bt.make_window(origin).describe()
        print(f"[{tag}] origin {w['forecast_origin_day']} "
              f"({w['forecast_origin_date']}) -> {w['validation_dates']}")

        e2, b2, v2, p2, m2, w2 = train_and_eval(
            bt, data, origin, best_cfg, cols, tag, f"model_06_window_{tag}",
            extra_note="Additional validation window using the configuration "
                       "selected on the inner window. Retrained from scratch on "
                       "origins before this window only.",
        )
        e2.save()
        print(f"   RMSE={m2['RMSE']:.4f}  MAE={m2['MAE']:.4f}  WAPE={m2['WAPE']:.4f}\n")
        rows.append({
            "window": tag, "origin_day": w2["forecast_origin_day"],
            "dates": w2["validation_dates"],
            "RMSE": m2["RMSE"], "MAE": m2["MAE"], "WAPE": m2["WAPE"],
        })
        del b2, v2

    df = pd.DataFrame(rows)
    out = config.ARTIFACTS_DIR / "multi_window_results.csv"
    df.to_csv(out, index=False)

    print("=" * 78)
    print("MULTI-WINDOW RESULTS (measured)")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\n  wrote {out.relative_to(config.PROJECT_ROOT)}")
    print(f"  total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
