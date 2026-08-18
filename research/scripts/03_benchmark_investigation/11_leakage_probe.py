"""
DIAGNOSTIC PROBE — what would an accidental leak have scored?

WHY THIS EXISTS
---------------
We have ruled out the ordinary explanations for the team's RMSE 2.0324:

  * it is not calibration — no rescaling of our predictions reaches below 2.1195
  * it is not the validation window — ours already has the highest mean demand of
    any 28-day window in the last two years
  * it is not per-target-day feature construction — that scored WORSE (2.1835)

The most common remaining explanation for a score that cannot be reproduced by
legitimate means is feature leakage: computing lag_1 / rolling_mean_7 relative to
each TARGET day rather than to the forecast origin. On day 20 of the horizon,
"yesterday's sales" is a real sales value from inside the validation window — a
number nobody would have on the day the forecast was actually made.

This script builds exactly that leaky feature set and measures it. The result is
a reference point for interpreting an unreproducible score. It is NOT a model, it
is NEVER used to forecast, and it is written to a clearly-labelled experiment id.

    python scripts/11_leakage_probe.py
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

import lightgbm as lgb

from pipeline import config, metrics, team_style
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment

VO = config.VALIDATION_ORIGIN_IDX
VALID_DAYS = VO + 1 + np.arange(config.HORIZON)
TRAIN_DAYS = np.arange(1214, VO + 1)

PARAMS = {
    "objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 128, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 32, "verbosity": -1,
    "num_threads": 0, "seed": config.RANDOM_SEED, "deterministic": True,
    "force_row_wise": True,
}


def main():
    t0 = time.time()
    print("=" * 78)
    print("LEAKAGE PROBE — deliberately unsafe features, for diagnosis only")
    print("=" * 78)
    print("  Building lag_1 / lag_7 / rolling windows RELATIVE TO EACH TARGET DAY.")
    print("  For horizon day 20 this reads sales from inside the validation window.")
    print("  This is exactly the mistake we designed our pipeline to prevent.\n")

    data = M5Data()
    lb = team_style.TeamStyleBuilder(data, min_lookback=1, lags=[1, 7, 14, 28])
    cols = lb.feature_columns

    # Confirm the probe really is leaky — the mirror image of our usual check.
    Xc, _, _ = lb.build(VALID_DAYS, with_target=False)
    corrupt = data.sales_wide.copy()
    corrupt[:, VO + 1:] = 9999
    Xd, _, _ = lb.build(VALID_DAYS, sales=corrupt, with_target=False)
    leaky = not np.array_equal(Xc, Xd, equal_nan=True)
    changed = [cols[j] for j in range(len(cols))
               if not np.array_equal(Xc[:, j], Xd[:, j], equal_nan=True)]
    print(f"  Confirmed leaky: {leaky}. Features that move when the future is "
          f"altered: {changed}")
    if not leaky:
        raise SystemExit("probe is not actually leaky — aborting, it would prove nothing")
    del Xd, corrupt

    exp = Experiment(
        "diagnostic_leakage_probe_DO_NOT_USE",
        model_type="LightGBM (DIAGNOSTIC ONLY)",
        objective="tweedie (variance_power=1.1)",
        feature_set="LEAKY_per_target_day_lag1",
        n_features=len(cols),
        features=list(cols),
        validation_days="d_1914 .. d_1941",
        validation_dates="2016-04-25 .. 2016-05-22",
        horizon=config.HORIZON,
        n_series=config.N_SERIES,
        validation_rows=int(config.N_SERIES * config.HORIZON),
        leaky_features=changed,
    )
    exp.warn("THIS EXPERIMENT DELIBERATELY LEAKS FUTURE SALES. It exists only to "
             "quantify what an accidental leak scores. It must never be used to "
             "produce a forecast or quoted as a result.")

    Xtr, ytr, _ = lb.build(TRAIN_DAYS, with_target=True)
    print(f"\n  training rows: {Xtr.shape[0]:,} x {Xtr.shape[1]}")
    dset = lgb.Dataset(Xtr, label=ytr, feature_name=list(cols),
                       categorical_feature=team_style.categorical_indices(cols),
                       free_raw_data=True)
    t = time.time()
    booster = lgb.train(PARAMS, dset, num_boost_round=400,
                        callbacks=[lgb.log_evaluation(period=0)])
    print(f"  trained in {time.time() - t:.0f}s")
    del Xtr, ytr, dset

    _, yv, _ = lb.build(VALID_DAYS, with_target=True)
    preds = np.clip(booster.predict(Xc), 0, None)
    m = metrics.evaluate(yv, preds)
    exp.set_metrics(**m)
    exp.set(training_rows=int(config.N_SERIES * len(TRAIN_DAYS)),
            hyperparameters=PARAMS, n_estimators=400,
            training_seconds=round(time.time() - t, 1))
    exp.save()

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  LEAKY probe                 : RMSE {m['RMSE']:.4f}  MAE {m['MAE']:.4f}")
    print(f"  our legitimate best (M4)    : RMSE 2.1210  MAE 1.0319")
    print(f"  team reported               : RMSE 2.0324  MAE 1.0869")
    print(f"  team-style (safe, 28d lag)  : RMSE 2.1835  MAE 1.0498")

    out = {
        "leaky_probe": {"RMSE": m["RMSE"], "MAE": m["MAE"], "WAPE": m["WAPE"]},
        "leaky_features": changed,
        "our_best": {"RMSE": 2.1210, "MAE": 1.0319},
        "team_reported": {"RMSE": 2.0324, "MAE": 1.0869},
        "interpretation_rules": (
            "If the leaky probe lands at or below the team's reported RMSE, then "
            "a per-target-day lag leak is a sufficient explanation for a score we "
            "cannot otherwise reproduce. If the probe is still well above their "
            "figure, leakage of this particular kind does NOT explain the gap and "
            "another cause must be sought."
        ),
    }
    (config.ARTIFACTS_DIR / "leakage_probe.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
