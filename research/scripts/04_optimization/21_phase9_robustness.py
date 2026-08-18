
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics, optimize
from pipeline.experiment import Experiment
from pipeline.features_v2 import V2_SETS

COLS = V2_SETS["v2_base"]

CANDIDATES = {
    "tweedie_1_1": {"objective": "tweedie", "tweedie_variance_power": 1.1},
    "tweedie_1_5": {"objective": "tweedie", "tweedie_variance_power": 1.5},
    "l1": {"objective": "regression_l1"},
}


def main():
    t0 = time.time()
    data_probe = optimize.Setup()
    cal = data_probe.data.calendar
    dates = pd.to_datetime(cal["date"])

    def idx_of(ds):
        return int(cal.index[dates == pd.Timestamp(ds)][0])

    windows = {
        "primary_spring_2016": config.VALIDATION_ORIGIN_IDX,
        "christmas_2015": idx_of("2015-12-25") - 14,
        "summer_2015": idx_of("2015-07-15"),
        "autumn_2015": idx_of("2015-10-01"),
    }
    del data_probe

    print("=" * 78)
    print("PHASE 9 — ROBUSTNESS ACROSS WINDOWS")
    print("=" * 78)
    for k, o in windows.items():
        print(f"  {k:<22} origin d_{o+1}")
    print()

    rows = []
    for wname, origin in windows.items():
        s = optimize.Setup(origin_idx=origin)
        print(f"\n--- {wname}: {s.window['validation_days']} "
              f"({s.window['validation_dates']}) ---")
        X, Y = optimize.build_matrix(s, COLS)
        for cname, params in CANDIDATES.items():
            booster, info = optimize.train(X, Y, COLS, params=params)
            p = optimize.predict(booster, s, COLS)
            d = optimize.diagnostics(s.y, p, s)
            print(f"   {cname:<14} RMSE={d['RMSE']:.4f}  MAE={d['MAE']:.4f}  "
                  f"hv={d['high_volume_RMSE']:.3f}")
            rows.append({"window": wname, "model": cname,
                         "origin_day": s.window["forecast_origin_day"],
                         "dates": s.window["validation_dates"],
                         "RMSE": d["RMSE"], "MAE": d["MAE"],
                         "high_vol_RMSE": d["high_volume_RMSE"]})
            e = Experiment(f"opt_09_robust_{wname}_{cname}", model_type="LightGBM",
                           objective=params["objective"],
                           feature_set_label=f"Robustness: {cname} on {wname}",
                           n_features=len(COLS), **s.describe(),
                           window_tag=wname, robustness_run=True)
            e.note("Phase 9. Retrained from scratch on origins before this window.")
            e.set_metrics(**d)
            e.set(hyperparameters=info["params"],
                  training_seconds=info["training_seconds"])
            e.save()
            del booster
        del X, Y, s

    df = pd.DataFrame(rows)
    df.to_csv(config.ARTIFACTS_DIR / "phase9_robustness.csv", index=False)

    print("\n" + "=" * 78)
    print("SUMMARY — consistency across windows")
    print("=" * 78)
    agg = df.groupby("model").agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        RMSE_worst=("RMSE", "max"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
    ).reset_index().sort_values("RMSE_mean")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\nPer window:")
    piv = df.pivot(index="window", columns="model", values="RMSE")
    print(piv.to_string(float_format=lambda v: f"{v:8.4f}"))

    agg.to_csv(config.ARTIFACTS_DIR / "phase9_robustness_summary.csv", index=False)
    (config.ARTIFACTS_DIR / "phase9_robustness.json").write_text(
        json.dumps({"per_run": rows, "summary": agg.to_dict(orient="records")},
                   indent=2, default=str), encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
