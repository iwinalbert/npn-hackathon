"""
Tweedie variance-power probe — the one legitimate lever our error analysis points at.

MOTIVATION (from measured evidence, not intuition)
--------------------------------------------------
Error analysis showed our model under-predicts the busiest series: high-volume
rows are 7.7% of the validation set but carry 61% of all squared error, with a
bias of -0.389. RMSE is dominated by exactly those rows.

The Tweedie variance power controls how much probability mass the objective puts
at zero versus in the tail. p -> 1 behaves like Poisson (lighter zero-inflation,
more weight on getting counts right); p -> 2 behaves like Gamma. We used 1.1
everywhere without ever testing it, so it is an untested assumption sitting
directly on top of the thing limiting our RMSE.

DISCIPLINE
----------
This runs on the INNER window (origin d_1885, targets d_1886..d_1913). The
primary window d_1914..d_1941 is not touched, so it remains an unbiased estimate
and nothing here can leak into the headline number.

    python scripts/12_tweedie_power_probe.py
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

INNER_ORIGIN = config.VALIDATION_ORIGIN_IDX - config.HORIZON   # d_1885
FEATURE_SET = "base_recency_listing"
POWERS = [1.1, 1.3, 1.5]          # 1.1 is our current setting


def main():
    t_start = time.time()
    print("=" * 78)
    print("TWEEDIE VARIANCE-POWER PROBE (inner window only)")
    print("=" * 78)

    data = M5Data()
    bt = Backtester(data)
    iw = bt.make_window(INNER_ORIGIN).describe()
    print(f"  inner validation : {iw['validation_days']} ({iw['validation_dates']})")
    print(f"  primary window   : d_1914 .. d_1941  <- untouched here\n")

    valid = bt.build_validation_frame(INNER_ORIGIN)
    y = valid["sales"].to_numpy()
    assert int(valid["target_day_idx"].max()) < config.VALIDATION_ORIGIN_IDX + 1

    cols = feature_sets.get(FEATURE_SET)
    origins = bt.training_origins(INNER_ORIGIN, n_origins=15)

    # Historical volume tiers, measured on pre-origin history only.
    hist = data.sales_wide[:, :INNER_ORIGIN + 1].mean(axis=1)
    tier = pd.Series(pd.cut(hist[valid["series_idx"].to_numpy()],
                            [-0.001, 0.2, 1.0, 3.0, np.inf],
                            labels=["very low", "low", "medium", "high"]))
    high = (tier == "high").to_numpy()

    rows = []
    for p in POWERS:
        print(f"[power={p}]")
        params = {"objective": "tweedie", "tweedie_variance_power": p,
                  "metric": "rmse"}
        exp = Experiment(
            f"probe_tweedie_power_{str(p).replace('.', '_')}",
            model_type="LightGBM", objective=f"tweedie (variance_power={p})",
            feature_set=FEATURE_SET, n_features=len(cols),
            tuning_window="INNER",
            validation_days=iw["validation_days"],
            validation_dates=iw["validation_dates"],
            horizon=config.HORIZON, n_series=config.N_SERIES,
            validation_rows=int(len(valid)),
        )
        exp.note("Probe of the Tweedie variance power on the INNER window. "
                 "Motivated by measured under-prediction of high-volume series. "
                 "The primary window is not used.")

        X, yt, binfo = models.build_training_matrix(
            bt, origins, cols, validation_origin=INNER_ORIGIN, verbose=False)
        booster, minfo = models.train_lightgbm(
            X, yt, cols, params=params, n_estimators=400, verbose=False)
        del X, yt

        preds = models.predict_nonneg(booster, valid[cols].to_numpy(np.float32))
        m = metrics.evaluate(y, preds)
        hm = metrics.evaluate(y[high], preds[high])

        exp.set_metrics(**m)
        exp.set(training_rows=binfo["rows"], hyperparameters=minfo["params"],
                n_estimators=400, training_seconds=minfo["training_seconds"],
                high_volume_RMSE=hm["RMSE"], high_volume_bias=hm["bias"])
        exp.save()

        print(f"   inner RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"bias={m['bias']:+.4f}")
        print(f"   high-volume tier: RMSE={hm['RMSE']:.4f}  bias={hm['bias']:+.4f}\n")

        rows.append({"power": p, "inner_RMSE": m["RMSE"], "inner_MAE": m["MAE"],
                     "bias": m["bias"], "high_vol_RMSE": hm["RMSE"],
                     "high_vol_bias": hm["bias"],
                     "train_seconds": minfo["training_seconds"]})
        del booster

    df = pd.DataFrame(rows).sort_values("inner_RMSE")
    df.to_csv(config.ARTIFACTS_DIR / "tweedie_power_probe.csv", index=False)

    print("=" * 78)
    print("RESULT (ranked by inner-window RMSE)")
    print("=" * 78)
    print(df.to_string(index=False))
    best = df.iloc[0]
    cur = df[df.power == 1.1].iloc[0]
    print(f"\n  current setting (1.1): inner RMSE {cur['inner_RMSE']:.4f}")
    print(f"  best setting ({best['power']}): inner RMSE {best['inner_RMSE']:.4f}")
    if best["power"] != 1.1:
        print(f"  -> a change of {best['inner_RMSE'] - cur['inner_RMSE']:+.4f} RMSE "
              f"on the inner window")
    else:
        print("  -> our existing setting is already the best of those tested")
    print(f"\n  total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
