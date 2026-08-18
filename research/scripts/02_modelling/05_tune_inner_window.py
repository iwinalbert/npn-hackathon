
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

INNER_ORIGIN = config.VALIDATION_ORIGIN_IDX - config.HORIZON
FEATURE_SET = "base_recency_listing"

TWEEDIE = {"objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse"}

GRID = [
    {"tag": "A_current_settings",  "n_estimators": 400,  "num_leaves": 128, "lr": 0.05, "origins": 15},
    {"tag": "B_more_rounds",       "n_estimators": 1200, "num_leaves": 128, "lr": 0.05, "origins": 15},
    {"tag": "C_rounds_and_leaves", "n_estimators": 1200, "num_leaves": 256, "lr": 0.05, "origins": 15},
    {"tag": "D_more_history",      "n_estimators": 2000, "num_leaves": 256, "lr": 0.05, "origins": 20},
]


def main() -> None:
    t_start = time.time()
    print("=" * 78)
    print("INNER-WINDOW CAPACITY SELECTION")
    print("=" * 78)

    data = M5Data()
    bt = Backtester(data)

    iw = bt.make_window(INNER_ORIGIN).describe()
    pw = bt.make_window(config.VALIDATION_ORIGIN_IDX).describe()
    print(f"  INNER  validation : {iw['validation_days']} ({iw['validation_dates']})")
    print(f"  PRIMARY validation: {pw['validation_days']} ({pw['validation_dates']})"
          f"  <- untouched by this script")

    inner_valid = bt.build_validation_frame(INNER_ORIGIN)
    y_inner = inner_valid["sales"].to_numpy()
    assert len(inner_valid) == config.N_SERIES * config.HORIZON

    first_primary_day = config.VALIDATION_ORIGIN_IDX + 1
    assert int(inner_valid["target_day_idx"].max()) < first_primary_day, \
        "inner validation overlaps the primary validation window"
    print(f"  guard: inner targets end at day index "
          f"{int(inner_valid['target_day_idx'].max())}, primary starts at "
          f"{first_primary_day} — no overlap\n")

    cols = feature_sets.get(FEATURE_SET)
    rows = []

    for cfg in GRID:
        print(f"[{cfg['tag']}] rounds={cfg['n_estimators']} leaves={cfg['num_leaves']} "
              f"lr={cfg['lr']} origins={cfg['origins']}")

        origins = bt.training_origins(INNER_ORIGIN, n_origins=cfg["origins"])
        params = dict(TWEEDIE)
        params.update({"num_leaves": cfg["num_leaves"], "learning_rate": cfg["lr"]})

        exp = Experiment(
            f"tune_inner_{cfg['tag']}",
            model_type="LightGBM",
            objective="tweedie (variance_power=1.1)",
            feature_set=FEATURE_SET,
            feature_groups=feature_sets.groups_in(FEATURE_SET),
            n_features=len(cols),
            tuning_window="INNER",
            validation_origin_day=iw["forecast_origin_day"],
            validation_dates=iw["validation_dates"],
            validation_days=iw["validation_days"],
            horizon=config.HORIZON,
            n_series=config.N_SERIES,
            validation_rows=int(len(inner_valid)),
            grid_config=cfg,
        )
        exp.note(
            "Capacity selection on the INNER window only. The primary validation "
            "window (d_1914..d_1941) is not used anywhere in this run, so it stays "
            "an honest held-out estimate."
        )

        X, y, binfo = models.build_training_matrix(
            bt, origins, cols, validation_origin=INNER_ORIGIN, verbose=False)
        print(f"   training rows: {binfo['rows']:,} ({binfo['memory_mb']} MB)")
        exp.set(training_data=binfo, training_rows=binfo["rows"],
                training_origins=binfo["origin_days"])

        booster, minfo = models.train_lightgbm(
            X, y, cols, params=params, n_estimators=cfg["n_estimators"], verbose=True)
        exp.set(**{k: v for k, v in minfo.items() if k != "params"})
        exp.set(hyperparameters=minfo["params"])
        del X, y

        preds = models.predict_nonneg(booster, inner_valid[cols].to_numpy(np.float32))
        if np.isnan(preds).any():
            exp.error("NaN predictions"); exp.save(); raise SystemExit("NaN predictions")

        m = metrics.evaluate(y_inner, preds)
        exp.set_metrics(**m)
        exp.save()
        print(f"   INNER RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"({minfo['training_seconds']}s)\n")

        rows.append({
            "tag": cfg["tag"], "n_estimators": cfg["n_estimators"],
            "num_leaves": cfg["num_leaves"], "learning_rate": cfg["lr"],
            "n_origins": cfg["origins"], "training_rows": binfo["rows"],
            "inner_RMSE": m["RMSE"], "inner_MAE": m["MAE"],
            "train_seconds": minfo["training_seconds"],
        })
        del booster

    df = pd.DataFrame(rows).sort_values("inner_RMSE")
    out = config.ARTIFACTS_DIR / "inner_window_tuning.csv"
    df.to_csv(out, index=False)

    print("=" * 78)
    print("INNER-WINDOW RESULTS (ranked by RMSE on d_1886..d_1913)")
    print("=" * 78)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\n  Selected configuration: {best['tag']}")
    print(f"    rounds={int(best['n_estimators'])}, leaves={int(best['num_leaves'])}, "
          f"lr={best['learning_rate']}, origins={int(best['n_origins'])}")
    print(f"    inner RMSE={best['inner_RMSE']:.4f}, MAE={best['inner_MAE']:.4f}")
    print("\n  This choice used ONLY the inner window. Script 06 applies it once to")
    print("  the primary window, which therefore remains an unbiased estimate.")
    print(f"\n  wrote {out.relative_to(config.PROJECT_ROOT)}")
    print(f"  total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
