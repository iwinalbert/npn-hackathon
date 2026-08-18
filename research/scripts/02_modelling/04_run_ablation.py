
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

N_TRAIN_ORIGINS = 15
N_ESTIMATORS = 400

LADDER = [
    "abl_1_calendar",
    "abl_2_calendar_demand",
    "abl_3_plus_recency",
    "abl_4_plus_price",
    "abl_5_plus_listing",
    "abl_6_plus_hierarchy",
    "abl_7_full",
]

TWEEDIE = {"objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse"}


def main() -> None:
    t_start = time.time()
    print("=" * 78)
    print("FEATURE-GROUP ABLATION — one group added per rung, everything else fixed")
    print("=" * 78)

    data = M5Data()
    bt = Backtester(data)
    VO = config.VALIDATION_ORIGIN_IDX
    valid = bt.build_validation_frame(VO)
    y_true = valid["sales"].to_numpy()
    origins = bt.training_origins(VO, n_origins=N_TRAIN_ORIGINS)
    wd = bt.make_window(VO).describe()

    print(f"  validation : {wd['validation_days']} ({wd['validation_dates']}), "
          f"{len(valid):,} predictions")
    print(f"  objective  : Tweedie (variance_power=1.1), {N_ESTIMATORS} rounds, untuned")
    print(f"  origins    : {len(origins)}\n")

    results = []
    prev = None
    for fs_name in LADDER:
        cols = feature_sets.get(fs_name)
        label = feature_sets.FEATURE_SET_LABELS[fs_name]
        groups = feature_sets.groups_in(fs_name)
        print(f"[{fs_name}] {label}")
        print(f"   {len(cols)} features, groups {groups}")

        exp = Experiment(
            f"ablation_{fs_name}",
            model_type="LightGBM",
            objective="tweedie (variance_power=1.1)",
            feature_set=fs_name,
            feature_set_label=label,
            feature_groups=groups,
            features=cols,
            n_features=len(cols),
            validation_origin_day=wd["forecast_origin_day"],
            validation_dates=wd["validation_dates"],
            validation_days=wd["validation_days"],
            horizon=config.HORIZON,
            n_series=config.N_SERIES,
            validation_rows=int(len(valid)),
        )
        exp.note("Feature-group ablation rung. Only the feature set differs between "
                 "rungs; objective, hyperparameters, origins and validation are fixed.")

        X, y, binfo = models.build_training_matrix(
            bt, origins, cols, validation_origin=VO, verbose=False)
        exp.set(training_data=binfo, training_rows=binfo["rows"],
                training_origins=binfo["origin_days"])

        booster, minfo = models.train_lightgbm(
            X, y, cols, params=TWEEDIE, n_estimators=N_ESTIMATORS, verbose=True)
        exp.set(**{k: v for k, v in minfo.items() if k != "params"})
        exp.set(hyperparameters=minfo["params"])
        del X, y

        preds = models.predict_nonneg(booster, valid[cols].to_numpy(np.float32))
        if np.isnan(preds).any():
            exp.error("NaN predictions"); exp.save(); raise SystemExit("NaN predictions")

        m = metrics.evaluate(y_true, preds)
        exp.set_metrics(**m)

        d_rmse = None if prev is None else m["RMSE"] - prev["RMSE"]
        d_mae = None if prev is None else m["MAE"] - prev["MAE"]
        exp.set(delta_rmse_vs_previous_rung=d_rmse, delta_mae_vs_previous_rung=d_mae)
        exp.save()

        dtxt = "" if d_rmse is None else f"   (dRMSE {d_rmse:+.4f}, dMAE {d_mae:+.4f})"
        print(f"   RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}{dtxt}\n")

        results.append({
            "rung": fs_name, "label": label, "groups": "".join(groups),
            "n_features": len(cols), "RMSE": m["RMSE"], "MAE": m["MAE"],
            "d_RMSE": d_rmse, "d_MAE": d_mae,
            "train_seconds": minfo["training_seconds"],
        })
        prev = m

    df = pd.DataFrame(results)
    out = config.ARTIFACTS_DIR / "ablation_results.csv"
    df.to_csv(out, index=False)

    print("=" * 78)
    print("ABLATION LADDER (measured)")
    print("=" * 78)
    print(f"  {'Configuration':<34} {'#F':>4} {'RMSE':>8} {'MAE':>8} "
          f"{'dRMSE':>9} {'dMAE':>9}")
    print("  " + "-" * 76)
    for r in results:
        dr = "  —" if r["d_RMSE"] is None else f"{r['d_RMSE']:+.4f}"
        dm = "  —" if r["d_MAE"] is None else f"{r['d_MAE']:+.4f}"
        print(f"  {r['label']:<34} {r['n_features']:>4} {r['RMSE']:>8.4f} "
              f"{r['MAE']:>8.4f} {dr:>9} {dm:>9}")
    print(f"\n  wrote {out.relative_to(config.PROJECT_ROOT)}")
    print(f"  total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
