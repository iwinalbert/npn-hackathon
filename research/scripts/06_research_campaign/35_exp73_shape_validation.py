
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
from pipeline.backtest import Backtester
from pipeline.experiment import Experiment
from pipeline.features_v2 import V2_SETS
from pipeline.features_v4 import FeatureBuilderV4, feature_set

BASE = V2_SETS["v2_base"]
SHAPE = feature_set()
SEEDS = [42, 7, 202]


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def setup(origin):
    s = optimize.Setup(origin_idx=origin)
    s.fb = FeatureBuilderV4(s.data)
    s.bt = Backtester(s.data, feature_builder=s.fb)
    s.valid = s.bt.build_validation_frame(origin)
    s.y = s.valid["sales"].to_numpy()
    return s


def fit_eval(s, cols, seed=42):
    X, Y = optimize.build_matrix(s, cols)
    b, info = optimize.train(X, Y, cols, params={"seed": seed,
                                                 "bagging_seed": seed,
                                                 "feature_fraction_seed": seed})
    del X, Y
    p = optimize.predict(b, s, cols)
    del b
    return metrics.rmse(s.y, p), metrics.mae(s.y, p), info["training_seconds"]


def main():
    t0 = time.time()
    banner("EXPERIMENT #73 — VALIDATING THE SHAPE-FEATURE EFFECT")

    probe = optimize.Setup()
    cal = probe.data.calendar
    dates = pd.to_datetime(cal["date"])
    idx = lambda ds: int(cal.index[dates == pd.Timestamp(ds)][0])
    WINDOWS = {
        "primary_spring_2016": config.VALIDATION_ORIGIN_IDX,
        "christmas_2015": idx("2015-12-25") - 14,
        "summer_2015": idx("2015-07-15"),
        "autumn_2015": idx("2015-10-01"),
    }
    del probe

    banner("TEST A — CROSS-WINDOW CONSISTENCY (both models retrained per window)")
    rows = []
    for wname, o in WINDOWS.items():
        s = setup(o)
        rb, mb, _ = fit_eval(s, BASE)
        rs, ms, _ = fit_eval(s, SHAPE)
        rows.append({"window": wname, "dates": s.window["validation_dates"],
                     "champion_RMSE": rb, "shape_RMSE": rs, "dRMSE": rs - rb,
                     "champion_MAE": mb, "shape_MAE": ms, "dMAE": ms - mb})
        print(f"  {wname:<20} champion {rb:.4f} / {mb:.4f}    "
              f"shape {rs:.4f} / {ms:.4f}    dRMSE {rs-rb:+.4f}  dMAE {ms-mb:+.4f}")
        del s

    W = pd.DataFrame(rows)
    wins = int((W.dRMSE < 0).sum())
    mean_d = float(W.dRMSE.mean())
    mean_dm = float(W.dMAE.mean())
    print(f"\n  shape wins {wins}/4 windows   mean dRMSE {mean_d:+.4f}   "
          f"mean dMAE {mean_dm:+.4f}")
    print(f"  probability of {wins}/4 by chance if the feature were useless: "
          f"{[1,4,6,4,1][wins]/16:.3f}")

    banner("TEST B — SEED SENSITIVITY (primary window, 3 seeds each)")
    s = setup(config.VALIDATION_ORIGIN_IDX)
    srows = []
    for seed in SEEDS:
        rb, mb, _ = fit_eval(s, BASE, seed)
        rs, ms, _ = fit_eval(s, SHAPE, seed)
        srows.append({"seed": seed, "champion_RMSE": rb, "shape_RMSE": rs,
                      "dRMSE": rs - rb, "dMAE": ms - mb})
        print(f"  seed {seed:<4} champion {rb:.4f}   shape {rs:.4f}   "
              f"dRMSE {rs-rb:+.4f}")
    S = pd.DataFrame(srows)
    seed_wins = int((S.dRMSE < 0).sum())
    print(f"\n  shape wins {seed_wins}/{len(SEEDS)} seeds   "
          f"mean dRMSE {S.dRMSE.mean():+.4f}")
    print(f"  champion RMSE across seeds: {S.champion_RMSE.min():.4f} - "
          f"{S.champion_RMSE.max():.4f} (spread {S.champion_RMSE.max()-S.champion_RMSE.min():.4f})")

    banner("DECISION")
    crit = {
        "wins_at_least_3_of_4_windows": wins >= 3,
        "mean_window_dRMSE_at_most_-0.003": mean_d <= -0.003,
        "wins_at_least_2_of_3_seeds": seed_wins >= 2,
        "mean_window_dMAE_not_positive": mean_dm <= 0,
    }
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    accepted = all(crit.values())
    print(f"\n  -> {'NEW CHAMPION' if accepted else 'CHAMPION STANDS'}")

    exp = Experiment("exp_73_shape_feature_validation",
                     model_type="validation of exp_72 (LightGBM Tweedie)",
                     objective="tweedie (variance_power=1.1)",
                     feature_set_label="Champion 32 vs Champion 32 + 4 shape features",
                     n_features=len(SHAPE), **s.describe())
    exp.note("Experiment #73. #72 met neither the pre-registered -0.010 magnitude "
             "threshold nor was it dismissible, since RMSE, MAE and high-volume "
             "error all moved the right way. This tests consistency instead of "
             "magnitude: 4 windows and 3 seeds, acceptance criteria fixed before "
             "running.")
    exp.set_metrics(RMSE=float(W[W.window == "primary_spring_2016"].shape_RMSE.iloc[0]),
                    MAE=float(W[W.window == "primary_spring_2016"].shape_MAE.iloc[0]))
    exp.set(cross_window=rows, window_wins=wins, mean_window_dRMSE=mean_d,
            mean_window_dMAE=mean_dm, seed_runs=srows, seed_wins=seed_wins,
            acceptance_criteria=crit, accepted=accepted,
            decision="NEW CHAMPION" if accepted else "CHAMPION STANDS")
    exp.save()

    W.to_csv(config.ARTIFACTS_DIR / "exp73_cross_window.csv", index=False)
    S.to_csv(config.ARTIFACTS_DIR / "exp73_seeds.csv", index=False)
    (config.ARTIFACTS_DIR / "exp73_summary.json").write_text(json.dumps({
        "cross_window": rows, "wins": wins, "mean_dRMSE": mean_d,
        "mean_dMAE": mean_dm, "seeds": srows, "seed_wins": seed_wins,
        "criteria": crit, "accepted": accepted}, indent=2, default=str),
        encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
