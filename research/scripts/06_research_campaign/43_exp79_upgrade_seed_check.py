
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics, optimize, recursive
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment
from pipeline.features_v2 import FeatureBuilderV2
from pipeline.features_v4 import V4_FEATURES
from pipeline.features_v5 import FeatureBuilderV5, CHAMPION_FEATURES, V5_FEATURES

REC_COLS_V5 = list(recursive.REC_COLS) + list(V4_FEATURES) + list(V5_FEATURES)
EXTRA_SEEDS = [7, 202]
W = 0.60


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


class SharedSetup(optimize.Setup):
    def __init__(self, data, origin_idx, n_origins=optimize.N_ORIGINS):
        self.data = data
        self.fb = FeatureBuilderV5(data)
        self.bt = Backtester(data, feature_builder=self.fb)
        self.origin_idx = origin_idx
        self.window = self.bt.make_window(origin_idx).describe()
        self.valid = self.bt.build_validation_frame(origin_idx)
        self.y = self.valid["sales"].to_numpy()
        self.origins = self.bt.training_origins(origin_idx, n_origins=n_origins)
        hist = data.sales_wide[:, :origin_idx + 1].mean(axis=1)
        self.tier = pd.Series(pd.cut(
            hist[self.valid["series_idx"].to_numpy()],
            [-0.001, 0.2, 1.0, 3.0, np.inf],
            labels=["very low", "low", "medium", "high"]))
        self.high = (self.tier == "high").to_numpy()
        self.is_zero = self.y == 0


def fit_direct(s, seed=42):
    X, Y = optimize.build_matrix(s, CHAMPION_FEATURES)
    b, _ = optimize.train(X, Y, CHAMPION_FEATURES,
                          params={"seed": seed, "bagging_seed": seed,
                                  "feature_fraction_seed": seed})
    del X, Y
    p = optimize.predict(b, s, CHAMPION_FEATURES)
    del b
    gc.collect()
    return p


def fit_recursive(s, seed, upgraded):
    cls = FeatureBuilderV5 if upgraded else FeatureBuilderV2
    cols = REC_COLS_V5 if upgraded else recursive.REC_COLS
    b, _ = recursive.train_one_step(s.data, s.origin_idx, seed=seed,
                                    builder_cls=cls, cols=cols)
    p, work = recursive.recursive_forecast(s.data, b, s.origin_idx,
                                           builder_cls=cls, cols=cols)
    ck = recursive.verify_no_future_leakage(s.data, work, s.origin_idx)
    del b, work
    gc.collect()
    if not ck["passed"]:
        raise SystemExit(f"STOP: leakage check failed: {ck}")
    return p


def main():
    t0 = time.time()
    banner("EXPERIMENT #79 — SEED STABILITY OF THE #77 MEMBER UPGRADE")
    log(f"  member A fixed at seed 42; B and B' at seeds {EXTRA_SEEDS}")
    log(f"  blend weight w = {W} (the shipped operating point)")
    log("\n  PRE-REGISTERED: E1 member B' beats B in >=5 of 6 cells")
    log("                  E2 blend A+B' beats A+B in >=5 of 6 cells")

    data = M5Data()
    cal = data.calendar
    dates = pd.to_datetime(cal["date"])
    idx = lambda ds: int(cal.index[dates == pd.Timestamp(ds)][0])
    WINDOWS = {
        "autumn_2015": idx("2015-10-01"),
        "christmas_2015": idx("2015-12-25") - 14,
    }

    rows = []
    for wname, o in WINDOWS.items():
        banner(f"{wname}  (origin d_{o+1})")
        s = SharedSetup(data, o)
        pa = fit_direct(s, 42)
        log(f"  A (seed 42)  RMSE {metrics.rmse(s.y, pa):.4f}")
        for seed in EXTRA_SEEDS:
            pb = fit_recursive(s, seed, upgraded=False)
            pb2 = fit_recursive(s, seed, upgraded=True)
            bl = np.clip(W * pa + (1 - W) * pb, 0, None)
            bl2 = np.clip(W * pa + (1 - W) * pb2, 0, None)
            r = {
                "window": wname, "seed": seed,
                "B_RMSE": metrics.rmse(s.y, pb), "B2_RMSE": metrics.rmse(s.y, pb2),
                "B_MAE": metrics.mae(s.y, pb), "B2_MAE": metrics.mae(s.y, pb2),
                "blend_AB_RMSE": metrics.rmse(s.y, bl),
                "blend_AB2_RMSE": metrics.rmse(s.y, bl2),
                "blend_AB_MAE": metrics.mae(s.y, bl),
                "blend_AB2_MAE": metrics.mae(s.y, bl2),
                "rho_A_B": float(np.corrcoef(pa - s.y, pb - s.y)[0, 1]),
                "rho_A_B2": float(np.corrcoef(pa - s.y, pb2 - s.y)[0, 1]),
                "blend_AB2_highvol": metrics.rmse(s.y[s.high], bl2[s.high]),
                "blend_AB_highvol": metrics.rmse(s.y[s.high], bl[s.high]),
            }
            r["dRMSE_member"] = r["B2_RMSE"] - r["B_RMSE"]
            r["dRMSE_blend"] = r["blend_AB2_RMSE"] - r["blend_AB_RMSE"]
            r["dMAE_blend"] = r["blend_AB2_MAE"] - r["blend_AB_MAE"]
            rows.append(r)
            log(f"  seed {seed:<4} B {r['B_RMSE']:.4f} -> B' {r['B2_RMSE']:.4f} "
                f"({r['dRMSE_member']:+.4f})   blend {r['blend_AB_RMSE']:.4f} -> "
                f"{r['blend_AB2_RMSE']:.4f} ({r['dRMSE_blend']:+.4f})")
            del pb, pb2, bl, bl2
            gc.collect()
        del s, pa
        gc.collect()

    seed42 = [
        {"window": "autumn_2015", "seed": 42, "dRMSE_member": -0.0284,
         "dRMSE_blend": -0.0105, "source": "exp_77"},
        {"window": "christmas_2015", "seed": 42, "dRMSE_member": -0.0108,
         "dRMSE_blend": -0.0053, "source": "exp_77"},
    ]
    all_cells = [{"window": r["window"], "seed": r["seed"],
                  "dRMSE_member": r["dRMSE_member"],
                  "dRMSE_blend": r["dRMSE_blend"], "source": "exp_79"}
                 for r in rows] + seed42

    banner("ALL SIX (window, seed) CELLS")
    log(f"  {'window':<18}{'seed':>6}{'dRMSE member':>15}{'dRMSE blend':>14}{'source':>10}")
    for c in sorted(all_cells, key=lambda x: (x["window"], x["seed"])):
        log(f"  {c['window']:<18}{c['seed']:>6}{c['dRMSE_member']:>+15.4f}"
            f"{c['dRMSE_blend']:>+14.4f}{c['source']:>10}")

    member_wins = sum(1 for c in all_cells if c["dRMSE_member"] < 0)
    blend_wins = sum(1 for c in all_cells if c["dRMSE_blend"] < 0)
    mean_member = float(np.mean([c["dRMSE_member"] for c in all_cells]))
    mean_blend = float(np.mean([c["dRMSE_blend"] for c in all_cells]))
    log(f"\n  member B' wins {member_wins}/6   mean dRMSE {mean_member:+.4f}")
    log(f"  blend A+B' wins {blend_wins}/6   mean dRMSE {mean_blend:+.4f}")

    banner("DECISION")
    crit = {"E1_member_wins_at_least_5_of_6": member_wins >= 5,
            "E2_blend_wins_at_least_5_of_6": blend_wins >= 5}
    for k, v in crit.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    confirmed = all(crit.values())
    log(f"\n  -> {'UPGRADE CONFIRMED SEED-STABLE' if confirmed else 'UPGRADE NOT SEED-STABLE — roll back to #76 member B'}")

    R = pd.DataFrame(rows)
    exp = Experiment("exp_79_upgrade_seed_check",
                     model_type="seed-stability check of exp_77",
                     objective="tweedie (variance_power=1.1)",
                     feature_set_label="member B (26f) vs B' (32f), seeds 7 and 202",
                     n_features=len(REC_COLS_V5),
                     validation_origin_day="autumn_2015 + christmas_2015",
                     horizon=config.HORIZON, n_series=config.N_SERIES)
    exp.note("Experiment #79. #77 was accepted on four windows at one seed and "
             "it changed the shipped forecast. This closes that gap on the two "
             "windows that carry the effect; the other two produced gains of "
             "-0.0005 and -0.0004, which are noise and cannot be confirmed or "
             "refuted by reseeding.")
    exp.note("Member A is held at seed 42 so that only the recursive members' "
             "seeds vary, isolating the question being asked.")
    exp.set_metrics(RMSE=float(R.blend_AB2_RMSE.mean()),
                    MAE=float(R.blend_AB2_MAE.mean()))
    exp.set(blend_weight=W, cells=all_cells, new_runs=rows,
            member_wins=member_wins, blend_wins=blend_wins,
            mean_member_dRMSE=mean_member, mean_blend_dRMSE=mean_blend,
            acceptance_criteria=crit, confirmed=confirmed,
            decision=("UPGRADE CONFIRMED SEED-STABLE" if confirmed
                      else "UPGRADE NOT SEED-STABLE"))
    exp.save()
    R.to_csv(config.ARTIFACTS_DIR / "exp79_seed_cells.csv", index=False)
    (config.ARTIFACTS_DIR / "exp79_summary.json").write_text(
        json.dumps({"cells": all_cells, "criteria": crit, "confirmed": confirmed,
                    "mean_member_dRMSE": mean_member,
                    "mean_blend_dRMSE": mean_blend}, indent=2, default=str),
        encoding="utf-8")
    log(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
