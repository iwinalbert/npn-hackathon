"""
PHASE 4 — Tweedie variance-power search (chosen on the INNER window)
PHASE 6 — objective comparison (L2 / Tweedie / Poisson / Gamma-style)

Discipline: the power is chosen using ONLY the inner window (d_1886..d_1913) and
then evaluated ONCE on the primary window. The objective comparison runs directly
on the primary window because it is a comparison, not a selection — no objective
is being picked by peeking.

    python scripts/17_phase4_6_objective.py
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

from pipeline import config, optimize
from pipeline.features_v2 import V2_SETS

POWERS = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
INNER_ORIGIN = config.VALIDATION_ORIGIN_IDX - config.HORIZON


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    cols = V2_SETS["v2_base"]           # hold features fixed at our current best

    # ==================================================================
    banner("PHASE 4 — TWEEDIE POWER SEARCH (inner window only)")
    # ==================================================================
    inner = optimize.Setup(origin_idx=INNER_ORIGIN)
    print(f"  inner window  : {inner.window['validation_days']} "
          f"({inner.window['validation_dates']})")
    print(f"  primary window: d_1914 .. d_1941  <- untouched in this phase\n")
    assert int(inner.valid["target_day_idx"].max()) < config.VALIDATION_ORIGIN_IDX + 1

    rows = []
    for p in POWERS:
        d = optimize.run(
            f"opt_04_power_{str(p).replace('.', '_')}", inner, cols,
            params={"objective": "tweedie", "tweedie_variance_power": p},
            label=f"Tweedie power {p} (INNER window)",
            notes=("Phase 4. Power selection on the inner window only; the primary "
                   "window plays no part in this choice.",),
            save_model=False, save_preds=False,
            extra={"tuning_window": "INNER", "tweedie_power": p})
        rows.append({"power": p, "inner_RMSE": d["RMSE"], "inner_MAE": d["MAE"],
                     "high_vol_RMSE": d["high_volume_RMSE"],
                     "high_vol_bias": d["high_volume_bias"],
                     "mean_pred_on_zero": d["mean_pred_on_zero_actual"],
                     "train_s": d["_train_s"]})

    pw = pd.DataFrame(rows).sort_values("inner_RMSE")
    pw.to_csv(config.ARTIFACTS_DIR / "phase4_power_search.csv", index=False)
    print("\n" + pw.to_string(index=False))
    best_power = float(pw.iloc[0]["power"])
    print(f"\n  selected on inner window: power {best_power} "
          f"(inner RMSE {pw.iloc[0]['inner_RMSE']:.4f})")
    del inner

    # ==================================================================
    banner("PHASE 4b — APPLY THE SELECTED POWER ONCE TO THE PRIMARY WINDOW")
    # ==================================================================
    s = optimize.Setup()
    applied = optimize.run(
        f"opt_04b_power_{str(best_power).replace('.', '_')}_primary", s, cols,
        params={"objective": "tweedie", "tweedie_variance_power": best_power},
        label=f"Tweedie power {best_power} (selected on inner, applied once)",
        notes=(f"Phase 4b. Power {best_power} was chosen using only "
               "d_1886..d_1913. This is its single unbiased evaluation on the "
               "primary window.",),
        extra={"tweedie_power": best_power,
               "selection_basis": "inner window d_1886..d_1913"})

    # ==================================================================
    banner("PHASE 6 — OBJECTIVE COMPARISON (same features, same window)")
    # ==================================================================
    print("  Gamma is deliberately excluded: it requires a strictly positive")
    print("  target, and 54% of our validation rows are exactly zero. Fitting it")
    print("  would require dropping or shifting zeros, which changes the problem.\n")

    objectives = [
        ("l2", {"objective": "regression"}, "L2 / squared error"),
        ("tweedie_1_1", {"objective": "tweedie", "tweedie_variance_power": 1.1},
         "Tweedie (power 1.1)"),
        ("poisson", {"objective": "poisson"}, "Poisson (count data)"),
        ("l1", {"objective": "regression_l1"}, "L1 / absolute error"),
    ]
    orows = []
    for tag, params, label in objectives:
        d = optimize.run(
            f"opt_06_obj_{tag}", s, cols, params=params,
            label=f"Objective: {label}",
            notes=("Phase 6. Only the objective changes; features, "
                   "hyperparameters, origins and validation window are fixed.",),
            save_model=False,
            extra={"objective_tag": tag})
        orows.append({"objective": label, "RMSE": d["RMSE"], "MAE": d["MAE"],
                      "dRMSE": d["RMSE"] - optimize.BEST_RMSE,
                      "dMAE": d["MAE"] - optimize.BEST_MAE,
                      "high_vol_RMSE": d["high_volume_RMSE"],
                      "mean_pred_on_zero": d["mean_pred_on_zero_actual"],
                      "train_s": d["_train_s"]})

    od = pd.DataFrame(orows).sort_values("RMSE")
    od.to_csv(config.ARTIFACTS_DIR / "phase6_objectives.csv", index=False)
    print("\n" + od.to_string(index=False))

    summary = {
        "power_search": rows,
        "selected_power": best_power,
        "applied_primary": {"RMSE": applied["RMSE"], "MAE": applied["MAE"],
                            "high_volume_RMSE": applied["high_volume_RMSE"],
                            "high_volume_bias": applied["high_volume_bias"]},
        "objectives": orows,
    }
    (config.ARTIFACTS_DIR / "phase4_6_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
