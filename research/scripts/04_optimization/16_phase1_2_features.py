
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, optimize
from pipeline.features_v2 import FeatureBuilderV2, V2_SETS, V2_LABELS


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    banner("PHASE 1 — OPTIMIZATION BASELINE")
    s = optimize.Setup()
    print(f"  window   : {s.window['validation_days']} ({s.window['validation_dates']})")
    print(f"  series   : {config.N_SERIES:,}   rows: {len(s.valid):,}")
    print(f"  origins  : {len(s.origins)} (d_{s.origins[0]+1} .. d_{s.origins[-1]+1})")
    print(f"  reference: RMSE {optimize.BEST_RMSE:.4f}  MAE {optimize.BEST_MAE:.4f}\n")

    print("  Leakage test on the extended feature builder...")
    cols_all = V2_SETS["v2_all"]
    clean = s.fb.build_origin_frame(s.origin_idx, include_target=False)
    corrupt = s.data.sales_wide.copy()
    corrupt[:, s.origin_idx + 1:] = 9999
    d2 = copy.copy(s.data); d2.sales_wide = corrupt
    dirty = FeatureBuilderV2(d2).build_origin_frame(s.origin_idx, include_target=False)
    changed = [c for c in cols_all
               if not np.array_equal(clean[c].to_numpy(), dirty[c].to_numpy(),
                                     equal_nan=np.issubdtype(clean[c].dtype, np.floating))]
    print(f"  {'PASS' if not changed else 'FAIL'} — {len(cols_all)} features, "
          f"{len(changed)} changed under future-sales corruption")
    if changed:
        raise SystemExit(f"STOP: leakage in {changed}")
    del dirty, corrupt, d2, clean

    results = []
    base = optimize.run(
        "opt_00_baseline_reproduce", s, V2_SETS["v2_base"],
        label="Current best 32 features (reproducibility check)",
        notes=("Phase 1. Identical configuration to model_04. If this does not "
               "reproduce RMSE 2.1210 exactly, the pipeline is not deterministic "
               "and every later comparison is unreliable.",),
        save_model=False)
    results.append(base)

    drift = abs(base["RMSE"] - optimize.BEST_RMSE)
    print(f"\n  reproducibility drift: {drift:.2e} "
          f"({'EXACT' if drift < 1e-9 else 'NOT EXACT — investigate'})")
    if drift > 1e-6:
        raise SystemExit("STOP: baseline did not reproduce; comparisons would be invalid")

    print(f"\n  zero-demand behaviour : mean prediction on true-zero rows = "
          f"{base['mean_pred_on_zero_actual']}")
    print(f"  positive rows         : predicted {base['mean_pred_on_positive_actual']} "
          f"vs actual mean {s.y[~s.is_zero].mean():.4f}")
    print(f"  high-volume tier      : RMSE {base['high_volume_RMSE']:.4f}, "
          f"bias {base['high_volume_bias']:+.4f}, "
          f"{base['high_volume_share_of_sq_error_pct']}% of squared error")
    print(f"  prediction spread     : p50 {base['pred_p50']}, p99 {base['pred_p99']}, "
          f"max {base['pred_max']}")

    banner("PHASE 2 — TARGETED FEATURE EXPERIMENTS (one group at a time)")
    for key in ["v2_A_demand", "v2_B_calendar", "v2_C_price",
                "v2_D_interactions", "v2_all"]:
        results.append(optimize.run(
            f"opt_02_{key}", s, V2_SETS[key], label=V2_LABELS[key],
            notes=("Phase 2. Base 32 features held fixed; only this group added. "
                   "Objective, hyperparameters, origins and validation window are "
                   "identical to the baseline.",),
            save_model=False))

    banner("PHASE 2 RESULTS")
    df = pd.DataFrame([{
        "experiment": r["_label"], "n_feat": r["_n_features"],
        "RMSE": r["RMSE"], "MAE": r["MAE"],
        "dRMSE": r["RMSE"] - optimize.BEST_RMSE,
        "dMAE": r["MAE"] - optimize.BEST_MAE,
        "high_vol_RMSE": r["high_volume_RMSE"],
        "train_s": r["_train_s"],
    } for r in results])
    print(df.to_string(index=False))
    df.to_csv(config.ARTIFACTS_DIR / "phase2_feature_results.csv", index=False)

    best = df.iloc[1:].sort_values("RMSE").iloc[0]
    print(f"\n  best feature group: {best['experiment']}  "
          f"(RMSE {best['RMSE']:.4f}, {best['dRMSE']:+.4f})")
    helped = df.iloc[1:][df.iloc[1:]["dRMSE"] < 0]
    print(f"  groups that improved RMSE at all: "
          f"{len(helped)} of {len(df)-1}")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
