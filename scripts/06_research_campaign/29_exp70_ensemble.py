"""
EXPERIMENT #70 — Variance-reduction ensemble of diverse strong RMSE models.

HYPOTHESIS
----------
The error autopsy decomposed MSE into 0.0049 bias-squared and 4.4939 variance:
99.89% of the error is variance. Every intervention tried so far has attacked
either bias (calibration, per-series correction, volume weighting) or information
(new features). None of those is the binding constraint, which is why all of them
failed.

Averaging several models that are individually strong but make DIFFERENT mistakes
cancels the part of the variance that comes from the fitting procedure itself
(bootstrap sampling, feature subsampling, split-point ties, seed). That is the
one canonical variance-reduction technique this project has never tested on
RMSE-competitive models.

WHY EXISTING EVIDENCE SUPPORTS IT
---------------------------------
- The variance decomposition above.
- Phase 8 blended Tweedie with L1 and RMSE got worse — but L1 is deliberately a
  poor RMSE model (it targets the median of a mostly-zero distribution). That
  experiment measured a metric trade-off, not variance reduction. It does not
  rule this out; it is a materially different hypothesis.
- Phase 4 showed Tweedie powers 1.1-1.3 all score within 0.006 of each other on
  the inner window, i.e. several near-equal-quality models exist to average.

MECHANISM
---------
For M models with individually equal error, equal-weight averaging leaves the
shared (irreducible) error untouched but shrinks the independent component by
roughly 1/M. The gain is bounded by how decorrelated the members' mistakes are,
which is measured and reported below.

LEAKAGE RISK: none beyond the champion's. Same 32 features, same origins, same
frozen-at-origin construction. No weights are fitted, so no selection touches the
validation window either — equal weights are chosen a priori.

VALIDATION: primary window d_1914..d_1941, all 30,490 series x 28 days.
Robustness on three further windows is run only if the primary gain warrants it.

SUCCESS CRITERION (fixed before running)
----------------------------------------
    PROMOTE to robustness testing if dRMSE <= -0.010
    NEW CHAMPION only if it also holds up across the robustness windows
    REJECT if dRMSE > -0.010 or MAE degrades by more than +0.020

    python scripts/29_exp70_ensemble.py
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

from pipeline import config, metrics, optimize
from pipeline.experiment import Experiment
from pipeline.features_v2 import V2_SETS

COLS = V2_SETS["v2_base"]
CHAMP_RMSE, CHAMP_MAE = 2.1210429411947650, 1.0319268155496617

# Six members. All are strong RMSE models (Tweedie 1.1-1.3, the band Phase 4
# showed to be near-equal). Diversity comes from seed, tree capacity, and the
# subsampling fractions — the sources of fit-to-fit variance.
MEMBERS = [
    {"tag": "m1_champion",  "tweedie_variance_power": 1.1, "num_leaves": 128,
     "seed": 42,  "feature_fraction": 0.8, "bagging_fraction": 0.8},
    {"tag": "m2_wide",      "tweedie_variance_power": 1.1, "num_leaves": 256,
     "seed": 7,   "feature_fraction": 0.6, "bagging_fraction": 0.7},
    {"tag": "m3_p12",       "tweedie_variance_power": 1.2, "num_leaves": 160,
     "seed": 101, "feature_fraction": 0.9, "bagging_fraction": 0.8},
    {"tag": "m4_p13_small", "tweedie_variance_power": 1.3, "num_leaves": 96,
     "seed": 202, "feature_fraction": 0.8, "bagging_fraction": 0.9},
    {"tag": "m5_deep",      "tweedie_variance_power": 1.1, "num_leaves": 192,
     "seed": 303, "feature_fraction": 0.7, "bagging_fraction": 0.6},
    {"tag": "m6_p12_wide",  "tweedie_variance_power": 1.2, "num_leaves": 224,
     "seed": 404, "feature_fraction": 0.85, "bagging_fraction": 0.75},
]

PROMOTE_RMSE = -0.010
REJECT_MAE = 0.020


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    banner("EXPERIMENT #70 — VARIANCE-REDUCTION ENSEMBLE")
    print("  Hypothesis: 99.89% of our MSE is variance. Averaging diverse strong")
    print("  models cancels the fit-to-fit component. Equal weights, fixed a priori,")
    print("  so nothing is selected using the validation window.\n")

    s = optimize.Setup()
    y = s.y
    print(f"  window : {s.window['validation_days']} ({s.window['validation_dates']})")
    print(f"  champion: RMSE {CHAMP_RMSE:.4f}  MAE {CHAMP_MAE:.4f}\n")

    # Build the training matrix ONCE and reuse it for every member.
    print("  building shared training matrix...")
    X, Y = optimize.build_matrix(s, COLS, verbose=True)

    preds = {}
    rows = []
    for mb in MEMBERS:
        params = {k: v for k, v in mb.items() if k != "tag"}
        params["objective"] = "tweedie"
        booster, info = optimize.train(X, Y, COLS, params=params)
        p = optimize.predict(booster, s, COLS)
        preds[mb["tag"]] = p
        r, m = metrics.rmse(y, p), metrics.mae(y, p)
        rows.append({"member": mb["tag"], "power": mb["tweedie_variance_power"],
                     "leaves": mb["num_leaves"], "seed": mb["seed"],
                     "RMSE": r, "MAE": m, "train_s": info["training_seconds"]})
        print(f"    {mb['tag']:<14} RMSE={r:.4f}  MAE={m:.4f}  "
              f"({info['training_seconds']}s)")
        del booster
    del X, Y

    md = pd.DataFrame(rows)

    # ---- decorrelation: how different are the members' mistakes? ----
    banner("MEMBER DIVERSITY")
    E = np.column_stack([preds[t] - y for t in preds])
    C = np.corrcoef(E.T)
    off = C[np.triu_indices_from(C, k=1)]
    print(f"  pairwise residual correlation: mean {off.mean():.4f}, "
          f"min {off.min():.4f}, max {off.max():.4f}")
    print("  (1.000 would mean identical mistakes and zero ensemble benefit)")

    # ---- equal-weight ensemble ----
    banner("ENSEMBLE (equal weights, no fitting)")
    P = np.mean(np.column_stack([preds[t] for t in preds]), axis=1)
    d = optimize.diagnostics(y, P, s)
    dr, dm = d["RMSE"] - CHAMP_RMSE, d["MAE"] - CHAMP_MAE

    print(f"  best single member : RMSE {md.RMSE.min():.4f}")
    print(f"  mean of members    : RMSE {md.RMSE.mean():.4f}")
    print(f"  ENSEMBLE           : RMSE {d['RMSE']:.4f}  MAE {d['MAE']:.4f}")
    print(f"  vs champion        : dRMSE {dr:+.4f}   dMAE {dm:+.4f}")
    print(f"  ensemble beats every member individually: "
          f"{bool(d['RMSE'] < md.RMSE.min())}")
    print(f"  high-volume RMSE   : {d['high_volume_RMSE']:.4f} (champion 5.9756)")

    # ---- how many members are worth having? ----
    print("\n  ensemble RMSE as members are added (in listed order):")
    cum = []
    for k in range(1, len(preds) + 1):
        pk = np.mean(np.column_stack([preds[t] for t in list(preds)[:k]]), axis=1)
        cum.append({"n_members": k, "RMSE": metrics.rmse(y, pk),
                    "MAE": metrics.mae(y, pk)})
        print(f"    {k} member(s): RMSE {cum[-1]['RMSE']:.4f}")

    decision = "PROMOTE to robustness testing" if dr <= PROMOTE_RMSE else "REJECT"
    if dm > REJECT_MAE:
        decision = "REJECT (MAE degradation)"
    banner("DECISION")
    print(f"  criterion fixed in advance: promote if dRMSE <= {PROMOTE_RMSE}")
    print(f"  measured dRMSE {dr:+.4f}, dMAE {dm:+.4f}")
    print(f"  -> {decision}")

    exp = Experiment(
        "exp_70_variance_reduction_ensemble",
        model_type="equal-weight ensemble of 6 LightGBM Tweedie models",
        objective="tweedie (powers 1.1-1.3)",
        feature_set_label="Champion 32 features; diversity via seed/leaves/subsampling",
        n_features=len(COLS), **s.describe())
    exp.note("Experiment #70. Attacks the variance component of MSE (99.89% of it) "
             "rather than bias or information, which is what every prior failed "
             "experiment attacked.")
    exp.note("Equal weights fixed a priori — no weight fitting, so the validation "
             "window is used only to score, never to choose.")
    exp.note("Materially different from the Phase 8 ensemble, which blended in L1 "
             "(a deliberately poor RMSE model) and therefore measured a metric "
             "trade-off rather than variance reduction.")
    exp.set_metrics(**d)
    exp.set(members=rows, delta_rmse_vs_best=round(dr, 6),
            delta_mae_vs_best=round(dm, 6),
            mean_pairwise_residual_corr=float(off.mean()),
            best_single_member_RMSE=float(md.RMSE.min()),
            ensemble_beats_all_members=bool(d["RMSE"] < md.RMSE.min()),
            cumulative_by_n_members=cum,
            training_seconds=float(md.train_s.sum()),
            decision=decision,
            decision_rule=f"promote if dRMSE <= {PROMOTE_RMSE}")
    exp.save()

    pd.DataFrame({"series_idx": s.valid["series_idx"].to_numpy(),
                  "target_day_idx": s.valid["target_day_idx"].to_numpy(),
                  "horizon": s.valid["horizon"].to_numpy(),
                  "y_true": y, "y_pred": np.round(P, 5)}).to_csv(
        config.PREDICTIONS_DIR / "exp_70_ensemble_validation.csv", index=False)
    md.to_csv(config.ARTIFACTS_DIR / "exp70_members.csv", index=False)
    (config.ARTIFACTS_DIR / "exp70_summary.json").write_text(json.dumps({
        "champion": {"RMSE": CHAMP_RMSE, "MAE": CHAMP_MAE},
        "ensemble": d, "delta": {"RMSE": dr, "MAE": dm},
        "members": rows, "cumulative": cum,
        "mean_pairwise_residual_corr": float(off.mean()),
        "decision": decision}, indent=2, default=str), encoding="utf-8")

    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
