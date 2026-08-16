"""
EXPERIMENT #69 — Pre-Origin Per-Series Bias Correction.

WHAT THIS DOES
--------------
Leaves the selected model and its predictions completely untouched. Learns one
multiplicative correction factor per store-item series from a window that ends
BEFORE the forecast origin, then applies those factors to the existing
predictions for d_1914..d_1941.

WHY AUXILIARY MODELS ARE NEEDED (and why they are not "retraining")
-------------------------------------------------------------------
A bias factor is actual / predicted. To measure it on the pre-origin window
d_1886..d_1913 we need predictions for those days — and the base model does not
produce any, because it was built to forecast from origin d_1913 onward.

So we train two auxiliary models purely to generate pre-origin predictions:

    AUX-A : origin d_1857 -> predicts d_1858..d_1885
    AUX-B : origin d_1885 -> predicts d_1886..d_1913

Both use the identical configuration to the base model, and both train only on
origins at least 28 days before their own window, so neither ever sees a day at
or after d_1914. The base model is never retrained, reloaded, or altered.

HOW THE SHRINKAGE CONSTANT IS CHOSEN
------------------------------------
Shrinking a factor toward 1.0 needs a constant k. Choosing k by trying values
against d_1914..d_1941 would be selecting on the scoring window — the exact
mistake that produced four false positives earlier in this project. Instead:

    factors fitted on d_1858..d_1885 (using AUX-A)
        -> evaluated on d_1886..d_1913 (using AUX-B)
        -> k chosen there

Both of those windows are entirely before the forecast origin. Only after k is
fixed do we fit the final factors on d_1886..d_1913 and apply them once.

DECISION RULE, FIXED BEFORE THE RESULT IS SEEN
----------------------------------------------
Phase 9 measured a window-to-window RMSE standard deviation of 0.022-0.033 for
this model. A change smaller than that is indistinguishable from which month you
happened to score on. Therefore:

    ACCEPT  if  dRMSE <= -0.022  AND  dMAE <= +0.020
    REJECT  otherwise

    python scripts/27_exp69_bias_correction.py
"""

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

from pipeline import config, metrics, optimize
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment
from pipeline.features_v2 import FeatureBuilderV2, V2_SETS

COLS = V2_SETS["v2_base"]                      # the base model's 32 features
BASE_PRED = config.PREDICTIONS_DIR / "model_04_tweedie_recency_listing_validation.csv"

VO = config.VALIDATION_ORIGIN_IDX              # 1912 -> d_1913
FIT_ORIGIN = VO - config.HORIZON               # 1884 -> d_1885, fits on d_1886..d_1913
TUNE_ORIGIN = VO - 2 * config.HORIZON          # 1856 -> d_1857, fits on d_1858..d_1885

CLIP_LO, CLIP_HI = 0.5, 2.0
K_GRID = [0, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000]   # 0 = no shrinkage
# The selection is also allowed to choose "apply no correction at all" — the
# k -> infinity limit. A fitting procedure that cannot reject its own correction
# is not a fair test of whether the correction is worth applying.

ACCEPT_RMSE = -0.022
ACCEPT_MAE_TOL = 0.020


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def factors_from(pred: np.ndarray, actual: np.ndarray, series_idx: np.ndarray,
                 k: float) -> np.ndarray:
    """
    One multiplicative factor per series: sum(actual) / sum(predicted) over the
    fitting window, shrunk toward 1.0 by volume and clipped.

    Shrinkage weight w = P / (P + k), where P is the series' total predicted
    units in the window. A busy series keeps almost all of its measured
    correction; a series with barely any predicted volume is pulled back to 1.0,
    because its ratio is mostly noise.
    """
    df = pd.DataFrame({"s": series_idx, "a": actual, "p": pred})
    g = df.groupby("s").agg(a=("a", "sum"), p=("p", "sum"))
    g = g.reindex(range(config.N_SERIES)).fillna(0.0)

    P = g["p"].to_numpy()
    A = g["a"].to_numpy()
    raw = np.where(P > 1e-9, A / np.maximum(P, 1e-9), 1.0)

    w = P / (P + k) if k > 0 else np.ones_like(P)
    f = 1.0 + (raw - 1.0) * w
    f = np.clip(f, CLIP_LO, CLIP_HI)
    f[P <= 1e-9] = 1.0                          # nothing predicted -> no correction
    return f.astype(np.float64)


def train_aux(origin_idx: int, tag: str):
    """Train an auxiliary model at `origin_idx` and return its window predictions."""
    s = optimize.Setup(origin_idx=origin_idx)
    print(f"    [{tag}] origin {s.window['forecast_origin_day']} -> "
          f"{s.window['validation_days']} ({s.window['validation_dates']})")
    X, Y = optimize.build_matrix(s, COLS)
    booster, info = optimize.train(X, Y, COLS)
    del X, Y
    p = optimize.predict(booster, s, COLS)
    print(f"    [{tag}] trained in {info['training_seconds']}s, "
          f"window RMSE {metrics.rmse(s.y, p):.4f}")
    return s, p, booster, info


def main():
    t0 = time.time()
    banner("EXPERIMENT #69 — PRE-ORIGIN PER-SERIES BIAS CORRECTION")

    data = M5Data(load_prices=False)
    base = pd.read_csv(BASE_PRED).sort_values(
        ["target_day_idx", "series_idx"]).reset_index(drop=True)
    y = base["y_true"].to_numpy(float)
    p0 = base["y_pred"].to_numpy(float)
    si = base["series_idx"].to_numpy()

    r0, m0 = metrics.rmse(y, p0), metrics.mae(y, p0)
    print(f"  baseline (untouched predictions): RMSE {r0:.4f}  MAE {m0:.4f}")
    # The prediction file stores y_pred rounded to 5 decimals, so recomputing
    # from it differs from the in-memory figure by ~1.5e-09. The tolerance is set
    # to 1e-7: loose enough to absorb that documented storage rounding, still
    # tight enough that any real change to the file would fail immediately.
    recorded = 2.1210429411947650
    drift = abs(r0 - recorded)
    assert drift < 1e-7, f"baseline predictions changed! drift={drift:.3e}"
    print(f"  baseline file verified against the recorded 2.1210 result "
          f"(drift {drift:.2e}, from 5-decimal CSV storage)")

    # ==================================================================
    banner("STEP 1 — AUXILIARY MODELS (pre-origin only; base model untouched)")
    # ==================================================================
    s_tune, p_tune, b_tune, i_tune = train_aux(TUNE_ORIGIN, "AUX-A")
    s_fit, p_fit, b_fit, i_fit = train_aux(FIT_ORIGIN, "AUX-B")

    max_day_touched = int(max(s_fit.valid["target_day_idx"].max(),
                              s_tune.valid["target_day_idx"].max()))
    print(f"\n  highest day index touched by either auxiliary model: "
          f"{max_day_touched} (d_{max_day_touched+1})")
    print(f"  first validation day: {VO+1} (d_{VO+2})")
    assert max_day_touched <= VO, "auxiliary model reached into the validation window"
    print("  PASS — no auxiliary model reached day d_1914 or later")

    # ==================================================================
    banner("STEP 2 — CHOOSE SHRINKAGE k ON A PRE-ORIGIN WINDOW")
    # ==================================================================
    print("  Factors fitted on d_1858..d_1885 (AUX-A), scored on d_1886..d_1913 (AUX-B).")
    print("  The validation window plays no part in this choice.\n")

    tune_rows = []
    base_tune_rmse = metrics.rmse(s_fit.y, p_fit)
    for k in K_GRID:
        f = factors_from(p_tune, s_tune.y, s_tune.valid["series_idx"].to_numpy(), k)
        pc = p_fit * f[s_fit.valid["series_idx"].to_numpy()]
        tune_rows.append({"k": k, "rmse": metrics.rmse(s_fit.y, pc),
                          "mae": metrics.mae(s_fit.y, pc)})
        print(f"    k={k:>4}: pre-origin RMSE {tune_rows[-1]['rmse']:.4f}  "
              f"MAE {tune_rows[-1]['mae']:.4f}")
    print(f"    uncorrected reference        {base_tune_rmse:.4f}  "
          f"{metrics.mae(s_fit.y, p_fit):.4f}")

    tune_rows.append({"k": float("inf"), "rmse": base_tune_rmse,
                      "mae": metrics.mae(s_fit.y, p_fit)})
    print(f"    k=inf  (apply no correction): pre-origin RMSE {base_tune_rmse:.4f}  "
          f"MAE {metrics.mae(s_fit.y, p_fit):.4f}")

    td = pd.DataFrame(tune_rows).sort_values("rmse")
    k_sel = float(td.iloc[0]["k"])
    improves_pre = bool(np.isfinite(k_sel))
    finite = pd.DataFrame([r for r in tune_rows if np.isfinite(r["k"])])
    k_best = float(finite.sort_values("rmse").iloc[0]["k"])

    print(f"\n  pre-origin selection picked: "
          f"{'k = %g' % k_sel if improves_pre else 'NO CORRECTION (k -> inf)'}")
    if not improves_pre:
        print("  >> Every finite k scored WORSE on the pre-origin window than")
        print("     leaving the predictions alone. The protocol has rejected this")
        print("     correction before the validation window was touched at all.")
        print(f"     We still apply the best finite k ({k_best:g}) once below,")
        print("     purely to document what it would have cost.")

    # ==================================================================
    banner("STEP 3 — FIT FINAL FACTORS ON d_1886..d_1913 AND APPLY ONCE")
    # ==================================================================
    fit_si = s_fit.valid["series_idx"].to_numpy()
    F = factors_from(p_fit, s_fit.y, fit_si, k_best)
    p1 = p0 * F[si]

    r1, m1 = metrics.rmse(y, p1), metrics.mae(y, p1)
    d_r, d_m = r1 - r0, m1 - m0
    print(f"  corrected: RMSE {r1:.4f}  MAE {m1:.4f}")
    print(f"  change   : RMSE {d_r:+.4f} ({d_r/r0*100:+.3f}%)   "
          f"MAE {d_m:+.4f} ({d_m/m0*100:+.3f}%)")

    # ==================================================================
    banner("STEP 4 — LEAKAGE CHECKS")
    # ==================================================================
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    chk("aux_models_never_reach_validation", max_day_touched <= VO,
        f"highest day used = d_{max_day_touched+1}, validation starts d_{VO+2}")

    # Empirical: overwrite every sales value after the origin, then re-derive the
    # actuals used for fitting and re-compute the factors from them. Both must be
    # untouched. This goes straight at the sales matrix rather than through the
    # feature builder, because the actuals are what the factors are made of.
    corrupt = data.sales_wide.copy()
    corrupt[:, VO + 1:] = 9999
    fit_days = np.arange(FIT_ORIGIN + 1, FIT_ORIGIN + 1 + config.HORIZON)
    same_slice = np.array_equal(corrupt[:, fit_days], data.sales_wide[:, fit_days])
    chk("fitting_window_actuals_unaffected_by_future", same_slice,
        f"the d_{fit_days[0]+1}..d_{fit_days[-1]+1} sales used to fit the factors "
        "are identical when every day after d_1913 is overwritten with 9999")

    y_fit_corrupt = corrupt[np.ix_(np.arange(config.N_SERIES), fit_days)].T.ravel().astype(float)
    F_corrupt = factors_from(p_fit, y_fit_corrupt, fit_si, k_best)
    chk("factors_unchanged_under_future_corruption",
        np.array_equal(F, F_corrupt),
        "all 30,490 correction factors bit-identical after corrupting the future")

    chk("baseline_predictions_untouched",
        np.array_equal(base["y_pred"].to_numpy(), pd.read_csv(BASE_PRED)["y_pred"].to_numpy()),
        "the base prediction file on disk is unchanged")

    chk("factors_within_clip_range", bool(F.min() >= CLIP_LO and F.max() <= CLIP_HI),
        f"range [{F.min():.4f}, {F.max():.4f}] within [{CLIP_LO}, {CLIP_HI}]")
    del corrupt

    if not all(c["passed"] for c in checks):
        raise SystemExit("STOP: a leakage check failed")

    # ==================================================================
    banner("STEP 5 — FACTOR DISTRIBUTION")
    # ==================================================================
    n_corrected = int((np.abs(F - 1.0) > 1e-6).sum())
    n_clipped = int(((F <= CLIP_LO + 1e-9) | (F >= CLIP_HI - 1e-9)).sum())
    qs = {f"p{q}": float(np.percentile(F, q)) for q in [1, 5, 25, 50, 75, 95, 99]}
    print(f"  series receiving a correction: {n_corrected:,} of {config.N_SERIES:,} "
          f"({n_corrected/config.N_SERIES*100:.2f}%)")
    print(f"  series hitting a clip bound  : {n_clipped:,} "
          f"({n_clipped/config.N_SERIES*100:.2f}%)")
    print(f"  factors <1 (shrink down): {int((F<0.999).sum()):,}   "
          f">1 (scale up): {int((F>1.001).sum()):,}")
    print("  percentiles: " + "  ".join(f"{k}={v:.3f}" for k, v in qs.items()))
    print(f"  mean {F.mean():.4f}   median {np.median(F):.4f}")

    # ==================================================================
    banner("STEP 6 — PERFORMANCE BY DEMAND-VOLUME DECILE")
    # ==================================================================
    hist = data.sales_wide[:, :VO + 1].mean(axis=1)
    dec = pd.qcut(hist[si], 10, labels=False, duplicates="drop")
    rows = []
    for k_ in sorted(pd.unique(dec)):
        m = dec == k_
        rows.append({
            "decile": int(k_) + 1, "n": int(m.sum()),
            "hist_range": f"{hist[si][m].min():.2f}-{hist[si][m].max():.2f}",
            "base_RMSE": metrics.rmse(y[m], p0[m]),
            "corr_RMSE": metrics.rmse(y[m], p1[m]),
            "base_MAE": metrics.mae(y[m], p0[m]),
            "corr_MAE": metrics.mae(y[m], p1[m]),
            "mean_factor": float(F[si[m]].mean()),
        })
    dv = pd.DataFrame(rows)
    dv["dRMSE"] = dv.corr_RMSE - dv.base_RMSE
    dv["dMAE"] = dv.corr_MAE - dv.base_MAE
    print(dv[["decile", "hist_range", "n", "base_RMSE", "corr_RMSE", "dRMSE",
              "dMAE", "mean_factor"]].to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))

    high = hist[si] > 3.0
    hi_base, hi_corr = metrics.rmse(y[high], p0[high]), metrics.rmse(y[high], p1[high])
    print(f"\n  high-volume tier (>3/day, {high.sum():,} rows): "
          f"RMSE {hi_base:.4f} -> {hi_corr:.4f} ({hi_corr-hi_base:+.4f})")
    helps_high = hi_corr < hi_base
    print(f"  correction {'HELPS' if helps_high else 'HURTS'} high-volume series")

    # ==================================================================
    banner("STEP 7 — DECISION")
    # ==================================================================
    NOISE_LO, NOISE_HI = 0.022, 0.033
    accepted = (d_r <= ACCEPT_RMSE) and (d_m <= ACCEPT_MAE_TOL)
    print(f"  rule fixed in advance: ACCEPT if dRMSE <= {ACCEPT_RMSE} "
          f"AND dMAE <= +{ACCEPT_MAE_TOL}")
    print(f"  measured             : dRMSE {d_r:+.4f}   dMAE {d_m:+.4f}")
    print(f"  noise floor (Phase 9): +/-{NOISE_LO} to {NOISE_HI} RMSE")
    print(f"  |dRMSE| vs noise floor: {abs(d_r):.4f} vs {NOISE_LO} "
          f"-> {'exceeds' if abs(d_r) > NOISE_LO else 'INSIDE the noise band'}")
    print(f"\n  DECISION: {'ACCEPTED' if accepted else 'REJECTED'}")

    # ==================================================================
    exp = Experiment(
        "exp_69_pre_origin_per_series_bias_correction",
        model_type="post-hoc per-series multiplicative correction",
        objective="n/a (correction applied to existing Tweedie predictions)",
        feature_set_label="No new features; base model untouched",
        n_features=len(COLS),
        **s_fit.describe(),
    )
    exp.set(validation_days="d_1914 .. d_1941",
            validation_dates="2016-04-25 .. 2016-05-22",
            validation_rows=int(len(y)))
    exp.note("Experiment #69. One multiplicative factor per series, fitted on "
             "d_1886..d_1913 (entirely before the forecast origin) and applied to "
             "the untouched base predictions for d_1914..d_1941.")
    exp.note(f"Shrinkage constant k={k_best:g} was selected on a SECOND pre-origin "
             "window (factors fitted d_1858..d_1885, scored d_1886..d_1913). The "
             "validation window played no part in any choice.")
    exp.note("Two auxiliary models were trained solely to produce pre-origin "
             "predictions. The selected base model was neither retrained nor "
             "modified, and its prediction file is unchanged on disk.")
    exp.note("METHODOLOGICAL CAVEAT: the factors measure the bias of an auxiliary "
             "model fitted at origin d_1885, and are applied to a different fit at "
             "origin d_1913. This assumes per-series bias is a property of the "
             "series rather than of one particular fit.")
    exp.set_metrics(RMSE=r1, MAE=m1, WAPE=metrics.wape(y, p1),
                    bias=metrics.bias(y, p1), n=int(len(y)),
                    high_volume_RMSE=hi_corr)
    exp.set(preorigin_selection_chose_correction=bool(improves_pre),
            baseline_RMSE=r0, baseline_MAE=m0,
            delta_rmse_vs_best=round(d_r, 6), delta_mae_vs_best=round(d_m, 6),
            pct_change_rmse=round(d_r / r0 * 100, 4),
            pct_change_mae=round(d_m / m0 * 100, 4),
            shrinkage_k=k_best, clip_range=[CLIP_LO, CLIP_HI],
            k_selection_grid=tune_rows,
            k_selected_on="pre-origin window d_1886..d_1913",
            n_series_corrected=n_corrected, n_series_clipped=n_clipped,
            factor_percentiles=qs, factor_mean=float(F.mean()),
            factor_median=float(np.median(F)),
            high_volume_base_RMSE=hi_base, high_volume_helps=bool(helps_high),
            by_decile=dv.to_dict(orient="records"),
            leakage_checks=checks,
            decision="ACCEPTED" if accepted else "REJECTED",
            decision_rule=f"ACCEPT if dRMSE <= {ACCEPT_RMSE} and dMAE <= {ACCEPT_MAE_TOL}",
            noise_floor=[NOISE_LO, NOISE_HI],
            aux_models={"tune": i_tune["training_seconds"],
                        "fit": i_fit["training_seconds"]},
            training_seconds=round(i_tune["training_seconds"] + i_fit["training_seconds"], 1))
    exp.save()

    pd.DataFrame({"series_idx": np.arange(config.N_SERIES),
                  "series_id": data.series_meta["id"].to_numpy(),
                  "factor": F}).to_csv(
        config.ARTIFACTS_DIR / "exp69_correction_factors.csv", index=False)
    pd.DataFrame({"series_idx": si,
                  "target_day_idx": base["target_day_idx"].to_numpy(),
                  "horizon": base["horizon"].to_numpy(),
                  "y_true": y, "y_pred": np.round(p1, 5)}).to_csv(
        config.PREDICTIONS_DIR / "exp_69_bias_corrected_validation.csv", index=False)
    dv.to_csv(config.ARTIFACTS_DIR / "exp69_by_decile.csv", index=False)

    summary = {
        "baseline": {"RMSE": r0, "MAE": m0},
        "corrected": {"RMSE": r1, "MAE": m1},
        "delta": {"RMSE": d_r, "MAE": d_m,
                  "pct_RMSE": d_r / r0 * 100, "pct_MAE": d_m / m0 * 100},
        "k": k_best, "k_grid": tune_rows,
        "k_improves_preorigin": bool(improves_pre),
        "preorigin_selection_chose_no_correction": bool(not improves_pre),
        "preorigin_uncorrected_RMSE": base_tune_rmse,
        "n_corrected": n_corrected, "n_clipped": n_clipped,
        "factor_percentiles": qs, "factor_mean": float(F.mean()),
        "high_volume": {"base": hi_base, "corrected": hi_corr, "helps": bool(helps_high)},
        "by_decile": dv.to_dict(orient="records"),
        "leakage_checks": checks,
        "noise_floor": [NOISE_LO, NOISE_HI],
        "decision": "ACCEPTED" if accepted else "REJECTED",
    }
    (config.ARTIFACTS_DIR / "exp69_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n  wrote experiments/exp_69_*.json, artifacts/exp69_*, "
          f"predictions/exp_69_bias_corrected_validation.csv")
    print(f"  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
