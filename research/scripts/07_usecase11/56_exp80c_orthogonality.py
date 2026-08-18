"""
EXPERIMENT #80c — IS THE ITEM-LEVEL GAIN ANYTHING MORE THAN A LEVEL CORRECTION?

INNER WINDOW ONLY. Read-only with respect to the champion.

WHY THIS EXISTS
---------------
Experiment #80b ran three negative controls and one of them fired. Rescaling
every prediction by a single fitted constant improved the inner window by
-0.0230, which is MORE than the -0.0160 that the whole item-level machinery
delivered. That control used the evaluation window's own truth to fit the
constant, so it is an oracle rather than an achievable method — but it exposes
something that has to be settled before anything is promoted:

    the champion over-forecasts this particular window by +0.0817 units per row
    (mean actual 1.3864, mean predicted 1.4681, 5.9% high),

whereas on the primary window it is essentially calibrated (bias -0.0081, and
the same global-rescale oracle is worth -0.0000 there). The inner window is
therefore ATYPICAL, and an alpha chosen on it may be measuring how well the
item-level model happens to undo a level anomaly rather than how much
cross-store information it carries.

THE TEST
--------
Split the item-level correction into the only two things it can be:

    ratio_i,d = Ahat_i,d / F_i,d           the multiplicative correction
    log ratio = (global mean)  +  (item-specific remainder)

Then compare, on identical rows:

  1. champion
  2. champion x best global constant                    (level only, oracle)
  3. champion reconciled, full correction                (level + item-specific)
  4. champion reconciled with the DEMEANED correction    (item-specific ONLY —
     the global component is removed, so this cannot act as a rescale)
  5. champion x best global constant, THEN reconciled with the demeaned
     correction                                          (both, non-overlapping)

If (4) is worth nothing, the direction is a calibration trick and must be
rejected as hierarchical forecasting. If (4) carries most of the gain, the
cross-store channel is real and the level anomaly is a separate matter.

    python scripts/07_usecase11/56_exp80c_orthogonality.py
"""

from __future__ import annotations

import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config, metrics
from pipeline.aggregate_level import AggregateLevel
from pipeline.champion_blend import champion_predictions
from pipeline.data_loader import M5Data

_probe = import_module("53_exp80_item_level_probe")

INNER_ORIGIN = config.VALIDATION_ORIGIN_IDX - config.HORIZON
ALPHAS = np.round(np.arange(0.0, 1.01, 0.05), 2)
OUT_JSON = config.ARTIFACTS_DIR / "uc11_exp80c_orthogonality.json"


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    banner("EXPERIMENT #80c — LEVEL CORRECTION vs CROSS-STORE INFORMATION")

    data = M5Data()
    champ = champion_predictions(data, INNER_ORIGIN)
    days = np.sort(np.unique(champ["target_day_idx"]))
    pos = {d: i for i, d in enumerate(days)}
    n_d = len(days)
    di = np.array([pos[d] for d in champ["target_day_idx"]])
    si = champ["series_idx"].astype(np.int64)
    Ym = np.zeros((config.N_SERIES, n_d))
    Pm = np.zeros((config.N_SERIES, n_d))
    Ym[si, di] = champ["y"]
    Pm[si, di] = champ["blend"]

    base = metrics.rmse(Ym.ravel(), Pm.ravel())
    base_mae = metrics.mae(Ym.ravel(), Pm.ravel())
    hist = data.sales_wide[:, :INNER_ORIGIN + 1].mean(axis=1)
    highm = np.tile((hist > 3.0)[:, None], (1, n_d))
    log(f"  champion   RMSE {base:.4f}  MAE {base_mae:.4f}  "
        f"bias {Pm.mean() - Ym.mean():+.4f}")

    agg = AggregateLevel(data, "item")
    g, n_g = agg.group_of_series, agg.n_groups
    F = np.zeros((n_g, n_d))
    np.add.at(F, g, Pm)
    p_agg, valid, _ = _probe.train_agg_model(agg, INNER_ORIGIN, objective="regression")
    Ahat = np.zeros((n_g, n_d))
    Ahat[valid["group_idx"].to_numpy(),
         [pos[d] for d in valid["target_day_idx"].to_numpy()]] = p_agg

    # --- decompose the multiplicative correction ------------------------
    ok = (F > 1e-6) & (Ahat > 1e-6)
    ratio = np.where(ok, Ahat / np.where(ok, F, 1.0), 1.0)
    logr = np.where(ok, np.log(np.where(ok, ratio, 1.0)), 0.0)
    # volume-weighted global component, so it matches what a global rescale does
    wts = np.where(ok, F, 0.0)
    g_mean = float((logr * wts).sum() / wts.sum())
    ratio_demeaned = np.where(ok, np.exp(logr - g_mean), 1.0)
    log(f"\n  global component of the item correction: "
        f"exp({g_mean:+.4f}) = x{np.exp(g_mean):.4f}")
    log(f"  spread of the item-specific remainder: sd(log) = "
        f"{float(np.sqrt((wts * (logr - g_mean) ** 2).sum() / wts.sum())):.4f}")

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(F[g] > 1e-9, Pm / np.where(F[g] > 1e-9, F[g], 1.0), 0.0)

    def score(P):
        return {"RMSE": metrics.rmse(Ym.ravel(), P.ravel()),
                "MAE": metrics.mae(Ym.ravel(), P.ravel()),
                "highvol_RMSE": metrics.rmse(Ym[highm], P[highm])}

    def best_over_alpha(make):
        b = None
        for a in ALPHAS:
            s = score(make(a))
            s["alpha"] = float(a)
            if b is None or s["RMSE"] < b["RMSE"]:
                b = s
        return b

    results = {}

    # 1 champion
    results["1_champion"] = {**score(Pm), "alpha": None}

    # 2 global rescale only (oracle constant)
    c = float((Ym * Pm).sum() / (Pm * Pm).sum())
    results["2_global_rescale_oracle"] = {**score(np.clip(c * Pm, 0, None)),
                                          "alpha": None, "c": c}

    # 3 full reconciliation
    results["3_reconcile_full"] = best_over_alpha(
        lambda a: np.clip(Pm + a * share * (Ahat - F)[g], 0, None))

    # 4 demeaned (item-specific only)
    Ahat_dm = ratio_demeaned * F
    results["4_reconcile_demeaned"] = best_over_alpha(
        lambda a: np.clip(Pm + a * share * (Ahat_dm - F)[g], 0, None))

    # 5 global rescale THEN demeaned reconciliation
    Pc = np.clip(c * Pm, 0, None)
    Fc = np.zeros((n_g, n_d))
    np.add.at(Fc, g, Pc)
    with np.errstate(divide="ignore", invalid="ignore"):
        share_c = np.where(Fc[g] > 1e-9, Pc / np.where(Fc[g] > 1e-9, Fc[g], 1.0), 0.0)
    Ahat_dm_c = ratio_demeaned * Fc
    results["5_rescale_then_demeaned"] = best_over_alpha(
        lambda a: np.clip(Pc + a * share_c * (Ahat_dm_c - Fc)[g], 0, None))

    banner("RESULTS")
    log(f"  {'variant':<28}{'alpha':>7}{'RMSE':>10}{'dRMSE':>10}"
        f"{'MAE':>10}{'dMAE':>10}{'highvol':>10}")
    for k, v in results.items():
        al = "-" if v["alpha"] is None else f"{v['alpha']:.2f}"
        log(f"  {k:<28}{al:>7}{v['RMSE']:>10.4f}{v['RMSE']-base:>+10.4f}"
            f"{v['MAE']:>10.4f}{v['MAE']-base_mae:>+10.4f}"
            f"{v['highvol_RMSE']:>10.4f}")

    dm = results["4_reconcile_demeaned"]["RMSE"] - base
    full = results["3_reconcile_full"]["RMSE"] - base
    banner("READING")
    log(f"  full correction              {full:+.4f}")
    log(f"  item-specific component only {dm:+.4f}  "
        f"({100 * dm / full:.0f}% of the full gain)")
    log(f"  global rescale oracle alone  "
        f"{results['2_global_rescale_oracle']['RMSE'] - base:+.4f}")
    log(f"  rescale + item-specific      "
        f"{results['5_rescale_then_demeaned']['RMSE'] - base:+.4f}")
    log("")
    if dm <= -0.005:
        log("  -> the item-specific (cross-store) component stands on its own.")
        log("     It is not a global calibration trick. The level anomaly of this")
        log("     window is a SEPARATE issue and must not be folded into alpha.")
    else:
        log("  -> the gain is essentially a level correction. Reject the")
        log("     hierarchical framing and treat calibration separately.")

    OUT_JSON.write_text(json.dumps({
        "window": f"origin d_{INNER_ORIGIN+1}",
        "champion_bias": float(Pm.mean() - Ym.mean()),
        "global_component_of_correction": float(np.exp(g_mean)),
        "results": results,
        "item_specific_share_of_gain_pct": float(100 * dm / full),
    }, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {OUT_JSON.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
