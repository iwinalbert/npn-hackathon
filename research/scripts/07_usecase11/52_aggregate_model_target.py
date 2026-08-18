
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics

PRED_FILE = config.PREDICTIONS_DIR / "exp_76_diversity_blend_validation.csv"
OUT_JSON = config.ARTIFACTS_DIR / "uc11_aggregate_model_target.json"


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    banner("USE CASE 11 — REQUIRED AGGREGATE-MODEL ACCURACY (read-only)")

    meta = pd.read_csv(config.SALES_EVAL_CSV,
                       usecols=["item_id", "store_id", "state_id", "dept_id"])
    pred = pd.read_csv(PRED_FILE)
    days = np.sort(pred["target_day_idx"].unique())
    day_pos = {d: i for i, d in enumerate(days)}
    Y = np.zeros((config.N_SERIES, len(days)))
    P = np.zeros((config.N_SERIES, len(days)))
    Y[pred["series_idx"].to_numpy(),
      pred["target_day_idx"].map(day_pos).to_numpy()] = pred["y_true"].to_numpy()
    P[pred["series_idx"].to_numpy(),
      pred["target_day_idx"].map(day_pos).to_numpy()] = pred["y_pred"].to_numpy()

    base_sse = float(((Y - P) ** 2).sum())
    n = Y.size
    log(f"  base RMSE {np.sqrt(base_sse / n):.4f}   SSE {base_sse:,.0f}   n {n:,}")

    E = Y - P
    out = {}

    for lname, key in [
        ("L9_store_dept", meta.store_id.astype(str) + "|" + meta.dept_id.astype(str)),
        ("L10_item", meta.item_id.astype(str)),
        ("L11_item_state", meta.item_id.astype(str) + "|" + meta.state_id.astype(str)),
    ]:
        g = pd.factorize(key)[0].astype(np.int32)
        n_g = int(g.max()) + 1
        A = np.zeros((n_g, len(days)))
        F = np.zeros((n_g, len(days)))
        np.add.at(A, g, Y)
        np.add.at(F, g, P)
        D = A - F

        Fg = F[g]
        share = np.where(Fg > 1e-9, P / np.where(Fg > 1e-9, Fg, 1.0), 0.0)

        c = np.zeros((n_g, len(days)))
        k = np.zeros((n_g, len(days)))
        np.add.at(c, g, E * share)
        np.add.at(k, g, share ** 2)

        S_cD = float((c * D).sum())
        S_kD2 = float((k * D * D).sum())
        S_k = float(k.sum())
        n_gd = n_g * len(days)

        d_sse_1 = -2 * S_cD + S_kD2
        alpha_star = S_cD / S_kD2 if S_kD2 > 0 else 0.0
        d_sse_star = -(S_cD ** 2) / S_kD2 if S_kD2 > 0 else 0.0

        var_max = (2 * S_cD - S_kD2) / S_k if S_k > 0 else 0.0
        rmse_target = float(np.sqrt(max(var_max, 0.0)))
        bu_rmse = metrics.rmse(A.ravel(), F.ravel())

        curve = []
        for mult in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
            var_eta = (bu_rmse * mult) ** 2
            gain_sse = -(S_cD ** 2) / (S_kD2 + S_k * var_eta)
            curve.append({
                "agg_RMSE": bu_rmse * mult,
                "vs_bottom_up_pct": (mult - 1) * 100,
                "bottom_RMSE": float(np.sqrt((base_sse + gain_sse) / n)),
                "bottom_dRMSE": float(np.sqrt((base_sse + gain_sse) / n)
                                      - np.sqrt(base_sse / n)),
            })

        out[lname] = {
            "n_groups": n_g, "n_group_days": n_gd,
            "bottom_up_aggregate_RMSE": bu_rmse,
            "sum_cD": S_cD, "sum_kD2": S_kD2, "sum_k": S_k,
            "oracle_alpha1_dSSE": d_sse_1,
            "oracle_alpha1_bottom_RMSE": float(np.sqrt((base_sse + d_sse_1) / n)),
            "oracle_alphastar": alpha_star,
            "oracle_alphastar_bottom_RMSE": float(np.sqrt((base_sse + d_sse_star) / n)),
            "breakeven_agg_RMSE_at_alpha1": rmse_target,
            "breakeven_vs_bottom_up_pct": (rmse_target / bu_rmse - 1) * 100,
            "shrunk_gain_curve": curve,
        }

        banner(f"{lname}   ({n_g} groups)")
        log(f"  bottom-up aggregate RMSE            {bu_rmse:10.4f}")
        log(f"  oracle (alpha=1)  bottom RMSE       "
            f"{out[lname]['oracle_alpha1_bottom_RMSE']:10.4f}")
        log(f"  oracle (alpha*={alpha_star:.3f}) bottom RMSE   "
            f"{out[lname]['oracle_alphastar_bottom_RMSE']:10.4f}")
        log(f"  break-even aggregate RMSE (alpha=1) {rmse_target:10.4f}   "
            f"({out[lname]['breakeven_vs_bottom_up_pct']:+.1f}% vs bottom-up)")
        log("\n  with alpha fitted, expected bottom-level gain by aggregate accuracy:")
        log(f"    {'agg RMSE':>10}{'vs BU':>9}{'bottom RMSE':>14}{'dRMSE':>10}")
        for r in curve:
            log(f"    {r['agg_RMSE']:>10.3f}{r['vs_bottom_up_pct']:>+8.0f}%"
                f"{r['bottom_RMSE']:>14.4f}{r['bottom_dRMSE']:>+10.4f}")

    banner("VERDICT INPUT")
    log("  A dedicated aggregate model is worth building only where the")
    log("  break-even target is a plausible improvement on the bottom-up sum.")
    for lname, v in out.items():
        log(f"    {lname:<16} needs {v['breakeven_agg_RMSE_at_alpha1']:.3f} "
            f"vs bottom-up {v['bottom_up_aggregate_RMSE']:.3f}  "
            f"({v['breakeven_vs_bottom_up_pct']:+.1f}%)")

    OUT_JSON.write_text(json.dumps({
        "source_predictions": PRED_FILE.name,
        "base_RMSE": float(np.sqrt(base_sse / n)),
        "levels": out,
    }, indent=2), encoding="utf-8")
    log(f"\n  wrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
