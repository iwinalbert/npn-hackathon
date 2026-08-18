
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics

PRED_FILE = config.PREDICTIONS_DIR / "exp_76_diversity_blend_validation.csv"
OUT_JSON = config.ARTIFACTS_DIR / "uc11_reconciliation_threshold.json"
OUT_CSV = config.ARTIFACTS_DIR / "uc11_reconciliation_threshold.csv"

LAMBDAS = np.round(np.arange(0.0, 1.001, 0.05), 2)


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    banner("USE CASE 11 — RECONCILIATION BREAK-EVEN THRESHOLD (read-only)")

    meta = pd.read_csv(config.SALES_EVAL_CSV,
                       usecols=["item_id", "store_id", "state_id", "dept_id"])
    pred = pd.read_csv(PRED_FILE)

    days = np.sort(pred["target_day_idx"].unique())
    day_pos = {d: i for i, d in enumerate(days)}
    n_s, n_d = config.N_SERIES, len(days)
    Y = np.zeros((n_s, n_d))
    P = np.zeros((n_s, n_d))
    Y[pred["series_idx"].to_numpy(),
      pred["target_day_idx"].map(day_pos).to_numpy()] = pred["y_true"].to_numpy()
    P[pred["series_idx"].to_numpy(),
      pred["target_day_idx"].map(day_pos).to_numpy()] = pred["y_pred"].to_numpy()

    base = metrics.rmse(Y.ravel(), P.ravel())
    log(f"  base bottom-level RMSE {base:.4f}   MAE {metrics.mae(Y.ravel(), P.ravel()):.4f}")

    groupings = {
        "L9_store_dept": pd.factorize(meta.store_id.astype(str) + "|"
                                      + meta.dept_id.astype(str))[0],
        "L10_item": pd.factorize(meta.item_id)[0],
        "L11_item_state": pd.factorize(meta.item_id.astype(str) + "|"
                                       + meta.state_id.astype(str))[0],
    }

    rows = []
    for lname, g in groupings.items():
        g = g.astype(np.int32)
        n_g = int(g.max()) + 1
        sizes = np.bincount(g, minlength=n_g).astype(float)
        A = np.zeros((n_g, n_d))
        F = np.zeros((n_g, n_d))
        np.add.at(A, g, Y)
        np.add.at(F, g, P)
        D = A - F

        banner(f"{lname}   ({n_g} groups, mean size {sizes.mean():.1f})")
        log(f"  bottom-up aggregate: RMSE {metrics.rmse(A.ravel(), F.ravel()):.4f}   "
            f"WAPE {metrics.wape(A.ravel(), F.ravel()):.4f}")
        log(f"  {'lambda':>7}{'prop':>11}{'equal':>11}{'shrunk':>11}"
            f"{'alpha*':>9}{'prop dRMSE':>12}")

        for lam in LAMBDAS:
            Dhat = lam * D
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(F > 1e-9, 1.0 + Dhat / F, 1.0)
            p_prop = np.clip(P * scale[g], 0, None)
            p_equal = np.clip(P + (Dhat / sizes[:, None])[g], 0, None)

            delta = P * scale[g] - P
            dd = float((delta ** 2).sum())
            alpha = float(((Y - P) * delta).sum() / dd) if dd > 1e-12 else 0.0
            p_shrunk = np.clip(P + alpha * delta, 0, None)

            r_prop = metrics.rmse(Y.ravel(), p_prop.ravel())
            r_equal = metrics.rmse(Y.ravel(), p_equal.ravel())
            r_shrunk = metrics.rmse(Y.ravel(), p_shrunk.ravel())
            rows.append({"level": lname, "lambda": float(lam),
                         "prop_RMSE": r_prop, "equal_RMSE": r_equal,
                         "shrunk_RMSE": r_shrunk, "alpha_star": alpha,
                         "prop_gain": r_prop - base,
                         "shrunk_gain": r_shrunk - base})
            if lam in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
                log(f"  {lam:>7.2f}{r_prop:>11.4f}{r_equal:>11.4f}{r_shrunk:>11.4f}"
                    f"{alpha:>9.3f}{r_prop - base:>+12.4f}")

    R = pd.DataFrame(rows)

    banner("BREAK-EVEN: smallest lambda at which each form beats the champion")
    summary = {}
    for lname in groupings:
        sub = R[R.level == lname]
        out = {}
        for form in ("prop", "shrunk"):
            col = f"{form}_gain"
            beat = sub[sub[col] < 0]
            out[f"{form}_breakeven_lambda"] = (float(beat["lambda"].min())
                                               if len(beat) else None)
            mean = sub[sub[col] <= -0.010]
            out[f"{form}_lambda_for_-0.010"] = (float(mean["lambda"].min())
                                                if len(mean) else None)
            out[f"{form}_gain_at_lambda_0.2"] = float(
                sub.loc[np.isclose(sub["lambda"], 0.2), col].iloc[0])
        summary[lname] = out
        log(f"  {lname:<16} prop break-even lambda = "
            f"{out['prop_breakeven_lambda']}   "
            f"lambda for -0.010 = {out['prop_lambda_for_-0.010']}   "
            f"gain at lambda=0.20 = {out['prop_gain_at_lambda_0.2']:+.4f}")

    R.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "source_predictions": PRED_FILE.name,
        "base_RMSE": base,
        "lambda_definition": ("A_hat = F + lambda*(A - F); lambda is the fraction "
                              "of the bottom-up aggregate discrepancy that a "
                              "dedicated aggregate model recovers, WITHOUT adding "
                              "any error of its own — optimistic by construction"),
        "summary": summary,
        "curve": rows,
    }, indent=2), encoding="utf-8")
    log(f"\n  wrote {OUT_CSV.name}, {OUT_JSON.name}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
