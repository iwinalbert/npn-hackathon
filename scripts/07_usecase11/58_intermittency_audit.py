"""
USE CASE 11 / REQUIREMENT 7 — INTERMITTENT DEMAND AUDIT.

READ-ONLY. Trains nothing. Writes uc11_intermittency_audit.json + a CSV.

The brief lists Croston, SBA, TSB, hurdle and segmentation as candidate
treatments for intermittency, and asks whether Tweedie is adequate. The project
has already tested the hurdle family twice (model_05_hurdle 2.1267,
opt_07_hurdle_v2 2.1241, opt_07_hurdle_v2_calibrated 2.1822, all worse than the
then-champion's 2.1210) and measured an intermittency-quintile segmentation
oracle of only -0.0025. Croston, SBA and TSB have NOT been tested, and
"Tweedie is fine" has never been checked REGIME BY REGIME — only in aggregate.
Both gaps are closed here.

WHAT IS COMPUTED
----------------
1. SYNTETOS-BOYLAN CLASSIFICATION of all 30,490 series from pre-origin history:

       ADI  = average inter-demand interval          cut at 1.32
       CV^2 = squared coefficient of variation of the NON-ZERO demand sizes
                                                      cut at 0.49

       smooth        ADI < 1.32 and CV^2 < 0.49
       erratic       ADI < 1.32 and CV^2 >= 0.49
       intermittent  ADI >= 1.32 and CV^2 < 0.49
       lumpy         ADI >= 1.32 and CV^2 >= 0.49

2. THE CHAMPION'S BEHAVIOUR IN EACH REGIME — RMSE, MAE, bias, share of total
   squared error, and share of series.

3. CROSTON / SBA / TSB, implemented here and fitted on pre-origin history only.
   Each produces one constant per series for all 28 horizon days, which is what
   those methods are: they estimate a rate, not a calendar. Scored in each
   regime against the champion.

4. THE ORACLE FOR REGIME SPECIALISATION — the best per-regime multiplicative
   recalibration, fitted on the evaluation window itself. It is the upper bound
   on anything a regime-specific model could achieve through level alone, and
   if it is negligible the whole segmentation idea is dead regardless of which
   model would sit in each segment.

5. THE ORACLE FOR A PER-REGIME BLEND with the best classical method — the upper
   bound on "use SBA where it wins, the champion elsewhere".

    python scripts/07_usecase11/58_intermittency_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics
from pipeline.data_loader import M5Data

PRED_FILE = config.PREDICTIONS_DIR / "exp_76_diversity_blend_validation.csv"
OUT_JSON = config.ARTIFACTS_DIR / "uc11_intermittency_audit.json"
ADI_CUT, CV2_CUT = 1.32, 0.49
HIST_DAYS = 728          # two years of pre-origin history for the classification
ALPHA = 0.1              # smoothing constant for Croston / SBA / TSB


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def classify(sales: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADI, CV^2 and a regime label per series, from the given history block."""
    nz = sales > 0
    counts = nz.sum(axis=1)
    n_days = sales.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        adi = np.where(counts > 0, n_days / np.maximum(counts, 1), np.inf)
    cv2 = np.zeros(sales.shape[0])
    for i in range(sales.shape[0]):
        v = sales[i][nz[i]]
        if v.size > 1 and v.mean() > 0:
            cv2[i] = (v.std() / v.mean()) ** 2
    lab = np.empty(sales.shape[0], dtype=object)
    lab[(adi < ADI_CUT) & (cv2 < CV2_CUT)] = "smooth"
    lab[(adi < ADI_CUT) & (cv2 >= CV2_CUT)] = "erratic"
    lab[(adi >= ADI_CUT) & (cv2 < CV2_CUT)] = "intermittent"
    lab[(adi >= ADI_CUT) & (cv2 >= CV2_CUT)] = "lumpy"
    lab[counts == 0] = "never sold"
    return adi, cv2, lab


def croston_family(sales: np.ndarray, alpha: float = ALPHA):
    """
    Croston, SBA (Syntetos-Boylan approximation) and TSB, vectorised over series.

    Croston  z <- alpha*size + (1-alpha)*z ;  p <- alpha*interval + (1-alpha)*p
             forecast = z / p
    SBA      = (1 - alpha/2) * Croston, the standard bias correction
    TSB      z <- alpha*size + (1-alpha)*z on demand periods
             prob <- beta*1 + (1-beta)*prob   every period (decays without demand)
             forecast = prob * z
    All three read the supplied history block only, which the caller slices to
    end at the forecast origin.
    """
    n, T = sales.shape
    z = np.zeros(n)
    p = np.ones(n)
    started = np.zeros(n, dtype=bool)
    gap = np.zeros(n)
    z_t = np.zeros(n)
    prob = np.zeros(n)
    beta = alpha

    for t in range(T):
        v = sales[:, t]
        d = v > 0
        gap += 1.0
        # --- Croston / SBA ---
        first = d & ~started
        z[first] = v[first]
        p[first] = 1.0
        started |= first
        upd = d & started & ~first
        z[upd] = alpha * v[upd] + (1 - alpha) * z[upd]
        p[upd] = alpha * gap[upd] + (1 - alpha) * p[upd]
        gap[d] = 0.0
        # --- TSB ---
        z_t[d] = alpha * v[d] + (1 - alpha) * z_t[d]
        prob = beta * d.astype(float) + (1 - beta) * prob

    with np.errstate(divide="ignore", invalid="ignore"):
        cro = np.where(p > 0, z / np.maximum(p, 1e-9), 0.0)
    return {"croston": cro, "sba": (1 - alpha / 2) * cro, "tsb": prob * z_t}


def main():
    banner("USE CASE 11 — INTERMITTENT DEMAND AUDIT (read-only)")
    data = M5Data()
    origin = config.VALIDATION_ORIGIN_IDX
    hist = data.sales_wide[:, max(0, origin + 1 - HIST_DAYS):origin + 1].astype(np.float64)
    log(f"  classification history: {hist.shape[1]} days ending at d_{origin+1}")

    adi, cv2, lab = classify(hist)
    log(f"  Syntetos-Boylan cuts: ADI {ADI_CUT}, CV^2 {CV2_CUT}")

    pred = pd.read_csv(PRED_FILE)
    y = pred["y_true"].to_numpy()
    p = pred["y_pred"].to_numpy()
    s_idx = pred["series_idx"].to_numpy()
    row_lab = lab[s_idx]
    resid = y - p
    tot_sq = float((resid ** 2).sum())
    base = metrics.rmse(y, p)
    log(f"  champion: RMSE {base:.4f}  MAE {metrics.mae(y, p):.4f}")

    # ------------------------------------------------------------------
    banner("A. REGIME COMPOSITION AND CHAMPION BEHAVIOUR")
    log(f"  {'regime':<14}{'series':>8}{'rows':>10}{'zero%':>8}{'meanY':>8}"
        f"{'RMSE':>9}{'MAE':>8}{'bias':>9}{'sqerr%':>9}")
    regimes = []
    for r in ["smooth", "erratic", "intermittent", "lumpy", "never sold"]:
        m = row_lab == r
        if m.sum() == 0:
            continue
        rec = {"regime": r, "n_series": int((lab == r).sum()), "n_rows": int(m.sum()),
               "zero_pct": float((y[m] == 0).mean() * 100),
               "mean_actual": float(y[m].mean()),
               "RMSE": metrics.rmse(y[m], p[m]), "MAE": metrics.mae(y[m], p[m]),
               "bias": float((p[m] - y[m]).mean()),
               "sq_error_share_pct": float((resid[m] ** 2).sum() / tot_sq * 100)}
        regimes.append(rec)
        log(f"  {r:<14}{rec['n_series']:>8}{rec['n_rows']:>10,}"
            f"{rec['zero_pct']:>8.1f}{rec['mean_actual']:>8.3f}"
            f"{rec['RMSE']:>9.4f}{rec['MAE']:>8.4f}{rec['bias']:>+9.4f}"
            f"{rec['sq_error_share_pct']:>9.2f}")

    # ------------------------------------------------------------------
    banner("B. CROSTON / SBA / TSB vs THE CHAMPION, PER REGIME")
    log("  Each classical method produces one constant per series for all 28")
    log("  days; that is what these methods are. Fitted on pre-origin history.")
    cf = croston_family(data.sales_wide[:, :origin + 1].astype(np.float64))
    classic = {k: v[s_idx] for k, v in cf.items()}
    classic["rolling_mean_28"] = np.repeat(
        data.sales_wide[:, origin - 27:origin + 1].mean(axis=1), 1)[s_idx]

    log(f"\n  {'regime':<14}{'champion':>10}" +
        "".join(f"{k:>12}" for k in classic))
    per_regime_classic = []
    for r in ["smooth", "erratic", "intermittent", "lumpy", "never sold"]:
        m = row_lab == r
        if m.sum() == 0:
            continue
        row = {"regime": r, "champion": metrics.rmse(y[m], p[m])}
        for k, v in classic.items():
            row[k] = metrics.rmse(y[m], v[m])
        per_regime_classic.append(row)
        log(f"  {r:<14}{row['champion']:>10.4f}" +
            "".join(f"{row[k]:>12.4f}" for k in classic))
    row = {"regime": "ALL", "champion": base}
    for k, v in classic.items():
        row[k] = metrics.rmse(y, v)
    per_regime_classic.append(row)
    log(f"  {'ALL':<14}{base:>10.4f}" + "".join(f"{row[k]:>12.4f}" for k in classic))

    beats = [(r["regime"], k) for r in per_regime_classic for k in classic
             if r[k] < r["champion"]]
    log(f"\n  regimes where a classical method beats the champion: "
        f"{beats if beats else 'NONE'}")

    # ------------------------------------------------------------------
    banner("C. ORACLE — WHAT REGIME SPECIALISATION COULD BE WORTH")
    log("  Best per-regime multiplicative rescale, fitted on the evaluation")
    log("  window itself. An upper bound no honest model can reach.")
    pc = p.copy()
    for r in np.unique(row_lab):
        m = row_lab == r
        denom = float((p[m] ** 2).sum())
        if denom > 0:
            pc[m] = p[m] * float((y[m] * p[m]).sum() / denom)
    oracle_scale = metrics.rmse(y, np.clip(pc, 0, None))
    log(f"    per-regime rescale oracle       RMSE {oracle_scale:.4f}  "
        f"({oracle_scale - base:+.4f})")

    log("\n  Best per-regime blend with each classical method, weight fitted on")
    log("  the evaluation window itself. Also an upper bound.")
    for k, v in classic.items():
        pb = p.copy()
        for r in np.unique(row_lab):
            m = row_lab == r
            d = v[m] - p[m]
            dd = float((d ** 2).sum())
            w = float(((y[m] - p[m]) * d).sum() / dd) if dd > 1e-9 else 0.0
            w = min(max(w, 0.0), 1.0)
            pb[m] = p[m] + w * d
        r_ = metrics.rmse(y, np.clip(pb, 0, None))
        log(f"    per-regime blend with {k:<16} RMSE {r_:.4f}  ({r_ - base:+.4f})")

    # ------------------------------------------------------------------
    banner("D. IS TWEEDIE MISCALIBRATED ANYWHERE?")
    log("  A wrong likelihood shows up as regime-dependent bias. Flat bias")
    log("  across regimes means the single Tweedie head is adequate.")
    for rec in regimes:
        rel = rec["bias"] / max(rec["mean_actual"], 1e-9) * 100
        log(f"    {rec['regime']:<14} bias {rec['bias']:+.4f}  "
            f"({rel:+.2f}% of mean demand)")

    OUT_JSON.write_text(json.dumps({
        "classification": {"ADI_cut": ADI_CUT, "CV2_cut": CV2_CUT,
                           "history_days": HIST_DAYS,
                           "alpha_smoothing": ALPHA},
        "champion_RMSE": base,
        "regimes": regimes,
        "classical_methods_per_regime": per_regime_classic,
        "regimes_where_classical_wins": beats,
        "per_regime_rescale_oracle_RMSE": oracle_scale,
        "per_regime_rescale_oracle_gain": oracle_scale - base,
    }, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(per_regime_classic).to_csv(
        config.ARTIFACTS_DIR / "uc11_intermittency_regimes.csv", index=False)
    log(f"\n  wrote {OUT_JSON.name} and uc11_intermittency_regimes.csv")


if __name__ == "__main__":
    main()
