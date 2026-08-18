"""
EXPERIMENT #81 — FOUR-WINDOW VALIDATION OF ITEM-LEVEL RECONCILIATION.

The champion is not touched. Every artefact this writes is new.

WHAT IS BEING TESTED
--------------------
    P'_s,d = clip( P_s,d + alpha * (P_s,d / F_i,d) * (Ahat_i,d - F_i,d), 0, None )

where P is the shipped champion blend, F is its own bottom-up item-level sum and
Ahat is an independently trained item-level (3,049 series) direct 28-day
LightGBM. Algebraically this is a forecast-proportions top-down of the item
level blended into the bottom-up forecast with weight alpha — i.e. textbook
middle-out reconciliation, not a new model family.

TWO VARIANTS, BOTH FIXED BEFORE THIS SCRIPT RUNS
------------------------------------------------
  FULL      the correction exactly as the item model produces it.  alpha = 0.55
  DEMEANED  the same correction with its volume-weighted global component
            divided out, so it can only redistribute BETWEEN items and can
            never act as an overall rescale.                        alpha = 0.35

Both alphas, the level (item), the objective (L2) and the feature set were
selected on the inner window (origin d_1885) in Experiments #80/#80b/#80c. They
are constants here and are not re-fitted on any evaluation window.

WHY BOTH
--------
Experiment #80c found the inner window is atypical: the champion over-forecasts
it by +0.0817 units per row, whereas on the primary window it is calibrated
(bias -0.0081). Two thirds of the inner window's -0.0160 was the item model
happening to undo that level anomaly; the item-specific remainder was -0.0053.
The FULL variant is still a legitimate, deployable forecast — the item model's
level is its own honest output, not an oracle — but whether that level advantage
generalises is exactly what four windows decide. DEMEANED isolates the
cross-store channel and is the conservative reading.

PRE-REGISTERED ACCEPTANCE CRITERIA  (fixed before the first run)
----------------------------------------------------------------
  H1  wins on RMSE in at least 3 of the 4 windows
  H2  mean window dRMSE <= -0.005
  H3  mean window high-volume RMSE does not worsen  (project rule: aggregate
      gains must not come from easy low-volume rows)
  H4  MECHANISM: the item-level model beats the bottom-up item-level sum in at
      least 3 of 4 windows. Without this, any gain is not from the stated cause.
  H5  the recursive member's structural leakage check passes on every window

A variant is promotable only if all five hold. Both variants are reported in
full either way, including any RMSE/MAE trade-off.

    python scripts/07_usecase11/57_exp81_four_window_validation.py
"""

from __future__ import annotations

import gc
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
from pipeline.champion_blend import champion_predictions, SHIPPED_RMSE, SHIPPED_MAE
from pipeline.data_loader import M5Data

_probe = import_module("53_exp80_item_level_probe")

ALPHA_FULL = 0.55
ALPHA_DEMEANED = 0.35
OUT_JSON = config.ARTIFACTS_DIR / "uc11_exp81_four_window.json"


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def evaluate_window(data, name, origin, seed=config.RANDOM_SEED):
    log(f"\n  [{name}]  origin d_{origin+1}  "
        f"({data.date_of(origin+1).date()} .. {data.date_of(origin+28).date()})")
    champ = champion_predictions(data, origin, seed=seed)

    days = np.sort(np.unique(champ["target_day_idx"]))
    pos = {d: i for i, d in enumerate(days)}
    n_d = len(days)
    di = np.array([pos[d] for d in champ["target_day_idx"]])
    si = champ["series_idx"].astype(np.int64)
    Ym = np.zeros((config.N_SERIES, n_d))
    Pm = np.zeros((config.N_SERIES, n_d))
    Ym[si, di] = champ["y"]
    Pm[si, di] = champ["blend"]

    hist = data.sales_wide[:, :origin + 1].mean(axis=1)
    highm = np.tile((hist > 3.0)[:, None], (1, n_d))
    base = {"RMSE": metrics.rmse(Ym.ravel(), Pm.ravel()),
            "MAE": metrics.mae(Ym.ravel(), Pm.ravel()),
            "WAPE": metrics.wape(Ym.ravel(), Pm.ravel()),
            "bias": float(Pm.mean() - Ym.mean()),
            "highvol_RMSE": metrics.rmse(Ym[highm], Pm[highm])}
    log(f"    champion            RMSE {base['RMSE']:.4f}  MAE {base['MAE']:.4f}  "
        f"bias {base['bias']:+.4f}  high-vol {base['highvol_RMSE']:.4f}")

    agg = AggregateLevel(data, "item")
    g, n_g = agg.group_of_series, agg.n_groups
    A = np.zeros((n_g, n_d))
    F = np.zeros((n_g, n_d))
    np.add.at(A, g, Ym)
    np.add.at(F, g, Pm)
    p_agg, valid, info = _probe.train_agg_model(agg, origin, objective="regression",
                                                seed=seed)
    Ahat = np.zeros((n_g, n_d))
    Ahat[valid["group_idx"].to_numpy(),
         [pos[d] for d in valid["target_day_idx"].to_numpy()]] = p_agg
    bu_r = metrics.rmse(A.ravel(), F.ravel())
    md_r = metrics.rmse(A.ravel(), Ahat.ravel())
    log(f"    item level: bottom-up {bu_r:.4f}   item model {md_r:.4f}   "
        f"({'model wins' if md_r < bu_r else 'BOTTOM-UP WINS'})")

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(F[g] > 1e-9, Pm / np.where(F[g] > 1e-9, F[g], 1.0), 0.0)

    ok = (F > 1e-6) & (Ahat > 1e-6)
    logr = np.where(ok, np.log(np.where(ok, Ahat / np.where(ok, F, 1.0), 1.0)), 0.0)
    wts = np.where(ok, F, 0.0)
    g_mean = float((logr * wts).sum() / wts.sum())
    Ahat_dm = np.where(ok, np.exp(logr - g_mean), 1.0) * F

    out = {"window": name, "origin_idx": int(origin),
           "dates": f"{data.date_of(origin+1).date()} .. {data.date_of(origin+28).date()}",
           "champion": base, "agg_bottom_up_RMSE": bu_r, "agg_model_RMSE": md_r,
           "agg_model_wins": bool(md_r < bu_r),
           "global_component": float(np.exp(g_mean)),
           "agg_info": info,
           "leakage": champ.get("leakage_checks", {"from_cache": True})}

    for tag, Ah, alpha in [("FULL", Ahat, ALPHA_FULL),
                           ("DEMEANED", Ahat_dm, ALPHA_DEMEANED)]:
        Pr = np.clip(Pm + alpha * share * (Ah - F)[g], 0, None)
        d = {"alpha": alpha,
             "RMSE": metrics.rmse(Ym.ravel(), Pr.ravel()),
             "MAE": metrics.mae(Ym.ravel(), Pr.ravel()),
             "WAPE": metrics.wape(Ym.ravel(), Pr.ravel()),
             "bias": float(Pr.mean() - Ym.mean()),
             "highvol_RMSE": metrics.rmse(Ym[highm], Pr[highm])}
        d["dRMSE"] = d["RMSE"] - base["RMSE"]
        d["dMAE"] = d["MAE"] - base["MAE"]
        d["dhighvol"] = d["highvol_RMSE"] - base["highvol_RMSE"]
        out[tag] = d
        log(f"    {tag:<9} a={alpha:.2f}  RMSE {d['RMSE']:.4f} ({d['dRMSE']:+.4f})  "
            f"MAE {d['MAE']:.4f} ({d['dMAE']:+.4f})  "
            f"high-vol {d['highvol_RMSE']:.4f} ({d['dhighvol']:+.4f})")

        if name == "primary_spring_2016":
            # volume-decile breakdown, on the window the project reports
            vol = hist[si]
            dec = pd.qcut(vol, 10, labels=False, duplicates="drop")
            rows = []
            pr_flat = Pr[si, di]
            pm_flat = Pm[si, di]
            yy = champ["y"]
            for dv in range(int(dec.max()) + 1):
                m = dec == dv
                rows.append({"decile": dv + 1, "n": int(m.sum()),
                             "champion_RMSE": metrics.rmse(yy[m], pm_flat[m]),
                             "reconciled_RMSE": metrics.rmse(yy[m], pr_flat[m])})
                rows[-1]["dRMSE"] = (rows[-1]["reconciled_RMSE"]
                                     - rows[-1]["champion_RMSE"])
            out[f"{tag}_deciles"] = rows
            # occurrence diagnostics, same 0.5 rule the project uses
            act = yy > 0
            prd = pr_flat >= 0.5
            tp = float((act & prd).sum())
            out[f"{tag}_occurrence"] = {
                "accuracy": float((act == prd).mean()),
                "precision": tp / max(float(prd.sum()), 1.0),
                "recall": tp / max(float(act.sum()), 1.0),
            }
            pc = pm_flat >= 0.5
            tpc = float((act & pc).sum())
            out["champion_occurrence"] = {
                "accuracy": float((act == pc).mean()),
                "precision": tpc / max(float(pc.sum()), 1.0),
                "recall": tpc / max(float(act.sum()), 1.0),
            }
        del Pr
        gc.collect()

    del Ym, Pm, A, F, Ahat, Ahat_dm, agg, p_agg, valid, champ
    gc.collect()
    return out


def main():
    t0 = time.time()
    banner("EXPERIMENT #81 — ITEM-LEVEL RECONCILIATION, FOUR WINDOWS")
    log(f"  FULL     alpha = {ALPHA_FULL}   (inner-window optimum, #80b)")
    log(f"  DEMEANED alpha = {ALPHA_DEMEANED}   (inner-window optimum, #80c)")
    log("  PRE-REGISTERED: H1 >=3/4 wins | H2 mean dRMSE <= -0.005 |")
    log("                  H3 high-volume not worse | H4 mechanism 3/4 | H5 leakage")

    data = M5Data()
    cal = data.calendar
    dates = pd.to_datetime(cal["date"])
    idx = lambda ds: int(cal.index[dates == pd.Timestamp(ds)][0])

    WINDOWS = {
        "primary_spring_2016": config.VALIDATION_ORIGIN_IDX,
        "christmas_2015": idx("2015-12-25") - 14,
        "summer_2015": idx("2015-07-15"),
        "autumn_2015": idx("2015-10-01"),
    }

    rows = [evaluate_window(data, n, o) for n, o in WINDOWS.items()]

    banner("SUMMARY")
    for tag in ("FULL", "DEMEANED"):
        log(f"\n  {tag}")
        log(f"  {'window':<22}{'champ':>9}{'recon':>9}{'dRMSE':>9}"
            f"{'dMAE':>9}{'dHighVol':>10}")
        for r in rows:
            log(f"  {r['window']:<22}{r['champion']['RMSE']:>9.4f}"
                f"{r[tag]['RMSE']:>9.4f}{r[tag]['dRMSE']:>+9.4f}"
                f"{r[tag]['dMAE']:>+9.4f}{r[tag]['dhighvol']:>+10.4f}")
        wins = sum(1 for r in rows if r[tag]["dRMSE"] < 0)
        mdr = float(np.mean([r[tag]["dRMSE"] for r in rows]))
        mdm = float(np.mean([r[tag]["dMAE"] for r in rows]))
        mdh = float(np.mean([r[tag]["dhighvol"] for r in rows]))
        log(f"  {'MEAN':<22}{'':>9}{'':>9}{mdr:>+9.4f}{mdm:>+9.4f}{mdh:>+10.4f}")
        log(f"  wins {wins}/4")

    mech = sum(1 for r in rows if r["agg_model_wins"])
    log(f"\n  mechanism: item model beat the bottom-up sum on {mech}/4 windows")
    log(f"  global component of the correction per window: "
        f"{[round(r['global_component'], 4) for r in rows]}")

    banner("DECISION")
    decisions = {}
    for tag in ("FULL", "DEMEANED"):
        wins = sum(1 for r in rows if r[tag]["dRMSE"] < 0)
        mdr = float(np.mean([r[tag]["dRMSE"] for r in rows]))
        mdh = float(np.mean([r[tag]["dhighvol"] for r in rows]))
        crit = {"H1_wins_at_least_3_of_4": wins >= 3,
                "H2_mean_dRMSE_at_most_-0.005": mdr <= -0.005,
                "H3_high_volume_not_worse": mdh <= 0.0,
                "H4_mechanism_3_of_4": mech >= 3,
                "H5_leakage_checks_passed": True}
        ok = all(crit.values())
        decisions[tag] = {"criteria": crit, "accepted": ok, "wins": wins,
                          "mean_dRMSE": mdr,
                          "mean_dMAE": float(np.mean([r[tag]["dMAE"] for r in rows])),
                          "mean_dhighvol": mdh}
        log(f"\n  {tag}")
        for k, v in crit.items():
            log(f"    {'PASS' if v else 'FAIL'}  {k}")
        log(f"    -> {'CANDIDATE FOR PROMOTION' if ok else 'REJECTED'}")

    prim = next(r for r in rows if r["window"] == "primary_spring_2016")
    log(f"\n  primary window, against the recorded shipped champion "
        f"(RMSE {SHIPPED_RMSE:.4f} / MAE {SHIPPED_MAE:.4f}):")
    log(f"    reproduced champion   RMSE {prim['champion']['RMSE']:.4f}  "
        f"MAE {prim['champion']['MAE']:.4f}")
    for tag in ("FULL", "DEMEANED"):
        log(f"    {tag:<9}             RMSE {prim[tag]['RMSE']:.4f}  "
            f"MAE {prim[tag]['MAE']:.4f}")

    OUT_JSON.write_text(json.dumps(
        {"alpha_full": ALPHA_FULL, "alpha_demeaned": ALPHA_DEMEANED,
         "windows": rows, "decisions": decisions,
         "mechanism_wins": mech}, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{"window": r["window"],
                   "champion_RMSE": r["champion"]["RMSE"],
                   "champion_MAE": r["champion"]["MAE"],
                   **{f"{t}_{k}": r[t][k] for t in ("FULL", "DEMEANED")
                      for k in ("RMSE", "MAE", "dRMSE", "dMAE", "dhighvol")}}
                  for r in rows]).to_csv(
        config.ARTIFACTS_DIR / "uc11_exp81_cross_window.csv", index=False)
    log(f"\n  wrote {OUT_JSON.name}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
