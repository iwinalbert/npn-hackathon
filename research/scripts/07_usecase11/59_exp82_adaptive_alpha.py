
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
from pipeline.champion_blend import champion_predictions
from pipeline.data_loader import M5Data

_probe = import_module("53_exp80_item_level_probe")

ALPHAS = np.round(np.arange(0.0, 1.01, 0.05), 2)
OUT_JSON = config.ARTIFACTS_DIR / "uc11_exp82_adaptive_alpha.json"


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def window_pieces(data, origin, seed=config.RANDOM_SEED):
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

    agg = AggregateLevel(data, "item")
    g, n_g = agg.group_of_series, agg.n_groups
    A = np.zeros((n_g, n_d))
    F = np.zeros((n_g, n_d))
    np.add.at(A, g, Ym)
    np.add.at(F, g, Pm)
    p_agg, valid, _ = _probe.train_agg_model(agg, origin, objective="regression",
                                             seed=seed)
    Ahat = np.zeros((n_g, n_d))
    Ahat[valid["group_idx"].to_numpy(),
         [pos[d] for d in valid["target_day_idx"].to_numpy()]] = p_agg

    ok = (F > 1e-6) & (Ahat > 1e-6)
    logr = np.where(ok, np.log(np.where(ok, Ahat / np.where(ok, F, 1.0), 1.0)), 0.0)
    wts = np.where(ok, F, 0.0)
    g_mean = float((logr * wts).sum() / wts.sum())
    Ahat_dm = np.where(ok, np.exp(logr - g_mean), 1.0) * F

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(F[g] > 1e-9, Pm / np.where(F[g] > 1e-9, F[g], 1.0), 0.0)

    hist = data.sales_wide[:, :origin + 1].mean(axis=1)
    out = {"Ym": Ym, "Pm": Pm, "F": F, "A": A, "Ahat": Ahat, "Ahat_dm": Ahat_dm,
           "share": share, "g": g, "n_d": n_d,
           "high": np.tile((hist > 3.0)[:, None], (1, n_d)),
           "agg_bu_RMSE": metrics.rmse(A.ravel(), F.ravel()),
           "agg_model_RMSE": metrics.rmse(A.ravel(), Ahat.ravel())}
    del champ, agg, p_agg, valid
    gc.collect()
    return out


def apply_alpha(w, variant, alpha):
    Ah = w["Ahat"] if variant == "FULL" else w["Ahat_dm"]
    return np.clip(w["Pm"] + alpha * w["share"] * (Ah - w["F"])[w["g"]], 0, None)


def main():
    t0 = time.time()
    banner("EXPERIMENT #82 — PER-ORIGIN alpha, SELECTED ON THE PRECEDING WINDOW")
    log("  PRE-REGISTERED: K1 >=3/4 wins | K2 mean dRMSE <= -0.005 |")
    log("                  K3 high-volume not worse | K4 alpha spread <= 0.40 |")
    log("                  K5 mechanism 3/4")

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

    rows = []
    for name, origin in WINDOWS.items():
        sel_origin = origin - config.HORIZON
        banner(f"{name}   eval origin d_{origin+1}   "
               f"alpha selected on d_{sel_origin+1}")

        log("  --- selection window ---")
        sel = window_pieces(data, sel_origin)
        chosen = {}
        for variant in ("FULL", "DEMEANED"):
            best_a, best_r = None, None
            for a in ALPHAS:
                r = metrics.rmse(sel["Ym"].ravel(), apply_alpha(sel, variant, a).ravel())
                if best_r is None or r < best_r:
                    best_a, best_r = float(a), r
            chosen[variant] = best_a
            log(f"    {variant:<9} alpha* = {best_a:.2f}   "
                f"(selection-window RMSE {best_r:.4f})")
        del sel
        gc.collect()

        log("  --- evaluation window ---")
        ev = window_pieces(data, origin)
        base_rmse = metrics.rmse(ev["Ym"].ravel(), ev["Pm"].ravel())
        base_mae = metrics.mae(ev["Ym"].ravel(), ev["Pm"].ravel())
        base_high = metrics.rmse(ev["Ym"][ev["high"]], ev["Pm"][ev["high"]])
        log(f"    champion  RMSE {base_rmse:.4f}  MAE {base_mae:.4f}  "
            f"high-vol {base_high:.4f}")
        log(f"    item level: bottom-up {ev['agg_bu_RMSE']:.4f}  "
            f"model {ev['agg_model_RMSE']:.4f}")

        rec = {"window": name, "origin_idx": int(origin),
               "selection_origin_idx": int(sel_origin),
               "champion_RMSE": base_rmse, "champion_MAE": base_mae,
               "champion_highvol": base_high,
               "agg_bu_RMSE": ev["agg_bu_RMSE"],
               "agg_model_RMSE": ev["agg_model_RMSE"],
               "agg_model_wins": bool(ev["agg_model_RMSE"] < ev["agg_bu_RMSE"])}
        for variant in ("FULL", "DEMEANED"):
            a = chosen[variant]
            Pr = apply_alpha(ev, variant, a)
            r = metrics.rmse(ev["Ym"].ravel(), Pr.ravel())
            m = metrics.mae(ev["Ym"].ravel(), Pr.ravel())
            h = metrics.rmse(ev["Ym"][ev["high"]], Pr[ev["high"]])
            rec[variant] = {"alpha": a, "RMSE": r, "MAE": m, "highvol_RMSE": h,
                            "dRMSE": r - base_rmse, "dMAE": m - base_mae,
                            "dhighvol": h - base_high}
            log(f"    {variant:<9} a={a:.2f}  RMSE {r:.4f} ({r-base_rmse:+.4f})  "
                f"MAE {m:.4f} ({m-base_mae:+.4f})  "
                f"high-vol {h:.4f} ({h-base_high:+.4f})")
            del Pr
        rows.append(rec)
        del ev
        gc.collect()

    banner("SUMMARY")
    decisions = {}
    mech = sum(1 for r in rows if r["agg_model_wins"])
    for variant in ("FULL", "DEMEANED"):
        log(f"\n  {variant}")
        log(f"  {'window':<22}{'alpha':>7}{'champ':>9}{'recon':>9}"
            f"{'dRMSE':>9}{'dMAE':>9}{'dHighVol':>10}")
        for r in rows:
            v = r[variant]
            log(f"  {r['window']:<22}{v['alpha']:>7.2f}{r['champion_RMSE']:>9.4f}"
                f"{v['RMSE']:>9.4f}{v['dRMSE']:>+9.4f}{v['dMAE']:>+9.4f}"
                f"{v['dhighvol']:>+10.4f}")
        wins = sum(1 for r in rows if r[variant]["dRMSE"] < 0)
        mdr = float(np.mean([r[variant]["dRMSE"] for r in rows]))
        mdm = float(np.mean([r[variant]["dMAE"] for r in rows]))
        mdh = float(np.mean([r[variant]["dhighvol"] for r in rows]))
        alphas = [r[variant]["alpha"] for r in rows]
        spread = max(alphas) - min(alphas)
        log(f"  {'MEAN':<22}{'':>7}{'':>9}{'':>9}{mdr:>+9.4f}{mdm:>+9.4f}{mdh:>+10.4f}")
        crit = {"K1_wins_at_least_3_of_4": wins >= 3,
                "K2_mean_dRMSE_at_most_-0.005": mdr <= -0.005,
                "K3_high_volume_not_worse": mdh <= 0.0,
                "K4_alpha_spread_at_most_0.40": spread <= 0.40,
                "K5_mechanism_3_of_4": mech >= 3}
        ok = all(crit.values())
        decisions[variant] = {"criteria": crit, "accepted": ok, "wins": wins,
                              "mean_dRMSE": mdr, "mean_dMAE": mdm,
                              "mean_dhighvol": mdh, "alphas": alphas,
                              "alpha_spread": spread}
        log(f"  wins {wins}/4   alphas {alphas} (spread {spread:.2f})")
        for k, v in crit.items():
            log(f"    {'PASS' if v else 'FAIL'}  {k}")
        log(f"    -> {'CANDIDATE FOR PROMOTION' if ok else 'REJECTED'}")

    log(f"\n  mechanism: item model beat the bottom-up sum on {mech}/4 windows")
    OUT_JSON.write_text(json.dumps({"windows": rows, "decisions": decisions,
                                    "mechanism_wins": mech},
                                   indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {OUT_JSON.name}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
