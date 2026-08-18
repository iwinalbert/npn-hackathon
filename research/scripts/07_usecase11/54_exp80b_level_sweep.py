
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics
from pipeline.aggregate_level import AggregateLevel
from pipeline.champion_blend import champion_predictions
from pipeline.data_loader import M5Data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_probe = import_module("53_exp80_item_level_probe")
train_agg_model = _probe.train_agg_model

INNER_ORIGIN = config.VALIDATION_ORIGIN_IDX - config.HORIZON
LEVELS = ["store_dept", "item", "item_state"]
ALPHAS = np.round(np.arange(0.0, 1.01, 0.05), 2)
OUT_JSON = config.ARTIFACTS_DIR / "uc11_exp80b_level_sweep.json"


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def reconcile(Pm, Ahat, F, g, alpha):
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(F[g] > 1e-9, Pm / np.where(F[g] > 1e-9, F[g], 1.0), 0.0)
    return np.clip(Pm + alpha * share * (Ahat - F)[g], 0, None)


def aggregate(M, g, n_g):
    out = np.zeros((n_g, M.shape[1]))
    np.add.at(out, g, M)
    return out


def main():
    t0 = time.time()
    banner("EXPERIMENT #80b — LEVEL SWEEP + NEGATIVE CONTROLS (inner window)")
    log(f"  origin d_{INNER_ORIGIN+1}, targets d_{INNER_ORIGIN+2}.."
        f"d_{INNER_ORIGIN+1+config.HORIZON}")

    data = M5Data()
    champ = champion_predictions(data, INNER_ORIGIN)

    days = np.sort(np.unique(champ["target_day_idx"]))
    pos = {d: i for i, d in enumerate(days)}
    n_d = len(days)
    di = np.array([pos[d] for d in champ["target_day_idx"]])
    si = champ["series_idx"].astype(np.int64)
    Ym = np.zeros((config.N_SERIES, n_d))
    Pm = np.zeros((config.N_SERIES, n_d))
    Dm = np.zeros((config.N_SERIES, n_d))
    Ym[si, di] = champ["y"]
    Pm[si, di] = champ["blend"]
    Dm[si, di] = champ["direct"]

    base_rmse = metrics.rmse(Ym.ravel(), Ym.ravel() * 0 + Pm.ravel())
    base_mae = metrics.mae(Ym.ravel(), Pm.ravel())
    hist = data.sales_wide[:, :INNER_ORIGIN + 1].mean(axis=1)
    highm = np.tile((hist > 3.0)[:, None], (1, n_d))
    base_high = metrics.rmse(Ym[highm], Pm[highm])
    log(f"  champion blend: RMSE {base_rmse:.4f}  MAE {base_mae:.4f}  "
        f"high-vol {base_high:.4f}")

    results, store = [], {}

    banner("PART 1 — one level at a time")
    log(f"  {'level':<12}{'groups':>8}{'aggBU':>9}{'aggMdl':>9}{'alpha*':>8}"
        f"{'RMSE':>10}{'dRMSE':>9}{'MAE':>9}{'dMAE':>9}{'highvol':>9}")

    for lvl in LEVELS:
        agg = AggregateLevel(data, lvl)
        g, n_g = agg.group_of_series, agg.n_groups
        A = aggregate(Ym, g, n_g)
        F = aggregate(Pm, g, n_g)
        p_agg, valid, info = train_agg_model(agg, INNER_ORIGIN,
                                             objective="regression")
        Ahat = np.zeros((n_g, n_d))
        Ahat[valid["group_idx"].to_numpy(),
             [pos[d] for d in valid["target_day_idx"].to_numpy()]] = p_agg
        bu_r = metrics.rmse(A.ravel(), F.ravel())
        md_r = metrics.rmse(A.ravel(), Ahat.ravel())

        best = None
        for a in ALPHAS:
            Pr = reconcile(Pm, Ahat, F, g, a)
            r = metrics.rmse(Ym.ravel(), Pr.ravel())
            rec = {"level": lvl, "alpha": float(a), "RMSE": r,
                   "MAE": metrics.mae(Ym.ravel(), Pr.ravel()),
                   "highvol_RMSE": metrics.rmse(Ym[highm], Pr[highm]),
                   "agg_bottom_up_RMSE": bu_r, "agg_model_RMSE": md_r,
                   "n_groups": n_g}
            rec["dRMSE"] = r - base_rmse
            rec["dMAE"] = rec["MAE"] - base_mae
            results.append(rec)
            if best is None or r < best["RMSE"]:
                best = rec
        store[lvl] = {"Ahat": Ahat, "F": F, "g": g, "n_g": n_g, "best": best,
                      "agg_info": info}
        log(f"  {lvl:<12}{n_g:>8}{bu_r:>9.3f}{md_r:>9.3f}{best['alpha']:>8.2f}"
            f"{best['RMSE']:>10.4f}{best['dRMSE']:>+9.4f}{best['MAE']:>9.4f}"
            f"{best['dMAE']:>+9.4f}{best['highvol_RMSE']:>9.4f}")
        del agg, p_agg, valid
        gc.collect()

    best_level = min(store, key=lambda k: store[k]["best"]["RMSE"])
    log(f"\n  best single level: {best_level}  "
        f"(alpha*={store[best_level]['best']['alpha']:.2f}, "
        f"dRMSE {store[best_level]['best']['dRMSE']:+.4f})")

    banner("PART 2 — sequential two-level reconciliation")
    seq = []
    for first in LEVELS:
        for second in LEVELS:
            if first == second:
                continue
            f1 = store[first]
            P1 = reconcile(Pm, f1["Ahat"], f1["F"], f1["g"], f1["best"]["alpha"])
            f2 = store[second]
            F2 = aggregate(P1, f2["g"], f2["n_g"])
            bestr = None
            for a in ALPHAS:
                P2 = reconcile(P1, f2["Ahat"], F2, f2["g"], a)
                r = metrics.rmse(Ym.ravel(), P2.ravel())
                if bestr is None or r < bestr["RMSE"]:
                    bestr = {"first": first, "second": second, "alpha2": float(a),
                             "RMSE": r, "MAE": metrics.mae(Ym.ravel(), P2.ravel()),
                             "highvol_RMSE": metrics.rmse(Ym[highm], P2[highm])}
            bestr["dRMSE"] = bestr["RMSE"] - base_rmse
            bestr["dMAE"] = bestr["MAE"] - base_mae
            seq.append(bestr)
            log(f"  {first:<12} then {second:<12} alpha2={bestr['alpha2']:.2f}  "
                f"RMSE {bestr['RMSE']:.4f} ({bestr['dRMSE']:+.4f})  "
                f"MAE {bestr['MAE']:.4f} ({bestr['dMAE']:+.4f})")
            del P1, F2
            gc.collect()
    best_seq = min(seq, key=lambda r: r["RMSE"])
    log(f"\n  best sequence: {best_seq['first']} then {best_seq['second']}  "
        f"dRMSE {best_seq['dRMSE']:+.4f}  vs best single "
        f"{store[best_level]['best']['dRMSE']:+.4f}")

    banner("PART 3 — negative controls at the best level")
    b = store[best_level]
    g, n_g, F = b["g"], b["n_g"], b["F"]
    A = aggregate(Ym, g, n_g)
    controls = {}

    c_star = float((Ym * Pm).sum() / (Pm * Pm).sum())
    Ahat_c1 = c_star * F
    agg_obj = AggregateLevel(data, best_level)
    naive = agg_obj.sales_wide[:, INNER_ORIGIN - 27:INNER_ORIGIN + 1].mean(axis=1)
    Ahat_c2 = np.repeat(naive[:, None], n_d, axis=1)
    Ahat_c3 = aggregate(Dm, g, n_g)

    for tag, Ah, why in [
            ("C1_global_rescale", Ahat_c1, f"Ahat = {c_star:.4f} x F"),
            ("C2_naive_aggregate", Ahat_c2, "Ahat = trailing 28-day aggregate mean"),
            ("C3_direct_member_sum", Ahat_c3, "Ahat = bottom-up sum of member A")]:
        best_c = None
        for a in ALPHAS:
            Pr = reconcile(Pm, Ah, F, g, a)
            r = metrics.rmse(Ym.ravel(), Pr.ravel())
            if best_c is None or r < best_c["RMSE"]:
                best_c = {"alpha": float(a), "RMSE": r,
                          "MAE": metrics.mae(Ym.ravel(), Pr.ravel())}
        best_c["dRMSE"] = best_c["RMSE"] - base_rmse
        best_c["agg_RMSE"] = metrics.rmse(A.ravel(), Ah.ravel())
        best_c["why"] = why
        controls[tag] = best_c
        log(f"  {tag:<22} {why:<42} alpha*={best_c['alpha']:.2f}  "
            f"dRMSE {best_c['dRMSE']:+.4f}")

    real = b["best"]["dRMSE"]
    log(f"\n  real aggregate model                                        "
        f"alpha*={b['best']['alpha']:.2f}  dRMSE {real:+.4f}")
    worst_control = min(c["dRMSE"] for c in controls.values())
    log(f"  gain NOT explained by the strongest control: "
        f"{real - worst_control:+.4f} of {real:+.4f} "
        f"({100 * (real - worst_control) / real:.0f}%)")

    banner("SELECTED CONFIGURATION (carried to the four-window validation)")
    log(f"  level           : {best_level}")
    log(f"  aggregate model : LightGBM L2, {len(_probe.AGG_FEATURES)} features, "
        f"{b['agg_info']['training_origins']} origins")
    log(f"  alpha           : {b['best']['alpha']:.2f}   (inner-window optimum)")
    log(f"  expected effect : dRMSE {b['best']['dRMSE']:+.4f}  "
        f"dMAE {b['best']['dMAE']:+.4f} on this window")

    pd.DataFrame(results).to_csv(
        config.ARTIFACTS_DIR / "uc11_exp80b_level_alpha_curve.csv", index=False)
    OUT_JSON.write_text(json.dumps({
        "window": f"origin d_{INNER_ORIGIN+1}",
        "champion": {"RMSE": base_rmse, "MAE": base_mae, "high_volume_RMSE": base_high},
        "per_level_best": {k: v["best"] for k, v in store.items()},
        "sequential": seq,
        "negative_controls": controls,
        "selected": {"level": best_level,
                     "alpha": store[best_level]["best"]["alpha"],
                     "objective": "regression (L2)"},
    }, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {OUT_JSON.name}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
