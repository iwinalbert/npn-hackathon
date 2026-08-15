"""
EXPERIMENT #80 (PROBE) — CAN AN ITEM-LEVEL MODEL BEAT THE BOTTOM-UP SUM, AND
DOES RECONCILING IT DOWNWARD IMPROVE THE STORE-ITEM FORECAST?

RUN ENTIRELY ON THE INNER WINDOW (origin d_1885, targets d_1886..d_1913).

WHY THE INNER WINDOW
--------------------
This is a go/no-go probe. Deciding whether to proceed on the basis of the
primary window would be selection on the evaluation data, and every weight or
threshold it produced would be contaminated. The inner window lies strictly
before the primary window's targets and before the final forecast origin, and it
is the same window Experiment #77 used to choose the blend weight. If this probe
passes, the full four-window validation follows in a separate script with every
constant fixed here carried across unchanged.

WHAT STEPS 1-3 ESTABLISHED
--------------------------
  * Bottom-up sums are already coherent; no incoherence exists to repair.
  * At every coarse level the recoverable error is negligible: knowing the TRUE
    store x dept daily total, perfectly, is worth only -0.0221 RMSE, and the
    chain total is worth -0.0007.
  * At item level (10 stores summed) the common component is 16.8% of all
    squared error and the true-aggregate oracle is worth -0.2272.
  * The algebra that converts that into a required aggregate accuracy assumes
    the aggregate model's error is uncorrelated with the bottom-up sum's error.
    That assumption is badly wrong in practice — both miss the same demand
    shocks — so it produces optimistic numbers and CANNOT be used as a
    prediction. Only a real aggregate forecast settles it.

WHAT IS MEASURED HERE
---------------------
  1. Item-level accuracy of the bottom-up sum vs a dedicated item-level model
     vs two naive aggregate benchmarks. (Mechanism check: if the item model
     cannot beat the bottom-up sum, the direction is dead.)
  2. The correlation between the two aggregate error series — the quantity that
     governs how much of the oracle survives.
  3. The exact bottom-level RMSE of the reconciled forecast

         P'_s = clip( P_s + alpha * (P_s / F) * (Ahat - F), 0, None )

     over a grid of alpha, plus MAE, high-volume RMSE and volume-decile
     behaviour at the best alpha.

PRE-REGISTERED PROCEED CRITERIA (fixed before the first run)
------------------------------------------------------------
  G1  the item-level model beats the bottom-up sum on item-level RMSE
  G2  the best-alpha reconciled bottom-level RMSE improves by at least -0.005
  G3  that alpha lies strictly inside the grid (not pinned at an endpoint,
      which would mean the optimum has not been bracketed)
  G4  the leakage corruption test on the aggregate feature builder passes

Failing any of them stops the direction here rather than spending four windows
on it.

    python scripts/07_usecase11/53_exp80_item_level_probe.py
"""

from __future__ import annotations

import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics, optimize
from pipeline.aggregate_level import (AggregateLevel, AggFeatureBuilder,
                                      AGG_FEATURES, agg_categorical_for)
from pipeline.champion_blend import champion_predictions
from pipeline.data_loader import M5Data

INNER_ORIGIN = config.VALIDATION_ORIGIN_IDX - config.HORIZON      # 1884 -> d_1885
N_AGG_ORIGINS = 30
ALPHAS = np.round(np.arange(0.0, 1.01, 0.05), 2)

OUT_JSON = config.ARTIFACTS_DIR / "uc11_exp80_probe.json"


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ----------------------------------------------------------------------
def train_agg_model(agg, origin_idx, *, objective="tweedie", power=1.1,
                    seed=config.RANDOM_SEED, n_origins=N_AGG_ORIGINS,
                    n_estimators=optimize.N_ESTIMATORS):
    """Direct 28-day model on the aggregate panel. Same harness philosophy as
    the bottom-level champion: fixed origins, 28-day blocks, no target may fall
    on or after the first evaluation day."""
    fb = AggFeatureBuilder(agg)
    newest = origin_idx - config.HORIZON
    origins = sorted(newest - i * config.HORIZON for i in range(n_origins))
    origins = [o for o in origins if o >= 400]

    frames = []
    for o in origins:
        f = fb.build_origin_frame(o, horizon=config.HORIZON, include_target=True)
        if int(f["target_day_idx"].max()) > origin_idx:
            raise AssertionError(
                f"LEAKAGE: aggregate training origin d_{o+1} targets day "
                f"{int(f['target_day_idx'].max())} beyond cutoff {origin_idx}")
        frames.append(f)
    train = pd.concat(frames, ignore_index=True)
    del frames
    if train["sales"].isna().any():
        raise AssertionError("aggregate training frame has unknown targets")

    params = dict(optimize.BASE_PARAMS)
    params.update({"objective": objective, "seed": seed, "bagging_seed": seed,
                   "feature_fraction_seed": seed})
    if objective == "tweedie":
        params["tweedie_variance_power"] = power
    else:
        params.pop("tweedie_variance_power", None)

    ds = lgb.Dataset(train[AGG_FEATURES].to_numpy(np.float32),
                     label=train["sales"].to_numpy(np.float32),
                     feature_name=list(AGG_FEATURES),
                     categorical_feature=agg_categorical_for(AGG_FEATURES),
                     free_raw_data=True)
    t0 = time.time()
    booster = lgb.train(params, ds, num_boost_round=n_estimators,
                        callbacks=[lgb.log_evaluation(period=0)])
    secs = round(time.time() - t0, 1)
    n_rows = len(train)
    del train, ds
    gc.collect()

    valid = fb.build_origin_frame(origin_idx, horizon=config.HORIZON,
                                  include_target=True)
    p = np.clip(booster.predict(valid[AGG_FEATURES].to_numpy(np.float32)), 0, None)
    info = {"objective": objective, "power": power if objective == "tweedie" else None,
            "training_rows": int(n_rows), "training_origins": len(origins),
            "training_origins_span": f"d_{origins[0]+1}..d_{origins[-1]+1}",
            "training_seconds": secs}
    del booster
    gc.collect()
    return p, valid, info


# ----------------------------------------------------------------------
def leakage_corruption_test(data, origin_idx):
    """
    Rebuild the aggregate features from a panel whose post-origin sales have
    been replaced with 9999 and require every feature to come back identical.

    The AggregateLevel object is reconstructed from the corrupted panel rather
    than patched, so nothing precomputed from the clean matrix can mask a leak.
    A companion check confirms the TARGET did change, so the test cannot pass
    vacuously.
    """
    clean = AggregateLevel(data, "item")
    f_clean = AggFeatureBuilder(clean).build_origin_frame(
        origin_idx, horizon=config.HORIZON, include_target=True)

    d2 = copy.copy(data)
    corrupted = data.sales_wide.copy()
    corrupted[:, origin_idx + 1:] = 9999
    d2.sales_wide = corrupted
    dirty = AggregateLevel(d2, "item")
    f_dirty = AggFeatureBuilder(dirty).build_origin_frame(
        origin_idx, horizon=config.HORIZON, include_target=True)

    mismatches = [c for c in AGG_FEATURES
                  if not np.array_equal(f_clean[c].to_numpy(),
                                        f_dirty[c].to_numpy(), equal_nan=True)]
    target_changed = not np.array_equal(f_clean["sales"].to_numpy(),
                                        f_dirty["sales"].to_numpy(),
                                        equal_nan=True)
    del d2, corrupted, dirty, f_clean, f_dirty
    gc.collect()
    return {"features_checked": len(AGG_FEATURES),
            "features_that_changed": mismatches,
            "target_changed_as_expected": bool(target_changed),
            "passed": (len(mismatches) == 0) and bool(target_changed)}


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    banner("EXPERIMENT #80 (PROBE) — ITEM-LEVEL MODEL + MIDDLE-OUT RECONCILIATION")
    log(f"  inner window: origin d_{INNER_ORIGIN+1}, "
        f"targets d_{INNER_ORIGIN+2}..d_{INNER_ORIGIN+1+config.HORIZON}")
    log("  PRE-REGISTERED: G1 item model beats bottom-up | G2 dRMSE <= -0.005 |")
    log("                  G3 alpha* interior | G4 corruption test passes")

    log("\n  loading data...")
    data = M5Data()

    # ---------------------------------------------------------------
    banner("STEP A — leakage corruption test on the aggregate builder")
    ck = leakage_corruption_test(data, INNER_ORIGIN)
    log(f"  features checked      : {ck['features_checked']}")
    log(f"  features that changed : {ck['features_that_changed'] or 'none'}")
    log(f"  target changed        : {ck['target_changed_as_expected']} (must be True)")
    log(f"  -> {'PASS' if ck['passed'] else 'FAIL'}")
    if not ck["passed"]:
        raise SystemExit("STOP: aggregate feature builder is not leakage-safe")

    # ---------------------------------------------------------------
    banner("STEP B — the champion's bottom-level forecast on this window")
    champ = champion_predictions(data, INNER_ORIGIN)
    y = champ["y"]
    P = champ["blend"]
    base_rmse = metrics.rmse(y, P)
    base_mae = metrics.mae(y, P)
    log(f"  member A direct     RMSE {metrics.rmse(y, champ['direct']):.4f}")
    log(f"  member B' recursive RMSE {metrics.rmse(y, champ['recursive']):.4f}")
    log(f"  BLEND w=0.60        RMSE {base_rmse:.4f}   MAE {base_mae:.4f}")

    # reshape bottom-level to (series x day)
    days = np.sort(np.unique(champ["target_day_idx"]))
    pos = {d: i for i, d in enumerate(days)}
    n_d = len(days)
    di = np.array([pos[d] for d in champ["target_day_idx"]])
    si = champ["series_idx"].astype(np.int64)
    Ym = np.zeros((config.N_SERIES, n_d))
    Pm = np.zeros((config.N_SERIES, n_d))
    Ym[si, di] = y
    Pm[si, di] = P

    # ---------------------------------------------------------------
    banner("STEP C — item-level: bottom-up sum vs a dedicated model")
    agg = AggregateLevel(data, "item")
    log(f"  {agg.describe()}")
    g = agg.group_of_series
    n_g = agg.n_groups
    A = np.zeros((n_g, n_d))
    F = np.zeros((n_g, n_d))
    np.add.at(A, g, Ym)
    np.add.at(F, g, Pm)

    log(f"\n  bottom-up sum          RMSE {metrics.rmse(A.ravel(), F.ravel()):8.4f}"
        f"   WAPE {metrics.wape(A.ravel(), F.ravel()):.4f}")

    # naive aggregate benchmarks, for context
    naive28 = np.repeat(agg.sales_wide[:, INNER_ORIGIN - 27:INNER_ORIGIN + 1]
                        .mean(axis=1)[:, None], n_d, axis=1)
    log(f"  naive rolling-mean-28  RMSE {metrics.rmse(A.ravel(), naive28.ravel()):8.4f}"
        f"   WAPE {metrics.wape(A.ravel(), naive28.ravel()):.4f}")

    candidates = {}
    for obj, power in [("tweedie", 1.1), ("tweedie", 1.3), ("regression", None)]:
        tag = f"{obj}" + (f"_{power}" if power else "_l2")
        p_agg, valid, info = train_agg_model(agg, INNER_ORIGIN,
                                             objective=obj, power=power or 1.1)
        Ahat = np.zeros((n_g, n_d))
        Ahat[valid["group_idx"].to_numpy(),
             [pos[d] for d in valid["target_day_idx"].to_numpy()]] = p_agg
        r = metrics.rmse(A.ravel(), Ahat.ravel())
        w = metrics.wape(A.ravel(), Ahat.ravel())
        rho = float(np.corrcoef((A - F).ravel(), (A - Ahat).ravel())[0, 1])
        candidates[tag] = {"Ahat": Ahat, "RMSE": r, "WAPE": w,
                           "rho_with_bottomup_error": rho, **info}
        log(f"  item model {tag:<16} RMSE {r:8.4f}   WAPE {w:.4f}   "
            f"rho(err_BU, err_agg) {rho:.4f}   ({info['training_seconds']}s)")
        del p_agg, valid
        gc.collect()

    best_tag = min(candidates, key=lambda k: candidates[k]["RMSE"])
    best = candidates[best_tag]
    log(f"\n  best item-level model: {best_tag}  RMSE {best['RMSE']:.4f}  "
        f"vs bottom-up {metrics.rmse(A.ravel(), F.ravel()):.4f}")
    G1 = best["RMSE"] < metrics.rmse(A.ravel(), F.ravel())
    log(f"  G1 (item model beats bottom-up): {'PASS' if G1 else 'FAIL'}")

    # ---------------------------------------------------------------
    banner("STEP D — reconcile downward and score at the BOTTOM level")
    log("  P'_s = clip( P_s + alpha * (P_s / F) * (Ahat - F), 0, None )")
    log(f"\n  {'alpha':>7}{'RMSE':>10}{'dRMSE':>10}{'MAE':>10}{'dMAE':>10}")

    hist = data.sales_wide[:, :INNER_ORIGIN + 1].mean(axis=1)
    high_series = hist > 3.0
    highm = np.tile(high_series[:, None], (1, n_d))

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(F[g] > 1e-9, Pm / np.where(F[g] > 1e-9, F[g], 1.0), 0.0)

    rows = []
    for tag, cand in candidates.items():
        D = cand["Ahat"] - F
        for a in ALPHAS:
            Pr = np.clip(Pm + a * share * D[g], 0, None)
            r = metrics.rmse(Ym.ravel(), Pr.ravel())
            m = metrics.mae(Ym.ravel(), Pr.ravel())
            rows.append({"model": tag, "alpha": float(a), "RMSE": r, "MAE": m,
                         "dRMSE": r - base_rmse, "dMAE": m - base_mae,
                         "highvol_RMSE": metrics.rmse(Ym[highm], Pr[highm])})
            if tag == best_tag and a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
                log(f"  {a:>7.2f}{r:>10.4f}{r-base_rmse:>+10.4f}"
                    f"{m:>10.4f}{m-base_mae:>+10.4f}")

    R = pd.DataFrame(rows)
    log(f"\n  best alpha per candidate model:")
    summary = {}
    for tag in candidates:
        sub = R[R.model == tag]
        b = sub.loc[sub.RMSE.idxmin()]
        summary[tag] = {"alpha_star": float(b.alpha), "RMSE": float(b.RMSE),
                        "dRMSE": float(b.dRMSE), "MAE": float(b.MAE),
                        "dMAE": float(b.dMAE),
                        "highvol_RMSE": float(b.highvol_RMSE),
                        "agg_RMSE": candidates[tag]["RMSE"],
                        "rho_with_bottomup_error":
                            candidates[tag]["rho_with_bottomup_error"]}
        log(f"    {tag:<16} alpha*={b.alpha:.2f}  RMSE {b.RMSE:.4f} "
            f"({b.dRMSE:+.4f})   MAE {b.MAE:.4f} ({b.dMAE:+.4f})   "
            f"high-vol {b.highvol_RMSE:.4f}")

    bestrow = R.loc[R.RMSE.idxmin()]
    base_high = metrics.rmse(Ym[highm], Pm[highm])
    log(f"\n  champion high-volume RMSE {base_high:.4f}  ->  "
        f"{bestrow.highvol_RMSE:.4f}  ({bestrow.highvol_RMSE - base_high:+.4f})")

    G2 = bool(bestrow.dRMSE <= -0.005)
    G3 = bool(0.0 < bestrow.alpha < 1.0)
    log(f"\n  G2 (dRMSE <= -0.005) : {'PASS' if G2 else 'FAIL'}  "
        f"({bestrow.dRMSE:+.4f})")
    log(f"  G3 (alpha* interior) : {'PASS' if G3 else 'FAIL'}  "
        f"(alpha*={bestrow.alpha:.2f})")
    log(f"  G4 (corruption test) : PASS")

    proceed = G1 and G2 and G3 and ck["passed"]
    banner(f"PROBE VERDICT: {'PROCEED to four-window validation' if proceed else 'STOP'}")

    R.to_csv(config.ARTIFACTS_DIR / "uc11_exp80_alpha_curve.csv", index=False)
    OUT_JSON.write_text(json.dumps({
        "window": {"origin_idx": INNER_ORIGIN,
                   "origin_day": f"d_{INNER_ORIGIN+1}",
                   "targets": f"d_{INNER_ORIGIN+2}..d_{INNER_ORIGIN+1+config.HORIZON}"},
        "leakage_corruption_test": ck,
        "champion_blend": {"RMSE": base_rmse, "MAE": base_mae,
                           "high_volume_RMSE": float(base_high)},
        "item_level": {
            "bottom_up_RMSE": metrics.rmse(A.ravel(), F.ravel()),
            "bottom_up_WAPE": metrics.wape(A.ravel(), F.ravel()),
            "naive_rm28_RMSE": metrics.rmse(A.ravel(), naive28.ravel()),
            "models": {k: {kk: vv for kk, vv in v.items() if kk != "Ahat"}
                       for k, v in candidates.items()},
            "best": best_tag,
        },
        "reconciliation": summary,
        "best_overall": {"model": str(bestrow.model), "alpha": float(bestrow.alpha),
                         "RMSE": float(bestrow.RMSE), "dRMSE": float(bestrow.dRMSE),
                         "MAE": float(bestrow.MAE), "dMAE": float(bestrow.dMAE)},
        "criteria": {"G1_item_model_beats_bottom_up": bool(G1),
                     "G2_dRMSE_at_most_-0.005": G2,
                     "G3_alpha_interior": G3,
                     "G4_corruption_test": ck["passed"]},
        "proceed": bool(proceed),
    }, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {OUT_JSON.name} and uc11_exp80_alpha_curve.csv")
    log(f"  wall time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
