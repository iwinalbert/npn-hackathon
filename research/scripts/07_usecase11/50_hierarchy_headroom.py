
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

OUT_JSON = config.ARTIFACTS_DIR / "uc11_hierarchy_headroom.json"
OUT_CSV = config.ARTIFACTS_DIR / "uc11_hierarchy_levels.csv"


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load_meta() -> pd.DataFrame:
    meta = pd.read_csv(config.SALES_EVAL_CSV,
                       usecols=["id", "item_id", "dept_id", "cat_id",
                                "store_id", "state_id"])
    if len(meta) != config.N_SERIES:
        raise SystemExit(f"expected {config.N_SERIES} series, got {len(meta)}")
    return meta


def build_levels(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    def codes(*cols):
        if len(cols) == 1:
            key = meta[cols[0]].astype(str)
        else:
            key = meta[cols[0]].astype(str)
            for c in cols[1:]:
                key = key + "|" + meta[c].astype(str)
        return pd.factorize(key)[0].astype(np.int32)

    return {
        "L1_total": np.zeros(len(meta), dtype=np.int32),
        "L2_state": codes("state_id"),
        "L3_store": codes("store_id"),
        "L4_cat": codes("cat_id"),
        "L5_dept": codes("dept_id"),
        "L6_state_cat": codes("state_id", "cat_id"),
        "L7_state_dept": codes("state_id", "dept_id"),
        "L8_store_cat": codes("store_id", "cat_id"),
        "L9_store_dept": codes("store_id", "dept_id"),
        "L10_item": codes("item_id"),
        "L11_item_state": codes("item_id", "state_id"),
    }


def main():
    t0 = time.time()
    banner("USE CASE 11 — HIERARCHICAL HEADROOM DIAGNOSTIC (read-only)")

    meta = load_meta()
    log(f"  hierarchy loaded: {len(meta)} series, "
        f"{meta.item_id.nunique()} items, {meta.store_id.nunique()} stores, "
        f"{meta.dept_id.nunique()} depts, {meta.cat_id.nunique()} cats, "
        f"{meta.state_id.nunique()} states")

    pred = pd.read_csv(PRED_FILE)
    log(f"  predictions   : {PRED_FILE.name}  ({len(pred)} rows)")

    days = np.sort(pred["target_day_idx"].unique())
    day_pos = {d: i for i, d in enumerate(days)}
    n_s, n_d = config.N_SERIES, len(days)
    Y = np.zeros((n_s, n_d), dtype=np.float64)
    P = np.zeros((n_s, n_d), dtype=np.float64)
    Y[pred["series_idx"].to_numpy(),
      pred["target_day_idx"].map(day_pos).to_numpy()] = pred["y_true"].to_numpy()
    P[pred["series_idx"].to_numpy(),
      pred["target_day_idx"].map(day_pos).to_numpy()] = pred["y_pred"].to_numpy()

    base_rmse = metrics.rmse(Y.ravel(), P.ravel())
    base_mae = metrics.mae(Y.ravel(), P.ravel())
    log(f"  base (bottom) : RMSE {base_rmse:.4f}   MAE {base_mae:.4f}   "
        f"n = {Y.size}")
    log(f"  window        : {n_d} days, d_{days.min()+1}..d_{days.max()+1}")

    E = Y - P
    total_sq = float((E ** 2).sum())

    levels = build_levels(meta)
    rows = []

    banner("PER-LEVEL: coherence, aggregate accuracy, decomposition, oracles")
    log(f"  {'level':<16}{'groups':>8}{'aggWAPE':>9}{'common%':>9}"
        f"{'O1 equal':>10}{'O2 prop':>10}{'O3 lsq':>10}")

    for name, g in levels.items():
        n_g = int(g.max()) + 1
        order = np.argsort(g, kind="stable")
        sizes = np.bincount(g, minlength=n_g)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

        A = np.zeros((n_g, n_d))
        F = np.zeros((n_g, n_d))
        np.add.at(A, g, Y)
        np.add.at(F, g, P)
        D = A - F

        agg = metrics.evaluate(A.ravel(), F.ravel())

        common_sq = float(((D ** 2) / sizes[:, None]).sum())
        common_share = common_sq / total_sq * 100.0

        P1 = P + (D / sizes[:, None])[g]
        o1 = metrics.rmse(Y.ravel(), np.clip(P1, 0, None).ravel())

        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(F > 1e-9, A / F, 1.0)
        P2 = P * scale[g]
        o2 = metrics.rmse(Y.ravel(), np.clip(P2, 0, None).ravel())

        denom = (D ** 2).sum(axis=1)
        num = np.zeros(n_s)
        for gi in range(n_g):
            idx = order[starts[gi]:starts[gi] + sizes[gi]]
            num[idx] = E[idx] @ D[gi]
        w = np.where(denom[g] > 1e-9, num / np.where(denom[g] > 1e-9, denom[g], 1.0), 0.0)
        P3 = P + w[:, None] * D[g]
        o3 = metrics.rmse(Y.ravel(), np.clip(P3, 0, None).ravel())

        rows.append({
            "level": name, "n_groups": n_g, "mean_group_size": float(sizes.mean()),
            "agg_RMSE": agg["RMSE"], "agg_MAE": agg["MAE"],
            "agg_WAPE": agg["WAPE"], "agg_bias": agg["bias"],
            "common_error_share_pct": common_share,
            "oracle1_equal_RMSE": o1, "oracle1_gain": o1 - base_rmse,
            "oracle2_prop_RMSE": o2, "oracle2_gain": o2 - base_rmse,
            "oracle3_lsq_RMSE": o3, "oracle3_gain": o3 - base_rmse,
        })
        log(f"  {name:<16}{n_g:>8}{agg['WAPE']:>9.4f}{common_share:>9.2f}"
            f"{o1 - base_rmse:>+10.4f}{o2 - base_rmse:>+10.4f}{o3 - base_rmse:>+10.4f}")

    R = pd.DataFrame(rows)

    banner("READING THE TABLE")
    best = R.loc[R.oracle3_gain.idxmin()]
    log(f"  Strongest oracle of any level: {best.level}  "
        f"{best.oracle3_gain:+.4f} RMSE (in-sample-fitted shares, unachievable).")
    log(f"  Its common-error share is {best.common_error_share_pct:.2f}% of all "
        "squared error;")
    log("  the remaining share is idiosyncratic and provably invisible to any")
    log("  aggregate-level information whatsoever.")
    log("")
    log("  ORACLE 2 (proportional top-down) is the realistic template: a real")
    log("  reconciliation would replace the TRUE aggregate with a FORECAST one,")
    log("  so its gain is bounded above by the oracle and shrinks with the")
    log("  aggregate model's own error.")

    banner("WHAT AN AGGREGATE MODEL WOULD HAVE TO ACHIEVE")
    log("  A top-down correction only helps if the aggregate forecast is closer")
    log("  to the truth than the bottom-up sum already is. Bottom-up aggregate")
    log("  WAPE per level is in the table above; for reference the bottom level")
    log(f"  itself is WAPE {metrics.wape(Y.ravel(), P.ravel()):.4f}.")
    for _, r in R.iterrows():
        log(f"    {r.level:<16} bottom-up aggregate WAPE {r.agg_WAPE:.4f}  "
            f"(accuracy {100*(1-r.agg_WAPE):.1f}%)")

    R.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "source_predictions": PRED_FILE.name,
        "note": ("Diagnostic run on the Experiment #76 blend (RMSE 2.0920), the "
                 "closest stored artefact to the shipped w=0.60 champion "
                 "(RMSE 2.0929). Conclusions are insensitive to that 0.0009."),
        "base_RMSE": base_rmse, "base_MAE": base_mae,
        "n_predictions": int(Y.size), "n_days": int(n_d),
        "levels": rows,
        "best_level_by_oracle3": str(best.level),
        "best_oracle3_gain": float(best.oracle3_gain),
    }, indent=2), encoding="utf-8")
    log(f"\n  wrote {OUT_CSV.name} and {OUT_JSON.name}")
    log(f"  wall time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
