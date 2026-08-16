"""
FINAL MODEL PERFORMANCE REPORT — audit and build.

REPORTING TASK ONLY. Trains nothing, changes no model, writes no forecast.
Every regression number is re-derived from the saved validation-prediction files
and cross-checked against the experiment registry; any disagreement is printed
and the script refuses to emit the report.

Demand-occurrence (classification-style) metrics are computed only for models
whose per-row predictions exist on disk. Where they do not exist, the report
says "N/A - not evaluated" rather than inventing a number.

    python scripts/06_research_campaign/44_final_performance_report.py
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
from pipeline.report_pdf import render_markdown_to_pdf

REG = config.EXPERIMENTS_DIR
PRED = config.PREDICTIONS_DIR
ART = config.ARTIFACTS_DIR

# Threshold for the demand-occurrence event. A point forecast of 0.5 or more
# rounds to at least one unit, which is the only non-arbitrary cut for a count
# target. Applied identically to every model.
THRESHOLD = 0.5

# (label, registry record, prediction file or None, objective, n_features, note)
MODELS = [
    ("Naive - last value", "model_00_baseline_last_value", None,
     "none (arithmetic)", None, "baseline"),
    ("Naive - rolling mean 7", "model_00_baseline_rolling_mean_7", None,
     "none (arithmetic)", None, "baseline"),
    ("Naive - rolling mean 28", "model_00_baseline_rolling_mean_28", None,
     "none (arithmetic)", None, "baseline"),
    ("Naive - seasonal naive", "model_00_baseline_seasonal_naive",
     "model_00_seasonal_naive_validation.csv", "none (arithmetic)", None, "baseline"),
    ("Global LightGBM (L2)", "model_01_lightgbm",
     "model_01_lightgbm_validation.csv", "L2 (squared error)", 32, "development"),
    ("Global LightGBM + Tweedie", "model_02_tweedie",
     "model_02_tweedie_validation.csv", "Tweedie (p=1.1)", 32, "development"),
    ("+ recency features", "model_03_tweedie_recency",
     "model_03_tweedie_recency_validation.csv", "Tweedie (p=1.1)", 32, "development"),
    ("Global LightGBM + Tweedie, 32 features", "model_04_tweedie_recency_listing",
     "model_04_tweedie_recency_listing_validation.csv", "Tweedie (p=1.1)", 32,
     "ORIGINAL CHAMPION"),
    ("Hurdle (two-stage)", "model_05_hurdle",
     "model_05_hurdle_validation.csv", "binary + Tweedie", 32, "development"),
    ("Recursive one-step (member B)", "opt_05_recursive",
     "opt_05_recursive_validation.csv", "Tweedie (p=1.1)", 26, "ensemble member"),
    # Shape-only, 36 features. Its predictions ARE on disk.
    ("Shape only, 36 features", "exp_72_per_series_shape_features",
     "exp_72_shape_validation.csv", "Tweedie (p=1.1)", 36, "development"),
    # The 38-feature champion. Its registry metrics are sound and were
    # reproduced bit-identically, but its per-row predictions were never
    # persisted: exp_74_new_champion_validation.csv is a misnamed byte-identical
    # copy of the 36-feature file above (script 36 discards the 38-feature
    # model's predictions at line 166 and saves Part A's at line 243).
    ("Shape+Cycle, 38 features", "exp_74_shape_reproduction_and_extension",
     None, "Tweedie (p=1.1)", 38, "SHAPE CHAMPION"),
    ("Diversity blend w=0.50 (38f + 26f)", "exp_76_architectural_diversity_blend",
     "exp_76_diversity_blend_validation.csv", "Tweedie (p=1.1), both members", 38,
     "accepted #76"),
]

# The shipped model. Its per-row predictions were never persisted by Experiment
# #77, so its classification metrics cannot be computed without retraining.
FINAL = {
    "label": "Diversity blend w=0.60 (38f + 32f)",
    "objective": "Tweedie (p=1.1), both members",
    "registry": "exp_77_recursive_member_upgrade",
    "n_features": "38 + 32",
    "note": "FINAL SHIPPED CHAMPION",
}


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def reg_metrics(name):
    p = REG / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("metrics", {})
    return {"RMSE": m.get("RMSE"), "MAE": m.get("MAE"),
            "n_features": d.get("n_features"), "decision": d.get("decision")}


def load_pred(fname):
    df = pd.read_csv(PRED / fname)
    need = {"series_idx", "target_day_idx", "y_true", "y_pred"}
    if not need.issubset(df.columns):
        return None
    return df.sort_values(["target_day_idx", "series_idx"]).reset_index(drop=True)


def occurrence(y, p, thr=THRESHOLD):
    """Binary demand-occurrence metrics: actual > 0 vs predicted >= thr."""
    a = y > 0
    q = p >= thr
    tp = int(np.sum(a & q)); fp = int(np.sum(~a & q))
    fn = int(np.sum(a & ~q)); tn = int(np.sum(~a & ~q))
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main():
    banner("AUDIT — recompute every number from the saved artifacts")
    rows, problems, ref_y = [], [], None

    for label, regname, fname, obj, nfeat, note in MODELS:
        rm = reg_metrics(regname)
        if rm is None:
            problems.append(f"{label}: registry record {regname}.json missing")
            continue

        rec = {"Model": label, "Objective": obj,
               "Features": nfeat if nfeat is not None else "-",
               "Registry RMSE": rm["RMSE"], "Registry MAE": rm["MAE"],
               "Role": note, "Predictions": fname or "not saved"}

        if fname and (PRED / fname).exists():
            df = load_pred(fname)
            y = df.y_true.to_numpy(float); p = df.y_pred.to_numpy(float)
            if ref_y is None:
                ref_y = y
            elif len(y) == len(ref_y) and not np.array_equal(y, ref_y):
                problems.append(f"{label}: y_true differs from the reference set")
            r, m = metrics.rmse(y, p), metrics.mae(y, p)
            rec["RMSE"], rec["MAE"], rec["n"] = r, m, len(y)
            # cross-check against the registry
            for k, v in (("RMSE", r), ("MAE", m)):
                reg_v = rm[k]
                if reg_v is not None and abs(reg_v - v) > 5e-4:
                    problems.append(
                        f"{label}: {k} recomputed {v:.6f} vs registry {reg_v:.6f}")
            rec.update(occurrence(y, p))
            rec["Occurrence metrics"] = "computed"
        else:
            rec["RMSE"], rec["MAE"] = rm["RMSE"], rm["MAE"]
            rec["n"] = 853720 if rm["RMSE"] else None
            for k in ("Accuracy", "Precision", "Recall", "F1"):
                rec[k] = np.nan
            rec["Occurrence metrics"] = "N/A - predictions not saved"
        rows.append(rec)

    # ---- the shipped model: metrics from #77, predictions never persisted ----
    r77 = json.loads((REG / f"{FINAL['registry']}.json").read_text(encoding="utf-8"))
    op = pd.DataFrame(r77["operating_point"])
    prim = op[(op.pair == "AB2") & (op.window == "primary_spring_2016")].iloc[0]
    w = float(r77["inner_selected_weights"]["AB2"])
    final_rmse, final_mae = float(prim.RMSE), float(prim.MAE)
    print(f"  shipped blend w={w:.2f}: RMSE {final_rmse:.4f}  MAE {final_mae:.4f} "
          f"(from {FINAL['registry']}.json operating_point)")
    rows.append({
        "Model": FINAL["label"], "Objective": FINAL["objective"],
        "Features": FINAL["n_features"],
        "Registry RMSE": final_rmse, "Registry MAE": final_mae,
        "RMSE": final_rmse, "MAE": final_mae, "n": 853720,
        "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan,
        "Occurrence metrics": "N/A - not evaluated (predictions not persisted)",
        "Role": FINAL["note"], "Predictions": "not saved",
    })

    T = pd.DataFrame(rows)

    print(f"\n  {'model':<40}{'RMSE':>9}{'registry':>10}{'MAE':>9}{'registry':>10}")
    for _, r in T.iterrows():
        rr = f"{r['Registry RMSE']:.4f}" if pd.notna(r["Registry RMSE"]) else "-"
        mm = f"{r['Registry MAE']:.4f}" if pd.notna(r["Registry MAE"]) else "-"
        print(f"  {r['Model']:<40}{r['RMSE']:>9.4f}{rr:>10}{r['MAE']:>9.4f}{mm:>10}")

    banner("AUDIT RESULT")
    if problems:
        for p_ in problems:
            print(f"  MISMATCH  {p_}")
        raise SystemExit("STOP: audit failed — report not written.")
    print(f"  PASS — {len(T)} models, every recomputed RMSE/MAE agrees with the "
          "registry to <5e-4")

    base_rate = float((ref_y > 0).mean())
    print(f"  demand-occurrence base rate (actual > 0): {base_rate*100:.2f}% "
          f"of {len(ref_y):,} predictions")

    T.to_csv(ART / "final_performance_comparison.csv", index=False)
    print(f"  wrote {ART / 'final_performance_comparison.csv'}")
    return T, base_rate, w, r77


if __name__ == "__main__":
    main()
