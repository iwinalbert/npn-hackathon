"""
MODEL_COMPARISON.csv — the verified metric table for the paper.

Every row is recomputed from the model's own saved prediction file. Rows without
a prediction file carry the registry's RMSE/MAE and N/A elsewhere; nothing is
estimated or carried across from a different model.

    python MY_RESEARCH_PAPER/build_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import config, metrics

OUT = Path(__file__).resolve().parent
REPRO = OUT / "reproduction"
PRED = config.PREDICTIONS_DIR
REG = config.EXPERIMENTS_DIR
THRESHOLD = 0.5
NA = "N/A"

# label, registry, prediction path (absolute or in predictions/validation), objective, features, class
ROWS = [
    ("Naive — last value", "model_00_baseline_last_value", None, "none (arithmetic)", "—", "baseline"),
    ("Naive — rolling mean 7", "model_00_baseline_rolling_mean_7", None, "none (arithmetic)", "—", "baseline"),
    ("Naive — rolling mean 28", "model_00_baseline_rolling_mean_28", None, "none (arithmetic)", "—", "baseline"),
    ("Naive — seasonal naive", "model_00_baseline_seasonal_naive", "model_00_seasonal_naive_validation.csv", "none (arithmetic)", "—", "baseline"),
    ("LightGBM, L2 objective", "model_01_lightgbm", "model_01_lightgbm_validation.csv", "L2 (squared error)", "32", "development"),
    ("LightGBM, Tweedie", "model_02_tweedie", "model_02_tweedie_validation.csv", "Tweedie p=1.1", "32", "development"),
    ("LightGBM, Tweedie + recency", "model_03_tweedie_recency", "model_03_tweedie_recency_validation.csv", "Tweedie p=1.1", "32", "development"),
    ("LightGBM, Tweedie 32f (champion v1)", "model_04_tweedie_recency_listing", "model_04_tweedie_recency_listing_validation.csv", "Tweedie p=1.1", "32", "champion v1"),
    ("Hurdle (two-stage)", "model_05_hurdle", "model_05_hurdle_validation.csv", "binary + Tweedie", "32", "development"),
    ("LightGBM, Tweedie p=1.5", "opt_04b_power_1_5_primary", "opt_04b_power_1_5_primary_validation.csv", "Tweedie p=1.5", "32", "development"),
    ("LightGBM, Poisson", "opt_06_obj_poisson", "opt_06_obj_poisson_validation.csv", "Poisson", "32", "development"),
    ("LightGBM, L1", "opt_06_obj_l1", "opt_06_obj_l1_validation.csv", "L1 (absolute error)", "32", "development"),
    ("Recursive one-step (member B)", "opt_05_recursive", "opt_05_recursive_validation.csv", "Tweedie p=1.1", "26", "ensemble member"),
    ("Team-style approach, frozen-origin", "model_08_team_style_reproduction", "model_08_team_style_validation.csv", "Tweedie p=1.1", "—", "reference"),
    ("Shape only 36f", "exp_72_per_series_shape_features", "exp_72_shape_validation.csv", "Tweedie p=1.1", "36", "development"),
    ("Shape+Cycle 38f (champion v2)", "exp_74_shape_reproduction_and_extension", None, "Tweedie p=1.1", "38", "champion v2"),
    ("Diversity blend w=0.50 (38f+26f)", "exp_76_architectural_diversity_blend", "exp_76_diversity_blend_validation.csv", "Tweedie p=1.1 ×2", "38+26", "champion v3"),
]

FINAL = ("Diversity blend w=0.60 (38f+32f)", "exp_77_recursive_member_upgrade",
         REPRO / "shipped_blend_w060_validation.csv", "Tweedie p=1.1 ×2", "38+32",
         "FINAL SHIPPED")

MEMBERS = [
    ("  - member A: direct 38f (seed 42)", None, REPRO / "shipped_blend_w060_validation.csv",
     "Tweedie p=1.1", "38", "member", "y_pred_direct"),
    ("  - member B2: recursive 32f (seed 42)", None, REPRO / "shipped_blend_w060_validation.csv",
     "Tweedie p=1.1", "32", "member", "y_pred_recursive"),
]


def occ(y, p, thr=THRESHOLD):
    a, q = y > 0, p >= thr
    tp = int(np.sum(a & q)); fp = int(np.sum(~a & q))
    fn = int(np.sum(a & ~q)); tn = int(np.sum(~a & ~q))
    pr = tp / (tp + fp) if (tp + fp) else float("nan")
    rc = tp / (tp + fn) if (tp + fn) else float("nan")
    return ((tp + tn) / len(y), pr, rc,
            2 * pr * rc / (pr + rc) if (pr + rc) else float("nan"))


def reg(name):
    """
    Reference RMSE/MAE for a model.

    Special case for exp_77: its `metrics` field records the w=0.50 blend used
    for the ACCEPTANCE TEST, not the w=0.60 configuration that was shipped. The
    shipped figures live in `operating_point`. Comparing the shipped model
    against `metrics` would compare two different models — that is a
    documentation inconsistency in the project, recorded in the paper, not a
    discrepancy in the numbers.
    """
    p = REG / f"{name}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    if name == "exp_77_recursive_member_upgrade":
        op = [r for r in d["operating_point"]
              if r["pair"] == "AB2" and r["window"] == "primary_spring_2016"][0]
        return {"RMSE": op["RMSE"], "MAE": op["MAE"]}
    return d.get("metrics", {})


def build_row(label, regname, predpath, obj, feats, cls, col="y_pred"):
    rec = {"Model": label, "Objective": obj, "Features": feats, "Class": cls}
    path = predpath if isinstance(predpath, Path) else (
        PRED / predpath if predpath else None)
    if path is not None and path.exists():
        df = pd.read_csv(path)
        y, p = df.y_true.to_numpy(float), df[col].to_numpy(float)
        m = metrics.evaluate(y, p)
        a, pr, rc, f1 = occ(y, p)
        rec.update({"RMSE": round(m["RMSE"], 4), "MAE": round(m["MAE"], 4),
                    "WAPE": round(m["WAPE"], 4), "Bias": round(m["bias"], 4),
                    "Demand Accuracy": round(a, 4), "Precision": round(pr, 4),
                    "Recall": round(rc, 4), "F1": round(f1, 4),
                    "Source": "recomputed from predictions"})
        if regname:
            r = reg(regname)
            for k, v in (("RMSE", m["RMSE"]), ("MAE", m["MAE"])):
                if r.get(k) is not None and abs(r[k] - v) > 5e-4:
                    rec["Source"] = (f"MISMATCH vs registry {k} "
                                     f"{r[k]:.4f} != {v:.4f}")
    else:
        r = reg(regname) if regname else {}
        rec.update({"RMSE": round(r["RMSE"], 4) if r.get("RMSE") else NA,
                    "MAE": round(r["MAE"], 4) if r.get("MAE") else NA,
                    "WAPE": round(r["WAPE"], 4) if r.get("WAPE") else NA,
                    "Bias": round(r["bias"], 4) if r.get("bias") is not None else NA,
                    "Demand Accuracy": NA, "Precision": NA, "Recall": NA, "F1": NA,
                    "Source": "registry record (no prediction file on disk)"})
    return rec


def main():
    rows = [build_row(*r) for r in ROWS]
    rows.append(build_row(*FINAL))
    rows += [build_row(m[0], m[1], m[2], m[3], m[4], m[5], m[6]) for m in MEMBERS]

    T = pd.DataFrame(rows)[
        ["Model", "Objective", "Features", "RMSE", "MAE", "WAPE", "Bias",
         "Demand Accuracy", "Precision", "Recall", "F1", "Class", "Source"]]
    T.to_csv(OUT / "MODEL_COMPARISON.csv", index=False)

    bad = T[T.Source.str.startswith("MISMATCH")]
    print(T.to_string(index=False).encode("ascii","replace").decode("ascii"))
    print(f"\n  rows: {len(T)}   mismatches: {len(bad)}")
    if len(bad):
        print(bad[["Model", "Source"]].to_string(index=False))
        raise SystemExit("STOP: registry mismatch")
    print(f"  wrote {OUT / 'MODEL_COMPARISON.csv'}")


if __name__ == "__main__":
    main()
