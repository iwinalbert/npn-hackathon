
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, experiment, metrics, optimize
from pipeline.features_v2 import V2_SETS

PRIMARY = "d_1914 .. d_1941"
NOISE_BAND = 0.013
MAE_VETO = 0.02


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def collect() -> pd.DataFrame:
    rows = []
    for r in experiment.load_all():
        if r.get("status") != "completed":
            continue
        if r.get("tuning_window") == "INNER" or r.get("robustness_run"):
            continue
        if r.get("validation_days") != PRIMARY:
            continue
        m = r.get("metrics", {})
        if "RMSE" not in m:
            continue
        name = r["experiment_name"]
        if name.startswith(("ablation_", "diagnostic_")):
            continue
        rows.append({
            "experiment": name,
            "label": r.get("feature_set_label") or r.get("objective", ""),
            "objective": r.get("objective", ""),
            "n_features": r.get("n_features"),
            "RMSE": m["RMSE"], "MAE": m["MAE"],
            "high_vol_RMSE": m.get("high_volume_RMSE"),
            "train_s": r.get("training_seconds", 0),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["experiment"])
    df["dRMSE"] = df["RMSE"] - optimize.BEST_RMSE
    df["dMAE"] = df["MAE"] - optimize.BEST_MAE
    return df.sort_values("RMSE").reset_index(drop=True)


def main():
    t0 = time.time()
    banner("PHASE 11 — FINAL MODEL SELECTION")

    df = collect()
    print(f"  {len(df)} leakage-safe candidates evaluated on {PRIMARY}\n")
    print(df[["experiment", "RMSE", "MAE", "dRMSE", "dMAE"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    top = df.iloc[0]
    incumbent = df[df.experiment == "opt_00_baseline_reproduce"].iloc[0]

    print(f"\n  lowest RMSE : {top['experiment']} ({top['RMSE']:.4f})")
    print(f"  incumbent   : {incumbent['experiment']} ({incumbent['RMSE']:.4f})")

    gain = incumbent["RMSE"] - top["RMSE"]
    mae_cost = top["MAE"] - incumbent["MAE"]
    print(f"  RMSE gain of the leader : {gain:+.4f}  "
          f"(noise band is +/-{NOISE_BAND})")
    print(f"  MAE cost of the leader  : {mae_cost:+.4f}")

    if gain < NOISE_BAND and mae_cost > MAE_VETO:
        selected = incumbent
        reason = (f"The lowest-RMSE candidate ({top['experiment']}) improves RMSE "
                  f"by only {gain:.4f}, which is inside the +/-{NOISE_BAND} "
                  f"window-to-window noise we measured, while costing "
                  f"{mae_cost:+.4f} MAE. That is not a real improvement, so the "
                  f"incumbent is retained.")
    else:
        selected = top
        reason = (f"{top['experiment']} has the lowest RMSE and does not breach "
                  f"the MAE veto.")

    print(f"\n  SELECTED: {selected['experiment']}")
    print(f"  reason  : {reason}")

    banner("PHASE 12 — FINAL 28-DAY FORECAST")

    incumbent_cfg = selected["experiment"] in (
        "opt_00_baseline_reproduce", "model_04_tweedie_recency_listing")
    fc_path = config.FINAL_FORECAST_DIR / "final_forecast_28day.csv"

    if incumbent_cfg and fc_path.exists():
        print("  The selected configuration is the one that already produced")
        print(f"  {fc_path.name}. Verifying that file rather than regenerating it,")
        print("  so no existing prediction is overwritten.\n")
        fc = pd.read_csv(fc_path)
        sub = pd.read_csv(config.SAMPLE_SUBMISSION_CSV, usecols=["id"])
        eval_ids = sub.loc[sub["id"].str.endswith("_evaluation"), "id"]
        vals = fc.iloc[:, 1:].to_numpy()
        checks = [
            ("rows_30490", len(fc) == config.N_SERIES, f"{len(fc):,} rows"),
            ("cols_F1_F28", list(fc.columns[1:]) == [f"F{i}" for i in range(1, 29)],
             f"{len(fc.columns)-1} forecast columns"),
            ("no_duplicate_ids", fc["id"].duplicated().sum() == 0, "0 duplicates"),
            ("no_nan", not np.isnan(vals).any(), "0 NaN"),
            ("no_negative", vals.min() >= 0, f"min {vals.min():.6f}"),
            ("ids_match_template", list(fc["id"]) == list(eval_ids),
             "ids and order match sample_submission.csv"),
        ]
        for n, ok, det in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {n}: {det}")
        if not all(c[1] for c in checks):
            raise SystemExit("STOP: existing forecast failed structure validation")
        forecast_note = ("existing file verified, not regenerated "
                         "(selected configuration unchanged)")
        fmean = float(vals.mean())
    else:
        raise SystemExit(
            "The selected model differs from the one behind the existing "
            "forecast. Regeneration would be required; stopping here rather "
            "than silently overwriting predictions/final_forecast_28day.csv.")

    summary = {
        "selected_model": selected["experiment"],
        "selection_reason": reason,
        "primary_RMSE": float(selected["RMSE"]),
        "primary_MAE": float(selected["MAE"]),
        "lowest_rmse_candidate": top["experiment"],
        "lowest_rmse_value": float(top["RMSE"]),
        "noise_band": NOISE_BAND,
        "forecast_file": str(fc_path.relative_to(config.PROJECT_ROOT)),
        "forecast_status": forecast_note,
        "forecast_mean": round(fmean, 5),
        "candidates": df.to_dict(orient="records"),
    }
    (config.ARTIFACTS_DIR / "final_selection.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    df.to_csv(config.ARTIFACTS_DIR / "final_scorecard.csv", index=False)

    print(f"\n  forecast: {fc_path.name} — {forecast_note}")
    print(f"  forecast mean {fmean:.4f} units/day")
    print(f"\n  wrote artifacts/final_selection.json and final_scorecard.csv")
    print(f"  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
