
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

import lightgbm as lgb

from pipeline import config, metrics, team_style
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment

VO = config.VALIDATION_ORIGIN_IDX
VALID_DAYS = VO + 1 + np.arange(config.HORIZON)

TRAIN_START = 1214
TRAIN_DAYS = np.arange(TRAIN_START, VO + 1)

TWEEDIE = {
    "objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 128, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 32, "verbosity": -1,
    "num_threads": 0, "seed": config.RANDOM_SEED, "deterministic": True,
    "force_row_wise": True,
}
N_ESTIMATORS = 400


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t_start = time.time()
    data = M5Data()
    tb = team_style.TeamStyleBuilder(data)
    cols = team_style.FEATURE_COLUMNS

    banner("SETUP")
    print(f"  validation : d_{VALID_DAYS[0]+1} .. d_{VALID_DAYS[-1]+1} "
          f"({data.date_of(VALID_DAYS[0]).date()} .. {data.date_of(VALID_DAYS[-1]).date()})")
    print(f"  training   : d_{TRAIN_DAYS[0]+1} .. d_{TRAIN_DAYS[-1]+1} "
          f"({len(TRAIN_DAYS)} days x {config.N_SERIES:,} series = "
          f"{len(TRAIN_DAYS)*config.N_SERIES:,} rows)")
    print(f"  features   : {len(cols)} (team-style, per-target-day, min lookback "
          f"{team_style.MIN_LOOKBACK}d)")

    banner("LEAKAGE TEST ON THE NEW FEATURE CONSTRUCTION")
    print("  A different feature builder needs its own proof, not inherited trust.")
    print("  Overwriting all sales after the origin with 9999 and rebuilding...")

    Xc, _, _ = tb.build(VALID_DAYS, with_target=False)
    corrupt = data.sales_wide.copy()
    corrupt[:, VO + 1:] = 9999
    Xd, _, _ = tb.build(VALID_DAYS, sales=corrupt, with_target=False)

    same = np.array_equal(Xc, Xd, equal_nan=True)
    print(f"  {'PASS' if same else 'FAIL'}  all {len(cols)} team-style features "
          f"{'unchanged' if same else 'CHANGED'} under future-sales corruption")
    if not same:
        bad = [cols[j] for j in range(len(cols))
               if not np.array_equal(Xc[:, j], Xd[:, j], equal_nan=True)]
        raise SystemExit(f"STOP: leakage detected in {bad}")
    del Xd, corrupt

    banner("BUILD TRAINING SET")
    t0 = time.time()
    Xtr, ytr, _ = tb.build(TRAIN_DAYS, with_target=True, verbose=True)
    build_s = time.time() - t0
    print(f"  built {Xtr.shape[0]:,} x {Xtr.shape[1]} in {build_s:.0f}s "
          f"({Xtr.nbytes/1e6:.0f} MB)")

    nan_rows = int(np.isnan(Xtr).any(axis=1).sum())
    print(f"  rows containing at least one NaN feature: {nan_rows:,} "
          f"({nan_rows/len(Xtr)*100:.2f}%) — left as NaN, never imputed")

    banner("TRAIN — LightGBM Tweedie, team-style features")
    exp = Experiment(
        "model_08_team_style_reproduction",
        model_type="LightGBM",
        objective="tweedie (variance_power=1.1)",
        feature_set="team_style_per_target_day",
        feature_set_label="Per-target-day lags/rollings with 28-day minimum lookback",
        features=list(cols),
        n_features=len(cols),
        validation_origin_day="d_1913",
        validation_days="d_1914 .. d_1941",
        validation_dates=(f"{data.date_of(VALID_DAYS[0]).date()} .. "
                          f"{data.date_of(VALID_DAYS[-1]).date()}"),
        horizon=config.HORIZON,
        n_series=config.N_SERIES,
        validation_rows=int(config.N_SERIES * config.HORIZON),
        training_rows=int(Xtr.shape[0]),
        training_days=f"d_{TRAIN_DAYS[0]+1} .. d_{TRAIN_DAYS[-1]+1}",
        methodology_source="INFORMED RECONSTRUCTION of the standard public M5 "
                           "recipe — the team's actual code/config is not present "
                           "anywhere in this project or on this machine.",
    )
    exp.note("Hyperparameters, objective and validation window are held IDENTICAL "
             "to our Model 2/4 so that the only variable is the feature "
             "construction (per-target-day vs fixed-origin).")
    exp.note(f"Leakage corruption test passed on this new builder: all "
             f"{len(cols)} features unchanged when post-origin sales overwritten.")

    dset = lgb.Dataset(Xtr, label=ytr, feature_name=list(cols),
                       categorical_feature=team_style.categorical_indices(),
                       free_raw_data=True)
    t0 = time.time()
    booster = lgb.train(TWEEDIE, dset, num_boost_round=N_ESTIMATORS,
                        callbacks=[lgb.log_evaluation(period=0)])
    train_s = time.time() - t0
    print(f"  trained in {train_s:.0f}s")
    del Xtr, ytr, dset

    exp.set(hyperparameters=TWEEDIE, n_estimators=N_ESTIMATORS,
            training_seconds=round(train_s, 1),
            feature_build_seconds=round(build_s, 1),
            nan_feature_rows=nan_rows)

    mpath = config.MODELS_DIR / "model_08_team_style_reproduction.txt"
    booster.save_model(str(mpath))
    exp.set(model_path=str(mpath.relative_to(config.PROJECT_ROOT)))

    banner("EVALUATE on d_1914..d_1941")
    _, yv, vmeta = tb.build(VALID_DAYS, with_target=True)
    preds = np.clip(booster.predict(Xc), 0, None)

    m = metrics.evaluate(yv, preds)
    exp.set_metrics(**m)
    exp.set(prediction_mean=round(float(preds.mean()), 6),
            prediction_max=round(float(preds.max()), 4),
            actual_mean=round(float(yv.mean()), 6))
    print(f"  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  WAPE={m['WAPE']:.4f}  "
          f"bias={m['bias']:+.4f}")
    print(f"  mean predicted={preds.mean():.4f}  mean actual={yv.mean():.4f}")

    pdf = pd.DataFrame({
        "series_idx": vmeta["series_idx"], "target_day_idx": vmeta["target_day_idx"],
        "y_true": yv, "y_pred": np.round(preds, 5),
    })
    ppath = config.PREDICTIONS_DIR / "model_08_team_style_validation.csv"
    pdf.to_csv(ppath, index=False)
    exp.set(prediction_path=str(ppath.relative_to(config.PROJECT_ROOT)))
    exp.save()

    banner("WHERE DOES THE DIFFERENCE COME FROM?")
    ours = pd.read_csv(config.PREDICTIONS_DIR /
                       "model_04_tweedie_recency_listing_validation.csv")
    ours = ours.sort_values(["target_day_idx", "series_idx"]).reset_index(drop=True)
    new = pdf.sort_values(["target_day_idx", "series_idx"]).reset_index(drop=True)
    assert np.array_equal(ours["y_true"].to_numpy(), new["y_true"].to_numpy()), \
        "the two experiments are not scoring the same rows"

    hist = data.sales_wide[:, :VO + 1]
    mean_hist = hist.mean(axis=1)
    tier = pd.cut(mean_hist[new["series_idx"].to_numpy()],
                  [-0.001, 0.2, 1.0, 3.0, np.inf],
                  labels=["very low (<0.2/day)", "low (0.2-1)",
                          "medium (1-3)", "high (>3)"])

    y = new["y_true"].to_numpy()
    tier = pd.Series(tier)
    rows = []
    for lab in tier.cat.categories:
        msk = (tier == lab).to_numpy()
        rows.append({
            "tier": str(lab), "n": int(msk.sum()),
            "actual_mean": float(y[msk].mean()),
            "ours_RMSE": metrics.rmse(y[msk], ours["y_pred"].to_numpy()[msk]),
            "team_style_RMSE": metrics.rmse(y[msk], new["y_pred"].to_numpy()[msk]),
            "ours_MAE": metrics.mae(y[msk], ours["y_pred"].to_numpy()[msk]),
            "team_style_MAE": metrics.mae(y[msk], new["y_pred"].to_numpy()[msk]),
        })
    tdf = pd.DataFrame(rows)
    tdf["dRMSE"] = tdf["team_style_RMSE"] - tdf["ours_RMSE"]
    tdf["dMAE"] = tdf["team_style_MAE"] - tdf["ours_MAE"]
    print(tdf.to_string(index=False))
    tdf.to_csv(config.ARTIFACTS_DIR / "team_style_by_volume_tier.csv", index=False)

    summary = {
        "team_reported": {"RMSE": 2.0324, "MAE": 1.0869,
                          "source": "provided by the team; methodology not documented"},
        "our_current_best_model_04": {"RMSE": 2.1210, "MAE": 1.0319},
        "team_style_reproduction": {"RMSE": m["RMSE"], "MAE": m["MAE"],
                                    "WAPE": m["WAPE"], "bias": m["bias"]},
        "validation_window": "d_1914..d_1941 (identical for all three of our runs)",
        "training_rows": int(config.N_SERIES * len(TRAIN_DAYS)),
        "n_features": len(cols),
        "by_volume_tier": tdf.to_dict(orient="records"),
        "leakage_test_passed": bool(same),
    }
    (config.ARTIFACTS_DIR / "team_reproduction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    banner("HEADLINE")
    print(f"  team reported (their setup, undocumented) : RMSE 2.0324  MAE 1.0869")
    print(f"  our current best (Model 4)                : RMSE 2.1210  MAE 1.0319")
    print(f"  team-style reproduction (our pipeline)    : RMSE {m['RMSE']:.4f}  "
          f"MAE {m['MAE']:.4f}")
    print(f"\n  total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
