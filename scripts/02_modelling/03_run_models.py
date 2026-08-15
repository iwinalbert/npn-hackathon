"""
Phases 2-7: Model 0 (naive baselines) through Model 5 (two-stage hurdle).

Every model is evaluated on the IDENTICAL validation window
(origin d_1913 -> targets d_1914..d_1941, all 30,490 series x 28 days =
853,720 predictions) using the identical metric code, so the comparison between
them is fair by construction.

    python scripts/03_run_models.py

Writes experiments/*.json, models/*.txt, predictions/*.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, feature_sets, metrics, models
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment

N_TRAIN_ORIGINS = 15          # 15 x 28 = 420 contiguous days of training coverage
N_ESTIMATORS = 400


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def save_predictions(name: str, valid: pd.DataFrame, preds: np.ndarray) -> str:
    """One row per (series, horizon day) with the prediction and the truth."""
    out = pd.DataFrame({
        "series_idx": valid["series_idx"].to_numpy(),
        "target_day_idx": valid["target_day_idx"].to_numpy(),
        "horizon": valid["horizon"].to_numpy(),
        "y_true": valid["sales"].to_numpy(),
        "y_pred": np.round(preds, 5),
    })
    path = config.PREDICTIONS_DIR / f"{name}_validation.csv"
    out.to_csv(path, index=False)
    return str(path.relative_to(config.PROJECT_ROOT))


def evaluate_and_record(exp: Experiment, valid: pd.DataFrame, preds: np.ndarray,
                        name: str, save_preds: bool = True) -> dict:
    y = valid["sales"].to_numpy()
    if np.isnan(preds).any():
        exp.error(f"{np.isnan(preds).sum()} NaN predictions — stopping")
        raise ValueError("NaN in predictions")
    if (preds < 0).any():
        exp.warn(f"{(preds < 0).sum()} negative predictions before clipping")

    m = metrics.evaluate(y, preds)
    exp.set_metrics(**m)
    exp.set(validation_rows=int(len(y)))
    if save_preds:
        exp.set(prediction_path=save_predictions(name, valid, preds))
    print(f"      RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
          f"WAPE={m['WAPE']:.4f}  bias={m['bias']:+.4f}")
    return m


def main() -> None:
    banner("SETUP")
    t0 = time.time()
    data = M5Data()
    bt = Backtester(data)
    print(f"  data loaded in {time.time() - t0:.1f}s")

    VO = config.VALIDATION_ORIGIN_IDX
    window = bt.make_window(VO)
    wd = window.describe()
    print(f"  validation origin : {wd['forecast_origin_day']} ({wd['forecast_origin_date']})")
    print(f"  validation window : {wd['validation_days']} ({wd['validation_dates']})")

    valid = bt.build_validation_frame(VO)
    y_true = valid["sales"].to_numpy()
    print(f"  validation frame  : {valid.shape[0]:,} rows "
          f"({config.N_SERIES:,} series x {config.HORIZON} days)")
    assert len(valid) == config.N_SERIES * config.HORIZON

    origins = bt.training_origins(VO, n_origins=N_TRAIN_ORIGINS)
    print(f"  training origins  : {len(origins)} "
          f"(d_{origins[0] + 1} .. d_{origins[-1] + 1}), "
          f"{len(origins) * config.HORIZON} days of coverage")
    print(f"  training dates    : {data.date_of(origins[0]).date()} .. "
          f"{data.date_of(origins[-1] + config.HORIZON).date()}")

    common = {
        "validation_origin_day": wd["forecast_origin_day"],
        "validation_dates": wd["validation_dates"],
        "validation_days": wd["validation_days"],
        "horizon": config.HORIZON,
        "n_series": config.N_SERIES,
        "validation_rows": int(len(valid)),
    }

    # ==================================================================
    banner("PHASE 2 — MODEL 0: NAIVE BASELINES (no model fitting)")
    # ==================================================================
    print("  Four rules, none of which fit any parameters. Their purpose is to say")
    print("  how hard this problem is before any learning happens.\n")

    baselines = {
        "seasonal_naive": (
            "Most recent SAME WEEKDAY on or before the origin",
            models.seasonal_naive_predict(data, VO)),
        "last_value": (
            "Repeat the origin day's sales for all 28 days",
            models.naive_last_value_predict(data, VO)),
        "rolling_mean_7": (
            "Repeat each series' own mean over the last 7 days",
            models.rolling_mean_predict(data, VO, 7)),
        "rolling_mean_28": (
            "Repeat each series' own mean over the last 28 days",
            models.rolling_mean_predict(data, VO, 28)),
    }

    baseline_results = {}
    for bname, (desc, preds) in baselines.items():
        print(f"  [{bname}] {desc}")
        exp = Experiment(
            f"model_00_baseline_{bname}",
            model_type="naive baseline (no fitting)",
            objective="n/a",
            feature_groups=["none — arithmetic rule only"],
            description=desc,
            **common,
        )
        m = evaluate_and_record(exp, valid, preds, f"model_00_{bname}",
                                save_preds=(bname == "seasonal_naive"))
        exp.set(training_rows=0, training_seconds=0.0).save()
        baseline_results[bname] = m

    best_base = min(baseline_results, key=lambda k: baseline_results[k]["RMSE"])
    print(f"\n  Best baseline by RMSE: {best_base} "
          f"(RMSE {baseline_results[best_base]['RMSE']:.4f})")

    # ==================================================================
    # Helper for the LightGBM experiments
    # ==================================================================
    def run_lgbm(exp_name: str, feature_set: str, objective_params: dict,
                 objective_label: str, note: str) -> dict:
        cols = feature_sets.get(feature_set)
        groups = feature_sets.groups_in(feature_set)
        print(f"  features : {feature_set} — {len(cols)} cols, groups {groups}")
        print(f"  objective: {objective_label}")

        exp = Experiment(
            exp_name,
            model_type="LightGBM",
            objective=objective_label,
            feature_set=feature_set,
            feature_set_label=feature_sets.FEATURE_SET_LABELS.get(feature_set, ""),
            feature_groups=groups,
            features=cols,
            n_features=len(cols),
            **common,
        )
        exp.note(note)

        X, y, binfo = models.build_training_matrix(
            bt, origins, cols, validation_origin=VO, verbose=False)
        print(f"  training : {binfo['rows']:,} rows, {binfo['memory_mb']} MB "
              f"(built in {binfo['build_seconds']}s)")
        exp.set(training_data=binfo, training_rows=binfo["rows"],
                training_origins=binfo["origin_days"])

        booster, minfo = models.train_lightgbm(
            X, y, cols, params=objective_params, n_estimators=N_ESTIMATORS,
            verbose=True)
        exp.set(**{k: v for k, v in minfo.items() if k != "params"})
        exp.set(hyperparameters=minfo["params"])
        del X, y

        mpath = config.MODELS_DIR / f"{exp_name}.txt"
        booster.save_model(str(mpath))
        exp.set(model_path=str(mpath.relative_to(config.PROJECT_ROOT)))

        t = time.time()
        preds = models.predict_nonneg(booster, valid[cols].to_numpy(np.float32))
        exp.set(prediction_seconds=round(time.time() - t, 1))

        m = evaluate_and_record(exp, valid, preds, exp_name)
        exp.save()
        return {"metrics": m, "booster": booster, "cols": cols, "preds": preds}

    # ==================================================================
    banner("PHASE 3 — MODEL 1: GLOBAL LIGHTGBM (L2 objective)")
    # ==================================================================
    print("  One model across all 30,490 series. No hyperparameter tuning, no early")
    print("  stopping — a fixed 400 rounds. Early stopping on the validation window")
    print("  would use validation to make a training decision and inflate its score.\n")
    m1 = run_lgbm(
        "model_01_lightgbm", "base",
        {"objective": "regression", "metric": "rmse"},
        "regression (L2)",
        "Baseline learned model. BASE features exclude recency and listing so "
        "Models 3 and 4 can measure what those groups add.",
    )

    # ==================================================================
    banner("PHASE 4 — MODEL 2: LIGHTGBM + TWEEDIE")
    # ==================================================================
    print("  Identical features to Model 1; ONLY the objective changes, so any")
    print("  difference is attributable to the loss function alone.\n")
    m2 = run_lgbm(
        "model_02_tweedie", "base",
        {"objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse"},
        "tweedie (variance_power=1.1)",
        "Tweedie is a distribution for non-negative outcomes with a spike at zero, "
        "which matches this target (68% of all historical rows are zero). Whether "
        "it actually helps here is measured, not assumed.",
    )

    # ==================================================================
    banner("PHASE 5 — MODEL 3: + RECENCY FEATURES")
    # ==================================================================
    print("  Model 2 + group C (days_since_last_sale, zero_streak_length,")
    print("  days_since_first_sale). Isolates the value of recency state.\n")
    m3 = run_lgbm(
        "model_03_tweedie_recency", "base_recency",
        {"objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse"},
        "tweedie (variance_power=1.1)",
        "Adds recency only. EDA found P(sale today) falls from 65.2% to 0.6% as the "
        "dry spell grows, the cleanest relationship in the dataset.",
    )

    # ==================================================================
    banner("PHASE 6 — MODEL 4: + LISTING-AWARE FEATURES")
    # ==================================================================
    print("  Model 3 + group D (days_since_first_listing, pre_listing).")
    print("  The foundation stage found pre_listing is 0% at this origin, so a")
    print("  measurable gain here is NOT expected. Testing it anyway is the point.\n")
    m4 = run_lgbm(
        "model_04_tweedie_recency_listing", "base_recency_listing",
        {"objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse"},
        "tweedie (variance_power=1.1)",
        "Adds listing-aware features. Foundation stage established pre_listing is "
        "constant zero at this forecast origin, so this is a genuine test of whether "
        "the proposed novelty feature earns its place rather than an assumption.",
    )

    # ==================================================================
    banner("PHASE 7 — MODEL 5: TWO-STAGE HURDLE")
    # ==================================================================
    print("  Stage 1: P(sales > 0)          — binary classifier")
    print("  Stage 2: E[units | sales > 0]   — Poisson regressor on positive rows only")
    print("  Final  : Stage1 x Stage2\n")

    # Use whichever feature set performed best among Models 2-4, chosen by RMSE.
    contenders = {
        "base": m2["metrics"]["RMSE"],
        "base_recency": m3["metrics"]["RMSE"],
        "base_recency_listing": m4["metrics"]["RMSE"],
    }
    hurdle_fs = min(contenders, key=contenders.get)
    print(f"  Feature set chosen by measured RMSE among Models 2-4: {hurdle_fs}")
    print(f"    {', '.join(f'{k}={v:.4f}' for k, v in contenders.items())}\n")

    cols = feature_sets.get(hurdle_fs)
    exp5 = Experiment(
        "model_05_hurdle",
        model_type="LightGBM two-stage hurdle",
        objective="stage1=binary, stage2=poisson",
        feature_set=hurdle_fs,
        feature_set_label=feature_sets.FEATURE_SET_LABELS.get(hurdle_fs, ""),
        feature_groups=feature_sets.groups_in(hurdle_fs),
        features=cols, n_features=len(cols),
        **common,
    )
    exp5.note(
        "Feature set was selected by measured validation RMSE across Models 2-4, "
        f"not chosen in advance ({contenders})."
    )

    X, y, binfo = models.build_training_matrix(
        bt, origins, cols, validation_origin=VO, verbose=False)
    exp5.set(training_data=binfo, training_rows=binfo["rows"],
             training_origins=binfo["origin_days"])
    print(f"  training : {binfo['rows']:,} rows")

    y_bin = (y > 0).astype(np.float32)
    pos = y > 0
    print(f"  positive rate in training: {y_bin.mean() * 100:.2f}% "
          f"({int(pos.sum()):,} positive rows for stage 2)")
    exp5.set(training_positive_rate=round(float(y_bin.mean()), 6),
             stage2_training_rows=int(pos.sum()))

    print("  [stage 1] binary classifier ...")
    b1, i1 = models.train_lightgbm(
        X, y_bin, cols,
        params={"objective": "binary", "metric": "binary_logloss"},
        n_estimators=N_ESTIMATORS, verbose=True)

    print("  [stage 2] poisson regressor on positive rows ...")
    b2, i2 = models.train_lightgbm(
        X[pos], y[pos], cols,
        params={"objective": "poisson", "metric": "poisson"},
        n_estimators=N_ESTIMATORS, verbose=True)
    del X, y, y_bin, pos

    exp5.set(stage1=i1, stage2=i2,
             training_seconds=round(i1["training_seconds"] + i2["training_seconds"], 1),
             hyperparameters={"stage1": i1["params"], "stage2": i2["params"]})

    p1 = config.MODELS_DIR / "model_05_hurdle_stage1.txt"
    p2 = config.MODELS_DIR / "model_05_hurdle_stage2.txt"
    b1.save_model(str(p1)); b2.save_model(str(p2))
    exp5.set(model_path=f"{p1.relative_to(config.PROJECT_ROOT)} + "
                        f"{p2.relative_to(config.PROJECT_ROOT)}")

    Xv = valid[cols].to_numpy(np.float32)
    t = time.time()
    prob = np.clip(b1.predict(Xv), 0.0, 1.0)
    mag = np.clip(b2.predict(Xv), 0.0, None)
    preds5 = prob * mag
    exp5.set(prediction_seconds=round(time.time() - t, 1))
    exp5.set(mean_predicted_probability=round(float(prob.mean()), 6),
             mean_predicted_magnitude=round(float(mag.mean()), 6))
    print(f"  mean P(sale)={prob.mean():.4f}, mean E[units|sale]={mag.mean():.4f}")

    m5 = evaluate_and_record(exp5, valid, preds5, "model_05_hurdle")
    exp5.save()

    # ==================================================================
    banner("PHASE 2-7 SUMMARY (measured, identical validation window)")
    # ==================================================================
    rows = [
        ("Model 0  seasonal naive", "none", baseline_results["seasonal_naive"]),
        ("Model 0  rolling_mean_28", "none", baseline_results["rolling_mean_28"]),
        ("Model 1  LightGBM L2", "base", m1["metrics"]),
        ("Model 2  LightGBM Tweedie", "base", m2["metrics"]),
        ("Model 3  + recency", "base_recency", m3["metrics"]),
        ("Model 4  + listing", "base_recency_listing", m4["metrics"]),
        ("Model 5  hurdle", hurdle_fs, m5),
    ]
    print(f"  {'Model':<28} {'Features':<22} {'RMSE':>8} {'MAE':>8} {'WAPE':>8}")
    print("  " + "-" * 76)
    for label, fsname, m in rows:
        print(f"  {label:<28} {fsname:<22} {m['RMSE']:>8.4f} "
              f"{m['MAE']:>8.4f} {m['WAPE']:>8.4f}")

    best = min(rows, key=lambda r: r[2]["RMSE"])
    print(f"\n  Lowest RMSE so far: {best[0]} (RMSE {best[2]['RMSE']:.4f}, "
          f"MAE {best[2]['MAE']:.4f})")
    print("\n  NOTE: the team's benchmark (RMSE 2.0324 / MAE 1.0869) was produced under")
    print("  a validation setup we have no documentation for. It is NOT a verified")
    print("  like-for-like comparison and is not treated as one.")
    print(f"\n  total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
