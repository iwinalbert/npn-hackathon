
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

import lightgbm as lgb

from pipeline import config, metrics, optimize
from pipeline.experiment import Experiment
from pipeline.features import FEATURE_GROUPS
from pipeline.features_v2 import FeatureBuilderV2, categorical_for

REC_COLS = (FEATURE_GROUPS["A_calendar"] + FEATURE_GROUPS["B_historical_demand"]
            + FEATURE_GROUPS["E_price"] + FEATURE_GROUPS["F_hierarchy"])

N_TRAIN_ORIGINS = 420
VO = config.VALIDATION_ORIGIN_IDX


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def build_one_step_training(s, origins):
    n = config.N_SERIES
    X = np.empty((n * len(origins), len(REC_COLS)), dtype=np.float32)
    Y = np.empty(n * len(origins), dtype=np.float32)
    t0 = time.time()
    for i, o in enumerate(origins):
        f = s.fb.build_origin_frame(o, horizon=1, include_target=True)
        assert int(f["target_day_idx"].max()) <= VO, "training target past cutoff"
        X[i * n:(i + 1) * n] = f[REC_COLS].to_numpy(np.float32)
        Y[i * n:(i + 1) * n] = f["sales"].to_numpy(np.float32)
        del f
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(origins)} origins ({time.time()-t0:.0f}s)")
    return X, Y


def recursive_forecast(data, booster, origin, horizon=config.HORIZON):
    n = config.N_SERIES
    n_days = data.sales_wide.shape[1]

    work = np.zeros((n, n_days), dtype=np.float32)
    work[:, :origin + 1] = data.sales_wide[:, :origin + 1]

    import copy as _copy
    preds = np.empty((horizon, n), dtype=np.float64)
    for h in range(1, horizon + 1):
        pseudo = origin + h - 1
        d2 = _copy.copy(data)
        d2.sales_wide = work
        fb = FeatureBuilderV2(d2)
        frame = fb.build_origin_frame(pseudo, horizon=1, include_target=False)
        p = np.clip(booster.predict(frame[REC_COLS].to_numpy(np.float32)), 0, None)
        preds[h - 1] = p
        work[:, pseudo + 1] = p.astype(np.float32)
        del fb, frame, d2
    return preds.ravel(), work


def main():
    t0 = time.time()
    s = optimize.Setup()

    banner("PHASE 5 — RECURSIVE FORECASTING")
    print(f"  window   : {s.window['validation_days']} ({s.window['validation_dates']})")
    print(f"  features : {len(REC_COLS)} (calendar + demand + price + hierarchy)")
    print(f"  model    : one-step-ahead, trained on {N_TRAIN_ORIGINS} daily origins\n")

    origins = list(range(VO - N_TRAIN_ORIGINS, VO))
    assert max(origins) + 1 <= VO
    print(f"  training origins: d_{origins[0]+1} .. d_{origins[-1]+1} "
          f"(targets up to d_{max(origins)+2})")

    print("\n  Building one-step training set...")
    X, Y = build_one_step_training(s, origins)
    print(f"  training matrix {X.shape} ({X.nbytes/1e6:.0f} MB)")

    ds = lgb.Dataset(X, label=Y, feature_name=list(REC_COLS),
                     categorical_feature=categorical_for(REC_COLS),
                     free_raw_data=True)
    params = dict(optimize.BASE_PARAMS)
    t = time.time()
    booster = lgb.train(params, ds, num_boost_round=optimize.N_ESTIMATORS,
                        callbacks=[lgb.log_evaluation(period=0)])
    train_s = time.time() - t
    print(f"  trained one-step model in {train_s:.0f}s")
    del X, Y, ds

    print("\n  Rolling forward 28 days, feeding predictions back...")
    t = time.time()
    preds, work = recursive_forecast(s.data, booster, VO)
    print(f"  rollout in {time.time()-t:.0f}s")

    real_future = s.data.sales_wide[:, VO + 1:VO + 1 + config.HORIZON]
    used_future = work[:, VO + 1:VO + 1 + config.HORIZON]
    identical = np.array_equal(real_future.astype(np.float32), used_future)
    print(f"  working matrix identical to real future sales? {identical} "
          f"(must be False)")
    if identical:
        raise SystemExit("STOP: recursion appears to have used real future sales")

    y = s.y
    d = optimize.diagnostics(y, preds, s)

    exp = Experiment(
        "opt_05_recursive", model_type="LightGBM one-step + recursive rollout",
        objective="tweedie (variance_power=1.1)",
        feature_set_label=f"Recursive, {len(REC_COLS)} features "
                          "(no recency/listing/horizon)",
        n_features=len(REC_COLS), features=list(REC_COLS), **s.describe())
    exp.note("Phase 5. A one-step-ahead model rolled forward 28 times, feeding "
             "its own predictions back as the previous day's sales. Real sales "
             "after the origin are never copied into the working matrix, so "
             "leakage is structurally impossible, and this was verified.")
    exp.set(training_rows=int(config.N_SERIES * len(origins)),
            training_origins_count=len(origins),
            hyperparameters=params, n_estimators=optimize.N_ESTIMATORS,
            training_seconds=round(train_s, 1),
            leakage_structural_check_passed=not identical)
    exp.set_metrics(**d)
    exp.set(delta_rmse_vs_best=round(d["RMSE"] - optimize.BEST_RMSE, 6),
            delta_mae_vs_best=round(d["MAE"] - optimize.BEST_MAE, 6))

    mp = config.MODELS_DIR / "opt_05_recursive_onestep.txt"
    booster.save_model(str(mp))
    exp.set(model_path=str(mp.relative_to(config.PROJECT_ROOT)))

    pp = config.PREDICTIONS_DIR / "opt_05_recursive_validation.csv"
    pd.DataFrame({
        "series_idx": s.valid["series_idx"].to_numpy(),
        "target_day_idx": s.valid["target_day_idx"].to_numpy(),
        "horizon": s.valid["horizon"].to_numpy(),
        "y_true": y, "y_pred": np.round(preds, 5),
    }).to_csv(pp, index=False)
    exp.set(prediction_path=str(pp.relative_to(config.PROJECT_ROOT)))
    exp.save()

    print(f"\n  RECURSIVE : RMSE={d['RMSE']:.4f} ({d['RMSE']-optimize.BEST_RMSE:+.4f})  "
          f"MAE={d['MAE']:.4f} ({d['MAE']-optimize.BEST_MAE:+.4f})")
    print(f"  DIRECT    : RMSE={optimize.BEST_RMSE:.4f}  MAE={optimize.BEST_MAE:.4f}")

    banner("ERROR ACCUMULATION ACROSS THE 28 DAYS")
    direct = pd.read_csv(config.PREDICTIONS_DIR /
                         "model_04_tweedie_recency_listing_validation.csv")
    direct = direct.sort_values(["horizon", "series_idx"]).reset_index(drop=True)
    hz = s.valid["horizon"].to_numpy()

    rows = []
    for h in range(1, config.HORIZON + 1):
        m = hz == h
        dm = direct["horizon"].to_numpy() == h
        rows.append({
            "horizon_day": h,
            "recursive_RMSE": metrics.rmse(y[m], preds[m]),
            "direct_RMSE": metrics.rmse(direct["y_true"].to_numpy()[dm],
                                        direct["y_pred"].to_numpy()[dm]),
            "recursive_MAE": metrics.mae(y[m], preds[m]),
            "direct_MAE": metrics.mae(direct["y_true"].to_numpy()[dm],
                                      direct["y_pred"].to_numpy()[dm]),
            "recursive_mean_pred": float(preds[m].mean()),
        })
    hdf = pd.DataFrame(rows)
    hdf["gap_RMSE"] = hdf["recursive_RMSE"] - hdf["direct_RMSE"]
    hdf.to_csv(config.ARTIFACTS_DIR / "phase5_horizon_errors.csv", index=False)
    print(hdf.head(8).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("   ...")
    print(hdf.tail(4).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    d1 = hdf.iloc[0]
    d28 = hdf.iloc[-1]
    print(f"\n  day 1  : recursive {d1['recursive_RMSE']:.4f} vs direct "
          f"{d1['direct_RMSE']:.4f}")
    print(f"  day 28 : recursive {d28['recursive_RMSE']:.4f} vs direct "
          f"{d28['direct_RMSE']:.4f}")
    print(f"  recursive mean prediction drifts from "
          f"{d1['recursive_mean_pred']:.4f} (day 1) to "
          f"{d28['recursive_mean_pred']:.4f} (day 28); actual mean is {y.mean():.4f}")

    verdict = ("recursive beats direct" if d["RMSE"] < optimize.BEST_RMSE
               else "direct beats recursive — recursion rejected")
    print(f"\n  VERDICT: {verdict}")

    (config.ARTIFACTS_DIR / "phase5_recursive_summary.json").write_text(
        json.dumps({"recursive": d, "direct_RMSE": optimize.BEST_RMSE,
                    "direct_MAE": optimize.BEST_MAE, "verdict": verdict,
                    "by_horizon": hdf.to_dict(orient="records")},
                   indent=2, default=str), encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
