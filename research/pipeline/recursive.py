
from __future__ import annotations

import copy as _copy
import time

import numpy as np

import lightgbm as lgb

from . import config, optimize
from .features import FEATURE_GROUPS
from .features_v2 import FeatureBuilderV2, categorical_for

REC_COLS = (FEATURE_GROUPS["A_calendar"] + FEATURE_GROUPS["B_historical_demand"]
            + FEATURE_GROUPS["E_price"] + FEATURE_GROUPS["F_hierarchy"])

N_TRAIN_ORIGINS = 420


def build_one_step_training(feature_builder, origins, cutoff_idx, cols=None):
    cols = REC_COLS if cols is None else cols
    n = config.N_SERIES
    X = np.empty((n * len(origins), len(cols)), dtype=np.float32)
    Y = np.empty(n * len(origins), dtype=np.float32)
    for i, o in enumerate(origins):
        f = feature_builder.build_origin_frame(o, horizon=1, include_target=True)
        if int(f["target_day_idx"].max()) > cutoff_idx:
            raise AssertionError(
                f"LEAKAGE: one-step origin d_{o+1} targets day "
                f"{int(f['target_day_idx'].max())} beyond cutoff {cutoff_idx}")
        X[i * n:(i + 1) * n] = f[cols].to_numpy(np.float32)
        Y[i * n:(i + 1) * n] = f["sales"].to_numpy(np.float32)
        del f
    return X, Y


def train_one_step(data, origin_idx, *, seed=config.RANDOM_SEED,
                   n_train_origins=N_TRAIN_ORIGINS,
                   n_estimators=optimize.N_ESTIMATORS, verbose=False,
                   builder_cls=FeatureBuilderV2, cols=None):
    cols = REC_COLS if cols is None else list(cols)
    fb = builder_cls(data)
    origins = list(range(origin_idx - n_train_origins, origin_idx))
    if min(origins) < 0:
        raise ValueError(f"origin {origin_idx} has less than {n_train_origins} "
                         "days of history behind it")
    t0 = time.time()
    X, Y = build_one_step_training(fb, origins, origin_idx, cols)
    ds = lgb.Dataset(X, label=Y, feature_name=list(cols),
                     categorical_feature=categorical_for(cols),
                     free_raw_data=True)
    params = dict(optimize.BASE_PARAMS)
    params.update({"seed": seed, "bagging_seed": seed,
                   "feature_fraction_seed": seed})
    booster = lgb.train(params, ds, num_boost_round=n_estimators,
                        callbacks=[lgb.log_evaluation(period=0)])
    del X, Y, ds
    info = {"params": params, "n_estimators": n_estimators,
            "training_seconds": round(time.time() - t0, 1),
            "training_origins_count": len(origins),
            "training_origins_span": f"d_{origins[0]+1} .. d_{origins[-1]+1}"}
    if verbose:
        print(f"      one-step model trained in {info['training_seconds']}s "
              f"({len(origins)} daily origins)")
    return booster, info


def recursive_forecast(data, booster, origin, horizon=config.HORIZON,
                       builder_cls=FeatureBuilderV2, cols=None):
    cols = REC_COLS if cols is None else list(cols)
    n = config.N_SERIES
    n_days = max(data.sales_wide.shape[1], origin + 1 + horizon)

    work = np.zeros((n, n_days), dtype=np.float32)
    work[:, :origin + 1] = data.sales_wide[:, :origin + 1]

    preds = np.empty((horizon, n), dtype=np.float64)
    for h in range(1, horizon + 1):
        pseudo = origin + h - 1
        d2 = _copy.copy(data)
        d2.sales_wide = work
        fb = builder_cls(d2)
        frame = fb.build_origin_frame(pseudo, horizon=1, include_target=False)
        p = np.clip(booster.predict(frame[cols].to_numpy(np.float32)), 0, None)
        preds[h - 1] = p
        work[:, pseudo + 1] = p.astype(np.float32)
        del fb, frame, d2
    return preds.ravel(), work


def verify_no_future_leakage(data, work, origin, horizon=config.HORIZON) -> dict:
    real_future = data.sales_wide[:, origin + 1:origin + 1 + horizon]
    used_future = work[:, origin + 1:origin + 1 + horizon]
    no_ground_truth = real_future.shape[1] == 0
    identical = (False if no_ground_truth else
                 bool(np.array_equal(real_future.astype(np.float32), used_future)))
    history_intact = bool(np.array_equal(
        data.sales_wide[:, :origin + 1].astype(np.float32), work[:, :origin + 1]))
    return {
        "future_matrix_equals_real_sales": identical,
        "pre_origin_history_intact": history_intact,
        "no_ground_truth_exists": no_ground_truth,
        "passed": (not identical) and history_intact,
    }
