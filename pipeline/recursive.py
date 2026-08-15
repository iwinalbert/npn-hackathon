"""
The RECURSIVE forecasting member, extracted from Phase 5 so other experiments
can reuse it.

The logic is copied verbatim from scripts/04_optimization/19_phase5_recursive.py
(Experiment `opt_05_recursive`, RMSE 2.1182 / MAE 1.0717 on the primary window).
Nothing about the model is changed here - the point of the extraction is that
Experiment #76 needs to retrain this exact configuration on windows other than
the primary one, and copying a 60-line rollout into another script would be the
wrong way to do that.

WHY IT IS A USEFUL MEMBER DESPITE BEING NO BETTER ON ITS OWN
-----------------------------------------------------------
It is a genuinely different forecaster, not a re-tuned copy of the champion:

    champion    DIRECT. Features frozen at the origin T, all 28 days predicted
                in one shot, `horizon` as a feature.
    this        RECURSIVE. One model that only ever predicts one day ahead,
                rolled forward 28 times, each step re-deriving its lag and
                rolling features from a working matrix that now contains its own
                previous outputs.

Their residuals correlate 0.953, against 0.990 between the direct models
Experiment #70 ensembled. That difference is the whole point.

HOW LEAKAGE IS MADE STRUCTURALLY IMPOSSIBLE
-------------------------------------------
The working sales matrix is rebuilt from scratch as

    work[:, :T+1] = real sales      (everything up to the origin)
    work[:, T+1:] = 0, then overwritten by our own predictions as we go

Real sales for the target window are never copied into it, so no code path can
read them. `verify_no_future_leakage` asserts this after every rollout rather
than trusting the construction.
"""

from __future__ import annotations

import copy as _copy
import time

import numpy as np

import lightgbm as lgb

from . import config, optimize
from .features import FEATURE_GROUPS
from .features_v2 import FeatureBuilderV2, categorical_for

# Calendar + demand + price + hierarchy. `horizon` is dropped because it is
# always 1 in a one-step-ahead model, and recency/listing are dropped because a
# fractional prediction like 0.3 would be counted as "a sale happened", which
# would corrupt days_since_last_sale during recursion.
REC_COLS = (FEATURE_GROUPS["A_calendar"] + FEATURE_GROUPS["B_historical_demand"]
            + FEATURE_GROUPS["E_price"] + FEATURE_GROUPS["F_hierarchy"])

N_TRAIN_ORIGINS = 420          # one row per series per origin, horizon = 1


def build_one_step_training(feature_builder, origins, cutoff_idx, cols=None):
    """One row per (series, origin) with target = the very next day."""
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
    """
    Train the one-step-ahead model on daily origins ending at `origin_idx`.

    `builder_cls` / `cols` default to the Phase-5 configuration that Experiment
    #76 validated. Passing a richer builder (e.g. FeatureBuilderV5) and the
    matching column list trains the same architecture on more information; the
    defaults are never changed by doing so.
    """
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
    """
    Roll forward one day at a time, feeding predictions back in.

    Returns (predictions, work) where `predictions` is horizon-major - h=1 for
    every series, then h=2, ... - which is the same ordering the Backtester uses
    for its validation frame.
    """
    cols = REC_COLS if cols is None else list(cols)
    n = config.N_SERIES
    # sales_wide stops at d_1941, but at the FINAL forecast origin the rollout
    # needs to write 28 days past it. Size the working matrix by what the
    # rollout actually needs, not by how much history happens to exist.
    n_days = max(data.sales_wide.shape[1], origin + 1 + horizon)

    # Real history up to and including the origin; everything after is zero and
    # will be filled only with our own predictions.
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
        work[:, pseudo + 1] = p.astype(np.float32)     # feed our own output back
        del fb, frame, d2
    return preds.ravel(), work


def verify_no_future_leakage(data, work, origin, horizon=config.HORIZON) -> dict:
    """
    Structural proof that the rollout never saw the answers.

    Two checks:
      1. the working matrix inside the horizon is NOT the real sales
      2. the working matrix before the origin IS exactly the real sales
         (so the model was not accidentally starved of legitimate history)
    """
    real_future = data.sales_wide[:, origin + 1:origin + 1 + horizon]
    used_future = work[:, origin + 1:origin + 1 + horizon]
    # At the final forecast origin there is no ground truth to compare against,
    # so check 1 is vacuous and says so rather than passing by accident.
    no_ground_truth = real_future.shape[1] == 0
    identical = (False if no_ground_truth else
                 bool(np.array_equal(real_future.astype(np.float32), used_future)))
    history_intact = bool(np.array_equal(
        data.sales_wide[:, :origin + 1].astype(np.float32), work[:, :origin + 1]))
    return {
        "future_matrix_equals_real_sales": identical,   # must be False
        "pre_origin_history_intact": history_intact,    # must be True
        "no_ground_truth_exists": no_ground_truth,      # True only at d_1941
        "passed": (not identical) and history_intact,
    }
