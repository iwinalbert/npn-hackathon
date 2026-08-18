"""
Model training and the memory-efficient assembly of training matrices.

MEMORY NOTE
-----------
Each forecast origin contributes 30,490 series x 28 days = 853,720 rows. Building
many origins as pandas DataFrames and concatenating them peaks at roughly twice
the final size. With ~5.7 GB free that is wasteful, so build_training_matrix
pre-allocates one float32 array and fills it origin by origin, freeing each
intermediate frame immediately.

WHY ORIGINS ARE SPACED 28 DAYS APART
------------------------------------
Origin T covers target days T+1..T+28, origin T-28 covers T-27..T, and so on.
At a stride of 28 the coverage is contiguous and non-overlapping: every historical
day appears exactly once in the training set. A shorter stride would repeat the
same (series, day) target under several origins, silently overweighting those days.
"""

from __future__ import annotations

import time

import numpy as np

from . import config
from .backtest import Backtester
from .features import CATEGORICAL_FEATURES

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


# ==========================================================================
# Training-matrix assembly
# ==========================================================================

def build_training_matrix(
    bt: Backtester,
    origins: list[int],
    feature_cols: list[str],
    series_idx: np.ndarray | None = None,
    horizon: int = config.HORIZON,
    validation_origin: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Assemble (X, y) for a list of forecast origins without a concat spike.

    Returns X (float32, n_rows x n_features), y (float32), and an info dict.
    """
    n_series = len(series_idx) if series_idx is not None else config.N_SERIES
    rows_per_origin = n_series * horizon
    total_rows = rows_per_origin * len(origins)

    X = np.empty((total_rows, len(feature_cols)), dtype=np.float32)
    y = np.empty(total_rows, dtype=np.float32)

    t0 = time.time()
    for i, o in enumerate(origins):
        frame = bt.fb.build_origin_frame(
            o, horizon=horizon, series_idx=series_idx, include_target=True
        )

        if validation_origin is not None:
            mx = int(frame["target_day_idx"].max())
            if mx > validation_origin:
                raise AssertionError(
                    f"LEAKAGE: origin d_{o + 1} produces a target on day index {mx}, "
                    f"beyond the training cutoff {validation_origin}"
                )
        if frame["sales"].isna().any():
            raise AssertionError(f"origin d_{o + 1} produced rows with unknown targets")

        s = i * rows_per_origin
        e = s + rows_per_origin
        X[s:e] = frame[feature_cols].to_numpy(dtype=np.float32)
        y[s:e] = frame["sales"].to_numpy(dtype=np.float32)
        del frame

        if verbose and (i + 1) % 5 == 0:
            print(f"      built {i + 1}/{len(origins)} origins "
                  f"({time.time() - t0:.0f}s)")

    info = {
        "n_origins": len(origins),
        "rows": int(total_rows),
        "n_features": len(feature_cols),
        "memory_mb": round(X.nbytes / 1e6, 1),
        "build_seconds": round(time.time() - t0, 1),
        "origin_days": [f"d_{o + 1}" for o in origins],
    }
    return X, y, info


def categorical_indices(feature_cols: list[str]) -> list[int]:
    """Positions of categorical columns within a feature list, for LightGBM."""
    return [i for i, c in enumerate(feature_cols) if c in CATEGORICAL_FEATURES]


# ==========================================================================
# Model 0 — naive baselines (no fitting of any kind)
# ==========================================================================

def seasonal_naive_predict(
    data, origin_idx: int, horizon: int = config.HORIZON,
    series_idx: np.ndarray | None = None,
) -> np.ndarray:
    """
    Seasonal naive: predict each target day using the most recent day at or
    before the origin that falls on the SAME weekday.

    For target day T+h, that day is T + h - 7*ceil(h/7):
        h=1..7   -> T+h-7   (last week, same weekday)
        h=8..14  -> T+h-14  (two weeks back)
        ...
    Never reaches past the origin, so it is safe for all 28 horizon days.

    Returned in horizon-major order to match the feature frames.
    """
    if series_idx is None:
        series_idx = np.arange(config.N_SERIES)

    preds = np.empty((horizon, len(series_idx)), dtype=np.float32)
    for h in range(1, horizon + 1):
        weeks_back = int(np.ceil(h / 7))
        src = origin_idx + h - 7 * weeks_back
        preds[h - 1] = data.sales_wide[series_idx, src]
    return preds.ravel()


def naive_last_value_predict(
    data, origin_idx: int, horizon: int = config.HORIZON,
    series_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Repeat the origin day's sales for all 28 days."""
    if series_idx is None:
        series_idx = np.arange(config.N_SERIES)
    v = data.sales_wide[series_idx, origin_idx].astype(np.float32)
    return np.tile(v, horizon)


def rolling_mean_predict(
    data, origin_idx: int, window: int, horizon: int = config.HORIZON,
    series_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Repeat each series' own trailing mean over `window` days ending at origin."""
    if series_idx is None:
        series_idx = np.arange(config.N_SERIES)
    start = max(0, origin_idx - window + 1)
    m = data.sales_wide[series_idx, start:origin_idx + 1].astype(np.float64).mean(axis=1)
    return np.tile(m.astype(np.float32), horizon)


# ==========================================================================
# LightGBM
# ==========================================================================

# Conservative, untuned defaults. Deliberately NOT hyperparameter-searched at
# this stage — and deliberately trained for a fixed number of rounds with no
# early stopping, because early stopping on the validation window would use the
# validation set to make a training decision and quietly inflate its own score.
DEFAULT_PARAMS: dict = {
    "objective": "regression",       # L2
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_threshold": 32,
    "verbosity": -1,
    "num_threads": 0,
    "seed": config.RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,
}

N_ESTIMATORS = 400


def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    params: dict | None = None,
    n_estimators: int = N_ESTIMATORS,
    verbose: bool = True,
) -> tuple["lgb.Booster", dict]:
    """Train a LightGBM model. Returns (booster, info)."""
    if lgb is None:
        raise ImportError("lightgbm is not installed")

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    cat_idx = categorical_indices(feature_cols)

    t0 = time.time()
    dset = lgb.Dataset(
        X, label=y,
        feature_name=list(feature_cols),
        categorical_feature=cat_idx,
        free_raw_data=True,
    )
    booster = lgb.train(
        p, dset,
        num_boost_round=n_estimators,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    secs = time.time() - t0

    info = {
        "params": p,
        "n_estimators": n_estimators,
        "training_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "categorical_features": [feature_cols[i] for i in cat_idx],
        "training_seconds": round(secs, 1),
    }
    if verbose:
        print(f"      trained in {secs:.0f}s "
              f"({X.shape[0]:,} rows x {X.shape[1]} features)")
    return booster, info


def predict_nonneg(booster, X: np.ndarray) -> np.ndarray:
    """
    Predict and clip negatives to zero.

    Unit sales cannot be negative. An L2-objective model can emit small negative
    values near zero; clipping is the standard, defensible fix and is applied
    identically to every model so comparisons stay fair. Tweedie and the hurdle
    model are non-negative by construction, so clipping is a no-op for them.
    """
    p = booster.predict(X, num_iteration=booster.best_iteration or None)
    return np.clip(p, 0.0, None).astype(np.float64)
