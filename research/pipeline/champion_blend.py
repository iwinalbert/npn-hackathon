"""
The SHIPPED champion, reproduced on demand for any origin, with caching.

NEW MODULE — READ-ONLY WITH RESPECT TO THE CHAMPION. It calls the existing
components (features_v5, recursive, optimize) exactly as Experiments #74-#79 did
and combines them at the shipped operating point. It defines no new modelling and
changes no existing file. Its only purpose is that later experiments need the
champion's per-row predictions on windows for which no prediction file was ever
saved, and re-deriving them by copying a training loop into each new script is
how reproductions drift.

THE SHIPPED CONFIGURATION  (Experiments #74, #76, #77, #79)
-----------------------------------------------------------
    member A   DIRECT   LightGBM tweedie(1.1), 38 features (CHAMPION_FEATURES),
                        400 rounds, 15 origins x 28 days
    member B'  RECURSIVE LightGBM tweedie(1.1), 32 features
                        (REC_COLS + the six shape/cycle features), one-step,
                        rolled forward 28 times
    blend      P = clip(0.60 * A + 0.40 * B', 0, None)

w = 0.60 was selected on an inner window (origin d_1885) in Experiment #77 and is
a fixed constant here — it is never re-fitted.

Primary window (origin d_1913, targets d_1914-d_1941), seed 42:
    RMSE 2.0929394   MAE 1.0395172
which is the number every candidate in this branch is measured against.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, optimize, recursive
from .backtest import Backtester
from .features_v2 import FeatureBuilderV2
from .features_v4 import V4_FEATURES
from .features_v5 import FeatureBuilderV5, CHAMPION_FEATURES, V5_FEATURES

#: member B' feature list, exactly as Experiment #77 defined it
REC_COLS_V5 = list(recursive.REC_COLS) + list(V4_FEATURES) + list(V5_FEATURES)

#: the shipped blend weight, selected on an inner window in Experiment #77
W_SHIPPED = 0.60

#: recorded primary-window scores of the shipped configuration
SHIPPED_RMSE = 2.0929394037324487
SHIPPED_MAE = 1.0395171989061582

CACHE_DIR = config.PREDICTIONS_ROOT / "uc11_cache"


class BlendSetup(optimize.Setup):
    """optimize.Setup on the champion's 38-feature builder, reusing one M5Data."""

    def __init__(self, data, origin_idx, n_origins=optimize.N_ORIGINS):
        self.data = data
        self.fb = FeatureBuilderV5(data)
        self.bt = Backtester(data, feature_builder=self.fb)
        self.origin_idx = origin_idx
        self.window = self.bt.make_window(origin_idx).describe()
        self.valid = self.bt.build_validation_frame(origin_idx)
        self.y = self.valid["sales"].to_numpy()
        self.origins = self.bt.training_origins(origin_idx, n_origins=n_origins)
        hist = data.sales_wide[:, :origin_idx + 1].mean(axis=1)
        self.tier = pd.Series(pd.cut(
            hist[self.valid["series_idx"].to_numpy()],
            [-0.001, 0.2, 1.0, 3.0, np.inf],
            labels=["very low", "low", "medium", "high"]))
        self.high = (self.tier == "high").to_numpy()
        self.is_zero = self.y == 0


def fit_direct(setup, seed=config.RANDOM_SEED):
    """Member A."""
    X, Y = optimize.build_matrix(setup, CHAMPION_FEATURES)
    booster, info = optimize.train(
        X, Y, CHAMPION_FEATURES,
        params={"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed})
    del X, Y
    p = optimize.predict(booster, setup, CHAMPION_FEATURES)
    del booster
    gc.collect()
    return p, info


def fit_recursive(data, origin_idx, seed=config.RANDOM_SEED):
    """Member B' — the upgraded recursive member, plus its leakage check."""
    booster, info = recursive.train_one_step(
        data, origin_idx, seed=seed,
        builder_cls=FeatureBuilderV5, cols=REC_COLS_V5)
    p, work = recursive.recursive_forecast(
        data, booster, origin_idx, builder_cls=FeatureBuilderV5, cols=REC_COLS_V5)
    checks = recursive.verify_no_future_leakage(data, work, origin_idx)
    del booster, work
    gc.collect()
    if not checks["passed"]:
        raise SystemExit(f"STOP: recursive leakage check failed: {checks}")
    return p, info, checks


def champion_predictions(data, origin_idx, *, seed=config.RANDOM_SEED,
                         cache=True, verbose=True):
    """
    Reproduce the shipped champion on one origin.

    Returns a dict with the two member forecasts, the blend, the truth and the
    leakage checks. Results are cached under predictions/uc11_cache/ keyed by
    origin and seed, because every experiment in this branch needs the same
    baseline and each rebuild costs several minutes.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"champion_blend_origin{origin_idx}_seed{seed}.csv"
    if cache and path.exists():
        df = pd.read_csv(path)
        if verbose:
            print(f"      champion blend loaded from cache: {path.name}", flush=True)
        return {
            "series_idx": df["series_idx"].to_numpy(),
            "target_day_idx": df["target_day_idx"].to_numpy(),
            "horizon": df["horizon"].to_numpy(),
            "y": df["y_true"].to_numpy(),
            "direct": df["p_direct"].to_numpy(),
            "recursive": df["p_recursive"].to_numpy(),
            "blend": df["p_blend"].to_numpy(),
            "from_cache": True,
        }

    t0 = time.time()
    s = BlendSetup(data, origin_idx)
    if verbose:
        print(f"      training member A (direct, {len(CHAMPION_FEATURES)} features)...",
              flush=True)
    pa, _ = fit_direct(s, seed=seed)
    if verbose:
        print(f"      training member B' (recursive, {len(REC_COLS_V5)} features)...",
              flush=True)
    pb, _, checks = fit_recursive(data, origin_idx, seed=seed)

    blend = np.clip(W_SHIPPED * pa + (1.0 - W_SHIPPED) * pb, 0, None)
    out = {
        "series_idx": s.valid["series_idx"].to_numpy(),
        "target_day_idx": s.valid["target_day_idx"].to_numpy(),
        "horizon": s.valid["horizon"].to_numpy(),
        "y": s.y, "direct": pa, "recursive": pb, "blend": blend,
        "leakage_checks": checks, "from_cache": False,
        "seconds": round(time.time() - t0, 1),
    }
    if cache:
        pd.DataFrame({
            "series_idx": out["series_idx"],
            "target_day_idx": out["target_day_idx"],
            "horizon": out["horizon"],
            "y_true": out["y"],
            "p_direct": np.round(pa, 6),
            "p_recursive": np.round(pb, 6),
            "p_blend": np.round(blend, 6),
        }).to_csv(path, index=False)
        if verbose:
            print(f"      cached -> {path.name}  ({out['seconds']}s)", flush=True)
    del s
    gc.collect()
    return out
