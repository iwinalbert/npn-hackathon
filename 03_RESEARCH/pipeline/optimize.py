"""
Shared harness for the optimization phases.

One place that knows how to: build the standard setup, train a LightGBM model on
a named feature set, evaluate it on the primary (or inner) window, compute the
diagnostics every phase report needs, and record the experiment.

Existing modules are reused rather than reimplemented — this calls
FeatureBuilder(V2), Backtester, metrics and Experiment directly.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import lightgbm as lgb

from . import config, metrics
from .backtest import Backtester
from .data_loader import M5Data
from .experiment import Experiment
from .features_v2 import FeatureBuilderV2, categorical_for

# Our current best, measured. Every delta in the optimization phases is quoted
# against these two numbers.
BEST_RMSE = 2.1210429411947650
BEST_MAE = 1.0319268155496617

BASE_PARAMS = {
    "objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 128, "max_depth": -1,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "max_cat_threshold": 32,
    "verbosity": -1, "num_threads": 0, "seed": config.RANDOM_SEED,
    "deterministic": True, "force_row_wise": True,
}
N_ESTIMATORS = 400
N_ORIGINS = 15


class Setup:
    """Loaded once and shared by every experiment in a phase script."""

    def __init__(self, origin_idx: int = config.VALIDATION_ORIGIN_IDX,
                 n_origins: int = N_ORIGINS):
        self.data = M5Data()
        self.fb = FeatureBuilderV2(self.data)
        self.bt = Backtester(self.data, feature_builder=self.fb)
        self.origin_idx = origin_idx
        self.window = self.bt.make_window(origin_idx).describe()
        self.valid = self.bt.build_validation_frame(origin_idx)
        self.y = self.valid["sales"].to_numpy()
        self.origins = self.bt.training_origins(origin_idx, n_origins=n_origins)

        # Volume tiers, measured on pre-origin history only.
        hist = self.data.sales_wide[:, :origin_idx + 1].mean(axis=1)
        self.tier = pd.Series(pd.cut(
            hist[self.valid["series_idx"].to_numpy()],
            [-0.001, 0.2, 1.0, 3.0, np.inf],
            labels=["very low", "low", "medium", "high"]))
        self.high = (self.tier == "high").to_numpy()
        self.is_zero = self.y == 0

    def describe(self) -> dict:
        return {
            "validation_origin_day": self.window["forecast_origin_day"],
            "validation_days": self.window["validation_days"],
            "validation_dates": self.window["validation_dates"],
            "horizon": config.HORIZON,
            "n_series": config.N_SERIES,
            "validation_rows": int(len(self.valid)),
            "training_origins": [f"d_{o + 1}" for o in self.origins],
        }


def diagnostics(y: np.ndarray, p: np.ndarray, s: Setup) -> dict:
    """The numbers every optimization report quotes."""
    m = metrics.evaluate(y, p)
    hi = metrics.evaluate(y[s.high], p[s.high])
    tot_sq = ((y - p) ** 2).sum()
    return {
        **m,
        "high_volume_RMSE": hi["RMSE"],
        "high_volume_MAE": hi["MAE"],
        "high_volume_bias": hi["bias"],
        "high_volume_share_of_sq_error_pct": round(
            float(((y[s.high] - p[s.high]) ** 2).sum() / tot_sq * 100), 2),
        "mean_pred_on_zero_actual": round(float(p[s.is_zero].mean()), 5),
        "mean_pred_on_positive_actual": round(float(p[~s.is_zero].mean()), 5),
        "pred_mean": round(float(p.mean()), 5),
        "pred_p50": round(float(np.percentile(p, 50)), 5),
        "pred_p99": round(float(np.percentile(p, 99)), 5),
        "pred_max": round(float(p.max()), 4),
        "pct_predictions_below_0_5": round(float((p < 0.5).mean() * 100), 2),
    }


def build_matrix(s: Setup, cols: list[str], origins=None, verbose=False):
    """Assemble (X, y) over the training origins for the given feature set."""
    origins = s.origins if origins is None else origins
    n = config.N_SERIES
    rows_per = n * config.HORIZON
    X = np.empty((rows_per * len(origins), len(cols)), dtype=np.float32)
    Y = np.empty(rows_per * len(origins), dtype=np.float32)
    t0 = time.time()
    for i, o in enumerate(origins):
        f = s.fb.build_origin_frame(o, horizon=config.HORIZON, include_target=True)
        if int(f["target_day_idx"].max()) > s.origin_idx:
            raise AssertionError(
                f"LEAKAGE: origin d_{o+1} targets day {int(f['target_day_idx'].max())} "
                f"beyond cutoff {s.origin_idx}")
        a, b = i * rows_per, (i + 1) * rows_per
        X[a:b] = f[cols].to_numpy(np.float32)
        Y[a:b] = f["sales"].to_numpy(np.float32)
        del f
    if verbose:
        print(f"      matrix {X.shape} in {time.time()-t0:.0f}s "
              f"({X.nbytes/1e6:.0f} MB)")
    return X, Y


def train(X, Y, cols, params=None, n_estimators=N_ESTIMATORS, weights=None):
    p = dict(BASE_PARAMS)
    if params:
        p.update(params)
    ds = lgb.Dataset(X, label=Y, weight=weights, feature_name=list(cols),
                     categorical_feature=categorical_for(cols),
                     free_raw_data=True)
    t0 = time.time()
    booster = lgb.train(p, ds, num_boost_round=n_estimators,
                        callbacks=[lgb.log_evaluation(period=0)])
    return booster, {"params": p, "n_estimators": n_estimators,
                     "training_seconds": round(time.time() - t0, 1)}


def predict(booster, s: Setup, cols: list[str]) -> np.ndarray:
    return np.clip(booster.predict(s.valid[cols].to_numpy(np.float32)),
                   0, None).astype(np.float64)


def run(name: str, s: Setup, cols: list[str], *, params=None,
        n_estimators=N_ESTIMATORS, weights=None, label="", notes=(),
        save_model=True, save_preds=True, extra=None) -> dict:
    """Train + evaluate + record one experiment. Returns the diagnostics dict."""
    print(f"  [{name}] {label}  ({len(cols)} features)")
    exp = Experiment(
        name, model_type="LightGBM",
        objective=(params or {}).get("objective", BASE_PARAMS["objective"]),
        feature_set_label=label, n_features=len(cols), features=list(cols),
        **s.describe(), **(extra or {}),
    )
    for n in notes:
        exp.note(n)

    X, Y = build_matrix(s, cols)
    exp.set(training_rows=int(X.shape[0]))
    booster, info = train(X, Y, cols, params, n_estimators, weights)
    del X, Y
    exp.set(hyperparameters=info["params"], n_estimators=info["n_estimators"],
            training_seconds=info["training_seconds"])

    p = predict(booster, s, cols)
    if np.isnan(p).any():
        exp.error("NaN predictions"); exp.save(); raise SystemExit("STOP: NaN")

    d = diagnostics(s.y, p, s)
    exp.set_metrics(**d)
    exp.set(delta_rmse_vs_best=round(d["RMSE"] - BEST_RMSE, 6),
            delta_mae_vs_best=round(d["MAE"] - BEST_MAE, 6))

    if save_model:
        mp = config.MODELS_DIR / f"{name}.txt"
        booster.save_model(str(mp))
        exp.set(model_path=str(mp.relative_to(config.PROJECT_ROOT)))
    if save_preds:
        pp = config.PREDICTIONS_DIR / f"{name}_validation.csv"
        pd.DataFrame({
            "series_idx": s.valid["series_idx"].to_numpy(),
            "target_day_idx": s.valid["target_day_idx"].to_numpy(),
            "horizon": s.valid["horizon"].to_numpy(),
            "y_true": s.y, "y_pred": np.round(p, 5),
        }).to_csv(pp, index=False)
        exp.set(prediction_path=str(pp.relative_to(config.PROJECT_ROOT)))
    exp.save()

    print(f"      RMSE={d['RMSE']:.4f} ({d['RMSE']-BEST_RMSE:+.4f})   "
          f"MAE={d['MAE']:.4f} ({d['MAE']-BEST_MAE:+.4f})   "
          f"highvol RMSE={d['high_volume_RMSE']:.3f}  ({info['training_seconds']}s)")
    d["_name"] = name
    d["_label"] = label
    d["_train_s"] = info["training_seconds"]
    d["_n_features"] = len(cols)
    return d
