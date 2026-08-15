"""
Forecast accuracy metrics.

The team's reported benchmark (LightGBM + Tweedie: RMSE 2.0324, MAE 1.0869) is
quoted in RMSE and MAE, so those are the two headline numbers this pipeline
computes. They are calculated over every (series, horizon-day) prediction in the
validation window — 30,490 series x 28 days = 853,720 values — with no weighting
and no series excluded, which is the only way a later comparison against that
benchmark can be apples-to-apples.

WAPE is included as a supporting metric because RMSE and MAE on a mostly-zero
target are hard to interpret on their own: a model that predicts 0 everywhere
scores deceptively well on MAE. WAPE expresses total error as a share of total
actual demand, which makes that failure mode visible.
"""

from __future__ import annotations

import numpy as np


def _as_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if np.isnan(yt).any():
        raise ValueError(
            "y_true contains NaN — this usually means the evaluation window runs "
            "past the last day with known sales."
        )
    return yt, yp


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error. Punishes large misses more than small ones."""
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error. The average size of the miss, ignoring direction."""
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def wape(y_true, y_pred) -> float:
    """
    Weighted Absolute Percentage Error: sum|error| / sum(actual).

    Guards against the "predict zero everywhere" trap — that strategy scores
    a WAPE of 1.0 (100% of demand unexplained) however good its MAE looks.
    """
    yt, yp = _as_pair(y_true, y_pred)
    denom = np.abs(yt).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(yt - yp).sum() / denom)


def bias(y_true, y_pred) -> float:
    """Mean signed error. Positive => over-forecasting on average."""
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.mean(yp - yt))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """All headline metrics in one dict."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(np.asarray(y_true).size),
    }


def evaluate_by_group(y_true, y_pred, group_labels) -> dict:
    """
    Metrics broken out by an arbitrary grouping (category, store, horizon day...).

    The EDA showed 68.6% of all units are FOODS and the top 10% of series drive
    54.4% of volume, so a single pooled number can hide a model that is failing
    badly on the long tail. This makes that visible.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    g = np.asarray(group_labels).ravel()

    out = {}
    for lab in np.unique(g):
        m = g == lab
        out[str(lab)] = {
            "RMSE": rmse(yt[m], yp[m]),
            "MAE": mae(yt[m], yp[m]),
            "n": int(m.sum()),
        }
    return out
