
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
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def wape(y_true, y_pred) -> float:
    yt, yp = _as_pair(y_true, y_pred)
    denom = np.abs(yt).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(yt - yp).sum() / denom)


def bias(y_true, y_pred) -> float:
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.mean(yp - yt))


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(np.asarray(y_true).size),
    }


def evaluate_by_group(y_true, y_pred, group_labels) -> dict:
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
