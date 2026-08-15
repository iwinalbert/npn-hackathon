"""
Year-over-year features — the last genuinely NEW information source available.

WHY THESE ARE DIFFERENT FROM THE FEATURES THAT ALREADY FAILED
-------------------------------------------------------------
Phase 2 added fourteen features and none helped. But every one of them was a
re-encoding of information the model already had: rolling_mean_14 and
demand_momentum sit between the existing 7- and 28-day windows;
rolling_zero_count_7 restates recency; the calendar and interaction terms are
combinations of columns already present.

These are different in kind. `lag_364` is what this exact product sold in this
exact store on the same weekday one year ago. No existing feature contains it —
the longest lookback in the champion is 28 days, and `month` / `week_of_year`
capture only a chain-wide seasonal average, not a per-series one.

LEAKAGE
-------
For a target day t = T + h with h <= 28, a 364-day lookback reads day t - 364,
which is at most T - 336. Every window used here therefore ends far before the
forecast origin, for every horizon day. Verified by corruption test in the
experiment script rather than asserted.

364 rather than 365: it is exactly 52 weeks, so the same-weekday alignment that
matters for retail demand is preserved.
"""

from __future__ import annotations

import numpy as np

from . import config
from .features_v2 import FeatureBuilderV2

YEAR = 364


class FeatureBuilderV3(FeatureBuilderV2):
    """FeatureBuilderV2 + year-over-year demand features."""

    def build_origin_frame(self, origin_idx, horizon=config.HORIZON,
                           series_idx=None, include_target=True):
        frame = super().build_origin_frame(
            origin_idx, horizon=horizon, series_idx=series_idx,
            include_target=include_target)

        s = self.d.sales_wide
        if series_idx is None:
            series_idx = np.arange(s.shape[0])
        n_s = len(series_idx)
        target_days = origin_idx + 1 + np.arange(horizon)

        # ---- target-day relative: same weekday one year earlier ----
        lag_y = np.empty((horizon, n_s), dtype=np.float32)
        wk_y = np.empty((horizon, n_s), dtype=np.float32)
        for i, t in enumerate(target_days):
            src = int(t) - YEAR
            if src < 0:
                lag_y[i] = np.nan
                wk_y[i] = np.nan
                continue
            lag_y[i] = s[series_idx, src].astype(np.float32)
            a = max(0, src - 3)
            b = min(s.shape[1], src + 4)          # +/- 3 days around it
            wk_y[i] = s[series_idx, a:b].astype(np.float64).mean(axis=1)
        frame["lag_364"] = lag_y.ravel()
        frame["rolling_mean_7_lag364"] = wk_y.ravel()

        # ---- origin-relative: how does this year compare with last year? ----
        cur_a, cur_b = max(0, origin_idx - 27), origin_idx + 1
        cur = s[series_idx, cur_a:cur_b].astype(np.float64).mean(axis=1)

        py_end = origin_idx - YEAR
        if py_end >= 0:
            py = s[series_idx, max(0, py_end - 27):py_end + 1].astype(np.float64).mean(axis=1)
        else:
            py = np.full(n_s, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            yoy = np.where(py > 0, cur / py, np.nan).astype(np.float32)
        frame["yoy_level_ratio"] = np.tile(yoy, horizon)
        frame["rolling_mean_28_lag364"] = np.tile(py.astype(np.float32), horizon)

        return frame


V3_FEATURES = ["lag_364", "rolling_mean_7_lag364",
               "yoy_level_ratio", "rolling_mean_28_lag364"]


def feature_set() -> list[str]:
    from .features_v2 import BASE32
    return list(BASE32) + V3_FEATURES
