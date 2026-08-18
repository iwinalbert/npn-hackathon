"""
The CHAMPION feature builder: 38 features (Experiments #72-#75).

Until now this class lived inside the one-off script that created it
(scripts/06_research_campaign/36_exp74_reproduce_and_extend.py). Every later
experiment that wants to build on the champion needs it, so it is promoted here
unchanged. The logic is copied verbatim from that script - this module adds no
behaviour, it only gives the champion a permanent home.

WHAT THE 38 FEATURES ARE
------------------------
    BASE32          the original champion feature set (features.py, groups A-G)
    + V4_FEATURES   wday_ratio_52w, wday_ratio_13w, snap_lift, weekend_lift
                    per-series WEEKLY shape - Experiment #72, validated in #73
                    across 4 windows and 3 seeds
    + V5_FEATURES   month_ratio, dom_ratio
                    per-series month-of-year and day-of-month shape -
                    Experiment #74 part B

Measured on the primary window (d_1914-d_1941): RMSE 2.1157, MAE 1.0287.

LEAKAGE
-------
Inherited unchanged from FeatureBuilderV4: every ratio is computed from sales
strictly at or before the forecast origin. The only target-day inputs are
calendar facts (which month it is, which day of the month), published years in
advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features_v2 import BASE32
from .features_v4 import FeatureBuilderV4, V4_FEATURES, _shrink

# Two years of history where available, matching Experiment #74.
CYCLE_WINDOW_DAYS = 728


class FeatureBuilderV5(FeatureBuilderV4):
    """V4 + per-series month-of-year and day-of-month profiles."""

    def _cycle_profiles(self, origin: int) -> dict:
        s = self.d.sales_wide
        cal = self.d.calendar
        a = max(0, origin + 1 - CYCLE_WINDOW_DAYS)
        blk = s[:, a:origin + 1].astype(np.float64)
        vol = blk.sum(axis=1)
        overall = blk.mean(axis=1)

        out = {}
        for col, n, name in [("month", 13, "month"), ("dom", 32, "dom")]:
            if col == "dom":
                key = pd.to_datetime(cal["date"]).dt.day.to_numpy()[a:origin + 1]
            else:
                key = cal["month"].to_numpy()[a:origin + 1]
            prof = np.ones((config.N_SERIES, n), dtype=np.float64)
            for v in range(1, n):
                m = key == v
                if m.any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        prof[:, v] = np.where(overall > 0,
                                              blk[:, m].mean(axis=1) / overall, 1.0)
            out[name] = _shrink(prof, vol)
        return out

    def build_origin_frame(self, origin_idx, horizon=config.HORIZON,
                           series_idx=None, include_target=True):
        frame = super().build_origin_frame(origin_idx, horizon=horizon,
                                           series_idx=series_idx,
                                           include_target=include_target)
        if series_idx is None:
            series_idx = np.arange(self.d.sales_wide.shape[0])
        n_s = len(series_idx)
        td = origin_idx + 1 + np.arange(horizon)
        cal = self.d.calendar
        months = cal["month"].to_numpy()[td]
        doms = pd.to_datetime(cal["date"]).dt.day.to_numpy()[td]

        prof = self._cycle_profiles(origin_idx)
        for name, keys in [("month_ratio", months), ("dom_ratio", doms)]:
            p = prof["month" if name == "month_ratio" else "dom"]
            vals = np.empty((horizon, n_s), dtype=np.float32)
            for i, k in enumerate(keys):
                vals[i] = p[series_idx, int(k)]
            frame[name] = vals.ravel()
        return frame


V5_FEATURES = ["month_ratio", "dom_ratio"]

#: The champion feature set, in the exact order Experiment #74 recorded it.
CHAMPION_FEATURES = list(BASE32) + list(V4_FEATURES) + V5_FEATURES

#: Measured champion scores on the primary window, for delta reporting.
CHAMPION_RMSE = 2.1156930820206945
CHAMPION_MAE = 1.0286892499701086


def feature_set() -> list[str]:
    return list(CHAMPION_FEATURES)
