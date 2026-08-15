"""
Per-series RESPONSE-SHAPE features.

WHY THESE ARE A DIFFERENT FAMILY FROM EVERYTHING ALREADY REJECTED
-----------------------------------------------------------------
Phase 2 tested fourteen features and none helped, but every one of them
described a series' LEVEL: rolling means over new windows, zero counts,
momentum ratios, price movements. Experiment #71's year-over-year features were
also level (last year's level).

These describe a series' SHAPE — how its demand is distributed across the week,
and how strongly it responds to SNAP and to weekends, relative to its own
average. The champion has `wday`, `is_weekend`, `snap` and `item_id`, so in
principle it could learn a 3,049 x 7 interaction from categorical splits, but
that is precisely the kind of high-cardinality interaction gradient-boosted
trees represent poorly.

EVIDENCE
--------
A read-only diagnostic (scripts/06_research_campaign/33_segmentation_diagnostic.py)
showed that a purely arithmetic predictor

    level(origin) x weekday_ratio(52 weeks of history)

beats a level-only predictor by 0.0578 RMSE out of sample. The signal is real.
The same diagnostic showed only ~10% of the per-series-x-weekday oracle gap is
recoverable from history, so expectations are modest.

LEAKAGE
-------
Every ratio is computed from sales strictly at or before the forecast origin.
The only target-day input is the calendar (which weekday it is, whether SNAP is
active), which is published years ahead. Verified by corruption test in the
experiment script rather than asserted.

SHRINKAGE
---------
A ratio measured on a series with almost no sales is noise. Each ratio is pulled
toward 1.0 with weight n/(n+k), where n is the series' observed unit volume in
the window - the same guard used in Experiment #69.
"""

from __future__ import annotations

import numpy as np

from . import config
from .features_v2 import FeatureBuilderV2, BASE32

SHRINK_K = 20.0


def _shrink(ratio: np.ndarray, volume: np.ndarray, k: float = SHRINK_K) -> np.ndarray:
    """Pull a ratio toward 1.0 when the series has little volume behind it."""
    w = volume / (volume + k)
    if ratio.ndim == 2:
        w = w[:, None]
    out = 1.0 + (ratio - 1.0) * w
    return np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)


class FeatureBuilderV4(FeatureBuilderV2):
    """FeatureBuilderV2 + four per-series response-shape features."""

    def _shape_profiles(self, origin: int) -> dict:
        s = self.d.sales_wide
        cal = self.d.calendar
        wd_all = cal["wday"].to_numpy()
        wknd_all = cal["is_weekend"].to_numpy().astype(bool)
        snap_m = self.d.snap_matrix
        snap_col = self.d.snap_col_of_series

        out = {}
        for weeks, tag in [(52, "52w"), (13, "13w")]:
            days = weeks * 7
            a = max(0, origin + 1 - days)
            blk = s[:, a:origin + 1].astype(np.float64)
            wd_blk = wd_all[a:origin + 1]
            overall = blk.mean(axis=1)
            vol = blk.sum(axis=1)

            prof = np.ones((config.N_SERIES, 8), dtype=np.float64)
            for w in range(1, 8):
                m = wd_blk == w
                if m.any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        prof[:, w] = np.where(overall > 0,
                                              blk[:, m].mean(axis=1) / overall, 1.0)
            out[f"wday_profile_{tag}"] = _shrink(prof, vol)

        # SNAP and weekend lift, 52 weeks, per series
        days = 52 * 7
        a = max(0, origin + 1 - days)
        blk = s[:, a:origin + 1].astype(np.float64)
        vol = blk.sum(axis=1)
        wknd_blk = wknd_all[a:origin + 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            we = blk[:, wknd_blk].mean(axis=1)
            wk = blk[:, ~wknd_blk].mean(axis=1)
            out["weekend_lift"] = _shrink(np.where(wk > 0, we / wk, 1.0), vol)

            # SNAP flag differs per state, so this must be done per state group
            snap_lift = np.ones(config.N_SERIES, dtype=np.float64)
            snap_blk = snap_m[a:origin + 1]              # (days, 3)
            for st in range(3):
                rows = snap_col == st
                if not rows.any():
                    continue
                on = snap_blk[:, st] == 1
                if on.any() and (~on).any():
                    m_on = blk[np.ix_(rows, np.where(on)[0])].mean(axis=1)
                    m_off = blk[np.ix_(rows, np.where(~on)[0])].mean(axis=1)
                    snap_lift[rows] = np.where(m_off > 0, m_on / m_off, 1.0)
            out["snap_lift"] = _shrink(snap_lift, vol)

        return out

    def build_origin_frame(self, origin_idx, horizon=config.HORIZON,
                           series_idx=None, include_target=True):
        frame = super().build_origin_frame(
            origin_idx, horizon=horizon, series_idx=series_idx,
            include_target=include_target)

        if series_idx is None:
            series_idx = np.arange(self.d.sales_wide.shape[0])
        n_s = len(series_idx)
        target_days = origin_idx + 1 + np.arange(horizon)
        wd_t = self.d.calendar["wday"].to_numpy()[target_days]

        prof = self._shape_profiles(origin_idx)

        # weekday ratio for each target day's own weekday (varies across horizon)
        for tag in ["52w", "13w"]:
            p = prof[f"wday_profile_{tag}"]
            vals = np.empty((horizon, n_s), dtype=np.float32)
            for i, w in enumerate(wd_t):
                vals[i] = p[series_idx, int(w)]
            frame[f"wday_ratio_{tag}"] = vals.ravel()

        # per-series lifts (constant across the horizon; combine with the
        # existing target-day `snap` / `is_weekend` columns)
        frame["snap_lift"] = np.tile(prof["snap_lift"][series_idx].astype(np.float32), horizon)
        frame["weekend_lift"] = np.tile(prof["weekend_lift"][series_idx].astype(np.float32), horizon)
        return frame


V4_FEATURES = ["wday_ratio_52w", "wday_ratio_13w", "snap_lift", "weekend_lift"]


def feature_set() -> list[str]:
    return list(BASE32) + V4_FEATURES
