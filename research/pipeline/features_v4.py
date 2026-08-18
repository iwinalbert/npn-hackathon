
from __future__ import annotations

import numpy as np

from . import config
from .features_v2 import FeatureBuilderV2, BASE32

SHRINK_K = 20.0


def _shrink(ratio: np.ndarray, volume: np.ndarray, k: float = SHRINK_K) -> np.ndarray:
    w = volume / (volume + k)
    if ratio.ndim == 2:
        w = w[:, None]
    out = 1.0 + (ratio - 1.0) * w
    return np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)


class FeatureBuilderV4(FeatureBuilderV2):

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

        days = 52 * 7
        a = max(0, origin + 1 - days)
        blk = s[:, a:origin + 1].astype(np.float64)
        vol = blk.sum(axis=1)
        wknd_blk = wknd_all[a:origin + 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            we = blk[:, wknd_blk].mean(axis=1)
            wk = blk[:, ~wknd_blk].mean(axis=1)
            out["weekend_lift"] = _shrink(np.where(wk > 0, we / wk, 1.0), vol)

            snap_lift = np.ones(config.N_SERIES, dtype=np.float64)
            snap_blk = snap_m[a:origin + 1]
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

        for tag in ["52w", "13w"]:
            p = prof[f"wday_profile_{tag}"]
            vals = np.empty((horizon, n_s), dtype=np.float32)
            for i, w in enumerate(wd_t):
                vals[i] = p[series_idx, int(w)]
            frame[f"wday_ratio_{tag}"] = vals.ravel()

        frame["snap_lift"] = np.tile(prof["snap_lift"][series_idx].astype(np.float32), horizon)
        frame["weekend_lift"] = np.tile(prof["weekend_lift"][series_idx].astype(np.float32), horizon)
        return frame


V4_FEATURES = ["wday_ratio_52w", "wday_ratio_13w", "snap_lift", "weekend_lift"]


def feature_set() -> list[str]:
    return list(BASE32) + V4_FEATURES
