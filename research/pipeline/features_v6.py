
from __future__ import annotations

import numpy as np

from . import config
from .features_v5 import FeatureBuilderV5, CHAMPION_FEATURES

SHORT_DAYS = 28
LONG_DAYS = 182
SHRINK_K = 20.0


def _shrink(ratio: np.ndarray, volume: np.ndarray, k: float = SHRINK_K) -> np.ndarray:
    w = volume / (volume + k)
    out = 1.0 + (ratio - 1.0) * w
    return np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 1.0)
    return np.nan_to_num(r, nan=1.0, posinf=1.0, neginf=1.0)


class FeatureBuilderV6(FeatureBuilderV5):

    def __init__(self, data):
        super().__init__(data)
        m = self.d.series_meta
        self._item_code = m["item_id_code"].to_numpy().astype(np.int64)
        self._sd_code = (m["store_id_code"].to_numpy().astype(np.int64)
                         * 100 + m["dept_id_code"].to_numpy().astype(np.int64))
        _, self._sd_code = np.unique(self._sd_code, return_inverse=True)
        self._n_item = int(self._item_code.max()) + 1
        self._n_sd = int(self._sd_code.max()) + 1

    def _window_mean(self, origin: int, days: int) -> np.ndarray:
        a = max(0, origin + 1 - days)
        blk = self.d.sales_wide[:, a:origin + 1]
        return blk.astype(np.float64).mean(axis=1)

    @staticmethod
    def _group_other_mean(vals, codes, n_groups):
        tot = np.bincount(codes, weights=vals, minlength=n_groups)
        cnt = np.bincount(codes, minlength=n_groups).astype(np.float64)
        other_sum = tot[codes] - vals
        other_cnt = cnt[codes] - 1.0
        return np.where(other_cnt > 0, other_sum / np.maximum(other_cnt, 1.0), vals)

    @staticmethod
    def _group_mean(vals, codes, n_groups):
        tot = np.bincount(codes, weights=vals, minlength=n_groups)
        cnt = np.bincount(codes, minlength=n_groups).astype(np.float64)
        return (tot / np.maximum(cnt, 1.0))[codes]

    def _cross_series(self, origin: int) -> dict:
        own_s = self._window_mean(origin, SHORT_DAYS)
        own_l = self._window_mean(origin, LONG_DAYS)

        oth_s = self._group_other_mean(own_s, self._item_code, self._n_item)
        oth_l = self._group_other_mean(own_l, self._item_code, self._n_item)
        vol_item = oth_l * LONG_DAYS

        xstore_momentum = _shrink(_safe_ratio(oth_s, oth_l), vol_item)
        xstore_rel_level = _shrink(_safe_ratio(oth_s, own_s),
                                   np.minimum(own_s, oth_s) * SHORT_DAYS)

        sd_s = self._group_mean(own_s, self._sd_code, self._n_sd)
        sd_l = self._group_mean(own_l, self._sd_code, self._n_sd)
        store_dept_momentum = _shrink(_safe_ratio(sd_s, sd_l), sd_l * LONG_DAYS)

        return {
            "xstore_momentum": xstore_momentum,
            "xstore_rel_level": xstore_rel_level,
            "store_dept_momentum": store_dept_momentum,
        }

    def build_origin_frame(self, origin_idx, horizon=config.HORIZON,
                           series_idx=None, include_target=True):
        frame = super().build_origin_frame(origin_idx, horizon=horizon,
                                           series_idx=series_idx,
                                           include_target=include_target)
        if series_idx is None:
            series_idx = np.arange(self.d.sales_wide.shape[0])
        xs = self._cross_series(origin_idx)
        for name, vals in xs.items():
            frame[name] = np.tile(vals[series_idx].astype(np.float32), horizon)
        return frame


V6_FEATURES = ["xstore_momentum", "xstore_rel_level", "store_dept_momentum"]

CHAMPION_PLUS_XSERIES = list(CHAMPION_FEATURES) + V6_FEATURES


def feature_set() -> list[str]:
    return list(CHAMPION_PLUS_XSERIES)
