"""
CROSS-SERIES features: what the rest of the chain is doing right now.

THE GAP THIS FILLS
------------------
Across Experiments #1-#78 every feature has been a function of ONE series' own
history, plus static hierarchy codes (item_id, dept_id, cat_id, store_id,
state_id) that tell the model WHICH item this is but nothing about how that item
is behaving anywhere else.

That matters because it explains why so many feature families were rejected.
rolling_mean_14, demand_momentum_7_28, the year-over-year set, the price-change
set - all of them are transformations of the same own-series signal, so they are
heavily collinear with lag_1/7/14/28 and rolling_mean_7/28, which the model
already has. Adding a re-encoding of information you already hold buys nothing.

The same item's recent sales in the OTHER NINE STORES is not a re-encoding. It
is an independent measurement of that item's current demand state, taken from
data this series has never seen. A national promotion, a supply problem, a
seasonal turn or a viral product shows up across the chain before it can be
distinguished from noise in one store's thin daily counts.

WHY RATIOS AND NOT LEVELS
-------------------------
The clearest lesson of the campaign so far: LEVEL features fail, SHAPE features
work. Phase 2 tested fourteen level features and none helped; Experiment #71's
year-over-year levels were rejected; but the per-series shape ratios of
Experiments #72-#74 were accepted across 4 windows and 3 seeds.

So every feature here is a RATIO describing a state or a trend, never a level:

    xstore_momentum     item's other-store mean over 28d
                        / item's other-store mean over 182d
                        -> is this item trending up or down chain-wide?

    xstore_rel_level    item's other-store mean over 28d
                        / this series' own mean over 28d
                        -> is this store over- or under-indexing on this item?

    store_dept_momentum this store x dept mean over 28d
                        / this store x dept mean over 182d
                        -> is footfall in this part of this store rising?

SELF-EXCLUSION
--------------
The cross-store aggregates exclude the series itself. Including it would smuggle
own-series level back in and make the feature partly a restatement of
rolling_mean_28, which is exactly the failure mode this module exists to avoid.

LEAKAGE
-------
Every quantity is computed from sales at or before the forecast origin, and the
values are constant across the 28-day horizon (like snap_lift and weekend_lift
before them). No target-day input is used at all. Verified by corruption test in
the experiment script rather than asserted here.

SHRINKAGE
---------
A ratio measured on thin data is noise, so each is pulled toward 1.0 with weight
n/(n+k) on the volume behind it - the same guard used in Experiments #69 and #72.
"""

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
    """The champion's 38 features plus three cross-series ratios."""

    def __init__(self, data):
        super().__init__(data)
        m = self.d.series_meta
        self._item_code = m["item_id_code"].to_numpy().astype(np.int64)
        # store x dept group, one code per (store, dept) pair
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
        """Mean of `vals` over each group, EXCLUDING the row itself."""
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

        # --- item across the OTHER stores -------------------------------
        oth_s = self._group_other_mean(own_s, self._item_code, self._n_item)
        oth_l = self._group_other_mean(own_l, self._item_code, self._n_item)
        vol_item = oth_l * LONG_DAYS

        xstore_momentum = _shrink(_safe_ratio(oth_s, oth_l), vol_item)
        xstore_rel_level = _shrink(_safe_ratio(oth_s, own_s),
                                   np.minimum(own_s, oth_s) * SHORT_DAYS)

        # --- this store x dept ------------------------------------------
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
