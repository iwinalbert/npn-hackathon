"""
AGGREGATE-LEVEL modelling: forecasting a hierarchy level directly, so that its
forecast can be reconciled back down onto the store-item series.

NEW MODULE. Nothing here modifies or imports-and-mutates any existing pipeline
component; the protected champion path is untouched.

WHY A SEPARATE BUILDER INSTEAD OF REUSING FeatureBuilder
--------------------------------------------------------
FeatureBuilder (and its V2/V4/V5 subclasses) hard-code config.N_SERIES = 30490 in
several places and read a per-series SNAP column keyed to one state. An item-level
series is a sum over ten stores in three states, so neither assumption holds. The
honest fix is a small purpose-built builder rather than bending the verified one
and risking a silent change to the champion's own feature values.

THE ONE RULE IS UNCHANGED
-------------------------
Standing at origin T we may use:
  * aggregated sales up to and including T  (ORIGIN-RELATIVE, held constant
    across all 28 horizon days)
  * the calendar and sell_prices for the target days, both of which the dataset
    genuinely publishes in advance  (TARGET-DAY)
and nothing else. Every array below is sliced with [:, :T+1] or indexed by a
target day, never both.

WHAT AN AGGREGATE MODEL KNOWS THAT THE BOTTOM MODEL DOES NOT
-------------------------------------------------------------
The bottom model sees one store's history of an item: a thin, mostly-zero count
series. Its implied item-level forecast is the sum of ten such independent
estimates. This model sees the pooled item history directly, so its lag and
rolling features carry roughly ten times the counts and a correspondingly smaller
estimation variance. That is a genuinely different information set, not a
re-encoding of one the bottom model already holds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .data_loader import M5Data

NOT_YET = -1
SHRINK_K = 20.0
WDAY_WINDOW_DAYS = 364        # 52 weeks


class AggregateLevel:
    """
    The bottom panel summed onto a named hierarchy level.

    level = "item"        3,049 groups  (one item across all 10 stores)
    level = "item_state"  9,147 groups
    level = "store_dept"     70 groups
    """

    LEVEL_KEYS = {
        "item": ["item_id"],
        "item_state": ["item_id", "state_id"],
        "item_cat": ["item_id"],
        "store_dept": ["store_id", "dept_id"],
    }

    def __init__(self, data: M5Data, level: str = "item"):
        if level not in self.LEVEL_KEYS:
            raise KeyError(f"unknown level '{level}'")
        self.level = level
        self.calendar = data.calendar
        self.day_to_week = data.day_to_week

        meta = data.series_meta
        keys = self.LEVEL_KEYS[level]
        key = meta[keys[0]].astype(str)
        for c in keys[1:]:
            key = key + "|" + meta[c].astype(str)
        codes, uniques = pd.factorize(key)
        self.group_of_series = codes.astype(np.int32)
        self.n_groups = len(uniques)

        # --- aggregated sales ------------------------------------------
        n_days = data.sales_wide.shape[1]
        self.sales_wide = np.zeros((self.n_groups, n_days), dtype=np.float32)
        np.add.at(self.sales_wide, self.group_of_series,
                  data.sales_wide.astype(np.float32))

        # --- group metadata (first member's hierarchy codes) ------------
        first_row = (pd.Series(np.arange(len(meta)))
                     .groupby(self.group_of_series).first().to_numpy())
        self.meta = pd.DataFrame({
            "item_code": meta["item_id_code"].to_numpy()[first_row],
            "dept_code": meta["dept_id_code"].to_numpy()[first_row],
            "cat_code": meta["cat_id_code"].to_numpy()[first_row],
            "store_code": meta["store_id_code"].to_numpy()[first_row],
            "state_code": meta["state_id_code"].to_numpy()[first_row],
        })
        self.members_per_group = np.bincount(
            self.group_of_series, minlength=self.n_groups).astype(np.float32)

        # --- aggregated price ------------------------------------------
        # Mean sell_price over the members that HAVE a price that week, plus the
        # count of members listing it. Both are published in advance for the
        # forecast weeks, exactly like the bottom-level price.
        if data.price_wide is not None:
            pw = data.price_wide
            priced = ~np.isnan(pw)
            s = np.zeros((self.n_groups, pw.shape[1]), dtype=np.float64)
            c = np.zeros((self.n_groups, pw.shape[1]), dtype=np.float64)
            np.add.at(s, self.group_of_series, np.nan_to_num(pw, nan=0.0))
            np.add.at(c, self.group_of_series, priced.astype(np.float64))
            with np.errstate(divide="ignore", invalid="ignore"):
                self.price_wide = np.where(c > 0, s / c, np.nan).astype(np.float32)
            self.n_priced = c.astype(np.float32)
        else:
            self.price_wide = None
            self.n_priced = None

        # --- SNAP at aggregate level -----------------------------------
        # An item-level group spans three states, so no single state flag
        # applies. The number of states with SNAP active that day is the
        # aggregate-level analogue, and is equally known in advance.
        snap = data.calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(np.int8)
        self.snap_count = snap.sum(axis=1).astype(np.int8)

        # --- event codes ------------------------------------------------
        self.event_codes = {}
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            vals = sorted(data.calendar[col].dropna().unique())
            m = {v: i + 1 for i, v in enumerate(vals)}
            self.event_codes[col] = (data.calendar[col].map(m).fillna(0)
                                     .to_numpy(np.int16))

        # --- first-ever sale (masked at the origin when used) -----------
        has_any = (self.sales_wide > 0).any(axis=1)
        self.first_sale_idx_full = np.where(
            has_any, np.argmax(self.sales_wide > 0, axis=1), 10 ** 6).astype(np.int32)

    def describe(self) -> dict:
        return {
            "level": self.level,
            "n_groups": int(self.n_groups),
            "mean_members_per_group": float(self.members_per_group.mean()),
            "n_days": int(self.sales_wide.shape[1]),
            "pct_zero_cells": float((self.sales_wide == 0).mean() * 100),
            "mean_daily_units": float(self.sales_wide.mean()),
        }


def _shrink(ratio: np.ndarray, volume: np.ndarray, k: float = SHRINK_K) -> np.ndarray:
    w = volume / (volume + k)
    if ratio.ndim == 2:
        w = w[:, None]
    return np.nan_to_num(1.0 + (ratio - 1.0) * w, nan=1.0, posinf=1.0, neginf=1.0)


class AggFeatureBuilder:
    """Leakage-safe direct-28-day feature frames for one AggregateLevel."""

    def __init__(self, agg: AggregateLevel):
        self.a = agg

    # ------------------------------------------------------------------
    def _demand(self, origin: int) -> dict[str, np.ndarray]:
        s = self.a.sales_wide
        out = {}
        for k in (1, 7, 14, 28):
            d = origin - k + 1
            out[f"lag_{k}"] = (s[:, d] if d >= 0
                               else np.full(s.shape[0], np.nan, np.float32)).astype(np.float32)
        for w in (7, 28, 91):
            blk = s[:, max(0, origin - w + 1):origin + 1].astype(np.float64)
            out[f"rolling_mean_{w}"] = blk.mean(axis=1).astype(np.float32)
            if w in (7, 28):
                out[f"rolling_std_{w}"] = blk.std(axis=1).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["momentum_28_91"] = np.where(
                out["rolling_mean_91"] > 0,
                out["rolling_mean_28"] / out["rolling_mean_91"],
                np.nan).astype(np.float32)
        return out

    # ------------------------------------------------------------------
    def _wday_ratio(self, origin: int) -> np.ndarray:
        """
        Per-group weekday profile over the trailing 52 weeks, shrunk toward 1.

        At bottom level this family was validated across four windows and three
        seeds (Experiments #72-#74). At aggregate level it is estimated from ~10x
        the counts, so it is strictly better determined here than there.
        """
        s = self.a.sales_wide
        a = max(0, origin + 1 - WDAY_WINDOW_DAYS)
        blk = s[:, a:origin + 1].astype(np.float64)
        wd = self.a.calendar["wday"].to_numpy()[a:origin + 1]
        overall = blk.mean(axis=1)
        vol = blk.sum(axis=1)
        prof = np.ones((s.shape[0], 8), dtype=np.float64)
        for w in range(1, 8):
            m = wd == w
            if m.any():
                with np.errstate(divide="ignore", invalid="ignore"):
                    prof[:, w] = np.where(overall > 0,
                                          blk[:, m].mean(axis=1) / overall, 1.0)
        return _shrink(prof, vol)

    # ------------------------------------------------------------------
    def _price(self, origin: int, target_days: np.ndarray):
        a = self.a
        if a.price_wide is None:
            n, h = a.n_groups, len(target_days)
            return (np.full(n, np.nan, np.float32),
                    np.full((h, n), np.nan, np.float32),
                    np.full((h, n), np.nan, np.float32))
        w_o = int(a.day_to_week[origin])
        w_s = max(0, w_o - config.PRICE_AVG_WINDOW_WEEKS + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            recent = np.nanmean(a.price_wide[:, w_s:w_o + 1].astype(np.float64),
                                axis=1).astype(np.float32)
        weeks = a.day_to_week[target_days]
        return recent, a.price_wide[:, weeks].T, a.n_priced[:, weeks].T

    # ------------------------------------------------------------------
    def build_origin_frame(self, origin_idx: int, horizon: int = config.HORIZON,
                           include_target: bool = True) -> pd.DataFrame:
        a = self.a
        n = a.n_groups
        target_days = origin_idx + 1 + np.arange(horizon)
        if target_days.max() >= config.N_CALENDAR_DAYS:
            raise ValueError("horizon runs past the end of calendar.csv")

        cal = a.calendar
        cols: dict[str, np.ndarray] = {
            "group_idx": np.tile(np.arange(n, dtype=np.int32), horizon),
            "target_day_idx": np.repeat(target_days.astype(np.int32), n),
            "horizon": np.repeat(np.arange(1, horizon + 1, dtype=np.int8), n),
        }

        # --- calendar (target-day) --------------------------------------
        for name, arr in [
            ("wday", cal["wday"].to_numpy(np.int16)),
            ("month", cal["month"].to_numpy(np.int16)),
            ("year", cal["year"].to_numpy(np.int16)),
            ("is_weekend", cal["is_weekend"].to_numpy(np.int8)),
            ("snap_count", a.snap_count),
            ("event_name_1", a.event_codes["event_name_1"]),
            ("event_type_1", a.event_codes["event_type_1"]),
            ("event_name_2", a.event_codes["event_name_2"]),
            ("event_type_2", a.event_codes["event_type_2"]),
        ]:
            cols[name] = np.repeat(arr[target_days], n)

        # --- demand (origin-relative) -----------------------------------
        for name, arr in self._demand(origin_idx).items():
            cols[name] = np.tile(arr, horizon)

        # --- weekday shape ----------------------------------------------
        prof = self._wday_ratio(origin_idx)
        wd_t = cal["wday"].to_numpy()[target_days]
        cols["wday_ratio_52w"] = np.concatenate(
            [prof[:, w].astype(np.float32) for w in wd_t])

        # --- recency ------------------------------------------------------
        first = np.where(a.first_sale_idx_full <= origin_idx,
                         a.first_sale_idx_full, NOT_YET)
        dsf = np.where(first >= 0, origin_idx - first, NOT_YET).astype(np.float32)
        cols["days_since_first_sale"] = np.tile(dsf, horizon)

        # --- hierarchy ----------------------------------------------------
        for c in ("item_code", "dept_code", "cat_code"):
            cols[c] = np.tile(a.meta[c].to_numpy(np.int16), horizon)
        cols["n_members"] = np.tile(a.members_per_group, horizon)

        # --- price ---------------------------------------------------------
        recent, price_t, npriced_t = self._price(origin_idx, target_days)
        cols["recent_avg_price"] = np.tile(recent, horizon)
        cols["sell_price"] = price_t.astype(np.float32).ravel()
        cols["n_stores_priced"] = npriced_t.astype(np.float32).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["price_rel_to_recent_avg"] = (
                cols["sell_price"] / cols["recent_avg_price"]).astype(np.float32)

        frame = pd.DataFrame(cols)

        if include_target:
            known = target_days <= config.LAST_KNOWN_DAY_IDX
            y = np.full((horizon, n), np.nan, dtype=np.float32)
            if known.any():
                y[known] = a.sales_wide[:, target_days[known]].T
            frame["sales"] = y.ravel()
        return frame


AGG_FEATURES = [
    # calendar (target-day)
    "wday", "month", "year", "is_weekend", "snap_count",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    # demand (origin-relative)
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_mean_91",
    "rolling_std_7", "rolling_std_28", "momentum_28_91",
    # shape
    "wday_ratio_52w",
    # recency
    "days_since_first_sale",
    # hierarchy
    "item_code", "dept_code", "cat_code", "n_members",
    # price
    "sell_price", "recent_avg_price", "price_rel_to_recent_avg", "n_stores_priced",
    # horizon
    "horizon",
]

AGG_CATEGORICAL = ["wday", "month", "year", "event_name_1", "event_type_1",
                   "event_name_2", "event_type_2", "item_code", "dept_code",
                   "cat_code"]


def agg_categorical_for(cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(cols) if c in AGG_CATEGORICAL]
