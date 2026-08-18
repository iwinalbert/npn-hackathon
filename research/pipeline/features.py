
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .data_loader import M5Data


NOT_YET = -1


class FeatureBuilder:

    def __init__(self, data: M5Data):
        self.d = data
        self._precompute()


    def _precompute(self) -> None:
        sales = self.d.sales_wide
        n_days = sales.shape[1]

        pos = np.arange(n_days, dtype=np.int16)
        tmp = np.where(sales > 0, pos[None, :], np.int16(-1))
        self.last_nonzero_upto = np.maximum.accumulate(tmp, axis=1)
        del tmp

        has_any = (sales > 0).any(axis=1)
        first_sale = np.argmax(sales > 0, axis=1).astype(np.int32)
        self.first_sale_idx_full = np.where(has_any, first_sale, np.int32(10**6))

        if self.d.price_wide is not None:
            priced = ~np.isnan(self.d.price_wide)
            has_price = priced.any(axis=1)
            first_price_week = np.argmax(priced, axis=1).astype(np.int32)
            week_first_day = (
                self.d.calendar.groupby("week_idx")["day_idx"].min()
                .reindex(range(self.d.price_wide.shape[1])).to_numpy()
            )
            self.first_price_day_idx_full = np.where(
                has_price, week_first_day[first_price_week], np.int32(10**6)
            ).astype(np.int32)
        else:
            self.first_price_day_idx_full = np.full(config.N_SERIES, 10**6, dtype=np.int32)

        self.event_maps = {}
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            vals = sorted(self.d.calendar[col].dropna().unique())
            self.event_maps[col] = {v: i + 1 for i, v in enumerate(vals)}

        self.event_codes = {
            col: self.d.calendar[col].map(self.event_maps[col]).fillna(0).to_numpy(np.int16)
            for col in self.event_maps
        }


    def _calendar_features(self, target_days: np.ndarray) -> dict[str, np.ndarray]:
        cal = self.d.calendar
        return {
            "wday": cal["wday"].to_numpy(np.int16)[target_days],
            "month": cal["month"].to_numpy(np.int16)[target_days],
            "year": cal["year"].to_numpy(np.int16)[target_days],
            "is_weekend": cal["is_weekend"].to_numpy(np.int8)[target_days],
            "event_name_1": self.event_codes["event_name_1"][target_days],
            "event_type_1": self.event_codes["event_type_1"][target_days],
            "event_name_2": self.event_codes["event_name_2"][target_days],
            "event_type_2": self.event_codes["event_type_2"][target_days],
        }


    def _demand_features(self, origin: int) -> dict[str, np.ndarray]:
        sales = self.d.sales_wide
        out: dict[str, np.ndarray] = {}

        for k in config.LAGS:
            day = origin - k + 1
            if day < 0:
                out[f"lag_{k}"] = np.full(sales.shape[0], np.nan, dtype=np.float32)
            else:
                out[f"lag_{k}"] = sales[:, day].astype(np.float32)

        for w in config.ROLLING_WINDOWS:
            start = max(0, origin - w + 1)
            window = sales[:, start:origin + 1].astype(np.float64)
            out[f"rolling_mean_{w}"] = window.mean(axis=1).astype(np.float32)
            out[f"rolling_std_{w}"] = window.std(axis=1).astype(np.float32)

        return out


    def _recency_features(self, origin: int) -> dict[str, np.ndarray]:
        last_nz = self.last_nonzero_upto[:, origin].astype(np.int32)
        days_since_last_sale = (origin - last_nz).astype(np.float32)

        first_sale = np.where(
            self.first_sale_idx_full <= origin, self.first_sale_idx_full, NOT_YET
        )
        days_since_first_sale = np.where(
            first_sale >= 0, (origin - first_sale).astype(np.float32), np.float32(NOT_YET)
        )

        return {
            "days_since_last_sale": days_since_last_sale,
            "zero_streak_length": days_since_last_sale.copy(),
            "days_since_first_sale": days_since_first_sale.astype(np.float32),
        }


    def _listing_arrays(self, origin: int, horizon: int) -> np.ndarray:
        known_through = origin + horizon
        return np.where(
            self.first_price_day_idx_full <= known_through,
            self.first_price_day_idx_full,
            NOT_YET,
        ).astype(np.int32)


    def _recent_avg_price(self, origin: int) -> np.ndarray:
        if self.d.price_wide is None:
            return np.full(config.N_SERIES, np.nan, dtype=np.float32)

        w_origin = int(self.d.day_to_week[origin])
        w_start = max(0, w_origin - config.PRICE_AVG_WINDOW_WEEKS + 1)
        window = self.d.price_wide[:, w_start:w_origin + 1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(window.astype(np.float64), axis=1).astype(np.float32)


    def _per_series_day_features(
        self, target_days: np.ndarray, series_idx: np.ndarray
    ) -> dict[str, np.ndarray]:
        snap = self.d.snap_matrix[np.ix_(target_days, self.d.snap_col_of_series[series_idx])]

        if self.d.price_wide is not None:
            weeks = self.d.day_to_week[target_days]
            price = self.d.price_wide[np.ix_(series_idx, weeks)].T
        else:
            price = np.full((len(target_days), len(series_idx)), np.nan, dtype=np.float32)

        return {
            "snap": snap.astype(np.int8).ravel(),
            "sell_price": price.astype(np.float32).ravel(),
        }


    def build_origin_frame(
        self,
        origin_idx: int,
        horizon: int = config.HORIZON,
        series_idx: np.ndarray | None = None,
        include_target: bool = True,
    ) -> pd.DataFrame:
        if series_idx is None:
            series_idx = np.arange(self.d.sales_wide.shape[0])
        n_s = len(series_idx)

        target_days = origin_idx + 1 + np.arange(horizon)
        if target_days.max() >= config.N_CALENDAR_DAYS:
            raise ValueError(
                f"horizon runs past the end of calendar.csv (last day index "
                f"{config.N_CALENDAR_DAYS - 1}, requested {target_days.max()})"
            )

        cols: dict[str, np.ndarray] = {}

        cols["series_idx"] = np.tile(series_idx.astype(np.int32), horizon)
        cols["origin_idx"] = np.full(n_s * horizon, origin_idx, dtype=np.int32)
        cols["target_day_idx"] = np.repeat(target_days.astype(np.int32), n_s)
        cols["horizon"] = np.repeat(np.arange(1, horizon + 1, dtype=np.int8), n_s)

        meta = self.d.series_meta
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            codes = meta[col + "_code"].to_numpy()[series_idx]
            cols[col] = np.tile(codes, horizon)

        for name, arr in self._calendar_features(target_days).items():
            cols[name] = np.repeat(arr, n_s)

        for name, arr in self._demand_features(origin_idx).items():
            cols[name] = np.tile(arr[series_idx], horizon)

        for name, arr in self._recency_features(origin_idx).items():
            cols[name] = np.tile(arr[series_idx], horizon)

        first_price = self._listing_arrays(origin_idx, horizon)[series_idx]
        first_price_tiled = np.tile(first_price, horizon)
        target_tiled = cols["target_day_idx"]

        days_since_listing = np.where(
            first_price_tiled >= 0,
            (target_tiled - first_price_tiled).astype(np.float32),
            np.float32(np.nan),
        )
        cols["days_since_first_listing"] = days_since_listing
        cols["pre_listing"] = (
            (first_price_tiled < 0) | (target_tiled < first_price_tiled)
        ).astype(np.int8)
        cols["first_price_day_idx"] = first_price_tiled.astype(np.int32)

        for name, arr in self._per_series_day_features(target_days, series_idx).items():
            cols[name] = arr

        recent_avg = self._recent_avg_price(origin_idx)[series_idx]
        cols["recent_avg_price"] = np.tile(recent_avg, horizon)

        with np.errstate(divide="ignore", invalid="ignore"):
            cols["price_rel_to_recent_avg"] = (
                cols["sell_price"] / cols["recent_avg_price"]
            ).astype(np.float32)

        cols["price_is_missing"] = np.isnan(cols["sell_price"]).astype(np.int8)

        frame = pd.DataFrame(cols)

        if include_target:
            known = target_days <= config.LAST_KNOWN_DAY_IDX
            if known.all():
                y = self.d.sales_wide[np.ix_(series_idx, target_days)].T.ravel()
                frame["sales"] = y.astype(np.float32)
            else:
                y = np.full((horizon, n_s), np.nan, dtype=np.float32)
                if known.any():
                    y[known] = self.d.sales_wide[np.ix_(series_idx, target_days[known])].T
                frame["sales"] = y.ravel()

        return frame


BOOKKEEPING_COLS = ["series_idx", "origin_idx", "target_day_idx", "first_price_day_idx"]
TARGET_COL = "sales"

FEATURE_GROUPS: dict[str, list[str]] = {
    "A_calendar": [
        "wday", "month", "year", "is_weekend",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap",
    ],
    "B_historical_demand": [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
    ],
    "C_recency": ["days_since_last_sale", "zero_streak_length", "days_since_first_sale"],
    "D_listing": ["days_since_first_listing", "pre_listing"],
    "E_price": ["sell_price", "recent_avg_price", "price_rel_to_recent_avg", "price_is_missing"],
    "F_hierarchy": ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    "G_horizon": ["horizon"],
}

CATEGORICAL_FEATURES = [
    "wday", "month", "year", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2", "item_id", "dept_id", "cat_id",
    "store_id", "state_id",
]


def all_feature_columns() -> list[str]:
    out: list[str] = []
    for group in FEATURE_GROUPS.values():
        out.extend(group)
    return out
