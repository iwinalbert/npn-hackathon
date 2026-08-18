
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(config.CALENDAR_CSV)

    if len(cal) != config.N_CALENDAR_DAYS:
        raise ValueError(
            f"calendar.csv has {len(cal)} rows, expected {config.N_CALENDAR_DAYS}"
        )

    cal["date"] = pd.to_datetime(cal["date"])
    cal["day_idx"] = cal["d"].str.slice(2).astype(int) - 1
    cal = cal.sort_values("day_idx").reset_index(drop=True)

    if not (cal["day_idx"].values == np.arange(config.N_CALENDAR_DAYS)).all():
        raise ValueError("calendar day indices are not a clean 0..N-1 sequence")

    weeks = np.sort(cal["wm_yr_wk"].unique())
    week_lookup = {w: i for i, w in enumerate(weeks)}
    cal["week_idx"] = cal["wm_yr_wk"].map(week_lookup).astype(np.int32)

    cal["is_weekend"] = cal["wday"].isin([1, 2]).astype(np.int8)

    cal.attrs["weeks"] = weeks
    return cal


def load_sales_wide() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(config.SALES_EVAL_CSV)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]

    if len(df) != config.N_SERIES:
        raise ValueError(f"sales file has {len(df)} rows, expected {config.N_SERIES}")
    if len(day_cols) != config.N_HISTORY_DAYS:
        raise ValueError(
            f"sales file has {len(day_cols)} day columns, expected {config.N_HISTORY_DAYS}"
        )

    day_numbers = np.array([int(c[2:]) for c in day_cols])
    if not (day_numbers == np.arange(1, config.N_HISTORY_DAYS + 1)).all():
        raise ValueError("day columns in the sales file are not in d_1..d_1941 order")

    series_meta = df[id_cols].copy()

    sales_wide = np.ascontiguousarray(df[day_cols].to_numpy(dtype=np.int16))

    return series_meta, sales_wide


def load_price_wide(series_meta: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    prices = pd.read_csv(
        config.SELL_PRICES_CSV,
        dtype={"store_id": "category", "item_id": "category",
               "wm_yr_wk": np.int32, "sell_price": np.float32},
    )

    weeks = calendar.attrs["weeks"]
    week_lookup = {w: i for i, w in enumerate(weeks)}

    key_to_row = {
        (st, it): i
        for i, (st, it) in enumerate(
            zip(series_meta["store_id"].to_numpy(), series_meta["item_id"].to_numpy())
        )
    }

    row_idx = np.fromiter(
        (key_to_row.get((st, it), -1)
         for st, it in zip(prices["store_id"].astype(str), prices["item_id"].astype(str))),
        dtype=np.int64, count=len(prices),
    )
    col_idx = prices["wm_yr_wk"].map(week_lookup).to_numpy()

    valid = (row_idx >= 0) & ~pd.isna(col_idx)
    if not valid.all():
        raise ValueError(f"{(~valid).sum()} price rows could not be mapped to a series/week")

    price_wide = np.full((config.N_SERIES, len(weeks)), np.nan, dtype=np.float32)
    price_wide[row_idx, col_idx.astype(np.int64)] = prices["sell_price"].to_numpy()

    return price_wide


class M5Data:

    def __init__(self, load_prices: bool = True):
        self.calendar = load_calendar()
        self.series_meta, self.sales_wide = load_sales_wide()
        self.price_wide = load_price_wide(self.series_meta, self.calendar) if load_prices else None

        self.day_to_week = self.calendar["week_idx"].to_numpy()
        self.dates = self.calendar["date"].to_numpy()

        self.cat_maps: dict[str, dict] = {}
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            uniques = sorted(self.series_meta[col].unique())
            self.cat_maps[col] = {v: i for i, v in enumerate(uniques)}
            self.series_meta[col + "_code"] = (
                self.series_meta[col].map(self.cat_maps[col]).astype(np.int16)
            )

        state_order = ["CA", "TX", "WI"]
        self.snap_col_of_series = (
            self.series_meta["state_id"].map({s: i for i, s in enumerate(state_order)})
            .to_numpy().astype(np.int8)
        )
        self.snap_matrix = self.calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(dtype=np.int8)

    def day_label(self, day_idx: int) -> str:
        return f"d_{day_idx + 1}"

    def date_of(self, day_idx: int) -> pd.Timestamp:
        return pd.Timestamp(self.dates[day_idx])

    def describe(self) -> dict:
        return {
            "n_series": int(self.sales_wide.shape[0]),
            "n_history_days": int(self.sales_wide.shape[1]),
            "n_calendar_days": int(len(self.calendar)),
            "n_price_weeks": int(self.price_wide.shape[1]) if self.price_wide is not None else None,
            "sales_matrix_mb": round(self.sales_wide.nbytes / 1e6, 1),
            "price_matrix_mb": round(self.price_wide.nbytes / 1e6, 1) if self.price_wide is not None else None,
            "first_date": str(self.date_of(0).date()),
            "last_sales_date": str(self.date_of(config.LAST_KNOWN_DAY_IDX).date()),
            "last_calendar_date": str(self.date_of(config.N_CALENDAR_DAYS - 1).date()),
        }
