
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from .data_loader import M5Data

MIN_LOOKBACK = 28

LAGS = [28, 35, 42, 56]
ROLL_WINDOWS = [7, 28, 56]

FEATURE_COLUMNS = (
    [f"lag_{k}" for k in LAGS]
    + [f"rolling_mean_{w}" for w in ROLL_WINDOWS]
    + ["rolling_std_7", "rolling_std_28", "rolling_max_28", "rolling_min_28"]
    + ["wday", "month", "year", "is_weekend",
       "event_name_1", "event_type_1", "snap"]
    + ["sell_price", "price_rel_to_recent_avg"]
    + ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
)

CATEGORICAL = ["wday", "month", "year", "event_name_1", "event_type_1",
               "item_id", "dept_id", "cat_id", "store_id", "state_id"]


class TeamStyleBuilder:

    def __init__(self, data: M5Data, min_lookback: int = MIN_LOOKBACK,
                 lags: list[int] | None = None):
        self.min_lookback = min_lookback
        self.lags = list(lags) if lags is not None else list(LAGS)
        self.lag_names = [f"lag_{k}" for k in self.lags]
        self.feature_columns = self.lag_names + FEATURE_COLUMNS[len(LAGS):]
        self.d = data
        cal = data.calendar
        self.wday = cal["wday"].to_numpy(np.int16)
        self.month = cal["month"].to_numpy(np.int16)
        self.year = cal["year"].to_numpy(np.int16)
        self.is_weekend = cal["is_weekend"].to_numpy(np.int8)

        ev1 = sorted(cal["event_name_1"].dropna().unique())
        et1 = sorted(cal["event_type_1"].dropna().unique())
        self.ev1 = cal["event_name_1"].map({v: i + 1 for i, v in enumerate(ev1)}) \
                                      .fillna(0).to_numpy(np.int16)
        self.et1 = cal["event_type_1"].map({v: i + 1 for i, v in enumerate(et1)}) \
                                      .fillna(0).to_numpy(np.int16)

        meta = data.series_meta
        self.hier = {c: meta[c + "_code"].to_numpy(np.int16)
                     for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]}


    def _day_block(self, t: int, sales: np.ndarray) -> dict[str, np.ndarray]:
        n = sales.shape[0]
        out: dict[str, np.ndarray] = {}

        for k, name in zip(self.lags, self.lag_names):
            src = t - k
            out[name] = (sales[:, src].astype(np.float32) if src >= 0
                         else np.full(n, np.nan, np.float32))

        end = t - self.min_lookback
        for w in ROLL_WINDOWS:
            start = end - w + 1
            if end < 0:
                out[f"rolling_mean_{w}"] = np.full(n, np.nan, np.float32)
                continue
            win = sales[:, max(0, start):end + 1].astype(np.float64)
            out[f"rolling_mean_{w}"] = win.mean(axis=1).astype(np.float32)
            if w == 7:
                out["rolling_std_7"] = win.std(axis=1).astype(np.float32)
            if w == 28:
                out["rolling_std_28"] = win.std(axis=1).astype(np.float32)
                out["rolling_max_28"] = win.max(axis=1).astype(np.float32)
                out["rolling_min_28"] = win.min(axis=1).astype(np.float32)

        out["wday"] = np.full(n, self.wday[t], np.int16)
        out["month"] = np.full(n, self.month[t], np.int16)
        out["year"] = np.full(n, self.year[t], np.int16)
        out["is_weekend"] = np.full(n, self.is_weekend[t], np.int8)
        out["event_name_1"] = np.full(n, self.ev1[t], np.int16)
        out["event_type_1"] = np.full(n, self.et1[t], np.int16)
        out["snap"] = self.d.snap_matrix[t, self.d.snap_col_of_series].astype(np.int8)

        wk = int(self.d.day_to_week[t])
        price = self.d.price_wide[:, wk].astype(np.float32)
        out["sell_price"] = price

        wk_ref = int(self.d.day_to_week[max(0, t - self.min_lookback)])
        w0 = max(0, wk_ref - 7)
        ref = self.d.price_wide[:, w0:wk_ref + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.nanmean(ref.astype(np.float64), axis=1)
            out["price_rel_to_recent_avg"] = (price / avg).astype(np.float32)

        for c, v in self.hier.items():
            out[c] = v
        return out


    def build(self, target_days, sales: np.ndarray | None = None,
              with_target: bool = True, verbose: bool = False):
        S = self.d.sales_wide if sales is None else sales
        target_days = np.asarray(target_days, dtype=int)
        n = S.shape[0]
        rows = n * len(target_days)
        FCOLS = self.feature_columns

        X = np.empty((rows, len(FCOLS)), dtype=np.float32)
        y = np.empty(rows, dtype=np.float32) if with_target else None
        series_idx = np.empty(rows, dtype=np.int32)
        day_idx = np.empty(rows, dtype=np.int32)

        t0 = time.time()
        for i, t in enumerate(target_days):
            blk = self._day_block(int(t), S)
            s, e = i * n, (i + 1) * n
            for j, c in enumerate(FCOLS):
                X[s:e, j] = blk[c]
            if with_target:
                y[s:e] = S[:, t].astype(np.float32)
            series_idx[s:e] = np.arange(n)
            day_idx[s:e] = t
            if verbose and (i + 1) % 100 == 0:
                print(f"      {i + 1}/{len(target_days)} days "
                      f"({time.time() - t0:.0f}s)")

        meta = pd.DataFrame({"series_idx": series_idx, "target_day_idx": day_idx})
        return X, y, meta


def categorical_indices(cols: list[str] | None = None) -> list[int]:
    cols = list(FEATURE_COLUMNS) if cols is None else list(cols)
    return [i for i, c in enumerate(cols) if c in CATEGORICAL]
