"""
"Team-style" per-target-day feature construction — a SEPARATE, ADDITIVE module.

Nothing in this file touches features.py, backtest.py or any existing experiment.
It exists so we can test a genuinely different methodology alongside our own,
not instead of it.

HOW THIS DIFFERS FROM OUR FIXED-ORIGIN PIPELINE
-----------------------------------------------
Our pipeline computes every history-derived feature ONCE at the forecast origin T
and holds it constant across all 28 forecast days. That mirrors the real task
exactly, and it is what makes lag_1 and rolling_mean_7 usable at all.

The far more common M5 recipe — the one in most public notebooks and tutorials —
instead builds one training row per (series, day) and computes lags RELATIVE TO
EACH TARGET DAY, using only lookbacks of 28 days or more:

    lag_28 for target day t  =  sales on day t-28

For a 28-day-ahead forecast from origin T, every target day t lies in
[T+1, T+28], so t-28 lies in [T-27, T] — never past the origin. It is therefore
leakage-safe, and it is verified as such by the corruption test in
scripts/10_team_reproduction.py rather than assumed.

Two consequences matter for the comparison:

  1. Features now VARY across the 28 horizon days instead of being flat. A
     high-volume series with a strong weekly rhythm gets a different lag_28 for
     each target day.
  2. Training rows multiply. Every historical day becomes a row, so a two-year
     training range yields ~21 million rows instead of the ~12.8 million our
     15-origin design produces — and, more importantly, they cover every day
     rather than 15 sampled windows.

WHAT IS AND IS NOT KNOWN
------------------------
No file anywhere in this project, or on this machine, documents the team's
actual feature set, split, or parameters. This module is therefore an
INFORMED RECONSTRUCTION of the standard public M5 approach, not a copy of
their code. Every report built on it says so.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from .data_loader import M5Data

# Minimum lookback. Any lag or rolling window must end at least this many days
# before the target day, otherwise it would reach inside the forecast horizon.
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
    """
    Builds per-target-day feature blocks with a configurable minimum lookback.

    `min_lookback` defaults to 28, which is the only leakage-safe setting for a
    28-day-ahead forecast. It is exposed as a parameter for ONE reason: to let
    scripts/10_team_reproduction.py deliberately construct a leaky variant
    (min_lookback=1) as a diagnostic probe, so we can measure what an
    accidental leak would score. That probe is never used to forecast.
    """

    def __init__(self, data: M5Data, min_lookback: int = MIN_LOOKBACK,
                 lags: list[int] | None = None):
        self.min_lookback = min_lookback
        self.lags = list(lags) if lags is not None else list(LAGS)
        self.lag_names = [f"lag_{k}" for k in self.lags]
        # Column order for this instance (lags vary with the constructor args).
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

    # ------------------------------------------------------------------

    def _day_block(self, t: int, sales: np.ndarray) -> dict[str, np.ndarray]:
        """Features for every series on one target day t."""
        n = sales.shape[0]
        out: dict[str, np.ndarray] = {}

        for k, name in zip(self.lags, self.lag_names):
            src = t - k
            out[name] = (sales[:, src].astype(np.float32) if src >= 0
                         else np.full(n, np.nan, np.float32))

        # Rolling windows END at t - min_lookback (inclusive).
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

        # Price relative to the item's own recent average, using only weeks that
        # are already at least MIN_LOOKBACK days behind the target day.
        wk_ref = int(self.d.day_to_week[max(0, t - self.min_lookback)])
        w0 = max(0, wk_ref - 7)
        ref = self.d.price_wide[:, w0:wk_ref + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.nanmean(ref.astype(np.float64), axis=1)
            out["price_rel_to_recent_avg"] = (price / avg).astype(np.float32)

        for c, v in self.hier.items():
            out[c] = v
        return out

    # ------------------------------------------------------------------

    def build(self, target_days, sales: np.ndarray | None = None,
              with_target: bool = True, verbose: bool = False):
        """
        Build a matrix for the given target days.

        Returns (X float32, y float32 or None, meta DataFrame).
        `sales` lets a caller pass a modified sales matrix (used by the leakage
        corruption test); it defaults to the real one.
        """
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
