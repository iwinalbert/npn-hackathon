"""
Loads the raw M5 files into compact in-memory structures.

WHY WIDE MATRICES INSTEAD OF THE 59M-ROW LONG TABLE
---------------------------------------------------
processed_dataset/sales_long_full.parquet has 59,181,090 rows. Loading it in full
costs several GB, and this machine has ~5.7 GB free. But the same information fits
in a few hundred MB if we keep it in its natural rectangular shape:

    sales_wide : (30490 series x 1941 days)  int16   ~118 MB
    price_wide : (30490 series x  282 weeks) float32  ~34 MB
    calendar   : 1969 rows                            negligible

Every feature this pipeline needs is then a fast array slice instead of a groupby
over 59 million rows. The long Parquet table is left completely untouched; we read
the raw CSVs directly (read-only) and cross-check our totals against the values
already verified from it.

DAY INDEXING CONVENTION (used everywhere in this codebase)
----------------------------------------------------------
Day indices are ZERO-BASED positions along the day axis:
    idx 0    <-> "d_1"    <-> 2011-01-29
    idx 1940 <-> "d_1941" <-> 2016-05-22  (last day with known sales)
    idx 1968 <-> "d_1969" <-> 2016-06-19  (last calendar day)
So: d_number == idx + 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

def load_calendar() -> pd.DataFrame:
    """
    Load calendar.csv, one row per date, indexed 0..1968 by day index.

    Adds:
      day_idx   zero-based day index (0 == d_1)
      week_idx  zero-based index into the sorted list of distinct wm_yr_wk values,
                so it can address columns of the price matrix directly
      is_weekend  1 on Saturday/Sunday, else 0

    Every column here is known in advance for the whole forecast horizon — the
    calendar file deliberately runs 28 days past the end of the sales data.
    """
    cal = pd.read_csv(config.CALENDAR_CSV)

    if len(cal) != config.N_CALENDAR_DAYS:
        raise ValueError(
            f"calendar.csv has {len(cal)} rows, expected {config.N_CALENDAR_DAYS}"
        )

    cal["date"] = pd.to_datetime(cal["date"])
    # "d_123" -> 122
    cal["day_idx"] = cal["d"].str.slice(2).astype(int) - 1
    cal = cal.sort_values("day_idx").reset_index(drop=True)

    if not (cal["day_idx"].values == np.arange(config.N_CALENDAR_DAYS)).all():
        raise ValueError("calendar day indices are not a clean 0..N-1 sequence")

    # Map each distinct Walmart week code to a column position in the price matrix.
    weeks = np.sort(cal["wm_yr_wk"].unique())
    week_lookup = {w: i for i, w in enumerate(weeks)}
    cal["week_idx"] = cal["wm_yr_wk"].map(week_lookup).astype(np.int32)

    # wday in this file is 1=Saturday .. 7=Friday, so Sat/Sun are 1 and 2.
    cal["is_weekend"] = cal["wday"].isin([1, 2]).astype(np.int8)

    cal.attrs["weeks"] = weeks
    return cal


# --------------------------------------------------------------------------
# Sales
# --------------------------------------------------------------------------

def load_sales_wide() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load sales_train_evaluation.csv.

    Returns
    -------
    series_meta : DataFrame, 30490 rows, one per store-item series, in file order.
                  Columns: id, item_id, dept_id, cat_id, store_id, state_id
    sales_wide  : int16 array of shape (30490, 1941). sales_wide[s, t] is the units
                  sold by series s on day index t.

    We use the *evaluation* file because it is a strict superset of
    sales_train_validation.csv: same 30,490 series, same values on shared days,
    plus 28 extra days of real observed history (d_1914..d_1941).

    Zeros are preserved exactly as recorded. Nothing is smoothed, dropped, or
    reinterpreted here or anywhere downstream.
    """
    df = pd.read_csv(config.SALES_EVAL_CSV)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]

    if len(df) != config.N_SERIES:
        raise ValueError(f"sales file has {len(df)} rows, expected {config.N_SERIES}")
    if len(day_cols) != config.N_HISTORY_DAYS:
        raise ValueError(
            f"sales file has {len(day_cols)} day columns, expected {config.N_HISTORY_DAYS}"
        )

    # Confirm the day columns are in ascending d_1..d_1941 order before we rely on
    # positional indexing for every downstream feature.
    day_numbers = np.array([int(c[2:]) for c in day_cols])
    if not (day_numbers == np.arange(1, config.N_HISTORY_DAYS + 1)).all():
        raise ValueError("day columns in the sales file are not in d_1..d_1941 order")

    series_meta = df[id_cols].copy()

    # Max observed value is 763, so int16 is safe and halves the memory vs int32.
    #
    # ascontiguousarray is not cosmetic. pandas hands back a Fortran-ordered block
    # here, and NumPy's pairwise summation groups elements according to memory
    # layout — so the same slice reduced from an F-ordered vs a C-ordered array can
    # differ in the last float bits. That is enough to make the leakage corruption
    # test (which compares features for exact equality) report a phantom failure.
    # Forcing row-major storage also matches our dominant access pattern: whole-row
    # slices per series.
    sales_wide = np.ascontiguousarray(df[day_cols].to_numpy(dtype=np.int16))

    return series_meta, sales_wide


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------

def load_price_wide(series_meta: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    """
    Load sell_prices.csv into a (30490 series x 282 weeks) float32 matrix.

    price_wide[s, w] is the price of series s during week-index w, or NaN if no
    price row exists for that store-item-week.

    A NaN here is NOT a data error. It means the store had no price on record for
    that product that week, which in this dataset lines up with the product not yet
    being part of that store's assortment. We leave it as NaN and let the model see
    the missingness (LightGBM handles NaN natively); we never invent a price.

    Prices extend 28 days past the end of the sales history, so the forecast
    horizon's prices are genuinely known in advance.
    """
    prices = pd.read_csv(
        config.SELL_PRICES_CSV,
        dtype={"store_id": "category", "item_id": "category",
               "wm_yr_wk": np.int32, "sell_price": np.float32},
    )

    weeks = calendar.attrs["weeks"]
    week_lookup = {w: i for i, w in enumerate(weeks)}

    # (store_id, item_id) -> row position in sales_wide
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
        # Every (store,item) in the price file should exist in the sales file, and
        # every week should be in the calendar. Surface it loudly if that breaks.
        raise ValueError(f"{(~valid).sum()} price rows could not be mapped to a series/week")

    price_wide = np.full((config.N_SERIES, len(weeks)), np.nan, dtype=np.float32)
    price_wide[row_idx, col_idx.astype(np.int64)] = prices["sell_price"].to_numpy()

    return price_wide


# --------------------------------------------------------------------------
# Convenience bundle
# --------------------------------------------------------------------------

class M5Data:
    """Everything the feature pipeline needs, loaded once and passed around."""

    def __init__(self, load_prices: bool = True):
        self.calendar = load_calendar()
        self.series_meta, self.sales_wide = load_sales_wide()
        self.price_wide = load_price_wide(self.series_meta, self.calendar) if load_prices else None

        # day_idx -> week_idx, for turning a day into its price column
        self.day_to_week = self.calendar["week_idx"].to_numpy()
        self.dates = self.calendar["date"].to_numpy()

        # Integer codes for the hierarchy columns. Saved so the exact same encoding
        # can be reused at prediction time.
        self.cat_maps: dict[str, dict] = {}
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            uniques = sorted(self.series_meta[col].unique())
            self.cat_maps[col] = {v: i for i, v in enumerate(uniques)}
            self.series_meta[col + "_code"] = (
                self.series_meta[col].map(self.cat_maps[col]).astype(np.int16)
            )

        # Which SNAP column applies to each series, as a column position into
        # [snap_CA, snap_TX, snap_WI]. SNAP is a US food-assistance benefit
        # (Supplemental Nutrition Assistance Program); the flag says whether the
        # benefit was usable in that state on that day. A CA store must read
        # snap_CA, not a blended flag.
        state_order = ["CA", "TX", "WI"]
        self.snap_col_of_series = (
            self.series_meta["state_id"].map({s: i for i, s in enumerate(state_order)})
            .to_numpy().astype(np.int8)
        )
        self.snap_matrix = self.calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(dtype=np.int8)

    def day_label(self, day_idx: int) -> str:
        """0 -> 'd_1'"""
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
