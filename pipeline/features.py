"""
Feature engineering for 28-day-ahead demand forecasting.

THE ONE RULE THAT GOVERNS THIS ENTIRE FILE
------------------------------------------
We are making a FIXED-ORIGIN, DIRECT 28-day forecast. Pick a day T ("the forecast
origin" — the last day whose sales we know). We must predict days T+1 .. T+28 all
at once, standing at T, without ever seeing what actually happened after T.

That splits every feature into exactly two kinds:

  1. ORIGIN-RELATIVE features — built from sales history up to and including T,
     then held CONSTANT across all 28 forecast days.
     Everything derived from past sales lives here: lags, rolling means/stds,
     days_since_last_sale, zero_streak_length.

     Why constant? Because on day T we genuinely do not know day T+5's sales, so
     we cannot build "the 7-day average ending at T+5" for it. Recomputing a lag
     per target day would silently require future sales. That is leakage, and it
     is the single most common way a forecasting backtest quietly lies to you.

  2. TARGET-DAY features — legitimately known in advance for every horizon day.
     The calendar (weekday, month, events, SNAP) is published years ahead, and
     this dataset also supplies sell_prices for the forecast weeks. These may and
     should vary across the 28 days.

Anything that does not fit cleanly into one of those two buckets does not belong
in the feature set.

WHAT WE NEVER DO (per the project's standing instructions)
----------------------------------------------------------
No smoothing of sales. No dropping of zero rows. No replacing zeros with NaN.
No assuming a zero means a stockout. No invented promotion labels. The model is
trained on the original observations exactly as recorded.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .data_loader import M5Data


# Sentinel used where a "days since <event>" quantity has no meaningful value yet
# (e.g. a product that has never sold as of the origin). Kept as an explicit
# number rather than NaN for the integer-valued recency features, so the value is
# visible and auditable; LightGBM splits on it fine.
NOT_YET = -1


class FeatureBuilder:
    """
    Builds leakage-safe feature frames for a given forecast origin.

    Usage:
        fb = FeatureBuilder(data)
        frame = fb.build_origin_frame(origin_idx=1912, horizon=28)
    """

    def __init__(self, data: M5Data):
        self.d = data
        self._precompute()

    # ------------------------------------------------------------------
    # One-time precomputation
    # ------------------------------------------------------------------

    def _precompute(self) -> None:
        """
        Precompute a few whole-history arrays that make per-origin feature building
        cheap. Each one is CAUSAL — reading it at column T only ever depends on
        columns <= T — which is what keeps this safe despite being computed once
        over the full matrix.
        """
        sales = self.d.sales_wide
        n_days = sales.shape[1]

        # last_nonzero_upto[s, t] = the largest day index <= t on which series s
        # sold at least one unit, or -1 if it had not sold at all by day t.
        #
        # np.maximum.accumulate is a running maximum, so column t is a function of
        # columns 0..t only. Reading column T therefore cannot see day T+1.
        pos = np.arange(n_days, dtype=np.int16)
        tmp = np.where(sales > 0, pos[None, :], np.int16(-1))
        self.last_nonzero_upto = np.maximum.accumulate(tmp, axis=1)
        del tmp

        # first_sale_idx_full[s] = first day index on which series s ever sold.
        # Used only after masking with "<= origin", so a first sale that happens
        # after the origin can never leak its position (see _recency_features).
        has_any = (sales > 0).any(axis=1)
        first_sale = np.argmax(sales > 0, axis=1).astype(np.int32)
        self.first_sale_idx_full = np.where(has_any, first_sale, np.int32(10**6))

        # first_price_day_idx_full[s] = first DAY index at which series s has a
        # recorded sell_price. Derived from the weekly price matrix, then mapped
        # back onto the day axis.
        if self.d.price_wide is not None:
            priced = ~np.isnan(self.d.price_wide)
            has_price = priced.any(axis=1)
            first_price_week = np.argmax(priced, axis=1).astype(np.int32)
            # first day index belonging to that week
            week_first_day = (
                self.d.calendar.groupby("week_idx")["day_idx"].min()
                .reindex(range(self.d.price_wide.shape[1])).to_numpy()
            )
            self.first_price_day_idx_full = np.where(
                has_price, week_first_day[first_price_week], np.int32(10**6)
            ).astype(np.int32)
        else:
            self.first_price_day_idx_full = np.full(config.N_SERIES, 10**6, dtype=np.int32)

        # Integer codes for calendar event fields. Built from calendar.csv, which
        # covers the whole forecast window, so no unseen category can appear later.
        self.event_maps = {}
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            vals = sorted(self.d.calendar[col].dropna().unique())
            # 0 is reserved for "no event on this day"
            self.event_maps[col] = {v: i + 1 for i, v in enumerate(vals)}

        self.event_codes = {
            col: self.d.calendar[col].map(self.event_maps[col]).fillna(0).to_numpy(np.int16)
            for col in self.event_maps
        }

    # ------------------------------------------------------------------
    # Feature group A — Calendar (TARGET-DAY, known in advance)
    # ------------------------------------------------------------------

    def _calendar_features(self, target_days: np.ndarray) -> dict[str, np.ndarray]:
        """
        Per-target-day calendar values. All of these are published in advance and
        are present in calendar.csv for the entire forecast window, so using them
        for a future day is legitimate, not leakage.

        SNAP = Supplemental Nutrition Assistance Program, a US food-assistance
        benefit. snap_CA / snap_TX / snap_WI flag whether the benefit was usable
        in that state on that day. Each series is matched to its OWN state's flag
        (a California store must read snap_CA), which is why SNAP is handled in
        _per_series_day_features rather than here.
        """
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

    # ------------------------------------------------------------------
    # Feature group B — Historical demand (ORIGIN-RELATIVE)
    # ------------------------------------------------------------------

    def _demand_features(self, origin: int) -> dict[str, np.ndarray]:
        """
        Lags and rolling statistics computed from sales up to and including the
        origin, then held constant across all 28 horizon days.

        LAG DEFINITION USED THROUGHOUT THIS PROJECT
        -------------------------------------------
        lag_k = sales on the day k days before the FIRST forecast day.
        The first forecast day is T+1, so:
            lag_1  = sales on day T      (the last day we know about)
            lag_7  = sales on day T-6
            lag_14 = sales on day T-13
            lag_28 = sales on day T-27
        Every one of those days is <= T, so all four lags are safe for all 28
        horizon days under a fixed origin.

        Rolling windows likewise END at the origin:
            rolling_mean_7  = mean sales over days [T-6,  T]
            rolling_mean_28 = mean sales over days [T-27, T]
        """
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
            # Accumulate in float64, then narrow to float32. Reducing directly in
            # float32 makes the result sensitive to the array's memory layout
            # (NumPy's pairwise summation groups by strides), which produces
            # last-bit differences that are meaningless numerically but break the
            # exact-equality leakage test. float64 accumulation removes that
            # sensitivity: the float32 result is then reproducible bit-for-bit.
            window = sales[:, start:origin + 1].astype(np.float64)
            out[f"rolling_mean_{w}"] = window.mean(axis=1).astype(np.float32)
            out[f"rolling_std_{w}"] = window.std(axis=1).astype(np.float32)  # ddof=0

        return out

    # ------------------------------------------------------------------
    # Feature group C — Recency (ORIGIN-RELATIVE)
    # ------------------------------------------------------------------

    def _recency_features(self, origin: int) -> dict[str, np.ndarray]:
        """
        How long has this series been "dry"? The EDA found this to be the single
        cleanest relationship in the whole dataset: the chance of selling today
        falls from 65.2% (sold yesterday) to 0.6% (29+ days without a sale).

        NOTE ON REDUNDANCY (an honest finding, verified in validation_checks.py):
        at a fixed origin, days_since_last_sale and zero_streak_length are the
        SAME number by construction — if the last sale was 3 days ago, then there
        are exactly 3 consecutive zero days ending at the origin. Both are built
        here because the project spec lists both, but they are perfectly
        correlated and one should be dropped before training. This is flagged in
        the foundation report rather than quietly hidden.
        """
        last_nz = self.last_nonzero_upto[:, origin].astype(np.int32)
        days_since_last_sale = (origin - last_nz).astype(np.float32)  # 0 if it sold on T

        # First-ever sale, but only if it happened on or before the origin.
        # If a series' first sale falls after T we must report "not yet", never
        # the actual future date.
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

    # ------------------------------------------------------------------
    # Feature group D — Listing awareness
    # ------------------------------------------------------------------

    def _listing_arrays(self, origin: int, horizon: int) -> np.ndarray:
        """
        first_price_day_idx as known at the origin.

        At origin T we know prices out to T+horizon (the dataset supplies future
        weeks). So "the first day this product had a price" is knowable as long as
        it falls within [0, T+horizon]. A first listing beyond that window is
        reported as NOT_YET rather than its true future date.

        Why this matters: the EDA found that a product's leading run of zero sales
        ends at almost exactly the same time as its leading run of missing prices
        (median gap 3 days; 99.48% of series within 7 days). That is strong
        evidence those early zeros mean "not stocked yet", not "nobody wanted it".
        Telling the model which zeros are which is the core of the listing-aware
        idea.
        """
        known_through = origin + horizon
        return np.where(
            self.first_price_day_idx_full <= known_through,
            self.first_price_day_idx_full,
            NOT_YET,
        ).astype(np.int32)

    # ------------------------------------------------------------------
    # Feature group E — Price
    # ------------------------------------------------------------------

    def _recent_avg_price(self, origin: int) -> np.ndarray:
        """
        Each series' own average price over the trailing weeks ending at the
        origin's week. Used to express price RELATIVE to what this product
        normally costs — raw price is dominated by cross-item scale (a $30 hobby
        item vs a $1 food item), which swamps the actual demand signal.

        Computed only from weeks up to the origin, so it cannot absorb a future
        price change.
        """
        if self.d.price_wide is None:
            return np.full(config.N_SERIES, np.nan, dtype=np.float32)

        w_origin = int(self.d.day_to_week[origin])
        w_start = max(0, w_origin - config.PRICE_AVG_WINDOW_WEEKS + 1)
        window = self.d.price_wide[:, w_start:w_origin + 1]

        with warnings.catch_warnings():
            # An all-NaN row here just means "never priced yet" — expected, not an error.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # float64 accumulation for the same layout-independence reason as the
            # rolling demand statistics above.
            return np.nanmean(window.astype(np.float64), axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Per-(series, day) blocks: SNAP and price
    # ------------------------------------------------------------------

    def _per_series_day_features(
        self, target_days: np.ndarray, series_idx: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Features that depend on BOTH the series and the target day, returned in
        horizon-major order (all series for h=1, then all series for h=2, ...).
        """
        # SNAP flag matched to each series' own state.
        snap = self.d.snap_matrix[np.ix_(target_days, self.d.snap_col_of_series[series_idx])]

        if self.d.price_wide is not None:
            weeks = self.d.day_to_week[target_days]
            price = self.d.price_wide[np.ix_(series_idx, weeks)].T  # -> (horizon, n_series)
        else:
            price = np.full((len(target_days), len(series_idx)), np.nan, dtype=np.float32)

        return {
            "snap": snap.astype(np.int8).ravel(),
            "sell_price": price.astype(np.float32).ravel(),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_origin_frame(
        self,
        origin_idx: int,
        horizon: int = config.HORIZON,
        series_idx: np.ndarray | None = None,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Build the full feature frame for one forecast origin.

        Parameters
        ----------
        origin_idx : the forecast origin T (zero-based day index). Sales up to and
                     including this day are visible; nothing after it is.
        horizon    : number of days to forecast (28).
        series_idx : optional subset of series row positions (for fast smoke tests).
        include_target : attach the true sales for the horizon days, when known.
                     Set False for the real future forecast, where no truth exists.

        Returns
        -------
        DataFrame with n_series * horizon rows, in horizon-major order:
        row (h-1)*n_series + s corresponds to series s on day origin+h.
        """
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

        # --- identifiers / bookkeeping -------------------------------------
        cols["series_idx"] = np.tile(series_idx.astype(np.int32), horizon)
        cols["origin_idx"] = np.full(n_s * horizon, origin_idx, dtype=np.int32)
        cols["target_day_idx"] = np.repeat(target_days.astype(np.int32), n_s)
        cols["horizon"] = np.repeat(np.arange(1, horizon + 1, dtype=np.int8), n_s)

        # --- F. Hierarchy (static; always known) ---------------------------
        meta = self.d.series_meta
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            codes = meta[col + "_code"].to_numpy()[series_idx]
            cols[col] = np.tile(codes, horizon)

        # --- A. Calendar (target-day) --------------------------------------
        for name, arr in self._calendar_features(target_days).items():
            cols[name] = np.repeat(arr, n_s)

        # --- B. Historical demand (origin-relative, constant over horizon) --
        for name, arr in self._demand_features(origin_idx).items():
            cols[name] = np.tile(arr[series_idx], horizon)

        # --- C. Recency (origin-relative) ----------------------------------
        for name, arr in self._recency_features(origin_idx).items():
            cols[name] = np.tile(arr[series_idx], horizon)

        # --- D. Listing awareness ------------------------------------------
        first_price = self._listing_arrays(origin_idx, horizon)[series_idx]
        first_price_tiled = np.tile(first_price, horizon)
        target_tiled = cols["target_day_idx"]

        # Negative => the product is not yet listed on that target day.
        # NOT_YET (-1) => no price on record anywhere in the known window.
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

        # --- SNAP + price (series x day) -----------------------------------
        for name, arr in self._per_series_day_features(target_days, series_idx).items():
            cols[name] = arr

        # --- E. Price, origin-relative + relative ---------------------------
        recent_avg = self._recent_avg_price(origin_idx)[series_idx]
        cols["recent_avg_price"] = np.tile(recent_avg, horizon)

        with np.errstate(divide="ignore", invalid="ignore"):
            cols["price_rel_to_recent_avg"] = (
                cols["sell_price"] / cols["recent_avg_price"]
            ).astype(np.float32)

        cols["price_is_missing"] = np.isnan(cols["sell_price"]).astype(np.int8)

        frame = pd.DataFrame(cols)

        # --- target ---------------------------------------------------------
        if include_target:
            known = target_days <= config.LAST_KNOWN_DAY_IDX
            if known.all():
                y = self.d.sales_wide[np.ix_(series_idx, target_days)].T.ravel()
                frame["sales"] = y.astype(np.float32)
            else:
                # Partially/entirely beyond known history: fill what we can, NaN rest.
                y = np.full((horizon, n_s), np.nan, dtype=np.float32)
                if known.any():
                    y[known] = self.d.sales_wide[np.ix_(series_idx, target_days[known])].T
                frame["sales"] = y.ravel()

        return frame


# --------------------------------------------------------------------------
# Column groupings, used by the report and by model training later
# --------------------------------------------------------------------------

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
    """Flat list of every modelling feature, in stable group order."""
    out: list[str] = []
    for group in FEATURE_GROUPS.values():
        out.extend(group)
    return out
