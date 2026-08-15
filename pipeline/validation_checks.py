"""
Correctness and leakage checks for the forecasting pipeline.

The headline check here is the FUTURE-SALES CORRUPTION TEST, which proves the
leakage claim empirically instead of asserting it in a comment:

    1. Build the feature frame normally at origin T.
    2. Take a copy of the sales matrix and overwrite EVERY day after T with an
       absurd value (9999 units).
    3. Rebuild the exact same feature frame from the corrupted matrix.
    4. If a single feature value changed, some feature was reading the future.

Features are supposed to be bit-for-bit identical between steps 1 and 3. Only the
target column may differ (it is, by definition, the future).

The mirror-image test matters just as much: corrupting future PRICES *should*
change the price features. Prices and the calendar are legitimately known ahead of
time for the forecast window — that is a property of this dataset, not a mistake.
If corrupting future prices changed nothing, it would mean we were failing to use
information we are entitled to use.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import config
from .data_loader import M5Data
from .features import FeatureBuilder, all_feature_columns


class CheckResult:
    """A single named check with a pass/fail verdict and supporting detail."""

    def __init__(self, name: str, passed: bool, detail: str = "", data: dict | None = None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.data = data or {}

    def __repr__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"

    def to_dict(self) -> dict:
        return {
            "check": self.name,
            "passed": bool(self.passed),
            "detail": self.detail,
            **({"data": self.data} if self.data else {}),
        }


# ==========================================================================
# 1. Source-data integrity
# ==========================================================================

def check_data_integrity(data: M5Data) -> list[CheckResult]:
    """Confirm the loaded matrices reproduce the independently verified totals."""
    out = []
    sales = data.sales_wide

    out.append(CheckResult(
        "series_count",
        sales.shape[0] == config.N_SERIES,
        f"{sales.shape[0]:,} series (expected {config.N_SERIES:,})",
    ))
    out.append(CheckResult(
        "history_day_count",
        sales.shape[1] == config.N_HISTORY_DAYS,
        f"{sales.shape[1]:,} days (expected {config.N_HISTORY_DAYS:,})",
    ))

    total = int(sales.sum(dtype=np.int64))
    out.append(CheckResult(
        "total_units_match",
        total == config.EXPECTED_TOTAL_UNITS,
        f"{total:,} units (expected {config.EXPECTED_TOTAL_UNITS:,})",
    ))

    zeros = int((sales == 0).sum())
    out.append(CheckResult(
        "zero_cell_count_match",
        zeros == config.EXPECTED_ZERO_CELLS,
        f"{zeros:,} zero cells = {zeros / sales.size * 100:.2f}% "
        f"(expected {config.EXPECTED_ZERO_CELLS:,})",
    ))

    mx = int(sales.max())
    out.append(CheckResult(
        "max_sales_match",
        mx == config.EXPECTED_MAX_SALES,
        f"max {mx} (expected {config.EXPECTED_MAX_SALES})",
    ))

    out.append(CheckResult(
        "no_negative_sales",
        int(sales.min()) == 0,
        f"min {int(sales.min())}",
    ))

    # Zeros must survive untouched — this pipeline is forbidden from altering them.
    out.append(CheckResult(
        "zeros_preserved_not_nan",
        not np.isnan(sales.astype(np.float32)).any(),
        "no zero was converted to NaN anywhere in loading",
    ))

    return out


def check_calendar_alignment(data: M5Data) -> list[CheckResult]:
    """The day-index <-> date mapping must be exact; every feature depends on it."""
    out = []
    expected = {
        0: ("d_1", "2011-01-29"),
        1912: ("d_1913", "2016-04-24"),
        1913: ("d_1914", "2016-04-25"),
        1940: ("d_1941", "2016-05-22"),
        1941: ("d_1942", "2016-05-23"),
        1968: ("d_1969", "2016-06-19"),
    }
    problems = []
    for idx, (label, date_str) in expected.items():
        got_label = data.day_label(idx)
        got_date = str(data.date_of(idx).date())
        if got_label != label or got_date != date_str:
            problems.append(f"idx {idx}: got {got_label}/{got_date}, expected {label}/{date_str}")

    out.append(CheckResult(
        "day_index_to_date_mapping",
        not problems,
        "all 6 anchor dates correct" if not problems else "; ".join(problems),
        data={f"idx_{i}": f"{l} = {d}" for i, (l, d) in expected.items()},
    ))

    out.append(CheckResult(
        "calendar_covers_forecast_horizon",
        len(data.calendar) == config.N_CALENDAR_DAYS,
        f"calendar has {len(data.calendar)} days vs {config.N_HISTORY_DAYS} sales days "
        f"= {len(data.calendar) - config.N_HISTORY_DAYS} future days available",
    ))
    return out


# ==========================================================================
# 2. Frame structure
# ==========================================================================

def check_frame_structure(frame: pd.DataFrame, n_series: int, horizon: int) -> list[CheckResult]:
    out = []
    expected_rows = n_series * horizon

    out.append(CheckResult(
        "row_count",
        len(frame) == expected_rows,
        f"{len(frame):,} rows (expected {n_series:,} series x {horizon} days = {expected_rows:,})",
    ))

    dup = frame.duplicated(subset=["series_idx", "target_day_idx"]).sum()
    out.append(CheckResult(
        "no_duplicate_series_day_rows",
        dup == 0,
        f"{dup} duplicate (series, target_day) pairs",
    ))

    n_days = frame["target_day_idx"].nunique()
    out.append(CheckResult(
        "validation_spans_exactly_28_days",
        n_days == horizon,
        f"{n_days} distinct target days (expected {horizon})",
    ))

    per_day = frame.groupby("target_day_idx").size()
    out.append(CheckResult(
        "every_series_present_on_every_day",
        bool((per_day == n_series).all()),
        f"rows per target day: min {per_day.min():,}, max {per_day.max():,} "
        f"(expected {n_series:,} for all)",
    ))

    horizons = sorted(frame["horizon"].unique())
    out.append(CheckResult(
        "horizon_values_1_to_28",
        horizons == list(range(1, horizon + 1)),
        f"horizon values {horizons[0]}..{horizons[-1]} ({len(horizons)} distinct)",
    ))
    return out


# ==========================================================================
# 3. Leakage: the corruption tests
# ==========================================================================

def _clone_with_sales(data: M5Data, new_sales: np.ndarray) -> M5Data:
    """Shallow clone sharing calendar/price but with a different sales matrix."""
    clone = copy.copy(data)
    clone.sales_wide = new_sales
    return clone


def check_no_future_sales_leakage(
    data: M5Data, origin_idx: int, horizon: int = config.HORIZON,
    series_idx: np.ndarray | None = None,
) -> list[CheckResult]:
    """
    THE critical test. Corrupt every sales value after the origin and confirm not
    one feature value moves.
    """
    fb_clean = FeatureBuilder(data)
    clean = fb_clean.build_origin_frame(origin_idx, horizon, series_idx, include_target=True)

    corrupt_sales = data.sales_wide.copy()
    corrupt_sales[:, origin_idx + 1:] = 9999          # absurd, unmissable value
    fb_dirty = FeatureBuilder(_clone_with_sales(data, corrupt_sales))
    dirty = fb_dirty.build_origin_frame(origin_idx, horizon, series_idx, include_target=True)

    feature_cols = [c for c in all_feature_columns() if c in clean.columns]

    changed = []
    for col in feature_cols:
        a, b = clean[col].to_numpy(), dirty[col].to_numpy()
        if not np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating)):
            changed.append(col)

    out = [CheckResult(
        "no_future_sales_in_features",
        not changed,
        (f"all {len(feature_cols)} features unchanged after overwriting every "
         f"post-origin sales value with 9999")
        if not changed else
        f"LEAKAGE DETECTED in: {changed}",
        data={"features_tested": len(feature_cols), "features_changed": changed},
    )]

    # Sanity counter-check: the TARGET must change, otherwise the corruption never
    # actually landed and the test above proves nothing.
    target_changed = not np.array_equal(
        clean["sales"].to_numpy(), dirty["sales"].to_numpy()
    )
    out.append(CheckResult(
        "corruption_actually_applied",
        target_changed,
        "target column did change, confirming the corruption reached the data "
        "(so the feature test above is meaningful)",
    ))
    return out


def check_future_covariates_are_used(
    data: M5Data, origin_idx: int, horizon: int = config.HORIZON,
    series_idx: np.ndarray | None = None,
) -> list[CheckResult]:
    """
    Mirror-image test: future PRICES are allowed, so corrupting them SHOULD change
    the price features. If nothing changed we would be throwing away legitimate
    information.
    """
    fb_clean = FeatureBuilder(data)
    clean = fb_clean.build_origin_frame(origin_idx, horizon, series_idx, include_target=False)

    clone = copy.copy(data)
    corrupt_price = data.price_wide.copy()
    future_weeks = np.unique(data.day_to_week[origin_idx + 1: origin_idx + 1 + horizon])
    corrupt_price[:, future_weeks] = 777.0
    clone.price_wide = corrupt_price

    dirty = FeatureBuilder(clone).build_origin_frame(
        origin_idx, horizon, series_idx, include_target=False
    )

    changed_price = not np.array_equal(
        clean["sell_price"].to_numpy(), dirty["sell_price"].to_numpy(), equal_nan=True
    )
    unchanged_demand = np.array_equal(
        clean["rolling_mean_28"].to_numpy(), dirty["rolling_mean_28"].to_numpy(), equal_nan=True
    )

    return [
        CheckResult(
            "future_prices_are_used",
            changed_price,
            "sell_price for horizon days responds to future price data "
            "(legitimate: sell_prices.csv covers the forecast weeks)",
        ),
        CheckResult(
            "price_corruption_does_not_touch_demand_features",
            unchanged_demand,
            "rolling_mean_28 unaffected by price changes, as expected",
        ),
    ]


def check_train_validation_separation(
    train: pd.DataFrame, validation_origin: int
) -> list[CheckResult]:
    """No training row may target a day inside the validation window."""
    first_val_day = validation_origin + 1
    max_target = int(train["target_day_idx"].max())
    max_origin = int(train["origin_idx"].max())

    return [
        CheckResult(
            "training_targets_precede_validation",
            max_target < first_val_day,
            f"latest training target is day index {max_target} "
            f"(d_{max_target + 1}); validation starts at {first_val_day} "
            f"(d_{first_val_day + 1})",
        ),
        CheckResult(
            "training_origins_precede_validation_origin",
            max_origin <= validation_origin - config.HORIZON,
            f"latest training origin is day index {max_origin} "
            f"(d_{max_origin + 1}), at least {config.HORIZON} days before the "
            f"validation origin {validation_origin} (d_{validation_origin + 1})",
        ),
    ]


# ==========================================================================
# 4. Feature sanity
# ==========================================================================

def check_feature_sanity(frame: pd.DataFrame, data: M5Data) -> list[CheckResult]:
    out = []

    # Lags and rolling means are unit counts: never negative.
    for col in ["lag_1", "lag_7", "lag_14", "lag_28",
                "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28"]:
        v = frame[col].to_numpy()
        finite = v[np.isfinite(v)]
        ok = len(finite) == 0 or finite.min() >= 0
        out.append(CheckResult(
            f"nonneg_{col}",
            ok,
            f"min {finite.min():.3f}, max {finite.max():.3f}" if len(finite) else "all NaN",
        ))

    # Recency features cannot exceed the length of history behind the origin.
    origin = int(frame["origin_idx"].iloc[0])
    dsl = frame["days_since_last_sale"].to_numpy()
    out.append(CheckResult(
        "days_since_last_sale_in_range",
        bool(dsl.min() >= 0 and dsl.max() <= origin + 1),
        f"range [{dsl.min():.0f}, {dsl.max():.0f}], history length {origin + 1}",
    ))

    # The redundancy the spec asks us to build — reported honestly rather than hidden.
    identical = bool(np.array_equal(
        frame["days_since_last_sale"].to_numpy(), frame["zero_streak_length"].to_numpy()
    ))
    out.append(CheckResult(
        "days_since_last_sale_vs_zero_streak_length",
        True,  # informational, not a failure
        ("IDENTICAL by construction at a fixed origin — one of the two is "
         "redundant and should be dropped before training"
         if identical else "the two features differ"),
        data={"identical": identical},
    ))

    # SNAP must be binary and state-matched.
    snap_vals = set(np.unique(frame["snap"].to_numpy()).tolist())
    out.append(CheckResult(
        "snap_is_binary",
        snap_vals.issubset({0, 1}),
        f"distinct SNAP values: {sorted(snap_vals)}",
    ))

    # Spot-check state matching on a single row against calendar.csv directly.
    row = frame.iloc[0]
    s_idx = int(row["series_idx"])
    t_idx = int(row["target_day_idx"])
    state = data.series_meta["state_id"].iloc[s_idx]
    expected_snap = int(data.calendar[f"snap_{state}"].iloc[t_idx])
    out.append(CheckResult(
        "snap_matched_to_series_own_state",
        int(row["snap"]) == expected_snap,
        f"series in {state} on {data.date_of(t_idx).date()}: "
        f"feature={int(row['snap'])}, calendar snap_{state}={expected_snap}",
    ))

    # Price sanity.
    price = frame["sell_price"].to_numpy()
    priced = price[np.isfinite(price)]
    out.append(CheckResult(
        "sell_price_positive_where_present",
        len(priced) == 0 or priced.min() > 0,
        f"{len(priced):,} priced rows, range "
        f"[{priced.min():.2f}, {priced.max():.2f}]" if len(priced) else "no priced rows",
    ))

    missing_pct = float(np.isnan(price).mean() * 100)
    out.append(CheckResult(
        "price_missingness_reported",
        True,  # informational
        f"{missing_pct:.2f}% of rows have no price (left as NaN, never imputed)",
        data={"missing_price_pct": round(missing_pct, 3)},
    ))

    # Target must be untouched raw counts.
    if "sales" in frame.columns:
        y = frame["sales"].to_numpy()
        out.append(CheckResult(
            "target_is_raw_nonneg_integers",
            bool(np.nanmin(y) >= 0 and np.all(np.equal(np.mod(y[~np.isnan(y)], 1), 0))),
            f"target range [{np.nanmin(y):.0f}, {np.nanmax(y):.0f}], "
            f"{float(np.nanmean(y == 0) * 100):.2f}% zeros — preserved, not smoothed",
        ))

    return out


def check_listing_feature_behaviour(
    data: M5Data, origins: list[int], horizon: int = config.HORIZON
) -> tuple[list[CheckResult], list[dict]]:
    """
    Probe the listing-aware features across several origins.

    Two things need establishing before we rely on these features:

    1. Do they ever actually fire? At a 2016 origin essentially every product has
       long since been listed, so `pre_listing` is constant zero and contributes
       nothing. If we only ever trained on recent origins the feature would be
       dead weight. This quantifies where it is and is not active.

    2. Does the listing signal hold up against the sales record? `pre_listing` is
       derived purely from sell_prices.csv; the sales come from
       sales_train_evaluation.csv. If rows flagged pre-listing really do show no
       sales, that is two independent files agreeing, not a circular argument.
    """
    from .features import FeatureBuilder

    fb = FeatureBuilder(data)
    rows = []
    for o in origins:
        has_truth = (o + horizon) <= config.LAST_KNOWN_DAY_IDX
        f = fb.build_origin_frame(o, horizon=horizon, include_target=has_truth)
        pl = f["pre_listing"].to_numpy()
        pm = f["price_is_missing"].to_numpy()
        rec = {
            "origin_day": f"d_{o + 1}",
            "origin_date": str(data.date_of(o).date()),
            "pre_listing_pct": round(float(pl.mean() * 100), 2),
            "price_missing_pct": round(float(pm.mean() * 100), 2),
            "pre_listing_equals_price_missing": bool(np.array_equal(pl, pm)),
        }
        if has_truth:
            y = f["sales"].to_numpy()
            m = pl == 1
            rec["pre_listing_mean_sales"] = round(float(y[m].mean()), 6) if m.any() else None
            rec["pre_listing_zero_pct"] = round(float((y[m] == 0).mean() * 100), 4) if m.any() else None
            rec["listed_mean_sales"] = round(float(y[~m].mean()), 4)
            rec["listed_zero_pct"] = round(float((y[~m] == 0).mean() * 100), 2)
        rows.append(rec)

    active = [r for r in rows if r["pre_listing_pct"] > 0]
    with_truth = [r for r in rows if r.get("pre_listing_zero_pct") is not None]

    results = [
        CheckResult(
            "listing_feature_activates_at_early_origins",
            len(active) > 0,
            "pre_listing is non-zero at " + ", ".join(
                f"{r['origin_day']} ({r['pre_listing_pct']}%)" for r in active
            ) if active else "pre_listing never fires at any tested origin",
        ),
        CheckResult(
            "pre_listing_rows_have_no_sales",
            all(r["pre_listing_zero_pct"] == 100.0 for r in with_truth) if with_truth else False,
            "; ".join(
                f"{r['origin_day']}: {r['pre_listing_zero_pct']}% zeros "
                f"(mean {r['pre_listing_mean_sales']}) vs {r['listed_zero_pct']}% "
                f"for listed rows"
                for r in with_truth
            ),
            data={"per_origin": with_truth},
        ),
        CheckResult(
            "pre_listing_vs_price_is_missing_redundancy",
            True,  # informational
            ("IDENTICAL at every tested origin — one of the two is redundant"
             if all(r["pre_listing_equals_price_missing"] for r in rows)
             else "the two features differ at some origins"),
            data={"identical_everywhere":
                  all(r["pre_listing_equals_price_missing"] for r in rows)},
        ),
    ]
    return results, rows


def check_target_matches_source(
    frame: pd.DataFrame, data: M5Data, n_spot: int = 200, seed: int = 42
) -> list[CheckResult]:
    """Randomly spot-check that the attached target really is the raw source cell."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frame), size=min(n_spot, len(frame)), replace=False)
    sub = frame.iloc[idx]

    truth = data.sales_wide[
        sub["series_idx"].to_numpy(), sub["target_day_idx"].to_numpy()
    ].astype(np.float32)
    got = sub["sales"].to_numpy().astype(np.float32)

    n_bad = int((truth != got).sum())
    return [CheckResult(
        "target_values_match_raw_source",
        n_bad == 0,
        f"{len(sub)} random rows spot-checked against sales_train_evaluation.csv, "
        f"{n_bad} mismatches",
    )]


# ==========================================================================
# Runner
# ==========================================================================

def summarize(results: list[CheckResult]) -> dict:
    passed = sum(1 for r in results if r.passed)
    return {
        "total_checks": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "all_passed": passed == len(results),
        "checks": [r.to_dict() for r in results],
    }
