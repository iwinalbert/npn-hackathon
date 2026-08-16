"""
CONTEXT BUILDER — the layer that keeps the backend the source of truth.

The assistant never queries the data itself and never receives the dataset. This
module decides what a question needs, fetches exactly that from the EXISTING
services, computes every number deterministically in Python, and hands the model
a small structured JSON document.

WHY DETERMINISTIC RETRIEVAL RATHER THAN LLM TOOL-CALLING
--------------------------------------------------------
Gemini supports function calling, which would let the model choose its own
queries. That was considered and rejected for this application:

  * every extra model round-trip adds latency and cost to a demo that must feel
    instant;
  * a model choosing its own arguments can choose wrong ones, and the failure is
    silent — it answers confidently about the wrong series;
  * deterministic routing is unit-testable, and the tests can assert exactly
    which numbers were available to the model for a given question.

The trade is flexibility: an unanticipated question falls back to a general
context rather than fetching something clever. That is the right way round for a
system whose whole claim is that its numbers are verifiable. The interface is
kept narrow so tool-calling could replace `resolve()` later without touching the
router or the provider.

EVERY DERIVED NUMBER IS COMPUTED HERE, NOT BY THE MODEL
-------------------------------------------------------
Trends, percentage changes, weekly aggregates and comparisons are calculated in
Python and passed in as facts. The model's job is to translate them into
English, never to do arithmetic.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import settings
from ..errors import NotFound
from . import accuracy as accuracy_svc
from . import hierarchy as hierarchy_svc
from . import insights as insights_svc
from . import meta as meta_svc
from . import series as series_svc

# --- intent detection -------------------------------------------------------
# Deliberately simple and readable. Order matters: the first match wins, and a
# selected series outranks everything because the user is looking at it.

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("accuracy", re.compile(
        r"\b(accura|rmse|mae|wape|bias|precision|recall|f1|error|reliab|trust|"
        r"how good|how well|performance|validat|backtest)\w*", re.I)),
    ("model", re.compile(
        r"\b(lightgbm|tweedie|model|ensemble|blend|direct|recursive|algorithm|"
        r"architecture|feature|train|leakage|horizon)\w*"
        # "how does it/this/the system work", with anything reasonable between
        r"|how\s+(does|do)\b.{0,40}\bwork", re.I)),
    ("ranking", re.compile(
        r"\b(which|what|top|highest|lowest|most|least|biggest|largest|rank|"
        r"attention|volatile|rising|falling|increas|decreas)\w*", re.I)),
    ("hierarchy", re.compile(
        r"\b(store|department|categor|state|aggregat|hierarch|roll[- ]?up|chain|"
        r"total)\w*", re.I)),
]


def detect_intent(question: str, has_series: bool) -> str:
    """Which family of data this question needs."""
    q = (question or "").strip()
    if has_series and not _PATTERNS[0][1].search(q) and not _PATTERNS[1][1].search(q):
        # A series is on screen and the question is not about metrics or the
        # model itself, so it is almost certainly about that series.
        return "series"
    for name, pattern in _PATTERNS:
        if pattern.search(q):
            return name
    return "series" if has_series else "general"


# --- deterministic descriptive statistics -----------------------------------

def _trend(values: list[float]) -> dict[str, Any]:
    """
    Direction and slope of a series, computed by least squares.

    Classified with a tolerance relative to the mean level, so a 0.01 unit/day
    drift on a product selling 100/day is correctly called "stable" rather than
    dressed up as a trend.
    """
    n = len(values)
    if n < 2:
        return {"direction": "unknown", "slope_per_day": 0.0, "change_pct": 0.0}

    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    slope = (sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / denom
             if denom else 0.0)

    total_change = slope * (n - 1)
    change_pct = (total_change / mean_y * 100) if mean_y > 1e-9 else 0.0

    # "material" = the fitted change across the window exceeds 10% of the level
    if abs(change_pct) < 10:
        direction = "stable"
    elif change_pct > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return {
        "direction": direction,
        "slope_per_day": round(slope, 4),
        "change_over_window_pct": round(change_pct, 1),
        "first_value": round(values[0], 3),
        "last_value": round(values[-1], 3),
        "mean": round(mean_y, 3),
    }


def _weekly(points: list[dict], key: str) -> list[dict]:
    """Collapse daily values into weeks so the context stays small."""
    out = []
    for w in range(0, len(points), 7):
        chunk = points[w:w + 7]
        if not chunk:
            continue
        vals = [float(p[key]) for p in chunk]
        out.append({
            "week": w // 7 + 1,
            "from": chunk[0].get("date"),
            "to": chunk[-1].get("date"),
            "total": round(sum(vals), 2),
            "daily_mean": round(sum(vals) / len(vals), 3),
        })
    return out


# --- context builders -------------------------------------------------------

def _series_context(store_id: str, item_id: str) -> dict[str, Any]:
    """Everything the assistant may say about one store-item series."""
    fc = series_svc.forecast(store_id, item_id, with_bands=True)
    hist = series_svc.history(store_id, item_id, days=91)
    plan = insights_svc.planning_summary(store_id, item_id)
    meta = fc["series"]

    fc_daily = [p["yhat"] for p in fc["forecast"]]
    hist_daily = [p["sales"] for p in hist["history"]]
    recent28 = hist_daily[-28:] if len(hist_daily) >= 28 else hist_daily

    # Backtest for this series, where truth exists. Absent for some series.
    backtest: dict[str, Any] | None = None
    try:
        bt = accuracy_svc.series_backtest(store_id, item_id, 1912)
        backtest = {
            "window": bt["window"],
            "rmse": bt["rmse"],
            "mae": bt["mae"],
            "total_actual": bt["total_actual"],
            "total_predicted": bt["total_predicted"],
            "basis": bt["basis"],
        }
    except NotFound:
        backtest = None

    prices = [p["sell_price"] for p in hist["history"] if p["sell_price"] is not None]
    events = sorted({p["event_name"] for p in hist["history"] if p["event_name"]})

    return {
        "kind": "series",
        "series": {
            "store": meta["store_id"],
            "item": meta["item_id"],
            "department": meta["dept_id"],
            "category": meta["cat_id"],
            "state": meta["state_id"],
            "demand_pattern": meta["regime"],
            "demand_pattern_meaning": meta.get("regime_explanation"),
            "volume_tier": meta["volume_tier"],
            "mean_daily_sales_all_time": round(meta["mean_daily_sales"], 3),
            "pct_days_with_no_sale": round(meta["zero_pct"], 1),
        },
        "forecast": {
            "horizon_days": len(fc_daily),
            "origin_day": fc["origin_day"],
            "origin_date": fc["origin_date"],
            "first_forecast_date": fc["forecast"][0]["date"] if fc["forecast"] else None,
            "last_forecast_date": fc["forecast"][-1]["date"] if fc["forecast"] else None,
            "total_28d": fc["total_28d"],
            "daily_mean": round(sum(fc_daily) / len(fc_daily), 3) if fc_daily else 0,
            "min_day": round(min(fc_daily), 3) if fc_daily else 0,
            "max_day": round(max(fc_daily), 3) if fc_daily else 0,
            "trend": _trend(fc_daily),
            "weekly": _weekly(fc["forecast"], "yhat"),
            "daily": [{"date": p["date"], "yhat": p["yhat"]} for p in fc["forecast"]],
        },
        "recent_history": {
            "days_supplied": len(hist_daily),
            "last_28_days_total": round(sum(recent28), 2),
            "last_28_days_daily_mean": round(sum(recent28) / len(recent28), 3) if recent28 else 0,
            "trend_last_28_days": _trend(recent28),
            "weekly_last_91_days": _weekly(hist["history"], "sales")[-13:],
        },
        "comparison": {
            "forecast_total_vs_last_28_days": plan.get("change_vs_recent"),
            "recent_28d_actual": plan.get("recent_28d_actual"),
            # `forecast_total_vs_last_28_days` is signed; spelling the direction
            # out in words keeps the model from having to interpret the sign,
            # which is the one part of this comparison it could get backwards
            # without the grounding check noticing (that check reads numbers,
            # not claims).
            "difference_direction": (
                None if plan.get("recent_28d_actual") is None else
                "higher" if fc["total_28d"] >= plan["recent_28d_actual"] else "lower"),
        },
        "uncertainty": {
            "planning_range_low": plan["planning_range"]["low"],
            "planning_range_high": plan["planning_range"]["high"],
            "basis": plan["planning_range"]["basis"],
            "is_model_produced_interval": False,
        },
        "historical_validation": backtest,
        "available_covariates": {
            "price": {
                "available": bool(prices),
                "recent_price": round(prices[-1], 2) if prices else None,
                "min_price_91d": round(min(prices), 2) if prices else None,
                "max_price_91d": round(max(prices), 2) if prices else None,
                "note": ("Price is an input the model uses as context. It is NOT "
                         "a causal lever and this system cannot answer "
                         "'what if I change the price'."),
            },
            "calendar_events_in_recent_history": events,
            "snap_days": {
                "available": True,
                "note": "SNAP benefit days for this store's state are a model input.",
            },
            "promotions": {
                "available": False,
                "note": "The dataset contains no promotion field at all.",
            },
        },
        "caveats": plan.get("caveats", []),
    }


def _accuracy_context() -> dict[str, Any]:
    windows = accuracy_svc.windows()
    primary = next((w for w in windows if w["is_primary_validation_window"]), windows[0])
    return {
        "kind": "accuracy",
        "primary_validation_window": primary,
        "all_windows": [
            {"window": f"{w['window_start']} to {w['window_end']}",
             "rmse": w["rmse"], "mae": w["mae"], "wape": w["wape"], "bias": w["bias"]}
            for w in windows
        ],
        "accuracy_by_aggregation_level": accuracy_svc.by_level(),
        "by_demand_pattern": accuracy_svc.by_regime(1912),
        "by_volume_tier": accuracy_svc.by_volume_tier(1912),
        "demand_occurrence": accuracy_svc.occurrence(1912),
        "members": accuracy_svc.members(1912),
        "error_by_horizon": accuracy_svc.by_horizon(1912)[:28],
        "reference_points": {
            "always_predict_zero_rmse": 3.9161,
            "rolling_28day_average_rmse": 2.2430,
            "note": "Baselines from the research, for scale.",
        },
        "critical_framing": (
            "All of these are HISTORICAL VALIDATION results on windows where the "
            "real outcome is known. They are not live deployment accuracy. No "
            "accuracy figure exists for the delivered 28-day forecast because "
            "that period has no recorded outcome."
        ),
    }


def _model_context() -> dict[str, Any]:
    return {
        "kind": "model",
        "model_card": meta_svc.model_card(),
        "capabilities": meta_svc.capabilities(),
        "members": accuracy_svc.members(1912),
        "architecture_explained": {
            "algorithm": "LightGBM, gradient-boosted decision trees",
            "objective": ("Tweedie with variance power 1.1, chosen because retail "
                          "demand is mostly zeros with occasional positive counts"),
            "direct_member": ("Predicts all 28 days at once from information frozen "
                              "at the forecast origin. 38 features. Weight 0.60."),
            "recursive_member": ("Predicts one day ahead, feeds its own prediction "
                                 "back in, and repeats 28 times. 32 features. "
                                 "Weight 0.40."),
            "why_blended": ("The two make different mistakes — their errors "
                            "correlate about 0.95, not 1.0 — so combining them "
                            "beats either alone."),
            "hierarchy": ("The model forecasts only the bottom store-item level. "
                          "Every aggregate is an exact sum of those forecasts, so "
                          "the hierarchy is coherent by construction."),
            "intermittent_demand": ("Handled by the Tweedie objective across all "
                                    "series, NOT by a separate Croston-style "
                                    "model. Croston, SBA and TSB were tested and "
                                    "lost in every demand regime."),
            "covariates": ("Price, calendar events, and state-matched SNAP days "
                           "are model inputs. There is no promotion field in the "
                           "dataset."),
            "leakage_prevention": ("Every feature is frozen at the forecast "
                                   "origin. Verified empirically by overwriting "
                                   "all post-origin sales with a nonsense value "
                                   "and requiring every feature to come back "
                                   "bit-for-bit identical."),
        },
    }


def _ranking_context(level: str, node_id: str) -> dict[str, Any]:
    movers = insights_svc.top_movers(level, node_id, 10)
    portfolio = insights_svc.portfolio_summary(level, node_id)
    return {
        "kind": "ranking",
        "scope": {"level": level, "node_id": node_id},
        "portfolio": portfolio,
        "rising_fastest": movers.get("rising", []),
        "falling_fastest": movers.get("falling", []),
        "ranking_basis": movers["basis"],
        "note": ("Rankings are by absolute unit change against the previous 28 "
                 "days, because percentage change on a near-zero series is noise."),
    }


def _hierarchy_context(level: str, node_id: str) -> dict[str, Any]:
    agg = hierarchy_svc.aggregate_forecast(level, node_id)
    daily = [p["yhat"] for p in agg["points"]]
    return {
        "kind": "hierarchy",
        "level": level,
        "node_id": node_id,
        "n_series_in_node": agg["n_series"],
        "forecast_total_28d": round(agg["total_28d"], 2),
        "forecast_daily_mean": round(sum(daily) / len(daily), 2) if daily else 0,
        "forecast_trend": _trend(daily),
        "measured_accuracy_at_this_level": hierarchy_svc.measured_accuracy(level),
        "levels_available": [
            {"level": lv["level"], "label": lv["label"], "nodes": lv["node_count"]}
            for lv in hierarchy_svc.list_levels()
        ],
        "coherence_note": ("Aggregates are exact sums of bottom-level store-item "
                           "forecasts."),
    }


def _general_context() -> dict[str, Any]:
    card = meta_svc.model_card()
    return {
        "kind": "general",
        "system": {
            "name": "Retail Demand Forecasting",
            "dataset": "Walmart M5",
            "what_it_forecasts": ("Daily units sold for each product in each "
                                  "store, 28 days ahead"),
            "series_count": card["n_series"],
            "horizon_days": card["horizon_days"],
            "forecast_period": card["forecast_dates"],
        },
        "model_card": card,
        "accuracy_by_aggregation_level": accuracy_svc.by_level(),
        "capabilities": meta_svc.capabilities(),
    }


# --- entry point ------------------------------------------------------------

def resolve(
    question: str,
    store_id: str | None = None,
    item_id: str | None = None,
    level: str = "total",
    node_id: str = "ALL",
) -> dict[str, Any]:
    """
    Build the smallest context that can answer this question.

    Always includes a compact `system` block so the assistant can identify the
    product and the frozen model without a second lookup.
    """
    has_series = bool(store_id and item_id)
    intent = detect_intent(question, has_series)

    if intent == "series" and has_series:
        body = _series_context(store_id, item_id)   # type: ignore[arg-type]
    elif intent == "accuracy":
        body = _accuracy_context()
    elif intent == "model":
        body = _model_context()
    elif intent == "ranking":
        body = _ranking_context(level, node_id)
    elif intent == "hierarchy":
        body = _hierarchy_context(level, node_id)
    else:
        body = _general_context()

    card = meta_svc.model_card()
    return {
        "intent": intent,
        "frozen_model": {
            "formula": card["blend_formula"],
            "status": card["status"],
            "validation_rmse": card["validation_rmse"],
            "validation_mae": card["validation_mae"],
            "validation_window": card["validation_window"],
            "validation_predictions": card["validation_n"],
            "horizon_days": card["horizon_days"],
            "series_count": card["n_series"],
        },
        "data": body,
    }


def collect_numbers(obj: Any, out: set[float] | None = None) -> set[float]:
    """
    Every numeric value anywhere in the context.

    Used by the grounding check in `genai.py` to test whether a figure the model
    quoted actually came from the data it was given.
    """
    if out is None:
        out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 4))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, out)
    elif isinstance(obj, str):
        for m in re.finditer(r"-?\d+(?:\.\d+)?", obj):
            try:
                out.add(round(float(m.group()), 4))
            except ValueError:
                pass
    return out
