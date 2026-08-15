"""
Model performance, served from verified artefacts only.

EVERY NUMBER HERE IS A MEASUREMENT, NOT AN ESTIMATE
---------------------------------------------------
All accuracy comes from the 8 cached backtest windows, where the champion's
members were retrained per window with a correct cutoff and scored against real
observed sales. Nothing is recomputed, approximated or extrapolated.

TWO RULES THIS MODULE EXISTS TO ENFORCE
---------------------------------------
1. **No accuracy is ever reported for the delivered forecast window.** No ground
   truth exists for d_1942-d_1969, so any figure claimed against it would be
   fabricated. `windows()` returns only origins where actuals exist.

2. **Accuracy is always qualified by aggregation level.** The same forecast is
   ~28% accurate per store-item and ~97% chain-wide. `by_level()` returns the
   whole ladder so a caller can never accidentally quote the wrong one.
"""

from __future__ import annotations

from ..cache import ttl_cache
from ..db import backtest_source, query, query_one
from ..errors import BadRequest, NotFound
from . import calendar as cal


@ttl_cache()
def windows() -> list[dict]:
    """The backtest windows available, newest first."""
    rows = query("SELECT * FROM window_metrics ORDER BY origin_idx DESC")
    out = []
    for r in rows:
        origin = int(r["origin_idx"])
        first, last = origin + 1, origin + 28
        out.append({
            "origin_idx": origin,
            "origin_day": cal.day_label(origin),
            "origin_date": cal.date_of(origin),
            "window_start": cal.date_of(first),
            "window_end": cal.date_of(last),
            "n_predictions": int(r["n"]),
            "rmse": round(float(r["rmse"]), 4),
            "mae": round(float(r["mae"]), 4),
            "wape": round(float(r["wape"]), 4),
            "bias": round(float(r["bias"]), 4),
            "accuracy_pct": round(100 * (1 - float(r["wape"])), 1),
            "rmse_direct_member": round(float(r["rmse_direct"]), 4),
            "rmse_recursive_member": round(float(r["rmse_recursive"]), 4),
            "member_residual_correlation": round(
                float(r["member_resid_corr"]), 4),
            "is_primary_validation_window": origin == 1912,
        })
    return out


def _require_window(origin_idx: int) -> dict:
    for w in windows():
        if w["origin_idx"] == origin_idx:
            return w
    valid = [w["origin_idx"] for w in windows()]
    raise NotFound(
        f"no backtest window at origin_idx {origin_idx}",
        valid_origins=valid,
        hint=("Accuracy can only be reported where ground truth exists. The "
              "delivered forecast window (d_1942-d_1969) has none."))


@ttl_cache()
def by_level() -> list[dict]:
    """
    Measured accuracy at each aggregation level.

    This is the single most important honesty control in the product: it makes
    the ~28% -> ~97% spread explicit instead of letting one number stand in for
    the whole system.
    """
    rows = query("SELECT * FROM level_accuracy ORDER BY n_groups")
    out = []
    for r in rows:
        wape = float(r["agg_WAPE"])
        out.append({
            "level": r["level"],
            "n_groups": int(r["n_groups"]),
            "rmse": round(float(r["agg_RMSE"]), 4),
            "mae": round(float(r["agg_MAE"]), 4),
            "wape": round(wape, 4),
            "accuracy_pct": round(100 * (1 - wape), 1),
        })
    out.append({
        "level": "L12_store_item", "n_groups": 30_490,
        "rmse": 2.0929, "mae": 1.0395, "wape": 0.7205,
        "accuracy_pct": round(100 * (1 - 0.7205), 1),
    })
    return out


@ttl_cache()
def by_horizon(origin_idx: int = 1912) -> list[dict]:
    """How error grows across the 28-day horizon."""
    _require_window(origin_idx)
    rows = query(
        f"""
        SELECT horizon,
               sqrt(avg((y_true - p_blend) ^ 2)) AS rmse,
               avg(abs(y_true - p_blend))        AS mae,
               avg(p_blend - y_true)             AS bias,
               avg(y_true)                       AS mean_actual
        FROM {backtest_source()} WHERE origin_idx = ?
        GROUP BY 1 ORDER BY 1
        """,
        [origin_idx],
    )
    return [{"horizon": int(r["horizon"]),
             "rmse": round(float(r["rmse"]), 4),
             "mae": round(float(r["mae"]), 4),
             "bias": round(float(r["bias"]), 4),
             "mean_actual": round(float(r["mean_actual"]), 4)} for r in rows]


@ttl_cache()
def by_regime(origin_idx: int = 1912) -> list[dict]:
    """
    Accuracy per Syntetos-Boylan demand regime.

    Directly evidences the intermittent-demand requirement: it shows the model's
    behaviour on the regimes that make this dataset hard.
    """
    _require_window(origin_idx)
    rows = query(
        f"""
        SELECT s.regime,
               count(*)                              AS n,
               count(DISTINCT b.series_idx)          AS n_series,
               sqrt(avg((b.y_true - b.p_blend) ^ 2)) AS rmse,
               avg(abs(b.y_true - b.p_blend))        AS mae,
               avg(b.p_blend - b.y_true)             AS bias,
               avg(b.y_true)                         AS mean_actual,
               avg(CASE WHEN b.y_true = 0 THEN 1.0 ELSE 0.0 END) AS zero_rate,
               sum((b.y_true - b.p_blend) ^ 2)       AS sq_error
        FROM {backtest_source()} b
        JOIN series s USING (series_idx)
        WHERE b.origin_idx = ?
        GROUP BY 1 ORDER BY sq_error DESC
        """,
        [origin_idx],
    )
    total_sq = sum(float(r["sq_error"]) for r in rows) or 1.0
    return [{"regime": r["regime"], "n": int(r["n"]),
             "n_series": int(r["n_series"]),
             "rmse": round(float(r["rmse"]), 4),
             "mae": round(float(r["mae"]), 4),
             "bias": round(float(r["bias"]), 4),
             "mean_actual": round(float(r["mean_actual"]), 4),
             "zero_rate_pct": round(100 * float(r["zero_rate"]), 1),
             "share_of_squared_error_pct": round(
                 100 * float(r["sq_error"]) / total_sq, 2)} for r in rows]


@ttl_cache()
def members(origin_idx: int = 1912) -> dict:
    """
    Direct vs recursive vs blend — why the ensemble exists.

    The blend was accepted because its members are architecturally different,
    not because averaging helps: a negative control attributed -0.0247 of the
    -0.0291 gain to architecture. Exposing the member split lets a user see that
    for themselves.
    """
    w = _require_window(origin_idx)
    row = query_one(
        f"""
        SELECT sqrt(avg((y_true - p_direct) ^ 2))    AS rmse_direct,
               sqrt(avg((y_true - p_recursive) ^ 2)) AS rmse_recursive,
               sqrt(avg((y_true - p_blend) ^ 2))     AS rmse_blend,
               avg(abs(y_true - p_direct))           AS mae_direct,
               avg(abs(y_true - p_recursive))        AS mae_recursive,
               avg(abs(y_true - p_blend))            AS mae_blend,
               corr(y_true - p_direct, y_true - p_recursive) AS resid_corr
        FROM {backtest_source()} WHERE origin_idx = ?
        """,
        [origin_idx],
    )
    d, r, b = (float(row["rmse_direct"]), float(row["rmse_recursive"]),
               float(row["rmse_blend"]))
    return {
        "origin_idx": origin_idx,
        "origin_day": w["origin_day"],
        "window": f"{w['window_start']} to {w['window_end']}",
        "members": [
            {"name": "Direct (38 features)", "weight": 0.60,
             "rmse": round(d, 4), "mae": round(float(row["mae_direct"]), 4)},
            {"name": "Recursive (32 features)", "weight": 0.40,
             "rmse": round(r, 4), "mae": round(float(row["mae_recursive"]), 4)},
        ],
        "blend": {"name": "Blend 0.60/0.40", "rmse": round(b, 4),
                  "mae": round(float(row["mae_blend"]), 4)},
        "gain_vs_best_member": round(b - min(d, r), 4),
        "residual_correlation": round(float(row["resid_corr"]), 4),
        "why_it_works": (
            "The members are architecturally different — one forecasts all 28 "
            "days directly, the other rolls a one-step model forward — so their "
            "errors are only partly shared. A negative control in Experiment "
            "#76 attributed -0.0247 of the -0.0291 gain to that architectural "
            "difference and only -0.0044 to averaging."),
    }


def series_backtest(store_id: str, item_id: str,
                    origin_idx: int = 1912) -> dict:
    """Predicted vs actual for one series in one window, with the member split."""
    w = _require_window(origin_idx)
    meta = query_one(
        "SELECT series_idx, id, item_id, store_id, state_id, regime, "
        "       volume_tier FROM series WHERE store_id = ? AND item_id = ?",
        [store_id, item_id])
    if not meta:
        raise NotFound(f"no series for store '{store_id}' and item '{item_id}'",
                       store_id=store_id, item_id=item_id)

    rows = query(
        f"""
        SELECT horizon, target_day_idx, y_true, p_direct, p_recursive, p_blend
        FROM {backtest_source()}
        WHERE origin_idx = ? AND series_idx = ?
        ORDER BY horizon
        """,
        [origin_idx, meta["series_idx"]],
    )
    if not rows:
        raise NotFound(f"no backtest rows for {store_id}/{item_id} at "
                       f"origin {origin_idx}")

    points, sq, ae, n = [], 0.0, 0.0, 0
    for r in rows:
        y, p = float(r["y_true"]), float(r["p_blend"])
        sq += (y - p) ** 2
        ae += abs(y - p)
        n += 1
        points.append({
            "date": cal.date_of(int(r["target_day_idx"])),
            "day_idx": int(r["target_day_idx"]),
            "horizon": int(r["horizon"]),
            "actual": y,
            "predicted": round(p, 4),
            "predicted_direct": round(float(r["p_direct"]), 4),
            "predicted_recursive": round(float(r["p_recursive"]), 4),
            "error": round(p - y, 4),
        })
    return {
        "series": meta,
        "origin_idx": origin_idx,
        "origin_day": w["origin_day"],
        "window": f"{w['window_start']} to {w['window_end']}",
        "points": points,
        "rmse": round((sq / n) ** 0.5, 4),
        "mae": round(ae / n, 4),
        "total_actual": round(sum(p["actual"] for p in points), 2),
        "total_predicted": round(sum(p["predicted"] for p in points), 2),
        "basis": ("Held-out backtest: the model's members were retrained with a "
                  "cutoff at this origin, so these predictions never saw the "
                  "days they are scored against."),
    }


def aggregate_backtest(level: str, node_id: str,
                       origin_idx: int = 1912) -> dict:
    """Predicted vs actual for a hierarchy node in one window."""
    from .hierarchy import _series_filter          # local: avoids a cycle
    w = _require_window(origin_idx)
    where, params = _series_filter(level, node_id)

    n_row = query_one(f"SELECT count(*) AS n FROM series {where}", params)
    if not n_row or n_row["n"] == 0:
        raise NotFound(f"no series found for {level}='{node_id}'")

    rows = query(
        f"""
        SELECT b.horizon, b.target_day_idx,
               sum(b.y_true) AS actual, sum(b.p_blend) AS predicted
        FROM {backtest_source()} b
        JOIN (SELECT series_idx FROM series {where}) s USING (series_idx)
        WHERE b.origin_idx = ?
        GROUP BY 1, 2 ORDER BY 1
        """,
        params + [origin_idx],
    )
    points = [{"date": cal.date_of(int(r["target_day_idx"])),
               "day_idx": int(r["target_day_idx"]),
               "horizon": int(r["horizon"]),
               "actual": round(float(r["actual"]), 2),
               "predicted": round(float(r["predicted"]), 2),
               "error": round(float(r["predicted"]) - float(r["actual"]), 2)}
              for r in rows]
    tot_a = sum(p["actual"] for p in points)
    tot_p = sum(p["predicted"] for p in points)
    ae = sum(abs(p["error"]) for p in points)
    return {
        "level": level, "node_id": node_id,
        "n_series": int(n_row["n"]),
        "origin_idx": origin_idx, "origin_day": w["origin_day"],
        "window": f"{w['window_start']} to {w['window_end']}",
        "points": points,
        "total_actual": round(tot_a, 2),
        "total_predicted": round(tot_p, 2),
        "wape": round(ae / tot_a, 4) if tot_a else None,
        "accuracy_pct": round(100 * (1 - ae / tot_a), 1) if tot_a else None,
        "note": ("Accuracy at this aggregation level, measured on held-out "
                 "actuals. It is higher than store-item accuracy because "
                 "independent errors cancel when summed."),
    }


@ttl_cache()
def error_bands(regime: str | None = None) -> list[dict]:
    """The empirical error-band table, exposed for transparency."""
    if regime:
        valid = {r["regime"] for r in query(
            "SELECT DISTINCT regime FROM error_bands")}
        if regime not in valid:
            raise BadRequest(f"unknown regime '{regime}'",
                             valid_regimes=sorted(valid))
        rows = query("SELECT * FROM error_bands WHERE regime = ? "
                     "ORDER BY horizon", [regime])
    else:
        rows = query("SELECT * FROM error_bands ORDER BY regime, horizon")
    return [{"regime": r["regime"], "horizon": int(r["horizon"]),
             "q05": round(float(r["q05"]), 4), "q25": round(float(r["q25"]), 4),
             "q50": round(float(r["q50"]), 4), "q75": round(float(r["q75"]), 4),
             "q95": round(float(r["q95"]), 4),
             "n": int(r["n"]),
             "normalised_sd": round(float(r["norm_sd"]), 4)} for r in rows]
