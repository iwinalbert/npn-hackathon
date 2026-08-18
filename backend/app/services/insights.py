
from __future__ import annotations

from ..cache import ttl_cache
from ..db import query, query_one
from ..errors import NotFound
from .series import BAND_SCALE_FLOOR, _bands_for_regime, _series_row


@ttl_cache()
def top_movers(level: str = "total", node_id: str = "ALL", limit: int = 20,
               direction: str = "both") -> dict:
    from .hierarchy import _series_filter
    where, params = _series_filter(level, node_id)

    rows = query(
        f"""
        WITH f AS (
            SELECT series_idx, sum(yhat) AS total_28d, avg(yhat) AS fc_daily
            FROM forecast GROUP BY 1
        )
        SELECT s.series_idx, s.id, s.item_id, s.store_id, s.dept_id, s.cat_id,
               s.regime, s.volume_tier,
               f.total_28d, f.fc_daily,
               s.mean_daily_28d AS recent_daily,
               f.fc_daily - s.mean_daily_28d AS delta_daily,
               (f.fc_daily - s.mean_daily_28d) * 28 AS delta_28d
        FROM series s JOIN f USING (series_idx)
        {where}
        """,
        params,
    )
    for r in rows:
        r["delta_pct"] = (round(100 * float(r["delta_daily"])
                                / float(r["recent_daily"]), 1)
                          if float(r["recent_daily"]) > 0.05 else None)

    def fmt(r: dict) -> dict:
        return {
            "series_idx": int(r["series_idx"]), "id": r["id"],
            "item_id": r["item_id"], "store_id": r["store_id"],
            "dept_id": r["dept_id"], "cat_id": r["cat_id"],
            "regime": r["regime"], "volume_tier": r["volume_tier"],
            "forecast_total_28d": round(float(r["total_28d"]), 2),
            "forecast_daily": round(float(r["fc_daily"]), 3),
            "recent_daily_28d": round(float(r["recent_daily"]), 3),
            "delta_daily": round(float(r["delta_daily"]), 3),
            "delta_28d": round(float(r["delta_28d"]), 2),
            "delta_pct": r["delta_pct"],
        }

    ups = sorted((r for r in rows if float(r["delta_daily"]) > 0),
                 key=lambda r: -float(r["delta_daily"]))[:limit]
    downs = sorted((r for r in rows if float(r["delta_daily"]) < 0),
                   key=lambda r: float(r["delta_daily"]))[:limit]

    out = {
        "level": level, "node_id": node_id, "n_series_considered": len(rows),
        "basis": ("Mean daily forecast over the 28-day horizon versus mean "
                  "daily actuals over the trailing 28 days before the origin. "
                  "Ranked by absolute unit change, because percentage change on "
                  "near-zero series is noise."),
    }
    if direction in ("both", "up"):
        out["rising"] = [fmt(r) for r in ups]
    if direction in ("both", "down"):
        out["falling"] = [fmt(r) for r in downs]
    return out


def planning_summary(store_id: str, item_id: str) -> dict:
    meta = _series_row(store_id, item_id)
    rows = query(
        "SELECT horizon, yhat FROM forecast WHERE series_idx = ? "
        "ORDER BY horizon", [meta["series_idx"]])
    if not rows:
        raise NotFound(f"no forecast for {store_id}/{item_id}")

    bands = _bands_for_regime(meta["regime"])
    total = lo_total = hi_total = 0.0
    weekly: list[dict] = []
    for r in rows:
        h, yhat = int(r["horizon"]), float(r["yhat"])
        total += yhat
        b = bands.get(h)
        if b:
            scale = max(yhat, BAND_SCALE_FLOOR) ** 0.5
            lo_total += max(0.0, yhat + float(b["q05"]) * scale)
            hi_total += max(0.0, yhat + float(b["q95"]) * scale)
        else:
            lo_total += yhat
            hi_total += yhat
        wk = (h - 1) // 7
        if len(weekly) <= wk:
            weekly.append({"week": wk + 1, "days": f"{h}-{min(h + 6, 28)}",
                           "expected": 0.0})
        weekly[wk]["expected"] += yhat

    for w in weekly:
        w["expected"] = round(w["expected"], 2)

    recent = query_one(
        "SELECT mean_daily_28d, mean_daily_91d FROM series WHERE series_idx = ?",
        [meta["series_idx"]])
    recent_28 = float(recent["mean_daily_28d"]) * 28 if recent else None

    return {
        "series": meta,
        "horizon_days": 28,
        "expected_total": round(total, 2),
        "expected_daily": round(total / 28, 3),
        "planning_range": {
            "low": round(lo_total, 2),
            "high": round(hi_total, 2),
            "basis": ("Sum of the per-day empirical p05-p95 backtest error band "
                      "for this demand regime. Measured model error, NOT a "
                      "model-produced prediction interval, and NOT a service-"
                      "level target."),
        },
        "recent_28d_actual": round(recent_28, 2) if recent_28 is not None else None,
        "change_vs_recent": (round(total - recent_28, 2)
                             if recent_28 is not None else None),
        "weekly_breakdown": weekly,
        "regime": meta["regime"],
        "caveats": [
            "No ground truth exists for this forecast window, so no accuracy "
            "figure applies to these specific numbers.",
            "A recorded zero may mean 'nobody wanted it' or 'it was out of "
            "stock'; the dataset cannot distinguish them, so neither can this.",
            "Forecasts are expected values and are not integers: 0.6 units/day "
            "is a meaningful rate, not a rounding error.",
        ],
    }


@ttl_cache()
def portfolio_summary(level: str = "total", node_id: str = "ALL") -> dict:
    from .hierarchy import _series_filter, measured_accuracy
    where, params = _series_filter(level, node_id)

    row = query_one(
        f"""
        WITH f AS (SELECT series_idx, sum(yhat) AS total_28d FROM forecast
                   GROUP BY 1)
        SELECT count(*)                         AS n_series,
               sum(f.total_28d)                 AS forecast_total,
               sum(s.mean_daily_28d) * 28       AS recent_total,
               avg(s.zero_pct)                  AS avg_zero_pct
        FROM series s JOIN f USING (series_idx)
        {where}
        """,
        params,
    )
    if not row or not row["n_series"]:
        raise NotFound(f"no series found for {level}='{node_id}'")

    regimes = query(
        f"""
        SELECT regime, count(*) AS n FROM series {where}
        GROUP BY 1 ORDER BY n DESC
        """,
        params,
    )
    fc = float(row["forecast_total"])
    rec = float(row["recent_total"])
    return {
        "level": level, "node_id": node_id,
        "n_series": int(row["n_series"]),
        "forecast_total_28d": round(fc, 1),
        "forecast_daily_avg": round(fc / 28, 1),
        "recent_total_28d": round(rec, 1),
        "change_vs_recent": round(fc - rec, 1),
        "change_pct": round(100 * (fc / rec - 1), 2) if rec else None,
        "avg_zero_day_pct": round(float(row["avg_zero_pct"]), 1),
        "regime_mix": [{"regime": r["regime"], "n_series": int(r["n"])}
                       for r in regimes],
        "expected_accuracy": measured_accuracy(level),
    }
