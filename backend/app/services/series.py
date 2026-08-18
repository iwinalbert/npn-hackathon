
from __future__ import annotations

from ..cache import ttl_cache
from ..config import settings
from ..db import history_source, query, query_one
from ..errors import NotFound
from . import calendar as cal

REGIME_EXPLANATION = {
    "smooth": ("Sells regularly with stable quantities. The easiest regime to "
               "forecast."),
    "erratic": ("Sells regularly but in highly variable quantities. Timing is "
                "predictable, size is not."),
    "intermittent": ("Long gaps between sales, but consistent size when it does "
                     "sell. The dominant regime in this dataset."),
    "lumpy": ("Both the timing and the size are irregular. The hardest regime; "
              "point forecasts here carry the widest error."),
    "never sold": ("No recorded sales in the classification window."),
}


def _series_row(store_id: str, item_id: str) -> dict:
    row = query_one(
        """
        SELECT series_idx, id, item_id, dept_id, cat_id, store_id, state_id,
               volume_tier, regime, mean_daily_sales, total_units, zero_pct,
               adi, cv2
        FROM series WHERE store_id = ? AND item_id = ?
        """,
        [store_id, item_id],
    )
    if not row:
        raise NotFound(
            f"no series for store '{store_id}' and item '{item_id}'",
            store_id=store_id, item_id=item_id,
            hint="Use /hierarchy/search to find valid identifiers.",
        )
    return row


def detail(store_id: str, item_id: str) -> dict:
    row = _series_row(store_id, item_id)
    row["adi"] = float(row["adi"]) if row["adi"] not in (None, float("inf")) else 999.0
    row["cv2"] = float(row["cv2"])
    row["regime_explanation"] = REGIME_EXPLANATION.get(row["regime"], "")
    return row


def history(store_id: str, item_id: str, days: int = 90) -> dict:
    meta = _series_row(store_id, item_id)
    origin = settings.forecast_origin_idx
    state = meta["state_id"]
    rows = query(
        f"""
        SELECT c.date AS date, h.day_idx, h.sales, h.sell_price,
               c.event_name_1, c.snap_CA, c.snap_TX, c.snap_WI
        FROM {history_source()} h
        JOIN calendar c ON c.day_idx = h.day_idx
        WHERE h.series_idx = ? AND h.day_idx > ? AND h.day_idx <= ?
        ORDER BY h.day_idx
        """,
        [meta["series_idx"], origin - days, origin],
    )
    points = []
    for r in rows:
        points.append({
            "date": str(r["date"])[:10],
            "day_idx": int(r["day_idx"]),
            "sales": int(r["sales"]),
            "sell_price": (round(float(r["sell_price"]), 2)
                           if r["sell_price"] is not None else None),
            "event_name": r["event_name_1"] or None,
            "snap": int(r.get(f"snap_{state}", 0) or 0),
        })
    return {
        "series": meta,
        "history": points,
        "from_date": points[0]["date"] if points else "",
        "to_date": points[-1]["date"] if points else "",
    }


BAND_SCALE_FLOOR = 1.0

BAND_BASIS = (
    "Empirical p05-p95 of (actual - predicted), measured on 8 held-out backtest "
    "windows for series in this demand regime at this horizon, rescaled by "
    "sqrt(forecast) to match the model's Tweedie variance assumption. This is "
    "OBSERVED MODEL ERROR, not a model-produced prediction interval — the frozen "
    "model emits point forecasts only."
)


@ttl_cache()
def _bands_for_regime(regime: str) -> dict[int, dict]:
    rows = query(
        "SELECT horizon, q05, q25, q50, q75, q95, n, norm_sd "
        "FROM error_bands WHERE regime = ? ORDER BY horizon",
        [regime],
    )
    return {int(r["horizon"]): r for r in rows}


def forecast(store_id: str, item_id: str, with_bands: bool = True) -> dict:
    meta = _series_row(store_id, item_id)
    rows = query(
        "SELECT horizon, day_idx, yhat FROM forecast "
        "WHERE series_idx = ? ORDER BY horizon",
        [meta["series_idx"]],
    )
    if not rows:
        raise NotFound(f"no forecast rows for series_idx {meta['series_idx']}")

    bands = _bands_for_regime(meta["regime"]) if with_bands else {}
    points = []
    for r in rows:
        h = int(r["horizon"])
        yhat = float(r["yhat"])
        lower = upper = None
        if h in bands:
            b = bands[h]
            scale = max(yhat, BAND_SCALE_FLOOR) ** 0.5
            lower = max(0.0, yhat + float(b["q05"]) * scale)
            upper = max(0.0, yhat + float(b["q95"]) * scale)
        points.append({
            "date": cal.date_of(int(r["day_idx"]))[:10],
            "day_idx": int(r["day_idx"]),
            "horizon": h,
            "yhat": round(yhat, 4),
            "lower": None if lower is None else round(lower, 4),
            "upper": None if upper is None else round(upper, 4),
        })

    origin_idx = cal.origin_day_idx()
    return {
        "series": meta,
        "origin_day": cal.day_label(origin_idx),
        "origin_date": cal.date_of(origin_idx)[:10],
        "forecast": points,
        "total_28d": round(sum(p["yhat"] for p in points), 4),
        "band_basis": BAND_BASIS if with_bands else None,
        "band_regime": meta["regime"] if with_bands else None,
    }


def list_series(store_id: str | None = None, item_id: str | None = None,
                dept_id: str | None = None, cat_id: str | None = None,
                state_id: str | None = None, regime: str | None = None,
                limit: int = 100, offset: int = 0) -> list[dict]:
    clauses, params = [], []
    for col, val in (("store_id", store_id), ("item_id", item_id),
                     ("dept_id", dept_id), ("cat_id", cat_id),
                     ("state_id", state_id), ("regime", regime)):
        if val:
            clauses.append(f"{col} = ?")
            params.append(val)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return query(
        f"""
        SELECT series_idx, id, item_id, dept_id, cat_id, store_id, state_id,
               volume_tier, regime, mean_daily_sales, zero_pct
        FROM series {where}
        ORDER BY mean_daily_sales DESC
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )
