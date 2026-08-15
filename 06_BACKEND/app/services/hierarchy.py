"""
Hierarchy navigation and coherent aggregation.

THE ONE IDEA THAT MATTERS HERE
------------------------------
Every aggregate this module returns is an exact SUM of bottom-level store-item
forecasts. That is not a convenience — it is what makes the hierarchy *coherent*
by construction: a store total always equals the sum of its items, at every
level, with no reconciliation step and no possibility of disagreement.

The research phase investigated whether producing independent forecasts at
aggregate levels and reconciling them downward could beat this (Experiments
#80-#82) and rejected it: above item level the true-aggregate oracle is worth at
most -0.0221 RMSE, and at item level the method failed its mechanism criterion on
2 of 4 windows. So bottom-up summation is not a simplification we settled for; it
is the approach the evidence selected.
"""

from __future__ import annotations

from ..cache import ttl_cache
from ..config import settings
from ..db import (SAFE_LEVELS, history_source, query, query_one,
                  validate_level)
from ..errors import BadRequest, NotFound

LEVEL_LABELS = {
    "total": "Total (all stores, all items)",
    "state": "State",
    "store": "Store",
    "category": "Category",
    "department": "Department",
    "state_category": "State × Category",
    "state_department": "State × Department",
    "store_category": "Store × Category",
    "store_department": "Store × Department",
    "item": "Item (across stores)",
    "item_state": "Item × State",
    "series": "Store-Item (bottom level)",
}

# Maps our level names onto the labels used in the Stage 7 measurement artefact
# (experiments/artifacts/uc11_hierarchy_levels.csv), so the UI can show the
# MEASURED accuracy for whatever level the user is looking at.
LEVEL_TO_MEASURED = {
    "total": "L1_total",
    "state": "L2_state",
    "store": "L3_store",
    "category": "L4_cat",
    "department": "L5_dept",
    "state_category": "L6_state_cat",
    "state_department": "L7_state_dept",
    "store_category": "L8_store_cat",
    "store_department": "L9_store_dept",
    "item": "L10_item",
    "item_state": "L11_item_state",
}


def _node_expr(cols: list[str]) -> str:
    """SQL expression producing a single node id from the level's columns."""
    if not cols:
        return "'ALL'"
    if len(cols) == 1:
        return cols[0]
    return " || '|' || ".join(cols)


@ttl_cache()
def list_levels() -> list[dict]:
    out = []
    for level, cols in SAFE_LEVELS.items():
        expr = _node_expr(cols)
        row = query_one(
            f"SELECT count(DISTINCT {expr}) AS n FROM series")
        out.append({
            "level": level,
            "label": LEVEL_LABELS[level],
            "node_count": int(row["n"]) if row else 0,
            "columns": cols,
        })
    return out


@ttl_cache()
def list_nodes(level: str, parent_level: str | None = None,
               parent_id: str | None = None, limit: int = 500) -> list[dict]:
    cols = validate_level(level)
    if not cols:
        return [{"level": level, "node_id": "ALL", "label": "All stores & items",
                 "n_series": _total_series(), "mean_daily_sales": None}]

    expr = _node_expr(cols)
    where, params = "", []
    if parent_level and parent_id:
        pcols = validate_level(parent_level)
        if not pcols:
            where = ""
        else:
            pexpr = _node_expr(pcols)
            where = f"WHERE {pexpr} = ?"
            params.append(parent_id)

    rows = query(
        f"""
        SELECT {expr} AS node_id,
               count(*) AS n_series,
               avg(mean_daily_sales) AS mean_daily_sales
        FROM series {where}
        GROUP BY 1 ORDER BY 1 LIMIT ?
        """,
        params + [limit],
    )
    return [{"level": level, "node_id": r["node_id"], "label": r["node_id"],
             "n_series": int(r["n_series"]),
             "mean_daily_sales": float(r["mean_daily_sales"])} for r in rows]


@ttl_cache()
def _total_series() -> int:
    row = query_one("SELECT count(*) AS n FROM series")
    return int(row["n"]) if row else 0


def _series_filter(level: str, node_id: str) -> tuple[str, list]:
    """WHERE clause selecting the series belonging to one hierarchy node."""
    cols = validate_level(level)
    if not cols:
        return "", []
    parts = node_id.split("|")
    if len(parts) != len(cols):
        raise BadRequest(
            f"node_id '{node_id}' does not match level '{level}', which needs "
            f"{len(cols)} part(s): {'|'.join(cols)}"
        )
    clause = " AND ".join(f"{c} = ?" for c in cols)
    return f"WHERE {clause}", parts


@ttl_cache()
def aggregate_forecast(level: str, node_id: str) -> dict:
    """Coherent bottom-up forecast for one hierarchy node."""
    where, params = _series_filter(level, node_id)

    n = query_one(f"SELECT count(*) AS n FROM series {where}", params)
    if not n or n["n"] == 0:
        raise NotFound(f"no series found for {level}='{node_id}'",
                       level=level, node_id=node_id)

    rows = query(
        f"""
        SELECT f.horizon, f.day_idx, sum(f.yhat) AS yhat
        FROM forecast f
        JOIN (SELECT series_idx FROM series {where}) s USING (series_idx)
        GROUP BY 1, 2 ORDER BY 1
        """,
        params,
    )
    return {
        "level": level,
        "node_id": node_id,
        "n_series": int(n["n"]),
        "points": rows,
        "total_28d": float(sum(r["yhat"] for r in rows)),
    }


@ttl_cache()
def aggregate_history(level: str, node_id: str, days: int = 90) -> list[dict]:
    """
    Daily actuals for one node over the trailing `days` before the origin.

    Joins the history sidecar to `series` on series_idx, which is why the
    sidecar carries no identifier columns: they live once in `series` instead of
    being repeated across 59.2M rows.
    """
    cols = validate_level(level)
    if cols:
        parts = node_id.split("|")
        where = "WHERE " + " AND ".join(f"s.{c} = ?" for c in cols)
        params = list(parts)
    else:
        where, params = "", []

    origin = settings.forecast_origin_idx
    return query(
        f"""
        SELECT c.date AS date, h.day_idx AS day_idx, sum(h.sales) AS sales
        FROM {history_source()} h
        JOIN series s USING (series_idx)
        JOIN calendar c ON c.day_idx = h.day_idx
        {where}
        {"AND" if where else "WHERE"} h.day_idx > ? AND h.day_idx <= ?
        GROUP BY 1, 2 ORDER BY 2
        """,
        params + [origin - days, origin],
    )


@ttl_cache()
def measured_accuracy(level: str) -> dict | None:
    """
    The accuracy MEASURED for this aggregation level during the Stage 7 audit.

    This exists so the UI can never show a single global accuracy number. The
    same forecast is ~28% accurate at store-item level and ~95% accurate
    chain-wide; showing either one next to the wrong view is the most likely way
    a demand-forecasting UI misleads its user.
    """
    key = LEVEL_TO_MEASURED.get(level)
    if key is None:
        return None
    row = query_one(
        "SELECT level, n_groups, agg_RMSE, agg_MAE, agg_WAPE "
        "FROM level_accuracy WHERE level = ?", [key])
    if not row:
        return None
    wape = float(row["agg_WAPE"])
    return {
        "measured_level": row["level"],
        "n_groups": int(row["n_groups"]),
        "rmse": round(float(row["agg_RMSE"]), 4),
        "mae": round(float(row["agg_MAE"]), 4),
        "wape": round(wape, 4),
        "accuracy_pct": round(100 * (1 - wape), 1),
        "basis": ("Measured on the held-out validation window d_1914-d_1941 "
                  "by aggregating the frozen model's bottom-level forecasts to "
                  "this level."),
    }


@ttl_cache()
def search(term: str, limit: int = 25) -> list[dict]:
    """Typeahead over item and store identifiers."""
    if len(term) < 2:
        raise BadRequest("search term must be at least 2 characters")
    like = f"%{term.upper()}%"
    return query(
        """
        SELECT series_idx, id, item_id, dept_id, cat_id, store_id, state_id,
               volume_tier, regime, mean_daily_sales, zero_pct
        FROM series
        WHERE upper(item_id) LIKE ? OR upper(store_id) LIKE ? OR upper(id) LIKE ?
        ORDER BY mean_daily_sales DESC
        LIMIT ?
        """,
        [like, like, like, limit],
    )
