"""Hierarchy navigation, coherent aggregation and search."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ..config import settings
from ..services import calendar as cal
from ..services import hierarchy as svc

router = APIRouter(prefix="/hierarchy", tags=["hierarchy"])


@router.get("/levels", summary="The 12 aggregation levels")
def levels() -> list[dict]:
    return svc.list_levels()


@router.get("/nodes", summary="Nodes at a level, optionally under a parent")
def nodes(
    level: str = Query(description="e.g. store, department, item"),
    parent_level: str | None = Query(default=None),
    parent_id: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[dict]:
    return svc.list_nodes(level, parent_level, parent_id, limit)


@router.get("/search", summary="Typeahead over items and stores")
def search(
    q: str = Query(min_length=2, description="item id, store id or series id"),
    limit: int = Query(default=25, ge=1, le=200),
) -> list[dict]:
    return svc.search(q, limit)


@router.get("/aggregate", summary="Coherent bottom-up forecast for a node")
def aggregate(
    level: str = Query(description="Aggregation level"),
    node_id: str = Query(default="ALL", description="Node id, '|'-joined"),
    history_days: int = Query(default=0, ge=0, le=730,
                              description="Include N days of actuals before the origin"),
) -> dict:
    agg = svc.aggregate_forecast(level, node_id)
    points = [{
        "date": cal.date_of(int(p["day_idx"]))[:10],
        "day_idx": int(p["day_idx"]),
        "horizon": int(p["horizon"]),
        "yhat": round(float(p["yhat"]), 4),
    } for p in agg["points"]]

    out = {
        "level": agg["level"],
        "node_id": agg["node_id"],
        "n_series": agg["n_series"],
        "origin_day": cal.day_label(cal.origin_day_idx()),
        "forecast": points,
        "total_28d": round(agg["total_28d"], 4),
        "expected_accuracy": svc.measured_accuracy(level),
    }
    if history_days:
        hist = svc.aggregate_history(level, node_id, history_days)
        out["history"] = [{"date": h["date"][:10], "sales": int(h["sales"])}
                          for h in reversed(hist)]
    return out
