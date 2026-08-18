"""Planning summaries derived from the frozen forecast."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ..config import settings
from ..services import insights as svc

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("/top-movers",
            summary="Series rising or falling most against their recent run-rate")
def top_movers(
    level: str = Query(default="total"),
    node_id: str = Query(default="ALL"),
    limit: int = Query(default=20, ge=1, le=100),
    direction: str = Query(default="both", pattern="^(both|up|down)$"),
) -> dict:
    return svc.top_movers(level, node_id, limit, direction)


@router.get("/summary", summary="Headline planning numbers for a node")
def summary(
    level: str = Query(default="total"),
    node_id: str = Query(default="ALL"),
) -> dict:
    return svc.portfolio_summary(level, node_id)


@router.get("/planning/{store_id}/{item_id}",
            summary="28-day planning view for one series")
def planning(store_id: str, item_id: str) -> dict:
    return svc.planning_summary(store_id, item_id)
