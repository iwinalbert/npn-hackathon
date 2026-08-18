"""Per store-item series: detail, history and forecast."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ..config import settings
from ..services import series as svc

router = APIRouter(prefix="/series", tags=["series"])


@router.get("", summary="List/filter series")
def list_series(
    store_id: str | None = None,
    item_id: str | None = None,
    dept_id: str | None = None,
    cat_id: str | None = None,
    state_id: str | None = None,
    regime: str | None = None,
    limit: int = Query(default=100, ge=1, le=settings.max_list_limit),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    return svc.list_series(store_id, item_id, dept_id, cat_id, state_id,
                           regime, limit, offset)


@router.get("/{store_id}/{item_id}", summary="Series metadata and regime")
def detail(store_id: str, item_id: str) -> dict:
    return svc.detail(store_id, item_id)


@router.get("/{store_id}/{item_id}/history", summary="Actual sales history")
def history(
    store_id: str, item_id: str,
    days: int = Query(default=settings.default_history_days, ge=1, le=1941),
) -> dict:
    return svc.history(store_id, item_id, days)


@router.get("/{store_id}/{item_id}/forecast",
            summary="Frozen 28-day forecast with empirical error bands")
def forecast(
    store_id: str, item_id: str,
    bands: bool = Query(default=True,
                        description="Attach empirical backtest error bands"),
) -> dict:
    return svc.forecast(store_id, item_id, with_bands=bands)
