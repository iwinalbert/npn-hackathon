"""Liveness and readiness."""
from __future__ import annotations

from fastapi import APIRouter

from .. import db
from ..cache import stats as cache_stats
from ..config import settings

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness — is the process up?")
def health() -> dict:
    return {"status": "ok", "version": settings.version, "app": settings.app_name}


@router.get("/ready", summary="Readiness — are the data artefacts usable?")
def ready() -> dict:
    """
    Readiness deliberately reports DEGRADED rather than failing outright when the
    history panel is missing: forecasts and backtests still work without it, and
    a partially-useful API beats a dead one during a demo.
    """
    detail = db.health()
    core_ok = detail.get("product_db_exists") and detail.get("tables")
    return {
        "ready": bool(core_ok),
        "degraded": bool(core_ok and not detail.get("panel_queryable")),
        "detail": detail,
        "cache": cache_stats(),
    }
