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
    detail = db.health()
    core_ok = bool(detail.get("tables"))
    sidecars_ok = (detail.get("history_queryable")
                   and detail.get("backtest_queryable"))
    return {
        "ready": bool(core_ok),
        "degraded": bool(core_ok and not sidecars_ok),
        "environment": settings.environment,
        "version": settings.version,
        "detail": detail,
        "cache": cache_stats(),
    }
