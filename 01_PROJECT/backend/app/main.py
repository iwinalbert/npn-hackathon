"""
NPN Demand Forecasting API.

Serves the frozen M5 forecasting model's output to the product frontend.

WHAT THIS PROCESS DELIBERATELY DOES NOT IMPORT
----------------------------------------------
The research pipeline. `pipeline/config.py` calls `mkdir()` at import time,
which is a filesystem side effect on the protected research tree and can raise
under a read-only mount. Keeping that import out of the API's startup path means:

  * the research tree can be mounted strictly read-only, or not at all;
  * the API starts in well under a second instead of waiting ~14 s for the
    59M-row panel to load;
  * a missing or broken model cannot stop the API from serving.

The pipeline is imported lazily, inside the inference service, only when a
verification job actually runs.

STARTUP CONTRACT
----------------
Startup NEVER fails because an artefact is missing. It logs the problem and
starts anyway, so `/ready` can report precisely what is wrong. A container that
refuses to boot tells an operator far less than one that boots and explains
itself.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import db
from .config import settings
from .errors import ApiError
from .routers import (accuracy, genai, health, hierarchy, inference, insights,
                      meta, series)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='{"time":"%(asctime)s","level":"%(levelname)s",'
           '"logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("npn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting %s v%s (env=%s)",
             settings.app_name, settings.version, settings.environment)
    try:
        detail = db.health()
        if detail.get("tables"):
            log.info("data layer ready: tables=%s history=%s backtest=%s",
                     detail["tables"], detail.get("history_queryable"),
                     detail.get("backtest_queryable"))
        else:
            log.warning("data layer NOT ready: %s. Build it with "
                        "'python tasks.py build-db'", detail.get("errors"))
        for err in detail.get("errors", []):
            log.warning("artefact problem: %s", err)
    except Exception as exc:                                   # noqa: BLE001
        log.warning("startup probe failed (serving will continue): %s", exc)

    if settings.is_production and settings.cors_allow_all:
        log.error("cors_allow_all is enabled in production — refusing to widen "
                  "CORS; falling back to the explicit allow-list")

    yield
    db.close_connection()
    log.info("shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    description=(
        "Serves 28-day-ahead demand forecasts for 30,490 Walmart store-item "
        "series from a **frozen** LightGBM Tweedie blend "
        "(0.60 × direct + 0.40 × recursive; validation RMSE 2.0929 / "
        "MAE 1.0395 on 853,720 held-out predictions).\n\n"
        "**Scientific honesty is part of the API contract.**\n\n"
        "* Error bands are *empirical backtest error*, never model-produced "
        "prediction intervals — the model emits point forecasts only.\n"
        "* Accuracy is always reported for the aggregation level being viewed, "
        "because the same forecast is ~28% accurate per store-item and ~97% "
        "chain-wide.\n"
        "* No accuracy is claimed for the delivered forecast window: it has no "
        "ground truth.\n\n"
        "See `/api/v1/meta/capabilities` for what is implemented, what the "
        "research phase rejected on evidence, and what is not supported."
    ),
)

# Explicit allow-list. `cors_allow_all` is a development-only escape hatch and
# is ignored when environment=production.
_origins = settings.cors_origins
_allow_all = settings.cors_allow_all and not settings.is_production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _origins,
    allow_credentials=False,          # no cookies; nothing to protect via CORS
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-ms"],
    max_age=600,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id and duration to every response."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    request.state.request_id = rid
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
        raise
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    if ms > settings.slow_request_ms:
        log.warning("slow request rid=%s %s %.0fms", rid, request.url.path, ms)
    return response


def _problem(request: Request, status: int, error: str, message: str,
             **extra) -> JSONResponse:
    body = {"error": error, "message": message,
            "request_id": getattr(request.state, "request_id", None)}
    body.update({k: v for k, v in extra.items() if v is not None})
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    """Every deliberate failure returns the same shape. Covers all subclasses."""
    payload = exc.to_payload()
    payload["request_id"] = getattr(request.state, "request_id", None)
    if exc.status_code >= 500:
        log.error("api error rid=%s %s: %s",
                  payload["request_id"], exc.error_type, exc.message)
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request,
                                   exc: RequestValidationError):
    return _problem(request, 422, "validation_error",
                    "one or more parameters are invalid",
                    detail=exc.errors())


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Service-layer guards (e.g. unknown hierarchy level) raise ValueError."""
    return _problem(request, 400, "bad_request", str(exc))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    """
    Last resort. Never leaks a traceback to the client; the request id ties the
    response to the full traceback in the logs.
    """
    log.exception("unhandled rid=%s path=%s",
                  getattr(request.state, "request_id", None), request.url.path)
    return _problem(request, 500, "internal_error",
                    "an unexpected error occurred; see server logs",
                    error_type=type(exc).__name__)


for r in (health.router, meta.router, hierarchy.router, series.router,
          accuracy.router, insights.router, inference.router, genai.router):
    app.include_router(r, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "app": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "api": settings.api_prefix,
        "health": f"{settings.api_prefix}/health",
        "model": "FROZEN — 0.60 × direct(38f) + 0.40 × recursive(32f)",
        "validation": {"rmse": 2.0929, "mae": 1.0395,
                       "window": "d_1914–d_1941", "n": 853_720},
    }
