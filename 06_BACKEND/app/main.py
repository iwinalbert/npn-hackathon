"""
NPN Demand Forecasting API.

Serves the frozen M5 forecasting model's output to the product frontend.

WHAT THIS PROCESS DELIBERATELY DOES NOT DO
------------------------------------------
It does not import the research pipeline. `pipeline/config.py` calls `mkdir()`
at import time, which is a filesystem side effect on the protected research tree
and can raise under a read-only mount. Keeping that import out of the API means:

  * the research tree can be mounted strictly read-only;
  * the API starts in well under a second instead of waiting ~14 s for the
    59M-row panel to load;
  * a crash in model code cannot take down the API.

Live model inference happens in a separate worker process (see app/worker/).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import db
from .config import settings
from .errors import ApiError
from .routers import health, hierarchy, meta, series

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s",'
           '"logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("npn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting %s v%s", settings.app_name, settings.version)
    try:
        detail = db.health()
        if detail.get("product_db_exists"):
            log.info("product db ready: %s", detail.get("tables"))
        else:
            log.warning("product database missing at %s — run "
                        "06_BACKEND/scripts/build_product_db.py",
                        settings.product_db)
    except Exception as exc:                                   # noqa: BLE001
        log.warning("startup db probe failed: %s", exc)
    yield
    db.close_connection()
    log.info("shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
    description=(
        "Serves 28-day-ahead demand forecasts for 30,490 Walmart store-item "
        "series from a **frozen** LightGBM Tweedie blend "
        "(0.60 × direct + 0.40 × recursive; validation RMSE 2.0929 / MAE 1.0395).\n\n"
        "**Scientific honesty is part of the contract.** Error bands are "
        "empirical backtest error, never model-produced intervals; accuracy is "
        "always reported for the aggregation level being viewed; and no accuracy "
        "is claimed for the delivered forecast window, which has no ground truth. "
        "See `/meta/capabilities` for what is implemented, what research rejected, "
        "and what is not supported."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,   # explicit allow-list, never "*"
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id and duration to every response."""
    rid = uuid.uuid4().hex[:12]
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
    if ms > 1000:
        log.warning("slow request rid=%s %s %.0fms", rid, request.url.path, ms)
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    payload = exc.to_payload()
    payload["request_id"] = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(db.DatabaseUnavailable)
async def db_unavailable_handler(request: Request, exc: db.DatabaseUnavailable):
    return JSONResponse(
        status_code=503,
        content={
            "error": "service_unavailable",
            "message": str(exc),
            "remedy": "python 06_BACKEND/scripts/build_product_db.py",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "bad_request", "message": str(exc),
                 "request_id": getattr(request.state, "request_id", None)},
    )


for r in (health.router, meta.router, hierarchy.router, series.router):
    app.include_router(r, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "app": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "api": settings.api_prefix,
        "model": "FROZEN — 0.60 × direct(38f) + 0.40 × recursive(32f)",
    }
