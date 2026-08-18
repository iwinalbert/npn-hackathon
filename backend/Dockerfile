# syntax=docker/dockerfile:1
#
# NPN Demand Forecasting API
#
# Two build targets:
#
#   --target api    minimal service. No ML libraries, no research tree needed.
#                   Serves every endpoint except /inference/*, which reports
#                   503 with a reason. Smallest and fastest to start.
#
#   --target full   adds lightgbm/numpy/pandas so live model verification works.
#                   Requires the frozen model files and the research `pipeline`
#                   package to be mounted at runtime. This is the default.
#
# Build:
#   docker build -f backend/Dockerfile --target full -t npn-api:latest .
#
# The build context is the REPOSITORY ROOT — not backend — because
# the `full` target needs research/pipeline. The root .dockerignore is an
# allow-list that reduces that context from 2,582 MB to 0.4 MB: the datasets,
# model binaries and predictions are mounted read-only at runtime, never copied.

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root from the start. The API only ever reads its data, so it never needs
# write access to any mounted volume.
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencies — separate layer so code changes do not reinstall the world
# ---------------------------------------------------------------------------
FROM base AS deps-api
COPY backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

FROM base AS deps-full
# libgomp1 is LightGBM's OpenMP runtime; the wheel will not import without it.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/requirements-inference.txt ./
RUN pip install -r requirements-inference.txt

# ---------------------------------------------------------------------------
# api — minimal runtime
# ---------------------------------------------------------------------------
FROM deps-api AS api

COPY --chown=app:app backend/app ./app
COPY --chown=app:app backend/scripts ./scripts

ENV NPN_ENVIRONMENT=production \
    NPN_DATA_DIR=/data/product \
    NPN_ENABLE_INFERENCE=false \
    NPN_LOG_LEVEL=INFO \
    PORT=8000

USER app
EXPOSE 8000

# Uses /ready, not /health: readiness proves the data layer is actually
# queryable, whereas liveness only proves the process is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys,json;\
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready',timeout=4));\
sys.exit(0 if r.get('ready') else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]

# ---------------------------------------------------------------------------
# full — adds live inference (DEFAULT TARGET)
# ---------------------------------------------------------------------------
FROM deps-full AS full

COPY --chown=app:app backend/app ./app
COPY --chown=app:app backend/scripts ./scripts
# The frozen research feature pipeline, read-only. Required to rebuild features
# for live inference. It is NOT modified — the container runs as a non-root user
# with no write access to it.
COPY --chown=app:app research/pipeline /research/pipeline

# `pipeline/config.py` calls mkdir(parents=True, exist_ok=True) on seven
# directories AT IMPORT TIME. With NPN_PROJECT_ROOT=/research that means
# /research/{experiments/artifacts,experiments/registry,reports,
# models/experiments,models/champion,predictions/validation,
# predictions/final_forecast}.
#
# Three of those do not otherwise exist in this image, and /research would be
# owned by root — so importing the pipeline as uid 10001 would fail with
# PermissionError the first time /inference/verify ran. Creating them here,
# owned by app, fixes that, and has a second effect worth stating: Docker seeds
# a NAMED volume from the image path it is mounted over, so the scratch volumes
# inherit app ownership instead of being created root-owned and unwritable.
#
# The two read-only bind mounts land on directories that now already exist, and
# `exist_ok=True` means Python never attempts mkdir on them.
RUN mkdir -p /research/experiments/artifacts /research/experiments/registry \
             /research/reports \
             /research/models/champion /research/models/experiments \
             /research/predictions/validation /research/predictions/final_forecast \
 && chown -R app:app /research

ENV NPN_ENVIRONMENT=production \
    NPN_DATA_DIR=/data/product \
    NPN_PROJECT_ROOT=/research \
    NPN_MODEL_DIRECT=/research/models/champion/model_11_blend_direct_final_forecast.txt \
    NPN_MODEL_RECURSIVE=/research/models/champion/model_12_blend_recursive_shape_final.txt \
    NPN_FORECAST_CSV=/research/predictions/final_forecast/final_forecast_28day_v3_diversity_blend.csv \
    NPN_ENABLE_INFERENCE=true \
    NPN_LOG_LEVEL=INFO \
    PYTHONPATH=/research

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys,json;\
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready',timeout=4));\
sys.exit(0 if r.get('ready') else 1)"

# Single worker by design: DuckDB opens the database read-only per process, and
# each additional worker adds a full copy of the model + panel during inference.
# Scale with replicas behind a load balancer rather than in-process workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
