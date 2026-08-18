
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

FROM base AS deps-api
COPY backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

FROM base AS deps-full
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/requirements-inference.txt ./
RUN pip install -r requirements-inference.txt

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

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys,json;\
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready',timeout=4));\
sys.exit(0 if r.get('ready') else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]

FROM deps-full AS full

COPY --chown=app:app backend/app ./app
COPY --chown=app:app backend/scripts ./scripts
COPY --chown=app:app research/pipeline /research/pipeline

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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
