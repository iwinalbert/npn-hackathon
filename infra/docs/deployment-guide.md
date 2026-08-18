# Deployment Guide

How to deploy this system, and the decisions you have to make when you do.

Procedures for a stack that is already running are in
[`runbook.md`](runbook.md).

---

## 1. What the deployable unit actually is

```
backend/                     the FastAPI service
frontend/                    the React app + nginx
backend/data/                the 130 MB generated data layer
docker-compose.yml           the stack definition
infra/compose/*.prod.yml     production hardening
```

**`research/` is not part of a deployment.** The default stack was verified
against an empty project root: the frozen 28-day forecast, hierarchy, accuracy,
insights, `/docs`, `/openapi.json` and every `/genai` route all respond, and
`/inference/status` reports unavailable with a reason. The machine running this
can have no research tree whatsoever.

The one exception is live model verification (`--inference`), which is opt-in
precisely so the default deployment carries no dependency on it.

---

## 2. The data layer: your one real decision

The API requires three files at `NPN_DATA_DIR` (`/data/product` in a container):

```
product.duckdb      20 MB
history.parquet     32 MB
backtest.parquet    78 MB
```

They are **generated** by `backend/scripts/build_product_db.py` from
frozen research artefacts, gitignored, and always rebuildable — so they need no
backup. But they must exist before the API reports ready, and the script that
builds them needs `research/`, which your deployment target probably does not
have.

Four ways to resolve that. Pick one deliberately:

| Approach | How | Good for | Cost |
|---|---|---|---|
| **Bind mount** *(current)* | build on the host, mount read-only | single host, VM, laptop | host must have the research tree, or the files copied to it |
| **Bake a data image** | `FROM scratch` + `COPY` the three files; mount its volume | clusters, immutable infra | a 130 MB image per data version; rebuild to update |
| **Init container** | run `build_product_db.py` into a shared volume before the API starts | Kubernetes | the init image needs the research artefacts (~143 MB) |
| **Object storage** | fetch from S3/GCS/Blob into an `emptyDir` at start | many replicas, frequent redeploys | network dependency at startup; needs credentials |

**Recommended for a cluster: bake a data image.** The files are immutable and
version-independent, so an image is the most honest representation of them, and
it removes any startup-time dependency.

Whatever you choose, the readiness probe (§4) is what makes it safe: the API
will simply never report ready until the data is actually there.

---

## 3. Deployment paths

### 3a. Single host / VM — the supported path

```bash
git clone <repo> && cd NPN_HACKATHON
pip install -r backend/requirements-dev.txt
pip install pyyaml

python tasks.py build-db
echo "GEMINI_API_KEY=your-key" > .env       # optional

python tasks.py preflight
python tasks.py docker-up --prod
python tasks.py smoke --prod
```

Put a TLS-terminating reverse proxy in front of port 8080 (§6).

### 3b. If the host has no research tree

Build the data layer on a machine that does have it, then ship the three files:

```bash
# on a machine WITH research
python tasks.py build-db
tar czf product-data.tgz -C backend/data \
    product.duckdb history.parquet backtest.parquet     # ~130 MB

# on the deployment host
mkdir -p backend/data
tar xzf product-data.tgz -C backend/data
python tasks.py preflight        # confirms all three, with sizes
```

### 3c. Container platforms (Cloud Run, App Runner, Container Apps, ECS)

The images are platform-neutral — no cloud-specific configuration anywhere. What
these platforms need from you:

- **Two services**, or one if you serve the SPA from a CDN and expose the API
  directly. If you split them, you must set `VITE_API_BASE_URL` at frontend
  **build** time and configure `NPN_CORS_ORIGINS` on the API, because you have
  given up the same-origin property.
- **The data layer**, via §2. Most serverless container platforms have no
  persistent volume — bake a data image or fetch from object storage.
- **Readiness probe** on `/api/v1/ready`, liveness on `/api/v1/health` (§4).
- **`GEMINI_API_KEY`** from the platform's secret manager, never an env file.
- **`linux/amd64`.** The dependency wheels were checked for
  `manylinux_2_28_x86_64`. On arm64 (Graviton, Apple Silicon) re-verify them
  before assuming a build will succeed without compiling.

### 3d. Kubernetes

Not provided, deliberately — untested manifests are worse than none. If you
write them, the stack maps cleanly:

- `api` Deployment, 1+ replicas, `readinessProbe` → `/api/v1/ready`,
  `livenessProbe` → `/api/v1/health`
- data layer as a `ReadOnlyMany` PVC, or an init container, or a baked image (§2)
- `frontend` Deployment + Service, with `API_HOST` pointing at the API Service
- `GEMINI_API_KEY` from a Secret
- Ingress terminating TLS in front of the frontend Service

Start from `docker-compose.yml` and
[`../compose/docker-compose.prod.yml`](../compose/docker-compose.prod.yml) —
between them they specify every probe, limit, capability and mount mode you need.

---

## 4. Probes

| Probe | Endpoint | Why |
|---|---|---|
| **Liveness** | `/api/v1/health` | process is alive. Restart if it fails. |
| **Readiness** | `/api/v1/ready` | data layer is **queryable**. Do not route traffic until true. |
| Frontend | `/healthz` | nginx is serving |

Getting this backwards is the most consequential probe mistake available here:
if readiness points at `/health`, an API with a missing or broken data mount is
declared ready and receives traffic it cannot serve.

Suggested timings — the API opens DuckDB at startup, so allow for it:

```yaml
readinessProbe:
  httpGet: { path: /api/v1/ready, port: 8000 }
  initialDelaySeconds: 15
  periodSeconds: 10
  failureThreshold: 3
livenessProbe:
  httpGet: { path: /api/v1/health, port: 8000 }
  initialDelaySeconds: 20
  periodSeconds: 30
  failureThreshold: 3
```

Add ~10 s to the initial delays when using the `full` (inference) image — it
imports a much heavier dependency stack.

---

## 5. Secrets

One secret exists: `GEMINI_API_KEY`, and it is **optional**. Without it every
other feature works and `/genai/status` explains why the assistant is
unavailable.

| Boundary | Guarantee |
|---|---|
| Not in any image layer | asserted by CI (`printenv` + filesystem scan) |
| Not in the build context | `.dockerignore` excludes `**/.env`; asserted by preflight |
| Not in the browser bundle | asserted by CI (`grep` over `dist/`) |
| Not in logs | `SecretStr` + `scrub_secrets()` on every response |
| Not in git | asserted by preflight over all tracked files |
| Injected at runtime only | `GEMINI_API_KEY: ${GEMINI_API_KEY:-}` |

**Docker is not a secret store.** For a real deployment use the platform's
secret manager and inject at runtime. Rotation procedure:
[`runbook.md`](runbook.md).

Note that each assistant request costs money. Rate limiting matters (§6).

---

## 6. What this stack deliberately does NOT provide

Four things, all of which belong at the edge rather than inside the containers.
If you expose this publicly, you must add them:

1. **TLS.** Terminate at a load balancer or reverse proxy in front of port 8080.
   The API already sets `--proxy-headers --forwarded-allow-ips *`, so it honours
   `X-Forwarded-Proto`.
2. **Authentication.** There is none. Every endpoint is public.
3. **Rate limiting.** Particularly on `/api/v1/genai/ask`, which costs money per
   request, and `/api/v1/inference/verify`, which costs ~1 GB of memory and ~45
   seconds per call.
4. **A Content-Security-Policy header.** `nginx.conf` sets `X-Frame-Options`,
   `X-Content-Type-Options` and `Referrer-Policy` but no CSP, because the app
   loads no third-party resources and the right policy depends on your edge.

---

## 7. Scaling

**Single worker per container, by design.** DuckDB opens the database read-only
per process, and each additional worker adds a full model + panel copy during
inference. `--workers 2` roughly doubles memory for no throughput gain on the
read path.

Scale with **replicas** behind a load balancer:

- The API is stateless. The only mutable state is an in-process response cache
  (`cache_ttl_seconds`, default 300) and the inference job registry.
- The data mount is read-only, so any number of replicas can share it.
- **Inference jobs are in-process.** A job started on replica A is invisible to
  replica B. If you run multiple replicas *and* enable `--inference`, either
  pin verification to one replica or use sticky sessions. This is the one place
  where "stateless" is not quite true.

---

## 8. Before you call it deployed

```bash
python tasks.py smoke --prod
```

`SMOKE TEST PASSED`, including
`frozen forecast CA_3/FOODS_3_090 total_28d=3331.3681 OK`.

That canary is the point. A deployment can answer 200 on every endpoint and
still be serving a stale or half-built data layer; the canary is what
distinguishes "responding" from "correct".

Then check the release checklist in [`runbook.md`](runbook.md) — it
includes the verification steps CI structurally cannot perform.
