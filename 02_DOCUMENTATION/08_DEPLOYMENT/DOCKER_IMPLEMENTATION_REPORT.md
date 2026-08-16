# DOCKER IMPLEMENTATION REPORT

**Retail Demand Forecasting** — Walmart M5 · Hierarchical 28-Day Forecasting
**Scope:** containerise the existing system for a teammate to deploy later
**Status:** configuration complete and statically verified · **images NOT built —
Docker is not installed on this machine**

> **Read this first.** Every path, permission, dependency and secret boundary has
> been checked by static analysis and by running the application under the
> container's exact environment contract. But `docker build` has never run here.
> §12 lists the first commands to run, and what to expect from each.

---

## 1. Architecture

```
DEFAULT STACK — no 03_RESEARCH dependency at all

Browser  ──►  frontend container (nginx :8080)
                 │  serves the built SPA
                 │  /api/  ──proxy──►  api container (uvicorn :8000, target: api)
                                          ├─ /data/product  ro   product data layer (125 MB)
                                          └─ HTTPS ──► Gemini API (only if a key is set)

WITH docker-compose.inference.yml — adds live model verification only

                                       api container (target: full)
                                          ├─ /research/pipeline                    in image (0.18 MB)
                                          ├─ /research/models/champion             ro mount
                                          └─ /research/predictions/final_forecast  ro mount
```

**The default deployment does not need the research tree.** Verified against an
empty project root: the frozen 28-day forecast (`total_28d = 3331.3681`),
hierarchy, accuracy, insights, `/docs`, `/openapi.json` and every `/genai` route
all respond; `/inference/status` reports unavailable with a reason. The machine
running this stack can have no `03_RESEARCH/` directory whatsoever.

One origin from the browser's point of view: nginx proxies `/api/` internally to
`api:8000`, so no CORS is exercised, no API host is compiled into the bundle, and
the Gemini key never leaves the API container.

## 2. Images

| | api target | full target *(default)* | frontend |
|---|---|---|---|
| Base | `python:3.13-slim` | `python:3.13-slim` + `libgomp1` | `node:22-alpine` → `nginx:1.27-alpine` |
| Deps | fastapi, uvicorn, pydantic, duckdb, google-genai | + lightgbm, numpy, pandas, pyarrow | nginx only |
| Live inference | disabled — `/inference/*` returns 503 with a reason | enabled | n/a |
| Needs `03_RESEARCH` | **no** | yes (two ro mounts + pipeline) | no |
| Compose default | **yes** | opt-in overlay | yes |
| Runs as | `app` (uid 10001) | `app` (uid 10001) | nginx (see §10) |

**Why `api` is the default.** A deployment should not depend on a 2 GB research
tree it does not need. `api` serves every route the product actually ships —
including the frozen forecast — from three generated files, so the deployable
unit is `01_PROJECT/` plus a data directory.

`full` exists because live model verification is a genuine credibility asset:
`POST /api/v1/inference/verify` re-runs the frozen boosters and shows
`max_abs_diff = 0.000e+00` against the shipped forecast. That is worth having,
but it is worth having *opt-in*, because it costs ~104 MB of extra wheels, ~1 GB
more memory, and a dependency on artefacts that live outside the product.

## 3. Build contexts — the part that needed fixing

The backend context **must** be the repository root, because the `full` target
does `COPY 03_RESEARCH/pipeline /research/pipeline`. A context rooted at
`01_PROJECT/` cannot reach the research tree.

That means the context root also contains ~2.3 GB of datasets, model binaries,
predictions and documentation. **`.dockerignore` was stale after the
reorganisation** — every path in it (`data/`, `models/`, `06_BACKEND/`,
`07_FRONTEND/` …) referred to directories that no longer exist, so every
exclusion was dead and the entire 2.3 GB would have been sent to the daemon.

It is now an **allow-list**: exclude each top-level area wholesale, then
re-include only the five paths a `COPY` actually names. A deny-list silently
starts shipping anything added later; this fails the other way.

Simulated with the builder's own rule (patterns in order, last match wins):

```
files SENT     :     56   (0.38 MB)
files excluded : 11,733   (2,582 MB)

    0.20 MB   32 files  01_PROJECT/backend      (app/, scripts/, 2 requirements files)
    0.18 MB   22 files  03_RESEARCH/pipeline
    0.00 MB    2 files  .dockerignore, .env.example
```

A **6,400× reduction**, and asserted rather than eyeballed: eleven required files
confirmed present, and `.env`, the product database, raw CSVs, model binaries,
predictions and the registry all confirmed absent.

The frontend uses its own context (`01_PROJECT/frontend`) with its own
`.dockerignore` excluding `node_modules`, `dist`, `.env` — already correct.

## 4. A container-breaking bug found and fixed

`pipeline/config.py` calls `mkdir(parents=True, exist_ok=True)` on seven
directories **at import time**. Reproduced against an empty root, it creates ten:

```
experiments/  experiments/artifacts/  experiments/registry/
models/  models/champion/  models/experiments/
predictions/  predictions/final_forecast/  predictions/validation/
reports/
```

In the `full` image `NPN_PROJECT_ROOT=/research`, and `/research` would be owned
by **root** while the container runs as **uid 10001**. Three of those directories
exist nowhere — so the first call to `/inference/verify` would have died with
`PermissionError`. The API itself would have started fine, which is what makes
this the kind of bug that surfaces during a demo.

Fixed in the Dockerfile — a project-side change, with the research tree
untouched:

```dockerfile
RUN mkdir -p /research/experiments/artifacts /research/experiments/registry \
             /research/reports \
             /research/models/champion /research/models/experiments \
             /research/predictions/validation /research/predictions/final_forecast \
 && chown -R app:app /research
```

This has a second effect worth knowing: Docker seeds a **named** volume from the
image path it is mounted over, so `npn-scratch` and `npn-scratch-preds` inherit
`app` ownership instead of being created root-owned and unwritable. The two
read-only bind mounts land on directories that now already exist, and
`exist_ok=True` means Python never attempts `mkdir` on them.

## 5. Compose services

Two files. `docker-compose.yml` is the deployable stack; `docker-compose.inference.yml` is an opt-in overlay that adds live model verification and, with it, the only dependency on `03_RESEARCH`.

| Service | Image | Port | Depends on |
|---|---|---|---|
| `api` | `npn-forecast-api:latest` (target `full`) | 8000 | — |
| `frontend` | `npn-forecast-frontend:latest` | 8080 | `api` → `service_healthy` |

The frontend waits for the API's **readiness**, not merely its process, so the
first page load cannot race the database open.

### Volumes

**Default stack — one mount, read-only:**

| Mount | Mode | Why |
|---|---|---|
| `./01_PROJECT/backend/data` → `/data/product` | **ro** | product.duckdb + two parquet sidecars: everything the API needs to serve every shipped route |

**With the inference overlay, three more:**

| Mount | Mode | Why |
|---|---|---|
| `./03_RESEARCH/models/champion` → `/research/models/champion` | **ro** | the two frozen boosters |
| `./03_RESEARCH/predictions/final_forecast` → `/research/predictions/final_forecast` | **ro** | the shipped forecast, compared against |
| `npn-scratch`, `npn-scratch-preds` | rw | absorb the import-time mkdirs (§4) |

No research path is ever writable. The frozen artefacts cannot be modified by
the running system even by accident.

### Environment

| Variable | Value | Notes |
|---|---|---|
| `NPN_DATA_DIR` | `/data/product` | |
| `NPN_PROJECT_ROOT` | `/research` | |
| `NPN_MODEL_DIRECT` / `_RECURSIVE` | absolute paths under `/research` | set in the Dockerfile |
| `NPN_FORECAST_CSV` | absolute path under `/research` | |
| `NPN_ENABLE_INFERENCE` | `true` | `false` in the `api` target |
| `NPN_CORS_ORIGINS` | dev origins | unused in this topology — same origin |
| `API_HOST` *(frontend)* | `api:8000` | substituted into nginx at start |
| **`GEMINI_API_KEY`** | `${GEMINI_API_KEY:-}` | **runtime interpolation only** |

## 5b. Why the Dockerfiles are not in a `docker/` directory

A `docker/backend/`, `docker/frontend/` layout was considered and rejected.
Build context is the reason:

* **The backend Dockerfile must resolve `03_RESEARCH/pipeline`** for the `full`
  target, so its context is the repository root either way. Moving the file
  changes nothing about that and adds a level of indirection between the
  Dockerfile and the tree it copies from.
* **The frontend Dockerfile's context IS its own directory.** It does
  `COPY package.json`, `COPY . .` and `COPY nginx.conf`. Moving it to
  `docker/frontend/` would separate it from `nginx.conf` and from the only
  context it can be built with — `docker build -f docker/frontend/Dockerfile
  01_PROJECT/frontend` is harder to read, not easier.
* A `docker/` directory would also be a fourth top-level area, which the
  three-area structure exists specifically to avoid.

Each Dockerfile therefore sits with the thing it builds, and the two compose
files sit at the root where the build contexts resolve from. What a deployment
teammate needs to find is `docker-compose.yml` and this report — both at
predictable locations.

## 6. Secret handling

The key is never in an image, a layer, a log, the bundle, or git.

| Control | Evidence |
|---|---|
| Not in any Dockerfile | no `GEMINI_API_KEY=<value>` anywhere; static check |
| Not in the build context | `.dockerignore` excludes `**/.env` and `**/.env.*`; simulated and asserted |
| Injected at runtime | `GEMINI_API_KEY: ${GEMINI_API_KEY:-}` — compose interpolation from host env or a root `.env` |
| Never reaches the browser | 0 key-shaped strings in `dist/`; the frontend has no AI SDK and no Gemini endpoint |
| Never logged | 0 occurrences in the API log across a full endpoint sweep |
| Never in a response | `SecretStr`, plus `scrub_secrets()` on every reply |
| Never in git | `git log -p --all` → 0 key-shaped strings; `.env` ignored |

Docker is **not** used as a secret store. For real deployment the teammate should
supply `GEMINI_API_KEY` through the platform's secret mechanism.

Leaving it unset is fully supported: the assistant reports why it is unavailable,
and the local guardrails still work because they never call Gemini at all.

## 7. Dependency policy — versions unchanged

The frozen model was validated against specific versions. None were changed. The
risk with a Linux image is that a pin has no prebuilt wheel and pip tries to
compile. Checked directly against PyPI for `cp313` / `manylinux_2_28`:

| Package | Wheel | Size |
|---|---|---|
| `lightgbm==4.7.0` | `py3-none-manylinux_2_27_x86_64.manylinux_2_28` | 3.5 MB |
| `numpy==2.5.1` | `cp313-cp313-manylinux_2_27.manylinux_2_28` | 16.7 MB |
| `pandas==3.0.5` | `cp313-cp313-manylinux_2_24.manylinux_2_28` | 10.9 MB |
| `pyarrow==25.0.1` | `cp313-cp313-manylinux_2_28` | 50.1 MB |
| `duckdb==1.5.5` | `cp313-cp313-manylinux_2_26.manylinux_2_28` | 21.5 MB |
| `fastapi==0.135.1`, `google-genai==2.18.1` | pure python | 1.2 MB |

**All prebuilt. No compilation, no version change.** `python:3.13-slim` is Debian
bookworm (glibc 2.36), which satisfies `manylinux_2_28`. `libgomp1` is installed
because LightGBM's wheel will not import without OpenMP — that is the only system
package required.

## 8. Verification performed

### Static (26/26 passed)
Portability (no Windows paths, no localhost as a service address), non-root in
both backend targets with `USER` before `CMD`, healthchecks in all three images,
base images pinned, secret boundaries, read-only research mounts, no wholesale
research copy, compose structure, SPA fallback, and envsubst restricted to
`API_HOST` so nginx's own `$host`/`$uri` are never clobbered.

### Runtime, under the container's exact environment contract
The API was started locally with the container's env vars — pipeline resolved
from an image-like `/research` root, model and forecast from mount-like absolute
paths:

| Check | Result |
|---|---|
| 15 endpoints (`health`, `ready`, `meta/*`, `hierarchy/*`, `series/*`, `accuracy/*`, `insights/*`, `inference/status`, `genai/*`) | **all 200** |
| Frozen forecast under container paths | `total_28d = 3331.3681` ✅ |
| Live inference availability | `available: true`, no blocking reasons — proves the pipeline imports from the image-like root |
| GenAI guardrail with no model call | forecast-mutation request **refused locally** |
| Secrets in logs | **0** |

### Application suites
| Suite | Result |
|---|---|
| Backend | **149 passed** |
| Frontend | **62 passed** |
| TypeScript | clean |
| Production build | 2.10 s, `dist/` 768 KB |
| Slow — live model reproduces the frozen forecast | **2 passed** |
| Protected artefacts | **522 files · 0 deleted · 0 modified** |
| Path resolution | **8 / 8** |

### NOT verified
`docker compose config`, `build`, `up`, container startup time, image sizes,
container memory. Docker is not installed here — no binary, no Docker Desktop, no
service. These are the first things to run (§12).

## 9. Resource expectations

Measured inputs, since images could not be built:

| | |
|---|---|
| Backend build context | 0.38 MB |
| Python wheels in the `full` image | ~104 MB |
| Frontend `dist/` | 768 KB |
| Product data layer (mounted, not baked) | 125 MB |
| Research read-only mounts | 143 MB |

Rough expectation: `api` target ~250–300 MB, `full` ~450–550 MB, frontend ~55 MB.
**These are estimates from wheel and base-image sizes, not measurements.**

The `full` image is larger because the frozen model requires LightGBM plus a
NumPy/Pandas/PyArrow stack to rebuild features. That is the cost of being able to
prove the model still reproduces its forecast; the `api` target avoids it.

Runtime memory: idle ~200 MB; a live inference job peaks near 1 GB because it
loads the 59.2M-row panel. The compose limit is 2 GB for that reason.

## 10. Known limitations

1. **No image has been built.** The single largest caveat.
2. **The product data layer is a prerequisite.** `01_PROJECT/backend/data/` must
   contain `product.duckdb`, `history.parquet` and `backtest.parquet` before
   `up`. It is gitignored (125 MB, rebuildable). If empty, the API starts but
   `/ready` reports not-ready, its healthcheck never passes, and the frontend —
   which waits for `service_healthy` — never starts. **That looks like a hang; it
   is a missing data layer.** Fix: `python tasks.py build-db`.
   *An in-container bootstrap service was considered and deliberately not built:
   it could not be tested here, and an untested bootstrap path is worse than a
   documented one-line prerequisite.*
3. **The frontend container's nginx master runs as root.** The official image's
   entrypoint needs root to render templates. Workers drop to `nginx`. Port 8080
   is unprivileged, so running fully rootless is possible but was not attempted
   without the ability to test it.
4. **Single worker by design.** DuckDB opens read-only per process and each
   worker would add a full model+panel copy during inference. Scale with
   replicas, not `--workers`.
5. **The inference overlay is the only research dependency, and is untested in
   a container.** The default stack was verified research-free by running the
   API against an empty project root. The overlay's mounts and the §4 permission
   fix are statically checked but have never been exercised by a real build.
6. **`linux/amd64` assumed.** The wheel check targeted `manylinux_2_28_x86_64`.
   On arm64 (Apple Silicon, Graviton) the wheels differ and must be re-checked.
7. **No TLS, no auth, no rate limiting.** Appropriate for a local production-like
   stack; all three belong at the edge in a real deployment.

## 11. Local run

```bash
python tasks.py build-db                   # ONCE — materialises the 125 MB data layer
echo "GEMINI_API_KEY=your-key" > .env      # optional, beside docker-compose.yml
docker compose up --build                  # http://localhost:8080
```

Or through the task runner, which works identically on Windows and prints a
clear message if Docker is not installed:

```bash
python tasks.py docker-build     # build images
python tasks.py docker-up        # start (detached) and print the URLs
python tasks.py docker-ps        # container status + health
python tasks.py docker-logs      # follow logs
python tasks.py docker-down      # stop and remove
```

Add `--inference` to any of those to use the overlay:

```bash
python tasks.py docker-up --inference
# equivalent to:
docker compose -f docker-compose.yml -f docker-compose.inference.yml up -d --build
```

| | |
|---|---|
| App | <http://localhost:8080> |
| API docs | <http://localhost:8000/docs> |
| Readiness | <http://localhost:8000/api/v1/ready> |

**The development workflow is unchanged.** `python tasks.py api` and
`python tasks.py ui` still run the app directly on the host with hot reload;
Docker is an additional path, not a replacement.

Without a `GEMINI_API_KEY` everything works except the assistant, which explains
why it is unavailable.

## 12. First commands once Docker is available

Run in order. Each has a specific thing to look for.

```bash
# 1. does compose resolve? interpolation, paths, volumes
docker compose config

# 2. build. Expect the context upload to be ~0.4 MB, not gigabytes —
#    if it says "Sending build context ... GB", .dockerignore is not being read.
docker compose build

# 3. start
docker compose up -d

# 4. both must reach healthy; the frontend will not start until api is healthy
docker compose ps

# 5. the checks that matter most
curl -s localhost:8000/api/v1/ready              # {"ready": true}
curl -s localhost:8080/ -o /dev/null -w '%{http_code}\n'          # 200
curl -s localhost:8080/forecast -o /dev/null -w '%{http_code}\n'  # 200 — SPA deep link
curl -s localhost:8080/api/v1/health              # 200 — proxy works
curl -s localhost:8000/api/v1/series/CA_3/FOODS_3_090/forecast \
  | python -m json.tool | grep total_28d          # 3331.3681

# 6. THE regression test for §4 — only applies to the inference overlay, which
#    is the only configuration that imports the pipeline
docker compose -f docker-compose.yml -f docker-compose.inference.yml up -d --build
docker compose exec api python -c "import pipeline.config; print('pipeline import OK')"
docker compose exec api curl -s -X POST localhost:8000/api/v1/inference/verify

# 7. non-root
docker compose exec api id                        # uid=10001(app)

# 8. no secret baked into the image
docker run --rm npn-forecast-api:latest printenv GEMINI_API_KEY   # must be EMPTY
docker compose exec api printenv GEMINI_API_KEY                   # set, if you supplied one

# 9. measurements this report could not take
docker images npn-forecast-api npn-forecast-frontend
docker stats --no-stream
```

If the build fails on a dependency, **do not relax a version pin** — those
versions validated the frozen model. Check the platform first (§10.5).

## 13. For whoever does the cloud deployment

What this repository gives you, and what it deliberately does not.

**Provided:** platform-neutral Dockerfiles and compose, no cloud-specific
configuration, runtime env-var configuration for every path, health and readiness
endpoints suitable for any orchestrator probe, non-root API, read-only artefact
mounts, and a frontend that hard-codes no API host.

**You will need to decide:**

1. **How the 125 MB product data layer reaches the container.** It is bind-mounted
   here. In a cluster: bake it into a data image, use an init container that runs
   `build_product_db.py`, or mount object storage. It is rebuildable from the
   frozen artefacts, so it does not need to be backed up — but it must exist
   before the API reports ready.
2. **Whether live inference is needed.** The default already says no: plain
   `docker compose up` gives you the lean image with no research mounts and no
   LightGBM. Only add `-f docker-compose.inference.yml` if you want the
   "watch the frozen model reproduce its forecast" endpoint, and note that it
   requires `03_RESEARCH/models/champion` and
   `03_RESEARCH/predictions/final_forecast` to be present on the host.
3. **`GEMINI_API_KEY` via the platform's secret manager**, not an env file.
4. **TLS, authentication and rate limiting at the edge.** None are in this stack.
   Note that each assistant request costs money — rate limiting matters.
5. **Architecture.** Re-verify the wheel matrix if not `linux/amd64`.
6. **Readiness probe:** use `/api/v1/ready`, not `/api/v1/health`. Health only
   proves the process is alive; ready proves the data layer is queryable.

---

## 14. Files changed in this milestone

| File | Change |
|---|---|
| `.dockerignore` | rewritten as an allow-list — the previous version was entirely stale and would have shipped a 2.3 GB context |
| `01_PROJECT/backend/Dockerfile` | pre-create + `chown` the ten directories `pipeline/config.py` makes at import; corrected the build-context comment |
| `docker-compose.yml` | `security_opt: no-new-privileges` on both services; corrected a stale comment claiming the API port was unpublished; documented the data-layer prerequisite and its hang-like failure mode |
| `02_DOCUMENTATION/08_DEPLOYMENT/DOCKER_IMPLEMENTATION_REPORT.md` | this document |

### Finalisation pass

| File | Change |
|---|---|
| `docker-compose.yml` | default target `full` → **`api`**; research mounts and scratch volumes moved out; memory limit 2G → 1G; documents that the stack needs no research tree |
| `docker-compose.inference.yml` | **new** — opt-in overlay carrying the entire `03_RESEARCH` dependency |
| `tasks.py` | `docker-config`, `docker-build`, `docker-up`, `docker-down`, `docker-logs`, `docker-ps`, each accepting `--inference`; explains itself if Docker is absent |
| `README.md` | Docker quick-start |
| this report | renamed from `DOCKER_DEPLOYMENT_REPORT.md` so there is one deployment document, not two |

No application code, no model, no dataset, no research artefact was modified.

> **Location note.** The task named
> `02_DOCUMENTATION/09_DEPLOYMENT/DOCKER_IMPLEMENTATION_REPORT.md`. The documentation
> tree already has `08_DEPLOYMENT/` (deployment) and `09_VALIDATION/`
> (verification), so this sits in `08_DEPLOYMENT/` beside `GIT_POLICY.md` rather
> than creating a second deployment folder.
