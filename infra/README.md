# infra

**Everything needed to build, ship, run and debug this system.**
Retail Demand Forecasting · Walmart M5 · NPN AIA Hackathon

If you are the DevOps engineer and you read one file, read
[`docs/onboarding.md`](docs/onboarding.md). It gets you from a fresh
clone to a verified running stack in about thirty minutes.

---

## The one thing that will bite you

The API needs a **130 MB product data layer** that is **not in git**.

```
backend/data/
    product.duckdb        20 MB
    history.parquet       32 MB
    backtest.parquet      78 MB
```

Without it the API starts fine, `/api/v1/ready` reports `ready: false`, its
healthcheck never passes, and the frontend — which waits for `service_healthy` —
never starts at all.

**That looks exactly like a hang, and it is not one.** It is the single most
likely reason your first deploy appears broken. Build it once:

```bash
python tasks.py build-db          # ~10 s
```

`preflight` checks for it before every `docker-up`, so you should never actually
hit this. It is documented anyway because the failure is so much more confusing
than the fix.

---

## What is in here

| Path | What it is |
|---|---|
| [`scripts/preflight.py`](scripts/preflight.py) | Pre-deploy gate. Data layer, compose validity, build-context hygiene, secret hygiene, Dockerfile hardening, toolchain. **Needs no Docker.** |
| [`scripts/smoke_test.py`](scripts/smoke_test.py) | Post-deploy verification against a running stack. Stdlib only — runs on a CI runner, a bastion, or inside the container. |
| [`compose/docker-compose.prod.yml`](compose/docker-compose.prod.yml) | Production hardening overlay: API off the host network, read-only root filesystems, all capabilities dropped, log rotation, CPU limits. |
| [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml) | The CI pipeline. Lives at the repo root because GitHub requires that path. |

### Diagrams

| | |
|---|---|
| [`system-architecture.png`](../docs/04_ARCHITECTURE/system-architecture.png) | What runs where — containers, internal network, read-only mounts, egress |
| [`data-pipeline.png`](../docs/04_ARCHITECTURE/data-pipeline.png) | Where the numbers come from, how the image is built, what each check proves |

Both are generated — edit and re-run the scripts in
[`docs/04_ARCHITECTURE/diagrams/`](../docs/04_ARCHITECTURE/diagrams/).

### Documentation

| Doc | Read it when |
|---|---|
| [`docs/onboarding.md`](docs/onboarding.md) | You are new. Start here. |
| [`docs/deployment-guide.md`](docs/deployment-guide.md) | You are deploying — locally, on a VM, or to a cloud. |
| [`docs/runbook.md`](docs/runbook.md) | Something is running and you need to operate it, or release. |
| [`docs/ci-cd.md`](docs/ci-cd.md) | You need to know what CI proves — and what it does not. |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Something is broken. Symptom → cause → fix. |
| [`docs/environment-reference.md`](docs/environment-reference.md) | You need to know what a variable does before you set it. |
| [`docs/handover.md`](docs/handover.md) | You want the honest state of this work: verified, unverified, and next. |

Background on how the containers were designed (and one container-breaking bug
that was found and fixed):
[`docs/08_DEPLOYMENT/DOCKER_IMPLEMENTATION_REPORT.md`](../docs/08_DEPLOYMENT/DOCKER_IMPLEMENTATION_REPORT.md).

---

## The whole workflow

Every command works identically on Windows, macOS and Linux. `make` is not
installed on the Windows machine this project is demonstrated on, so `tasks.py`
is the primary entry point; the `Makefile` is a thin wrapper over it.

```bash
python tasks.py build-db        # ONCE   -- materialise the data layer
python tasks.py preflight       # gate   -- is it safe to build?
python tasks.py docker-up       # start  -- runs preflight first, then up -d
python tasks.py smoke           # verify -- is it actually serving correctly?
python tasks.py docker-logs     # watch
python tasks.py docker-down     # stop
```

Flags compose:

```bash
python tasks.py docker-up --prod              # + production hardening
python tasks.py docker-up --inference         # + live model verification
python tasks.py smoke --prod                  # test through the proxy
python tasks.py preflight --skip-data         # config checks only (CI)
```

| | |
|---|---|
| App | <http://localhost:8080> |
| API docs | <http://localhost:8000/docs> *(not published under `--prod`)* |
| Readiness | <http://localhost:8000/api/v1/ready> |

---

## The two probes, and why the difference matters

| Endpoint | Proves | Use it for |
|---|---|---|
| `/api/v1/health` | the process is alive | **liveness** probe |
| `/api/v1/ready` | the data layer is **queryable** | **readiness** probe |

Point your orchestrator's readiness probe at `/ready`. A stack with an empty
data mount answers `/health` with 200 forever while serving nothing — which is
precisely the failure described at the top of this page.

---

## Status

The container configuration is complete and statically verified. **No image has
ever been built** — Docker is not installed on the development machine. What is
verified, what is not, and what to run first once Docker is available:
[`docs/handover.md`](docs/handover.md).
