# Runbook

Operating a running stack. Procedures, not explanations — the reasoning lives in
[`deployment-guide.md`](deployment-guide.md).

---

## Health at a glance

```bash
python tasks.py docker-ps                       # both must say (healthy)
curl -s localhost:8000/api/v1/ready             # {"ready": true, ...}
curl -s localhost:8080/healthz                  # ok
python tasks.py smoke                           # full verification, ~2 s
```

| Signal | Healthy | Meaning if not |
|---|---|---|
| `docker compose ps` | both `(healthy)` | see [`troubleshooting.md`](troubleshooting.md) |
| `/api/v1/health` | 200 | the API process is down |
| `/api/v1/ready` | `ready: true` | process is up, **data layer is not queryable** |
| `/healthz` (8080) | `ok` | nginx is down |
| smoke canary | `3331.3681` | **serving the wrong data** — do not release |

---

## Release checklist

Run in order. Do not skip a step because the previous one passed.

```bash
# 1. clean state
git status                                  # no unexpected changes
git pull

# 2. data layer present and correct
python tasks.py build-db                    # idempotent; safe to re-run
python tasks.py preflight                   # must say PREFLIGHT PASSED

# 3. the checks CI cannot run (it has no data layer)
python tasks.py test                        # 149 backend tests
python tasks.py ui-test                     # 62 frontend tests
python tasks.py verify-integrity            # no frozen artefact changed

# 4. build and start
python tasks.py docker-up --prod

# 5. verify the deployment, not the build
python tasks.py smoke --prod

# 6. eyeball it
#    open http://localhost:8080, load a series, refresh on a sub-route
```

**Release gate:** step 5 must report `SMOKE TEST PASSED` with the canary line
`total_28d=3331.3681 OK`. Nothing else counts as verified.

Steps 2, 3 and 5 are exactly the things CI cannot prove, because a CI runner has
no data layer. See [`ci-cd.md`](ci-cd.md).

---

## Deploy a new version

```bash
python tasks.py docker-up --prod      # rebuilds changed layers, recreates
python tasks.py smoke --prod
```

Compose recreates only containers whose image or config changed. The API is
stateless and the data mount is read-only, so there is no migration and no
state to preserve.

Brief downtime is expected — this is a single-replica stack. For zero-downtime
you need two replicas behind a load balancer; see
[`deployment-guide.md`](deployment-guide.md) §7.

---

## Roll back

There is no database migration and no persistent state, so rollback is just
running the previous code.

```bash
git log --oneline -10
git checkout <previous-good-sha>
python tasks.py docker-up --prod
python tasks.py smoke --prod
```

The data layer is **version-independent** — it is generated from frozen
artefacts that never change. You do not need to rebuild it to roll back.

---

## Restart a service

```bash
docker compose restart api               # just the API
docker compose restart frontend
python tasks.py docker-down && python tasks.py docker-up    # everything
```

After any restart: `python tasks.py smoke`.

---

## Logs

```bash
python tasks.py docker-logs                        # follow both, last 100
docker compose logs -f api                         # one service
docker compose logs --since 15m api                # a time window
docker compose logs api | grep -iE 'error|warn'    # errors only
```

Under `--prod`, logs rotate at 10 MB × 5 files per service (30 MB and 50 MB
ceilings). Without `--prod` they are **unbounded** — a long-running stack on a
dev box will eventually fill the disk. That is the main reason to use `--prod`
for anything left running.

The API never logs the Gemini key: it is a `SecretStr`, and every response is
passed through `scrub_secrets()`.

---

## Rotate the Gemini API key

```bash
# 1. new key from https://aistudio.google.com/apikey
# 2. update the .env beside docker-compose.yml (or your platform's secret store)
# 3. recreate the API so it picks up the new environment
docker compose up -d --force-recreate api
# 4. confirm
curl -s localhost:8000/api/v1/genai/status
python tasks.py smoke
# 5. revoke the OLD key in the Google console -- this step is the one that
#    actually makes the rotation meaningful
```

The key exists only in the API container's environment. It is never in an image
layer, never in the JavaScript bundle, never in a log, never in a response body.
`preflight` and CI both assert this.

---

## Rebuild the data layer

Do this if the canary is wrong, `/ready` reports false, or the files were
deleted.

```bash
python tasks.py clean-db      # removes product.duckdb (always rebuildable)
python tasks.py build-db      # ~10 s
python tasks.py preflight
docker compose restart api    # the mount is read-only; the API must reopen it
python tasks.py smoke
```

`build-db` reads the frozen research artefacts, so it must run on a machine that
has `research/`. The **resulting files** can then be shipped anywhere — the
running stack itself never needs the research tree.

---

## Enable live model verification

Opt-in. It re-runs the two frozen LightGBM boosters and checks they still
reproduce the shipped forecast — measured `max_abs_diff = 0.000e+00`.

```bash
python tasks.py docker-up --inference
curl -X POST localhost:8000/api/v1/inference/verify
```

Costs: a larger image (+~104 MB of wheels), ~1 GB more memory during a run, and
a dependency on `research/models/champion` and
`research/predictions/final_forecast` being present on the host. A run takes
roughly 45 seconds; nginx's proxy read timeout is set to 180 s to accommodate it.

Do not enable this in a deployment that does not need it.

---

## Capacity

| | |
|---|---|
| API memory, idle | ~200 MB |
| API memory, inference run | ~1 GB (loads the 59.2M-row panel) |
| API memory limit | 1 GB default · 2 GB with `--inference` |
| Frontend memory | ~20 MB, limit 128 MB |
| Workers | **1, by design** |

**Do not raise `--workers`.** DuckDB opens the database read-only per process,
and each worker adds a full model + panel copy during inference. Scale with
replicas behind a load balancer instead.

```bash
docker stats --no-stream          # live usage
```

---

## Incident: the app is down

```bash
# 1. what is actually running?
python tasks.py docker-ps

# 2. why did it stop?
docker compose logs --tail 100 api
docker compose logs --tail 100 frontend

# 3. distinguish "process down" from "data down"
curl -s localhost:8000/api/v1/health      # process
curl -s localhost:8000/api/v1/ready       # data layer

# 4. restart
docker compose restart api

# 5. if it will not come back, fall back to the host run to isolate
#    whether the problem is the app or the container layer
python tasks.py api
python tasks.py smoke --skip-web
```

If the host run works and the container run does not, the problem is
configuration — a mount, an environment variable, or a permission — not the
application. Go to [`troubleshooting.md`](troubleshooting.md).

---

## Incident: wrong numbers

**Stop. Do not restart anything yet** — a restart destroys evidence.

```bash
curl -s localhost:8000/api/v1/series/CA_3/FOODS_3_090/forecast | python -m json.tool
# total_28d MUST be 3331.3681

curl -s localhost:8000/api/v1/meta/provenance | python -m json.tool
ls -la backend/data/
python tasks.py verify-integrity
```

Most likely cause: a stale or partially built `product.duckdb`. Rebuild it (see
above). If `verify-integrity` reports a changed frozen artefact, that is a much
more serious finding — escalate rather than rebuild, because the rebuild would
overwrite the evidence.

---

## Regular maintenance

| Task | When |
|---|---|
| `python tasks.py smoke` | after every deploy, restart, or config change |
| `docker system prune -f` | weekly on a dev box — build caches accumulate |
| Check log volume | weekly if running **without** `--prod` |
| Rotate the Gemini key | per your organisation's policy |
| `python tasks.py verify-integrity` | before any release |

---

## Command reference

```bash
python tasks.py help              # all commands

python tasks.py build-db          # generate the data layer            (once)
python tasks.py preflight         # pre-deploy gate     (+ --skip-data)
python tasks.py docker-up         # start   (+ --prod, --inference)
python tasks.py smoke             # verify  (+ --prod, --skip-web, --api URL)
python tasks.py docker-ps         # status and health
python tasks.py docker-logs       # follow logs
python tasks.py docker-down       # stop
python tasks.py clean-db          # delete the data layer  (rebuildable)
python tasks.py verify-all        # backend + frontend + build + integrity
```

`make <target>` wraps the same commands where `make` is available; pass flags as
`make docker-up ARGS=--prod`.
