# Troubleshooting

Symptom → cause → fix. Ordered by how often each one actually happens.

**First move, always:**

```bash
python tasks.py preflight        # is the configuration sane?
python tasks.py docker-ps        # what is actually running?
python tasks.py smoke            # what specifically is broken?
```

`smoke` names the failing component. Start there rather than guessing.

---

## 1. `docker compose up` appears to hang. The frontend never starts.

**By far the most common failure, and it is not a hang.**

**Cause.** `backend/data/` is empty or incomplete. The API starts,
`/api/v1/ready` reports `ready: false`, its healthcheck therefore never passes,
and the frontend — which waits on `depends_on: condition: service_healthy` —
correctly never starts.

**Confirm:**

```bash
ls -la backend/data/          # want three files, ~130 MB total
curl -s localhost:8000/api/v1/ready      # ready: false, with a reason
docker compose ps                        # api: starting / unhealthy
```

**Fix:**

```bash
python tasks.py build-db
python tasks.py preflight                # confirms all three, with sizes
docker compose restart api
```

`docker-up` now runs preflight first, so this should be caught before the stack
starts at all.

---

## 2. Docker sends gigabytes of build context

**Symptom.** `transferring context: 2.3GB` instead of ~0.4 MB. Builds crawl.

**Cause.** `.dockerignore` is not being applied, or a new top-level directory
has no rule covering it. The file is written as an allow-list enumerated **per
top-level area**, so a newly added top-level folder is matched by nothing and is
sent wholesale. This has already happened once, when `study-materials`
was added after `.dockerignore` was written.

**Confirm:**

```bash
python tasks.py preflight
#   [  ok  ] top-level coverage  every top-level directory has a rule
#   [  ok  ] context size        56 files | 0.38 MB
```

**Fix.** Add the directory to `.dockerignore`:

```
06_WHATEVER/**
```

Also check you are building from the **repository root**. The backend Dockerfile
needs it — the `full` target does `COPY research/pipeline` — and a
`.dockerignore` in a different context directory does not apply.

---

## 3. `SMOKE TEST FAILED` — the canary is wrong

```
[ FAIL ] api: frozen forecast  total_28d=2891.44, expected 3331.3681
         -- the deployed data layer is NOT the validated one
```

**This is a release blocker.** The stack is serving, but serving wrong numbers.

**Causes, in order of likelihood:**

1. `product.duckdb` is stale or was built from different artefacts
2. `build_product_db.py` was interrupted, leaving a partial database
3. The mount points at the wrong directory
4. A frozen research artefact changed — serious

**Diagnose:**

```bash
curl -s localhost:8000/api/v1/meta/provenance | python -m json.tool
ls -la backend/data/
docker compose exec api ls -la /data/product
python tasks.py verify-integrity
```

**Fix.** For 1–3, rebuild:

```bash
python tasks.py clean-db && python tasks.py build-db
docker compose restart api && python tasks.py smoke
```

For 4 — if `verify-integrity` reports a changed artefact — **do not rebuild.**
The rebuild overwrites the evidence. Escalate.

---

## 4. A container will not start under `--prod` but works without it

**Cause.** `--prod` sets `read_only: true`. Something needs to write to a path
that has no `tmpfs`.

**Confirm:**

```bash
docker compose logs api | tail -30
# look for: "Read-only file system"
```

**Fix.** Add the specific path to that service's `tmpfs` list in
[`../compose/docker-compose.prod.yml`](../compose/docker-compose.prod.yml):

```yaml
    tmpfs:
      - /the/path/it/needs:size=32m
```

nginx already has the three it needs (`/var/cache/nginx`, `/var/run`, `/tmp`).
Add paths one at a time and note why — a read-only root is worth keeping.

Related: nginx also needs `CHOWN`, `SETGID` and `SETUID` back after `cap_drop:
ALL`, because its entrypoint drops privileges itself. Removing those three stops
the container booting.

---

## 5. `502 Bad Gateway` from the frontend

The SPA loads; every API call inside it fails.

**Cause.** nginx cannot reach the API. Either `API_HOST` is wrong, or the API is
not healthy.

**Confirm:**

```bash
docker compose ps                                  # is api healthy?
docker compose exec frontend env | grep API_HOST   # want api:8000
docker compose exec frontend wget -qO- http://api:8000/api/v1/health
```

**Fix.** `API_HOST` must be the **service name and internal port** — `api:8000`.
Not `localhost:8000` (that is the frontend container itself) and not the host's
address.

If the API is unhealthy, this is really problem 1.

---

## 6. nginx exits immediately: `host not found in upstream`

**Cause.** `envsubst` renders `${API_HOST}` into the config **before** nginx
starts, so `proxy_pass` gets a literal hostname — which nginx resolves at
startup and refuses to start without.

This bites whenever the two containers are not on a shared user-defined network,
for example `docker run` on the default bridge.

**Fix.** Put them on the same network:

```bash
docker network create npn-net
docker run -d --name api  --network npn-net npn-forecast-api:latest
docker run -d --name web  --network npn-net -e API_HOST=api:8000 \
  npn-forecast-frontend:latest
```

Compose does this for you. This only comes up when running containers by hand.

---

## 7. SPA deep links 404 on refresh

Clicking around works. Refreshing on `/forecast` gives a 404.

**Cause.** nginx's `try_files` fallback is missing or wrong. The client-side
router never sees the request.

**Confirm:**

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:8080/forecast   # want 200
```

**Fix.** `nginx.conf` must contain:

```nginx
location / {
    try_files $uri $uri/ /index.html;
}
```

If it is present and still failing, `envsubst` may have blanked nginx's own
`$uri` variable. `NGINX_ENVSUBST_FILTER=API_HOST` in the frontend Dockerfile is
what prevents that — confirm it is still set.

`smoke_test.py` checks this on every run.

---

## 8. The AI assistant says it is unavailable

**Not a bug** unless you configured a key. Every other feature is unaffected.

```bash
curl -s localhost:8000/api/v1/genai/status | python -m json.tool
```

| Reported reason | Fix |
|---|---|
| no key configured | put `GEMINI_API_KEY=...` in the `.env` beside `docker-compose.yml`, then `docker compose up -d --force-recreate api` |
| model 404 / NOT_FOUND | Google retired the model id. Set `NPN_GEMINI_MODEL=gemini-flash-latest` |
| timeout | check egress to `generativelanguage.googleapis.com` |

The key must be in the `.env` next to `docker-compose.yml`, not the one in
`backend/` — compose reads its own.

---

## 9. `/inference/*` returns 503

**By design** on the default stack. The lean `api` image has no LightGBM and no
research mounts, and it reports so with a reason rather than failing obscurely.

**Fix**, only if you actually want it:

```bash
python tasks.py docker-up --inference
```

That needs `research/models/champion` and
`research/predictions/final_forecast` present on the host.

---

## 10. `PermissionError` on `/research/...` during inference

**Cause.** `pipeline/config.py` calls `mkdir()` on seven directories **at import
time**. With `NPN_PROJECT_ROOT=/research`, three of them do not otherwise exist,
and `/research` would be root-owned while the container runs as uid 10001.

**This is already fixed** in the Dockerfile, which pre-creates and `chown`s all
ten paths. If you see it, the `full` image was built from a modified Dockerfile.

```bash
docker compose exec api id                              # uid=10001(app)
docker compose exec api ls -la /research
docker compose exec api python -c "import pipeline.config; print('ok')"
```

**Fix.** Restore the `RUN mkdir -p ... && chown -R app:app /research` block and
rebuild. Do not "fix" this by running as root.

---

## 11. Port already in use

```bash
python tasks.py stop-api            # port 8000, handles the Windows orphan case
python tasks.py stop-api 8080
```

On Windows, `uvicorn --reload` binds in a reloader parent and serves from a
spawned child. Kill the parent and the child survives, keeps the port, and keeps
serving whatever code it loaded at startup — so the app answers `/health`
perfectly while missing every route added since. `netstat` still attributes the
socket to the dead parent, so a plain `taskkill` reports "process not found".
`stop-api` handles this specifically.

---

## 12. `pip install` tries to compile a wheel

**Cause.** No prebuilt wheel for your platform. Almost always means you are not
on `linux/amd64` — arm64 (Graviton, Apple Silicon) has a different wheel matrix.

**Do not relax a version pin.** Those exact versions validated the frozen model.

**Fix.** Build for amd64, or re-verify the wheel matrix for your architecture
before changing anything:

```bash
docker build --platform linux/amd64 -f backend/Dockerfile .
```

`libgomp1` is LightGBM's OpenMP runtime and is required for the `full` target —
the wheel will not import without it.

---

## 13. CI is red but it works locally

Expected in one specific case: **CI has no data layer**, so data-dependent
backend tests skip and `/ready` is false there. That is by design and documented
in [`ci-cd.md`](ci-cd.md).

If CI fails on something else, the likely candidates are:

| CI failure | Meaning |
|---|---|
| `openapi.json is stale` | run `python tasks.py openapi` and commit |
| preflight `top-level coverage` | you added a top-level folder — add it to `.dockerignore` |
| `An API key shape was found in the built bundle` | a `VITE_`-prefixed variable leaked a secret into the frontend. Serious — Vite inlines those at build time |
| `runs as uid 0` | a Dockerfile lost its `USER app` line, or moved it after `CMD` |

---

## Escalation

Collect this before asking for help:

```bash
python tasks.py preflight --json  > preflight.json
python tasks.py smoke --json      > smoke.json
docker compose ps                 > ps.txt
docker compose logs --tail 200    > logs.txt
docker compose config             > resolved-compose.yml
python --version; node --version; docker compose version
```

Redact `logs.txt` before sharing it outside the team. It should contain no
secrets — the API is built so that it cannot — but check rather than assume.
