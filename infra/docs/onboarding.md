# DevOps Onboarding

**From a fresh clone to a verified running stack.** About 30 minutes, most of it
waiting for installs.

You do not need to understand the forecasting model to operate this system. You
do need to understand one thing about it, and it is in step 0.

---

## Step 0 — the one concept that matters

This is a **frozen-model** deployment. The model was trained, validated, and then
frozen. Nothing in production trains, retrains, or updates it. The API reads
precomputed results out of a database and serves them.

That has three consequences that shape every operational decision:

1. **There is no training job, no GPU, no model registry to sync.** The
   deployable unit is a stateless API plus a read-only data file.
2. **The data layer is generated, not stored.** A 130 MB DuckDB file plus two
   parquet sidecars, built from frozen artefacts by a script. It is gitignored
   and always rebuildable, so it never needs backing up — but it **must exist**
   before the API can serve anything.
3. **Correctness is checkable with one number.** The frozen forecast for series
   `CA_3 / FOODS_3_090` totals **3331.3681** over 28 days. If a deployment
   returns that, it is serving the validated data. If it returns anything else,
   it is not. `smoke_test.py` asserts exactly this.

Point 2 causes almost every "it's broken" report. Point 3 resolves almost every
"is it really working?" question.

---

## Step 1 — prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.11+ (3.13 used here) | everything |
| Node.js | 22+ | frontend build and tests |
| Docker + Compose v2 | any recent | containers |
| git | any | — |

```bash
python --version    # 3.11+
node --version      # 22+
docker compose version
```

If Docker is missing, everything except the container steps still works:
`python tasks.py api` and `python tasks.py ui` run the app directly on the host.

---

## Step 2 — install dependencies

```bash
# backend (from the repository root)
pip install -r backend/requirements-dev.txt
pip install pyyaml                      # preflight only

# frontend
cd frontend && npm ci && cd ../..
```

`npm ci`, not `npm install` — it honours the lockfile exactly and fails loudly
if `package.json` and the lockfile disagree.

**Do not relax a version pin** if something fails to install. Those exact
versions validated the frozen model. Diagnose the platform instead
(see [`troubleshooting.md`](troubleshooting.md)).

---

## Step 3 — build the data layer

```bash
python tasks.py build-db
```

Takes ~10 seconds and writes three files into `backend/data/`. Verify:

```bash
python tasks.py preflight
```

You want `PREFLIGHT PASSED`. Two warnings are normal and expected:

- `.env for compose absent` — you have not set a Gemini API key. Optional.
- `docker not on PATH` — only if Docker is genuinely not installed.

---

## Step 4 — run it on the host first

Containers add a layer of indirection. Confirm the application works before
adding it.

```bash
python tasks.py api          # terminal 1 -> http://localhost:8000/docs
python tasks.py ui           # terminal 2 -> http://localhost:5173
```

In a third terminal:

```bash
python tasks.py smoke --skip-web
```

Expected: **8 passed, 0 failures**, including
`frozen forecast CA_3/FOODS_3_090 total_28d=3331.3681 OK`.

That single line is your proof the whole chain works: process → DuckDB → query →
serialisation → HTTP.

Stop with Ctrl-C. If port 8000 is somehow still held afterwards — a known
Windows `uvicorn --reload` orphan — use `python tasks.py stop-api`.

---

## Step 5 — run it in containers

```bash
python tasks.py docker-up        # runs preflight, then builds and starts
python tasks.py docker-ps        # both services must reach (healthy)
python tasks.py smoke            # now including the frontend and the proxy
```

Then open <http://localhost:8080>.

Watch the build output for one thing: the context upload should be about
**0.4 MB**. If Docker reports sending gigabytes, `.dockerignore` is not being
read — stop and fix that before anything else.

Tear down with `python tasks.py docker-down`.

---

## Step 6 — the production topology

```bash
python tasks.py docker-up --prod
python tasks.py smoke --prod
```

`--prod` layers [`../compose/docker-compose.prod.yml`](../compose/docker-compose.prod.yml)
on top. Differences you will notice immediately:

- **Port 8000 is gone.** The API is reachable only through the frontend's nginx
  proxy, which is why `smoke --prod` tests through port 8080.
- Root filesystems are read-only, all Linux capabilities are dropped, and logs
  rotate at 10 MB × 5 files.

If a container fails to start under `--prod` but works without it, the cause is
almost certainly a missing writable path. See
[`troubleshooting.md`](troubleshooting.md) → "read-only filesystem".

---

## Step 7 — know what you own

```
                    ┌──────────────────────────────────────┐
   Browser ────────►│  frontend  nginx :8080               │
                    │    · serves the built React SPA      │
                    │    · proxies /api/ ──────────┐       │
                    └──────────────────────────────┼───────┘
                                                   │ internal network
                    ┌──────────────────────────────▼───────┐
                    │  api  uvicorn :8000  (non-root)      │
                    │    · /data/product   read-only mount │
                    │    · HTTPS ──► Gemini  (only if a    │
                    │                 key is configured)   │
                    └──────────────────────────────────────┘
```

**One origin.** The browser only ever talks to the frontend. That means no CORS
is exercised, no API host is compiled into the JavaScript bundle, and the Gemini
key never leaves the API container.

**Two images, one Dockerfile.** The backend Dockerfile has two targets:

| Target | Contains | Research tree needed | Default |
|---|---|---|---|
| `api` | FastAPI + DuckDB, no ML libraries | **no** | **yes** |
| `full` | + LightGBM/NumPy/Pandas/PyArrow | yes (two read-only mounts) | opt-in |

The default deployment needs **nothing** from `research/`. Live model
verification is the only feature that does, and it is opt-in via `--inference`.

---

## Step 8 — read these, in this order

1. [`runbook.md`](runbook.md) — day-to-day operation and the release checklist
2. [`troubleshooting.md`](troubleshooting.md) — skim it now, so you recognise symptoms later
3. [`ci-cd.md`](ci-cd.md) — what the pipeline proves and what it cannot
4. [`handover.md`](handover.md) — what is verified, what is not, what to do next

---

## Quick reference

```bash
python tasks.py help              # every command
python tasks.py preflight         # safe to build?
python tasks.py docker-up         # start        (+ --prod, --inference)
python tasks.py smoke             # serving correctly?
python tasks.py docker-ps         # health
python tasks.py docker-logs       # follow logs
python tasks.py docker-down       # stop
python tasks.py verify-all        # full local check suite
```

**Golden rule:** never deploy without `preflight` passing, and never call a
deploy finished without `smoke` passing.
