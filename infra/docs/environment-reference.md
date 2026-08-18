# Environment Reference

Every environment variable this system reads, what it does, and whether you
should touch it.

Source of truth: [`backend/app/config.py`](../../backend/app/config.py).
Backend settings all take an **`NPN_`** prefix; the field name uppercased is the
variable name (`data_dir` → `NPN_DATA_DIR`).

Legend: **Set it** = you will normally configure this · **Rarely** = valid but
uncommon · **Do not** = changing it invalidates the frozen model's contract.

---

## The five you will actually set

| Variable | Value in a container | Why |
|---|---|---|
| `NPN_ENVIRONMENT` | `production` | selects production behaviour |
| `NPN_DATA_DIR` | `/data/product` | where the three data files are mounted |
| `NPN_LOG_LEVEL` | `INFO` | `DEBUG` for diagnosis, never routinely |
| `GEMINI_API_KEY` | *(from a secret store)* | optional AI assistant |
| `API_HOST` *(frontend)* | `api:8000` | where nginx proxies `/api/` |

Everything else has a working default.

---

## Paths

| Variable | Default | Container | Notes |
|---|---|---|---|
| `NPN_DATA_DIR` | `backend/data` | `/data/product` | **Required at runtime.** Must contain `product.duckdb`, `history.parquet`, `backtest.parquet`. Mounted read-only. |
| `NPN_PROJECT_ROOT` | `research` | `/research` | Inference only. Ignored on the lean `api` image. |
| `NPN_MODEL_DIRECT` | under `research/models/champion` | `/research/models/champion/model_11_...txt` | Inference only. |
| `NPN_MODEL_RECURSIVE` | under `research/models/champion` | `/research/models/champion/model_12_...txt` | Inference only. |
| `NPN_FORECAST_CSV` | under `research/predictions/final_forecast` | `/research/predictions/final_forecast/...csv` | Inference only — the shipped forecast that verification compares against. |

Every path is overridable, which is what lets a container set all of them
explicitly and depend on nothing about the checkout layout.

---

## Service

| Variable | Default | Set it? | Notes |
|---|---|---|---|
| `NPN_ENVIRONMENT` | `development` | **Set it** | `development` \| `staging` \| `production` |
| `NPN_LOG_LEVEL` | `INFO` | Rarely | `DEBUG` is verbose; do not leave it on |
| `NPN_SLOW_REQUEST_MS` | `1000` | Rarely | requests slower than this are logged |
| `NPN_API_PREFIX` | `/api/v1` | **Do not** | the frontend and nginx both assume it |
| `NPN_APP_NAME` | `NPN Demand Forecasting API` | Rarely | cosmetic |
| `NPN_VERSION` | `2.0.0` | Rarely | reported by `/meta/*` |

---

## CORS

| Variable | Default | Notes |
|---|---|---|
| `NPN_CORS_ORIGINS` | dev localhost origins | Comma-separated **or** a JSON list. |
| `NPN_CORS_ALLOW_ALL` | `false` | Development escape hatch. **Never enable in production.** |

In the shipped topology CORS is **never exercised** — the browser talks only to
the frontend, and nginx proxies `/api/` internally, so everything is same-origin.
These matter in exactly two cases: frontend development with `npm run dev`, and
a split-origin deployment where the API is exposed directly. The `--prod`
overlay sets `NPN_CORS_ORIGINS` to empty on purpose, so a cross-origin request
in production fails loudly rather than silently matching a localhost entry.

---

## Performance

| Variable | Default | Notes |
|---|---|---|
| `NPN_CACHE_TTL_SECONDS` | `300` | In-process response cache. Per replica, not shared. |
| `NPN_MAX_LIST_LIMIT` | `500` | Cap on list endpoint page size. |
| `NPN_DEFAULT_HISTORY_DAYS` | `90` | Default history window. |
| `NPN_MAX_HISTORY_DAYS` | `1941` | The full history. Raising it does nothing — there is no more data. |

---

## Inference (opt-in)

| Variable | Default | Container | Notes |
|---|---|---|---|
| `NPN_ENABLE_INFERENCE` | `true` | `false` on `api`, `true` on `full` | The lean image has no LightGBM; `/inference/*` returns 503 with a reason. |
| `NPN_INFERENCE_MAX_CONCURRENT` | `1` | | **Leave at 1.** Each run loads the 59.2M-row panel — ~1 GB. |
| `NPN_INFERENCE_TIMEOUT_SECONDS` | `600` | | A run takes ~45 s. |
| `NPN_INFERENCE_JOB_TTL_SECONDS` | `3600` | | How long completed jobs stay queryable. |

Jobs are held **in process**. With multiple replicas, a job started on one is
invisible to the others.

---

## AI assistant (optional)

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | unset | Also accepted as `NPN_GEMINI_API_KEY`. **The only secret in the system.** |
| `NPN_GEMINI_MODEL` | `gemini-3.7-flash` | Google retires model ids without much notice. If the assistant starts returning 404, set `gemini-flash-latest` — an alias that never 404s, at the cost of the model changing under you. |
| `NPN_GENAI_ENABLED` | `true` | `false` disables the assistant even with a key. |
| `NPN_GENAI_TIMEOUT_SECONDS` | `30.0` | |
| `NPN_GENAI_MAX_QUESTION_CHARS` | `800` | |
| `NPN_GENAI_MAX_OUTPUT_TOKENS` | `2048` | Gemini 3.x spends this budget on internal reasoning **before** writing. Lowering it truncates answers mid-sentence. |
| `NPN_GENAI_THINKING_BUDGET` | `0` | `0` disables extended thinking, `-1` lets the model decide. Zero is right here: facts arrive precomputed and the model is forbidden from doing arithmetic. |
| `NPN_GENAI_TEMPERATURE` | `0.2` | |

**Both spellings of the key work**, from a real environment variable or a `.env`
file. `GEMINI_API_KEY` is unprefixed because that is what Google's tooling and
most hosting platforms use.

Which `.env` is read depends on how you start the stack:

| Start method | File read |
|---|---|
| `docker compose up` | `.env` beside `docker-compose.yml` (repo root) |
| `python tasks.py api` | `backend/.env` |

Putting the key in the wrong one is a common and confusing mistake.

Leaving it unset is fully supported: every other feature works and
`/genai/status` explains why the assistant is unavailable. The local guardrails
still function, because they never call Gemini at all.

---

## Domain constants — do not change

| Variable | Value | Why not |
|---|---|---|
| `NPN_N_SERIES` | `30490` | the number of store-item series in M5 |
| `NPN_HORIZON` | `28` | the frozen forecast horizon |
| `NPN_FORECAST_ORIGIN_IDX` | `1940` | zero-based; `d_1941`, the forecast origin |
| `NPN_HISTORY_DAYS_TOTAL` | `1941` | days of history that exist |

These mirror the frozen pipeline. Changing one does not change the model — it
makes the API describe the data incorrectly.

---

## Frontend

| Variable | When | Notes |
|---|---|---|
| `API_HOST` | **runtime** | Where nginx proxies `/api/`. Service name + internal port: `api:8000`. Not `localhost`. |
| `NGINX_ENVSUBST_FILTER` | runtime | Set to `API_HOST` in the Dockerfile. **Do not remove.** Without it `envsubst` substitutes *every* defined variable and blanks nginx's own `$host`, `$uri`, `$remote_addr`. |
| `VITE_API_BASE_URL` | **build** | Deliberately unset. The bundle then defaults to same-origin `/api/v1`. Set it **only** for a split-origin deployment — and remember it is baked into the JavaScript at build time. |
| `VITE_DEV_API_TARGET` | dev | Where `npm run dev` proxies `/api`. `python tasks.py ui <port>` sets it for you. |

**Never put a secret in a `VITE_`-prefixed variable.** Vite inlines them into
the bundle at build time, which publishes them to every user. CI greps `dist/`
for key shapes specifically to catch this.

---

## Container-only

| Variable | Set in | Notes |
|---|---|---|
| `PYTHONPATH=/research` | `full` Dockerfile | so `pipeline` is importable |
| `PORT` | `api` Dockerfile | `8000`; the `CMD` hardcodes the port |
| `PYTHONUNBUFFERED=1` | base | logs are not swallowed by buffering |
| `PYTHONDONTWRITEBYTECODE=1` | base | needed for a read-only root filesystem |

---

## Precedence

For the backend, highest first:

1. real environment variables (what a container and a secret manager use)
2. the `.env` file for the working directory
3. defaults in `config.py`

For the frontend: `VITE_*` are frozen at **build** time and cannot be changed
afterwards; `API_HOST` is applied at **container start** by `envsubst`. That
asymmetry is the reason the API host is not compiled into the bundle — one image
works in every environment.
