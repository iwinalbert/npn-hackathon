# BACKEND IMPLEMENTATION REPORT — Phase 2

**NPN_HACKATHON — Cognizant Use Case 11**
**Scope:** production-oriented FastAPI backend over the frozen forecasting model
**Status:** complete — 28 endpoints, 80 tests passing, container-ready

---

## 1. What was built

A deployable REST API that serves the frozen champion's forecasts, its measured
accuracy, and a live model-verification path — without the research tree being
present at runtime.

```
FROZEN MODEL  ─┐
               ├─→ build_product_db.py ─→ 130 MB portable data layer ─→ FastAPI ─→ JSON
RESEARCH DATA ─┘        (offline, read-only)
```

| Layer | Implementation |
|---|---|
| Data | DuckDB + 2 sorted parquet sidecars, all product-owned |
| API | FastAPI, layered routers → services → db |
| Model serving | Lazy-loaded frozen boosters behind an inference service boundary |
| Jobs | In-process thread pool for the ~48 s verification run |
| Packaging | Multi-target Dockerfile (`api` / `full`), compose, split requirements |

---

## 2. The one architecture change, and why

**The plan said:** query `data/processed/sales_long_full.parquet` in place —
"duplicating 287 MB of history would be waste."

**Inspection showed that is a hard Linux blocker.** DuckDB bakes an *absolute
path* into any view over an external file. The Phase-1 database contained:

```sql
CREATE VIEW panel AS SELECT * FROM
  read_parquet('C:/Users/Rishi/OneDrive/Desktop/NPN_HACKATHON/data/processed/sales_long_full.parquet');
```

That view is dead the moment the file is opened anywhere else. Separately, the
research parquet is *denormalised*: `item_id`, `store_id`, event names and SNAP
flags repeat on every one of 59.2M rows.

**Measured alternatives:**

| Approach | Size | Single-series history query | Portable |
|---|---|---|---|
| Research parquet in place | 287 MB external | 320 ms | ✗ absolute path |
| DuckDB table + index | 1,066 MB | 2 ms | ✓ |
| **Sorted product sidecar** | **31.7 MB** | **8–13 ms** | **✓** |

**Decision: product-owned sorted parquet sidecars.** Smaller, ~25× faster, and
portable. Keeping only `(series_idx, day_idx, sales, sell_price)` and sorting by
series lets DuckDB's zone maps prune nearly everything.

**Consequence:** the API needs **no access to the research tree at all**. That is
what makes the container deployable with research artefacts mounted read-only —
or not mounted.

| Artefact | Before | After |
|---|---|---|
| product.duckdb | 295 MB | 19.9 MB |
| history | 287 MB external | 31.7 MB sidecar |
| backtest | in DuckDB | 78.4 MB sidecar |
| **Total** | **582 MB** | **130 MB** |

---

## 3. Model-serving strategy

Unchanged from the plan: **two tiers**.

### Tier 1 — precomputed (every user-facing request)
The forecast for `d_1942–d_1969` is a *fixed* quantity: the model is frozen and
its covariates are published, so exactly one correct answer exists and it is
already computed. Served from DuckDB in ~5 ms.

### Tier 2 — live verification (opt-in, ~48 s)
`POST /inference/verify` reloads the frozen boosters, rebuilds features from the
raw panel via the frozen `pipeline` package, re-runs both members, blends at
w=0.60 and compares against the shipped artefact.

**Measured result: `max_abs_diff = 0.000e+00` across all 853,720 predictions.**
The live model reproduces the shipped forecast bit-for-bit.

```
[  0.0s]  2%  loading frozen boosters
[  2.0s] 10%  loading sales panel (~14 s)
[ 15.8s] 25%  running direct member (38 features)
[ 19.8s] 40%  running recursive member (28-step rollout)
[ 47.9s] done  VERDICT: MATCH
timings: data 13.95s · direct 4.31s · recursive 28.38s · total 47.29s
```

### Model loading efficiency
The two boosters (~29 MB, 0.055 s) are cached for the process lifetime under a
double-checked lock — **no request ever reloads a model binary**. The 800 MB
`M5Data` panel is deliberately *not* cached: it is loaded per job and released,
keeping idle RSS near 200 MB. A container idling at 1 GB to save 14 s on a rarely
used endpoint is the wrong shape.

### What inference refuses to do
| Refused | Reason |
|---|---|
| Inference at an origin before `d_1941` | Both boosters were trained with data up to `d_1941`; running them earlier is leakage. Historical accuracy comes from the cached per-window artefacts instead |
| Retraining / re-blending | The model is frozen |
| Price or covariate what-if | Measured price response is non-monotone and not causal (plan §14.3) |

The service also asserts the panel ends exactly at the frozen origin before
running, and refuses otherwise.

---

## 4. API endpoints (28)

Base path `/api/v1`. Interactive docs at `/docs`, schema at `/openapi.json`.

### Health
| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness — process is up |
| `GET /ready` | readiness — artefacts present, tables queryable, sidecars readable, cache stats |

### Meta
| Endpoint | Purpose |
|---|---|
| `GET /meta/model` | frozen model card incl. artefact SHA-256 |
| `GET /meta/capabilities` | implemented / rejected / not-supported matrix |
| `GET /meta/provenance` | source + hash of everything served |

### Hierarchy
| Endpoint | Purpose |
|---|---|
| `GET /hierarchy/levels` | the 12 aggregation levels |
| `GET /hierarchy/nodes` | nodes at a level, optionally under a parent |
| `GET /hierarchy/search` | typeahead over items/stores |
| `GET /hierarchy/aggregate` | coherent roll-up + level-matched accuracy + optional history |

### Series
| Endpoint | Purpose |
|---|---|
| `GET /series` | filtered listing |
| `GET /series/{store}/{item}` | metadata, volume tier, demand regime |
| `GET /series/{store}/{item}/history` | actuals, price, events, SNAP |
| `GET /series/{store}/{item}/forecast` | 28-day forecast + empirical error band |

### Accuracy (verified artefacts only)
| Endpoint | Purpose |
|---|---|
| `GET /accuracy/windows` | the 8 backtest windows with ground truth |
| `GET /accuracy/levels` | measured accuracy at every aggregation level |
| `GET /accuracy/horizon` | how error grows across 28 days |
| `GET /accuracy/regimes` | accuracy per Syntetos-Boylan regime |
| `GET /accuracy/members` | direct vs recursive vs blend decomposition |
| `GET /accuracy/error-bands` | the empirical band table |
| `GET /accuracy/backtest/{store}/{item}` | predicted vs actual, one series |
| `GET /accuracy/backtest` | predicted vs actual, one hierarchy node |

### Insights
| Endpoint | Purpose |
|---|---|
| `GET /insights/summary` | headline planning numbers for a node |
| `GET /insights/top-movers` | series rising/falling vs recent run-rate |
| `GET /insights/planning/{store}/{item}` | 28-day planning view with margin |

### Inference
| Endpoint | Purpose |
|---|---|
| `GET /inference/status` | availability, refusals, runtime, job stats |
| `POST /inference/verify` | start a verification run → job id |
| `GET /inference/jobs` | recent jobs |
| `GET /inference/jobs/{id}` | poll one job |

### Examples

```bash
curl localhost:8000/api/v1/series/CA_3/FOODS_3_090/forecast
```
```json
{
  "series": {"item_id": "FOODS_3_090", "store_id": "CA_3",
             "regime": "smooth", "volume_tier": "high"},
  "origin_day": "d_1941", "origin_date": "2016-05-22",
  "forecast": [
    {"date": "2016-05-23", "horizon": 1, "yhat": 96.2665,
     "lower": 79.1, "upper": 121.98}
  ],
  "total_28d": 3331.3683,
  "band_basis": "Empirical p05-p95 of (actual - predicted) ... This is OBSERVED
                 MODEL ERROR, not a model-produced prediction interval ...",
  "band_regime": "smooth"
}
```

```bash
curl "localhost:8000/api/v1/hierarchy/aggregate?level=store&node_id=CA_3"
```
```json
{
  "level": "store", "node_id": "CA_3", "n_series": 3049,
  "total_28d": 189101.2,
  "expected_accuracy": {"measured_level": "L3_store", "wape": 0.053,
                        "accuracy_pct": 94.7,
                        "basis": "Measured on the held-out validation window ..."},
  "coherence_note": "Aggregates are exact sums of the bottom-level store-item
                     forecasts, so the hierarchy is coherent by construction."
}
```

Error shape is uniform:
```json
{"error": "not_found", "message": "no series for store 'ZZ_9' and item 'NOPE'",
 "context": {"hint": "Use /hierarchy/search to find valid identifiers."},
 "request_id": "a1b2c3d4e5f6"}
```

---

## 5. Configuration

All settings are environment variables with the `NPN_` prefix; every path is
overridable and nothing depends on the checkout layout.

| Variable | Default | Purpose |
|---|---|---|
| `NPN_DATA_DIR` | `backend/data` | the three product artefacts |
| `NPN_PROJECT_ROOT` | repo root | research tree (inference only) |
| `NPN_MODEL_DIRECT` / `NPN_MODEL_RECURSIVE` | `models/champion/...` | frozen boosters |
| `NPN_FORECAST_CSV` | `predictions/final_forecast/...` | artefact to verify against |
| `NPN_ENVIRONMENT` | `development` | `production` disables the CORS escape hatch |
| `NPN_CORS_ORIGINS` | localhost dev ports | comma-separated **or** JSON list |
| `NPN_ENABLE_INFERENCE` | `true` | set `false` for a lean API-only container |
| `NPN_LOG_LEVEL` | `INFO` | |
| `NPN_CACHE_TTL_SECONDS` | `300` | |
| `NPN_INFERENCE_MAX_CONCURRENT` | `1` | |

---

## 6. Deployment

### Build the data layer once (host or image build)
```bash
python tasks.py build-db        # ~10 s → 130 MB
```

### Docker
```bash
docker build -f backend/Dockerfile --target full -t npn-api .
docker compose up
```

| Target | Contents | Use |
|---|---|---|
| `api` | fastapi + uvicorn + pydantic + duckdb | lean; `/inference/*` returns 503 with a reason |
| `full` | + lightgbm, numpy, pandas, `pipeline/` | default; live verification works |

**Deployment properties:**
- runs as **non-root** (`uid 10001`), so mounted research artefacts cannot be modified;
- research volumes mounted **`:ro`**;
- `HEALTHCHECK` polls `/ready` (proves the data layer is queryable), not `/health`;
- single uvicorn worker by design — DuckDB opens read-only per process and each
  worker would add a full model+panel copy during inference; scale with replicas;
- `libgomp1` installed in the `full` target (LightGBM will not import without it);
- 1.7 GB of research data is **mounted, never `COPY`'d** — see `.dockerignore`;
- memory limit 2 GB: idle ~200 MB, inference peak ~1 GB.

### Startup contract
Startup **never fails** on a missing artefact. It logs the problem and serves
anyway so `/ready` can explain what is wrong. A container that refuses to boot
tells an operator less than one that boots and reports.

---

## 7. Testing

```bash
python tasks.py test          # 78 fast tests, ~9 s
cd 06_BACKEND && python -m pytest -m slow   # 2 slow tests, ~60 s
```

**80 tests, all passing.**

| File | n | Covers |
|---|---|---|
| `test_health.py` | 4 | liveness, readiness shape, request-id contract |
| `test_meta.py` | 5 | model-card values, capability matrix, price-what-if guard |
| `test_hierarchy.py` | 16 | levels, nodes, search, **coherence**, level-matched accuracy, injection rejection |
| `test_series.py` | 10 | history ordering, forecast window, band bracketing/scaling, band wording |
| `test_accuracy.py` | 12 | 8 windows, published metrics, **no window without ground truth**, member decomposition |
| `test_insights.py` | 9 | totals consistent across endpoints, movers ranking, planning caveats |
| `test_inference.py` | 5 | availability, refusals, job lifecycle, **live verification MATCH** (slow) |
| `test_deployment.py` | 9 | env config, no Windows paths, no baked paths, no eager ML imports, deps declared, 503 degradation, **portability proof** (slow) |
| `test_integrity.py` | 10 | **freeze regression guard** |

### The two tests that matter most

**`test_api_serves_with_no_research_tree_present`** — copies only the three
product artefacts to a temp dir, points `NPN_PROJECT_ROOT` at an empty directory,
and runs the API in a clean subprocess. Result: readiness true, 12 levels, 28
forecast points, chain total > 1M, 8 accuracy windows, 30 history points — and
inference correctly reports unavailable with reasons. **This is the deployment
proof.**

**`test_verification_reproduces_the_frozen_forecast`** — full live inference,
`verdict == MATCH`, `max_abs_diff == 0.0`.

### Freeze regression guard
Model SHA-256 must match `CHAMPION_MANIFEST.json`; the hashes the API advertises
must match the files on disk; served forecasts must equal the frozen CSV
row-for-row; the chain total must equal the artefact sum; the backtest must still
reproduce RMSE 2.0929 / MAE 1.0395; `p_blend` must still equal
`0.60·direct + 0.40·recursive`; bands must still cover ~90%.

---

## 8. Security

| Concern | Treatment |
|---|---|
| SQL injection | Level names whitelisted against a module constant; all values bound parameters; **no f-string SQL with request data**. A test fires `x'; DROP TABLE series; --` at the level parameter and asserts 400 + table intact |
| CORS | Explicit allow-list; `cors_allow_all` ignored when `environment=production` |
| Write protection | DuckDB opened `read_only=True`; parquet read-only; container non-root; research volumes `:ro` |
| Error leakage | Tracebacks never returned; request id correlates response to logs |
| DoS | Hard limits on all list endpoints; inference capped at 1 concurrent job |
| Auth | **Not implemented** — read-only public data, no PII, no mutations. Documented as deliberate; the dependency hook is left in place |

---

## 9. Performance (measured)

| Operation | Time |
|---|---|
| Series forecast | ~5 ms |
| Series detail | ~4 ms |
| Aggregate (cached) | ~7 ms |
| Aggregate + 30d history | ~445 ms cold, cached after |
| Series history 60d | ~9 ms (was 320 ms via research parquet) |
| Accuracy windows / members | 17–22 ms |
| Top movers (3,049 series) | ~17 ms |
| Regime accuracy (6.8M rows) | ~393 ms |
| Live verification | ~48 s |
| API startup | < 1 s |
| Data layer build | 9.8 s |
| Fast test suite | ~9 s |

Idle RSS ~200 MB; inference peak ~1 GB.

---

## 10. Known limitations

1. **Docker was not built or run.** Docker is unavailable on the development
   machine. The Dockerfile and compose file are written to spec and statically
   validated by tests (non-root, healthcheck, `:ro` mounts, targets), but **an
   actual image build has not been executed.** First build should happen early
   on a Linux host — the pins (`pandas 3.0.5`, `numpy 2.5.1`, Python 3.13) are
   recent and are the most likely source of a build-time surprise.
2. **Jobs are in-process.** They do not survive a restart and are not shared
   across replicas. Fine for one verification endpoint at demo scale; replace
   with a broker if the API is ever scaled horizontally. `/inference/status`
   states this in its response rather than hiding it.
3. **Single worker.** Multiple uvicorn workers would each hold a DuckDB handle
   and, during inference, a full model + panel copy. Scale with replicas.
4. **Error-band calibration is in-sample** to the backtest windows the quantiles
   were fitted on. With 6.8M observations across 140 cells this is negligible,
   but it is not an out-of-sample coverage guarantee.
5. **Inference needs the research `pipeline` package**, which calls `mkdir()` at
   import. The compose file mounts writable scratch volumes for the two
   directories it touches. If that proves awkward in a target environment, the
   `api` target avoids the issue entirely.
6. **No authentication.** Deliberate — see §8.
7. **`/accuracy/regimes` is the slowest endpoint** (~393 ms) because it scans
   6.8M backtest rows grouped by regime. Cached after first call; could be
   precomputed into the build if it matters.

---

## 11. What was NOT touched

The frozen research layer. Verified after implementation:

```
files before : 520      MISSING (deleted) : 0
files after  : 522      CHANGED (rewritten): 0
raw-data MD5 : 1bce33213f53e8d7b5136bec2f4e67bc (unchanged)
-> PASS: no protected artefact was deleted or modified
```

The two added files under a protected root are the Phase-1 organisation scripts
(`62_experiment_classification.py`, `63_verify_paths.py`), not research artefacts.

Everything the backend wrote went to `backend/` and `docs/`.
