# 06_BACKEND — Forecasting API

FastAPI service that serves the **frozen** M5 demand-forecasting model's output.

```
Status: Phase 2 complete — 28 endpoints, 80 tests passing, container-ready
```

Full implementation report: [`08_DOCUMENTATION/BACKEND_IMPLEMENTATION_REPORT.md`](../08_DOCUMENTATION/BACKEND_IMPLEMENTATION_REPORT.md)

---

## Quick start

```bash
python tasks.py build-db     # build the 130 MB data layer (~10 s, one time)
python tasks.py api          # http://localhost:8000  · docs at /docs
python tasks.py test         # 78 fast tests, ~9 s

cd 06_BACKEND && python -m pytest -m slow    # +2 slow tests (~60 s):
                                             # portability proof + live inference
```

### Docker

```bash
python tasks.py build-db                     # produce 06_BACKEND/data/
docker compose up --build                    # http://localhost:8000
```

| Target | Contents | Use |
|---|---|---|
| `api` | fastapi, uvicorn, pydantic, duckdb | lean; `/inference/*` returns 503 with a reason |
| `full` | + lightgbm, numpy, pandas, `pipeline/` | default; live model verification works |

Runs as non-root, mounts research artefacts `:ro`, healthchecks on `/ready`.
**Not yet built** — Docker is unavailable on the development machine; see the
implementation report §10.

`make` equivalents exist in the project `Makefile` for Unix/CI. `tasks.py` works
everywhere, including the Windows machine this is demonstrated on.

---

## What this service is

It answers questions about a **fixed** 28-day forecast produced by a frozen
model. The forecast for `d_1942–d_1969` is not a variable quantity: the model is
frozen and its covariates are published in advance, so there is exactly one
correct answer and it is precomputed. This API makes that answer navigable,
aggregatable and honest about its own accuracy.

**The frozen model** — see `02_MODEL/MODEL_FREEZE.md`:

```
0.60 × Direct(38 features) + 0.40 × Recursive(32 features)
LightGBM Tweedie(1.1), 400 rounds, seed 42
Validation (d_1914–d_1941): RMSE 2.0929 · MAE 1.0395 · 853,720 predictions
```

---

## Architecture

```
routers/   HTTP only — validation, status codes, serialisation
services/  all SQL and all domain logic
db.py      DuckDB access, read-only, identifier whitelisting
cache.py   in-process TTL cache
worker/    background job runner for the ~48 s verification run
services/inference.py  the ONLY module that imports the research pipeline
```

### The API does not import the research pipeline

`pipeline/config.py` calls `mkdir()` at import time. That is a filesystem side
effect on the protected research tree and can raise under a read-only mount. By
keeping that import out of the API:

* the research tree can be mounted strictly read-only;
* the API starts in well under a second rather than waiting ~14 s for the
  59M-row panel to load;
* model code cannot crash the API.

The pipeline is imported LAZILY inside `services/inference.py`, only when a
verification job actually runs. A container built without lightgbm still serves
every other endpoint and reports why inference is unavailable.

### Data layer

Three product-owned files, built once from the frozen artefacts:

| File | Size | Contents |
|---|---|---|
| `data/product.duckdb` | 19.9 MB | series, forecast, calendar, error bands, level accuracy, window metrics, model card |
| `data/history.parquet` | 31.7 MB | 59.2M rows of actuals, sorted by series |
| `data/backtest.parquet` | 78.4 MB | 6.8M rows across 8 backtest windows |

**The API needs nothing else at runtime** — not the research tree, not
`data/raw/`, not the 287 MB research parquet. That is what makes the container
deployable with research artefacts mounted read-only, or absent entirely. It is
proven by `test_api_serves_with_no_research_tree_present`.

Phase 1 queried the research parquet in place. That was replaced because DuckDB
bakes **absolute paths** into views over external files (the database contained a
`C:/Users/...` path), and because the research parquet repeats identifier columns
across all 59.2M rows. The sidecar is 31.7 MB instead of 287 MB and answers a
single-series query in 8–13 ms instead of 320 ms. Full reasoning in
`scripts/build_product_db.py`.

Nothing under `data/`, `models/`, `predictions/`, `experiments/` or `reports/`
is ever opened in write mode.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | liveness |
| GET | `/api/v1/ready` | readiness + table counts; reports *degraded* if a sidecar is missing |
| GET | `/api/v1/meta/model` | frozen model card incl. artefact SHA-256 |
| GET | `/api/v1/meta/capabilities` | **implemented / rejected / not-supported matrix** |
| GET | `/api/v1/meta/provenance` | source and hash of everything served |
| GET | `/api/v1/hierarchy/levels` | the 12 aggregation levels |
| GET | `/api/v1/hierarchy/nodes` | nodes at a level |
| GET | `/api/v1/hierarchy/search` | typeahead over items/stores |
| GET | `/api/v1/hierarchy/aggregate` | coherent roll-up + level-matched accuracy |
| GET | `/api/v1/series` | filtered series listing |
| GET | `/api/v1/series/{store}/{item}` | metadata, volume tier, demand regime |
| GET | `/api/v1/series/{store}/{item}/history` | actual sales, price, events, SNAP |
| GET | `/api/v1/series/{store}/{item}/forecast` | 28-day forecast + empirical error band |
| GET | `/api/v1/accuracy/windows` | the 8 backtest windows that have ground truth |
| GET | `/api/v1/accuracy/levels` | measured accuracy at every aggregation level |
| GET | `/api/v1/accuracy/horizon` | how error grows across the 28 days |
| GET | `/api/v1/accuracy/regimes` | accuracy per Syntetos-Boylan demand regime |
| GET | `/api/v1/accuracy/members` | direct vs recursive vs blend decomposition |
| GET | `/api/v1/accuracy/error-bands` | the empirical band table |
| GET | `/api/v1/accuracy/backtest/{store}/{item}` | predicted vs actual, one series |
| GET | `/api/v1/accuracy/backtest` | predicted vs actual, one hierarchy node |
| GET | `/api/v1/insights/summary` | headline planning numbers for a node |
| GET | `/api/v1/insights/top-movers` | series rising/falling vs recent run-rate |
| GET | `/api/v1/insights/planning/{store}/{item}` | 28-day planning view with margin |
| GET | `/api/v1/inference/status` | can live inference run here, and what it refuses |
| POST | `/api/v1/inference/verify` | re-run the frozen model and verify (~48 s) |
| GET | `/api/v1/inference/jobs` · `/jobs/{id}` | job list and polling |

Interactive documentation at `/docs`; schema at `06_BACKEND/openapi.json`
(`python tasks.py openapi`).

## Live model serving

`POST /inference/verify` reloads the frozen boosters, rebuilds features from the
raw panel, re-runs both members, blends at w=0.60 and compares the result against
the shipped artefact.

**Measured: `max_abs_diff = 0.000e+00` across all 853,720 predictions** — the
live model reproduces the shipped forecast bit-for-bit.

The two boosters are cached for the process lifetime, so no request ever reloads
a model binary. The 800 MB panel is loaded per job and released, keeping idle
RSS near 200 MB.

Inference **refuses** to run at an origin before `d_1941` (the boosters were
trained to that origin — earlier inference would be leakage), to retrain, or to
run covariate what-if scenarios.

---

## Two contracts the API will not break

### 1. Accuracy is always level-matched

The same forecast is ~28% accurate per store-item and ~97% chain-wide. There is
no global accuracy number in this API, because publishing one would guarantee it
eventually appears next to the wrong view. `/hierarchy/aggregate` returns the
accuracy **measured for the level being requested**.

### 2. Error bands are measurements, not model output

The frozen model emits point forecasts only. `lower`/`upper` are the empirical
p05–p95 of `(actual − predicted)` observed on 8 held-out backtest windows,
grouped by demand regime and horizon, rescaled by `sqrt(forecast)`.

The `sqrt` scaling is not cosmetic. Pooling raw residuals by volume tier — the
first approach — was measured to be invalid: inside the single "high" tier the
residual standard deviation ranges from 3.3 to 21.6 depending on series size, so
the band was far too narrow for large series and far too wide for small ones.
Normalising by `sqrt(max(ŷ, 1))` collapses that spread to ~1.4×, and the exponent
matches the model's own Tweedie variance power (1.1 ⇒ sd ∝ μ^0.55).

**Measured coverage of the resulting p05–p95 band: 90.0%**, and 89.9–90.1% in
every regime and at every horizon. Calibration is in-sample to the backtest
windows; with 6.8M observations across 140 cells, overfitting is negligible.

Every response carries `band_basis` stating in words that this is observed error
and not a model-produced interval.

---

## Testing

```bash
python tasks.py test        # 45 tests
```

| File | Covers |
|---|---|
| `test_health.py` | liveness, readiness, request-id contract |
| `test_meta.py` | model card values, capability matrix, **guards that price what-if stays declared unsupported** |
| `test_hierarchy.py` | levels, nodes, search, **coherence** (children sum to parent), level-matched accuracy, injection rejection |
| `test_series.py` | history ordering, forecast window, band bracketing/scaling, band-basis wording |
| `test_accuracy.py` | 8 windows, published metrics, **no window without ground truth**, member decomposition |
| `test_insights.py` | totals agree across endpoints, movers ranking, planning caveats |
| `test_inference.py` | availability, refusals, job lifecycle, **live verification MATCH** |
| `test_deployment.py` | env config, no Windows paths, no baked paths, no eager ML imports, deps declared, 503 degradation, **portability proof** |
| `test_integrity.py` | **freeze regression guard** — see below |

### The freeze regression guard

`test_integrity.py` is the reason a model swap cannot happen silently:

* model file SHA-256 must match `CHAMPION_MANIFEST.json`;
* the hashes the API advertises must match the files on disk;
* served forecasts must equal the frozen CSV row-for-row;
* chain-wide total must equal the sum of the frozen artefact;
* cached backtest must still reproduce **RMSE 2.0929 / MAE 1.0395**;
* `p_blend` must still equal `0.60·direct + 0.40·recursive`;
* error bands must still cover ~90%.

---

## Performance (measured on the development machine)

| Operation | Time |
|---|---|
| Series forecast | ~5 ms |
| Aggregate (cached) | ~7 ms |
| Aggregate + 30 days history | ~445 ms first call |
| Series history, 60 days | **~9 ms** (was 320 ms in Phase 1) |
| Accuracy windows / members | 17–22 ms |
| Regime accuracy (6.8M rows) | ~393 ms |
| Live verification | ~48 s |
| API startup | < 1 s |
| Fast test suite | ~9 s |
| Data layer build | 9.8 s |

Idle RSS ~200 MB; inference peak ~1 GB.

---

## Correction to an earlier note

An earlier version of this file said running the recursive member "takes
minutes". That was wrong — it conflated **training** (416 s) with **inference**.
Measured from the saved boosters: direct member 4.1 s for all 853,720 rows,
recursive rollout ~29 s, so a **complete blend re-forecast is ~33 s**. Live
inference is therefore viable and is planned for Phase 5 as a verification
endpoint.

---

## What this service will never do

* retrain, re-tune or re-blend the model;
* run the shipped boosters at an origin earlier than `d_1941` (they were trained
  to that origin — using them earlier would be leakage);
* offer price what-if simulation. The frozen model's measured response to
  simulated price changes is non-monotone and sometimes economically backwards
  (a 10% cut predicted −74% demand on one high-volume series). It is a
  forecaster that uses price as context, not a causal elasticity model.
  See `08_DOCUMENTATION/PRODUCT_ARCHITECTURE_PLAN.md` §14.3.
