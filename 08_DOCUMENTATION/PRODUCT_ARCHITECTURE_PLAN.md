# PRODUCT ARCHITECTURE & IMPLEMENTATION PLAN

**NPN_HACKATHON — Cognizant Use Case 11**
**Phase:** product development around the frozen forecasting model
**Status:** proposal for review — no implementation started

> **Use Case 11.** *"Build a forecasting model that handles hierarchical
> aggregation, external covariates (price/promo/holiday) and intermittent demand,
> producing accurate 28-day-ahead forecasts per store/item."*

> **Addendum — 2026-08. This document is the pre-implementation proposal and is
> kept as written.** Everything in it was built. One component was added
> afterwards and is therefore *not* described below: an **AI Forecast Assistant**
> (a Gemini-backed explanatory layer over the same read-only API — 4 endpoints
> under `/genai`, one React page, no write path to the model). Its architecture,
> key handling and guardrails are documented in
> [`GENAI_IMPLEMENTATION_REPORT.md`](GENAI_IMPLEMENTATION_REPORT.md). It changes
> nothing about §2 (the inference boundary), §9 (data storage) or §28 (what must
> remain untouched); it consumes the API described in §10 rather than extending it.

---

# 1. Current-state assessment

## 1.1 What exists and works

| Layer | State |
|---|---|
| Frozen model | `model_11` (direct 38f) + `model_12` (recursive 32f), LightGBM Tweedie(1.1), blend w=0.60 |
| Validated performance | RMSE **2.0929** / MAE **1.0395** on 853,720 held-out predictions; reproduced exactly on 4 windows during the Stage 7 audit |
| Shipped forecast | 30,490 × F1..F28 for `d_1942–d_1969`, structurally validated |
| Research pipeline | 22 modules, 61 scripts, 86 experiment records, 27 reports — all path-locked and reproducible |
| Backtest artifacts | **8 cached windows** in `predictions/uc11_cache/` with `y_true`, `p_direct`, `p_recursive`, `p_blend` per row |
| Measured hierarchy accuracy | all 12 M5 levels, `experiments/artifacts/uc11_hierarchy_levels.csv` |
| Organisation layer | `01_DATA` … `99_ARCHIVE`, integrity-verified |

## 1.2 Measurements I took before designing

Everything below was measured on this machine, not assumed. **These numbers drive
the architecture.**

| Operation | Cost |
|---|---|
| Load both frozen boosters | **0.99 s** |
| Load `M5Data` (sales + price matrices) | **13.7 s** |
| `FeatureBuilderV5` precompute | 0.24 s |
| **Direct member, full panel** (853,720 rows) | **4.13 s** (0.77 build + 3.36 predict) |
| Direct member, single series | 0.66 s |
| **Recursive member, per rollout step** | **1.02 s** |
| **Recursive member, full 28-step rollout** | **~29 s** |
| **Full blend re-forecast, all 30,490 series** | **≈ 33 s** |
| Inference service peak RSS | **~800 MB** |
| DuckDB query on the 59.2M-row parquet | **0.10 – 0.16 s** |

**I must correct an estimate I gave during the organisation phase.** I stated in
`06_BACKEND/README.md` that running the recursive member "takes minutes". That
conflated *training* (416 s) with *inference*. Pure inference from the saved
boosters is ~29 s. **Live re-forecasting is viable**, which materially widens the
design space. That README will be corrected during implementation.

## 1.3 Constraints created by the current architecture

| Constraint | Consequence |
|---|---|
| `pipeline/config.py` derives all paths from `parent.parent` | The research tree cannot move; the product must live alongside it |
| `pipeline/config.py` calls `mkdir()` **at import time** (lines 64–66) | Importing the research pipeline has a filesystem side effect. In a read-only container mount this can raise `EROFS`. **Design consequence: the API process must not import the research pipeline** |
| `FeatureBuilderV4/V5` hard-code `config.N_SERIES` in shape/cycle profiles | Feature building is inherently full-panel; per-series inference still pays the full-panel precompute (~0.65 s floor) |
| Boosters were trained at origin `d_1941` | Using them at an **earlier** origin is leakage. Backtest views must use the cached per-window artifacts, never the shipped boosters |
| `requirements.txt` omits `duckdb`, `fastapi`, `uvicorn` (all installed) | Must be pinned before deployment |
| Project is **not under version control** (0 tracked files) | Flagged in `ORGANIZATION_AUDIT.md` §8.2; product code needs a repo |

## 1.4 Existing scaffolds — my assessment

`06_BACKEND/` and `07_FRONTEND/` contain only a README and `.gitkeep`. There is
nothing to keep or discard; they are directory reservations. **I will use them as
the roots for new code**, which is the right call — they sit outside the
path-locked research tree, so nothing I build can disturb reproducibility.

---

# 2. Final ML model and the inference boundary

```
ŷ(s,h) = clip( 0.60 · Direct₃₈(s,h) + 0.40 · Recursive₃₂(s,h) , 0, ∞ )
```

**The inference boundary is deliberately narrow.** The product may:

- read the frozen boosters and run them **forward from `d_1941`** with the real,
  published covariates;
- read precomputed forecasts and cached backtest artifacts;
- aggregate, slice, compare and visualise those numbers.

The product may **not**:

- retrain, re-tune, re-blend or re-seed anything;
- run the shipped boosters at an origin earlier than `d_1941` (leakage);
- perturb model inputs off-distribution and present the result as a forecast
  (see §14.3 — I tested this and it fails badly).

---

# 3. Proposed system architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  BROWSER — React + TypeScript + Vite SPA                         │
│  Explorer · Forecast · Accuracy · Hierarchy · Model Card         │
└───────────────────────────┬──────────────────────────────────────┘
                            │ REST/JSON
┌───────────────────────────▼──────────────────────────────────────┐
│  API — FastAPI (uvicorn)                                         │
│  routers · schemas · service layer · OpenAPI docs                │
│  ── does NOT import the research pipeline ──                     │
└──────────┬────────────────────────────────┬──────────────────────┘
           │ SQL (read-only)                │ job submit / poll
┌──────────▼─────────────────┐   ┌──────────▼──────────────────────┐
│  DATA — DuckDB             │   │  INFERENCE WORKER (separate     │
│  • product.duckdb (built)  │   │  process, imports pipeline)     │
│  • ATTACH parquet directly │   │  • loads model_11 + model_12    │
│    (read-only)             │   │  • forward re-forecast @ d_1941 │
└──────────┬─────────────────┘   │  • ~33 s per full run           │
           │                     └──────────┬──────────────────────┘
┌──────────▼────────────────────────────────▼──────────────────────┐
│  PROTECTED RESEARCH LAYER — READ-ONLY, NEVER WRITTEN             │
│  data/ · models/ · predictions/ · experiments/ · reports/        │
└──────────────────────────────────────────────────────────────────┘
```

**The load-bearing decision: the API never imports the research pipeline.** It
serves from DuckDB only. Live inference is delegated to a separate worker process
that does import it. This buys:

- the API starts in <1 s instead of ~15 s;
- the API has no filesystem side effects on the research tree (§1.3);
- a slow or crashed inference run cannot take down the API;
- the API can be containerised without LightGBM or the 1.78 GB research tree.

---

# 4. Component responsibilities

| Component | Owns | Explicitly does not |
|---|---|---|
| **React SPA** | Presentation, navigation, chart interaction, client state | Any aggregation logic, any accuracy computation |
| **FastAPI app** | HTTP, validation, serialisation, error mapping, OpenAPI | Model loading, feature building |
| **Service layer** | Query composition, aggregation, error-band lookup, caching | SQL string building in routers, chart formatting |
| **DuckDB** | All slicing/aggregation over history, forecast and backtest | Storing anything the research layer owns |
| **Build script** | One-time materialisation of `product.duckdb` from protected artefacts | Writing anywhere outside `06_BACKEND/data/` |
| **Inference worker** | Loading frozen boosters, forward re-forecast, verification | Retraining, weight changes, earlier-origin inference |
| **Research layer** | Ground truth for everything | Being modified by the product in any way |

---

# 5. Data flow

**Read path (default, every user interaction):**

```
browser → GET /api/... → service → DuckDB → (parquet + product.duckdb) → JSON → chart
```
Typical latency: **< 200 ms**.

**Verification path (on demand, opt-in):**

```
browser → POST /api/inference/verify → job queued → worker loads frozen boosters
       → direct (4 s) + recursive rollout (29 s) → compare against shipped artefact
       → job result → browser polls → verdict + max abs diff
```
Typical latency: **~35 s**, shown with progress.

**Build path (once, at setup):**

```
protected artefacts (read-only) → 06_BACKEND/scripts/build_product_db.py
                                → 06_BACKEND/data/product.duckdb
```

---

# 6. Backend design

**Framework: FastAPI + uvicorn** — already installed; native Pydantic validation;
auto-generated OpenAPI/Swagger, which is direct evidence for the *deployment and
integration* criterion; async I/O suits a poll-based job endpoint.

```
06_BACKEND/
├── app/
│   ├── main.py                 FastAPI app, CORS, exception handlers, lifespan
│   ├── config.py               settings via pydantic-settings + .env
│   ├── db.py                   DuckDB connection pool, read-only guards
│   ├── cache.py                in-process TTL/LRU cache
│   ├── routers/
│   │   ├── health.py           liveness/readiness
│   │   ├── meta.py             model card, capability matrix, data provenance
│   │   ├── hierarchy.py        levels, nodes, children, search
│   │   ├── series.py           per store-item history + forecast
│   │   ├── forecast.py         aggregated forecast at any hierarchy node
│   │   ├── accuracy.py         per-level accuracy, backtest windows, error bands
│   │   ├── insights.py         planning summaries, regime classification
│   │   └── inference.py        verification job submit/poll
│   ├── services/               one module per router, all SQL lives here
│   ├── schemas/                Pydantic response models
│   └── worker/
│       ├── runner.py           subprocess job manager
│       └── verify_forecast.py  the ONLY module importing `pipeline`
├── scripts/build_product_db.py
├── data/product.duckdb         generated, git-ignored
├── tests/
├── requirements.txt
└── Dockerfile
```

**Why not Flask/Django:** Flask lacks native validation and OpenAPI; Django is far
too heavy for a read-mostly analytical API.

---

# 7. Frontend design

**Framework: React 18 + TypeScript + Vite.** *UI/UX* is an explicit scoring
category and the brief states the result must not look like a notebook.

**Considered and rejected: Streamlit.** It would be ~2 days faster, but it looks
like a data-science tool, caps layout control, and couples UI to Python. Given
UI/UX is scored directly, that ceiling is the wrong trade. *(If the timeline
compresses hard, Streamlit is the documented fallback — it can reach ~70% of the
demo value at ~40% of the effort.)*

- **Styling:** Tailwind CSS — fast, consistent, no bespoke design system needed.
- **Charts:** Recharts — declarative, composable, good defaults. ECharts is the
  fallback if the 1,941-point history views need canvas rendering.
- **State/data:** TanStack Query — caching, background refetch and request
  dedup for free; no Redux needed for a read-mostly app.
- **Routing:** React Router.

```
07_FRONTEND/
├── src/
│   ├── api/            typed client generated from OpenAPI
│   ├── components/     charts/, layout/, ui/
│   ├── pages/          Dashboard, Explorer, SeriesDetail, Hierarchy,
│   │                   Accuracy, ModelCard
│   ├── hooks/
│   └── lib/            formatting, regime labels, colour scales
├── index.html
├── vite.config.ts
└── Dockerfile
```

---

# 8. Model-serving / inference strategy

**Two tiers, chosen because of the measurements in §1.2.**

### Tier 1 — precomputed serving (default)
Every user-facing forecast comes from the frozen artefact via DuckDB. Sub-200 ms,
deterministic, zero risk. This is correct, not a shortcut: the forecast for
`d_1942–d_1969` is a *fixed* quantity — the model is frozen, the inputs are
published, so there is exactly one right answer and it is already computed.

### Tier 2 — live verification (on demand)
A `POST /api/inference/verify` job loads the frozen boosters, rebuilds features
from the raw panel, runs both members, blends at w=0.60, and **compares the result
against the shipped artefact**, returning max absolute deviation and a pass/fail.

This is worth building because it proves, live and on stage:
1. the model files load and run;
2. the feature pipeline is intact;
3. the shipped forecast is authentic and reproducible;
4. the system has a genuine model-serving path, not just a CSV reader.

It is honest — it makes no claim beyond reproduction — and it takes ~33 s, which
is a reasonable "watch it run" demo moment.

**Rejected: request-time re-forecasting for arbitrary user inputs.** See §14.3.

---

# 9. Data-storage strategy

**DuckDB, in two parts, both read-only from the API's perspective.**

1. **Direct parquet attachment.** DuckDB queries
   `data/processed/sales_long_full.parquet` (59.2M rows) *in place* at 0.10–0.16 s
   with predicate pushdown. No ETL, no duplication, no modification of protected
   data.
2. **`06_BACKEND/data/product.duckdb`** — a small generated database holding only
   what does not already exist in queryable form:

| Table | Rows | Source |
|---|---|---|
| `forecast` | 853,720 | shipped forecast, unpivoted to (series, day, ŷ) |
| `series_meta` | 30,490 | hierarchy + volume tier + Syntetos-Boylan regime |
| `backtest` | ~6.8M | the 8 cached windows, with `p_direct`/`p_recursive`/`p_blend`/`y_true` |
| `error_bands` | ~280 | residual quantiles by (volume tier × horizon) |
| `level_accuracy` | 12 | measured accuracy per M5 level (Stage 7) |
| `model_card` | 1 | frozen-model metadata + hashes |

**Why not Postgres:** it would need a server, a container, a migration story and
an ETL of 59M rows — all cost, no benefit for a read-only analytical workload.
**Why not raw pandas in memory:** ~4 GB resident and slow group-bys; DuckDB does
it out-of-core in milliseconds.

**Caching:** in-process TTL cache on hot aggregates (hierarchy roll-ups, level
accuracy). No Redis — it adds a deployment dependency for a workload DuckDB
already answers in ~100 ms.

---

# 10. API design

Versioned under `/api/v1`. All responses Pydantic-typed; all errors RFC-7807-style.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/ready` | liveness; readiness includes DB + artefact checks |
| GET | `/meta/model` | frozen model card: architecture, metrics, hashes, freeze status |
| GET | `/meta/capabilities` | **the honesty matrix** — implemented / rejected / future (§14) |
| GET | `/meta/provenance` | artefact paths, SHA-256, row counts, build timestamp |
| GET | `/hierarchy/levels` | the 12 M5 levels with node counts |
| GET | `/hierarchy/nodes?level=&parent=` | children of a node |
| GET | `/hierarchy/search?q=` | typeahead over items/stores |
| GET | `/series/{store_id}/{item_id}` | metadata, regime, volume tier, price |
| GET | `/series/{store_id}/{item_id}/history?from=&to=` | actuals + price + events |
| GET | `/series/{store_id}/{item_id}/forecast` | 28-day forecast + empirical band |
| GET | `/forecast/aggregate?level=&node=` | coherent roll-up + history |
| GET | `/accuracy/levels` | measured accuracy at each of the 12 levels |
| GET | `/accuracy/backtest?origin=&node=` | predicted vs actual, one cached window |
| GET | `/accuracy/members?origin=` | direct vs recursive vs blend decomposition |
| GET | `/accuracy/error-bands?tier=&horizon=` | empirical residual quantiles |
| GET | `/insights/top-movers?level=&node=&n=` | largest forecast-vs-recent shifts |
| GET | `/insights/planning/{store_id}/{item_id}` | 28-day expected demand + band |
| POST | `/inference/verify` | start verification job → `{job_id}` |
| GET | `/inference/jobs/{job_id}` | poll status/result |

---

# 11. User workflows

**W1 — Planner: "what will this item sell in this store?"**
Search → series page → 28-day forecast over 90 days of history, with the empirical
error band, regime label and 28-day expected total.

**W2 — Manager: "what does my whole store need?"**
Hierarchy → pick store → coherent aggregate forecast → top movers → drill into any
department or item.

**W3 — Sceptic: "why should I believe this?"**
Accuracy page → measured accuracy at *their* aggregation level → backtest replay on
a window with known truth → predicted vs actual overlay.

**W4 — Evaluator: "is the model real?"**
Model card → capability matrix → **Run verification** → watch the frozen model
recompute the forecast in ~33 s and report exact agreement.

**W5 — Analyst: "how does the blend help?"**
Member decomposition on any backtest window: direct vs recursive vs blend, with the
measured residual correlation (0.949) that explains why the ensemble works.

---

# 12. Forecast visualisation strategy

| View | Chart | Why this one |
|---|---|---|
| Series detail | Line: history (solid) → forecast (dashed), shaded empirical band, origin marker | The single most legible way to show a 28-day-ahead forecast in context |
| Intermittent series | Bar for actuals + line for forecast | 68% of rows are zero; a line over zeros reads as a flat artefact, bars read as events |
| Hierarchy roll-up | Stacked area by child node | Shows composition and that aggregates are exact sums |
| Accuracy by level | Horizontal bar, 12 levels, WAPE-accuracy | Makes the 28%→95% story immediate and honest |
| Backtest replay | Overlay predicted vs actual + residual strip | Truth exists here, so show it |
| Member decomposition | Three-line overlay + residual-correlation callout | Explains the ensemble visually |
| Error concentration | Volume-decile bar | Communicates that the top decile carries 66% of squared error |

**Design rules, driven by the data's actual shape:**
- Never show a single global "accuracy" number — always the one matching the
  current aggregation level.
- Never render confidence bands as if the model produced them; empirical bands are
  labelled *"observed backtest error, not a model interval"*.
- Never show accuracy against the delivered forecast window — no ground truth
  exists there.
- Forecasts are continuous (0.63 units is meaningful); state whether a view rounds.

---

# 13. How the system addresses Use Case 11

| Requirement | How the product demonstrates it | Evidence |
|---|---|---|
| **Hierarchical aggregation** | Navigate all 12 M5 levels; every aggregate is an exact sum of bottom-level forecasts, i.e. coherent by construction; accuracy shown per level | `uc11_hierarchy_levels.csv` (measured, all 12 levels) |
| **External covariates — price** | Price shown on every series chart alongside demand; price features are inside the model (ablation-verified) | `ablation_abl_4_plus_price`; 30,490/30,490 series priced for all 28 forecast days |
| **External covariates — promo** | **Honestly reported as unavailable** — M5 has no promotion field; capability matrix says so | perfect-discount-correction oracle −0.0002 |
| **External covariates — holiday** | Events/SNAP overlaid on charts; both are model inputs | 28/28 forecast days covered |
| **Intermittent demand** | Every series labelled with its Syntetos-Boylan regime; Tweedie's per-regime adequacy shown against Croston/SBA/TSB | Champion wins in **all four** regimes |
| **28-day-ahead per store/item** | The core product: 30,490 series × 28 days | shipped forecast, RMSE 2.0929 |

---

# 14. What is genuinely supported vs what is not

### A. Implemented and validated
28-day forecasts for all 30,490 store-item series · coherent hierarchical
aggregation at 12 levels · measured accuracy per level · intermittency regime
classification · backtest accuracy across 8 windows with member decomposition ·
price/holiday/SNAP as model inputs.

### B. Investigated during research and **rejected** — will be shown as rejected, not built
| Idea | Measured outcome |
|---|---|
| Hierarchical reconciliation (top-down/MinT/middle-out) | oracle ≤ −0.0221 above item level; item-level failed mechanism 2/4 |
| Croston / SBA / TSB | lose to the champion in **every** regime |
| Promotion/discount covariates | perfect-correction oracle −0.0002 |
| Regime/volume/category segmentation | oracle −0.0008 |
| Hurdle models | 2.1267 / 2.1241, both worse |
| Per-series or global recalibration | −0.0014 global; Exp #69 failed |

Surfacing these *as rejected*, with numbers, is a strength: it shows the use case
was engaged with scientifically rather than decorated with buzzwords.

### C. Safe product features built around the frozen model
Exploration and search · aggregation · history-vs-forecast comparison ·
**empirical error bands** from backtest residuals (clearly labelled as observed
error, not model intervals) · planning summaries · top movers · live verification
job · capability matrix.

### D. Future work — **must not be claimed as implemented**
Model-produced prediction intervals · causal price/promotion response · live
retraining · stockout/censoring detection · multi-tenant auth · real-time data
ingestion.

### 14.3 A feature I tested and am rejecting: price what-if scenarios

A price-scenario slider was the most attractive candidate for demonstrating
"external covariates" interactively. **I implemented it as a probe and measured
the model's response before proposing it.** Perturbing `sell_price` for the 28
forecast weeks on the frozen model gives:

| Series | −20% | −10% | base | +10% | +20% |
|---|---|---|---|---|---|
| FOODS_3_090 / CA_3 | **−85.5%** | **−74.2%** | — | −1.4% | −0.4% |
| HOBBIES_1_001 / CA_1 | −7.1% | +4.2% | — | +0.1% | −2.2% |
| FOODS_2_360 / TX_2 | **−48.2%** | +7.4% | — | −0.3% | −1.4% |

The response is **non-monotone, inconsistent in sign, and economically backwards**
— a price cut predicts demand collapse. The cause is that the model is a
*forecaster* using price as a contextual feature (`price_rel_to_recent_avg`), not
a causal elasticity model; perturbed prices are off-distribution for series whose
price is historically stable.

**Decision: do not build it.** A slider that craters demand when an evaluator cuts
a price would destroy the credibility of the whole solution in one gesture. This
belongs in category **D**, and the capability matrix will say so explicitly.

---

# 15. Security considerations

Realistic for a demo-stage analytical product:

| Concern | Treatment |
|---|---|
| Authentication | **Not implemented.** Read-only public data, no PII, no mutations. Adding auth would be theatre. Documented as a deliberate decision, with the hook (`Depends(get_current_user)`) left in place |
| SQL injection | All identifiers whitelisted against `series_meta`; all values passed as DuckDB bound parameters. **No f-string SQL** |
| Path traversal | No user input reaches a filesystem path; artefact paths are settings constants |
| DoS via expensive queries | Hard `LIMIT` on all list endpoints; hierarchy queries bounded by level cardinality; verification jobs rate-limited to 1 concurrent |
| CORS | Explicit allow-list, not `*` |
| Research-tree integrity | DuckDB opened `read_only=True` on the parquet; the API process never writes outside `06_BACKEND/data/` |
| Dependency supply chain | Fully pinned `requirements.txt` / `package-lock.json` |
| Secrets | None required; `.env` for config only, git-ignored |

---

# 16. Error handling

| Failure | Response |
|---|---|
| Unknown store/item | `404` with the nearest valid suggestions |
| Malformed params | `422` from Pydantic with field-level detail |
| `product.duckdb` missing | `/ready` fails with the exact build command to run |
| Parquet unreadable | `/ready` reports degraded mode; forecast endpoints still work |
| Verification job fails | Job status `failed` + captured traceback tail; **the API stays up** |
| Verification mismatch | Job succeeds with `verdict: MISMATCH` and the max deviation — **reported, never hidden** |
| Frontend request failure | Error boundary per panel; one failed chart never blanks the page |
| Empty result | Explicit empty-state UI, never a blank chart |

Structured JSON logging with a request-id echoed in the response header.

---

# 17. Performance considerations

| Path | Target | Basis |
|---|---|---|
| Series history + forecast | < 200 ms | measured 0.16 s worst-case DuckDB query |
| Hierarchy aggregate | < 250 ms | measured 0.11–0.13 s |
| Cached aggregates | < 20 ms | in-process cache |
| Verification job | ~35 s | measured 4.1 s + 29 s + load |
| API RSS | < 300 MB | no pipeline import, no model in-process |
| Worker RSS | ~800 MB | measured |
| Frontend bundle | < 400 kB gzipped | Vite code-splitting per route |

Charts downsample history server-side (`from`/`to` params) — 1,941 raw points per
series is more than any chart needs; default to 90 days pre-origin.

---

# 18. Deployment strategy

**Docker Compose, two services, one command.**

```yaml
services:
  api:       # FastAPI + DuckDB; mounts research tree READ-ONLY
  frontend:  # Vite build served by nginx, proxies /api → api
```

- Research tree mounted **`:ro`** — the container *cannot* modify protected
  artefacts even if code tried.
- `product.duckdb` built at image build time (or first boot) into a named volume.
- The 301 MB parquet and 1.78 GB research tree are **mounted, not baked** — keeps
  the image small and avoids copying protected data.
- Health checks on `/health`; frontend waits on API readiness.
- **Non-Docker path preserved:** `uvicorn app.main:app --reload` + `npm run dev`,
  because a laptop demo must never depend on Docker being healthy.
- A `Makefile` provides `make build-db`, `make dev`, `make up`, `make test`.

### Things that will make deployment awkward — flagged now
1. **`pipeline/config.py` mkdir-on-import** — with a read-only mount this can
   raise. *Mitigated by design:* the API never imports it; the worker gets a
   read-write mount for `predictions/uc11_cache/` only, or runs outside the
   container.
2. **1.78 GB of artefacts** — mount, never `COPY`.
3. **Bleeding-edge pins** (`pandas 3.0.5`, `numpy 2.5.1`, Python 3.13) — the image
   must pin the exact Python minor or LightGBM/pandas may not resolve.
4. **`requirements.txt` is missing `duckdb`, `fastapi`, `uvicorn`** despite them
   being installed — must be fixed before any container build.
5. **No version control** — product code needs a repo before multi-person work.
6. **Windows-authored, Linux-deployed** — all code uses `pathlib`; CI on Linux will
   catch any strays.

---

# 19. Testing strategy

| Level | Scope |
|---|---|
| **Integrity tests** (highest value) | Forecast row count = 30,490; no NaN/negatives; aggregate of children == parent (coherence); model-card hashes match `CHAMPION_MANIFEST.json` |
| Unit | Aggregation helpers, error-band lookup, regime classification, formatters |
| API contract | Every endpoint: happy path, 404, 422; snapshot the OpenAPI schema |
| Data-layer | Each SQL service against a small fixture DB |
| Frontend | Vitest for hooks/formatters; React Testing Library for chart containers |
| E2E | Playwright: W1–W4 workflows against a running stack |
| **Regression guard** | A test asserting the served forecast still matches the frozen artefact hash — fails loudly if anyone swaps the model |

The regression guard matters most: it makes accidental violation of the freeze a
**test failure**, not a silent product change.

---

# 20. Development roadmap

Sized for a small team; **Phase 3 is the minimum demoable cut line.**

| Phase | Deliverable | Est. |
|---|---|---|
| **0** | `build_product_db.py`; pin deps; scaffold both apps; `make dev` runs | 0.5 d |
| **1** | API core: health, meta, hierarchy, series, forecast + tests | 1.0 d |
| **2** | Frontend shell, Explorer, Series Detail with history+forecast chart | 1.5 d |
| **3** | Aggregation views + accuracy-by-level → **demoable end-to-end** | 1.0 d |
| **4** | Backtest replay + member decomposition | 1.0 d |
| **5** | Verification job + model card + capability matrix | 0.5 d |
| **6** | Insights, error bands, empty/error states, polish | 0.5 d |
| **7** | Docker, E2E tests, README, demo script | 1.0 d |
| | **Total** | **~7 days** |

Cut order under pressure: 6 → 4 → 5. Never cut Phase 3 or the capability matrix.

---

# 21. File-by-file implementation plan

### Phase 0
| File | Purpose |
|---|---|
| `06_BACKEND/scripts/build_product_db.py` | Read protected artefacts read-only → write `data/product.duckdb`. Builds all 6 tables; computes error bands from backtest residuals; embeds hashes |
| `06_BACKEND/requirements.txt` | Pin fastapi, uvicorn, duckdb, pydantic-settings, pytest, httpx (+ lightgbm/pandas/numpy for the worker) |
| `06_BACKEND/app/config.py` | Settings: artefact paths, DB path, CORS origins, cache TTL |
| `06_BACKEND/app/db.py` | DuckDB connection factory, read-only parquet attach, identifier whitelist |
| `07_FRONTEND/package.json`, `vite.config.ts`, `tailwind.config.js` | Scaffold + `/api` proxy |
| `Makefile` | `build-db`, `dev`, `up`, `test` |

### Phase 1
`app/main.py` · `app/cache.py` · `routers/{health,meta,hierarchy,series,forecast}.py`
· matching `services/*` and `schemas/*` · `tests/test_{health,hierarchy,series}.py`

### Phase 2
`src/api/client.ts` (typed from OpenAPI) · `src/pages/{Dashboard,Explorer,SeriesDetail}.tsx`
· `src/components/charts/{ForecastChart,IntermittentChart}.tsx` ·
`src/components/layout/{AppShell,Sidebar}.tsx` · `src/hooks/{useSeries,useHierarchy}.ts`

### Phase 3
`routers/accuracy.py` + `services/accuracy.py` · `src/pages/{Hierarchy,Accuracy}.tsx` ·
`src/components/charts/{RollupArea,LevelAccuracyBar}.tsx`

### Phase 4
`services/backtest.py` · `src/components/charts/{BacktestOverlay,MemberDecomposition}.tsx`

### Phase 5
`app/worker/verify_forecast.py` (**the only file importing `pipeline`**) ·
`app/worker/runner.py` · `routers/inference.py` · `src/pages/ModelCard.tsx` ·
`src/components/CapabilityMatrix.tsx`

### Phase 6–7
`routers/insights.py` · error boundaries/empty states · `Dockerfile` ×2 ·
`docker-compose.yml` · `tests/e2e/` · `06_BACKEND/README.md` (corrected) ·
`08_DOCUMENTATION/DEMO_SCRIPT.md`

---

# 22. Risks and mitigation

| Risk | Sev | Mitigation |
|---|---|---|
| Someone "improves" the model mid-build | High | Freeze doc + hash regression test that fails CI |
| Demo shows a misleading accuracy number | High | Level-matched accuracy enforced in the API contract; no global number exists to display |
| Live verification fails on stage | Med | Tier 1 is fully independent; verification is opt-in and its failure is a job status, not an outage |
| Off-distribution scenario features creep back in | Med | §14.3 documents the measurement; capability matrix ships in-product |
| Frontend scope creep eats the timeline | Med | Phase 3 cut line; phases 4–6 are independently droppable |
| DuckDB/parquet path differs in container | Low | Settings-driven paths; `/ready` verifies artefacts at boot |
| Bleeding-edge dependency conflicts in Docker | Med | Pin exact Python minor; build the image early in Phase 0, not Phase 7 |
| No version control | Med | Initialise a repo for product code before Phase 1 |

---

# 23. What must remain untouched from the ML research layer

**Read-only, always:**
`data/raw/` · `data/processed/` · `models/` · `predictions/` (except the worker's
own cache dir) · `experiments/` · `reports/` · `docs/` · `pipeline/` · `scripts/` ·
`MY_RESEARCH_PAPER/`

**Product code writes only to** `06_BACKEND/`, `07_FRONTEND/`, and
`08_DOCUMENTATION/`.

Enforced three ways: read-only container mount; `read_only=True` DuckDB handles;
and a CI integrity check reusing
`scripts/08_organization/61_integrity_manifest.py`.

---

# THIS IS THE ARCHITECTURE I RECOMMEND

**A React + TypeScript SPA over a FastAPI service that reads a DuckDB layer built
from the frozen artefacts, with a separate, opt-in inference worker that
reproduces the shipped forecast live from the frozen model.**

Why this is the strongest balance:

**Correctness.** The API never imports the research pipeline and never writes to
it; the container mounts it read-only; a hash regression test makes any violation
of the freeze a build failure. The forecast served is the frozen artefact — there
is exactly one right answer and we serve it.

**Hackathon impact.** It demonstrates all four Use Case 11 pillars with *measured*
evidence rather than assertions, and the verification job gives a live "the model
is real, watch it reproduce the forecast in 33 seconds" moment. The capability
matrix turns the research's rejected hypotheses into a credibility asset instead
of hiding them.

**Usability.** Four concrete workflows for four real personas, with the
level-matched-accuracy rule preventing the single most likely way a demand
forecasting UI misleads its user.

**Engineering quality.** Clean layer separation, typed contracts end-to-end,
auto-generated OpenAPI, SQL confined to a service layer, tests that assert the
*scientific* invariants (coherence, artefact hashes) and not just HTTP codes.

**Implementation feasibility.** ~7 days with a cut line at day 4; every major
dependency is already installed; every performance target is backed by a
measurement I took today rather than an estimate.

**Scalability.** DuckDB handles 59M rows at ~100 ms without a server; the API is
stateless and horizontally scalable; the worker is already a separate process, so
it becomes a queue consumer whenever that is needed.

**Scientific honesty.** This is where the architecture earns most. The product
ships the capability matrix in-product; refuses to fabricate prediction intervals;
labels empirical bands as observed backtest error; shows no accuracy for a window
with no ground truth; forbids earlier-origin inference with the shipped boosters;
and **drops the most seductive feature available — price what-if — because I
measured the model's price response and it is backwards.** A product that declines
to ship a misleading slider is worth more than one that ships it.

### Two decisions I need from you before Phase 0
1. **Frontend stack:** React+TS as recommended (~7 days), or Streamlit fallback
   (~4 days, materially lower UI/UX ceiling)?
2. **Version control:** may I initialise a git repository for the product code?
   Without it, multi-person work in Phase 1+ is risky.

**Stopping here as instructed.** No implementation has begun.
