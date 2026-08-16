# FRONTEND IMPLEMENTATION REPORT

**Retail Demand Forecasting** — Walmart M5 · Hierarchical 28-Day Forecasting
**Scope:** React + TypeScript application over the frozen forecasting model
**Status:** complete — 8 pages, 30 tests passing, production build verified against a live API

---

## 1. What was built

A forecasting product, not a dashboard. Eight pages that walk an evaluator from
*what problem is this solving* to *why should I trust the result*, with every
number sourced from the API and every metric explained in one plain sentence.

```
DATA → SELECTION → HISTORY → MODEL → 28-DAY FORECAST → ACCURACY → DECISION SUPPORT
```

The sidebar is grouped along that flow (Forecasting / Evidence / How it works)
so the shape of the argument is visible before any page is opened.

---

## 2. Architecture

```
Browser
  └─ React 18 + TypeScript (Vite)
       ├─ pages/          8 route-level screens
       ├─ components/     ui primitives · charts · explanatory diagrams
       ├─ api/            typed client + one hook per endpoint
       └─ lib/            pure formatting and chart transforms
            │
            │  same-origin  /api/v1
            ▼
       nginx (prod) or Vite proxy (dev)
            ▼
       FastAPI  →  DuckDB + frozen model
```

**One rule governs the whole layer: the frontend computes nothing the backend
could compute.** No aggregation, no metric derivation, no re-implementation of
the blend. The only client-side logic is presentation formatting and merging
history with forecast onto a shared axis — both pure functions, both unit-tested.

### Technology choices

| Choice | Why | Rejected alternative |
|---|---|---|
| React 18 + TypeScript | Typed responses turn a backend contract change into a compile error rather than a blank chart | plain JS — no contract safety |
| Vite | Fast builds, native code-splitting, `import.meta.env` config | CRA (unmaintained), Next.js (SSR buys nothing for an authenticated-free analytical SPA) |
| Tailwind | One config file holds the entire palette, so the restrained dark theme cannot drift | CSS modules — more files, more drift |
| Recharts | Declarative composition; the composed chart this product needs most (history → forecast with a band) is straightforward and correct | ECharts (heavier, imperative), D3 (too low-level for the deadline) |
| TanStack Query | Cache, dedup, loading/error states. The model is frozen, so most responses are immutable and long stale times are *correct* | Redux — the only real state is the server cache |

**No component library.** A pre-built kit would have fought the deliberately
restrained visual language and added weight for components this app does not need.

---

## 3. Pages

| Route | Purpose | Key API calls |
|---|---|---|
| `/` **Overview** | The 30-second answer. Headline metrics, accuracy-by-level bars, pipeline diagram, provenance | `meta/model`, `accuracy/levels`, `accuracy/occurrence`, `meta/provenance` |
| `/forecast` **Forecast** | Search a store-item; history flows into the 28-day forecast; planning view with weekly breakdown | `hierarchy/search`, `series/*/history`, `series/*/forecast`, `insights/planning/*` |
| `/hierarchy` **Hierarchy** | Drill chain → state → store → dept → item; shows aggregates are exact sums | `hierarchy/levels`, `hierarchy/nodes`, `hierarchy/aggregate`, `accuracy/levels` |
| `/insights` **Insights** | Portfolio summary, regime mix, products rising/falling fastest | `insights/summary`, `insights/top-movers`, `hierarchy/nodes` |
| `/accuracy` **Accuracy** | 8 windows, horizon decay, regimes, volume tiers, occurrence + confusion matrix, member split | 6 `accuracy/*` endpoints |
| `/validation` **Validation** | Replay a historical window with known outcomes; predicted vs actual + daily miss | `accuracy/windows`, `accuracy/backtest*` |
| `/model` **Model** | Architecture, specification, **live model re-run**, artefact fingerprints | `meta/model`, `accuracy/members`, `inference/*` |
| `/methodology` **Methodology** | Data explanation, intermittent demand, covariates, capability matrix, limitations | `meta/capabilities`, `accuracy/regimes`, `meta/provenance` |

**Components:** 6 UI primitives (`Card`, `Metric`, `Badge`, states, `Explain`,
`Caveat`, `Async`), 1 composed chart (`ForecastChart`), 4 explanatory diagrams
(pipeline, hierarchy, validation, data row), 1 app shell.

The `Async` boundary renders loading → error → empty → data uniformly, so no
screen can accidentally ship a blank panel.

---

## 4. Design system

Deep navy-slate, never pure black, so cards separate from the page and charts
stay readable.

| Token | Value | Role |
|---|---|---|
| `base` | `#0B1220` | page background |
| `surface` | `#131C2E` | cards |
| `elevated` | `#1B2740` | hover / raised |
| `line` / `line-strong` | `#243149` / `#31415F` | borders |
| `ink` / `ink-muted` / `ink-dim` | `#E8EDF6` / `#9AA9C2` / `#6B7C99` | text hierarchy |
| `forecast` | `#4EA8F0` | model output — the one accent |
| `actual` | `#C6D2E4` | observed reality |
| `good` / `warn` / `bad` | `#3FB98B` / `#D9A63C` / `#E0665F` | semantic states |

**One accent for "the model", one neutral for "reality".** Everything else is
semantic. No gradients, no glassmorphism, no glow. Animations are 150–200 ms
fades and hover transitions only, and all of them collapse under
`prefers-reduced-motion`.

Numbers use tabular figures (`font-variant-numeric: tnum`) so metrics and tables
do not jitter between values.

---

## 5. Charts and visualisations

| Visualisation | Where | Notes |
|---|---|---|
| History → forecast composed chart | Forecast, Hierarchy | Solid neutral line for actuals, **dashed** accent line for forecast, shaded empirical band, labelled origin reference line |
| Accuracy-by-level bars | Overview, Hierarchy | Makes the 28% → 97% spread immediate |
| Horizon decay line | Accuracy | RMSE and MAE across the 28 days |
| Volume-tier bars + error-share meters | Accuracy | Shows the top tier is 7.8% of rows but 61% of squared error |
| Regime cards | Accuracy, Methodology | Per Syntetos-Boylan class |
| Confusion matrix | Accuracy | 2×2 grid, labelled in words — never colour alone |
| Member comparison bars | Accuracy, Model | Direct vs recursive vs blend |
| Backtest overlay + daily-miss bars | Validation | Predicted vs actual where truth exists |
| Regime-mix stacked bar | Insights | Portfolio composition |
| 4 HTML/flex diagrams | Overview, Model, Methodology | Pipeline, hierarchy, validation, data row — reflow on narrow screens, readable in order by a screen reader |

**Encoding never relies on colour alone.** History vs forecast is carried by dash
pattern and position; the confusion matrix is labelled in words; every bar has a
numeric label.

---

## 6. Scientific integrity in the UI

This is the part that makes it a research demonstrator rather than a dashboard.

| Rule | How it is enforced |
|---|---|
| No fabricated metrics | Every figure comes from an API response. A test asserts that when the API fails, **no** numbers render — only an error state |
| No fake confidence intervals | Bands are labelled *"observed model error, NOT a model-produced prediction interval"*, carried verbatim from the backend's `band_basis`. A test asserts that wording reaches the DOM |
| No global accuracy number | Accuracy is always shown for the level being viewed. The Overview shows four levels side by side precisely so one number cannot stand in for the system |
| Validation ≠ live accuracy | The Accuracy and Validation pages both open with a prominent caveat. No accuracy is shown against the delivered forecast window, which has no ground truth |
| Rejected research is shown as rejected | `/methodology` renders the backend's capability matrix — implemented / rejected / not-supported — with the measured evidence for each |
| **No price what-if slider** | Deliberately absent, and `/methodology` explains why: the frozen model's measured price response is non-monotone and sometimes economically backwards |

### A correction made during implementation

The occurrence metrics in the original brief were garbled. Computed from the
verified backtest artefact using the research's documented 0.5-unit rule:

| Metric | Brief said | **Computed** |
|---|---|---|
| Accuracy | 0.8068 | **0.6980** |
| Precision | 0.7088 | **0.6321** |
| Recall | 0.8068 | **0.8068** ✓ |
| F1 | 0.7549 | **0.7088** |

The brief's "accuracy" was the recall value and its "precision" was the F1 value.
**The UI shows the computed truth**, and a test pins 69.8% / 63.2% / 80.7% so the
inflated figures cannot creep back in.

### Two backend endpoints added

The brief asked for occurrence and high-volume metrics that the API did not
expose. Rather than omit them or hard-code them, they were added as real
computations over the verified backtest artefact:

- `GET /accuracy/occurrence` — confusion matrix + accuracy/precision/recall/F1,
  with the backend's own caveat that the model was never trained to classify
- `GET /accuracy/volume-tiers` — RMSE, bias and error share by volume tier,
  stating explicitly that tiers here use full-history means, which differs
  slightly from the research report's per-origin tiering

No model, dataset, prediction or experiment record was touched.

---

## 7. API integration

`src/api/` is the entire contract boundary:

- **`types.ts`** — hand-written to mirror `openapi.json`. Written by hand rather
  than generated so that the fields carrying honesty statements (`band_basis`,
  `expected_accuracy`, `caveat`) are explicit and cannot be silently dropped.
- **`client.ts`** — one `apiFetch`. Base URL resolves to `VITE_API_BASE_URL` or
  falls back to same-origin `/api/v1`. Maps structured backend errors to
  `ApiError` with a `userMessage` safe to render.
- **`hooks.ts`** — one hook per endpoint, 27 in total. Frozen data uses
  `staleTime: Infinity`; the inference job hook polls at 1.5 s and stops the
  moment the job settles.

**All 27 endpoints verified against the live API** (see §9).

---

## 8. Accessibility

- Semantic landmarks: `<nav aria-label="Main">`, `<main>`, `<section>`, `<dl>` for metrics
- Visible focus ring on every interactive element, offset against the page background
- Toggle groups use `aria-pressed`; the nav uses `aria-current`; errors use `role="alert"`; spinners use `role="status"`
- Tables carry `<caption>` (screen-reader only) and `<th scope="col">`
- Charts sit in figures with `sr-only` captions describing what they show
- No information conveyed by colour alone
- `prefers-reduced-motion` collapses all animation
- Contrast: `ink` on `base` ≈ 14:1, `ink-muted` on `surface` ≈ 7:1 — both above WCAG AA

---

## 9. Verification results

| Check | Result |
|---|---|
| Frontend tests | **30 passed** (3 files) |
| Backend tests | **78 passed**, 2 slow deselected |
| TypeScript | clean, `--noEmit` |
| Production build | **succeeds**, 5.2 s, 712 modules |
| All 27 frontend-consumed endpoints, live | **27 OK / 0 failed** |
| Production bundle served + hitting live API | **verified** — `/`, `/forecast`, `/methodology` all HTTP 200; API returns RMSE 2.0929 / MAE 1.0395 / FROZEN through the proxy |
| Hard-coded localhost in `src/` | none (only a comment saying it is not hard-coded) |
| Hard-coded API host in the **built bundle** | **none** |
| Windows paths in shipped code | none (only gitignored `.pyc` bytecode) |
| Research artefacts modified | **0 deleted, 0 changed** |
| Raw-data MD5 | `1bce3321…` unchanged |
| Docker image build | **NOT VERIFIED — see §11** |

### Bundle

| Chunk | Raw | Gzip |
|---|---|---|
| react | 157.9 kB | 51.8 kB |
| query | 47.3 kB | 14.7 kB |
| index (app shell + Overview) | 30.0 kB | 9.7 kB |
| css | 20.6 kB | 4.9 kB |
| **initial load** | **~256 kB** | **~81 kB** |
| charts (lazy) | 416.3 kB | 112.2 kB |
| per-page chunks | 2.9–11.2 kB | 1.2–3.8 kB |

Charts load only when a chart route is opened, so the largest dependency is off
the critical path.

### Test coverage

| File | n | Covers |
|---|---|---|
| `format.test.ts` | 13 | compact/percent/signed/date formatting; the history↔forecast merge — including that it never puts an actual on a forecast day, never invents a forecast for a past day, and emits a band only where the API supplied both bounds |
| `client.test.ts` | 8 | base URL is same-origin; query serialisation; structured error → `ApiError`; 503 messaging; network failure; non-JSON error body |
| `pages.test.tsx` | 9 | loading state; branding; **exact API metrics rendered**; occurrence values not inflated; backend caveat reaches the screen; multi-level accuracy; **error state renders no fabricated numbers**; all 8 nav links; API-offline badge; labelled nav landmark |

---

## 10. Deployment

```bash
python tasks.py build-db      # once — 130 MB product data layer
docker compose up --build     # http://localhost:8080
```

Two services. The browser talks only to the frontend container; nginx proxies
`/api/` to the API over the internal network, so there is **one origin, no CORS
in play, and no API host in the JavaScript bundle**.

| Property | Detail |
|---|---|
| Frontend image | multi-stage: `node:22-alpine` builds, `nginx:1.27-alpine` serves |
| API host | `API_HOST` env var, substituted into the nginx template at container start |
| envsubst safety | `NGINX_ENVSUBST_FILTER=API_HOST` — only that variable is substituted, so nginx's own `$host`/`$uri` are never clobbered |
| SPA routing | `try_files … /index.html` so deep links survive a hard refresh |
| Caching | hashed assets immutable for 1 year; `index.html` `no-cache` |
| Proxy timeout | `180s` — live model verification takes ~45 s |
| Health | `/healthz` on the frontend; frontend `depends_on` the API being **healthy**, not merely started |
| Research tree | mounted `:ro`; both containers non-root |
| Memory | frontend limit 128 MB; API 2 GB |

**Without Docker**, the same stack runs as:

```bash
python tasks.py api      # :8000
python tasks.py ui       # :5173  (dev, proxied)
# or, to serve the real production bundle:
npm --prefix 07_FRONTEND run preview   # :4173, /api proxied
```

---

## 11. Limitations

1. **The Docker images have never been built.** Docker is not installed on the
   development machine. Both Dockerfiles and the compose file are written to
   spec and statically validated (valid YAML, correct service graph, non-root,
   healthchecks, `:ro` mounts), and the production bundle they serve has been
   verified against a live API — but **no image build has been executed**. See
   §12 for the exact commands.
2. **No end-to-end browser test.** Component tests run against a mocked API and
   the built bundle was verified over HTTP, but there is no Playwright suite
   driving a real browser through the primary flow. This was traded for depth of
   honesty assertions within the time available.
3. **Error-band calibration is in-sample** to the backtest windows the quantiles
   were fitted on. Stated on-screen, and negligible at 6.8M observations across
   140 cells, but it is not an out-of-sample coverage guarantee.
4. **Search is server-side and exact-ish.** `hierarchy/search` matches on
   substrings of item/store ids; there is no fuzzy matching or typo tolerance.
5. **One default series is hard-coded as a landing example**
   (`CA_3 / FOODS_3_090`) so the Forecast page is never empty on first load. It
   is a navigation convenience, not data — the numbers come from the API.
6. **Mobile is supported, not optimised.** Layouts reflow and the nav collapses,
   but this is a desktop analytical tool and the dense tables assume width.
7. **No authentication.** Consistent with the backend: read-only public data, no
   PII, no mutations.

---

## 12. Remaining step: build and run the containers

Cannot be executed here. On a machine with Docker:

```bash
cd <project root>
python tasks.py build-db                  # if 01_PROJECT/backend/data is empty
docker compose build                      # builds both images
docker compose up
# then check:
curl -f http://localhost:8080/healthz                 # frontend alive
curl -f http://localhost:8080/api/v1/ready            # proxy reaches the API
open http://localhost:8080                            # the application
```

**What to watch for on the first build**, in likelihood order:

1. **API image dependency resolution.** `pandas 3.0.5` / `numpy 2.5.1` / Python
   3.13 are recent pins. If a wheel is unavailable for the base image, this is
   where it surfaces. The pins are deliberate — they are the versions the model
   was validated with — so prefer changing the base image over loosening them.
2. **`libgomp1`** is installed in the API `full` target because LightGBM will
   not import without it. If inference reports unavailable, check that first.
3. **`npm ci`** requires `package-lock.json`, which is committed.
4. If the frontend renders but every panel errors, the proxy is not reaching the
   API: check `API_HOST` and that the API container is healthy.

A lean alternative that avoids the ML dependency chain entirely:

```bash
docker compose build --build-arg target=api    # or edit target: api in compose
```
The API then serves everything except `/inference/*`, which reports 503 with a
reason, and the Model page renders that state correctly.

---

## 13. Future improvements

- Playwright end-to-end smoke test through the primary flow
- Server-driven fuzzy search, and recently-viewed series
- Saved comparisons (pin two series side by side)
- CSV export of a forecast or planning view
- A compact "print/share" view of the Overview for judging
- Out-of-sample band calibration on a held-out window not used to fit the quantiles

---

## 14. What was not touched

The frozen research layer. Verified after implementation:

```
files before : 520      MISSING (deleted) : 0
files after  : 522      CHANGED (rewritten): 0
raw-data MD5 : 1bce33213f53e8d7b5136bec2f4e67bc (unchanged)
-> PASS: no protected artefact was deleted or modified
```

The two added files under a protected root are Phase-1 organisation scripts, not
research artefacts. All frontend work is confined to `01_PROJECT/frontend/`, with the two
additive backend endpoints in `01_PROJECT/backend/app/` and this report in
`02_DOCUMENTATION/`.
