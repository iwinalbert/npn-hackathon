# 07_FRONTEND — Retail Demand Forecasting

React + TypeScript application over the frozen forecasting model.

```
Status: complete — 9 pages, 44 tests passing, production build verified
```

Full implementation report:
[`08_DOCUMENTATION/FRONTEND_IMPLEMENTATION_REPORT.md`](../08_DOCUMENTATION/FRONTEND_IMPLEMENTATION_REPORT.md)

---

## Quick start

```bash
# 1. the API must be running (it serves all data)
python tasks.py build-db      # once
python tasks.py api           # http://localhost:8000

# 2. the frontend
python tasks.py ui            # http://localhost:5173
```

`npm run dev` proxies `/api` to `http://127.0.0.1:8000`, so the browser sees a
single origin and CORS is never exercised in development either.

| Command | What it does |
|---|---|
| `python tasks.py ui` | dev server with hot reload |
| `python tasks.py ui-build` | production build into `dist/` |
| `python tasks.py ui-test` | 30 tests |
| `npm run typecheck` | TypeScript, no emit |
| `npm run preview` | serve the built bundle locally |

---

## Stack, and why

| Choice | Reason |
|---|---|
| **React 18 + TypeScript** | Typed API responses catch a backend contract change at compile time rather than as a blank chart |
| **Vite** | Fast builds, native code-splitting, `import.meta.env` for configuration |
| **Tailwind** | A constrained palette enforced in one config file; no bespoke CSS to drift |
| **Recharts** | Declarative composition, good defaults, and the one chart shape this product needs most (history flowing into forecast) is straightforward to build correctly |
| **TanStack Query** | Caching, dedup and loading/error state for free. The model is frozen, so most responses are immutable — long stale times are correct, not lazy |

**Deliberately not used:** a component library (would fight the deliberately
restrained visual language), Redux (the app is read-mostly; server cache is the
only real state), and a charting library with a canvas renderer (unnecessary at
these data volumes after server-side windowing).

---

## Pages

| Route | Purpose |
|---|---|
| `/` | Overview — the 30-second answer: what is forecast, how well, on what data |
| `/forecast` | Search a store-item, see its history flow into the 28-day forecast, plus a planning view |
| `/hierarchy` | Drill from chain to item; shows that aggregates are exact sums of bottom-level forecasts |
| `/insights` | Portfolio summary and the products rising or falling fastest |
| `/accuracy` | Measured performance: 8 windows, horizon, regimes, volume tiers, occurrence, member split |
| `/validation` | Replay a historical window where the true outcome is known |
| `/model` | Architecture, specification, and a live re-run of the frozen model |
| `/methodology` | Data, intermittent demand, covariates, capability matrix, honest limitations |
| `/assistant` | AI Forecast Assistant — ask about a forecast in plain language |

Overview loads eagerly; every other route is lazy-loaded so the 112 KB chart
bundle is not on the critical path.

### The assistant page

Optional: it is driven by `/genai/*`, and if the API has no `GEMINI_API_KEY` the
page explains why it is unavailable and states that the rest of the application
works without it — no broken input box, no silent failure.

**No key or AI SDK exists in this bundle.** Questions go to the FastAPI backend,
which holds the key and calls Gemini server-side. A test asserts the rendered DOM
never contains anything key-shaped.

Every answer carries a provenance strip: which data family it drew on, how long
it took, and whether **every number in the reply traced back to backend data**.
When a figure cannot be traced, the UI names it and marks it unverified rather
than hiding it. Design in
[`08_DOCUMENTATION/GENAI_IMPLEMENTATION_REPORT.md`](../08_DOCUMENTATION/GENAI_IMPLEMENTATION_REPORT.md).

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `VITE_API_BASE_URL` | *(unset)* | Leave unset behind a shared proxy — the client falls back to same-origin `/api/v1`. Set only for a split-origin deployment |
| `VITE_DEV_API_TARGET` | `http://127.0.0.1:8000` | Dev-server proxy target only |
| `API_HOST` *(container)* | `api:8000` | Where nginx proxies `/api/` at runtime |

**No API host is compiled into the bundle by default.** A test asserts that
`API_BASE` never matches `localhost`, an IP, or an absolute URL.

---

## Deployment

```bash
docker compose up --build     # http://localhost:8080
```

Multi-stage build: node compiles the bundle, nginx serves the static output and
proxies `/api/` to the API service over the internal network.

- `NGINX_ENVSUBST_FILTER=API_HOST` — only `API_HOST` is substituted into the
  nginx template, so nginx's own `$host` / `$uri` variables are never clobbered.
- Hashed assets cached for a year; `index.html` never cached.
- `try_files` fallback so client-side routes survive a hard refresh.
- `proxy_read_timeout 180s` because live model verification takes ~45 s.
- `/healthz` for container health checks.

**The Docker image has not been built** — Docker is unavailable on the
development machine. See the implementation report for the exact command to run
and what to check.

---

## Two things this UI will not do

**It will not show a price what-if slider.** The research measured the frozen
model's response to simulated price changes and found it non-monotone and
sometimes economically backwards. The model uses price as forecasting context,
not as a causal lever, so no such control is offered and `/methodology` says so.

**It will not present validation accuracy as live accuracy.** Every figure comes
from held-out windows with known outcomes. The delivered 28-day forecast has no
recorded outcome, so no accuracy is ever quoted against it.

---

## Testing

```bash
python tasks.py ui-test
```

44 tests across four files: pure formatting/transform logic, the API client
(URL building, error mapping, network failure), component rendering against a
mocked API, and the assistant. The load-bearing assertions are the honesty ones
— that the app renders exactly the numbers the API returned, that a failed
request produces an error state rather than a plausible-looking placeholder,
that backend caveats reach the screen, and that an AI answer containing an
untraceable number is flagged as unverified instead of shown as fact.
