# PROJECT STRUCTURE

**Retail Demand Forecasting** — Walmart M5 · Hierarchical 28-Day Forecasting

The repository is split by the question each area answers.

| | Answers | Contains |
|---|---|---|
| **`backend/`** | *What do we ship?* | FastAPI + DuckDB service, tests, Dockerfile |
| **`frontend/`** | *What do we ship?* | React app, tests, Dockerfile, nginx.conf |
| **`docs/`** | *What does it mean?* | Every document that explains, validates or presents the work |
| **`research/`** | *How did we get here?* | The full research history: pipeline, data, models, experiments, reports |
| **`infra/`** | *How do we run it?* | CI, pre-deploy gate, smoke tests, production overlay, runbook |
| **`study-materials/`** | *How does the team learn it?* | Per-role guides, two depths each |

The split is by **role**, not by file type. A PDF in `research/reports/` is a
research artefact; the same PDF in `docs/11_SUBMISSION/` is a
deliverable. Both are correct, and each is where it is for a reason.

---

## Root

Five files stay at the top level because they must:

| File | Why it cannot move |
|---|---|
| `docker-compose.yml` | Its build context is the repository root. The `full` API image does `COPY research/pipeline /research/pipeline` — a context rooted at `backend/` could not reach the research tree |
| `.dockerignore` | Only honoured at the build context root |
| `tasks.py` | The single entry point, and it drives *both* the product and the research verification scripts |
| `Makefile` | Unix/CI equivalent of the same |
| `.gitignore`, `.gitattributes`, `.env.example` | Repository-wide by definition |

---

## `backend/` and `frontend/` — the product

```
backend/      FastAPI + DuckDB. 34 endpoints, 149 tests, Dockerfile,
              requirements, and data/ (the generated 130 MB product layer)
frontend/     React + TypeScript. 9 pages, 62 tests, Dockerfile, nginx.conf
```

**A fresh machine needs only these two folders plus `docker-compose.yml` to run
the product.** The API serves entirely from three generated files in
`backend/data/` — `product.duckdb`, `history.parquet`, `backtest.parquet` — and
needs no access to raw data, model binaries, or the research tree. That is
proven by `test_api_serves_with_no_research_tree_present`, which launches the
app against an empty project root.

The research tree is required only for two optional things: rebuilding the data
layer (`python tasks.py build-db`) and the live inference verification endpoint.

## `docs/` — the explanations

Numbered so the reading order is the thinking order: what it is, what the model
is, what the data is, how it is built, then how it was proved.

| Folder | Contents |
|---|---|
| `01_PROJECT_OVERVIEW/` | This file; the ML project overview; the file-level index |
| `02_MODEL/` | `MODEL_FREEZE.md` and `FROZEN_CHAMPION/` — the manifest **and the two binaries whose SHA-256 it certifies**, kept together because neither proves anything alone |
| `03_DATA/` | Dataset description, provenance, immutability guarantees |
| `04_ARCHITECTURE/` | The architecture proposed and approved before implementation |
| `05_BACKEND/` | Backend implementation report: data layer, API, model serving |
| `06_FRONTEND/` | Frontend implementation report: pages, design system, integrity rules |
| `07_GENAI/` | AI assistant: Gemini integration, key handling, guardrails |
| `08_DEPLOYMENT/` | Docker deployment report; Git policy — what is versioned and what is deliberately not |
| `09_VALIDATION/` | Organisation audits, this reorganisation's report, and `_integrity/` — the SHA-256 manifests over all 522 protected artefacts |
| `10_RESEARCH_REPORT/` | The paper, the performance report, the Use Case 11 compliance report, the experiment classification |
| `11_SUBMISSION/` | Deliverable copies: the forecast CSV and the three PDFs |

## `research/` — how we got here

```
research/
├── pipeline/          reusable source package — the research library
├── scripts/           one-off run scripts, numbered by stage (01_foundation … 08_organization)
├── data/              raw/ (immutable competition CSVs) + processed/ (the 59.2M-row panel)
├── models/            champion/ (the frozen selection) + experiments/ (everything else)
├── predictions/       final_forecast/ (the deliverable) + validation/ (backtests)
├── experiments/       registry/ (86 JSON records) + artifacts/ (result tables)
├── reports/           25 stage reports, filed by phase
├── docs/              problem statement, dataset guides, EDA, approach documents
└── MY_RESEARCH_PAPER/ the paper's build scripts, figures and reproduction data
```

**This folder is one indivisible unit.** `pipeline/config.py` derives
`PROJECT_ROOT` as its own parent and resolves `data/`, `models/`,
`predictions/`, `experiments/` and `reports/` as siblings. `MY_RESEARCH_PAPER/`
and every script under `scripts/` use the same idiom. Moving any one of them
out on its own breaks path resolution for all of them — which is precisely why
they moved together, and why the move required **no change to
`pipeline/config.py` at all**.

Nothing here is needed to serve the product. Everything here is needed to
justify it.

---

## Navigation

| I want to… | Go to |
|---|---|
| Run the product | root → `python tasks.py api` / `ui` |
| Understand the architecture | `docs/04_ARCHITECTURE/` |
| Check what the model is and that it is frozen | `docs/02_MODEL/MODEL_FREEZE.md` |
| See measured accuracy | `docs/10_RESEARCH_REPORT/` |
| See what was tried and rejected | `docs/10_RESEARCH_REPORT/EXPERIMENT_CLASSIFICATION.md`, then `research/experiments/registry/` |
| Verify nothing was tampered with | `python tasks.py verify-integrity` |
| Rebuild the product data layer | `python tasks.py build-db` |
| Containerise / deploy | `docs/08_DEPLOYMENT/DOCKER_IMPLEMENTATION_REPORT.md` |

## The frozen model

```
ŷ = 0.60 × Direct LightGBM Tweedie(1.1, 38 features)
  + 0.40 × Recursive LightGBM Tweedie(1.1, 32 features)
400 rounds · seed 42 · deterministic
```

Validated on 853,720 held-out predictions (`d_1914`–`d_1941`):

| Metric | Value |
|---|---|
| RMSE | 2.0929 |
| MAE | 1.0395 |
| WAPE | 0.7205 |
| Bias | −0.0224 |

Occurrence ("did it sell at all", 0.5-unit rule): **accuracy 0.6980, precision
0.6321, recall 0.8068, F1 0.7088** — computed from the backtest artefact, and
worth stating explicitly because these four are easy to transpose.

No figure on this page comes from the delivered forecast window: `d_1942`–
`d_1969` has no recorded ground truth, so no accuracy exists for it.
