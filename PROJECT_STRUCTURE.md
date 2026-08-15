# PROJECT STRUCTURE

**NPN_HACKATHON — Walmart M5 store-item demand forecasting**

The ML/forecasting phase is complete and the champion is **FROZEN**. This file
explains where everything lives and where the next phase happens.

---

## The five things most people want

| I want… | Path |
|---|---|
| **The frozen model + its freeze document** | `02_MODEL/` → [`MODEL_FREEZE.md`](02_MODEL/MODEL_FREEZE.md) |
| **The final 28-day forecast** | `03_FORECASTS/final_forecast_28day_v3_diversity_blend.csv` |
| **The research report** | `05_REPORTS/FINAL_RESEARCH_REPORT/` |
| **What to hand in** | `09_SUBMISSION/` |
| **Where to build next** | `06_BACKEND/` and `07_FRONTEND/` |

---

## The frozen champion

```
0.60 x Direct (38 features)  +  0.40 x Recursive (32 features)
both LightGBM, Tweedie(1.1), 400 rounds, seed 42

Validation (d_1914–d_1941, 853,720 predictions)
    RMSE 2.0929      MAE 1.0395      WAPE 0.7205      bias -0.0224

STATUS: FROZEN — do not modify without deliberate approval
```

| | Path |
|---|---|
| Freeze document | `02_MODEL/MODEL_FREEZE.md` |
| Frozen copies | `02_MODEL/FROZEN_CHAMPION/` |
| Canonical sources | `models/champion/model_11_…txt`, `model_12_…txt` |
| Hash manifest | `02_MODEL/FROZEN_CHAMPION/CHAMPION_MANIFEST.json` |

---

## Two layers, and why

This project has **two** top-level layers, deliberately:

### Layer 1 — the numbered delivery layer (new)

`01_DATA/` … `99_ARCHIVE/`. Navigation, frozen copies of deliverables, and the
backend/frontend workspaces. **Safe to reorganise.**

### Layer 2 — the research pipeline (unchanged, path-locked)

`pipeline/`, `data/`, `models/`, `predictions/`, `experiments/`, `reports/`,
`docs/`, `scripts/`, `MY_RESEARCH_PAPER/`. **Do not move any of these.**

`pipeline/config.py` resolves every path from

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

so `pipeline/` must stay exactly one level below the project root, and every
folder it names must keep its current name and position. `MY_RESEARCH_PAPER/`
has the same constraint through its own `parent.parent`. Moving any of them
breaks all 86 experiments' reproducibility and the research-paper build.

The delivery layer therefore holds **copies and pointers**, never relocations.

---

## Top-level folders

| Folder | Contains | Notes |
|---|---|---|
| `01_DATA/` | Pointer to the real datasets + integrity hashes | Data itself stays in `data/` |
| `02_MODEL/` | **Frozen champion** + `MODEL_FREEZE.md` | Byte-verified copies |
| `03_FORECASTS/` | The shipped 28-day forecast | Copy; flags the stale M5 submission |
| `04_EXPERIMENTS/` | `EXPERIMENT_CLASSIFICATION.md` — all 86 records sorted | Index only; registry not moved |
| `05_REPORTS/` | `FINAL_RESEARCH_REPORT/` — paper, reports, figures | Copies |
| `06_BACKEND/` | FastAPI service — 28 endpoints, 80 tests | `python tasks.py api` |
| `07_FRONTEND/` | React + TypeScript app — 8 pages, 30 tests | `python tasks.py ui` |
| `08_DOCUMENTATION/` | Structure, audit, integrity manifests | |
| `09_SUBMISSION/` | Copies of final deliverables only | Originals untouched |
| `99_ARCHIVE/` | Deliberately empty | Nothing was archived or deleted; reasons documented |

## Research pipeline folders (do not move)

| Folder | Contains |
|---|---|
| `pipeline/` | 22 modules — config, data loader, 6 feature builders, backtest, metrics, models, recursive, champion_blend, aggregate_level, validation checks, report renderer |
| `data/raw/` | 5 immutable competition CSVs |
| `data/processed/` | 59.2M-row parquet + build audits |
| `models/champion/` | shipped model (`model_11`, `model_12`) + 3 superseded champions |
| `models/experiments/` | 11 experimental models |
| `predictions/final_forecast/` | 3 forecasts + 1 stale M5 submission |
| `predictions/validation/` | 28 backtest prediction files |
| `predictions/uc11_cache/` | 8 cached champion reproductions (Stage 7) |
| `experiments/registry/` | 86 experiment records |
| `experiments/artifacts/` | 74 result tables and diagnostics |
| `scripts/01_…08_` | 58 run scripts, numbered chronologically |
| `reports/` | 27 stage reports (PDF + markdown) + 19 charts |
| `docs/` | problem statement, dataset guides, EDA, approach |
| `MY_RESEARCH_PAPER/` | paper sources, build scripts, figures, reproduction |

---

## Where the next phase happens

```
06_BACKEND/     FastAPI + DuckDB over the frozen model      python tasks.py api
07_FRONTEND/    React + TypeScript product UI               python tasks.py ui
```

Run the whole stack with `docker compose up --build` (http://localhost:8080),
or verify everything with `python tasks.py verify-all`.

Both have READMEs covering what data is available, what to serve, and the
data-shape realities that will otherwise bite (68% zeros, accuracy varying
~28%→~95% by aggregation level, no ground truth for the forecast window, point
forecasts with no intervals).

**The backend serves precomputed forecasts, not live inference, on the request
path.** All 853,720 predictions already exist, and the forecast is a fixed
quantity because the model is frozen and its covariates are published. Live
inference is still available on demand at `POST /api/v1/inference/verify`, which
reloads the frozen model and re-derives the forecast in ~45 s to prove the
published artefact is reproducible.

---

## Reproducing the research

```bash
python -m pip install -r requirements.txt
python scripts/01_foundation/01_foundation_check.py                  # 46 integrity + leakage checks
python scripts/06_research_campaign/41_exp77_blend_final_forecast.py # rebuild the shipped forecast
```

Scripts locate the project root by walking up to the folder containing
`pipeline/config.py`, so they run from any working directory.

---

## Known documentation drift

`README.md`, `PROJECT_INDEX.md`, `models/champion/README.md` and
`predictions/final_forecast/README.md` still describe the **superseded**
32-feature model (RMSE 2.1210) as the champion. They pre-date the blend and were
left untouched during reorganisation because they document protected artefacts.
See `08_DOCUMENTATION/ORGANIZATION_AUDIT.md` for the full list and exact lines.
