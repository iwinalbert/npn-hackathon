# Retail Demand Forecasting

**Walmart M5 · Hierarchical 28-Day Forecasting**
NPN AIA Hackathon · Cognizant Use Case 11

Forecast daily unit sales for **30,490 Walmart store-item series**, 28 days ahead —
and be honest about how accurate that can be.

---

## The repository, in three folders

| | | |
|---|---|---|
| **`01_PROJECT/`** | *what we ship* | FastAPI backend, React frontend, tests, containers |
| **`02_DOCUMENTATION/`** | *what we explain* | Model, data, architecture, validation, reports, submission |
| **`03_RESEARCH/`** | *how we got here* | Pipeline, datasets, 86 experiments, models, research reports |

Full map: [`02_DOCUMENTATION/01_PROJECT_OVERVIEW/PROJECT_STRUCTURE.md`](02_DOCUMENTATION/01_PROJECT_OVERVIEW/PROJECT_STRUCTURE.md)

## Run it

```bash
python tasks.py build-db     # once — builds the 130 MB product data layer (~10 s)
python tasks.py api          # http://localhost:8000   · docs at /docs
python tasks.py ui           # http://localhost:5173
```

```bash
python tasks.py test         # 149 backend tests, ~3 s
python tasks.py ui-test      # 62 frontend tests
python tasks.py verify-all   # everything, plus artefact integrity
python tasks.py help         # all commands
```

Optional AI assistant: put `GEMINI_API_KEY=…` in `01_PROJECT/backend/.env`
(gitignored). Without it every other feature works unchanged.

## The model

```
ŷ = 0.60 × Direct LightGBM Tweedie(1.1, 38 features)
  + 0.40 × Recursive LightGBM Tweedie(1.1, 32 features)
```

**Frozen.** Validated on 853,720 held-out predictions (`d_1914`–`d_1941`):
**RMSE 2.0929 · MAE 1.0395 · WAPE 0.7205 · bias −0.0224**.

Accuracy depends entirely on the level you aggregate to — 28.5% at
store-item-day, 94.5% chain-wide — because 54% of individual store-item-days are
zero and independent errors cancel on aggregation. The product always shows the
figure matching the decision being made.

The delivered window `d_1942`–`d_1969` has **no ground truth**, so no accuracy is
ever quoted against it.

## What this project deliberately does not do

- **No price what-if simulation.** The frozen model's measured response to
  simulated price cuts is non-monotone and sometimes economically backwards. It
  uses price as forecasting context, not as a causal lever.
- **No prediction intervals.** It emits point forecasts. The ranges shown are
  *observed past error*, labelled as such.
- **No promotion modelling.** The dataset has no promotion field, and nothing
  here pretends to recover one.

Reasoning and measurements: [`02_DOCUMENTATION/04_ARCHITECTURE/`](02_DOCUMENTATION/04_ARCHITECTURE/).
