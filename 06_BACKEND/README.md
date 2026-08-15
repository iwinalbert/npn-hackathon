# 06_BACKEND — workspace (empty, not yet built)

Backend development happens **here**. Nothing has been implemented yet.

## Ground rules

1. **The model is frozen.** Read `02_MODEL/MODEL_FREEZE.md` before writing code.
   The backend consumes forecasts; it does not retrain, re-tune or re-blend.
2. **Do not write into the research tree.** `data/`, `models/`, `predictions/`,
   `experiments/`, `reports/` and `pipeline/` are read-only from the backend's
   point of view. Backend outputs belong under `06_BACKEND/`.
3. **Serve the forecast, not the model.** All 30,490 x 28 predictions are already
   computed in `03_FORECASTS/final_forecast_28day_v3_diversity_blend.csv`
   (17 MB, loads instantly). Loading the LightGBM boosters and running the
   28-step recursive rollout takes minutes and is not a request-time operation.

## What is available to serve

| Data | Path | Size |
|---|---|---|
| 28-day forecast, per store-item | `03_FORECASTS/final_forecast_28day_v3_diversity_blend.csv` | 30,490 x 28 |
| Historical actuals | `data/raw/sales_train_evaluation.csv` | 30,490 x 1,941 |
| Calendar, events, SNAP | `data/raw/calendar.csv` | 1,969 days |
| Prices | `data/raw/sell_prices.csv` | weekly |
| Item/store hierarchy | id columns of `sales_train_evaluation.csv` | 3,049 items x 10 stores |
| Validation predictions (for accuracy displays) | `MY_RESEARCH_PAPER/reproduction/shipped_blend_w060_validation.csv` | 853,720 rows |

## Reading the model's own metadata

If the API needs to report what model produced a forecast:

```python
import json, pathlib
m = json.loads(pathlib.Path("02_MODEL/FROZEN_CHAMPION/CHAMPION_MANIFEST.json").read_text())
m["frozen_champion"]["primary_window_RMSE"]   # 2.0929...
```

## One accuracy caveat that must reach the UI

Accuracy depends entirely on the aggregation level:

| Level | Accuracy (1 − WAPE) |
|---|---|
| store-item-day (what is forecast) | ~28% |
| item across all stores, per day | ~71% |
| whole store, per day | ~93% |
| whole chain, per day | ~95% |

Showing the ~28% figure next to a chain-level total badly understates the system;
showing ~95% next to a single store-item badly overstates it. Whichever level a
screen aggregates to, quote the matching number.

Also: **no accuracy figure applies to the delivered forecast window** — no ground
truth exists for `d_1942–d_1969`. Any "accuracy" shown must be labelled as
measured on the `d_1914–d_1941` validation window.
