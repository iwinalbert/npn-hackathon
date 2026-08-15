# 07_FRONTEND — workspace (empty, not yet built)

Frontend development happens **here**. Nothing has been implemented yet.

## Ground rules

1. The frontend talks to `06_BACKEND/`. It should not read `data/`, `models/`
   or `predictions/` directly.
2. The model is frozen — see `02_MODEL/MODEL_FREEZE.md`.

## Things the data will force you to handle

| Reality | Consequence for the UI |
|---|---|
| 68% of historical store-item-days are zero | Charts of a single series will look mostly flat at zero. Sparklines at item or store level read far better than at store-item level |
| Forecasts are continuous, actuals are integers | A forecast of 0.63 units is meaningful (an expected value), not a rounding error. Decide deliberately whether to round in the UI, and say which you are showing |
| Accuracy varies ~28% → ~95% by aggregation level | Never show a single global "accuracy" number. Show the one matching the current view's aggregation level |
| No ground truth exists for the forecast window | Do not render an "accuracy" or "error" panel against the delivered forecast. Any accuracy shown is from the `d_1914–d_1941` validation window and must be labelled as such |
| 30,490 series | Any "all series" table needs virtualisation or server-side paging |
| Point forecasts only, no intervals | Do not render confidence bands. The model does not produce them, and inventing them would misrepresent it |

## Useful groupings

3 states → 10 stores → 3 categories → 7 departments → 3,049 items →
30,490 store-item series. Aggregates are exact sums of the bottom-level forecast
(the forecast is coherent by construction), so a store total is just the sum of
its rows.
