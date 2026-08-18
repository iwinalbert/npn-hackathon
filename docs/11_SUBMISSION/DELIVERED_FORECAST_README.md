# 03_FORECASTS — the deliverable forecast

## The shipped forecast

```
final_forecast_28day_v3_diversity_blend.csv     30,490 rows x (id + F1..F28)
```

A byte-identical copy of
`predictions/final_forecast/final_forecast_28day_v3_diversity_blend.csv`,
produced by Experiment #78 from the FROZEN champion
(0.60 x direct 38f + 0.40 x recursive 32f) at forecast origin `d_1941`.

| Property | Value |
|---|---|
| Covers | `d_1942 … d_1969` (2016-05-23 → 2016-06-19) |
| Series | 30,490 store-item combinations |
| Structure | validated: no NaN, no negatives, no duplicate ids |
| Value range | 0.0035 … 146.70, mean 1.5333 |

**No accuracy figure applies to this file.** No ground truth exists for
`d_1942–d_1969` in any dataset. The honest estimate of its quality is the
validation result on `d_1914–d_1941`: RMSE 2.0929 / MAE 1.0395.

## Other forecasts that exist (and are NOT this one)

Left in place under `predictions/final_forecast/`, superseded but preserved:

| File | Model | Status |
|---|---|---|
| `final_forecast_28day.csv` | model_07, 32 features, RMSE 2.1210 | superseded |
| `final_forecast_28day_v2_shape_cycle.csv` | model_10, 38 features, RMSE 2.1157 | superseded |
| `submission_m5_format.csv` | built from `final_forecast_28day.csv` | **STALE — see below** |

## ⚠ Known issue: the M5-format submission is stale

`predictions/final_forecast/submission_m5_format.csv` was generated from the
**superseded** `model_07` forecast, not from the frozen champion. This was
verified during the reorganisation by comparing its evaluation block against both
forecasts:

```
submission evaluation-block == final_forecast_28day (model_07, OLD)  : True
submission evaluation-block == v3_diversity_blend (FROZEN champion)  : False
```

It was **not** regenerated here, because doing so requires running the models and
this was an organisation task. It is therefore excluded from `docs/11_SUBMISSION/`.
If an M5-format submission of the frozen champion is needed, regenerating it is a
deliberate modelling action requiring approval.
