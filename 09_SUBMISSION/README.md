# 09_SUBMISSION — final deliverables (COPIES)

Every file here is a **copy**. No original was moved or deleted.

| File | Copy of | What it is |
|---|---|---|
| `final_forecast_28day_v3_diversity_blend.csv` | `predictions/final_forecast/` | **the forecast** — 30,490 series x F1..F28, `d_1942–d_1969` |
| `MY_RESEARCH_PAPER.pdf` | `MY_RESEARCH_PAPER/` | the research paper |
| `FINAL_MODEL_PERFORMANCE_REPORT.pdf` | `reports/` | model comparison, metrics, acceptance record |
| `USE_CASE_11_COMPLIANCE_AND_RESEARCH_REPORT.pdf` | `reports/` | Use Case 11 compliance matrix |

The forecast copy was verified SHA-256-identical to its source.

## What the model is

```
0.60 x Direct (38 features) + 0.40 x Recursive (32 features)
Validation (d_1914–d_1941): RMSE 2.0929   MAE 1.0395
STATUS: FROZEN
```

## ⚠ Deliberately NOT included: the M5-format submission

`predictions/final_forecast/submission_m5_format.csv` is **stale** — it was built
from the superseded `model_07` forecast, not from the frozen champion. Verified
by comparing its evaluation block against both forecasts:

```
== final_forecast_28day (model_07, OLD)  : True
== v3_diversity_blend (FROZEN champion)  : False
```

Including it would ship the wrong model's numbers under the champion's name.
Regenerating it requires running the models, which is out of scope for an
organisation task and needs explicit approval.
