# Final forecast — the deliverable

| File | Shape |
|---|---|
| `final_forecast_28day.csv` | 30,490 rows × (id + F1..F28) — one row per store-item series |
| `submission_m5_format.csv` | 60,980 rows — the full M5 layout (validation + evaluation blocks) |

Covers **d_1942–d_1969 (2016-05-23 → 2016-06-19)**. Structure validated: no NaN,
no negatives, no duplicate ids, order matching `data/raw/sample_submission.csv`.

**No accuracy figure applies to this window** — no ground truth for it exists in
any file. The honest estimate of its quality is the validation result
(RMSE 2.1210 / MAE 1.0319) measured on d_1914–d_1941.

Backtest predictions for every experiment are in `predictions/validation/`.
