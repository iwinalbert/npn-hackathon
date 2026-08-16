# Champion model

| File | Role |
|---|---|
| `model_04_tweedie_recency_listing.txt` | The selected model. Global LightGBM, Tweedie(1.1), 32 features, 400 trees / 128 leaves, seed 42. Validation RMSE **2.1210**, MAE **1.0319**. |
| `model_07_final_forecast.txt` | The same configuration retrained with the forecast origin moved to d_1941, used to produce `predictions/final_forecast/`. |

The same configuration also appears in the registry as `opt_00_baseline_reproduce`,
`model_06_tuned_primary` and `ablation_abl_7_full`. Those are independent re-runs
that reproduce the score to every decimal — kept deliberately as evidence the
pipeline is deterministic. `model_04...` is the canonical artefact.

Models from other experimental runs are in `models/experiments/`.
