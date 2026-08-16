# Final Model Selection

*Phase 11 — choosing on measured evidence against a rule fixed in advance. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## The rule, set before the results were seen

1. leakage-safe (a hard gate, not a ranking factor)
2. lowest RMSE on the primary window
3. MAE as tie-break — and as a veto if a trivial RMSE gain costs a lot of MAE
4. robust across windows
5. reasonable training time
6. explainable
7. novelty only if experimentally supported

## Every candidate

![Scorecard](charts/final_scorecard.png)

| Model | RMSE | MAE | ΔRMSE | ΔMAE |
|---|---|---|---|---|
| opt_05_recursive | 2.1182 | 1.0717 | -0.0029 | +0.0398 |
| model_04_tweedie_recency_listing | 2.1210 | 1.0319 | +0.0000 | +0.0000 |
| opt_00_baseline_reproduce | 2.1210 | 1.0319 | +0.0000 | +0.0000 |
| model_06_tuned_primary | 2.1210 | 1.0319 | +0.0000 | +0.0000 |
| opt_06_obj_tweedie_1_1 | 2.1210 | 1.0319 | +0.0000 | +0.0000 |
| opt_03_highvol_calibration | 2.1210 | 1.0319 | +0.0000 | -0.0000 |
| opt_02_v2_A_demand | 2.1233 | 1.0312 | +0.0022 | -0.0007 |
| opt_07_hurdle_v2 | 2.1241 | 1.0300 | +0.0031 | -0.0020 |
| model_02_tweedie | 2.1256 | 1.0315 | +0.0045 | -0.0005 |
| opt_02_v2_D_interactions | 2.1256 | 1.0314 | +0.0045 | -0.0005 |
| model_03_tweedie_recency | 2.1258 | 1.0320 | +0.0048 | +0.0001 |
| opt_04b_power_1_5_primary | 2.1263 | 1.0289 | +0.0053 | -0.0030 |
| model_09_tweedie_power_1_5 | 2.1263 | 1.0289 | +0.0053 | -0.0030 |
| model_05_hurdle | 2.1267 | 1.0324 | +0.0057 | +0.0004 |
| opt_08_ensemble_tweedie_l1 | 2.1272 | 1.0128 | +0.0062 | -0.0191 |
| opt_02_v2_C_price | 2.1281 | 1.0307 | +0.0071 | -0.0012 |
| opt_02_v2_all | 2.1320 | 1.0313 | +0.0110 | -0.0007 |
| opt_02_v2_B_calendar | 2.1327 | 1.0326 | +0.0116 | +0.0007 |
| opt_06_obj_l2 | 2.1351 | 1.0388 | +0.0141 | +0.0069 |
| opt_03_volume_weight_cap5 | 2.1371 | 1.0335 | +0.0161 | +0.0016 |
| opt_03_volume_weight_cap3 | 2.1376 | 1.0336 | +0.0165 | +0.0016 |
| opt_06_obj_poisson | 2.1379 | 1.0350 | +0.0169 | +0.0031 |
| model_01_lightgbm | 2.1467 | 1.0411 | +0.0256 | +0.0092 |
| opt_07_hurdle_v2_calibrated | 2.1822 | 1.0195 | +0.0612 | -0.0124 |
| model_08_team_style_reproduction | 2.1835 | 1.0498 | +0.0625 | +0.0179 |
| model_00_baseline_rolling_mean_28 | 2.2430 | 1.0657 | +0.1219 | +0.0338 |
| opt_06_obj_l1 | 2.2432 | 0.9591 | +0.1221 | -0.0728 |
| model_00_baseline_rolling_mean_7 | 2.2487 | 1.0683 | +0.1276 | +0.0364 |
| model_00_baseline_seasonal_naive | 2.6769 | 1.2440 | +0.5559 | +0.2120 |
| model_00_baseline_last_value | 2.8936 | 1.3730 | +0.7725 | +0.3411 |

## Decision

**Selected: `opt_00_baseline_reproduce`** — RMSE 2.1210, MAE 1.0319.

The lowest-RMSE candidate (opt_05_recursive) improves RMSE by only 0.0029, which is inside the +/-0.013 window-to-window noise we measured, while costing +0.0398 MAE. That is not a real improvement, so the incumbent is retained.

## Final forecast

- File: `predictions\final_forecast_28day.csv` — existing file verified, not regenerated (selected configuration unchanged)
- Mean forecast: 1.48287 units per series per day
- 30,490 rows, columns F1–F28, no NaN, no negatives, ids and order matching `sample_submission.csv`

> No accuracy figure can be quoted for the forecast window itself (d_1942–d_1969) — no ground truth for it exists in any file. The validation result above is the only honest estimate.
