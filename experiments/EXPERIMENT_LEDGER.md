# Experiment Ledger

All 86 experiments, in chronological order. Every row links to its JSON record in `experiments/registry/`, which contains the full configuration, hyperparameters, leakage checks and conclusion for that run.

**How to look something up**

- *What was Experiment 69?* — see Stage 5 below, or open `experiments/registry/exp_69_pre_origin_per_series_bias_correction.json`
- *What files belong to an experiment?* — the `model` and `predictions` columns give the paths
- *Why was it accepted or rejected?* — the `notes` and `decision` fields in the JSON, and the stage report in `reports/`
- *What was the final conclusion?* — `reports/05_diagnostics_and_research/AUTONOMOUS_RESEARCH_CAMPAIGN_REPORT.pdf`

Validation window for every scored run: **d_1914–d_1941** (2016-04-25 → 2016-05-22), 30,490 series × 28 days = 853,720 predictions. Rows marked *inner* were scored on d_1886–d_1913 and are tuning runs, not comparable to the primary figures.

## Stage 1 — Foundation & baselines  (9 runs)

| Experiment | RMSE | MAE | Window | Model file | Predictions |
|---|---|---|---|---|---|
| `model_00_baseline_last_value` | 2.8936 | 1.3730 | primary | — | — |
| `model_00_baseline_rolling_mean_28` | 2.2430 | 1.0657 | primary | — | — |
| `model_00_baseline_rolling_mean_7` | 2.2487 | 1.0683 | primary | — | — |
| `model_00_baseline_seasonal_naive` | 2.6769 | 1.2440 | primary | — | predictions\model_00_seasonal_naive_validation.csv |
| `model_01_lightgbm` | 2.1467 | 1.0411 | primary | models\model_01_lightgbm.txt | predictions\model_01_lightgbm_validation.csv |
| `model_02_tweedie` | 2.1256 | 1.0315 | primary | models\model_02_tweedie.txt | predictions\model_02_tweedie_validation.csv |
| `model_03_tweedie_recency` | 2.1258 | 1.0320 | primary | models\model_03_tweedie_recency.txt | predictions\model_03_tweedie_recency_validation.csv |
| `model_04_tweedie_recency_listing` | 2.1210 | 1.0319 | primary | models\model_04_tweedie_recency_listing.txt | predictions\model_04_tweedie_recency_listing_validation.csv |
| `model_05_hurdle` | 2.1267 | 1.0324 | primary | models\model_05_hurdle_stage1.txt + models\model_05_hurdle_stage2.txt | predictions\model_05_hurdle_validation.csv |

## Stage 2 — Ablation & tuning  (15 runs)

| Experiment | RMSE | MAE | Window | Model file | Predictions |
|---|---|---|---|---|---|
| `ablation_abl_1_calendar` | 3.6393 | 1.6591 | primary | — | — |
| `ablation_abl_2_calendar_demand` | 2.1584 | 1.0500 | primary | — | — |
| `ablation_abl_3_plus_recency` | 2.1614 | 1.0549 | primary | — | — |
| `ablation_abl_4_plus_price` | 2.1418 | 1.0461 | primary | — | — |
| `ablation_abl_5_plus_listing` | 2.1537 | 1.0494 | primary | — | — |
| `ablation_abl_6_plus_hierarchy` | 2.1374 | 1.0324 | primary | — | — |
| `ablation_abl_7_full` | 2.1210 | 1.0319 | primary | — | — |
| `model_06_tuned_primary` | 2.1210 | 1.0319 | primary | models\model_06_tuned_primary.txt | predictions\model_06_tuned_primary_validation.csv |
| `model_06_window_christmas_2015` | 2.1851 | 0.9231 | — | — | — |
| `model_06_window_summer_2015` | 2.1405 | 0.9746 | — | — | — |
| `model_07_final_forecast` | — | — | — | models\model_07_final_forecast.txt | predictions\final_forecast_28day.csv |
| `tune_inner_A_current_settings` | 2.0899 | 1.0173 | inner | — | — |
| `tune_inner_B_more_rounds` | 2.1127 | 1.0200 | inner | — | — |
| `tune_inner_C_rounds_and_leaves` | 2.0977 | 1.0113 | inner | — | — |
| `tune_inner_D_more_history` | 2.0955 | 1.0074 | inner | — | — |

## Stage 3 — Benchmark investigation  (6 runs)

| Experiment | RMSE | MAE | Window | Model file | Predictions |
|---|---|---|---|---|---|
| `diagnostic_leakage_probe_DO_NOT_USE` | 1.9165 | 0.9754 | primary | — | — |
| `model_08_team_style_reproduction` | 2.1835 | 1.0498 | primary | models\model_08_team_style_reproduction.txt | predictions\model_08_team_style_validation.csv |
| `model_09_tweedie_power_1_5` | 2.1263 | 1.0289 | primary | models\model_09_tweedie_power_1_5.txt | predictions\model_09_tweedie_power_1_5_validation.csv |
| `probe_tweedie_power_1_1` | 2.0899 | 1.0173 | inner | — | — |
| `probe_tweedie_power_1_3` | 2.0793 | 1.0124 | inner | — | — |
| `probe_tweedie_power_1_5` | 2.0766 | 1.0087 | inner | — | — |

## Stage 4 — Optimization campaign  (38 runs)

| Experiment | RMSE | MAE | Window | Model file | Predictions |
|---|---|---|---|---|---|
| `opt_00_baseline_reproduce` | 2.1210 | 1.0319 | primary | — | predictions\opt_00_baseline_reproduce_validation.csv |
| `opt_02_v2_A_demand` | 2.1233 | 1.0312 | primary | — | predictions\opt_02_v2_A_demand_validation.csv |
| `opt_02_v2_B_calendar` | 2.1327 | 1.0326 | primary | — | predictions\opt_02_v2_B_calendar_validation.csv |
| `opt_02_v2_C_price` | 2.1281 | 1.0307 | primary | — | predictions\opt_02_v2_C_price_validation.csv |
| `opt_02_v2_D_interactions` | 2.1256 | 1.0314 | primary | — | predictions\opt_02_v2_D_interactions_validation.csv |
| `opt_02_v2_all` | 2.1320 | 1.0313 | primary | — | predictions\opt_02_v2_all_validation.csv |
| `opt_03_highvol_calibration` | 2.1210 | 1.0319 | primary | — | — |
| `opt_03_volume_weight_cap3` | 2.1376 | 1.0336 | primary | — | — |
| `opt_03_volume_weight_cap5` | 2.1371 | 1.0335 | primary | — | — |
| `opt_04_power_1_1` | 2.0899 | 1.0173 | inner | — | — |
| `opt_04_power_1_2` | 2.0845 | 1.0147 | inner | — | — |
| `opt_04_power_1_3` | 2.0793 | 1.0124 | inner | — | — |
| `opt_04_power_1_4` | 2.0812 | 1.0120 | inner | — | — |
| `opt_04_power_1_5` | 2.0766 | 1.0087 | inner | — | — |
| `opt_04_power_1_6` | 2.0825 | 1.0090 | inner | — | — |
| `opt_04_power_1_7` | 2.0923 | 1.0092 | inner | — | — |
| `opt_04_power_1_8` | 2.1040 | 1.0096 | inner | — | — |
| `opt_04b_power_1_5_primary` | 2.1263 | 1.0289 | primary | models\opt_04b_power_1_5_primary.txt | predictions\opt_04b_power_1_5_primary_validation.csv |
| `opt_05_recursive` | 2.1182 | 1.0717 | primary | models\opt_05_recursive_onestep.txt | predictions\opt_05_recursive_validation.csv |
| `opt_06_obj_l1` | 2.2432 | 0.9591 | primary | — | predictions\opt_06_obj_l1_validation.csv |
| `opt_06_obj_l2` | 2.1351 | 1.0388 | primary | — | predictions\opt_06_obj_l2_validation.csv |
| `opt_06_obj_poisson` | 2.1379 | 1.0350 | primary | — | predictions\opt_06_obj_poisson_validation.csv |
| `opt_06_obj_tweedie_1_1` | 2.1210 | 1.0319 | primary | — | predictions\opt_06_obj_tweedie_1_1_validation.csv |
| `opt_07_hurdle_v2` | 2.1241 | 1.0300 | primary | — | — |
| `opt_07_hurdle_v2_calibrated` | 2.1822 | 1.0195 | primary | — | — |
| `opt_08_ensemble_tweedie_l1` | 2.1272 | 1.0128 | primary | — | — |
| `opt_09_robust_autumn_2015_l1` | 2.3087 | 0.9251 | — | — | — |
| `opt_09_robust_autumn_2015_tweedie_1_1` | 2.1869 | 0.9977 | — | — | — |
| `opt_09_robust_autumn_2015_tweedie_1_5` | 2.1733 | 0.9929 | — | — | — |
| `opt_09_robust_christmas_2015_l1` | 2.3005 | 0.8503 | — | — | — |
| `opt_09_robust_christmas_2015_tweedie_1_1` | 2.1851 | 0.9231 | — | — | — |
| `opt_09_robust_christmas_2015_tweedie_1_5` | 2.1731 | 0.9166 | — | — | — |
| `opt_09_robust_primary_spring_2016_l1` | 2.2432 | 0.9591 | primary | — | — |
| `opt_09_robust_primary_spring_2016_tweedie_1_1` | 2.1210 | 1.0319 | primary | — | — |
| `opt_09_robust_primary_spring_2016_tweedie_1_5` | 2.1263 | 1.0289 | primary | — | — |
| `opt_09_robust_summer_2015_l1` | 2.2948 | 0.8926 | — | — | — |
| `opt_09_robust_summer_2015_tweedie_1_1` | 2.1405 | 0.9746 | — | — | — |
| `opt_09_robust_summer_2015_tweedie_1_5` | 2.1573 | 0.9683 | — | — | — |

## Stage 5 — Autonomous research  (3 runs)

| Experiment | RMSE | MAE | Window | Model file | Predictions |
|---|---|---|---|---|---|
| `exp_69_pre_origin_per_series_bias_correction` | 2.1286 | 1.0333 | primary | — | — |
| `exp_70_variance_reduction_ensemble` | 2.1261 | 1.0290 | primary | — | — |
| `exp_71_year_over_year_features` | 2.1564 | 1.0344 | primary | — | — |

## Stage 6 — Shape features and architectural diversity  (8 runs)

| Experiment | RMSE | MAE | Window | Model file | Predictions |
|---|---|---|---|---|---|
| `exp_72_per_series_shape_features` | 2.1163 | 1.0299 | primary | — | predictions\validation\exp_72_shape_validation.csv |
| `exp_73_shape_feature_validation` | 2.1148 | 1.0290 | primary | — | — |
| `exp_74_shape_reproduction_and_extension` | 2.1157 | 1.0287 | primary | models\experiments\exp_74_shape_champion_primary.txt | predictions\validation\exp_74_new_champion_validation.csv |
| `exp_75_new_champion_final_forecast` | — | — | — | models\champion\model_10_shape_cycle_final_forecast.txt | predictions\final_forecast\final_forecast_28day_v2_shape_cycle.csv |
| `exp_76_architectural_diversity_blend` | 2.0920 | 1.0441 | primary | — | predictions\validation\exp_76_diversity_blend_validation.csv |
| `exp_77_recursive_member_upgrade` | 2.0915 | 1.0433 | primary | — | — |
| `exp_78_blend_final_forecast` | — | — | — | models\champion\model_11_blend_direct_final_forecast.txt + model_12_blend_recursive_shape_final.txt | predictions\final_forecast\final_forecast_28day_v3_diversity_blend.csv |
| `exp_79_upgrade_seed_check` | — | — | autumn + christmas | — | — |

**Stage 6 notes.** #72 measured a real but sub-threshold shape effect and was
recorded REJECT on magnitude; #73 re-tested it for consistency (4/4 windows,
3/3 seeds) and accepted it; #74 reproduced that independently and added two more
cyclical axes, giving the 38-feature champion. #76 blended that champion with the
Phase-5 recursive model — a negative control attributed -0.0247 of the -0.0291
gain to architecture and only -0.0044 to averaging, which is why #70's ensemble
of six mutually-correlated direct models had failed. #77 gave the recursive
member the champion's six shape features (accepted, D1 4/4, D2 -0.0042, D3 3/4)
and selected the blend weight w=0.60 on an inner window. #79 confirmed the #77
upgrade is seed-stable (6/6 cells).

**Shipped configuration.** `0.60 x direct(38 features) + 0.40 x recursive(32
features)`. Across four evaluation windows: mean dRMSE **-0.0242** and mean dMAE
**+0.0186** against the direct champion alone; primary window RMSE **2.0929** /
MAE **1.0395**. The MAE cost is a deliberate, disclosed trade-off — the w-frontier
is recorded in `exp77_operating_point.csv`.

**Rejected before training, on measured evidence** (`exp76_headroom_diagnostic.json`,
`segmentation_diagnostic.json`, and the checks in Stage 6 scripts): cross-store /
cross-item features (joint oracle upper bound -0.0055), pre-launch row exclusion
(0.48% of training rows at the primary origin), ghost-stockout filtering (0.1997%
of cells), per-category specialisation (oracle -0.0025), per-horizon
specialisation (oracle -0.0081), and a third near-duplicate ensemble member
(lost 3 of 4 windows).

## Stage 7 — Use Case 11 compliance branch  (7 runs)

| Experiment | RMSE | MAE | Window | Model file | Predictions |
|---|---|---|---|---|---|
| `exp_80_item_level_reconciliation_probe` | 2.0517 | 1.0192 | inner | — | — |
| `exp_80b_hierarchy_level_sweep` | 2.0517 | 1.0192 | inner | — | — |
| `exp_80c_level_vs_crossstore` | 2.0624 | 1.0297 | inner | — | — |
| `exp_81_reconciliation_fixed_alpha` | 2.0890 | 1.0384 | primary | — | — |
| `exp_82_reconciliation_adaptive_alpha` | 2.0890 | 1.0384 | primary | — | — |
| `exp_83_covariate_audit` | — | — | primary | — | — |
| `exp_84_intermittency_audit` | — | — | primary | — | — |

**Stage 7 notes.** An audit of the shipped system against every clause of Use
Case 11, then investigation of the one requirement it did not satisfy —
hierarchical forecasting. The champion was reproduced from scratch on all four
evaluation windows and matched its recorded scores to four decimals every time
(2.0929 / 2.1547 / 2.1074 / 2.1496).

*Rejected before training, on exact oracle bounds* (`uc11_hierarchy_levels.csv`):
reconciliation at every level above item — knowing the TRUE chain total is worth
−0.0007 RMSE and the true store × dept total −0.0221. Promotion/discount
covariates (`uc11_covariate_audit.json`): a perfect per-discount-bin correction
is worth −0.0002. Croston / SBA / TSB (`uc11_intermittency_audit.json`): beaten
by the champion in every Syntetos-Boylan regime, with a regime-specialisation
oracle of −0.0008.

*Trained and rejected on a pre-registered criterion*: item-level middle-out
reconciliation, under two protocols. Both improved mean RMSE, mean MAE and mean
high-volume RMSE at once (#81 DEMEANED −0.0088 / −0.0009 / −0.0352; #82 DEMEANED
−0.0093 / −0.0007 / −0.0406) and both won 3 of 4 windows, but both failed the
MECHANISM criterion: the item-level model beats the champion's own bottom-up sum
on only 2 of 4 windows. #82 tested the deployable fix — α selected on the 28 days
preceding each origin — and found it moves α the WRONG way into the one window
where the item model collapses (α 0.55→0.60, loss +0.0158→+0.0198). The item
model's advantage does not persist across consecutive windows, so its benefit is
not knowable at forecast time.

This converges with Experiment #76, which measured a joint oracle of −0.0055 for
cross-store/cross-item *features*: two independent attacks on the same
information channel reach the same wall.

**Verdict: KEEP CHAMPION.** The shipped `0.60 x direct(38f) + 0.40 x
recursive(32f)` blend stands unchanged at RMSE 2.0929 / MAE 1.0395.

---

*Generated from `experiments/registry/`. Four groups of records share byte-identical model or prediction files — these are independent reproductions of the same configuration and are kept deliberately as evidence that the pipeline is deterministic.*