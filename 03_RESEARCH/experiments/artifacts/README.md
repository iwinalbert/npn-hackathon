# Experiment artifacts

Result tables and diagnostics produced by the runs in `../registry/`. Kept as a
flat directory because scripts write here via `config.ARTIFACTS_DIR`; grouping
into subfolders would break the write paths on re-run.

| Prefix / file | Produced by |
|---|---|
| `foundation_checks.json`, `feature_summary.csv`, `sample_features.csv` | Stage 1 foundation |
| `ablation_results.csv`, `inner_window_tuning.csv`, `multi_window_results.csv` | Stage 2 ablation & tuning |
| `error_analysis.json`, `feature_importance.csv` | Stage 2 analysis |
| `final_selection.json`, `final_scorecard.csv`, `final_forecast_summary.json` | final model selection |
| `team_*`, `leakage_probe.json`, `tweedie_power_*` | Stage 3 benchmark investigation |
| `phase2_… phase9_*` | Stage 4 optimization campaign |
| `error_autopsy.json`, `autopsy_worst_200_rows.csv` | Stage 5 error autopsy |
| `exp69_*`, `exp70_*`, `exp71_*` | Experiments #69–71 |
| `exp72_*` … `exp79_*` | Stage 6 shape features and the diversity blend |
| `uc11_*` | Stage 7 Use Case 11 compliance branch (Experiments #80–#84) |
