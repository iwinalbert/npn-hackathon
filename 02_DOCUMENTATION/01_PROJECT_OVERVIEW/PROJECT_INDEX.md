# Project Index

Where everything lives. For the narrative, see [README.md](../../README.md).

## The five things most people want

| I want… | Path |
|---|---|
| **The final 28-day forecast** | `predictions/final_forecast/final_forecast_28day.csv` |
| **The champion model** | `models/champion/model_04_tweedie_recency_listing.txt` |
| **The headline results** | `reports/04_optimization/FINAL_ML_RESULTS_REPORT.pdf` |
| **All 71 experiments** | `experiments/EXPERIMENT_LEDGER.md` |
| **Proof there is no leakage** | `reports/01_foundation/ML_PIPELINE_FOUNDATION_REPORT.pdf` |

## Final artefacts

| Artefact | Path |
|---|---|
| Selected model | `models/champion/model_04_tweedie_recency_listing.txt` |
| Model retrained for the forecast | `models/champion/model_07_final_forecast.txt` |
| Champion validation predictions | `predictions/validation/model_04_tweedie_recency_listing_validation.csv` |
| Final forecast (30,490 × F1–F28) | `predictions/final_forecast/final_forecast_28day.csv` |
| Full M5-format submission (60,980 rows) | `predictions/final_forecast/submission_m5_format.csv` |
| Final metrics + scorecard | `experiments/artifacts/final_scorecard.csv`, `final_selection.json` |
| Feature pipeline | `pipeline/features.py` (+ `features_v2.py`, `features_v3.py` for tested extensions) |
| Validation configuration | `pipeline/config.py`, `pipeline/backtest.py` |

## Source code (`pipeline/`)

| Module | Responsibility |
|---|---|
| `config.py` | **all paths and dataset constants — the single point of truth** |
| `data_loader.py` | raw CSVs → compact wide matrices (read-only) |
| `features.py` | the 32 champion features, groups A–G |
| `features_v2.py` | +14 tested candidates (rejected, Experiment #70 era) |
| `features_v3.py` | +4 year-over-year features (rejected, Experiment #71) |
| `backtest.py` | fixed-origin train / validation / future frame assembly |
| `metrics.py` | RMSE, MAE, WAPE, bias |
| `validation_checks.py` | correctness + empirical leakage corruption tests |
| `optimize.py` | shared experiment harness |
| `models.py` | training wrappers and naive baselines |
| `team_style.py` | per-target-day reconstruction used in the benchmark investigation |
| `charts.py`, `report_pdf.py` | figures and markdown→PDF rendering |

## Scripts (`scripts/`) — numbered chronologically

| Group | Scripts | What it covers |
|---|---|---|
| `01_foundation/` | 01–02 | integrity + leakage checks, foundation report |
| `02_modelling/` | 03–09 | baselines → hurdle, ablation, tuning, final forecast, reports |
| `03_benchmark_investigation/` | 10–15 | team reproduction, leakage probe, Tweedie probe, comparison reports |
| `04_optimization/` | 16–24 | features, high-volume, Tweedie sweep, recursive, ensemble, robustness, selection |
| `05_diagnostics/` | 25–26 | error autopsy and its report |
| `06_research_campaign/` | 27–31 | Experiments #69–71 and the campaign report |

## Reports (`reports/`)

| Stage | Contents |
|---|---|
| `01_foundation/` | pipeline foundation, leakage methodology |
| `02_modelling/` | Models 0–5, model comparison, full project report |
| `03_benchmark_investigation/` | team fair comparison, team approach vs ours |
| `04_optimization/` | baseline, features, high-volume, Tweedie, recursive, objectives, hurdle, ensemble, robustness, selection, **final results** |
| `05_diagnostics_and_research/` | error autopsy, Experiment #69, autonomous research campaign |
| `charts/` | 18 figures used across the reports |

Each report exists as both `.pdf` (deliverable) and `.md` (source).

## Documentation (`docs/`)

| Folder | Contents |
|---|---|
| `01_problem_statement/` | PS11 walkthrough; the other team's end-to-end approach doc |
| `02_dataset/` | dataset guides (`DATASET_SUMMARY.md`, `DATASET_EXPLAINED.*`), Cognizant references |
| `03_exploratory_analysis/` | first-pass exploration: 9 charts, summary CSVs, step JSONs |
| `04_eda/` | formal EDA: report, methodology, 26 charts, 9 stat dumps, 33 tables |
| `05_approach/` | planning documents and the final approach + supporting evidence |

## Data (`data/`)

| Path | Contents | Mutability |
|---|---|---|
| `data/raw/` | the 5 original competition CSVs | **immutable — never written to** |
| `data/processed/` | `sales_long_full.parquet` (59.2M rows), inspection sample, build/audit reports | build output, not regenerated |

## Experiments (`experiments/`)

- `EXPERIMENT_LEDGER.md` — the index; start here
- `registry/` — 71 JSON records (configuration, hyperparameters, metrics, leakage checks, decision)
- `artifacts/` — result tables and diagnostics (ablations, robustness, autopsy, scorecards)

## Entry points

```bash
python scripts/01_foundation/01_foundation_check.py    # verify integrity + leakage
python scripts/02_modelling/08_final_forecast.py       # regenerate the final forecast
```
