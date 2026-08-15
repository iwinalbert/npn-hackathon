# M5 Retail Demand Forecasting — Problem Statement 11

**NPN AIA Hackathon · St. Joseph's College of Engineering**

Forecast daily unit sales for **30,490 Walmart store-item series**, 28 days ahead.

---

## 1. Purpose

Retail demand forecasting decides how much stock sits on a shelf. Forecast too
low and customers find gaps; too high and stock ties up cash or spoils. This
project builds a leakage-verified forecasting pipeline for that decision and,
just as importantly, establishes **how accurate such a forecast can honestly be**
on this data.

## 2. Problem statement

Predict units sold for every (product, store, day) combination over the next 28
days, given ~5.3 years of daily history. Two properties dominate the problem:

- **68% of all historical rows are zero** — most products do not sell every day
  in every store (*intermittent demand*).
- **No promotion field and no inventory field exist**, so a recorded zero could
  mean "nobody wanted it" or "it was not on the shelf". The data cannot tell
  them apart, and this project never pretends otherwise.

## 3. Dataset

Public M5 Forecasting — Accuracy data (Walmart), in `data/raw/` (**immutable**).

| | |
|---|---|
| Series | 30,490 (3,049 products × 10 stores) |
| Locations | 10 stores across California, Texas, Wisconsin |
| History | 1,941 days, 2011-01-29 → 2016-05-22, no gaps |
| Long-format rows | 59,181,090 |
| Total units sold | 66,927,173 |
| Zero-sales rows | 40,241,819 (68.00%) |
| Largest single-day sale | 763 units |

Known in advance for the forecast window: calendar, weekday, holidays/events,
SNAP flags (a US food-assistance benefit, flagged per state per day), and
`sell_price`. Not known: the sales themselves.

## 4. Forecasting objective

Predict **d_1942 – d_1969 (2016-05-23 → 2016-06-19)** for all 30,490 series.
Primary metric **RMSE**, secondary **MAE**.

## 5. Final model

**Global LightGBM with a Tweedie objective**, one model across all 30,490 series.

| | |
|---|---|
| Objective | `tweedie`, `variance_power = 1.1` |
| Features | 32, in 7 groups (calendar, historical demand, recency, listing, price, hierarchy, horizon) |
| Trees / leaves / learning rate | 400 / 128 / 0.05 |
| Training rows | 12,805,800 (15 origins × 30,490 series × 28 days) |
| Seed | 42, `deterministic=True` |
| Model file | `models/champion/model_04_tweedie_recency_listing.txt` |
| Retrained for the deliverable | `models/champion/model_07_final_forecast.txt` |

> **On naming.** The champion configuration appears under several experiment ids
> (`model_04_tweedie_recency_listing`, `opt_00_baseline_reproduce`,
> `model_06_tuned_primary`, `ablation_abl_7_full`). These are the *same*
> configuration re-run at different stages, and they reproduce the same score to
> every decimal — deliberate evidence that the pipeline is deterministic.
> `model_04...` is the canonical artefact; `model_07...` is that same
> configuration retrained with the origin moved to d_1941 to produce the
> delivered forecast.

## 6. Final validation metrics

Measured on 853,720 predictions (30,490 series × 28 days):

| Metric | Value |
|---|---|
| **RMSE** | **2.1210** |
| **MAE** | **1.0319** |
| WAPE | 0.7152 |
| Bias | −0.0704 |

Accuracy depends entirely on the level you aggregate to:

| Level | Accuracy (1 − WAPE) |
|---|---|
| Store-item-day (what is forecast) | 28.5% |
| Item across all stores, per day | 70.9% |
| Whole store, per day | 92.9% |
| Whole chain, per day | 94.5% |

The low bottom-level figure is expected: 54% of individual store-item-days are
zero, and errors are largely independent, so they cancel on aggregation. Use the
figure that matches the decision being made.

For reference: a naive "repeat the last 28-day average" scores RMSE 2.2430, and
predicting zero everywhere scores 3.9161.

## 7. Validation methodology

**Fixed-origin, direct 28-day backtest.**

| Block | Days | Dates |
|---|---|---|
| Training | d_1 … d_1913 | 2011-01-29 … 2016-04-24 |
| Validation | d_1914 … d_1941 | 2016-04-25 … 2016-05-22 |
| Final forecast | d_1942 … d_1969 | 2016-05-23 … 2016-06-19 |

Every history-derived feature is frozen at the forecast origin and held constant
across all 28 days; only the calendar and price vary per day, because only those
are genuinely published in advance.

**The leakage guarantee is proved, not asserted.** Every sales value after the
origin is overwritten with 9999, all features are rebuilt, and every one must
come back bit-for-bit identical. A companion check confirms the target *did*
change, so the test cannot pass vacuously. It caught a real float32 issue on its
first run, and it is re-run against every new feature builder rather than
inherited.

Robustness: the model was retrained and rescored on four separate 28-day windows.
Mean RMSE 2.1584, **standard deviation 0.033** — the noise scale against which
every claimed improvement in this project is judged.

## 8. Final forecast

```
predictions/final_forecast/final_forecast_28day.csv    30,490 rows x F1..F28
predictions/final_forecast/submission_m5_format.csv    60,980 rows (full M5 layout)
```

Validated: no NaN, no negatives, no duplicate ids, order matching
`sample_submission.csv`. **No accuracy figure can be quoted for this window** —
no ground truth for d_1942–d_1969 exists in any file. The validation result above
is the only honest estimate.

## 9. Repository structure

```
data/
  raw/            the 5 original CSVs — IMMUTABLE, never opened in write mode
  processed/      sales_long_full.parquet (59.2M rows) + build/audit reports
pipeline/         reusable source package — all file paths resolve via config.py
scripts/          one-off run scripts, grouped by stage, numbered chronologically
  01_foundation/  02_modelling/  03_benchmark_investigation/
  04_optimization/  05_diagnostics/  06_research_campaign/
experiments/
  registry/       71 JSON records, one per experiment (the ledger)
  artifacts/      result tables and diagnostics those runs produced
  EXPERIMENT_LEDGER.md   <- start here to navigate the experiments
models/
  champion/       the selected model + its retrain for the forecast
  experiments/    models from other experimental runs
predictions/
  final_forecast/ the deliverable
  validation/     backtest predictions, one file per experiment
reports/          25 PDF reports (+ markdown sources), filed by stage
docs/             problem statement, dataset guides, EDA, approach documents
```

## 10. How to reproduce the final pipeline

```bash
python -m pip install -r requirements.txt

python scripts/01_foundation/01_foundation_check.py      # 46 integrity + leakage checks
python scripts/02_modelling/03_run_models.py             # baselines through the hurdle model
python scripts/02_modelling/08_final_forecast.py         # retrain champion, write the forecast
```

Paths are resolved centrally in `pipeline/config.py`; scripts locate the project
root by walking up to the folder containing `pipeline/config.py`, so they work
from any working directory.

## 11. Historical experiments

All 71 are in `experiments/registry/`, indexed by
**[`experiments/EXPERIMENT_LEDGER.md`](experiments/EXPERIMENT_LEDGER.md)** — which
experiment, what it scored, which model and prediction files belong to it.

## 12. Important reports

| Report | Location |
|---|---|
| **Complete results and scorecard** | `reports/04_optimization/FINAL_ML_RESULTS_REPORT.pdf` |
| Full project narrative | `reports/02_modelling/FINAL_ML_PROJECT_REPORT.pdf` |
| Where the error actually is | `reports/05_diagnostics_and_research/ERROR_AUTOPSY_REPORT.pdf` |
| Why we stopped, and the ceiling | `reports/05_diagnostics_and_research/AUTONOMOUS_RESEARCH_CAMPAIGN_REPORT.pdf` |
| Leakage + validation design | `reports/01_foundation/ML_PIPELINE_FOUNDATION_REPORT.pdf` |
| Comparison with the team benchmark | `reports/03_benchmark_investigation/TEAM_FAIR_COMPARISON_REPORT.pdf` |

## 13. Known limitations

- Results come from one primary window; other windows differ by ±0.02–0.03 RMSE.
- Point forecasts only — no uncertainty intervals, which inventory decisions want.
- Stockouts and promotions are unobservable in this dataset; nothing here recovers
  them, and no feature claims to.
- The comparison against the team's reported benchmark (RMSE 2.0324 / MAE 1.0869)
  is **not** like-for-like — their validation window, horizon, series count and
  metric code are undocumented. Our MAE is better; their reported RMSE is lower.
- `zero_streak_length` duplicates `days_since_last_sale`, and `pre_listing`
  duplicates `price_is_missing`. Both redundancies were measured and reported
  rather than silently dropped.

## 14. Current status

**Complete.** 71 experiments run; the champion stands at RMSE 2.1210 / MAE 1.0319;
the 28-day forecast is generated and structurally validated.

The research campaign concluded that **RMSE < 2.0 is not achievable with the
information in this dataset**, on three independent lines of evidence: six
architecturally different models produce residuals correlated at 0.9897 (so
model-side improvement is exhausted); every feature family has been tested and
none helped; and 2.0 sits below what an oracle that already knows each series'
true 28-day mean achieves (1.9818). What would change the answer is not a better
model but **more information** — a promotions calendar, inventory records, or
footfall — none of which exists here.
