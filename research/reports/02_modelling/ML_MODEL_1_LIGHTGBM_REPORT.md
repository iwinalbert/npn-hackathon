# Model 1 — Global LightGBM

*Generated 2026-08-14 from executed run `model_01_lightgbm`.*

> **Terms used in this report.** **RMSE** (Root Mean Squared Error) measures how far predictions land from actual sales, counting big misses much more heavily than small ones — lower is better. **MAE** (Mean Absolute Error) is the plain average size of the miss. **WAPE** expresses total error as a share of total actual demand, which exposes a model that scores well simply by predicting near-zero everywhere. **Leakage** means letting information into the model that would not have existed at the moment the forecast was really made. **SNAP** is the US Supplemental Nutrition Assistance Program, a food-assistance benefit; the dataset records, per state per day, whether it was usable. **Intermittent demand** describes a product that sells on some days and records zero on many others.

## Objective

Find out whether a learned model beats simple arithmetic, and establish the reference that every later modelling idea must improve on.

## What we did

We trained a single **global** LightGBM model — one model for all 30,490 series, rather than 30,490 separate models. LightGBM is a gradient-boosted decision tree method: it builds many small trees in sequence, each one correcting the mistakes of the trees before it. A global model lets a sparse item borrow patterns (weekends, holidays, SNAP days) learned from thousands of other items, which a per-series model cannot do. It is also the only tractable option here: fitting and maintaining 30,490 separate models would be far slower for no obvious gain.

The feature set for this model deliberately **excludes** recency and listing features, so that Models 3 and 4 can measure exactly what those groups contribute.

## Data used

| | |
|---|---|
| Training rows | 12,805,800 |
| Training origins | 15 (each contributes 30,490 series x 28 days) |
| Features | 27 |
| Feature groups | A, B, E, F, G |
| Feature set | `base` — Calendar + Historical demand + Price + Hierarchy + Horizon |

Training origins are spaced 28 days apart, so their 28-day target blocks tile the history contiguously and no (series, day) target is counted twice.

## Model configuration

| Setting | Value |
|---|---|
| Model | LightGBM |
| Objective | regression (L2) |
| learning_rate | 0.05 |
| num_leaves | 128 |
| max_depth | -1 |
| min_data_in_leaf | 100 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| lambda_l2 | 1.0 |
| seed | 42 |
| Boosting rounds | 400 |
| Categorical features | 12 handled natively by LightGBM |
| Random seed | 42 |
| Training time | 65.1s |

> **No early stopping was used, deliberately.** Stopping when the validation score stops improving would let the validation window influence a training decision, and the resulting score would flatter itself. A fixed number of rounds keeps the held-out estimate honest.

## Validation design

Every model in this project is scored on exactly the same window, with the same metric code, so the comparisons between them are fair by construction.

| | |
|---|---|
| Forecast origin | d_1913 |
| Days predicted | d_1914 .. d_1941 |
| Dates predicted | 2016-04-25 .. 2016-05-22 |
| Horizon | 28 days |
| Series | 30,490 |
| Predictions scored | 853,720 |

The model sees no sales at all from the validation window. It is given only the calendar, event, SNAP and price information for those days, all of which are genuinely published in advance.

## Results (measured)

| Metric | Value |
|---|---|
| RMSE | **2.1467** |
| MAE | **1.0411** |
| WAPE | 0.7216 |
| Bias (mean predicted − mean actual) | -0.0590 |
| Predictions scored | 853,720 |

### Comparison against the previous step

| | RMSE | MAE |
|---|---|---|
| Model 0 — best naive (rolling mean 28) | 2.2430 | 1.0657 |
| **This model** | **2.1467** | **1.0411** |
| Change | -0.0963 | -0.0247 |

Measured verdict: this change **improved accuracy**.

## What the model learned

The strongest inputs are the recent-demand features. That is expected: a product's own recent sales level is by far the best available clue to its next 28 days. Section results are quantified in the final comparison report.

## Leakage checks

The foundation stage established the guarantee this model inherits, and it was verified by experiment rather than asserted: every sales value after the forecast origin was overwritten with an absurd number (9999), the features were rebuilt, and all 32 came back **bit-for-bit identical**. A companion check confirmed the target column did change, proving the corruption really reached the data.

In addition, the training-set builder refuses to run if any training row targets a day inside the validation window — that is an assertion in code, not a convention.

## Limitations

- Untuned hyperparameters — this is a reference point, not an optimised model.
- A single L2 objective treats a miss on a zero-sales day the same as a miss on a high-volume day, which does not match a target where 68% of historical rows are zero.
- Point forecasts only; no uncertainty interval is produced.

## Conclusion and next step

Learning clearly beats arithmetic. The next question is whether the loss function is right for this target, which Model 2 tests by changing the objective and nothing else.
