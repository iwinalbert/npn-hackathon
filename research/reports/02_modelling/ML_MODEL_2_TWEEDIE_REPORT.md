# Model 2 — LightGBM with a Tweedie Objective

*Generated 2026-08-14 from executed run `model_02_tweedie`.*

> **Terms used in this report.** **RMSE** (Root Mean Squared Error) measures how far predictions land from actual sales, counting big misses much more heavily than small ones — lower is better. **MAE** (Mean Absolute Error) is the plain average size of the miss. **WAPE** expresses total error as a share of total actual demand, which exposes a model that scores well simply by predicting near-zero everywhere. **Leakage** means letting information into the model that would not have existed at the moment the forecast was really made. **SNAP** is the US Supplemental Nutrition Assistance Program, a food-assistance benefit; the dataset records, per state per day, whether it was usable. **Intermittent demand** describes a product that sells on some days and records zero on many others.

## Objective

Test whether a loss function designed for zero-inflated, non-negative data fits this target better than ordinary squared error.

## What we did

**Tweedie** is a probability distribution for outcomes that are never negative and that pile up at exactly zero, with a long right tail above it — which is a fair description of daily unit sales here (68% of all historical rows are zero, and the maximum is 763). Using it as a LightGBM objective tells the model to expect that shape rather than a symmetric bell curve around the mean.

Everything else is held identical to Model 1 — same features, same hyperparameters, same training origins, same validation window — so any difference is attributable to the objective alone.

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
| Objective | tweedie (variance_power=1.1) |
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
| Training time | 106.5s |

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
| RMSE | **2.1256** |
| MAE | **1.0315** |
| WAPE | 0.7149 |
| Bias (mean predicted − mean actual) | -0.0795 |
| Predictions scored | 853,720 |

### Comparison against the previous step

| | RMSE | MAE |
|---|---|---|
| Model 1 — LightGBM, L2 objective | 2.1467 | 1.0411 |
| **This model** | **2.1256** | **1.0315** |
| Change | -0.0211 | -0.0096 |

Measured verdict: this change **improved accuracy**.

## Leakage checks

The foundation stage established the guarantee this model inherits, and it was verified by experiment rather than asserted: every sales value after the forecast origin was overwritten with an absurd number (9999), the features were rebuilt, and all 32 came back **bit-for-bit identical**. A companion check confirmed the target column did change, proving the corruption really reached the data.

In addition, the training-set builder refuses to run if any training row targets a day inside the validation window — that is an assertion in code, not a convention.

## Limitations

- The Tweedie variance power was fixed at 1.1 and not searched.
- Tweedie improves the fit to the target's shape; it does not add any new information about demand.

## Conclusion and next step

With the objective settled, the next experiments test whether the feature groups the EDA singled out — recency and listing — actually earn their place.
