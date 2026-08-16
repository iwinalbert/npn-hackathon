# Model 5 — Two-Stage Hurdle Model

*Generated 2026-08-14 from executed run `model_05_hurdle`.*

> **Terms used in this report.** **RMSE** (Root Mean Squared Error) measures how far predictions land from actual sales, counting big misses much more heavily than small ones — lower is better. **MAE** (Mean Absolute Error) is the plain average size of the miss. **WAPE** expresses total error as a share of total actual demand, which exposes a model that scores well simply by predicting near-zero everywhere. **Leakage** means letting information into the model that would not have existed at the moment the forecast was really made. **SNAP** is the US Supplemental Nutrition Assistance Program, a food-assistance benefit; the dataset records, per state per day, whether it was usable. **Intermittent demand** describes a product that sells on some days and records zero on many others.

## Objective

With most rows at zero, predicting whether a sale happens and predicting how big it is are arguably two different problems. A hurdle model asks them separately and multiplies the answers.

## What we did

**Stage 1** is a classifier estimating P(sales > 0) — the probability the item sells at all that day. **Stage 2** is a Poisson regressor estimating E[units | sales > 0], trained only on rows where a sale actually happened (5,123,355 rows). The final forecast is the two multiplied together.

A worked example: if Stage 1 says there is a 40% chance of selling, and Stage 2 says that when it does sell it typically moves 2.3 units, the forecast is 0.40 x 2.3 = 0.92 units.

Measured on the validation window: mean P(sale) = 0.4361, mean E[units | sale] = 2.3218.

## Data used

| | |
|---|---|
| Training rows | 12,805,800 |
| Training origins | 15 (each contributes 30,490 series x 28 days) |
| Features | 32 |
| Feature groups | A, B, C, D, E, F, G |
| Feature set | `base_recency_listing` — BASE + Recency + Listing (all 32 features) |

Training origins are spaced 28 days apart, so their 28-day target blocks tile the history contiguously and no (series, day) target is counted twice.

## Model configuration

| Setting | Value |
|---|---|
| Model | LightGBM two-stage hurdle |
| Objective | stage1=binary, stage2=poisson |
| Boosting rounds | — |
| Categorical features | 0 handled natively by LightGBM |
| Random seed | 42 |
| Training time | 177.0s |

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
| RMSE | **2.1267** |
| MAE | **1.0324** |
| WAPE | 0.7155 |
| Bias (mean predicted − mean actual) | -0.0721 |
| Predictions scored | 853,720 |

### Comparison against the previous step

| | RMSE | MAE |
|---|---|---|
| Model 4 — best single-stage model so far | 2.1210 | 1.0319 |
| **This model** | **2.1267** | **1.0324** |
| Change | +0.0057 | +0.0004 |

Measured verdict: this change **made accuracy worse**.

## What the model learned

The two stages behave sensibly on their own terms — the predicted probability is close to the observed positive rate, and the magnitude model produces plausible conditional volumes. The difficulty is that multiplying two separately-fitted estimates compounds the error in both, whereas a single Tweedie model is already fitting a distribution with a spike at zero and so is solving the same problem in one step.

## Leakage checks

The foundation stage established the guarantee this model inherits, and it was verified by experiment rather than asserted: every sales value after the forecast origin was overwritten with an absurd number (9999), the features were rebuilt, and all 32 came back **bit-for-bit identical**. A companion check confirmed the target column did change, proving the corruption really reached the data.

In addition, the training-set builder refuses to run if any training row targets a day inside the validation window — that is an assertion in code, not a convention.

## Limitations

- Stage 2 used a Poisson objective; gamma or log-normal alternatives were not explored.
- Neither stage was tuned.
- Two models must be trained and stored instead of one.

## Conclusion and next step

Measured result: the hurdle structure made accuracy worse relative to the best single-stage model. Complexity that does not pay for itself is not carried forward.
