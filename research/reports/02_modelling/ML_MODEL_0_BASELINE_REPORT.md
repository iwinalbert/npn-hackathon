# Model 0 — Naive Baselines

*Generated 2026-08-14 from executed runs.*

> **Terms used in this report.** **RMSE** (Root Mean Squared Error) measures how far predictions land from actual sales, counting big misses much more heavily than small ones — lower is better. **MAE** (Mean Absolute Error) is the plain average size of the miss. **WAPE** expresses total error as a share of total actual demand, which exposes a model that scores well simply by predicting near-zero everywhere. **Leakage** means letting information into the model that would not have existed at the moment the forecast was really made. **SNAP** is the US Supplemental Nutrition Assistance Program, a food-assistance benefit; the dataset records, per state per day, whether it was usable. **Intermittent demand** describes a product that sells on some days and records zero on many others.

## Objective

Establish how hard this forecasting problem actually is before any machine learning is involved. Without a baseline, a machine-learning RMSE is just a number — there is no way to tell whether it represents real skill or whether repeating last week's sales would have done just as well.

## What we did

Four rules were applied to all 30,490 series. **None of them fit any parameters** — there is no training step, no data is learned from. Each one simply copies some piece of the recent past forward across all 28 days.

## Results (measured)

| Baseline rule | RMSE | MAE | WAPE |
|---|---|---|---|
| Mean of the last 28 days, repeated **(best)** | 2.2430 | 1.0657 | 0.7386 |
| Mean of the last 7 days, repeated | 2.2487 | 1.0683 | 0.7404 |
| Seasonal naive — repeat the most recent same weekday | 2.6769 | 1.2440 | 0.8622 |
| Last value — repeat the origin day's sales for 28 days | 2.8936 | 1.3730 | 0.9516 |

The strongest naive rule is **Mean of the last 28 days, repeated**, at RMSE 2.2430 and MAE 1.0657.

Two things are worth noticing. First, averaging beats copying: both rolling-mean rules clearly outperform seasonal-naive and last-value. On a target where most days are zero and individual days are noisy, a smoothed recent level is a better guess than any single recent day. Second, the bar this sets is genuinely high — any learned model has to beat RMSE 2.2430 before it has demonstrated it is worth its complexity at all.

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

## Limitations

- These rules cannot react to anything: not a holiday, not a SNAP day, not a price change, not a weekend.
- They apply one number to all 28 days, so they cannot express that a Saturday sells more than a Tuesday.
- No uncertainty estimate is produced.

## Conclusion and next step

The problem has a meaningful floor: RMSE 2.2430 is achievable with arithmetic alone. Next we train a single global LightGBM model on engineered features and check whether learning actually buys anything over this.
