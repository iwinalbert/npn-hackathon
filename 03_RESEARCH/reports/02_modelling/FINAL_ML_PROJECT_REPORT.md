# Final ML Project Report

*Generated 2026-08-14. Every quantitative claim in this report comes from an experiment that actually ran; the underlying records are in `experiments/`.*

> **Terms used in this report.** **RMSE** (Root Mean Squared Error) measures how far predictions land from actual sales, counting big misses much more heavily than small ones — lower is better. **MAE** (Mean Absolute Error) is the plain average size of the miss. **WAPE** expresses total error as a share of total actual demand, which exposes a model that scores well simply by predicting near-zero everywhere. **Leakage** means letting information into the model that would not have existed at the moment the forecast was really made. **SNAP** is the US Supplemental Nutrition Assistance Program, a food-assistance benefit; the dataset records, per state per day, whether it was usable. **Intermittent demand** describes a product that sells on some days and records zero on many others.

> **How to read the labels.** **FACT** = measured or directly verified from the data. **INTERPRETATION** = our reading of a fact, which another analyst could reasonably dispute. **HYPOTHESIS** = untested.

---

## 1. The problem

Forecast daily unit sales for the next 28 days, for 30,490 store-item combinations (3,049 products x 10 Walmart stores across California, Texas and Wisconsin). The forecast window is **d_1942 to d_1969, 2016-05-23 to 2016-06-19**. No sales for those days exist in any file.

Getting this wrong is expensive in both directions: forecast too low and shelves run empty, too high and stock sits and spoils.

## 2. The dataset

**FACT** — verified directly against the raw files:

| Property | Value |
|---|---|
| Store-item series | 30,490 |
| Days of history | 1,941 (2011-01-29 to 2016-05-22, no gaps) |
| Long-format rows | 59,181,090 |
| Total units sold | 66,927,173 |
| Zero-sales rows | 40,241,819 (68.00%) |
| Largest single-day sale | 763 units |
| Rows with no price on record | 20.78% |

## 3. What the EDA found

The prior EDA stage established several things that shaped this build (all **FACT**): 68% of rows are zero; weekend sales run 31.1% above weekdays; SNAP days lift sales 12.7% overall and 17.3% within FOODS; named holidays move in opposite directions (Christmas −99.95%, Labor Day +27.5%); and the probability of a sale falls from 65.2% to 0.6% as a dry spell lengthens.

## 4. Why this is hard

- **Intermittent demand.** Most series do not sell every day, so there is no smooth curve to extrapolate.
- **Scale.** 30,490 series must be forecast at once, ranging from 130 units a day to fewer than 20 units in five years.
- **Zero is ambiguous.** A zero can mean nobody bought it, or it was not stocked, or it was out of stock. The dataset has no inventory field, so these cannot be told apart. We never pretended otherwise.
- **No promotion data.** A real driver of retail spikes is simply absent from the files.

## 5. Data preparation

**Nothing in `raw_dataset/` or `processed_dataset/` was modified.** The pipeline reads the raw CSVs read-only into compact matrices (30,490 x 1,941 sales, 118 MB; 30,490 x 282 prices, 34 MB) rather than materialising the 59-million-row long table, which would not fit comfortably in the ~5.7 GB of free memory on this machine.

Sales were **not** smoothed, zeros were **not** removed or converted to missing, no stockout was inferred, no promotion label was invented, and missing prices were left missing.

## 6. Feature engineering

32 features in seven groups. The organising principle is that at a fixed forecast origin T, a feature is either built from history up to T and held constant across all 28 days, or it is genuinely known in advance for each target day.

| Group | Features | Kind |
|---|---|---|
| A Calendar | weekday, month, year, weekend, event name/type x2, SNAP | known in advance |
| B Historical demand | lag 1/7/14/28, rolling mean & std over 7/28 days | origin-relative |
| C Recency | days_since_last_sale, zero_streak_length, days_since_first_sale | origin-relative |
| D Listing | days_since_first_listing, pre_listing | mixed |
| E Price | sell_price, recent average, price relative to average, price missing | known in advance |
| F Hierarchy | item, department, category, store, state | static |
| G Horizon | how many days ahead this prediction is | known |

## 7. Leakage prevention

**FACT.** The guarantee is tested, not asserted. Every sales value after the origin was overwritten with 9999, all features rebuilt, and all 32 came back bit-for-bit identical. A counter-check confirmed the target column did change, so the test was not vacuous. The training builder additionally refuses to run if any training row targets a day inside the validation window.

This test earned its keep: on its first run it flagged `rolling_std_28`. Investigation showed the inputs were byte-identical and the difference was float32 rounding (5.1e-07 relative) caused by differing memory layouts. Rather than loosen the test, the root cause was fixed — C-contiguous storage and float64 accumulation — and exact equality now holds.

## 8. Backtesting

| Block | Days | Dates |
|---|---|---|
| Training | d_1 .. d_1913 | 2011-01-29 .. 2016-04-24 |
| Validation | d_1914 .. d_1941 | 2016-04-25 .. 2016-05-22 |
| Final forecast | d_1942 .. d_1969 | 2016-05-23 .. 2016-06-19 |

Random train/test splitting would be wrong here: it would let the model learn from May while being tested on April. Time-series validation must cut on time.

## 9-14. What each experiment measured

| Step | What changed | RMSE | MAE | Verdict |
|---|---|---|---|---|
| Model 0 (rolling mean 28) | no learning at all | 2.2430 | 1.0657 | reference |
| Model 1 | global LightGBM, L2 objective | 2.1467 | 1.0411 | improved accuracy |
| Model 2 | objective -> Tweedie | 2.1256 | 1.0315 | improved accuracy |
| Model 3 | + recency features | 2.1258 | 1.0320 | made no meaningful difference |
| Model 4 | + listing features | 2.1210 | 1.0319 | made no meaningful difference |
| Model 5 | two-stage hurdle | 2.1267 | 1.0324 | made accuracy worse |
| Model 6 | capacity tuned on an inner window | 2.1210 | 1.0319 | made no meaningful difference |

### The three findings that matter

**1. Tweedie helped (FACT).** Changing only the objective improved RMSE from 2.1467 to 2.1256. Matching the loss function to a zero-inflated target is worth more than most feature work here.

**2. Recency did not help (FACT).** Two independent experimental designs — the Model 2/3 comparison and the ablation ladder — both put the effect at or below noise. **INTERPRETATION:** the rolling means already encode it. A series whose 28-day average is zero is, by definition, in a long dry spell; an explicit counter restates what the model can already see.

**3. Listing-awareness did not help either (FACT).** The underlying observation is real, and was confirmed more strongly than the EDA had put it — rows flagged pre-listing have a 100.00% zero-sales rate. But at this forecast origin **0% of rows are pre-listing**, so the feature is constant across everything it is asked to predict. **INTERPRETATION:** a true description of the data is not automatically a useful feature.

## 15. Model comparison

Full detail is in `FINAL_MODEL_COMPARISON_REPORT.pdf`. The one-line summary of the ablation ladder: historical demand features account for about 41% of the achievable error reduction, and everything else is a rounding error by comparison.

## 16. Error analysis

### What the model relied on

![Feature importance](charts/feature_importance.png)

> Importance shows what the model **used**, not what **causes** sales. A feature ranking highly is not evidence of a causal relationship.

### Where the error lives

| | |
|---|---|
| Share of all squared error from the worst 1% of rows | 55.18% |
| Validation rows with actual sales = 0 | 54.44% |
| Mean prediction on those zero rows | 0.5849 |
| Mean prediction where actual > 0 | 2.3134 |
| Mean actual where actual > 0 | 3.1665 |

**INTERPRETATION:** the error is dominated by a small number of high-volume rows, and the model systematically under-predicts the busiest days while placing a small positive value on days that turn out to be zero. That is the classic conservative compromise a squared-error-family objective makes on a zero-inflated target.

### Category

| Group | Rows | Actual mean | Pred mean | RMSE | MAE | Share of total error |
|---|---|---|---|---|---|---|
| FOODS | 402,360 | 2.069 | 1.939 | 2.662 | 1.337 | 74.21% |
| HOBBIES | 158,200 | 0.732 | 0.703 | 1.628 | 0.739 | 10.92% |
| HOUSEHOLD | 293,160 | 0.967 | 0.955 | 1.396 | 0.772 | 14.87% |

### Historical volume tier

| Group | Rows | Actual mean | Pred mean | RMSE | MAE | Share of total error |
|---|---|---|---|---|---|---|
| high (>3) | 65,968 | 7.585 | 7.196 | 5.976 | 3.639 | 61.33% |
| medium (1-3) | 160,244 | 2.185 | 2.076 | 2.291 | 1.505 | 21.90% |
| low (0.2-1) | 398,160 | 0.797 | 0.761 | 1.174 | 0.782 | 14.30% |
| very low (<0.2/day) | 229,348 | 0.280 | 0.267 | 0.643 | 0.384 | 2.47% |

![Error by volume](charts/error_by_volume.png)

**INTERPRETATION:** the busiest products are where the model struggles most, and they are a small minority of rows. This is where any further effort would pay off.

### Historical sparsity

| Group | Rows | Actual mean | Pred mean | RMSE | MAE | Share of total error |
|---|---|---|---|---|---|---|
| <50% zeros | 184,072 | 3.706 | 3.556 | 3.681 | 2.051 | 64.94% |
| 50-75% zeros | 262,696 | 1.317 | 1.237 | 1.901 | 1.062 | 24.70% |
| 75-95% zeros | 361,704 | 0.535 | 0.506 | 1.023 | 0.582 | 9.86% |
| >95% zeros | 45,248 | 0.219 | 0.207 | 0.645 | 0.306 | 0.49% |

### Does accuracy decay across the 28 days?

![RMSE by horizon](charts/rmse_by_horizon.png)

**FACT:** day 1 is the most accurate. Beyond that the pattern is uneven rather than a clean decay — day-to-day demand variation within the window matters more than distance from the origin. **INTERPRETATION:** because every origin-relative feature is held constant across all 28 days, the model has no more information about day 2 than about day 28; what changes is only the calendar. That is a deliberate consequence of the fixed-origin design.

## 17. Final model selection

**Model 4  + listing** — chosen mechanically as the lowest RMSE on the primary validation window, not by preference.

| | |
|---|---|
| Objective | tweedie (variance_power=1.1) |
| Features | 32 |
| Training rows | 12,805,800 |
| Validation RMSE | **2.1210** |
| Validation MAE | **1.0319** |

## 18. The 28-day forecast

The selected configuration was retrained with the forecast origin moved to d_1941 (2016-05-22) and used to predict **d_1942 .. d_1969 (2016-05-23 .. 2016-06-19)**.

| Check | Result |
|---|---|
| Rows | 30,490 (one per series) |
| Forecast columns | F1..F28 |
| Duplicate ids | 0 |
| NaN values | 0 |
| Negative predictions | 0 |
| Structure checks passed | 10/10 |

> **No accuracy figure can be quoted for this forecast.** d_1942..d_1969 has no ground truth in any file. The only honest estimate of its quality is the validation result above (RMSE 2.1210).

## 19. Novelty — what actually survived

The project's proposed novelty was *Listing-Aware + Recency-Aware Demand Forecasting*. **We tested it and it did not hold up.** Neither feature group produced a measurable improvement, and the hurdle model did not beat a single-stage Tweedie model.

We are not presenting it anyway. What we can defend is the method rather than the mechanism:

1. **An empirically verified leakage guarantee.** Not a claim in a slide — a corruption test that overwrites the future and proves 32 features are unchanged. It caught a real issue on its first run.
2. **Hypotheses tested and dropped on evidence.** Three plausible, well-motivated ideas were measured and abandoned because the numbers did not support them. The ablation table shows exactly what each idea was worth.
3. **Honest separation of description from prediction.** The pre-listing finding is real (100.00% zero-sales rate) and useless for this horizon (0% of forecast rows). Recognising that distinction is the actual insight.

**INTERPRETATION:** a team that can show which of its ideas failed is more credible than one that reports only successes.

## 20. Limitations

- Results come from one primary validation window; other windows give different error levels.
- Hyperparameters were tuned only over a small grid on an inner window.
- No uncertainty intervals are produced; these are point forecasts.
- The comparison against the team benchmark is not like-for-like, because their methodology is undocumented.
- Stockouts and promotions remain unobservable; nothing here recovers them.
- `pre_listing` duplicates `price_is_missing`, and `zero_streak_length` duplicates `days_since_last_sale`. Both redundancies were measured and left in place rather than silently dropped mid-experiment.

## 21. Future work

- Obtain the team's validation methodology and run a genuine head-to-head.
- Broader hyperparameter search on the inner window.
- Per-horizon or per-segment models for the high-volume tail, which carries most of the error.
- Quantile forecasts for inventory decisions, where the cost of under- and over-stocking is asymmetric.
- Recursive forecasting, so lag_1 becomes usable beyond day 1.

## 22. Questions judges may ask

**Why LightGBM?**

It handles 12.8 million training rows in about two minutes on a laptop, takes categorical features natively, handles missing prices without imputation, and is the strongest published family on this dataset. We also measured it against naive baselines rather than assuming.

**Why Tweedie?**

Tweedie models non-negative outcomes with a spike at zero, which matches a target where 68% of rows are zero. We measured it: RMSE improved from 2.1467 to 2.1256 with only the objective changed.

**Why not an LSTM or Transformer?**

30,490 short, mostly-zero series is not a regime where sequence models have an advantage over gradient-boosted trees, and they would cost far more compute for an unproven gain. Given our measured result that even recency features add nothing on top of rolling means, extra sequence modelling capacity is unlikely to be the binding constraint.

**Why are there so many zero sales?**

Most products do not sell every day in every store. Some zeros are also structural: before a product is listed in a store it records zeros by definition. We measured that pre-listing rows are 100.00% zero.

**What is SNAP?**

The US Supplemental Nutrition Assistance Program, a food-assistance benefit. The calendar records per state and day whether it was usable. The EDA measured a +12.7% overall lift and +17.3% within FOODS — the effect lands where domain knowledge says it should, which is a good internal consistency check.

**What is intermittent demand?**

A product that sells on some days and records zero on many others, rather than a smooth daily flow.

**What is leakage, and how do you know you have none?**

Leakage is letting information into the model that would not have existed when the forecast was really made. We overwrite every sales value after the forecast origin with 9999, rebuild the features, and check all 32 are bit-for-bit identical. They are.

**Why 28 days?**

That is what the task defines: sample_submission.csv has columns F1 to F28, and the calendar and price files extend exactly 28 days past the last day of sales.

**Why a fixed-origin backtest?**

Because it reproduces the real task. We stand on one day and predict the next 28 at once. Random splitting would let the model learn from the future to explain the past.

**Why one global model instead of 30,490 separate ones?**

A global model lets a sparse item borrow weekend, holiday and SNAP patterns learned from thousands of other items. Many series sell only a handful of units in five years and could not support their own model.

**How do you know the model is not overfitting?**

The validation window was never used to make any training decision — no early stopping, and hyperparameters were chosen on a separate earlier window (d_1886 to d_1913) so the primary window stayed untouched.

**Why did recency not matter, when the EDA said it was the strongest signal?**

Both statements are true. The dry-spell relationship is real in the data, but the rolling-mean features already capture it — a series with a 28-day average of zero is in a dry spell by definition. The EDA measured a relationship; we measured incremental predictive value. They are different questions.

**What happens if the hurdle model performs worse?**

It did, so we did not use it. That is documented in Model 5's report rather than quietly dropped.

**How does this compare with the team's model?**

We cannot say fairly. We have their two numbers but not their validation dates, series count, or metric code. We label theirs as a team-reported benchmark under their own setup and deliberately do not compute a percentage difference.

**Can this generalise to another store or item?**

Within this chain, yes — it is one global model already covering 10 stores and 3,049 products, and a new item would get predictions from hierarchy and calendar features. A genuinely new chain would need retraining.

**What information is available for the future forecast?**

Calendar, weekday, month, holidays, SNAP flags and sell_price are all present for d_1942 to d_1969 — we verified 100% price coverage. Only the sales themselves are missing, which is what we predict.

---

*Every figure in this report traces to a JSON record in `experiments/` written by an executed run. Where something was not measured, the report says so.*