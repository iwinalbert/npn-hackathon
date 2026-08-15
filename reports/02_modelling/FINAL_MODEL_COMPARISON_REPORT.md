# Final Model Comparison

*Generated 2026-08-14. Every number below comes from an executed run scored on the identical validation window.*

> **Terms used in this report.** **RMSE** (Root Mean Squared Error) measures how far predictions land from actual sales, counting big misses much more heavily than small ones — lower is better. **MAE** (Mean Absolute Error) is the plain average size of the miss. **WAPE** expresses total error as a share of total actual demand, which exposes a model that scores well simply by predicting near-zero everywhere. **Leakage** means letting information into the model that would not have existed at the moment the forecast was really made. **SNAP** is the US Supplemental Nutrition Assistance Program, a food-assistance benefit; the dataset records, per state per day, whether it was usable. **Intermittent demand** describes a product that sells on some days and records zero on many others.

## How to read this

All models were scored on the same 28 days (2016-04-25 .. 2016-05-22), across all 30,490 series, using the same metric code — 853,720 predictions each. Ranking is by RMSE, with MAE as the secondary metric, decided before the results were seen.

## Full comparison (measured)

| Model | Objective | Feature groups | RMSE | MAE | WAPE | Training time |
|---|---|---|---|---|---|---|
| Model 4  + listing **<-- best** | tweedie (variance_power=1.1) | A B C D E F G | 2.1210 | 1.0319 | 0.7152 | 116s |
| Model 6  tuned Tweedie | tweedie (variance_power=1.1) | A B C D E F G | 2.1210 | 1.0319 | 0.7152 | 115s |
| Model 2  LightGBM Tweedie | tweedie (variance_power=1.1) | A B E F G | 2.1256 | 1.0315 | 0.7149 | 106s |
| Model 3  + recency | tweedie (variance_power=1.1) | A B C E F G | 2.1258 | 1.0320 | 0.7153 | 115s |
| Model 5  hurdle (2-stage) | stage1=binary, stage2=poisson | A B C D E F G | 2.1267 | 1.0324 | 0.7155 | 177s |
| Model 1  LightGBM L2 | regression (L2) | A B E F G | 2.1467 | 1.0411 | 0.7216 | 65s |
| Model 0  rolling mean 28 | n/a | none | 2.2430 | 1.0657 | 0.7386 | — |
| Model 0  rolling mean 7 | n/a | none | 2.2487 | 1.0683 | 0.7404 | — |
| Model 0  seasonal naive | n/a | none | 2.6769 | 1.2440 | 0.8622 | — |
| Model 0  last value | n/a | none | 2.8936 | 1.3730 | 0.9516 | — |
| *Team-reported benchmark* | *LightGBM Tweedie* | *not documented* | *2.0324* | *1.0869* | *—* | *—* |

![Model comparison](charts/model_comparison.png)

> **Model 4 and Model 6 are numerically identical, and that is the expected result.** The capacity search in script 05 chose the settings Model 4 was already using, so Model 6 retrained the same configuration on the same data. Reproducing the earlier score to every decimal place is a useful check that the pipeline is deterministic — same inputs, same seed, same answer.

> **RMSE and MAE do not agree on the winner.** Model 4 has the lowest RMSE while Model 2 has the lowest MAE, and the gaps in both directions are small. RMSE was fixed as the primary metric before any results were seen, so Model 4 is selected — but the honest reading is that Models 2, 3, 4 and 6 are all within noise of one another.

## The team benchmark — why this is not a like-for-like comparison

We were given two numbers (RMSE 2.0324, MAE 1.0869) and no methodology. Before treating any difference as meaningful, the specification requires checking whether the two setups match. We cannot check, because the following are all unknown to us:

- which validation dates were used, and whether the horizon was 28 days
- whether all 30,490 series were scored, or a subset
- whether predictions were clipped at zero
- how the features were built, and what leakage controls applied
- whether the metric was computed over the same 853,720 predictions

> **Therefore this is labelled a team-reported benchmark under their own validation setup, and no percentage improvement or degradation is calculated against it.** Doing that arithmetic would imply a shared methodology that has not been established. If the team can supply their validation dates, series count and metric code, a fair comparison can be computed in minutes — the harness is already built.

For reference only: our best measured RMSE is 2.1210 against their reported 2.0324. Taken at face value that is behind their figure, but the caveat above is the honest headline, not the number.

## Feature-group ablation (measured)

Each rung adds one feature group on top of the previous rung. Objective, hyperparameters, training origins and validation window are identical throughout, so each change in RMSE is attributable to the group just added.

| Configuration | Features | RMSE | MAE | ΔRMSE | ΔMAE |
|---|---|---|---|---|---|
| A. Calendar only | 9 | 3.6393 | 1.6591 | — | — |
| B. Calendar + Historical demand | 17 | 2.1584 | 1.0500 | -1.4809 | -0.6091 |
| C. + Recency | 20 | 2.1614 | 1.0549 | +0.0030 | +0.0048 |
| D. + Price | 24 | 2.1418 | 1.0461 | -0.0197 | -0.0088 |
| E. + Listing-aware | 26 | 2.1537 | 1.0494 | +0.0120 | +0.0033 |
| F. + Hierarchy | 31 | 2.1374 | 1.0324 | -0.0164 | -0.0170 |
| G. Full feature set (+ horizon) | 32 | 2.1210 | 1.0319 | -0.0163 | -0.0005 |

![Ablation ladder](charts/ablation_ladder.png)

### What the ladder actually says

**Historical demand is doing nearly all of the work.** Going from calendar-only to calendar-plus-demand improves RMSE by 1.4809 — around 41% of the starting error. Every other group combined moves it by a fraction of that.

**Recency and listing did not help.** In this ladder both came out slightly negative. In the separately-controlled Model 2/3/4 comparison the signs differed slightly, which tells us these effects are within run-to-run noise rather than real. Two independent experimental designs agreeing that an effect is indistinguishable from zero is a result, and it is reported as one.

## Additional validation windows (measured)

The best configuration retrained and rescored on other 28-day periods, to check the result is not an artefact of one lucky window.

| Window | Origin | Dates | RMSE | MAE | WAPE |
|---|---|---|---|---|---|
| primary_spring_2016 | d_1913 | 2016-04-25 .. 2016-05-22 | 2.1210 | 1.0319 | 0.7152 |
| christmas_2015 | d_1778 | 2015-12-12 .. 2016-01-08 | 2.1851 | 0.9231 | 0.7750 |
| summer_2015 | d_1629 | 2015-07-16 .. 2015-08-12 | 2.1405 | 0.9746 | 0.7396 |

Error levels differ substantially between periods. That is expected — demand itself differs between periods — and it is the reason a single-window score should never be quoted as though it were a universal accuracy figure.

## Conclusion

The best measured model is **Model 4  + listing**, at RMSE 2.1210 and MAE 1.0319. It was selected mechanically by the pre-agreed metric, not chosen.

The uncomfortable finding is that the two feature groups the project had nominated as its novelty — recency and listing-awareness — do not measurably improve the forecast, and neither does the two-stage hurdle structure. What does work is unglamorous: recent-demand features, a Tweedie objective, price, hierarchy, and enough model capacity.
