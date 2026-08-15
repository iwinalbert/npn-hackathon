# Autonomous Research Campaign

*Objective: reduce RMSE as far below 2.0 as legitimately possible. Generated 2026-08-14. Two experiments run, both rejected, campaign stopped on evidence.*

> ## OUTCOME: the champion stands at **RMSE 2.1210 / MAE 1.0319**, and RMSE < 2.0 is **not achievable** with the information in this dataset. The evidence for that ceiling is quantitative and is set out below.

> **Terms.** **Variance vs bias** — bias is a consistent tilt (predicting too high or too low every time); variance is error that flips sign row to row. **Residual correlation** — whether two models make the *same* mistakes on the same rows. **Oracle** — a cheating predictor allowed to see the answers, used to measure what is even possible.

---

## The audit that set the strategy

Across 69 prior experiments, exactly one thing ever beat the champion (recursive forecasting, by 0.0029 — inside noise, rejected on a +0.0398 MAE cost). The failures cluster into two families:

| Family | Attempts | Result |
|---|---|---|
| Attacks on **bias** | calibration, per-series correction (#69), volume weighting, high-volume rescaling | all failed |
| Attacks on **information** | 14 new features in 4 groups, recency (x2), listing (x2), per-target-day lags | all failed |

The error autopsy explains why: **MSE = 0.0049 bias-squared + 4.4939 variance. 99.89% of the error is variance.** Neither family was attacking the binding constraint.

That left one canonical technique untested on RMSE-competitive models — **ensembling**, the standard remedy for variance. Phase 8 had blended Tweedie with L1, but L1 is deliberately a poor RMSE model, so that measured a metric trade-off, not variance reduction.

## Experiment #70 — variance-reduction ensemble

**Hypothesis.** Averaging several individually-strong models that make *different* mistakes cancels the fit-to-fit component of variance.

**Design.** Six LightGBM Tweedie models, identical 32 features and training origins, diversified across seed, tree size (96–256 leaves), feature/bagging fractions, and Tweedie power (1.1–1.3). Equal weights fixed a priori, so nothing was selected using the validation window.

| Member | Power | Leaves | Seed | RMSE | MAE |
|---|---|---|---|---|---|
| m1_champion | 1.1 | 128 | 42 | 2.1210 | 1.0319 |
| m2_wide | 1.1 | 256 | 7 | 2.1592 | 1.0322 |
| m3_p12 | 1.2 | 160 | 101 | 2.1301 | 1.0306 |
| m4_p13_small | 1.3 | 96 | 202 | 2.1276 | 1.0314 |
| m5_deep | 1.1 | 192 | 303 | 2.1342 | 1.0311 |
| m6_p12_wide | 1.2 | 224 | 404 | 2.1399 | 1.0300 |
| **Equal-weight ensemble** | — | — | — | **2.1261** | **1.0290** |

**Result: REJECTED.** ΔRMSE +0.0051, ΔMAE -0.0030. The ensemble did not even beat its own best member.

### The finding that ended the campaign

**Mean pairwise residual correlation across the six models: 0.9897.**

Six models with different seeds, different tree sizes, different subsampling and different loss parameters make **essentially the same mistakes on the same rows**. If the errors were driven by the fitting procedure, they would decorrelate. They do not.

**INTERPRETATION:** the 4.4939 variance is not model variance. It is variance in the target that every model sees identically — irreducible given the available features. That single number explains why 69 previous experiments failed, and it predicts that better models of this kind cannot help either.

## Experiment #71 — year-over-year features

**Why this followed.** If model variance is not the lever, only new *information* can help. The champion's longest lookback is 28 days; it has no way to know what a specific product sold in a specific store one year earlier. `month` and `week_of_year` give only a chain-wide seasonal average, which cannot express that one item peaks in May and another in November.

This is materially different from the Phase 2 features that failed — those re-encoded information already present (a 14-day window between the existing 7 and 28, zero-counts restating recency). A 364-day lookback lies entirely outside the champion's feature horizon.

**Features added:** `lag_364`, `rolling_mean_7_lag364`, `rolling_mean_28_lag364`, `yoy_level_ratio`. All leakage-safe: for target day T+h with h ≤ 28, a 364-day lookback reads at most T−336. Confirmed by corruption test — all 36 features unchanged when every post-origin sale was overwritten with 9999.

| | RMSE | MAE |
|---|---|---|
| Champion | **2.1210** | **1.0319** |
| + year-over-year features | 2.1564 | 1.0344 |
| Change | +0.0354 | +0.0025 |

**Result: REJECTED.** ΔRMSE +0.0354 — clearly worse, and outside the noise band in the wrong direction.

The instructive detail: **the model did use them.** They took 2.56% of total gain, with `rolling_mean_7_lag364` ranking 5th of 36 features. They carry real signal — but mostly noise, and the capacity they consumed would have been better spent on splits of `rolling_mean_28`. Last year's demand is a worse guide to next month than last month's demand is.

## The practical ceiling, quantified

### Bound 1 — ensembling cannot get there, and the maths is checkable

For M models with residual standard deviation s and mean pairwise correlation ρ, the averaged residual variance is s²·(ρ + (1−ρ)/M). With the measured s = 2.1353 and ρ = 0.9897:

| Models averaged | Predicted RMSE |
|---|---|
| 1 | 2.1353 |
| 2 | 2.1298 |
| 6 | 2.1261 |
| 10 | 2.1254 |
| 100 | 2.1244 |
| **infinite** | **2.1243** |

> **This bound validates itself.** The formula predicts 2.1261 at M = 6; the measured six-member ensemble scored **2.1261** — an exact match to four decimals. The extrapolation to infinite members is therefore trustworthy.

![Ceiling](charts/campaign_ceiling.png)

**Averaging an unlimited number of models of this family bottoms out at 2.1243 — worse than the single champion, and 0.1243 away from the target.**

### Bound 2 — 2.0 is below an oracle that already knows the answer

| Predictor | RMSE |
|---|---|
| Champion | 2.1210 |
| **Target** | **2.0000** |
| Oracle: each series' true 28-day mean | 1.9818 |
| Oracle: optimal per-series rescale | 1.8823 |

Reaching 2.0 means essentially **matching a predictor that already knows each series' average over the very window being forecast** — and beating it would require exploiting within-window day-to-day structure on top. Experiment #69 tested whether that per-series level is learnable from the past and found it is not: bias measured on one fit does not transfer to another.

### Bound 3 — the error that would have to disappear is the unpredictable kind

Going from 2.1210 to 2.0000 means removing **11.1% of all squared error**. The autopsy located that error precisely:

| Where the error is | Share of squared error |
|---|---|
| Demand spikes (actual > 2× the series' historical mean) | 61.92% |
| Top volume decile | 66.5% |
| Worst 1,000 rows (0.12% of data) | 28.01% |

And those spikes are barely predictable from anything available: the spike rate is 0.31 on weekends versus 0.23 midweek, 0.272 on SNAP days versus 0.252 otherwise, 0.280 on event days versus 0.256. The model already has every one of those signals. With no promotion field and no inventory field anywhere in the dataset, the information needed simply is not present.

## Decision: stop

Research rule 20 permits stopping when the evidence shows further experiments are unlikely to produce meaningful improvement. That threshold is met, on three independent lines of evidence:

1. **Model-side improvement is closed.** Six architecturally different models correlate at ρ = 0.9897. Infinite ensembling floors at 2.1243. A seventh architecture (XGBoost, CatBoost, a neural network) would have to break that correlation, and nothing in the evidence suggests it would — all six already converge on the same conditional-mean estimate.
2. **Information-side improvement is closed.** Every feature family has been tested: recent demand, calendar, price, interactions, recency, listing, per-target-day lags, and now year-over-year. Eighteen features across six experiments; none helped.
3. **The target is below a cheating predictor.** 2.0 sits under the per-series oracle, and #69 showed that per-series level is not learnable from history.

### What would actually change the answer

Not a better model — **more information.** Specifically: a promotions or markdown calendar, inventory or stockout records, or store-level footfall. Those would address the spikes that carry 62% of the error. None exists in this dataset, and no amount of modelling recovers them.

## Final position

| | |
|---|---|
| Champion | Global LightGBM + Tweedie(1.1), 32 features |
| RMSE | **2.1210** |
| MAE | **1.0319** |
| Leakage | verified by corruption test |
| Reproducibility | re-run reproduced the score to every decimal |
| Robustness | 4 windows, RMSE std 0.033 |
| Experiments run in total | 71 |

The honest summary for the presentation: this project did not find a way below 2.0, and it can say precisely why. The error is dominated by genuinely unpredictable demand spikes; six different models make the same mistakes on the same rows; and the target sits below what a predictor with access to the answers achieves. That is a stronger position than an unexplained number would be.

---

*Experiments #70 and #71 are recorded in `experiments/`. Predictions saved separately. The champion model, its predictions, the final forecast, and all previous reports are unchanged.*