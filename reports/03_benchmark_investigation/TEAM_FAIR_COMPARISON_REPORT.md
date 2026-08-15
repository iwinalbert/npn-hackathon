# Team Benchmark — Fair Comparison Investigation

*Generated 2026-08-14. Every figure attributed to us comes from an experiment that actually ran and is recorded in `experiments/`. The three team figures are quoted exactly as supplied.*

> **Headline: the comparison is NOT currently fair, and we could not reproduce the team's result.** Their methodology is not documented anywhere in this project or on this machine, and their reported RMSE/MAE combination was not reproduced by any of four methodologies we tested. This report explains exactly what we ruled out and what would settle the question.

> **Terms.** **RMSE** punishes large misses much more heavily than small ones. **MAE** is the plain average miss. **Leakage** means letting information into the model that would not have existed when the forecast was really made — for example using "yesterday's sales" for day 20 of a 28-day forecast, when yesterday is itself still in the future. **Tweedie** is a loss function for non-negative data with many zeros.

---

## 1. What the team did — what we could actually find

**We searched for it properly before assuming anything.** A full-text search across every file in the project for their reported numbers (2.0324, 1.0869, 2.0770, 1.1187, 2.1434, 1.1275) and for the strings `RandomForest`, `XGBoost`, `Random Forest` returned only:

- coincidental digit sequences inside our own EDA tables and per-series statistics (e.g. a series whose standard deviation happens to be 2.0324)
- our own model files, prediction files and reports
- one docstring in `pipeline/metrics.py` that we wrote ourselves

We also searched outside the project — Desktop, Downloads and the wider OneDrive folder — for notebooks, scripts, model files or result files. Nothing relating to the team's models exists on this machine.

> **Therefore: the team's methodology is UNDOCUMENTED and could not be inspected.** Everything below labelled "team" is either a number they supplied or an explicitly-flagged reconstruction. We did not invent their method and we do not claim to know it.

### What we know versus what we do not

| Question | Status |
|---|---|
| Their reported RMSE / MAE | **Known** — supplied by the team |
| That they used the same raw and processed datasets | **Stated** by the team |
| Training cutoff / validation start / validation end | **UNKNOWN** |
| Number of forecast days | **UNKNOWN** |
| Number of series scored | **UNKNOWN** |
| Direct or recursive forecasting | **UNKNOWN** |
| Whether predictions were clipped at zero | **UNKNOWN** |
| Whether zero rows were dropped or reweighted | **UNKNOWN** |
| Exact lag / rolling feature definitions | **UNKNOWN** |
| LightGBM parameters and Tweedie variance power | **UNKNOWN** |
| How RMSE and MAE were computed (pooled? per-series? aggregated?) | **UNKNOWN** |

## 2. Feature comparison

Step 1 of the brief asked for a feature-by-feature table. We can fill in our column honestly and must leave theirs unknown — writing anything else would be inventing their method.

| Feature | Used by team? | Used by us? | Difference | Reproduce? |
|---|---|---|---|---|
| `lag_1` | **UNKNOWN** | yes (origin-relative) | cannot be determined | not possible without their spec |
| `lag_7` | **UNKNOWN** | yes (origin-relative) | cannot be determined | not possible without their spec |
| `lag_14` | **UNKNOWN** | yes (origin-relative) | cannot be determined | not possible without their spec |
| `lag_21` | **UNKNOWN** | no | cannot be determined | not possible without their spec |
| `lag_28` | **UNKNOWN** | yes (origin-relative) | cannot be determined | not possible without their spec |
| `lag_35` | **UNKNOWN** | no (tested in reproduction) | cannot be determined | not possible without their spec |
| `lag_56` | **UNKNOWN** | no (tested in reproduction) | cannot be determined | not possible without their spec |
| `rolling_mean_7` | **UNKNOWN** | yes | cannot be determined | not possible without their spec |
| `rolling_mean_14` | **UNKNOWN** | no | cannot be determined | not possible without their spec |
| `rolling_mean_28` | **UNKNOWN** | yes — our single strongest feature (74% of model gain) | cannot be determined | not possible without their spec |
| `rolling_mean_56` | **UNKNOWN** | no (tested in reproduction) | cannot be determined | not possible without their spec |
| `rolling_std_7` | **UNKNOWN** | yes | cannot be determined | not possible without their spec |
| `rolling_std_28` | **UNKNOWN** | yes | cannot be determined | not possible without their spec |
| `rolling_min / rolling_max` | **UNKNOWN** | no (tested in reproduction) | cannot be determined | not possible without their spec |
| `price features` | **UNKNOWN** | yes — sell_price, recent avg, relative price, missing flag | cannot be determined | not possible without their spec |
| `calendar features` | **UNKNOWN** | yes — weekday, month, year, weekend | cannot be determined | not possible without their spec |
| `SNAP` | **UNKNOWN** | yes — matched to each series' own state | cannot be determined | not possible without their spec |
| `events` | **UNKNOWN** | yes — event_name_1/2, event_type_1/2 | cannot be determined | not possible without their spec |
| `store / category / dept / item / state ids` | **UNKNOWN** | yes — all five, native categoricals | cannot be determined | not possible without their spec |
| `target encoding` | **UNKNOWN** | no | cannot be determined | not possible without their spec |
| `recency (days_since_last_sale, zero_streak)` | **UNKNOWN** | yes — measured as no help | cannot be determined | not possible without their spec |
| `listing-aware (pre_listing, days_since_first_listing)` | **UNKNOWN** | yes — measured as no help | cannot be determined | not possible without their spec |

> Every cell in the "Used by team?" column is unknown for the same reason: there is no artefact of their work to read. This table is included because the brief asked for it, not because it tells us anything about them.

## 3. Is the validation identical?

**Unknown, and this matters more than anything else in this report.** Our setup is fully specified and was held constant across every experiment we have ever run:

| | Ours |
|---|---|
| Forecast origin | d_1913 (2016-04-24) |
| Validation days | d_1914 .. d_1941 (2016-04-25 .. 2016-05-22) |
| Horizon | 28 days |
| Series | 30,490 |
| Predictions scored | 853,720 |
| Forecasting mode | direct multi-horizon, fixed origin |
| Clipping | predictions clipped at 0 |
| Metric | pooled over all rows, unweighted |

We did not change any of this, per the brief. If the team used a different window, a different horizon, a subset of series, or a different metric implementation, then the two sets of numbers are not measuring the same thing and no arithmetic comparison between them is meaningful.

## 4. What we actually ran

Four configurations, all scored on our validation window, all on the same 853,720 predictions, all with the same metric code.

| Configuration | RMSE | MAE | Leakage-safe? |
|---|---|---|---|
| Diagnostic leaky probe (deliberately unsafe) | 1.9165 | 0.9754 | **NO — by design** |
| *Team reported (their setup)* | *2.0324* | *1.0869* | *unknown* |
| **Our best model (Model 4)** | **2.1210** | **1.0319** | **yes — verified** |
| Team-style reproduction (28-day lookback) | 2.1835 | 1.0498 | yes — verified |

![Comparison](charts/team_comparison_scatter.png)

The team's other two models sit at RMSE 2.0770 / MAE 1.1187 (Random Forest, MSE) and RMSE 2.1434 / MAE 1.1275 (XGBoost, Poisson).

## 5. Diagnosing the difference — what we ruled out

Rather than guess, we tested the plausible explanations one at a time.

### Ruled out: prediction calibration or clipping

We rescaled our existing predictions by every constant factor from 0.9 to 2.0. The best RMSE any rescaling can achieve is **2.1195** — still well above their 2.0324. Scaling up made *both* metrics worse, not one better. So their result is not our model with a different multiplier or clipping rule; the prediction vector itself must be different.

### Ruled out: a different validation window

Their MAE (1.0869) is *higher* than ours (1.0319). MAE tracks the demand level of the window being scored. We measured the mean daily sales of every 28-day window in the last two years: the range is 1.0622 to **1.4428**, and the maximum is our own window. For their MAE to come from a higher-demand window at our error rate, that window would need a mean of about 1.52 — higher than any window that exists in the data. So window choice alone cannot produce their numbers.

### Ruled out: per-target-day lag construction

The most common public M5 recipe builds one row per (series, day) with lags of 28 days or more, rather than computing features once at the origin. We implemented it (`pipeline/team_style.py`, 25 features, 21,312,510 training rows), verified it leakage-safe with the same corruption test, and held the objective, hyperparameters and validation window identical to ours. It scored **2.1835 / 1.0498** — worse than our model on both metrics.

Broken down by how much each series normally sells:

| Volume tier | Rows | Actual mean | Our RMSE | Team-style RMSE | Difference |
|---|---|---|---|---|---|
| very low (<0.2/day) | 229,348 | 0.280 | 0.6431 | 0.6471 | +0.0040 |
| low (0.2-1) | 398,160 | 0.797 | 1.1743 | 1.1780 | +0.0037 |
| medium (1-3) | 160,244 | 2.185 | 2.2912 | 2.3235 | +0.0323 |
| high (>3) | 65,968 | 7.585 | 5.9756 | 6.2255 | +0.2499 |

The 28-day-lookback version is worse everywhere and worst on high-volume series. That is the explanation: our origin-relative features include `lag_1` and `rolling_mean_7` measured right up to the forecast origin, which is fresher information than a 28-day-old lag. Freshness matters most for the busiest products.

### Not ruled out: a per-target-day leak

The remaining common explanation for a score that cannot be reproduced legitimately is leakage — computing `lag_1` or `rolling_mean_7` relative to each *target* day instead of the forecast origin. On day 20 of the horizon that reads a real sales value from inside the validation window, which nobody would have had on the day the forecast was made.

We built exactly that, confirmed it leaky with the corruption test (10 features moved when the future was altered), and measured it: **RMSE 1.9165, MAE 0.9754**.

> **How to read that, carefully.** A leak of this kind scores *better* than the team's reported RMSE (1.9165 vs 2.0324), so it is *sufficient* to produce a number in their range. That is **not** evidence that they leaked. It only establishes that their RMSE is reachable by a mechanism we know produces invalid results, and is not reachable by any valid mechanism we tested. It is a reason to check, not an accusation.

### The part that no explanation covers

Their MAE (1.0869) is worse than **every** configuration we measured — worse than our best (1.0319), worse than the safe team-style reproduction (1.0498), and worse than the leaky probe (0.9754). Meanwhile their RMSE is better than both of our legitimate models. Lower RMSE with higher MAE means comparatively fewer large misses but more medium-sized ones, and we could not produce that combination with any of four methodologies. This is the strongest single sign that their numbers were produced under a different evaluation setup rather than simply by a better model.

## 6. Comparison table, with the honest labels

| Approach | RMSE | MAE | Features | Validation window | Notes |
|---|---|---|---|---|---|
| Team reported — LightGBM Tweedie | 2.0324 | 1.0869 | unknown | **unknown** | reported by team; not independently verified |
| Team reported — Random Forest MSE | 2.0770 | 1.1187 | unknown | **unknown** | reported by team |
| Team reported — XGBoost Poisson | 2.1434 | 1.1275 | unknown | **unknown** | reported by team |
| Our current best — LightGBM Tweedie | 2.1210 | 1.0319 | 32 | d_1914..d_1941, 30,490 series | leakage-verified |
| Our team-style reproduction | 2.1835 | 1.0498 | 25 | d_1914..d_1941, 30,490 series | leakage-verified reconstruction |
| Diagnostic leaky probe | 1.9165 | 0.9754 | 25 | d_1914..d_1941, 30,490 series | **invalid — diagnosis only** |

### Why we are not reporting a percentage difference

The brief asks for a percentage improvement or degradation, but only after establishing that the methodology matches. It does not match — or rather, we cannot establish that it matches, which for this purpose is the same thing. Computing `(2.0324 - 2.1210) / 2.0324 = -4.36%` would imply the two numbers measure the same quantity on the same rows. They may not. The arithmetic is shown here so nobody has to wonder what it would have been, and it should not be quoted as a result.

## 7. The one improvement lever we tested

Our error analysis established that high-volume series are 7.7% of rows but carry **61% of all squared error**, and that we systematically under-predict them. The Tweedie variance power controls how much the objective concentrates on zeros versus the tail, and we had never tested it — 1.1 was an untested assumption sitting directly on the thing limiting our RMSE.

Tested on the **inner** window (d_1886..d_1913) so the primary window stays an unbiased estimate:

| Tweedie power | Inner RMSE | Inner MAE | Bias | High-volume RMSE | High-volume bias |
|---|---|---|---|---|---|
| 1.1 *(current)* | 2.0899 | 1.0173 | +0.0283 | 5.9380 | +0.3166 |
| 1.3 | 2.0793 | 1.0124 | +0.0176 | 5.8955 | +0.2494 |
| 1.5 | 2.0766 | 1.0087 | +0.0022 | 5.8722 | +0.1289 |

On the inner window, power 1.5 looked like a clear win: RMSE improved by -0.0133, MAE improved, and — exactly as the error analysis predicted — the high-volume bias shrank from +0.317 to +0.129. Every signal pointed the same way.

### Then we tested it on the primary window — and it did not hold

Because the power was selected using only the inner window, applying it once to d_1914..d_1941 is a clean unbiased test. We ran it:

| | RMSE | MAE | High-volume RMSE | High-volume bias |
|---|---|---|---|---|
| Model 4 — power 1.1 | **2.1210** | 1.0319 | 5.9756 | −0.389 |
| Model 9 — power 1.5 | 2.1263 | **1.0289** | 5.9931 | -0.422 |
| Change | +0.0053 | -0.0030 | +0.0175 | — |

**The improvement did not transfer.** RMSE got slightly worse, and the high-volume bias moved the wrong way on this window (−0.42) even though it had improved on the other one (+0.13). A −0.0133 gain on one 28-day window turned into a +0.0053 loss on the next.

> **Decision: do not change the Tweedie power.** Model 4 with power 1.1 remains our best model. This is the correct outcome of a disciplined process — we formed a hypothesis from measured evidence, tested it properly, and it failed. Had we selected on the primary window instead, we would have shipped a change that was really just noise.

It is also a caution about the team comparison itself: a 0.013 swing between adjacent 28-day windows is ordinary noise here, and the gap being discussed is only about 0.09.

## 8. What we should change, and what we should not

### Should NOT change

- **Our validation setup.** It is fully specified, leakage-verified, and consistent across every experiment. Changing it to match an unknown setup would destroy the one thing we can actually defend.
- **Our origin-relative feature design.** We tested the main alternative and it was worse (2.1835 vs 2.1210).
- **Do not adopt per-target-day lags to chase the benchmark.** They are only better when they reach into the forecast window, which is exactly the thing that makes a forecast worthless in production.
- **Do not re-add recency or listing features as novelty.** Previously measured as no help; nothing here changes that.

### Should change / do next, in priority order

1. **Ask the team for five specific things** — their validation dates, series count, horizon, whether lags are computed relative to the target day or the origin, and their metric code. Four of the five are one-line answers, and they would settle this entirely. The harness to run a real head-to-head already exists.
2. **Attack the high-volume tail.** It is 7.7% of rows and 61% of our error. Options with evidence behind them: volume-weighted training, a separate model for the high tier, or per-horizon models.
3. **Test recursive forecasting.** The leaky probe is not just a diagnostic — it is an upper bound. It says that perfect knowledge of recent sales during the horizon would be worth about 0.2045 RMSE. A recursive strategy feeds the model's own predictions back in as lags, which is legitimate and captures some fraction of that headroom. That is the single most promising legitimate direction this investigation has produced.

## 9. Is the comparison genuinely fair?

**No, and it cannot be made fair from our side alone.** We reproduced everything reproducible: same dataset, same window, same metric code, same 853,720 predictions, and a good-faith reconstruction of the standard public recipe. What we cannot reproduce is a methodology we have never seen.

What we can state with confidence:

- Our result is leakage-verified by an empirical corruption test. Their leakage status is unknown.
- Our number is reproducible: an independently retrained run matched it to four decimal places.
- Under our methodology, our model has the better MAE and their reported RMSE is lower.
- The RMSE/MAE combination they report was not reproducible by any of four methodologies we tested, which suggests a difference in evaluation rather than purely in modelling.

> **We are not claiming to beat the team, and we are not conceding that they beat us.** Neither claim is supportable on the evidence. What is supportable is that our pipeline is verified, reproducible and honest about its own limits — and that a genuine comparison is five questions away.

---

*All figures attributed to us come from executed runs recorded in `experiments/`: `model_04_tweedie_recency_listing`, `model_08_team_style_reproduction`, `diagnostic_leakage_probe_DO_NOT_USE`, and `probe_tweedie_power_*`. No existing model, experiment or report was modified to produce this comparison.*