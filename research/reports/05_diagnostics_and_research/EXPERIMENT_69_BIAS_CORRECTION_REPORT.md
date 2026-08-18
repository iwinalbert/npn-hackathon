# Experiment #69 — Pre-Origin Per-Series Bias Correction

*Executed 2026-08-14. The selected model was not retrained or modified, and its prediction file is unchanged on disk.*

> ## DECISION: **REJECTED**

> **Terms.** **Bias correction** — multiplying each product's forecast by a number learned from how wrong that product's forecasts were recently. **Pre-origin** — using only days before the last day we are allowed to see (d_1913). **Shrinkage** — pulling a correction toward 1.0 (i.e. toward "leave it alone") when there is too little data to trust it. **RMSE** — average error with big misses punished heavily; lower is better.

---

## Result

| | RMSE | MAE |
|---|---|---|
| Baseline (untouched) | **2.1210** | **1.0319** |
| After correction | 2.1286 | 1.0333 |
| Absolute change | +0.0076 | +0.0013 |
| Percentage change | +0.356% | +0.130% |

**The correction made both metrics worse.** It was rejected — and, more importantly, it was rejected by its own fitting procedure before the validation window was ever touched.

## What was done

The idea came from the error autopsy, which found that per-series bias persists within the validation window (split-half correlation r = +0.557) and that an oracle per-series rescaling would reach RMSE 1.8823. If that bias could be learned from the past, it would be the largest available gain in the project.

### The problem that shaped the design

A bias factor is *actual ÷ predicted*. To measure it on the pre-origin window d_1886–d_1913 we need predictions for those days — and the selected model does not produce any, because it forecasts from d_1913 onward. So two **auxiliary models** were trained purely to generate pre-origin predictions:

| Model | Origin | Predicts | Purpose |
|---|---|---|---|
| AUX-A | d_1857 | d_1858–d_1885 | fit factors used to choose the shrinkage constant |
| AUX-B | d_1885 | d_1886–d_1913 | fit the final factors, and score AUX-A's factors |

Both use the identical configuration to the selected model, and both train only on origins at least 28 days before their own window. **The selected model itself was never retrained, reloaded, or altered.**

### Choosing the shrinkage constant without cheating

Shrinking a factor toward 1.0 needs a constant *k*. Picking *k* by trying values against d_1914–d_1941 would be selecting on the scoring window — the exact mistake that produced four false positives earlier in this project. So *k* was chosen entirely on pre-origin data: factors fitted on d_1858–d_1885, scored on d_1886–d_1913.

Critically, the search was **allowed to choose "apply no correction at all"** (the k → ∞ limit). A fitting procedure that cannot reject its own correction is not a fair test.

| k | Pre-origin RMSE | Pre-origin MAE |
|---|---|---|
| 0 | 2.5625 | 1.1531 |
| 2 | 2.5568 | 1.1478 |
| 5 | 2.5488 | 1.1413 |
| 10 | 2.5368 | 1.1326 |
| 20 | 2.5164 | 1.1199 |
| 50 | 2.4714 | 1.0976 |
| 100 | 2.4208 | 1.0790 |
| 250 | 2.3262 | 1.0553 |
| 500 | 2.2521 | 1.0412 |
| 1000 | 2.1870 | 1.0311 |
| 5000 | 2.1044 | 1.0199 |
| ∞ (no correction) **← chosen** | 2.0899 | 1.0173 |

![k selection](charts/exp69_k_selection.png)

**Every single finite value of k scored worse than leaving the predictions alone.** The trend is monotone: the more the correction is shrunk away, the better the result gets. The procedure was trying to tell us to abandon the correction entirely, and at k = 5000 — where factors are nearly 1.0 — it still had not caught up with doing nothing.

We applied the best finite k (5000) once anyway, purely to document what it would have cost. That is the number in the result table above.

## Safeguards applied

| Safeguard | Implementation |
|---|---|
| Shrink toward 1.0 for low-volume series | weight = P / (P + k), where P is the series' total predicted units in the fitting window |
| Clip factors | hard clip to [0.5, 2.0]; observed range was [0.8942, 1.1655], so **no factor hit a bound** |
| No validation actuals in fitting | verified empirically, see below |
| Nothing else changed | no new features, no ensemble, no weighting, no store-level term, no retraining |

## Leakage checks

| Check | Result | Detail |
|---|---|---|
| `aux_models_never_reach_validation` | PASS | highest day used = d_1913, validation starts d_1914 |
| `fitting_window_actuals_unaffected_by_future` | PASS | the d_1886..d_1913 sales used to fit the factors are identical when every day after d_1913 is overwritten with 9999 |
| `factors_unchanged_under_future_corruption` | PASS | all 30,490 correction factors bit-identical after corrupting the future |
| `baseline_predictions_untouched` | PASS | the base prediction file on disk is unchanged |
| `factors_within_clip_range` | PASS | range [0.8942, 1.1655] within [0.5, 2.0] |

The second and third checks are the decisive ones: every sales value after d_1913 was overwritten with 9999, and both the actuals used for fitting and all 30,490 resulting correction factors came back **bit-for-bit identical**. No d_1914–d_1941 sales entered the correction at any point.

## Distribution of the correction factors

| | |
|---|---|
| Series receiving a correction | 30,469 of 30,490 (99.93%) |
| Series hitting a clip bound | 0 (0.00%) |
| Factors below 1 (forecast reduced) | 7,317 |
| Factors above 1 (forecast increased) | 6,820 |
| Mean / median | 0.9999 / 0.9999 |
| Range | [0.8942, 1.1655] |

| Percentile | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| Factor | 0.9846 | 0.9953 | 0.9991 | 0.9999 | 1.0008 | 1.0045 | 1.0125 |

At the selected shrinkage the factors are almost all within ±1.5% of 1.0. That is the safeguard doing its job — and it means this experiment tested a *very gentle* correction. Even that gentle version made things worse.

## Performance by demand-volume decile

| Decile | Historical mean/day | Rows | Base RMSE | Corrected RMSE | ΔRMSE | ΔMAE | Mean factor |
|---|---|---|---|---|---|---|---|
| 1 | 0.01-0.09 | 85,428 | 0.5586 | 0.5586 | -0.0000 | +0.0000 | 1.0000 |
| 2 | 0.09-0.15 | 85,400 | 0.6457 | 0.6458 | +0.0000 | +0.0000 | 1.0000 |
| 3 | 0.15-0.22 | 85,288 | 0.7661 | 0.7662 | +0.0000 | +0.0000 | 1.0000 |
| 4 | 0.22-0.32 | 85,792 | 0.9043 | 0.9049 | +0.0006 | +0.0001 | 1.0000 |
| 5 | 0.32-0.44 | 85,232 | 1.0442 | 1.0445 | +0.0002 | +0.0001 | 1.0000 |
| 6 | 0.44-0.62 | 85,204 | 1.2016 | 1.2021 | +0.0004 | +0.0002 | 1.0001 |
| 7 | 0.62-0.87 | 85,428 | 1.3731 | 1.3734 | +0.0003 | +0.0002 | 1.0001 |
| 8 | 0.88-1.32 | 85,204 | 1.7540 | 1.7554 | +0.0014 | +0.0004 | 1.0000 |
| 9 | 1.32-2.44 | 85,400 | 2.3294 | 2.3315 | +0.0021 | +0.0011 | 0.9998 |
| 10 | 2.44-130.95 | 85,344 | 5.4723 | 5.4999 | +0.0276 | +0.0112 | 0.9988 |

![By decile](charts/exp69_by_decile.png)

### Does it help high-volume series?

**No — it hurts them most.** On the >3 units/day tier (65,968 rows), RMSE moves from **5.9756 to 6.0079** (+0.0322).

Almost all of the damage sits in decile 10, which absorbs +0.0276 of the +0.0076 total. Deciles 1–3 are unchanged to four decimal places. This is precisely inverted from the hypothesis: the autopsy identified high-volume series as the place to gain, and this is the only place the correction does real harm.

## Comparison against the validation noise floor

| | |
|---|---|
| Measured change | +0.0076 RMSE |
| Window-to-window noise (Phase 9) | ±0.022 to 0.033 RMSE |
| Verdict | the change is **inside** the noise band |

Strictly, a degradation of 0.0076 is too small to distinguish from noise on its own. It is not the validation number that condemns this experiment — it is the pre-origin evidence, where the correction lost to "do nothing" at **every** setting tested, by margins of 0.014 to 0.47. Two independent windows agreeing on the direction is what makes this a real negative rather than an unlucky draw.

## Why it failed — the interesting part

The autopsy's evidence was real. Per-series bias genuinely does persist at r = +0.557, and correcting the second half of the validation window using the first half genuinely does gain 0.068 RMSE. So why does the same idea lose here?

**Because the autopsy measured persistence *within a single model fit*, and this experiment required it to transfer *between* fits.**

- In the autopsy, both halves came from one model — `model_04`, fitted at origin d_1913. The bias measured on days 1–14 and the bias on days 15–28 belong to the same trained object.
- Here, the factors are measured on AUX-B (fitted at origin d_1885) and applied to `model_04` (fitted at origin d_1913). These are different fits, trained on different origin sets.

**INTERPRETATION:** per-series bias appears to be a property of *a particular fit* rather than a stable property of *the series*. Retrain the model on a later origin and the pattern of which products it over- and under-shoots substantially reshuffles. That is consistent with the autopsy's own finding that 99.89% of the error is variance rather than bias — there simply is not a persistent per-product tilt for a correction to lock onto.

This is a genuinely useful negative result, because it closes the largest remaining hypothesis in the project with evidence rather than assumption.

## Decision

The rule was fixed in writing before the result was seen:

> ACCEPT if ΔRMSE ≤ −0.022 **and** ΔMAE ≤ +0.020. REJECT otherwise.

Measured: ΔRMSE **+0.0076**, ΔMAE **+0.0013**.

## **REJECTED**

The selected model stands unchanged at RMSE 2.1210 / MAE 1.0319. The corrected predictions are saved separately for the record and are **not** part of the final forecast.

### What this means for the project

This was the strongest remaining hypothesis, and the one with the best prior evidence behind it. Its failure is informative: combined with the autopsy's variance decomposition, it indicates the model is close to what this feature set and this data can deliver, and that the residual error is dominated by genuinely unpredictable day-to-day demand rather than by a correctable systematic tilt.

Hypotheses 2 and 3 from the autopsy — store-level correction and a separate high-volume model — remain untested. Given that this experiment failed worst precisely on high-volume series, expectations for both should now be lower than the autopsy suggested.

---

*Experiment record: `experiments/exp_69_pre_origin_per_series_bias_correction.json`. Factors: `artifacts/exp69_correction_factors.csv`. Corrected predictions: `predictions/exp_69_bias_corrected_validation.csv`. The baseline model, its predictions, and all previous reports are unchanged.*