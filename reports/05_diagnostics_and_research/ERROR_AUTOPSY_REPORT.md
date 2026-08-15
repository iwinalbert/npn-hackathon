# Error Autopsy — Global LightGBM + Tweedie

*Deep post-mortem of the selected model (RMSE 2.1210). Generated 2026-08-14. Read-only: no model was trained or modified, no existing artifact overwritten. Everything below is computed from the 853,720 validation predictions already on disk.*

> **Terms.** **Squared error** — the miss, squared, which is what RMSE averages; it makes one big miss count more than many small ones. **Bias** — a consistent tendency to predict too high or too low. **Variance** — error that flips sign from row to row. **Oracle** — a cheating predictor allowed to see the answers, used here only to measure how much error is even removable.

---

## The four findings that matter

1. **The error is almost pure variance, not bias.** MSE splits into 0.0049 bias-squared and 4.4939 variance — bias is **0.11%** of the total. There is no systematic tilt to correct globally, which is exactly why every calibration and rescaling attempt failed.
2. **The error is extraordinarily concentrated.** **602 series out of 30,490 — 2% — carry half of all squared error.** The worst 1% of series carry 40%.
3. **What looks like horizon decay is not horizon decay.** RMSE correlates only +0.40 with how far ahead the day is, but +0.91 with how busy that day happens to be. The model is not degrading with distance; later days in this window are simply busier.
4. **Per-series bias is persistent and therefore learnable** — the single most actionable thing in this report. See Hypothesis 1.

## Where the error lives: demand volume

Rows split into ten equal groups by the series' own historical daily mean.

| Decile | Historical mean/day | Rows | Actual | Predicted | Bias | RMSE | Share of squared error |
|---|---|---|---|---|---|---|---|
| 1 | 0.01-0.09 | 85,428 | 0.196 | 0.188 | -0.007 | 0.559 | 0.69% |
| 2 | 0.09-0.15 | 85,400 | 0.295 | 0.280 | -0.016 | 0.646 | 0.93% |
| 3 | 0.15-0.22 | 85,288 | 0.395 | 0.378 | -0.018 | 0.766 | 1.30% |
| 4 | 0.22-0.32 | 85,792 | 0.529 | 0.509 | -0.020 | 0.904 | 1.83% |
| 5 | 0.32-0.44 | 85,232 | 0.668 | 0.631 | -0.038 | 1.044 | 2.42% |
| 6 | 0.44-0.62 | 85,204 | 0.845 | 0.807 | -0.039 | 1.202 | 3.20% |
| 7 | 0.62-0.87 | 85,428 | 1.074 | 1.038 | -0.036 | 1.373 | 4.19% |
| 8 | 0.88-1.32 | 85,204 | 1.506 | 1.421 | -0.085 | 1.754 | 6.83% |
| 9 | 1.32-2.44 | 85,400 | 2.302 | 2.205 | -0.097 | 2.329 | 12.06% |
| 10 | 2.44-130.95 | 85,344 | 6.621 | 6.272 | -0.349 | 5.472 | 66.54% |

The top decile alone is **66.5%** of all squared error; the top two are 78.6%. Bias grows monotonically with volume — from −0.007 in the quietest decile to −0.349 in the busiest — so the model under-predicts busy series systematically, but the effect is small next to the variance.

![Concentration](charts/autopsy_concentration.png)

## Systematic under-prediction

| | |
|---|---|
| Rows predicted below actual | 33.43% |
| Share of squared error they carry | **72.46%** |
| Average shortfall when we under-predict | 1.6487 units |
| Average excess when we over-predict | 0.7222 units |

We under-predict on only a third of rows, but those rows produce nearly three quarters of the damage. Over-prediction is frequent and cheap; under-prediction is rare and expensive. That asymmetry is the signature of a right-skewed target: most days are quiet, and the occasional busy day is very busy.

## The worst individual observations

The worst **1,000 rows — 0.12% of the data — carry 28.01% of all squared error.** Of those, **88.1% are under-predictions**, with a median actual of 35 units against a median prediction of 9.5.

Their defining feature: the median actual is **3.69x that series' own historical mean**. These are not modelling mistakes on ordinary days — they are demand spikes.

| Series | Date | Actual | Predicted | Historical mean |
|---|---|---|---|---|
| `FOODS_3_376_WI_2_evaluation` | 2016-05-06 | 179 | 39.9 | 22.0 |
| `FOODS_3_234_WI_2_evaluation` | 2016-05-03 | 143 | 23.1 | 26.2 |
| `FOODS_3_444_WI_2_evaluation` | 2016-05-15 | 106 | 0.1 | 9.1 |
| `FOODS_3_816_WI_2_evaluation` | 2016-05-04 | 107 | 6.0 | 13.1 |
| `FOODS_3_090_WI_3_evaluation` | 2016-05-06 | 0 | 99.2 | 63.5 |
| `FOODS_3_234_WI_2_evaluation` | 2016-05-11 | 115 | 21.4 | 26.2 |
| `FOODS_3_498_WI_2_evaluation` | 2016-05-07 | 100 | 8.8 | 18.1 |
| `FOODS_3_498_WI_2_evaluation` | 2016-05-11 | 103 | 13.2 | 18.1 |

Note how many are `FOODS_3` in `WI_2`. That is not coincidence — see the store table below.

Zooming out: rows where actual exceeded twice the series' historical mean are **25.87% of the data and 61.92% of the squared error**. Spikes are the problem.

## Hierarchy

| Store | Rows | Actual mean | RMSE | Share of squared error | RMSE per unit of demand |
|---|---|---|---|---|---|
| WI_2 | 85,372 | 1.804 | 3.035 | 20.47% | 1.683 |
| CA_4 | 85,372 | 0.924 | 1.440 | 4.61% | 1.559 |
| TX_1 | 85,372 | 1.142 | 1.752 | 6.82% | 1.533 |
| WI_3 | 85,372 | 1.344 | 2.035 | 9.21% | 1.514 |
| TX_3 | 85,372 | 1.366 | 2.032 | 9.18% | 1.488 |
| TX_2 | 85,372 | 1.350 | 1.959 | 8.53% | 1.451 |
| CA_1 | 85,372 | 1.570 | 2.198 | 10.74% | 1.400 |
| CA_2 | 85,372 | 1.557 | 2.008 | 8.96% | 1.290 |
| CA_3 | 85,372 | 2.054 | 2.618 | 15.23% | 1.274 |
| WI_1 | 85,372 | 1.317 | 1.676 | 6.25% | 1.273 |

**WI_2 is a genuine outlier.** It carries **20.5% of all squared error** from 10% of the rows, and it is worst on the normalised measure too: its RMSE per unit of demand (1.68) is the highest of the ten stores, while CA_3 — which sells *more* — sits at 1.27. WI_2 is not simply busy; it is genuinely more volatile.

By category, FOODS is 74.2% of squared error, and FOODS_3 alone is 53.4%. FOODS_1 stands out for bias: −0.335, the largest of any department.

## How much error is even removable?

This is the part that should change what we do next. Each row below is a predictor allowed to cheat in one controlled way. They are not models — they are ceilings.

| Predictor | RMSE | What it tells us |
|---|---|---|
| Naive: repeat the historical 28-day mean | 2.2430 | A legitimate constant-per-series predictor |
| **Our model** | **2.1210** | Where we are |
| Best single global multiplier | 2.1195 | Global calibration is worth ~0.0015. Nothing there. |
| *Team-reported benchmark* | *2.0324* | *Sits between us and the per-series oracle* |
| Oracle: each series' true window mean | 1.9818 | The best any constant-per-series forecast could do |
| Oracle: optimal multiplier per series | 1.8823 | The best per-series *recalibration* of our own model |
| Oracle: each series x weekday mean | 1.6764 | Adding a perfect weekly profile |
| Oracle: perfect on the worst 1% of rows | 1.4200 | If spikes were solved |
| Oracle: perfect on the >3/day tier | 1.3189 | If the busy 7.7% were solved |

![Oracles](charts/autopsy_oracles.png)

### Reading the budget

Our model sits **0.122 better than a naive historical constant** and **0.239 worse than a per-series recalibration of itself that knows the answers.** That second number is the realistic headroom, and it is the largest single opportunity this autopsy found.

It also settles a question the project has been circling. The team's reported 2.0324 lies **between** our 2.1210 and the per-series oracle at 1.9818. It is therefore not physically impossible to reach by legitimate means — but reaching it would require capturing about 37% of the entire gap to a model that already knows each series' correct scaling.

## Ranked hypotheses for reducing RMSE

### 1. Per-series bias correction — HIGH confidence, largest measured upside

**The evidence is unusually strong for a hypothesis that has not been tested yet.**

- The oracle per-series rescale reaches **1.8823** — a headroom of 0.239.
- That bias **persists**: splitting the validation window in half and correlating each series' correction factor between halves gives **r = +0.557** (multiplicative) and **+0.579** (additive), across 20,280 series with enough volume to measure. Persistent bias is learnable bias; noise would give r ≈ 0.
- A direct demonstration: correction factors derived from days 1–14, applied to days 15–28, move RMSE from **2.1846 to 2.1165** — a gain of **0.0681**.

![Bias stability](charts/autopsy_bias_stability.png)

For scale: that demonstrated gain is roughly **twenty times larger than anything the entire optimization campaign produced**, and about double the ±0.022–0.033 window-to-window noise floor we measured. It is the first candidate that clears the noise bar by a comfortable margin.

> **The honest caveat, stated up front.** The demonstration corrects the second half of the window using the first half of the *same* window — a gap of days. In production the correction would have to be learned from a period *before* the forecast origin, so the gap is 14–42 days instead. Whether the bias survives that longer gap is exactly the experiment to run, and it is a clean one: fit per-series factors on d_1886–d_1913, apply to d_1914–d_1941, change nothing else.

Practical guards: shrink the factor toward 1.0 for low-volume series (the split-half test already restricted to series with meaningful volume), and clip the factor to something like [0.5, 2.0] so a single odd series cannot misbehave.

### 2. Store-level correction, especially WI_2 — MEDIUM confidence, robust

WI_2 carries **20.5% of all squared error** from 10% of rows and has the worst RMSE-per-unit-demand of the ten stores. A store-level (or store×department) correction is a much lower-variance version of Hypothesis 1 — far fewer parameters, so far less risk of fitting noise. Worth running alongside H1 as the conservative variant.

### 3. A separate model for the top volume decile — MEDIUM confidence

Decile 10 is 66.5% of squared error. Phase 3 already showed that *weighting* those rows inside one model makes things worse — but a genuinely separate model is a different intervention, and the concentration is extreme enough to justify one attempt. Expect modest returns.

### 4. Spike modelling — LOW confidence, likely a dead end

Spikes dominate the error (61.92% of it), so this looks attractive — until you check whether they are predictable. They are barely calendar-linked:

| Condition | Spike rate |
|---|---|
| Weekend | 0.311 |
| Weekday | 0.238 |
| SNAP day | 0.272 |
| Non-SNAP day | 0.252 |
| Event day | 0.280 |
| Ordinary day | 0.255 |

The lifts are small, and the model already has every one of these features. With no promotion field and no inventory field in the dataset, the remaining spikes are not predictable from anything we hold. This is where the irreducible error lives.

### 5. Per-horizon models — LOW value, deprioritise

RMSE correlates +0.40 with horizon but +0.91 with the day's demand level. Week 1 averages 1.8863 and week 4 2.0678, but that gap is explained by *which days those are*, not by forecast distance. There is little genuine decay to fix.

### 6. Global calibration or rescaling — CLOSED

Bias is 0.11% of MSE, and the best possible single multiplier is worth 0.0016. This has now been tested three separate ways and found empty each time. It should not be attempted again.

## Summary of recommendations

| Rank | Hypothesis | Evidence strength | Measured/bounded upside | Verdict |
|---|---|---|---|---|
| 1 | Per-series bias correction from a pre-origin window | **Strong** — r=+0.56 persistence, oracle 1.8823 | ~0.068 demonstrated within-window | **RUN FIRST** |
| 2 | Store / store×dept correction (WI_2 focus) | Medium — 20.5% of error in one store | smaller but lower variance | **RUN as the safe variant** |
| 3 | Separate model for top volume decile | Medium — 67% of error | unknown | Optional |
| 4 | Spike modelling | Weak — spikes barely calendar-linked | probably ~0 | Skip |
| 5 | Per-horizon models | Weak — decay is an artefact | ~0 | Skip |
| 6 | Global calibration | **Closed** — tested 3x | 0.0015 | Do not repeat |

> **One caution before anyone runs these.** Hypotheses 1–3 all correct the model using recent observed error. That is legitimate — it uses only data before the forecast origin — but it is also exactly the kind of adjustment that looked good on an inner window and evaporated on the primary window four separate times in the optimization campaign. Every one of these must be fitted on a pre-origin window and evaluated once on the untouched primary window, and judged against the ±0.022–0.033 noise floor rather than against zero.

---

*Read-only autopsy. Source: 853,720 validation predictions from `model_04_tweedie_recency_listing`. Full numbers in `artifacts/error_autopsy.json`; the 200 worst rows in `artifacts/autopsy_worst_200_rows.csv`. No model was trained or modified.*