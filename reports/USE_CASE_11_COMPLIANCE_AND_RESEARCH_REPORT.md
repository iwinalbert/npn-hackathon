# Use Case 11 — Compliance Audit and Hierarchical Forecasting Research

**Project:** NPN_HACKATHON — Walmart M5 store-item demand forecasting
**Branch:** Stage 7, Experiments #80–#84
**Scope:** audit the shipped system against every clause of Use Case 11, then
investigate the one requirement it did not yet satisfy. The protected champion
was never modified, retrained over, or replaced.

> **Use Case 11.** *"Build a forecasting model that handles hierarchical
> aggregation, external covariates (price/promo/holiday) and intermittent
> demand, producing accurate 28-day-ahead forecasts per store/item."*

---

# 1. Verdict

**B. KEEP CHAMPION.**

The shipped model stands unchanged:

```
0.60 x Direct LightGBM Tweedie(1.1), 38 features
0.40 x Recursive LightGBM Tweedie(1.1), 32 features
primary window   RMSE 2.0929   MAE 1.0395   WAPE 0.7205   bias -0.0224
```

Three new directions were investigated. Two were rejected before any model was
trained, on measured headroom. The third — formal hierarchical reconciliation —
was trained and validated across four disjoint windows under **two** independent
protocols, and both were rejected on the same pre-registered mechanism
criterion, despite each improving mean RMSE, mean MAE and mean high-volume RMSE
simultaneously.

| Experiment | Protocol | Wins | Mean ΔRMSE | Mechanism | Decision |
|---|---|---|---|---|---|
| #81 FULL | α fixed at inner-window optimum | 3/4 | −0.0074 | 2/4 | REJECTED |
| #81 DEMEANED | α fixed at inner-window optimum | 3/4 | −0.0088 | 2/4 | REJECTED |
| #82 FULL | α per origin, preceding window | 3/4 | −0.0061 | 2/4 | REJECTED |
| #82 DEMEANED | α per origin, preceding window | 3/4 | −0.0093 | 2/4 | REJECTED |

Section 6 explains why that rejection is the right call, and Section 6.1 shows
that the natural fix — choosing α adaptively from data available at forecast
time — moves α the *wrong way* on the one window where it matters.

The champion was reproduced from scratch on all four evaluation windows during
this branch and matched its recorded scores to four decimals every time
(2.0929 / 2.1547 / 2.1074 / 2.1496). That is an independent reproducibility
result, not a claim inherited from earlier stages.

---

# 2. Requirement gap analysis

Assessed against the code, features, artefacts and experiment records as they
stood **before** this branch began.

| # | Requirement | Status before | Why |
|---|---|---|---|
| A | Hierarchical aggregation | **~ PARTIAL** | The hierarchy entered the model only as five categorical features (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`). No forecast was ever produced at an aggregate level, no coherence property was stated, no reconciliation was attempted, and aggregate accuracy was quoted at four ad-hoc levels in the README without a per-level measurement. |
| B1 | Price covariates | **✓ SATISFIED** | `sell_price` (target-day), `recent_avg_price` (origin-relative), `price_rel_to_recent_avg`, `price_is_missing`. Price dynamics were additionally tested and rejected (`opt_02_v2_C_price`, 2.1281 vs 2.1210). |
| B2 | Promotion covariates | **~ PARTIAL** | **M5 contains no promotion field.** Nothing in `calendar.csv`, `sales_train_*.csv` or `sell_prices.csv` flags a promotion, display, feature ad or coupon. Promotion is observable only through price. |
| B3 | Holiday / calendar covariates | **✓ SATISFIED** | `event_name_1/2`, `event_type_1/2`, `wday`, `month`, `year`, `is_weekend`, plus state-matched SNAP. |
| C | Intermittent demand | **~ PARTIAL** | Tweedie(1.1) selected by a power sweep, and the hurdle family tested twice and rejected. But Croston, SBA and TSB — named in the brief — had never been tested, and adequacy had only been checked in aggregate, never regime by regime. |
| D | 28-day-ahead forecasting | **✓ SATISFIED** | Fixed-origin direct 28-day design; every history feature frozen at the origin; 46 foundation checks; corruption test overwriting all post-origin sales with 9999 and requiring bit-identical features. |
| E | Per store-item forecasting | **✓ SATISFIED** | 30,490 series × 28 days = 853,720 predictions, none excluded, none weighted. |

Three gaps therefore drove this branch: **A** (no hierarchical forecasting at
all), **B2** (promotion never represented explicitly) and **C** (classical
intermittent methods never measured).

---

# 3. Hierarchical aggregation

## 3.1 Coherence

The system forecasts only the bottom level, so every aggregate is a sum of
bottom-level forecasts. That is bottom-up reconciliation and it is **trivially
coherent** — there is no incoherence to repair, because no independent aggregate
forecast exists to disagree with. Coherence was never the open question.

The open question was whether information living at an aggregate level could
reduce **bottom-level** error, which is what the project is scored on.

## 3.2 Aggregate accuracy at all 12 M5 levels

Measured from the champion's own residuals (`uc11_hierarchy_levels.csv`):

| Level | Groups | Bottom-up WAPE | Accuracy | Common share of squared error |
|---|---|---|---|---|
| L1 total | 1 | 0.0283 | 97.2% | 0.06% |
| L2 state | 3 | 0.0396 | 96.0% | 0.13% |
| L3 store | 10 | 0.0530 | 94.7% | 0.23% |
| L4 category | 3 | 0.0379 | 96.2% | 0.12% |
| L5 department | 7 | 0.0518 | 94.8% | 0.26% |
| L6 state × cat | 9 | 0.0514 | 94.9% | 0.23% |
| L7 state × dept | 21 | 0.0680 | 93.2% | 0.44% |
| L8 store × cat | 30 | 0.0692 | 93.1% | 0.42% |
| L9 store × dept | 70 | 0.0891 | 91.1% | 0.78% |
| L10 item | 3,049 | 0.2920 | 70.8% | **16.76%** |
| L11 item × state | 9,147 | 0.4446 | 55.5% | **36.36%** |
| L12 store-item | 30,490 | 0.7236 | 27.6% | — |

"Common share" is exact, not estimated. For a group of *n* members with
discrepancy *D*, the bottom-level squared error splits as
`sum(e²) = D²/n + sum((e - D/n)²)`. The first term is the entire portion any
aggregate-level information could ever address; the second is idiosyncratic and
provably invisible to it.

## 3.3 Oracle bounds — why the coarse levels were rejected without training

Each oracle is handed the **true** aggregate and allowed to redistribute it
perfectly. No reconciliation method can beat its own oracle.

| Level | Equal-share | Proportional (top-down) | In-sample-fitted shares |
|---|---|---|---|
| L1 total | −0.0007 | −0.0024 | −0.0655 |
| L3 store | −0.0024 | −0.0100 | −0.0985 |
| L5 department | −0.0028 | −0.0047 | −0.1060 |
| L9 store × dept | −0.0084 | −0.0221 | −0.1412 |
| L10 item | −0.1864 | **−0.2272** | −0.4004 |
| L11 item × state | −0.4331 | **−0.5154** | −0.6868 |

Knowing the true chain total *perfectly* is worth −0.0007 RMSE. Knowing the true
store × department daily total *perfectly* is worth −0.0221. **Every level above
item is dead**, and no amount of modelling effort changes that, because the
information simply is not there. That result alone retired bottom-up, top-down,
middle-out and MinT at those levels without a single training run.

The item-bearing levels are different, and were investigated properly.

## 3.4 Experiments #80 / #80b / #80c — selection and controls (inner window)

All selection was done on the inner window (origin d_1885, targets
d_1886–d_1913), strictly before the primary window's targets and before the
final forecast origin.

- **Objective.** Tweedie(1.1), which wins at the bottom level, is clearly *wrong*
  at the aggregate level: item-level RMSE 10.79 versus 8.18 for L2. Aggregates
  are neither sparse nor zero-inflated. L2 was selected.
- **Level.** item (−0.0160) beat item × state (−0.0137) and store × dept
  (−0.0122).
- **Leakage.** A corruption test rebuilt all 30 aggregate features from a panel
  whose post-origin sales were replaced with 9999. All 30 came back identical;
  the target changed, so the test could not pass vacuously.
- **Negative controls.** Three were run. One fired: an oracle global rescale was
  worth −0.0230, *more* than the method's −0.0160.

That last result forced Experiment #80c, which decomposed the correction:

| Variant | RMSE | ΔRMSE |
|---|---|---|
| champion | 2.0677 | — |
| global rescale (oracle) | 2.0447 | −0.0230 |
| full reconciliation | 2.0517 | −0.0160 |
| **item-specific component only** | 2.0624 | **−0.0053** |
| rescale + item-specific | 2.0415 | −0.0262 |

Two thirds of the headline gain was the item model happening to undo a level
anomaly specific to that window: the champion over-forecasts the inner window by
**+0.0817 units per row**, while on the primary window it is calibrated
(bias −0.0224, and the same rescale oracle is worth −0.0000 there). Promoting on
the headline number would have shipped a calibration artefact fitted to one
atypical month.

Both variants — FULL and DEMEANED (global component divided out, so it can only
redistribute between items) — were carried forward with α fixed at their
inner-window optima, 0.55 and 0.35.

## 3.5 Experiment #81 — four-window validation

```
P'(s,d) = clip( P(s,d) + a * (P(s,d) / F(i,d)) * (Ahat(i,d) - F(i,d)), 0, None )
```

which is forecast-proportions top-down of the item level blended into the
bottom-up forecast — textbook middle-out reconciliation.

**FULL, α = 0.55**

| Window | Champion | Reconciled | ΔRMSE | ΔMAE | ΔHigh-vol |
|---|---|---|---|---|---|
| primary_spring_2016 | 2.0929 | 2.0913 | −0.0017 | −0.0074 | +0.0028 |
| christmas_2015 | 2.1547 | 2.1705 | **+0.0158** | +0.0001 | **+0.0820** |
| summer_2015 | 2.1074 | 2.0888 | −0.0187 | −0.0156 | −0.0728 |
| autumn_2015 | 2.1496 | 2.1246 | −0.0250 | −0.0127 | −0.1164 |
| **Mean** | | | **−0.0074** | **−0.0089** | **−0.0261** |

**DEMEANED, α = 0.35**

| Window | Champion | Reconciled | ΔRMSE | ΔMAE | ΔHigh-vol |
|---|---|---|---|---|---|
| primary_spring_2016 | 2.0929 | 2.0890 | −0.0040 | −0.0011 | −0.0106 |
| christmas_2015 | 2.1547 | 2.1599 | **+0.0052** | +0.0041 | **+0.0320** |
| summer_2015 | 2.1074 | 2.0916 | −0.0159 | −0.0033 | −0.0660 |
| autumn_2015 | 2.1496 | 2.1289 | −0.0208 | −0.0034 | −0.0961 |
| **Mean** | | | **−0.0088** | **−0.0009** | **−0.0352** |

| Criterion | FULL | DEMEANED |
|---|---|---|
| H1 wins ≥ 3/4 | PASS (3/4) | PASS (3/4) |
| H2 mean ΔRMSE ≤ −0.005 | PASS (−0.0074) | PASS (−0.0088) |
| H3 high-volume not worse | PASS (−0.0261) | PASS (−0.0352) |
| **H4 mechanism ≥ 3/4** | **FAIL (2/4)** | **FAIL (2/4)** |
| H5 leakage checks | PASS | PASS |
| **Decision** | **REJECTED** | **REJECTED** |

## 3.6 Why it fails, and why the failure is informative

The mechanism criterion asks whether the item-level model actually beats the
champion's own bottom-up sum. It does so on only two of four windows — and the
per-window outcome tracks that margin almost perfectly:

| Window | Item model vs bottom-up | ΔRMSE (FULL) |
|---|---|---|
| christmas_2015 | 10.6% **worse** | +0.0158 |
| primary_spring_2016 | 1.3% worse | −0.0017 |
| summer_2015 | 0.3% **better** | −0.0187 |
| autumn_2015 | 3.5% better | −0.0250 |

This is exactly what reconciliation theory predicts, and it is the reason to
reject rather than a reason to hope. The physics is right; the input is not
reliably available. The champion's bottom-up sum is *already* a strong
item-level forecaster, and a dedicated item-level model beats it only sometimes.

This **converges with earlier project evidence from a completely different
angle.** Experiment #76 measured a joint oracle of only −0.0055 for cross-store
and cross-item *features* and rejected them before training. This branch
attacked the same information channel through formal reconciliation with a
separately trained aggregate model and hit the same wall. Two independent
methods agreeing is far stronger than either alone: the bottom-level model
already extracts what cross-store information this dataset contains, through
global pooling and the `item_id` categorical.

---

# 4. External covariates

## 4.1 Availability at the real forecast origin (d_1941)

Checked against the raw files for d_1942–d_1969, the genuinely unknown window.

| Covariate | Available for all 28 days | Needs forecasting? | Source |
|---|---|---|---|
| `wday`, `month`, `year`, `is_weekend` | 28/28 | No | `calendar.csv` |
| `event_name_1/2`, `event_type_1/2` | 28/28 | No | `calendar.csv` |
| `snap_CA/TX/WI` | 28/28 | No | `calendar.csv` |
| `sell_price` | **30,490 / 30,490 series, all 5 future weeks** | No | `sell_prices.csv` |
| **promotion** | **0/28 — field does not exist** | n/a | — |

Events genuinely present in the forecast window: MemorialDay, NBAFinalsStart,
NBAFinalsEnd, Ramadan starts, Father's day. Future price is used legitimately:
the dataset publishes it, so the forecasting scenario genuinely assumes it is
known at forecast time. No future actual-sales-derived quantity is used
anywhere.

## 4.2 Residual structure — is anything left to explain?

A covariate whose residuals are already flat has nothing left to give.

**Discount against the 52-week regular price** (reference frozen at the origin —
the retail-standard promotion proxy, and a genuinely different quantity from the
week-on-week price changes already rejected in `opt_02_v2_C_price`):

| Bin | n | Actual | Predicted | Residual | Share of sq. error |
|---|---|---|---|---|---|
| price ≥ regular | 2,705 | 1.999 | 2.089 | −0.090 | 0.49% |
| 0–2% off | 770,926 | 1.412 | 1.412 | +0.000 | 88.94% |
| 2–5% off | 19,290 | 1.628 | 1.579 | +0.049 | 2.48% |
| 5–10% off | 29,506 | 1.407 | 1.291 | +0.115 | 2.52% |
| 10–20% off | 27,877 | 2.230 | 2.119 | +0.111 | 5.13% |
| 20–30% off | 2,210 | 1.035 | 1.050 | −0.015 | 0.17% |
| 30%+ off | 1,206 | 0.513 | 0.755 | −0.242 | 0.26% |

89% of rows sit in the "no meaningful discount" bin with a residual of +0.000.
Events (+0.014 vs −0.025) and SNAP (+0.007 vs +0.010) are equally flat.

**Oracle for a perfect discount correction**, fitted on the evaluation window
itself and therefore unreachable:

| Correction | ΔRMSE |
|---|---|
| per discount bin | −0.0002 |
| per weeks-at-current-price bin | −0.0012 |
| per discount × price-age cell | −0.0019 |

**Rejected without training.** A promotion proxy cannot pay for itself when a
perfect version of it is worth −0.0002.

One real but small effect is documented rather than fixed: items repriced within
the last week are badly under-forecast (actual 1.706 vs predicted 0.711,
residual +0.995) — but they are 0.46% of rows, so perfect treatment is worth
−0.0012 overall.

---

# 5. Intermittent demand

Syntetos-Boylan classification from 728 days of pre-origin history
(ADI cut 1.32, CV² cut 0.49):

| Regime | Series | Rows | Zero % | Mean actual | RMSE | Bias | Share of sq. error |
|---|---|---|---|---|---|---|---|
| smooth | 2,216 | 62,048 | 11.7 | 5.993 | 4.1661 | −0.0077 | 28.82% |
| erratic | 673 | 18,844 | 14.7 | 5.467 | 5.3957 | −0.0981 | 14.68% |
| intermittent | 23,247 | 650,916 | 60.9 | 0.831 | 1.2790 | −0.0093 | 28.50% |
| lumpy | 4,342 | 121,576 | 47.7 | 1.774 | 2.9326 | +0.0133 | 27.98% |
| never sold | 12 | 336 | 67.3 | 0.530 | 1.0428 | −0.4199 | 0.01% |

## 5.1 Croston, SBA and TSB — measured, not assumed

Each fitted on pre-origin history only (α = 0.1), producing one rate per series
across all 28 days, which is what these methods are.

| Regime | Champion | Croston | SBA | TSB | Rolling mean 28 |
|---|---|---|---|---|---|
| smooth | **4.1661** | 4.5121 | 4.4982 | 4.5641 | 4.6191 |
| erratic | **5.3957** | 5.5322 | 5.5126 | 5.5913 | 5.7174 |
| intermittent | **1.2790** | 1.3781 | 1.3741 | 1.3417 | 1.3625 |
| lumpy | **2.9326** | 3.1407 | 3.1343 | 3.0217 | 3.0706 |
| **ALL** | **2.0920** | 2.2380 | 2.2317 | 2.2084 | 2.2430 |

**There is no regime in which any classical intermittent method beats the
champion.** Not one.

## 5.2 Is Tweedie adequate?

A wrong likelihood shows up as regime-dependent bias. Bias is flat: −0.13% of
mean demand on smooth, −1.79% erratic, −1.12% intermittent, +0.75% lumpy. The
only outlier is the 12 "never sold" series (0.01% of squared error).

## 5.3 Oracles for specialisation

| Upper bound (fitted on the evaluation window) | ΔRMSE |
|---|---|
| per-regime multiplicative rescale | −0.0008 |
| per-regime blend with Croston | −0.0048 |
| per-regime blend with SBA | −0.0046 |
| per-regime blend with TSB | −0.0003 |
| per-regime blend with rolling mean 28 | +0.0000 |

Combined with the existing evidence — hurdle tested twice (2.1267, 2.1241,
both worse), intermittency-quintile segmentation oracle −0.0025 — the single
Tweedie head is confirmed adequate. **Requirement C is satisfied by Tweedie, and
that is now a measured claim rather than an assumption.**

---

# 6. Why a candidate that improved three metrics was still rejected

Experiment #81's DEMEANED variant improved mean RMSE (−0.0088), mean MAE
(−0.0009) and mean high-volume RMSE (−0.0352) at once — something almost nothing
in this project's 79 prior experiments achieved. It was rejected anyway. The
reasoning matters more than the number.

1. **The pre-registered mechanism criterion failed.** H4 was fixed before the
   run precisely to prevent accepting a gain whose cause cannot be verified.
   Waiving it after seeing a favourable mean is the exact failure mode the
   research brief forbids.
2. **The gain is not attributable.** It tracks whether the item-level model
   happens to beat the bottom-up sum that month — knowable only after the fact.
   The expected gain at the real forecast origin is therefore not −0.0088; it is
   a draw from a distribution containing +0.0158.
3. **It harms the operationally most important window.** Christmas is where
   retail forecasting matters most, and FULL costs +0.0158 RMSE and +0.0820
   high-volume RMSE there.
4. **The incumbent was held to a higher bar.** The shipped blend was accepted on
   4/4 windows, 3/3 seeds, with a negative control isolating the cause. Promoting
   a 3/4 result with a failed mechanism test would lower the project's evidential
   standard, not raise its accuracy.

**What would overturn this.** An item-level model that beats the bottom-up sum
on 3+ of 4 windows, or a leakage-safe pre-origin signal that reliably predicts
which regime the next 28 days fall into. Experiment #82 tested the second of
these directly, and it failed.

## 6.1 Experiment #82 — the deployable per-origin α, and why it does not rescue the method

At any forecast origin *T* the 28 days ending at *T* are fully observed, so α can
be chosen on `(T-28, T]` and applied to `(T, T+28]`. At the real forecast origin
d_1941 the selection window is d_1914–d_1941 — exactly the data a practitioner
standing on 2016-05-22 would hold. This is a complete, leakage-safe, deployable
rule, and it was pre-registered before Experiment #81 produced any result.

| Window | α selected on | FULL α | FULL ΔRMSE | DEMEANED α | DEMEANED ΔRMSE |
|---|---|---|---|---|---|
| primary_spring_2016 | d_1885 | 0.55 | −0.0017 | 0.35 | −0.0040 |
| christmas_2015 | d_1750 | 0.60 | **+0.0198** | 0.50 | **+0.0145** |
| summer_2015 | d_1601 | 0.80 | −0.0151 | 0.65 | −0.0187 |
| autumn_2015 | d_1679 | 0.80 | −0.0275 | 0.75 | −0.0291 |
| **Mean** | | | **−0.0061** | | **−0.0093** |

| Criterion | FULL | DEMEANED |
|---|---|---|
| K1 wins ≥ 3/4 | PASS (3/4) | PASS (3/4) |
| K2 mean ΔRMSE ≤ −0.005 | PASS (−0.0061) | PASS (−0.0093) |
| K3 high-volume not worse | PASS (−0.0229) | PASS (−0.0406) |
| K4 α spread ≤ 0.40 | PASS (0.25) | PASS (0.40) |
| **K5 mechanism ≥ 3/4** | **FAIL (2/4)** | **FAIL (2/4)** |
| **Decision** | **REJECTED** | **REJECTED** |

**The decisive observation is not the mean — it is which way α moved.** Going
into christmas_2015, the one window where the item-level model collapses (10.98
versus 9.92 for the bottom-up sum), the preceding window told the rule to be
*more* aggressive: α rose from 0.55 to 0.60 (FULL) and from 0.35 to 0.50
(DEMEANED). The loss consequently grew from +0.0158 to +0.0198 and from +0.0052
to +0.0145.

A working detector would have shrunk α toward zero there. Instead the adaptive
rule amplified the harm, because **the item model's advantage does not persist
from one 28-day window to the next.** The selected α also swings across half the
grid (0.55 → 0.60 → 0.80 → 0.80) without tracking the quantity it needs to.

Against the fixed-α protocol the adaptive one is a wash — FULL is slightly worse
(−0.0061 vs −0.0074), DEMEANED slightly better (−0.0093 vs −0.0088) — so the
extra machinery buys nothing.

This is the strongest form the negative result could take. The rejection is not
"we could not tune it well enough"; it is that **the quantity you would have to
tune on is not predictable from the information available at forecast time.**

---

# 7. Use Case 11 final compliance matrix

| Requirement | Status | Evidence |
|---|---|---|
| Hierarchical aggregation | **✓** | Bottom-up coherent by construction; accuracy measured at all 12 M5 levels (`uc11_hierarchy_levels.csv`); reconciliation investigated formally at 3 levels with exact oracle bounds; 4-window validation (#81); rejected on evidence, not omitted |
| External price covariates | **✓** | 4 price features; 30,490/30,490 series priced for all 28 future days; residuals flat; perfect-correction oracle −0.0002 |
| Promotion covariates | **~** | **No promotion field exists in M5.** Represented by price-relative proxies; discount-vs-52-week-regular audited, oracle −0.0002; rejected on measurement |
| Holiday / calendar covariates | **✓** | 4 event fields + weekday/month/year/weekend + state-matched SNAP; 28/28 days available at the real origin; residuals flat |
| Intermittent demand | **✓** | Tweedie(1.1) by power sweep; hurdle rejected twice; Croston/SBA/TSB now measured and beaten in every regime; bias flat across regimes; specialisation oracle −0.0008 |
| 28-day-ahead forecasting | **✓** | Fixed-origin direct 28-day; 46 foundation checks; 9999-corruption test on both the champion's builder and the 30 new aggregate features; recursive rollout structurally verified per window |
| Per store-item forecasting | **✓** | 853,720 predictions (30,490 × 28), none excluded or weighted |
| Leakage-safe | **✓** | Structural + empirical corruption tests; all training origins satisfy `origin + horizon ≤ validation_origin`; α and every constant selected on pre-origin windows only |
| Validated accuracy | **✓** | 4 disjoint windows, multiple seeds, paired comparisons on identical rows; champion reproduced to 4 decimals on all 4 windows in this branch |

**One requirement remains PARTIAL, and it cannot be made otherwise:** M5 has no
promotion field. Marking it satisfied would require inventing a covariate the
data does not contain.

---

# 8. What must not change

1. **The champion configuration.** `0.60 × direct(38f) + 0.40 × recursive(32f)`,
   Tweedie(1.1), 400 rounds, seed 42. Reproduced exactly four times in this
   branch.
2. **The blend weight w = 0.60**, selected on an inner window in #77.
3. **The Tweedie objective at the bottom level.** Now confirmed against
   Croston/SBA/TSB in every demand regime, not just in aggregate.
4. **The fixed-origin direct-28-day design and its corruption tests.**
5. **The validation protocol** — four disjoint windows, pre-registered criteria,
   selection on pre-origin windows only.
6. **The raw data.** All five MD5s re-verified unchanged during this branch.

## Directions now closed on measured evidence

| Direction | Evidence | Verdict |
|---|---|---|
| Reconciliation above item level | true-aggregate oracle ≤ −0.0221 | dead |
| Item-level reconciliation, fixed α | 3/4 windows but mechanism 2/4 | rejected |
| Item-level reconciliation, adaptive α | 3/4 windows, mechanism 2/4, α moves the wrong way on the failing window | rejected |
| Cross-store / cross-item features | joint oracle −0.0055 (#76) | dead |
| Promotion / discount covariates | perfect-correction oracle −0.0002 | dead |
| Croston / SBA / TSB | lose in every regime | dead |
| Regime segmentation | oracle −0.0008 | dead |
| Hurdle models | 2.1267 / 2.1241, both worse | dead |
| Global or per-series recalibration | −0.0014 global; #69 failed | dead |

---

# 9. Limitations

1. **The promotion covariate does not exist** in this dataset and no feature
   here recovers it.
2. **Stockouts remain unobservable.** A zero can mean "nobody wanted it" or "it
   was not on the shelf", and nothing distinguishes them.
3. **Item-level reconciliation is rejected, not disproven.** It helps materially
   on two of four windows, and both protocols improved mean RMSE, MAE and
   high-volume RMSE together. What defeats it is that its benefit is not
   predictable at forecast time — Experiment #82 showed the obvious pre-origin
   signal moves α the wrong way precisely when it matters. A genuinely
   predictive regime signal would revive the direction.
4. **α selected on a single window proved unreliable**, and the adaptive
   replacement was no better. Both are documented rather than hidden.
5. **Point forecasts only.** No predictive intervals, which inventory decisions
   want.
6. **Documentation drift.** `README.md`, `PROJECT_INDEX.md`,
   `models/champion/README.md` and `predictions/final_forecast/README.md` still
   describe the 32-feature single model at RMSE 2.1210 as the champion, and the
   final-forecast README quotes 2.1210 as the estimate for the delivered window.
   The delivered file is `final_forecast_28day_v3_diversity_blend.csv` from the
   0.60/0.40 blend at 2.0929. These files were left untouched by this branch;
   correcting them is a separate, deliberate decision.

---

# 10. Artefacts produced

New code (nothing existing was modified):

| Path | Purpose |
|---|---|
| `pipeline/aggregate_level.py` | aggregate-level panel + leakage-safe feature builder |
| `pipeline/champion_blend.py` | faithful reproduction of the shipped champion, with caching |
| `scripts/07_usecase11/50_hierarchy_headroom.py` | 12-level oracle diagnostic |
| `scripts/07_usecase11/51_reconciliation_threshold.py` | break-even curve |
| `scripts/07_usecase11/52_aggregate_model_target.py` | required aggregate accuracy |
| `scripts/07_usecase11/53_exp80_item_level_probe.py` | inner-window go/no-go |
| `scripts/07_usecase11/54_exp80b_level_sweep.py` | level sweep + negative controls |
| `scripts/07_usecase11/56_exp80c_orthogonality.py` | level vs cross-store decomposition |
| `scripts/07_usecase11/57_exp81_four_window_validation.py` | four-window validation |
| `scripts/07_usecase11/59_exp82_adaptive_alpha.py` | per-origin α protocol |
| `scripts/07_usecase11/55_covariate_audit.py` | requirement-6 audit |
| `scripts/07_usecase11/58_intermittency_audit.py` | requirement-7 audit |
| `scripts/07_usecase11/60_record_experiments.py` | registry bookkeeping |

Artefacts are `experiments/artifacts/uc11_*`; cached champion reproductions are
`predictions/uc11_cache/`. No file under `data/raw/`, `models/champion/`,
`predictions/final_forecast/` or `experiments/registry/` was modified.
