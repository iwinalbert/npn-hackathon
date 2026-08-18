# Research Branch — Demand Segmentation Investigation

*Autonomous research branch. Generated 2026-08-14. Experiments #72-#75, plus one read-only diagnostic.*

> ## OUTCOME: the segmentation hypothesis was **rejected on evidence**, but the investigation it triggered produced **the project's first validated improvement**. New champion: **RMSE 2.1157 / MAE 1.0287** (was 2.1210 / 1.0319).

> **Terms.** **Oracle** — a predictor allowed to see the answers, used to measure what is even possible. **Paired comparison** — two models trained and scored on the *same* window, so window difficulty cancels. **Shape** — how a series spreads demand across the week, relative to its own average.

---

## 1. Was segmentation novel, and was it promising?

**Novel: partly.** Volume *weighting* was tested and failed (+0.0165). The hurdle model is a segmentation by zero/non-zero and failed twice. Per-series correction failed in #69. But a genuinely **separate model per segment** had never been run.

**Promising: no** — and that was establishable without training anything. If segment specialisation can help, then at minimum giving each segment its own *oracle* multiplier, fitted with full knowledge of the answers, should beat a single global one. It barely does:

| Segmentation scheme | Groups | Oracle RMSE | Gain vs champion |
|---|---|---|---|
| global (single multiplier) | 1 | 2.1195 | -0.0016 |
| volume decile | 10 | 2.1191 | -0.0019 |
| intermittency quintile (zero%) | 5 | 2.1186 | -0.0025 |
| volatility quintile (CV) | 5 | 2.1194 | -0.0017 |
| spike-rate quintile | 5 | 2.1183 | -0.0028 |
| category (3) | 3 | 2.1185 | -0.0025 |
| department (7) | 7 | 2.1183 | -0.0027 |
| store (10) | 10 | 2.1157 | -0.0054 |
| store x category (30) | 30 | 2.1139 | -0.0071 |
| volume decile x weekday | 70 | 2.1181 | -0.0029 |
| PER-SERIES (30,490 groups) | 30,490 | 1.8823 | -0.2388 |

Every coarse scheme — volume, intermittency, volatility, spike-rate, category, department, store, store x category, volume x weekday — tops out at **-0.0071**, against a +/-0.022-0.033 noise floor. Only per-series has real headroom, and Experiment #69 already proved per-series correction does not transfer between fits.

> **No segmentation experiment was run, and that was the right call.** An oracle ceiling 3-5x below the noise floor cannot be beaten by a real model.

## 2. The lead that replaced it

The same diagnostic asked a second question. The error autopsy had measured two oracles: per-series **constant** 1.9818, per-series **x weekday** 1.6764. That 0.31 gap says a per-series weekly *shape* is worth far more than a per-series *level* — and the champion had no direct representation of it, because a 3,049 x 7 interaction is exactly what trees express badly.

A purely arithmetic check, no model involved:

| Predictor | RMSE |
|---|---|
| level only | 2.2430 |
| level x weekday ratio (8w history) | 2.2352 |
| level x weekday ratio (13w history) | 2.2174 |
| level x weekday ratio (26w history) | 2.2030 |
| level x weekday ratio (52w history) | 2.1851 |
| ORACLE per-series x weekday | 1.6764 |

`level x weekday-ratio(52w)` beats level-only by **-0.0578** out of sample. Real signal, recoverable from history alone. But only **10.2%** of the oracle gap is recoverable — most of 1.6764 was the oracle fitting validation-window noise — so expectations were set modestly.

## 3. Experiment #72 — per-series shape features

Four features on top of the champion's 32: `wday_ratio_52w`, `wday_ratio_13w`, `snap_lift`, `weekend_lift`. Each is a ratio to the series' own average, shrunk toward 1.0 by volume, computed only from sales at or before the origin. Leakage corruption test passed.

**Materially different from the 18 features already rejected.** Every Phase-2 feature and every Experiment #71 feature described *level*. These describe *shape*.

| | RMSE | MAE | High-volume RMSE |
|---|---|---|---|
| Champion | 2.1210 | 1.0319 | 5.9756 |
| + shape | 2.1163 | 1.0299 | 5.9565 |
| Change | -0.0047 | -0.0021 | -0.0191 |

**Formally REJECTED** against the pre-registered -0.010 threshold. But this was the first experiment in 72 to move RMSE, MAE *and* the high-volume tier the right way at once.

## 4. Experiment #73 — applying the right instrument

A single-window magnitude test is the wrong instrument for a small effect. The noise floor describes how one window's score wanders; it says nothing about how often a useless feature would win on *four* windows and *three* seeds. Criteria were fixed before running.

![Cross-window](charts/exp73_cross_window.png)

| Window | Champion | + Shape | dRMSE | dMAE |
|---|---|---|---|---|
| primary_spring_2016 | 2.1264 | 2.1148 | -0.0115 | -0.0026 |
| christmas_2015 | 2.2140 | 2.1875 | -0.0265 | -0.0037 |
| summer_2015 | 2.1375 | 2.1340 | -0.0035 | -0.0037 |
| autumn_2015 | 2.1883 | 2.1852 | -0.0031 | -0.0017 |
| **mean** | | | **-0.0112** | **-0.0029** |

| Seed | Champion | + Shape | dRMSE |
|---|---|---|---|
| 42 | 2.1264 | 2.1148 | -0.0115 |
| 7 | 2.1285 | 2.1214 | -0.0071 |
| 202 | 2.1306 | 2.1138 | -0.0168 |

**4/4 windows and 3/3 seeds — 7 of 7 paired comparisons favour shape.** Under a sign test that is p ~ 0.008. All four pre-registered criteria passed.

The champion's own RMSE varies 2.1264-2.1306 across seeds (spread 0.0042); the mean shape gain (-0.0112) is about 2.7x that. Because these are *paired* comparisons on identical windows, the between-window noise floor is not the relevant yardstick — window difficulty cancels out.

## 5. Experiment #74 — reproduction and extension

**Reproduction (required before any champion change):** a from-scratch re-run gave 2.116324 against 2.1163 recorded — drift 2.4e-05. Reproduced.

**Extension:** if shape is the mechanism, other cyclical axes should add a little. Added per-series `month_ratio` and `dom_ratio` (day-of-month).

| Window | Shape (36) | + Cycle (38) | Delta |
|---|---|---|---|
| primary | 2.1163 | 2.1157 | -0.0006 |
| christmas_2015 | 2.1837 | 2.1843 | +0.0006 |
| summer_2015 | 2.1214 | 2.1180 | -0.0035 |
| autumn_2015 | 2.1747 | 2.1512 | -0.0235 |

Wins 3 of 4 windows, so it was accepted under the same standard — but honestly, the incremental value is **-0.0006 on the primary window**, essentially nothing. The validated core of this result is the four *shape* features; the two cycle features are a rounding error carried along because they met the criterion.

## 6. New champion

| | Old champion | New champion |
|---|---|---|
| Features | 32 | 38 (32 + 4 shape + 2 cycle) |
| RMSE | 2.1210 | **2.1157** (-0.0053) |
| MAE | 1.0319 | **1.0287** (-0.0032) |
| Validation | one window | four windows + three seeds, all favourable |
| Model | `models/champion/model_04_...txt` | `models/champion/model_10_shape_cycle_final_forecast.txt` |
| Forecast | `final_forecast_28day.csv` | `final_forecast_28day_v2_shape_cycle.csv` |

The old champion, its predictions and its forecast are preserved unchanged. The new forecast passed all six structure checks and correlates 0.99434 with the previous one — a refinement, not a different answer.

## 7. What this does and does not mean

**It is a genuine, validated improvement** — the first in the project to survive multi-window and multi-seed paired testing, with a mechanism identified in advance from oracle analysis and independently confirmed by an arithmetic check.

**It is also small.** -0.0053 RMSE does not change the story about the practical ceiling. Experiment #70's finding stands: six architecturally different models have residuals correlated at 0.9897, and RMSE < 2.0 remains out of reach — 2.0 sits below the per-series oracle at 1.9818.

**The segmentation hypothesis itself was wrong**, and the way it was wrong was informative. Asking *why* a segment-level fix could not work pointed straight at what the model was actually missing: not a level adjustment per group, but a shape representation per series.

## 8. Highest-value next direction

1. **More shape axes at finer resolution.** The mechanism is validated and the weekly axis alone averaged -0.011. A per-series weekday x SNAP interaction, or a weekday profile conditioned on recency, is the natural next step and is cheap.
2. **Re-run the ensemble (#70) on top of the shape model.** #70 failed because members were near-identical (rho = 0.9897). Shape features change what the model attends to, so member diversity may now differ. One cheap check.
3. **Not worth revisiting:** segmentation (oracle-bounded at -0.007), per-series bias correction (#69), year-over-year level (#71), volume weighting, global calibration.

---

*Experiments #72-#75 and diagnostic 33 are recorded in `experiments/registry/`. All previous champion artefacts preserved unchanged.*