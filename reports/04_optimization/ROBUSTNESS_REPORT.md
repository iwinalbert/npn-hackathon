# Robustness Across Windows

*Phase 9 — the same models scored on four different 28-day periods. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## Why this is the most important phase

Several changes in this campaign looked like improvements on one window and reversed on another. This phase measures how large that window-to-window variation actually is — which tells us how big a difference has to be before it means anything.

## Results (each model retrained per window)

| Window | Dates | l1 | tweedie_1_1 | tweedie_1_5 |
|---|---|---|---|---|
| autumn_2015 | 2015-10-02 .. 2015-10-29 | 2.3087 | 2.1869 | 2.1733 |
| christmas_2015 | 2015-12-12 .. 2016-01-08 | 2.3005 | 2.1851 | 2.1731 |
| primary_spring_2016 | 2016-04-25 .. 2016-05-22 | 2.2432 | 2.1210 | 2.1263 |
| summer_2015 | 2015-07-16 .. 2015-08-12 | 2.2948 | 2.1405 | 2.1573 |

![Robustness](charts/phase9_robustness.png)

## Consistency summary

| Model | Mean RMSE | Std dev | Worst window | Mean MAE | MAE std |
|---|---|---|---|---|---|
| tweedie_1_5 | 2.1575 | 0.0221 | 2.1733 | 0.9767 | 0.0472 |
| tweedie_1_1 | 2.1584 | 0.0329 | 2.1869 | 0.9818 | 0.0457 |
| l1 | 2.2868 | 0.0296 | 2.3087 | 0.9068 | 0.0464 |

## The finding that reframes the whole project

RMSE varies by roughly **±0.033** across windows for the same model. Almost every "improvement" tested in this campaign was smaller than that.

Put plainly: the differences we have been chasing are smaller than the natural variation between one month and the next. That is why inner-window gains kept failing to transfer — they were noise, and our discipline of selecting on a separate window is what caught them.

It also puts the comparison with the team's benchmark in perspective. The disputed gap is 0.0886, only about 2.7x this natural window variation — and their validation window is unknown.
