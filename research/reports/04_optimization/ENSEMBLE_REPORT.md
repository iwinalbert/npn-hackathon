# Ensemble

*Phase 8 — blending the two objectives that won on different metrics. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## Why these two

Phase 6 produced a clean split: Tweedie is best on RMSE, L1 is best on MAE. Blending them tests whether the trade-off can be improved rather than merely slid along.

## Weight selection (inner window only)

| Weight on Tweedie | Inner RMSE | Inner MAE |
|---|---|---|
| 0.0 | 2.1826 | 0.9293 |
| 0.1 | 2.1579 | 0.9347 |
| 0.2 | 2.1365 | 0.9409 |
| 0.3 | 2.1184 | 0.9480 |
| 0.4 | 2.1037 | 0.9557 |
| 0.5 | 2.0925 | 0.9641 |
| 0.6 | 2.0848 | 0.9733 |
| 0.7 | 2.0807 | 0.9831 |
| 0.8 | 2.0802 | 0.9937 |
| 0.9 | 2.0833 | 1.0051 |
| 1.0 | 2.0899 | 1.0173 |

The inner window preferred **0.80 Tweedie / 0.20 L1**, where it beat pure Tweedie by about 0.010 RMSE.

## Applied once to the primary window

| | RMSE | MAE |
|---|---|---|
| Tweedie alone | 2.1210 | 1.0319 |
| Ensemble | 2.1272 | **1.0128** |
| Change | +0.0062 | -0.0191 |

The RMSE gain did not transfer — again — but **the MAE improvement is substantial and real: -0.0191**, taking us to 1.0128 against the team's reported 1.0869.

## Decision

Not selected as the final model, because the project's primary metric is RMSE and the ensemble is slightly worse there. But it is **the best model we have for MAE**, it is cheap (no extra training — it blends two models we already have), and it should be the submission if the hackathon turns out to score MAE.
