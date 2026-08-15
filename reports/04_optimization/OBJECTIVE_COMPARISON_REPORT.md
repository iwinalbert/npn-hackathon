# Objective Comparison

*Phase 6 — four loss functions, identical features and window. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## What a loss function does

The objective is the model's definition of "wrong". Change it and you change what the model tries hardest to get right. Everything else here is held fixed, so the differences are the objective alone.

## Results (measured)

| Objective | RMSE | MAE | ΔRMSE | ΔMAE | Mean prediction on true-zero rows |
|---|---|---|---|---|---|
| Tweedie (power 1.1) | 2.1210 | 1.0319 | +0.0000 | +0.0000 | 0.5848 |
| L2 / squared error | 2.1351 | 1.0388 | +0.0141 | +0.0069 | 0.6006 |
| Poisson (count data) | 2.1379 | 1.0350 | +0.0169 | +0.0031 | 0.5915 |
| L1 / absolute error | 2.2432 | 0.9591 | +0.1221 | -0.0728 | 0.2509 |

## The headline finding

**Tweedie wins on RMSE. L1 wins overwhelmingly on MAE.**

L1 (absolute error) reaches MAE **0.9591** — 0.0728 better than our Tweedie model, and well below the team's reported 1.0869. It pays for that with +0.1221 RMSE.

The mechanism is visible in the last column: L1 chases the *median* rather than the mean, and with 54% of rows at zero the median is often zero. So L1 pushes predictions down (mean 0.25 on empty days versus 0.58 for Tweedie), which is exactly right for MAE and exactly wrong for RMSE.

> **The practical lesson: pick the objective that matches the metric you are judged on.** If this hackathon scores RMSE, use Tweedie. If it scores MAE, switch to L1 and gain far more than any feature engineering in this project delivered. We have both models trained and ready.

Gamma was excluded deliberately: it requires a strictly positive target, and 54% of our rows are exactly zero. Fitting it would mean dropping or shifting those rows, which changes the problem rather than the model.
