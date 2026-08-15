# Hurdle Model — Second Attempt

*Phase 7 — trying to rescue the project's original novelty. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## The idea

Split the problem in two: **will this item sell at all today?** and **if it sells, how much?** Multiply the answers. With most rows at zero this is intuitively appealing, and it was the project's original proposed novelty.

The first attempt lost (2.1267 versus 2.1210) using Poisson for the magnitude stage. This attempt changes that stage to Tweedie and adds a calibration factor chosen on the inner window.

## Results

| Variant | RMSE | MAE | ΔRMSE | ΔMAE |
|---|---|---|---|---|
| Single model (reference) | 2.1210 | 1.0319 | — | — |
| Hurdle v1 (Poisson stage 2) | 2.1267 | 1.0324 | +0.0057 | +0.0005 |
| Hurdle v2 (Tweedie stage 2) | 2.1241 | 1.0300 | +0.0031 | -0.0020 |
| Hurdle v2 + calibration x0.88 | 2.1822 | 1.0195 | +0.0612 | -0.0124 |

Tweedie improved the hurdle over its Poisson version, but it still does not reach the single model. The calibration factor, which looked strongly helpful on the inner window, made the primary window substantially worse — the same non-transfer we saw in Phase 4.

## Why the hurdle keeps losing

**INTERPRETATION.** A Tweedie model is already a hurdle model. The Tweedie distribution has a point mass at zero and a continuous positive part — it is fitting "does it sell" and "how much" jointly, in one estimator. Splitting them by hand means fitting two models and multiplying, which compounds both of their errors instead of sharing information between them.

## Decision

**The hurdle is not part of the final model.** It stays in the project as a conceptual contribution and a documented negative result: we proposed it, tested it twice, improved it, and it still lost. That is a more defensible story than shipping it anyway.
