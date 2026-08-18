# Tweedie Optimization

*Phase 4 — an eight-point search over the Tweedie variance power. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## What the Tweedie power controls

Tweedie sits between two familiar distributions. A power near 1 behaves like a Poisson (counting things); a power near 2 behaves like a Gamma (positive continuous amounts). In between it can put mass exactly at zero *and* have a long right tail — which is what daily unit sales look like. We had been using 1.1 without ever testing it.

## Search on the inner window

| Power | Inner RMSE | Inner MAE | High-volume RMSE | High-volume bias |
|---|---|---|---|---|
| 1.1 *(previous setting)* | 2.0899 | 1.0173 | 5.938 | +0.317 |
| 1.2 | 2.0845 | 1.0147 | 5.906 | +0.297 |
| 1.3 | 2.0793 | 1.0124 | 5.896 | +0.249 |
| 1.4 | 2.0812 | 1.0120 | 5.896 | +0.258 |
| 1.5 | 2.0766 | 1.0087 | 5.872 | +0.129 |
| 1.6 | 2.0825 | 1.0090 | 5.902 | +0.125 |
| 1.7 | 2.0923 | 1.0092 | 5.939 | +0.043 |
| 1.8 | 2.1040 | 1.0096 | 5.981 | -0.063 |

![Power curve](charts/phase4_power_curve.png)

A clean U-shape with a minimum at **power 1.5**. The high-volume bias falls steadily as the power rises — exactly the behaviour Phase 3 said we wanted.

## Then the honest part

The power was selected using only the inner window, so applying it once to the primary window is a clean test. It did not survive:

| | Inner window | Primary window |
|---|---|---|
| Power 1.1 (previous) | 2.0899 | 2.1210 |
| Power 1.5 (selected) | 2.0766 | 2.1263 |
| Change | -0.0133 | +0.0053 |

A gain of about 0.013 on one window became a loss of about 0.005 on the next. MAE did improve slightly (-0.0030).

> **Decision: keep power 1.1.** The improvement was not real. Had we selected on the scoring window instead of an inner one, we would have shipped noise and called it a result.
