# Optimization Baseline

*Phase 1 — re-running the current best model unchanged, to prove it reproduces before anything is optimised. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## Why this step exists

Before changing anything we re-ran the existing best configuration from scratch. If it does not reproduce its own score exactly, then every later comparison is measuring randomness rather than our changes.

## Result

| | Reference | Re-run | Drift |
|---|---|---|---|
| RMSE | 2.121043 | 2.121043 | +0.00e+00 |
| MAE | 1.031927 | 1.031927 | +0.00e+00 |

**Reproduced exactly.** Same seed, same data, same answer to every decimal place. The pipeline is deterministic.

## Baseline behaviour we will try to improve

| Diagnostic | Value | What it means |
|---|---|---|
| High-volume RMSE | 5.9756 | Error on the busiest 7.7% of rows |
| High-volume bias | -0.3891 | Negative = we under-predict busy days |
| Share of squared error from that tier | 61.33% | Where RMSE actually comes from |
| Mean prediction on true-zero rows | 0.58485 | We place a small positive value on days that turn out empty |
| Mean prediction where sales happened | 2.31342 | Against an actual mean of about 3.17 |
| Prediction spread (p50 / p99 / max) | 0.61136 / 12.44237 / 151.1688 | The forecast is heavily concentrated near zero |

The picture is consistent: the model is cautious. It under-predicts busy days and puts a little weight on quiet ones. Squared error punishes the first far more than the second, which is why over 60% of RMSE sits in a small minority of rows.

## Leakage check

The extended feature builder used in later phases was put through the same corruption test: every sales value after the forecast origin was overwritten with 9999 and all features rebuilt. All 46 came back identical. The guarantee is re-earned, not inherited.

## Next

Phase 2 adds candidate features one group at a time.
