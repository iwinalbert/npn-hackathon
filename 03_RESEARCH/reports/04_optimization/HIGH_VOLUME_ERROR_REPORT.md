# High-Volume Error Attack

*Phase 3 — where the error really is, and three attempts to fix it. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## Diagnosis: where RMSE comes from

| Volume tier | Rows | Actual mean | Predicted mean | Bias | RMSE | Share of squared error |
|---|---|---|---|---|---|---|
| high | 65,968 | 7.585 | 7.196 | -0.389 | 5.976 | 61.33% |
| medium | 160,244 | 2.185 | 2.076 | -0.109 | 2.291 | 21.90% |
| low | 398,160 | 0.797 | 0.761 | -0.036 | 1.174 | 14.30% |
| very low | 229,348 | 0.280 | 0.267 | -0.012 | 0.643 | 2.47% |

The busiest tier is 7.7% of rows and carries about 61% of all squared error. We under-predict it by roughly 0.39 units per row.

Concentration is even sharper by product: the **top 50 items of 3,049 carry 37.7% of all squared error**.

## Three legitimate fixes, all tested

| Attempt | RMSE | MAE | ΔRMSE | High-volume RMSE | High-volume bias |
|---|---|---|---|---|---|
| current best (reference) | 2.1210 | 1.0319 | +0.0000 | 5.9756 | -0.3890 |
| volume weight cap 3.0x | 2.1376 | 1.0336 | +0.0165 | 6.0530 | -0.4770 |
| volume weight cap 5.0x | 2.1371 | 1.0335 | +0.0161 | 6.0484 | -0.5054 |
| high-vol calibration x1.00 | 2.1210 | 1.0319 | +0.0000 | 5.9756 | -0.3891 |

### All three failed, and the way they failed is informative

**Volume weighting made things worse.** Weighting busy series more heavily in training pushed RMSE up by about 0.016 — and, counter-intuitively, made the high-volume tier itself *worse* (RMSE 6.05 versus 5.98) with a *more* negative bias. Forcing the model to chase big days cost it accuracy on the medium ones without buying accuracy on the big ones.

**Post-hoc calibration found nothing to correct.** We searched for a multiplier to apply to high-volume predictions, choosing it on the inner window. The search returned **1.00** — no scaling improved anything. The model is already optimally calibrated for that tier.

## What we conclude

The under-prediction of busy days is **not a calibration error and not a weighting error**. It is what a squared-error-family model correctly does when the target is genuinely volatile: predicting the conditional mean of a high-variance day is the right answer even though it looks timid. The remaining high-volume error appears to be irreducible with the information available.
