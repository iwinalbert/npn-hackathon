# Feature Optimization

*Phase 2 — fourteen new candidate features, tested one group at a time. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## What we tested

Every group was added on top of the same 32-feature baseline, with the objective, hyperparameters, training origins and validation window held fixed. So any change is attributable to the group alone.

| Group | Features added |
|---|---|
| A. Short-term demand | `rolling_mean_14`, `rolling_std_14`, `rolling_zero_count_7`, `demand_momentum_7_28` |
| B. Calendar | `day_of_month` (payday effect), `week_of_year` |
| C. Price dynamics | `price_pct_change_1w`, `price_pct_change_4w`, `price_vs_origin_pct` |
| D. Interactions | `snap_food`, `snap x category`, `snap x store`, `weekend x category`, `event x category` |

## Results (measured)

| Configuration | Features | RMSE | MAE | ΔRMSE | ΔMAE |
|---|---|---|---|---|---|
| Current best 32 features (reproducibility check) | 32 | 2.1210 | 1.0319 | +0.0000 | +0.0000 |
| + A. Short-term demand dynamics (4 features) | 36 | 2.1233 | 1.0312 | +0.0022 | -0.0007 |
| + B. Calendar expansion (2 features) | 34 | 2.1327 | 1.0326 | +0.0116 | +0.0007 |
| + C. Price dynamics (3 features) | 35 | 2.1281 | 1.0307 | +0.0071 | -0.0012 |
| + D. Interactions (5 features) | 37 | 2.1256 | 1.0314 | +0.0045 | -0.0005 |
| + All v2 groups (14 features) | 46 | 2.1320 | 1.0313 | +0.0110 | -0.0007 |

**0 of 5 groups improved RMSE.**

Every group made RMSE slightly worse. Three of them (A, C, D) made MAE very slightly better, by around a thousandth — far too small to act on.

## What this means

This is the third independent time the project has reached the same conclusion. The original feature ablation found that everything beyond recent-demand features moved RMSE by hundredths; recency and listing features were measured as no help twice; and now fourteen fresh candidates, including three the other team's document specifically recommends, also fail to help.

**Interpretation:** the feature space is saturated. `rolling_mean_28` alone accounts for about 74% of the model's gain, and additional views of the same recent-demand signal are redundant. The remaining error is not missing-information error — it is genuine day-to-day randomness in retail demand.

## Decision

Keep the 32-feature set. No feature added in this phase is carried forward.
