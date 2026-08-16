# Recursive Forecasting

*Phase 5 — predicting one day at a time and feeding predictions back. Generated 2026-08-14.*

> **Terms.** **RMSE** — average error with big misses punished far more heavily; lower is better. **MAE** — the plain average error. **Leakage** — letting the model see information that would not have existed when the forecast was really made. **Tweedie** — a loss function for data that is never negative and mostly zero. **Objective / loss function** — the definition of "wrong" that the model tries to minimise. **Inner window** — an earlier 28-day period we tune on, so the real scoring window stays untouched.

> Every number here comes from an experiment that actually ran. Failed experiments are reported alongside successful ones — in this campaign most of them failed, and that is the finding.

---

## The idea, in plain English

Our normal pipeline is **direct**: it freezes what it knows on the last real day and predicts all 28 days at once. That means it knows no more about day 28 than about day 2 — only the calendar changes.

**Recursive** instead predicts day 1, then pretends that prediction actually happened and uses it to predict day 2, and so on. This is allowed because only our own output is fed back; the real future sales are never touched. The risk is error accumulation — one bad early guess poisons everything after it.

## How we made leakage impossible

The working history is rebuilt from scratch: real sales up to the origin, zeros afterwards, then overwritten only by our own predictions. The real values for the forecast days are never copied in, so no code path can reach them. Verified, not assumed.

## Result

| | RMSE | MAE |
|---|---|---|
| Direct (current pipeline) | 2.1210 | 1.0319 |
| Recursive | **2.1182** | 1.0717 |
| Change | -0.0029 | +0.0398 |

This was the only configuration in the entire campaign to lower RMSE — but it cost a large amount of MAE.

![Horizon](charts/phase5_horizon.png)

## Error accumulation is visible

| Horizon day | Direct RMSE | Recursive RMSE | Recursive mean prediction |
|---|---|---|---|
| 1 | 1.8329 | 1.7999 | 1.2463 |
| 2 | 1.6749 | 1.6479 | 1.2195 |
| 3 | 1.7446 | 1.7408 | 1.2113 |
| 7 | 2.1812 | 2.2069 | 1.8372 |
| 14 | 2.1951 | 2.2008 | 1.5458 |
| 21 | 2.7977 | 2.7854 | 1.9324 |
| 28 | 2.4172 | 2.4300 | 1.8483 |

Recursion wins clearly in the first few days, where the fed-back values are still close to reality, and loses later as its own errors compound. The give-away is drift: the average prediction climbs from 1.25 on day 1 to 1.85 on day 28, while the actual average is 1.44. The model is progressively feeding itself optimism.

## Verdict

**Rejected.** The RMSE gain of 0.0029 is inside the ±0.013 window-to-window noise we measured, while the MAE loss of +0.0398 is roughly thirteen times larger than the gain. The brief's own guard-rail applies: do not optimise RMSE at the cost of severe MAE degradation.

Recursion remains the most interesting idea we tested, and it is worth presenting as an experiment with a clear diagnosis rather than as a component of the final model.
