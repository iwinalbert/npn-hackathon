# Presentation Script — Forecasting Evaluation

For presenting this role's work to hackathon judges. Roughly 3 minutes spoken.
**Bold** terms are the ones worth pausing on and defining out loud.

---

## Script

A forecast without honest evaluation is just a guess with extra
confidence, so this part of the project exists to make sure every number
we show anyone is actually true, and true in the way we say it's true.

The core number is a **backtest**: a validation window — d_1914 through
d_1941 in this dataset's day numbering — where we already know what
actually happened, so we can compare the model's prediction against real,
recorded outcomes. Across all 30,490 series, that's **853,720 individual
predictions**, scored at RMSE 2.0929 and MAE 1.0395. RMSE, root mean
squared error, penalizes big misses harder than small ones; MAE, mean
absolute error, is the plainer number — on average, the model's daily miss
is about one unit per product.

Here's the part that actually matters for trust, not just accuracy: we
report accuracy at **multiple aggregation levels**, on purpose, because a
single number here would be misleading. Individual product-level accuracy
is meaningfully noisier than store-level, which is noisier than
chain-wide. That's not a flaw we're hiding — it's arithmetic. Errors on
individual, noisy, near-zero-selling products partially cancel out when
you sum them across a whole store, and cancel out further across a whole
chain. Showing only the best-looking aggregate number would be technically
true and practically dishonest. So we show all of them, labeled by level,
every time.

One more distinction we're strict about: a **planning range** is not a
confidence interval. This model produces **point forecasts** — one number
per day — not a probability distribution. The range we show alongside it
is *observed historical error*, not a statistically derived interval, and
we say so explicitly rather than letting it imply more mathematical
rigor than it has.

The story I'd actually lead with, if I only had one: the original project
brief specified four occurrence-detection numbers — accuracy 0.8068,
precision 0.7088, recall 0.8076, F1 0.7082. When we recomputed those
directly from the verified backtest artifact, we found the brief had them
shifted — the number labeled "accuracy" was actually the recall value, and
the number labeled "precision" was actually the F1 score. We caught it,
recomputed the correct values, and reported the correction rather than
quietly shipping the brief's numbers unchecked. That's the whole point of
this role: not to make the numbers look good, but to make sure they're
*right*.

---

## Key terms glossary (for Q&A)

| Term | One-line definition |
|---|---|
| Backtest | Comparing model predictions against a window where the true outcome is already known |
| RMSE / MAE | Two ways of scoring average miss size — RMSE punishes big misses harder, MAE is the plain average |
| Aggregation level | How far you zoom out before reporting accuracy — item, store, or chain-wide |
| Point forecast | A single predicted number per day, not a probability distribution |
| Planning range | Observed historical error shown as a range — not a statistical confidence interval |

---

## Likely judge questions

- **"Why does accuracy look worse at the item level — is the model bad at
  that granularity?"** → no, it's inherent noise in low-volume individual
  series; the same model is ~97% accurate at chain level, because
  individual errors partially cancel in aggregate.
- **"Is the planning range a real confidence interval?"** → no, and we
  never call it one — it's the range of error actually observed
  historically, stated plainly as that.
