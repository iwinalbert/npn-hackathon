# Presentation Script — Feature Engineer

For presenting this role's work to hackathon judges. Roughly 3 minutes spoken.
**Bold** terms are the ones worth pausing on and defining out loud.

---

## Script

The frozen model isn't one forecaster — it's two, and they were deliberately
built to disagree. We call them **DIRECT** and **RECURSIVE**, and the
difference between them is entirely a feature-engineering decision, not a
tuning decision.

The DIRECT model has **38 features** and predicts all 28 days of the
forecast in a single pass, using only information that existed at the
**forecast origin** — the last day of known history. The RECURSIVE model
has **32 features** and instead predicts one day ahead, then feeds that
prediction back in as if it were real, and repeats that 28 times to walk
forward. Same underlying algorithm, same training data — genuinely
different feature sets and a genuinely different mechanism for reaching day
28. That's on purpose: two models that make the *same* kind of mistake add
nothing when you combine them. Two models that make *different* mistakes
do — and empirically, these two disagree just enough. Their errors
correlate at about 0.95, not 1.0, and that half-a-percent of disagreement
is exactly what the blend captures.

The feature that matters most in a project like this isn't any individual
signal — price, calendar events, SNAP benefit days, recency — it's
**leakage prevention**. A feature "leaks" when it accidentally contains
information from *after* the point the model is supposed to be predicting
from — for example, a "days since last sale" feature that was computed
using data from next week. That kind of bug doesn't crash anything and
doesn't look wrong in a spot check; it just makes your validation accuracy
lie to you, because the model quietly cheated during training and won't be
able to cheat in production. We didn't just review the feature code for
this — we **proved** it: every feature was recomputed after overwriting
every single post-origin sales value with a nonsense number, and the test
requires every feature to come back byte-for-byte identical to what it was
before. If even one feature changed, that would mean it was looking into
the future. None did.

Two things worth naming if a judge pushes on the model's honest
limitations: there's no promotion field anywhere in this dataset, so the
model was never given the chance to learn from promotions — it can't
answer "what if we ran a promotion," and we say so explicitly rather than
letting the model guess. And price is included as a feature the model
*conditions on*, not a lever it understands causally — the measured
response to price in this data is non-monotone, so we never claim the
model can answer "what happens if we change the price."

---

## Key terms glossary (for Q&A)

| Term | One-line definition |
|---|---|
| Forecast origin | The last day of known history; the cutoff point every feature must respect |
| Direct model | Predicts all 28 days at once, in a single pass |
| Recursive model | Predicts one day, feeds it back in, repeats 28 times |
| Leakage | A feature accidentally using information from after the point it's predicting from |
| Error correlation | How similarly two models' mistakes line up; lower correlation makes blending more valuable |

---

## Likely judge questions

- **"How do you actually prove there's no leakage, not just claim it?"** →
  overwrite everything after the origin with garbage, recompute every
  feature, require bit-for-bit identical output.
- **"Why not just pick whichever of the two models scores better?"** → the
  blend beats either one alone, because they're wrong in different places,
  not because one is better everywhere.
