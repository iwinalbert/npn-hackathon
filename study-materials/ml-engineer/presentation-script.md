# Presentation Script — ML Engineer

For presenting this role's work to hackathon judges. Roughly 3 minutes spoken.
**Bold** terms are the ones worth pausing on and defining out loud.

---

## Script

Behind the model you're seeing today are **86 tracked experiments** — every
one logged with its full configuration, its leakage checks, and a written
decision for why it was accepted or rejected, across five research stages:
foundation and baselines, ablation and tuning, benchmark investigation,
optimization, and diagnostics. This wasn't 86 random attempts — it's an
honest, inspectable record of how we actually got here, including the
things that didn't work.

We started with **baselines**, because a model is only impressive relative
to something. The naive "just predict last value" baseline scores an RMSE
of 2.89. A slightly smarter 28-day rolling average gets to 2.24. The first
LightGBM model — no tuning, just the algorithm applied honestly — already
beats both at 2.15. That's the real starting line: not zero, but a
meaningfully hard baseline to clear.

The model we shipped is called **Tweedie**, which is a probability
distribution built for exactly this kind of data: mostly zeros, with
occasional positive counts. Most retail products don't sell every single
day — a lot of days are just zero — and a model trained assuming a normal
distribution handles that badly. Tweedie doesn't.

The final, **frozen champion** is a blend: 0.60 times a direct model, plus
0.40 times a recursive model, achieving an RMSE of **2.0929** on 853,720
held-out predictions. "Frozen" here is a specific engineering commitment,
not a description — it means a regression test hashes the two model
binaries and fails the build the instant either one changes, whether from
a bug, an accidental retrain, or someone trying to sneak in an
"improvement" without re-validating everything downstream. Every accuracy
figure this product ever shows, anywhere, traces back to this exact,
unchanging pair of files. We also fixed the random seed — **seed 42** — and
turned on deterministic training, so the training process itself is
100% reproducible, not just approximately similar run to run.

If a judge asks why we blend two models instead of picking the best one:
because they're not two attempts at the same thing, they're two
*architecturally different* forecasters — one direct, one recursive — and
their errors correlate at about 0.95, not 1.0. That gap is where the blend
earns its keep.

---

## Key terms glossary (for Q&A)

| Term | One-line definition |
|---|---|
| Experiment registry | A logged record of every model run — config, checks, and decision — not just the winners |
| Baseline | A deliberately simple comparison point that any real model must beat to justify its existence |
| Tweedie distribution | A statistical distribution suited to data that's mostly zeros with occasional positive spikes |
| Frozen model | A model whose binaries are hash-checked so any change fails the build automatically |
| Deterministic training | Training that produces the exact same result every time it's re-run, via a fixed seed |

---

## Likely judge questions

- **"How do we know 2.0929 isn't cherry-picked from a lucky run?"** →
  seed 42, deterministic training, and 853,720 held-out predictions, not
  a small sample.
- **"What happens if someone wants to improve the model later?"** → they
  can, but the hash check means it can never happen *silently* — any change
  requires deliberately re-validating everything that depends on it.
