# Presentation Script — Data Engineer

For presenting this role's work to hackathon judges. Roughly 3 minutes spoken.
**Bold** terms are the ones worth pausing on and defining out loud — judges
notice when a term is used correctly and explained, not just dropped.

---

## Script

Every forecast this system produces traces back to five CSV files from the
Walmart **M5 competition** — calendar data, two versions of the sales
history, sell prices, and a sample submission format. Before anything else
happened, we treated those five files as **immutable**: never opened in
write mode, and every one of them has a recorded **MD5 checksum** — a short
fingerprint of a file's exact contents — so if a byte ever changed, we'd
know instantly, without having to eyeball 60 megabytes of numbers.

From those raw files, we built what we call the **long panel**: a single,
flat table with 59.2 million rows — one row per store, per item, per day.
That's the format every downstream model actually needs, because the raw
files ship demand as a wide table, one column per day, which is convenient
for storage but useless for training a model that has to reason about time.

The scale here is genuinely large: **30,490 series**. A "series" is one
specific product in one specific store — so this isn't one forecast, it's
30,490 of them, running through the same pipeline simultaneously. And
they're not independent — they sit inside a real retail **hierarchy**:
state, then store, then category, then department, then item. That
hierarchy matters later, because it's what lets us report accuracy at
different zoom levels instead of just one number that hides more than it
shows.

The part I'm proudest of isn't the pipeline itself — it's the **integrity
manifest**. Every one of 520 protected files across this project — raw
data, trained models, predictions, reports — has a SHA-256 hash on record.
Before any release, we re-hash all 520 and diff against the recorded
values. If even one file was silently regenerated, corrupted, or swapped,
that diff catches it immediately. That's not a nice-to-have — for a project
whose entire pitch is "the model is frozen and reproducible," being able to
*prove* nothing quietly changed underneath it is the whole point.

One design decision worth mentioning if asked: we deliberately did **not**
move or duplicate the raw data closer to the code that uses it, even though
that would have looked tidier. The pipeline's own configuration derives
every single path from one root, and every one of 58 pipeline scripts and
86 tracked experiments depends on that root staying exactly where it was
when those experiments ran. Reproducibility outranked folder aesthetics —
and if a judge asks "why didn't you clean that up," that's the honest
answer: because cleaning it up would have broken the ability to reproduce
every experiment that came before it.

---

## Key terms glossary (for Q&A)

| Term | One-line definition |
|---|---|
| M5 competition | The Walmart hierarchical retail forecasting benchmark this whole project is built on |
| MD5 / SHA-256 checksum | A short fingerprint of a file's exact contents; any change to the file changes the fingerprint |
| Long panel | Data reshaped to one row per (store, item, day) — the format models actually need |
| Series | One specific product in one specific store, tracked independently |
| Hierarchy | The nested grouping (state → store → category → department → item) that lets accuracy be reported at different levels |
| Integrity manifest | The recorded hash of every protected file, used to prove nothing changed silently |

---

## Likely judge questions

- **"How do you know the data hasn't drifted or been corrupted?"** → the
  520-file SHA-256 manifest, re-checked before every release.
- **"Why 30,490 series instead of one aggregate model?"** → because the
  business question is "how much of *this* product should *this* store
  stock," and that only exists at the bottom of the hierarchy — aggregates
  are derived from it, not the other way round.
