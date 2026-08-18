# Architecture diagrams

The two PNGs in the parent folder are **generated**, not hand-drawn. Edit the
script, re-run it, and commit both.

```bash
pip install matplotlib          # the only dependency
python docs/04_ARCHITECTURE/diagrams/make_arch.py
python docs/04_ARCHITECTURE/diagrams/make_pipeline.py
```

| Script | Output | Answers |
|---|---|---|
| `make_arch.py` | `../system-architecture.png` | what runs where — containers, network, mounts, egress |
| `make_pipeline.py` | `../data-pipeline.png` | where the numbers come from, how the image is built, what each check proves |

Both render at **200 dpi** (~2,600 px wide) — sharp enough to project, to print
on A3, or to crop a single box out of for a slide.

## Why matplotlib rather than Graphviz or Mermaid

Neither was available and neither was wanted:

- **Graphviz is not installed** on this machine, and its layout engine puts
  boxes where the algorithm prefers. An architecture drawing is *read*, so the
  reader's path through it is the thing being designed — that has to be placed
  by hand.
- **Mermaid** fixes its theme at render time, so a diagram exported from it
  cannot follow a light/dark context, and its box sizing is not controllable to
  the degree these annotated enclosures need.

matplotlib gives absolute coordinate control and crisp text at any DPI, with a
dependency the project already has.

## The one rule to keep

**Box heights are computed from content by `bh()` — never passed in.**

The first draft passed heights by hand and the last line of six different boxes
rendered directly on top of its own bottom border. If you add a body line to a
box, the box grows on its own; what you must then check is that it has not
collided with whatever sits *below* it, since the vertical budget in each
section is hand-allocated.

Colour carries meaning and should not be reassigned:

| | |
|---|---|
| teal | the deployed unit |
| ochre | frozen artefact, read-only |
| grey dashed | outside the default deployment |
