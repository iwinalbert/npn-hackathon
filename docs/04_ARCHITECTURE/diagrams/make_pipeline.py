from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG        = "#FFFFFF"
INK       = "#111A22"
INK2      = "#3D4A56"
MUTED     = "#6B7884"
RULE      = "#C8D1D9"

ACCENT    = "#0E7A85"
ACCENT_BG = "#E7F3F4"
FROZEN    = "#9C6526"
FROZEN_BG = "#F8F0E5"
GHOST     = "#8B98A5"
GHOST_BG  = "#F4F6F8"
OK        = "#2C7A54"
WARN      = "#B0432F"

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

TITLE_DROP, LINE_1, LINE_GAP, PAD_BOT = 3.2, 3.9, 2.9, 2.7


def bh(n):
    return 6.2 if n == 0 else TITLE_DROP + LINE_1 + (n - 1) * LINE_GAP + PAD_BOT


fig, ax = plt.subplots(figsize=(16, 13.2), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 160)
ax.set_ylim(0, 132)
ax.axis("off")


def box(x, y, w, title, lines=(), *, edge=ACCENT, face=ACCENT_BG,
        dashed=False, tag=None, tsize=9.6, lsize=7.3):
    lines = list(lines)
    h = bh(len(lines))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=1.5, edgecolor=edge, facecolor=face,
        linestyle=(0, (4, 2.6)) if dashed else "solid", zorder=2))
    ty = y + h - TITLE_DROP
    ax.text(x + 2.4, ty, title, fontsize=tsize, family=MONO, weight="bold",
            color=INK, va="center", ha="left", zorder=3)
    if tag:
        ax.text(x + w - 2.4, ty, tag, fontsize=6.3, family=MONO, color=edge,
                va="center", ha="right", weight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.30", facecolor=BG,
                          edgecolor=edge, linewidth=0.9))
    ly = ty - LINE_1
    for ln in lines:
        ax.text(x + 2.4, ly, ln, fontsize=lsize, family=MONO, color=INK2,
                va="center", ha="left", zorder=3)
        ly -= LINE_GAP
    return h


def arrow(x1, y1, x2, y2, label=None, *, color=None, dashed=False,
          lx=None, ly=None, lsize=6.8):
    color = color or MUTED
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.5, color=color, zorder=4,
        linestyle=(0, (4, 2.4)) if dashed else "solid",
        shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2,
                ly if ly is not None else (y1 + y2) / 2 + 2.4,
                label, fontsize=lsize, family=MONO, color=color,
                va="center", ha="center", zorder=5,
                bbox=dict(boxstyle="square,pad=0.28", facecolor=BG,
                          edgecolor="none"))


def heading(y, text):
    ax.text(4, y, text, fontsize=10.2, family=MONO, weight="bold",
            color=INK, va="center")
    ax.plot([4, 156], [y - 2.6, y - 2.6], color=RULE, linewidth=1.0, zorder=1)


ax.text(4, 127.5, "Retail Demand Forecasting — Data & Delivery Pipeline",
        fontsize=21, family=SANS, weight="bold", color=INK, va="center")
ax.text(4, 122.9,
        "Where the numbers come from, how the image is built, and what each "
        "check actually proves",
        fontsize=9.6, family=SANS, color=MUTED, va="center")
ax.plot([4, 156], [119.5, 119.5], color=RULE, linewidth=1.1, zorder=1)

heading(114, "1  ·  WHERE THE NUMBERS COME FROM")

chain = [
    (4, "M5 raw data", ["5 competition CSVs", "immutable, never edited"],
     FROZEN, FROZEN_BG, "RO"),
    (35, "research/pipeline", ["feature build + training", "86 experiments"],
     GHOST, GHOST_BG, None),
    (66, "frozen champion", ["2 LightGBM boosters", "SHA-256 certified"],
     FROZEN, FROZEN_BG, "RO"),
    (97, "build_product_db.py", ["normalise · sort by series", "row-group · ~10 s"],
     ACCENT, ACCENT_BG, None),
    (128, "backend/data", ["130 MB · duckdb + 2 pq", "the ONLY runtime input"],
     ACCENT, ACCENT_BG, None),
]
CY = 96
for i, (x, t, ls, ec, fc, tg) in enumerate(chain):
    box(x, CY, 28, t, ls, edge=ec, face=fc, tag=tg, tsize=8.6)
    if i < len(chain) - 1:
        arrow(x + 28, CY + bh(2) / 2, x + 31, CY + bh(2) / 2)

ax.text(4, 92.0,
        "Everything left of build_product_db.py is research and never ships. "
        "Everything right of it is the deployable unit.",
        fontsize=7.8, family=SANS, color=MUTED, va="center")

box(4, 72, 90, "THE FROZEN MODEL",
    ["ŷ = 0.60 × Direct    LightGBM Tweedie(1.1, 38 features)",
     "  + 0.40 × Recursive LightGBM Tweedie(1.1, 32 features)",
     "validated on 853,720 held-out predictions, d_1914–d_1941"],
    edge=FROZEN, face=FROZEN_BG, tsize=9.4, lsize=7.6)

box(98, 72, 58, "HELD-OUT ACCURACY",
    ["RMSE  2.0929        WAPE  0.7205",
     "MAE   1.0395        bias −0.0224",
     "28.5% at store-item-day · 94.5% chain-wide"],
    edge=ACCENT, face=ACCENT_BG, tsize=9.4, lsize=7.6)

ax.text(4, 67.5,
        "Accuracy depends entirely on the level you aggregate to — 54% of "
        "store-item-days are zero, and independent errors cancel on "
        "aggregation. The delivered window d_1942–d_1969 has no ground truth, "
        "so no accuracy is ever quoted against it.",
        fontsize=7.8, family=SANS, color=MUTED, va="center")

heading(60, "2  ·  BUILD  —  one Dockerfile, two targets")

box(4, 34, 48, "target: api          (default)",
    ["fastapi · uvicorn · pydantic", "duckdb · google-genai",
     "no ML libraries at all", "/inference/* → 503 with a reason",
     "needs research/ : NO"],
    tsize=9.0)

box(56, 34, 48, "target: full         (opt-in)",
    ["+ lightgbm · numpy · pandas", "+ pyarrow · libgomp1  (~104 MB)",
     "live model verification works", "needs 2 read-only research mounts",
     "needs research/ : YES"],
    edge=FROZEN, face=FROZEN_BG, tsize=9.0)

box(108, 34, 48, "BUILD CONTEXT",
    ["repo root — the full target does", "COPY research/pipeline",
     "", ".dockerignore is an ALLOW-LIST:", "2,582 MB  →  0.38 MB  (56 files)"],
    edge=GHOST, face=GHOST_BG, dashed=True, tsize=9.0)

ax.text(4, 30.0,
        "A deny-list silently starts shipping whatever is added later; the "
        "allow-list fails the other way. preflight.py fails the build if any "
        "top-level directory has no rule — which is how a leak of 14 PDFs was "
        "caught.",
        fontsize=7.8, family=SANS, color=MUTED, va="center")

heading(25, "3  ·  CI  —  and what a green run does NOT prove")

jobs = [
    (4,   "preflight",   "config · context\nsecrets · hardening"),
    (35,  "backend",     "imports · schema\nnon-data tests"),
    (66,  "frontend",    "typecheck · 62 tests\nbuild · no key in dist/"),
    (97,  "images",      "build 3 · boot\nuid 10001 · no secret"),
    (128, "deploy-gate", "verdict + what was\nNOT proven"),
]
JY = 9
for i, (x, name, sub) in enumerate(jobs):
    box(x, JY, 28, name, sub.split("\n"),
        edge=ACCENT if i < 4 else INK,
        face=ACCENT_BG if i < 4 else GHOST_BG, tsize=9.0, lsize=7.0)
    if i < len(jobs) - 1:
        arrow(x + 28, JY + bh(2) / 2, x + 31, JY + bh(2) / 2)

ax.text(4, 5.0, "PROVEN IN CI", fontsize=7.6, family=MONO, weight="bold",
        color=OK, va="center")
ax.text(26, 5.0,
        "config validity · build-context hygiene · secret hygiene · frontend "
        "suite · images build and boot non-root · the API degrades correctly "
        "with no data",
        fontsize=7.4, family=SANS, color=INK2, va="center")

ax.text(4, 0.6, "ONLY ON A HOST", fontsize=7.6, family=MONO, weight="bold",
        color=WARN, va="center")
ax.text(26, 0.6,
        "the frozen-forecast canary 3331.3681 · the data-dependent backend "
        "tests · the API reaching ready:true  —  a CI runner has no data layer",
        fontsize=7.4, family=SANS, color=INK2, va="center")

out = pathlib.Path(__file__).resolve().parent.parent / "data-pipeline.png"
fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.34)
print(f"wrote {out}")
