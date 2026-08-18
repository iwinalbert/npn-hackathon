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

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

TITLE_DROP = 3.2
LINE_1     = 3.9
LINE_GAP   = 2.9
PAD_BOT    = 2.7


def bh(n_lines: int) -> float:
    if n_lines == 0:
        return 6.2
    return TITLE_DROP + LINE_1 + (n_lines - 1) * LINE_GAP + PAD_BOT


fig, ax = plt.subplots(figsize=(16, 12.6), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 160)
ax.set_ylim(0, 126)
ax.axis("off")


def box(x, y, w, title, lines=(), *, edge=ACCENT, face=ACCENT_BG,
        dashed=False, tag=None, tsize=10, lsize=7.4):
    lines = list(lines)
    h = bh(len(lines))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=1.5, edgecolor=edge, facecolor=face,
        linestyle=(0, (4, 2.6)) if dashed else "solid", zorder=2))

    ty = y + h - TITLE_DROP
    ax.text(x + 2.4, ty, title, fontsize=tsize, family=MONO,
            weight="bold", color=INK, va="center", ha="left", zorder=3)

    if tag:
        ax.text(x + w - 2.4, ty, tag, fontsize=6.3, family=MONO,
                color=edge, va="center", ha="right", weight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.30", facecolor=BG,
                          edgecolor=edge, linewidth=0.9))

    ly = ty - LINE_1
    for ln in lines:
        ax.text(x + 2.4, ly, ln, fontsize=lsize, family=MONO,
                color=INK2, va="center", ha="left", zorder=3)
        ly -= LINE_GAP
    return h


def zone(x, y, w, h, label, *, edge=RULE, lc=MUTED):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=1.4, edgecolor=edge, facecolor="none",
        linestyle=(0, (6, 3.5)), zorder=1))
    ax.text(x + 4.5, y + h, f"  {label}  ", fontsize=7.2, family=MONO,
            color=lc, va="center", ha="left", weight="bold", zorder=3,
            bbox=dict(boxstyle="square,pad=0.34", facecolor=BG,
                      edgecolor="none"))


def arrow(x1, y1, x2, y2, label=None, *, color=None, dashed=False,
          lx=None, ly=None, lsize=7.0, ha="center"):
    color = color or MUTED
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.5, color=color, zorder=4,
        linestyle=(0, (4, 2.4)) if dashed else "solid",
        shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2,
                ly if ly is not None else (y1 + y2) / 2,
                label, fontsize=lsize, family=MONO, color=color,
                va="center", ha=ha, zorder=5,
                bbox=dict(boxstyle="square,pad=0.30", facecolor=BG,
                          edgecolor="none"))


ax.text(4, 121.5, "Retail Demand Forecasting — System Architecture",
        fontsize=21, family=SANS, weight="bold", color=INK, va="center")
ax.text(4, 116.9,
        "Walmart M5  ·  30,490 store-item series  ·  28-day horizon  "
        "·  frozen model, served from a precomputed data layer",
        fontsize=9.6, family=SANS, color=MUTED, va="center")
ax.plot([4, 156], [113.5, 113.5], color=RULE, linewidth=1.1, zorder=1)

box(26, 99, 60, "BROWSER",
    ["React SPA · calls the relative path /api/v1/…",
     "no API host is compiled into the bundle"],
    edge=GHOST, face=GHOST_BG, dashed=True, tsize=10.5)

arrow(56, 99, 56, 94.6, "HTTP :8080", lx=58.8, ly=96.8, ha="left")

zone(4, 30, 108, 63, "DOCKER HOST")
zone(8, 59, 100, 30, "INTERNAL NETWORK  ·  api:8000 is not published",
     edge=ACCENT, lc=ACCENT)

box(12, 64, 42, "frontend  ·  nginx",
    ["serves the built static bundle",
     "try_files → /index.html  (SPA deep links)",
     "proxy_pass → ${API_HOST}/api/",
     "proxy_read_timeout 180s"],
    tag=":8080 PUB")

box(64, 64, 44, "api  ·  uvicorn / FastAPI",
    ["python:3.13-slim · uid 10001, non-root",
     "34 routes · single worker by design",
     "/health → the process is alive",
     "/ready  → the data layer is queryable"],
    tag=":8000 INT")

arrow(54, 73, 64, 73, "/api/", ly=75.9)

arrow(86, 64, 86, 55.1, "reads", lx=88.4, ly=61.6, ha="left")
box(64, 40, 44, "/data/product",
    ["product.duckdb   20 MB  relational tables",
     "history.parquet  32 MB  59.2M rows of actuals",
     "backtest.parquet 78 MB  6.8M backtest rows"],
    edge=FROZEN, face=FROZEN_BG, tag="RO")

box(12, 37.1, 48, "NO RESEARCH TREE REQUIRED",
    ["The default stack serves every shipped route —",
     "including the frozen 28-day forecast — from the",
     "three files at right. Verified against an empty",
     "project root: research/ can be absent entirely."],
    edge=GHOST, face=GHOST_BG, dashed=True, tsize=8.8, lsize=7.2)

box(120, 64, 36, "Google Gemini API",
    ["external · HTTPS egress,",
     "from the api container only",
     "unset key → the assistant",
     "reports why; all else works"],
    edge=GHOST, face=GHOST_BG, dashed=True, tsize=9.6, lsize=7.2)
arrow(108, 73, 120, 73, "optional", ly=75.9, dashed=True)

zone(4, 4, 108, 24,
     "OPT-IN OVERLAY  ·  --inference  ·  the ONLY dependency on research/",
     edge=FROZEN, lc=FROZEN)

box(10, 9, 32, "models/champion",
    ["the two frozen", "LightGBM boosters"],
    edge=FROZEN, face=FROZEN_BG, tag="RO", tsize=8.8)
box(46, 9, 34, "predictions/final_forecast",
    ["the shipped forecast,", "compared against"],
    edge=FROZEN, face=FROZEN_BG, tag="RO", tsize=8.2)
box(84, 9, 24, "scratch volumes",
    ["absorb import-time", "mkdir() calls only"],
    edge=GHOST, face=GHOST_BG, dashed=True, tag="RW", tsize=8.8)

arrow(86, 40, 86, 28.5, dashed=True)

ax.text(120, 108, "READING THE DIAGRAM", fontsize=8.4, family=MONO,
        weight="bold", color=INK, va="center")
ax.plot([120, 156], [105.8, 105.8], color=RULE, linewidth=1.0)

for i, (ec, fc, dsh, txt) in enumerate([
        (ACCENT, ACCENT_BG, False, "the deployed unit"),
        (FROZEN, FROZEN_BG, False, "frozen artefact, read-only"),
        (GHOST,  GHOST_BG,  True,  "outside the default deploy")]):
    ly = 101.5 - i * 5.6
    ax.add_patch(FancyBboxPatch(
        (120, ly - 1.6), 5.4, 3.2,
        boxstyle="round,pad=0,rounding_size=0.5",
        linewidth=1.5, edgecolor=ec, facecolor=fc,
        linestyle=(0, (3, 2)) if dsh else "solid", zorder=2))
    ax.text(127.8, ly, txt, fontsize=7.8, family=SANS, color=INK2,
            va="center", ha="left")

ax.text(120, 56, "INSIDE THE api CONTAINER", fontsize=8.4, family=MONO,
        weight="bold", color=INK, va="center")
ax.plot([120, 156], [53.8, 53.8], color=RULE, linewidth=1.0)

layers = [
    ("routers/   8 · HTTP + validation", ACCENT, ACCENT_BG),
    ("services/  9 · domain logic",      ACCENT, ACCENT_BG),
    ("cache.py + db.py  ·  read-only",   ACCENT, ACCENT_BG),
    ("/data/product  ·  duckdb + 2 pq",  FROZEN, FROZEN_BG),
]
top = 49.0
for i, (name, ec, fc) in enumerate(layers):
    y = top - bh(0)
    box(120, y, 36, name, edge=ec, face=fc, tsize=8.2)
    if i < len(layers) - 1:
        arrow(138, y, 138, y - 3.0)
    top = y - 3.0

ax.text(120, 12.0,
        "Routers never touch the database.\n"
        "Only db.py speaks to DuckDB, so the\n"
        "read-only guarantee is enforced in\n"
        "one place rather than trusted in ten.",
        fontsize=7.4, family=SANS, color=MUTED, va="top", ha="left",
        linespacing=1.7)

ax.plot([4, 156], [1.4, 1.4], color=RULE, linewidth=1.0)
ax.text(4, -1.6,
        "Browser → frontend only.  nginx proxies /api/ across the internal "
        "network, so there is one origin: no CORS, no API host in the bundle, "
        "and no path for the Gemini key to reach the browser.",
        fontsize=7.8, family=SANS, color=MUTED, va="center", ha="left")

out = pathlib.Path(__file__).resolve().parent.parent / "system-architecture.png"
fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.34)
print(f"wrote {out}")
