"""
Markdown -> PDF renderer for pipeline stage reports.

Deliberately small: headings, paragraphs, pipe tables, bullet/numbered lists,
blockquote callouts, fenced code blocks, bold/inline-code spans, horizontal rules.
That is all our reports use, and a full markdown engine would be a dependency we
do not need.

Handles escaped pipes (\\|) inside table cells, which a naive splitter would
mistake for column separators.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable, Image, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

NAVY = colors.HexColor("#1a2a4a")
ACCENT = colors.HexColor("#2c5f8a")
LIGHTBG = colors.HexColor("#eef3f8")
CALLOUTBG = colors.HexColor("#fff8e6")
CALLOUTBORDER = colors.HexColor("#d9a441")
GREY = colors.HexColor("#555555")
ROWALT = colors.HexColor("#f5f7fa")
GRIDCOL = colors.HexColor("#c7d0da")

_PIPE = "\x00PIPE\x00"


def _styles():
    s = getSampleStyleSheet()
    add = s.add
    add(ParagraphStyle("DocTitle", fontSize=21, leading=26, textColor=NAVY,
                       fontName="Helvetica-Bold", spaceAfter=6, alignment=TA_LEFT))
    add(ParagraphStyle("DocSubtitle", fontSize=11.5, leading=15.5, textColor=GREY,
                       fontName="Helvetica-Oblique", spaceAfter=3))
    add(ParagraphStyle("H1", fontSize=15.5, leading=19.5, textColor=NAVY,
                       fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=7))
    add(ParagraphStyle("H2", fontSize=12.5, leading=16.5, textColor=ACCENT,
                       fontName="Helvetica-Bold", spaceBefore=11, spaceAfter=5))
    add(ParagraphStyle("H3", fontSize=11, leading=14.5, textColor=NAVY,
                       fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4))
    add(ParagraphStyle("Body2", fontSize=9.3, leading=13.2, spaceAfter=6,
                       fontName="Helvetica"))
    add(ParagraphStyle("Bullet2", fontSize=9.3, leading=13, spaceAfter=3,
                       fontName="Helvetica"))
    add(ParagraphStyle("Callout", fontSize=9.2, leading=13, spaceAfter=4,
                       fontName="Helvetica-Oblique",
                       textColor=colors.HexColor("#5a4200")))
    add(ParagraphStyle("Cell", fontSize=7.9, leading=10.3, fontName="Helvetica"))
    add(ParagraphStyle("CellHead", fontSize=8.1, leading=10.5,
                       fontName="Helvetica-Bold", textColor=colors.white))
    add(ParagraphStyle("Mono", fontSize=7.6, leading=10.0, fontName="Courier",
                       backColor=LIGHTBG))
    return s


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', text)
    return text


def _parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for ln in lines:
        ln = ln.strip().replace("\\|", _PIPE)
        if ln.startswith("|"):
            ln = ln[1:]
        if ln.endswith("|"):
            ln = ln[:-1]
        rows.append([c.strip().replace(_PIPE, "|") for c in ln.split("|")])
    return [r for r in rows if not all(re.match(r"^:?-+:?$", c) for c in r)]


def _table_flowable(rows: list[list[str]], s, avail_width: float) -> Table:
    n_col = max(len(r) for r in rows)
    data = [[Paragraph(_inline(h), s["CellHead"]) for h in rows[0]]]
    for r in rows[1:]:
        r = list(r) + [""] * (n_col - len(r))
        data.append([Paragraph(_inline(c), s["Cell"]) for c in r[:n_col]])

    t = Table(data, colWidths=[avail_width / n_col] * n_col, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, GRIDCOL),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ROWALT))
    t.setStyle(TableStyle(style))
    return t


def render_markdown_to_pdf(
    md_path: str | Path,
    pdf_path: str | Path,
    title: str,
    subtitles: list[str] | None = None,
    footer: str = "",
    page_break_on_h1: bool = True,
) -> Path:
    """Render a markdown file to a styled PDF. Returns the output path."""
    md_path, pdf_path = Path(md_path), Path(pdf_path)
    s = _styles()
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    avail = LETTER[0] - 1.1 * inch
    story = [Paragraph(title, s["DocTitle"])]
    for sub in (subtitles or []):
        story.append(Paragraph(_inline(sub), s["DocSubtitle"]))
    story += [Spacer(1, 6), HRFlowable(width="100%", thickness=1.2, color=ACCENT),
              Spacer(1, 13)]

    i, n, first_h1 = 0, len(lines), True
    while i < n:
        raw = lines[i]
        st = raw.strip()

        if st == "":
            i += 1
            continue

        if st == "---":
            story += [Spacer(1, 4),
                      HRFlowable(width="100%", thickness=0.6, color=GRIDCOL),
                      Spacer(1, 4)]
            i += 1
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", st)
        if m:
            lvl, txt = len(m.group(1)), _inline(m.group(2))
            if lvl == 1:
                if first_h1:
                    first_h1 = False          # title already rendered above
                    i += 1
                    continue
                if page_break_on_h1:
                    story.append(PageBreak())
                story += [Paragraph(txt, s["H1"]),
                          HRFlowable(width="100%", thickness=0.8, color=ACCENT),
                          Spacer(1, 6)]
            else:
                story.append(Paragraph(txt, s["H2" if lvl == 2 else "H3"]))
            i += 1
            continue

        # Image:  ![alt](relative/or/absolute/path.png)
        mimg = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", st)
        if mimg:
            alt, src = mimg.group(1), mimg.group(2)
            p = Path(src)
            if not p.is_absolute():
                p = (md_path.parent / src).resolve()
            if p.exists():
                iw, ih = ImageReader(str(p)).getSize()
                w = min(avail, iw)
                story += [Spacer(1, 6),
                          Image(str(p), width=w, height=ih * (w / iw)),
                          Spacer(1, 8)]
            else:
                story.append(Paragraph(f"<i>[missing image: {alt} — {src}]</i>",
                                       s["Body2"]))
            i += 1
            continue

        if st.startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = _parse_table(block)
            if rows:
                story += [Spacer(1, 4), _table_flowable(rows, s, avail), Spacer(1, 8)]
            continue

        if st.startswith(">"):
            block = []
            while i < n and lines[i].strip().startswith(">"):
                block.append(lines[i].strip()[1:].strip())
                i += 1
            p = Paragraph(_inline(" ".join(block)), s["Callout"])
            t = Table([[p]], colWidths=[avail])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CALLOUTBG),
                ("BOX", (0, 0), (-1, -1), 0.8, CALLOUTBORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story += [Spacer(1, 4), t, Spacer(1, 6)]
            continue

        if st.startswith("```"):
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            body = ("\n".join(code)
                    .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace("\n", "<br/>").replace(" ", "&nbsp;"))
            story += [Paragraph(body, s["Mono"]), Spacer(1, 6)]
            continue

        if re.match(r"^(-|\d+\.)\s+", st):
            items = []
            while i < n:
                mm = re.match(r"^(-|\d+\.)\s+(.*)$", lines[i].strip())
                if not mm:
                    break
                items.append(Paragraph(_inline(mm.group(2)), s["Bullet2"]))
                i += 1
            story += [ListFlowable([ListItem(x, leftIndent=6) for x in items],
                                   bulletType="bullet", start="•",
                                   leftIndent=14, bulletFontSize=7),
                      Spacer(1, 4)]
            continue

        para = [st]
        i += 1
        while i < n:
            nx = lines[i].strip()
            if (nx == "" or nx == "---" or nx.startswith(("|", ">", "```"))
                    or re.match(r"^#{1,3}\s+", nx) or re.match(r"^(-|\d+\.)\s+", nx)):
                break
            para.append(nx)
            i += 1
        story.append(Paragraph(_inline(" ".join(para)), s["Body2"]))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=LETTER,
        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title=title, author="NPN Hackathon — M5 Forecasting Pipeline",
    )

    def _page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(0.55 * inch, 0.35 * inch, footer)
        canvas.drawRightString(LETTER[0] - 0.55 * inch, 0.35 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return pdf_path
