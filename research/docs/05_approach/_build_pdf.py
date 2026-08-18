"""
Renders FINAL_PROJECT_APPROACH.md into FINAL_PROJECT_APPROACH.pdf.
Read-only w.r.t. every existing project file; only touches FINAL_APPROACH/.
A lightweight markdown -> reportlab renderer: headings, paragraphs, pipe-tables,
bullet lists, blockquotes (used here for callouts), bold spans, hr.
"""
import re
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "FINAL_PROJECT_APPROACH.md")
PDF_PATH = os.path.join(HERE, "FINAL_PROJECT_APPROACH.pdf")

NAVY = colors.HexColor("#1a2a4a")
ACCENT = colors.HexColor("#2c5f8a")
LIGHTBG = colors.HexColor("#eef3f8")
CALLOUTBG = colors.HexColor("#fff8e6")
CALLOUTBORDER = colors.HexColor("#d9a441")
GREY = colors.HexColor("#555555")
ROWALT = colors.HexColor("#f5f7fa")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="DocTitle", fontSize=22, leading=27, textColor=NAVY,
                           fontName="Helvetica-Bold", spaceAfter=6, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="DocSubtitle", fontSize=12, leading=16, textColor=GREY,
                           fontName="Helvetica-Oblique", spaceAfter=4))
styles.add(ParagraphStyle(name="H1", fontSize=16, leading=20, textColor=NAVY,
                           fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8,
                           borderWidth=0, borderColor=ACCENT))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=17, textColor=ACCENT,
                           fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="H3", fontSize=11.5, leading=15, textColor=NAVY,
                           fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyText2", fontSize=9.3, leading=13.2, spaceAfter=6,
                           fontName="Helvetica", alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Bullet2", fontSize=9.3, leading=13, spaceAfter=3,
                           fontName="Helvetica"))
styles.add(ParagraphStyle(name="Callout", fontSize=9.3, leading=13, spaceAfter=4,
                           fontName="Helvetica-Oblique", textColor=colors.HexColor("#5a4200")))
styles.add(ParagraphStyle(name="Cell", fontSize=8.2, leading=10.8, fontName="Helvetica"))
styles.add(ParagraphStyle(name="CellHead", fontSize=8.4, leading=10.8, fontName="Helvetica-Bold",
                           textColor=colors.white))
styles.add(ParagraphStyle(name="Mono", fontSize=8.0, leading=10.5, fontName="Courier",
                           backColor=LIGHTBG))
styles.add(ParagraphStyle(name="TOCItem", fontSize=9.6, leading=15, fontName="Helvetica"))


def inline_md(text):
    """Convert **bold**, `code`, and escape for reportlab mini-HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', text)
    return text


_PIPE_PLACEHOLDER = "\x00PIPE\x00"


def parse_table(lines):
    rows = []
    for ln in lines:
        ln = ln.strip().replace("\\|", _PIPE_PLACEHOLDER)
        if ln.startswith("|"):
            ln = ln[1:]
        if ln.endswith("|"):
            ln = ln[:-1]
        cells = [c.strip().replace(_PIPE_PLACEHOLDER, "|") for c in ln.split("|")]
        rows.append(cells)
    # drop separator row (---|---)
    rows = [r for r in rows if not all(re.match(r"^:?-+:?$", c) for c in r)]
    return rows


def make_table_flowable(rows, col_count):
    data = []
    header = rows[0]
    data.append([Paragraph(inline_md(h), styles["CellHead"]) for h in header])
    for r in rows[1:]:
        r = r + [""] * (col_count - len(r))
        data.append([Paragraph(inline_md(c), styles["Cell"]) for c in r[:col_count]])

    avail_width = 7.1 * inch
    col_width = avail_width / col_count
    t = Table(data, colWidths=[col_width] * col_count, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d0da")),
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


def build_story(md_text):
    lines = md_text.split("\n")
    story = []
    i = 0
    n = len(lines)

    # Title block from first H1
    story.append(Paragraph("Final Project Approach", styles["DocTitle"]))
    story.append(Paragraph("M5 Retail Demand Forecasting &mdash; Problem Statement 11", styles["DocSubtitle"]))
    story.append(Paragraph("Senior Review, Discrepancy Resolution &amp; Team-Ready ML Strategy", styles["DocSubtitle"]))
    story.append(Paragraph("NPN AIA Hackathon &mdash; St. Joseph's College of Engineering", styles["DocSubtitle"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT))
    story.append(Spacer(1, 14))

    skip_first_h1 = True

    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if stripped == "":
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#c7d0da")))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = inline_md(m.group(2))
            if level == 1:
                if skip_first_h1:
                    skip_first_h1 = False
                    i += 1
                    continue
                story.append(PageBreak())
                story.append(Paragraph(text, styles["H1"]))
                story.append(HRFlowable(width="100%", thickness=0.8, color=ACCENT))
                story.append(Spacer(1, 6))
            elif level == 2:
                story.append(Paragraph(text, styles["H2"]))
            else:
                story.append(Paragraph(text, styles["H3"]))
            i += 1
            continue

        # italic-only line used as document meta (e.g. *Problem Statement...*)
        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            story.append(Paragraph(f"<i>{inline_md(stripped[1:-1])}</i>", styles["DocSubtitle"]))
            i += 1
            continue

        # Table block
        if stripped.startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = parse_table(block)
            if rows:
                col_count = max(len(r) for r in rows)
                tbl = make_table_flowable(rows, col_count)
                story.append(Spacer(1, 4))
                story.append(tbl)
                story.append(Spacer(1, 8))
            continue

        # Blockquote / callout (>)
        if stripped.startswith(">"):
            block = []
            while i < n and lines[i].strip().startswith(">"):
                block.append(lines[i].strip()[1:].strip())
                i += 1
            text = " ".join(block)
            p = Paragraph(inline_md(text), styles["Callout"])
            tbl = Table([[p]], colWidths=[7.1 * inch])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CALLOUTBG),
                ("BOX", (0, 0), (-1, -1), 0.8, CALLOUTBORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(Spacer(1, 4))
            story.append(tbl)
            story.append(Spacer(1, 6))
            continue

        # Code block fenced
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code_text = "\n".join(code_lines)
            code_text = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_text = code_text.replace("\n", "<br/>").replace(" ", "&nbsp;")
            story.append(Paragraph(code_text, styles["Mono"]))
            story.append(Spacer(1, 6))
            continue

        # Bullet list
        if re.match(r"^(-|\d+\.)\s+", stripped):
            items = []
            while i < n:
                s2 = lines[i].strip()
                mm = re.match(r"^(-|\d+\.)\s+(.*)$", s2)
                if not mm:
                    break
                items.append(Paragraph(inline_md(mm.group(2)), styles["Bullet2"]))
                i += 1
            story.append(ListFlowable(
                [ListItem(it, leftIndent=6) for it in items],
                bulletType="bullet", start="•", leftIndent=14, bulletFontSize=7
            ))
            story.append(Spacer(1, 4))
            continue

        # Regular paragraph (collect contiguous non-blank lines)
        para_lines = [stripped]
        i += 1
        while i < n and lines[i].strip() != "" and not re.match(r"^(#{1,3})\s+", lines[i].strip()) \
                and not lines[i].strip().startswith("|") and not lines[i].strip().startswith(">") \
                and not lines[i].strip() == "---" and not re.match(r"^(-|\d+\.)\s+", lines[i].strip()) \
                and not lines[i].strip().startswith("```"):
            para_lines.append(lines[i].strip())
            i += 1
        text = " ".join(para_lines)
        story.append(Paragraph(inline_md(text), styles["BodyText2"]))

    return story


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=LETTER,
        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Final Project Approach — M5 Retail Demand Forecasting",
        author="NPN Hackathon Team — Senior ML Review",
    )

    story = build_story(md_text)

    def add_page_number(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawString(0.55 * inch, 0.35 * inch, "FINAL_APPROACH / FINAL_PROJECT_APPROACH.pdf — Review stage, no model trained")
        canvas.drawRightString(LETTER[0] - 0.55 * inch, 0.35 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
