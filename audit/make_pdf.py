"""Render docs/AUDIT.md as a PDF (docs/AUDIT.pdf).

Needs the `reportlab` Python package, which is not part of the pixi
environment:  python3 -m pip install reportlab
Usage: python3 audit/make_pdf.py [input.md] [output.pdf]
"""
import re
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

src = sys.argv[1] if len(sys.argv) > 1 else "docs/AUDIT.md"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/AUDIT.pdf"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5, leading=13)
cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8, leading=10)
head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, leading=22, alignment=0, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5, keepWithNext=True)

GREEN = colors.HexColor("#1b7f3b")
RED = colors.HexColor("#b3261e")
GREY = colors.HexColor("#5f6368")
HEAD_BG = colors.HexColor("#2f3b52")
STRIPE = colors.HexColor("#f3f5f8")
RULE = colors.HexColor("#c9ced6")


def inline(text):
    """Markdown bold and code spans to reportlab markup."""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


# column widths (points) by header row; anything else is spread evenly
WIDTHS = {
    ("Box result", "Boxes"): [150, 60],
    ("Number result", "Numbers"): [150, 60],
    ("Box", "Ch.", "Source", "Example", "Numbers compared", "Matching", "Result"): [32, 30, 48, 330, 90, 62, 100],
    ("Box", "Ch.", "Quantity", "Book", "Port", "Difference", "Result"): [32, 30, 300, 80, 90, 90, 90],
    ("Box", "Ch.", "Quantity", "Book", "Port", "Difference / note", "Result"): [32, 30, 285, 88, 88, 150, 90],
}


def result_markup(text):
    if text.startswith("match"):
        return f'<font color="#1b7f3b"><b>{escape(text)}</b></font>'
    if text.lower().startswith("differs") or text == "last digit":
        return f'<font color="#b3261e"><b>{escape(text)}</b></font>'
    if text == "not comparable":
        return f'<font color="#5f6368"><b>{escape(text)}</b></font>'
    return inline(text)


def make_table(rows, avail):
    header = tuple(rows[0])
    widths = WIDTHS.get(header) or [avail / len(header)] * len(header)
    data = [[Paragraph(inline(c), head) for c in rows[0]]]
    result_col = len(header) - 1 if header[-1] == "Result" else None
    for r in rows[1:]:
        cells = []
        for i, c in enumerate(r):
            cells.append(Paragraph(result_markup(c) if i == result_col else inline(c), cell))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(doc.leftMargin, 9 * mm, "Financial Numerical Recipes in C++: Mojo port, audit against the book")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


page = landscape(A4)
doc = SimpleDocTemplate(
    dst,
    pagesize=page,
    leftMargin=15 * mm,
    rightMargin=15 * mm,
    topMargin=14 * mm,
    bottomMargin=16 * mm,
    title="Audit: book examples against the Mojo port",
    author="financial-recipes-mojo",
)
avail = page[0] - doc.leftMargin - doc.rightMargin

story = []
lines = open(src, encoding="utf-8").read().split("\n")
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("# "):
        story.append(Paragraph(inline(line[2:]), h1))
        i += 1
    elif line.startswith("## "):
        story.append(Paragraph(inline(line[3:]), h2))
        i += 1
    elif line.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
            i += 1
        story.append(make_table(rows, avail))
        story.append(Spacer(1, 6))
    elif line.startswith("- "):
        items = []
        while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")):
            if lines[i].startswith("- "):
                items.append(lines[i][2:])
            else:
                items[-1] += " " + lines[i].strip()
            i += 1
        story.append(
            ListFlowable(
                [ListItem(Paragraph(inline(t), body), leftIndent=12) for t in items],
                bulletType="bullet",
                start="•",
                leftIndent=12,
            )
        )
        story.append(Spacer(1, 4))
    elif line.strip() == "":
        i += 1
    else:
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ")):
            para.append(lines[i].strip())
            i += 1
        text = " ".join(para)
        if text.startswith("Generated by"):
            continue
        story.append(Paragraph(inline(text), body))
        story.append(Spacer(1, 5))

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", dst)
