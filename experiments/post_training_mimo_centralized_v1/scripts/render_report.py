"""Deterministic PDF layout of the versioned Markdown cycle report (ReportLab)."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether


def inline(text):
    text = html.escape(text.replace("\u2014", " - ").replace("\u2013", "-"))
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="7.2">\1</font>', text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def render(source, output):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MainTitle",fontName="Helvetica-Bold",fontSize=22,leading=26,
                             textColor=colors.HexColor("#173b49"),spaceAfter=16))
    styles.add(ParagraphStyle(name="SectionTitle",fontName="Helvetica-Bold",fontSize=12,leading=15,
                             textColor=colors.HexColor("#196775"),spaceBefore=10,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportBody",fontName="Helvetica",fontSize=9.2,leading=13,
                             spaceAfter=7,alignment=TA_LEFT,splitLongWords=True,allowWidows=0,allowOrphans=0))
    styles.add(ParagraphStyle(name="Cell",fontName="Helvetica",fontSize=7.8,leading=11,spaceAfter=0))
    doc = SimpleDocTemplate(str(output),pagesize=(612,792),leftMargin=48,rightMargin=48,
                            topMargin=50,bottomMargin=48,title="Rineya Titan 0.6B Qualification Cycle",
                            author="Rineya research",invariant=1)
    story=[]
    for block in re.split(r"\n\s*\n", source.read_text(encoding="utf-8").strip()):
        lines=block.splitlines()
        if lines[0].startswith("# "):
            story.append(Paragraph(inline(lines[0][2:]),styles["MainTitle"]))
        elif lines[0].startswith("## "):
            story.append(Paragraph(inline(lines[0][3:]),styles["SectionTitle"]))
        elif all(line.startswith("|") for line in lines):
            rows=[]
            for line in lines:
                cells=[c.strip() for c in line.strip("|").split("|")]
                if all(re.fullmatch(r"[-:]+",c) for c in cells):
                    continue
                rows.append([Paragraph(inline(c),styles["Cell"]) for c in cells])
            table=Table(rows,colWidths=[83,156,83,194],repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dcecef")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#196775")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7f8")]),
                ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
                ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            story.extend([table,Spacer(1,8)])
        else:
            story.append(Paragraph(inline(" ".join(lines)),styles["ReportBody"]))
    def page(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#b8ced3"))
        canvas.line(48,32,564,32)
        canvas.setFont("Helvetica",7)
        canvas.setFillColor(colors.HexColor("#52646a"))
        canvas.drawString(48,21,"RINEYA RESEARCH  |  30 SEPTEMBER 2026  |  NO EFFICACY CLAIM")
        canvas.drawRightString(564,21,str(doc.page))
        canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    render(args.source,args.output)
