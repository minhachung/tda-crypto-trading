#!/usr/bin/env python
"""
Build a publication-quality PDF from RESULTS.md using reportlab.

Produces a paper-style document with:
- Cover page (title, author, date, repo)
- Properly formatted body (justified, indented paragraphs)
- Tables rendered as Table objects (not bare markdown)
- Embedded figures from results/figures/*.png
- Proper page breaks at section headings
"""

import os
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle, KeepTogether,
)
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path('/Users/vhsy.o34/Library/Mobile Documents/iCloud~md~obsidian/Documents/MinhaChung/tda-crypto-trading')


def make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles['title'] = ParagraphStyle(
        'PaperTitle', parent=base['Title'],
        fontSize=20, leading=24, alignment=TA_CENTER,
        spaceAfter=20, textColor=colors.black,
        fontName='Helvetica-Bold',
    )
    styles['author'] = ParagraphStyle(
        'Author', parent=base['Normal'],
        fontSize=12, leading=16, alignment=TA_CENTER,
        textColor=colors.HexColor('#333333'),
        fontName='Helvetica',
    )
    styles['date'] = ParagraphStyle(
        'Date', parent=base['Normal'],
        fontSize=11, leading=14, alignment=TA_CENTER,
        textColor=colors.HexColor('#666666'),
        fontName='Helvetica-Oblique',
    )
    styles['repo'] = ParagraphStyle(
        'Repo', parent=base['Normal'],
        fontSize=10, leading=12, alignment=TA_CENTER,
        textColor=colors.HexColor('#0066cc'),
        fontName='Helvetica',
    )
    styles['h1'] = ParagraphStyle(
        'PaperH1', parent=base['Heading1'],
        fontSize=15, leading=20, spaceBefore=18, spaceAfter=10,
        fontName='Helvetica-Bold', textColor=colors.HexColor('#1a1a1a'),
    )
    styles['h2'] = ParagraphStyle(
        'PaperH2', parent=base['Heading2'],
        fontSize=12.5, leading=17, spaceBefore=14, spaceAfter=7,
        fontName='Helvetica-Bold', textColor=colors.HexColor('#222222'),
    )
    styles['h3'] = ParagraphStyle(
        'PaperH3', parent=base['Heading3'],
        fontSize=11, leading=14, spaceBefore=10, spaceAfter=5,
        fontName='Helvetica-Bold', textColor=colors.HexColor('#333333'),
    )
    styles['body'] = ParagraphStyle(
        'PaperBody', parent=base['Normal'],
        fontSize=10.5, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=8, fontName='Helvetica',
        textColor=colors.HexColor('#1a1a1a'),
    )
    styles['abstract'] = ParagraphStyle(
        'Abstract', parent=base['Normal'],
        fontSize=10, leading=13.5, alignment=TA_JUSTIFY,
        spaceAfter=6, fontName='Helvetica',
        textColor=colors.HexColor('#1a1a1a'),
        leftIndent=20, rightIndent=20,
    )
    styles['caption'] = ParagraphStyle(
        'Caption', parent=base['Normal'],
        fontSize=9.5, leading=12, alignment=TA_CENTER,
        spaceAfter=10, fontName='Helvetica-Oblique',
        textColor=colors.HexColor('#444444'),
    )
    styles['code'] = ParagraphStyle(
        'Code', parent=base['Code'],
        fontSize=8.5, leading=11, fontName='Courier',
        textColor=colors.HexColor('#222222'),
        backColor=colors.HexColor('#f5f5f5'),
        borderColor=colors.HexColor('#dddddd'),
        borderWidth=0.5, borderPadding=4,
        leftIndent=10, rightIndent=10,
        spaceAfter=10,
    )
    styles['list'] = ParagraphStyle(
        'List', parent=styles['body'],
        leftIndent=24, bulletIndent=10,
    )
    return styles


def parse_md_inline(text):
    """Convert inline markdown (bold, italic, code, links) to reportlab markup."""
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="9">\1</font>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  r'<link href="\2" color="blue">\1</link>', text)
    # Math/math-like in $...$
    text = re.sub(r'\$([^$]+)\$', r'<i>\1</i>', text)
    text = text.replace('—', '—').replace('–', '–')
    return text


def parse_table(lines):
    """Parse markdown table to list of rows."""
    rows = []
    for line in lines:
        if line.strip().startswith('|---') or '---|' in line.replace(':', ''):
            continue
        if '|' not in line:
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        rows.append(cells)
    return rows


def build_table(rows, max_width=6.5):
    """Convert parsed rows into a reportlab Table."""
    if not rows:
        return None
    rendered = []
    for row in rows:
        rendered_row = [Paragraph(parse_md_inline(c),
                       ParagraphStyle('TC', fontSize=8.5, leading=11,
                                      fontName='Helvetica',
                                      textColor=colors.HexColor('#1a1a1a')))
                        for c in row]
        rendered.append(rendered_row)

    n_cols = len(rendered[0])
    col_width = (max_width * inch) / n_cols
    table = Table(rendered, colWidths=[col_width] * n_cols, hAlign='CENTER')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#222222')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#888888')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#f8f8f8'), colors.white]),
    ]))
    return table


def fix_check_marks(s):
    """Replace markdown checkmarks with reportlab-safe text."""
    return s.replace('✓', '[ok]').replace('✗', '[x]').replace('✅', '[ok]').replace('❌', '[x]')


def figure_path(name):
    p = PROJECT_ROOT / 'results' / 'figures' / name
    return str(p) if p.exists() else None


def build_paper(md_path, pdf_path):
    md = Path(md_path).read_text()
    styles = make_styles()
    story = []

    # Cover page
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(
        "Persistent Homology Detects<br/>"
        "Weak but Statistically Significant<br/>"
        "Predictive Structure in<br/>"
        "Cryptocurrency Returns",
        styles['title']
    ))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Minha Chung", styles['author']))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("May 2026", styles['date']))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "https://github.com/minhachung/tda-crypto-trading",
        styles['repo']
    ))
    story.append(Spacer(1, 1.5 * inch))

    # Pull-quote box on cover
    headline_style = ParagraphStyle(
        'Headline', parent=styles['body'],
        fontSize=11, leading=16, alignment=TA_CENTER,
        leftIndent=40, rightIndent=40,
        textColor=colors.HexColor('#222222'),
        fontName='Helvetica-Oblique',
    )
    story.append(Paragraph(
        "<b>Headline result:</b> 61.77% cross-validated direction accuracy at "
        "the 3-day horizon (n = 2,260, Wilson 95% CI [59.75%, 63.75%]) with an "
        "empirical permutation p-value &le; 0.04 (0/25 shuffled-data runs "
        "matched) and a held-out test accuracy of 69.32% "
        "(n = 315, CI [63.90%, 74.05%]). Statistically rigorous; "
        "economically marginal under typical retail fees.",
        headline_style
    ))
    story.append(PageBreak())

    # Body content
    lines = md.split('\n')
    i = 0
    skip_top_title = False
    in_code = False
    code_buf = []

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith('```'):
            if in_code:
                code_text = '\n'.join(code_buf).replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(
                    f'<font face="Courier" size="9">{code_text}</font>'.replace('\n', '<br/>'),
                    styles['code']
                ))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Skip the very first H1 (replaced by cover page)
        if line.startswith('# ') and not skip_top_title:
            skip_top_title = True
            i += 1
            continue

        if line.startswith('## '):
            heading = line[3:].strip()
            if heading.startswith('1.') or heading.startswith('2.') or heading.startswith('3.'):
                story.append(PageBreak())
            story.append(Paragraph(parse_md_inline(heading), styles['h1']))
            i += 1
            continue
        elif line.startswith('### '):
            story.append(Paragraph(parse_md_inline(line[4:].strip()), styles['h2']))
            i += 1
            continue
        elif line.startswith('#### '):
            story.append(Paragraph(parse_md_inline(line[5:].strip()), styles['h3']))
            i += 1
            continue

        # Tables (start with |)
        if line.strip().startswith('|'):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                tbl_lines.append(lines[i])
                i += 1
            rows = parse_table(tbl_lines)
            rows = [[fix_check_marks(c) for c in row] for row in rows]
            tbl = build_table(rows)
            if tbl:
                story.append(Spacer(1, 6))
                story.append(tbl)
                story.append(Spacer(1, 12))
            continue

        # Lists
        if re.match(r'^[\s]*[-*]\s', line):
            list_lines = []
            while i < len(lines) and (re.match(r'^[\s]*[-*]\s', lines[i]) or lines[i].strip() == ''):
                if lines[i].strip():
                    list_lines.append(re.sub(r'^[\s]*[-*]\s', '', lines[i]))
                i += 1
            for item in list_lines:
                story.append(Paragraph(
                    f'• {parse_md_inline(item)}',
                    styles['list']
                ))
            continue

        # Numbered lists
        if re.match(r'^[\s]*\d+\.\s', line):
            list_lines = []
            while i < len(lines) and (re.match(r'^[\s]*\d+\.\s', lines[i]) or lines[i].strip() == ''):
                if lines[i].strip():
                    list_lines.append(re.sub(r'^[\s]*\d+\.\s', '', lines[i]))
                i += 1
            for j, item in enumerate(list_lines, start=1):
                story.append(Paragraph(
                    f'{j}. {parse_md_inline(item)}',
                    styles['list']
                ))
            continue

        # Horizontal rule
        if line.strip() == '---':
            story.append(Spacer(1, 6))
            i += 1
            continue

        # Plain paragraph
        if line.strip():
            # Inline figure references — embed actual PNG when known
            if 'fig1_horizon_sweep' in line.lower():
                fp = figure_path('fig1_horizon_sweep.png')
                if fp:
                    story.append(Image(fp, width=6 * inch, height=3.6 * inch))
                    story.append(Paragraph(
                        "<b>Figure 1.</b> Direction accuracy by prediction horizon "
                        "with 95% Wilson confidence intervals. Green bars indicate "
                        "statistical significance.", styles['caption']
                    ))
                    i += 1
                    continue

            # Detect abstract section
            in_abstract = (i > 0 and any('## Abstract' in lines[k]
                            for k in range(max(0, i - 30), i)))
            if not any(line.startswith(h)
                       for h in ['## ', '### ', '#### ']):
                buf = [line]
                j = i + 1
                while j < len(lines) and lines[j].strip() and not lines[j].startswith('#') \
                    and not lines[j].strip().startswith('|') \
                    and not lines[j].strip().startswith('-') \
                    and not re.match(r'^[\s]*\d+\.\s', lines[j]) \
                    and not lines[j].strip().startswith('```'):
                    buf.append(lines[j])
                    j += 1
                paragraph_text = ' '.join(buf).strip()
                if paragraph_text:
                    style = styles['abstract'] if in_abstract else styles['body']
                    story.append(Paragraph(parse_md_inline(paragraph_text), style))
                i = j
                continue

        i += 1

    # Append figures section at end (visible in body)
    story.append(PageBreak())
    story.append(Paragraph("Appendix A — Figures", styles['h1']))

    for figname, caption in [
        ('fig1_horizon_sweep.png',
         '<b>Figure 1.</b> Direction accuracy by prediction horizon, with 95% '
         'Wilson confidence intervals. Five of six horizons (green) achieve '
         'statistical significance.'),
        ('fig2_sharpe_horizon.png',
         '<b>Figure 2.</b> Mean Sharpe ratio (left axis, blue) and mean fold '
         'return (right axis, orange) by prediction horizon. Sharpe improves '
         'as horizon lengthens — fees become a smaller fraction of per-trade '
         'move size.'),
        ('fig3_per_asset.png',
         '<b>Figure 3.</b> Per-asset direction accuracy (left) and TDA strategy '
         'vs buy-and-hold returns (right) at the best horizon. ADA and SOL show '
         'the highest predictability; BTC sits near chance.'),
        ('fig4_progression.png',
         '<b>Figure 4.</b> Validation methodology progression v1–v9. Left: '
         'direction accuracy with Wilson CI lower bound across versions. Right: '
         'sample size (log scale, n=2 to n=4,266) shows how each iteration '
         'expanded statistical power. Color coding distinguishes CV-only '
         'validation tiers (blue), the headline result and rigorous tier '
         '(green), and the continuous walk-forward profitability tier '
         '(purple).'),
    ]:
        fp = figure_path(figname)
        if fp:
            story.append(Spacer(1, 12))
            story.append(Image(fp, width=6.4 * inch, height=4 * inch))
            story.append(Paragraph(caption, styles['caption']))

    def page_footer(canv, doc):
        canv.saveState()
        canv.setFont('Helvetica', 9)
        canv.setFillColor(colors.HexColor('#666666'))
        canv.drawString(inch, 0.5 * inch,
                        'Topological Data Analysis in Crypto Returns — Minha Chung')
        canv.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page}")
        canv.restoreState()

    def cover_footer(canv, doc):
        if doc.page == 1:
            return
        page_footer(canv, doc)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=inch, rightMargin=inch,
        topMargin=0.75 * inch, bottomMargin=0.85 * inch,
        title="Persistent Homology in Cryptocurrency Returns",
        author="Minha Chung",
    )

    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=page_footer)
    print(f"Wrote: {pdf_path}")


if __name__ == '__main__':
    md = sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / 'results/RESULTS.md'
    pdf = sys.argv[2] if len(sys.argv) > 2 else PROJECT_ROOT / 'results/PAPER.pdf'
    build_paper(md, pdf)
