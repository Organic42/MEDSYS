"""
Build a single-column Word manuscript (for review and tracked changes) from
medsys-paper.tex. Compile the LaTeX first (with intermediates kept), because
citation, table, figure and equation numbers come from medsys-paper.aux and
the reference list from medsys-paper.bbl, so the Word copy matches the PDF.

  tectonic -X compile --keep-intermediates medsys-paper.tex
  python make_docx.py [out.docx]

Needs python-docx, matplotlib (equations) and PyMuPDF (figures).
"""
import io
import os
import re
import sys

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'medsys-paper.tex')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'medsys-paper.docx')
FONT = 'Times New Roman'
MONO = 'Consolas'

tex = re.sub(r'(?<!\\)%.*', '', open(SRC, encoding='utf-8').read())
aux = open(os.path.join(HERE, 'medsys-paper.aux'), encoding='utf-8').read()
bbl = open(os.path.join(HERE, 'medsys-paper.bbl'), encoding='utf-8').read()
CITE = dict(re.findall(r'\\bibcite\{([^}]+)\}\{(\d+)\}', aux))
LABEL = {k: v for k, v in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', aux)}


# ── brace-aware helpers ──────────────────────────────────────────────────

def group(s, i):
    """Return (content, end) for the {...} group starting at s[i] == '{'."""
    assert s[i] == '{', s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '{' and (j == 0 or s[j - 1] != '\\'):
            depth += 1
        elif s[j] == '}' and s[j - 1] != '\\':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError('unbalanced braces')


def split_top(s, sep):
    """Split s on sep outside braces (and not escaped)."""
    out, depth, cur, i = [], 0, '', 0
    while i < len(s):
        if s[i] == '{' and (i == 0 or s[i - 1] != '\\'):
            depth += 1
        elif s[i] == '}' and s[i - 1] != '\\':
            depth -= 1
        if depth == 0 and s.startswith(sep, i) and (i == 0 or s[i - 1] != '\\'):
            out.append(cur)
            cur, i = '', i + len(sep)
            continue
        cur += s[i]
        i += 1
    out.append(cur)
    return out


# ── inline LaTeX to runs ─────────────────────────────────────────────────

MATH_SYMBOLS = {
    r'\approx': '≈', r'\Delta': 'Δ', r'\eta': 'η', r'\sigma': 'σ', r'\tau': 'τ', r'\pm': '±',
    r'\times': '×', r'\leq': '≤', r'\geq': '≥', r'\le': '≤', r'\ge': '≥', r'\sum': 'Σ',
    r'\in': '∈', r'\cdot': '·', r'\ldots': '…', r'\infty': '∞', r'\max': 'max', r'\circ': '°',
    r'\{': '{', r'\}': '}', r'\%': '%', r'\,': '\u2009', r'\;': ' ', r'\!': '', r'\mm': '\u00a0mm',
    r'\textstyle': '', r'\left': '', r'\right': '', r'\sqrt': '√',
}
TEXT_SYMBOLS = {r'\%': '%', r'\&': '&', r'\_': '_', r'\#': '#', r'\,': '\u2009', r'\mm': '\u00a0mm',
                r'\ ': ' ', r'\\': ' '}


def math_runs(m, base):
    """Inline math to runs with sub/superscripts."""
    m = re.sub(r'\\(?:text|mathrm|mathit)\{([^}]*)\}', r'\1', m)
    m = re.sub(r'\\tfrac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', m)
    m = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', m)
    m = re.sub(r'\\bar\s*\{?([A-Za-z])\}?', '\\1\u0304', m)
    for k in sorted(MATH_SYMBOLS, key=len, reverse=True):
        m = m.replace(k, MATH_SYMBOLS[k])
    m = re.sub(r'\s*([=≈≤≥<>])\s*', r' \1 ', m).strip()   # spacing around relations
    runs, i, buf = [], 0, ''
    while i < len(m):
        c = m[i]
        if c in '^_':
            if buf:
                runs.append(dict(base, text=buf))
                buf = ''
            if i + 1 < len(m) and m[i + 1] == '{':
                inner, i = group(m, i + 1)
            else:
                inner, i = m[i + 1], i + 2
            runs.append(dict(base, text=inner.replace('{', '').replace('}', ''),
                             sup=(c == '^'), sub=(c == '_')))
            continue
        if c not in '{}':
            buf += c
        i += 1
    if buf:
        runs.append(dict(base, text=buf))
    return runs


ACCENTS = {'"': '̈', "'": '́', '`': '̀', '^': '̂', '~': '̃', 'c': '̧'}


def accents(s):
    """LaTeX accents ({\\"o}, \\'e, \\c{c}) to Unicode; drop bibliography spacing commands."""
    import unicodedata
    s = re.sub(r'\\hskip[^\\]*\\relax', ' ', s)
    s = re.sub(r'\{?\\(["\'`^~]|c\s)\{?([A-Za-z])\}?\}?', lambda m: m.group(2) + ACCENTS[m.group(1).strip()], s)
    return unicodedata.normalize('NFC', s)


def inline(s, base=None):
    """Inline LaTeX to a list of run dicts: text, italic, bold, mono, red, sup, sub."""
    base = dict(base or {})
    s = accents(s)
    s = s.replace('---', '—').replace('--', '–').replace('``', '“').replace("''", '”')
    runs, buf, i = [], '', 0

    def flush():
        nonlocal buf
        if buf:
            runs.append(dict(base, text=buf))
            buf = ''

    while i < len(s):
        c = s[i]
        if c == '$':
            j = s.index('$', i + 1)
            flush()
            runs += math_runs(s[i + 1:j], base)
            i = j + 1
            continue
        if c == '~':
            buf += '\u00a0'
            i += 1
            continue
        if c in '{}':
            i += 1
            continue
        if c == '\\':
            m = re.match(r'\\([A-Za-z]+|.)', s[i:])
            name = m.group(1)
            j = i + len(m.group(0))
            cmd = '\\' + name
            if cmd in TEXT_SYMBOLS:
                buf += TEXT_SYMBOLS[cmd]
                i = j
                continue
            style = {'emph': {'italic': True}, 'textit': {'italic': True}, 'textbf': {'bold': True},
                     'texttt': {'mono': True}, 'lvl': {'mono': True}, 'url': {'mono': True},
                     'todo': {'red': True}}.get(name)
            if style is not None or name in ('cite', 'ref', 'label', 'MakeLowercase'):
                while j < len(s) and s[j] == ' ':
                    j += 1
                arg, j = group(s, j)
                flush()
                if name == 'cite':
                    runs.append(dict(base, text=', '.join(f'[{CITE.get(k.strip(), "?")}]' for k in arg.split(','))))
                elif name == 'ref':
                    runs.append(dict(base, text=LABEL.get(arg, '?')))
                elif name == 'todo':
                    runs += inline('[TODO: ' + arg + ']', dict(base, red=True))
                elif name == 'url':
                    runs.append(dict(base, text=arg, mono=True))
                elif name != 'label':
                    runs += inline(arg, dict(base, **style))
                i = j
                continue
            i = j                                          # drop unknown commands, keep their text
            continue
        buf += c
        i += 1
    flush()
    return runs


# ── document helpers ─────────────────────────────────────────────────────

doc = Document()
for sec in doc.sections:
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(1)
normal = doc.styles['Normal']
normal.font.name = FONT
normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15


def add_runs(p, runs, size=None):
    for r in runs:
        run = p.add_run(r['text'])
        run.italic = r.get('italic')
        run.bold = r.get('bold')
        run.font.superscript = r.get('sup')
        run.font.subscript = r.get('sub')
        if r.get('mono'):
            run.font.name = MONO
        if r.get('red'):
            run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        if size:
            run.font.size = Pt(size)


def para(text, align=None, size=None, base=None, indent=True):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    elif indent:
        p.paragraph_format.first_line_indent = Inches(0.25)
    add_runs(p, inline(' '.join(text.split()), base), size)
    return p


def heading(text, level):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, inline(text, {'bold': level == 1, 'italic': level == 2}), 11)
    return p


def cell_shading_none(table):
    tbl = table._tbl
    margins = OxmlElement('w:tblCellMar')                 # tight cell padding (about 0.03 in)
    for side in ('left', 'right'):
        e = OxmlElement(f'w:{side}')
        e.set(qn('w:w'), '45')
        e.set(qn('w:type'), 'dxa')
        margins.append(e)
    tbl.tblPr.append(margins)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'bottom'):
        e = OxmlElement(f'w:{edge}')
        e.set(qn('w:val'), 'single')
        e.set(qn('w:sz'), '8')
        e.set(qn('w:color'), '000000')
        borders.append(e)
    tbl.tblPr.append(borders)


def row_rule(row):
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement('w:tcBorders')
        e = OxmlElement('w:bottom')
        e.set(qn('w:val'), 'single')
        e.set(qn('w:sz'), '4')
        e.set(qn('w:color'), '000000')
        b.append(e)
        tcPr.append(b)


ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']


# ── equations, figures and tables ────────────────────────────────────────

plt.rcParams['mathtext.fontset'] = 'stix'                 # Times-like, to match the text


def equation(body):
    """Render a display equation as an image, numbered like LaTeX (every equation is numbered)."""
    eq_count[0] += 1
    m = re.sub(r'\\label\{[^}]*\}', '', body).strip().rstrip(',.').strip()
    m = m.replace(r'\textstyle', '').replace(r'\text{', r'\mathrm{').replace(r'\tfrac', r'\frac')
    m = m.replace("s_a'^2", "s_a'^{2}").replace("s_a^2", "s_a^{2}")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f'${m}$', fontsize=14)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    px_w = Image.open(buf).size[0]
    buf.seek(0)
    row = doc.add_table(rows=1, cols=3).rows[0]             # borderless: spacer, equation, number
    for cell, w in zip(row.cells, (0.9, 4.7, 0.9)):
        cell.width = Inches(w)
    p = row.cells[1].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(buf, width=Inches(min(4.5, px_w / 300 * 0.95)))   # natural size
    q = row.cells[2].paragraphs[0]
    q.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    q.add_run(f'({eq_count[0]})')


eq_count = [0]


def figure(block, wide):
    path = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block).group(1)
    pdf = fitz.open(os.path.join(HERE, 'figures', path))
    png = pdf[0].get_pixmap(dpi=220).tobytes('png')
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(io.BytesIO(png), width=Inches(6.3 if wide else 3.6))
    cap_text, _ = group(block, block.index(r'\caption') + len(r'\caption'))
    label = re.search(r'\\label\{([^}]+)\}', block).group(1)
    cp = doc.add_paragraph()
    add_runs(cp, [{'text': f'Fig. {LABEL[label]}. '}] + inline(' '.join(cap_text.split())), 9.5)


def table(block):
    cap_text, _ = group(block, block.index(r'\caption') + len(r'\caption'))
    label = re.search(r'\\label\{([^}]+)\}', block).group(1)
    inp = re.search(r'\\input\{([^}]+)\}', block)
    if inp:
        block = open(os.path.join(HERE, inp.group(1)), encoding='utf-8').read()
        block = re.sub(r'(?<!\\)%.*', '', block)
    start = block.index(r'\begin{tabular}')
    spec, j = group(block, block.index('{', start + len(r'\begin{tabular}')))
    body = block[j:block.index(r'\end{tabular}')]
    body = re.sub(r'\\cmidrule(\([^)]*\))?\{[^}]*\}', '', body)
    for cmd in (r'\toprule', r'\bottomrule'):
        body = body.replace(cmd, '')
    rows, header_rows = [], 0
    for part in split_top(body, r'\\'):
        if r'\midrule' in part:
            if not header_rows:
                header_rows = len(rows)
            part = part.replace(r'\midrule', '')
        if part.strip():
            rows.append(part)
    def width(cell):
        mc = re.match(r'\s*\\multicolumn\{(\d+)\}', cell)
        return int(mc.group(1)) if mc else 1
    ncols = max(sum(width(c) for c in split_top(r, '&')) for r in rows)
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.keep_with_next = True
    tp.paragraph_format.space_before = Pt(8)
    add_runs(tp, [{'text': f'TABLE {LABEL[label]}\n'}] + inline(' '.join(cap_text.split())), 9.5)
    t = doc.add_table(rows=len(rows), cols=ncols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_shading_none(t)
    inches = [0.35] * ncols                                # widest single-column entry per column
    for ri, r in enumerate(rows):
        col = 0
        for c in split_top(r, '&'):
            if not c.strip().startswith(r'\multicolumn') and col < ncols:
                runs = inline(' '.join(c.split()))
                # 8.5 pt Times averages about 0.058 in per character, Consolas about 0.068
                w = sum(len(x['text']) * (0.068 if x.get('mono') else 0.058) for x in runs)
                inches[col] = max(inches[col], min(w, 1.9) + 0.12)
            col += 1
    excess = sum(inches) - 6.5                             # take any excess from the widest columns
    wide = [k for k, w in enumerate(inches) if w > 0.9 and k > 0]   # never break the row labels
    if excess > 0 and wide:
        pool = sum(inches[k] for k in wide)
        for k in wide:
            inches[k] -= excess * inches[k] / pool
    widths = [Inches(w) for w in inches]
    for ri, r in enumerate(rows):
        col = 0
        for c in split_top(r, '&'):
            span, text = 1, c.strip()
            if text.startswith(r'\multicolumn'):
                n, k = group(text, len(r'\multicolumn'))
                _, k = group(text, k)                      # column spec
                span = int(n)
                text, _ = group(text, k)
            cell = t.cell(ri, col)
            if span > 1:
                cell = cell.merge(t.cell(ri, min(col + span - 1, ncols - 1)))
            cell.paragraphs[0].text = ''
            add_runs(cell.paragraphs[0], inline(' '.join(text.split()), {'bold': ri < header_rows}), 8.5)
            col += span
        if ri == header_rows - 1:
            row_rule(t.rows[ri])
    t.autofit = False
    for row in t.rows:
        for k, cell in enumerate(row.cells):
            cell.width = widths[min(k, ncols - 1)]
    for k, column in enumerate(t.columns):
        column.width = widths[k]
    doc.add_paragraph()


# ── front matter ─────────────────────────────────────────────────────────

title, _ = group(tex, tex.index(r'\title') + len(r'\title'))
para(title, align=WD_ALIGN_PARAGRAPH.CENTER, size=17, base={'bold': True})
authors = re.findall(r'\\IEEEauthorblockN\{([^}]*)\}\s*\\IEEEauthorblockA\{([^}]*)\}', tex)
at = doc.add_table(rows=1, cols=len(authors))
for k, (name, aff) in enumerate(authors):
    p = at.cell(0, k).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, [{'text': name, 'bold': True}], 11.5)
    for line in aff.split(r'\\'):
        q = at.cell(0, k).add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(q, inline(line.strip()), 10)
doc.add_paragraph()

abstract = tex[tex.index(r'\begin{abstract}') + 16:tex.index(r'\end{abstract}')]
p = doc.add_paragraph()
add_runs(p, [{'text': 'Abstract—', 'bold': True, 'italic': True}] + inline(' '.join(abstract.split()), {'bold': True}), 10.5)
keywords = tex[tex.index(r'\begin{IEEEkeywords}') + 20:tex.index(r'\end{IEEEkeywords}')]
p = doc.add_paragraph()
add_runs(p, [{'text': 'Index Terms—', 'bold': True, 'italic': True}] + inline(' '.join(keywords.split()), {'bold': True}), 10.5)

# ── body ─────────────────────────────────────────────────────────────────

body = tex[tex.index(r'\section{Introduction}'):tex.index(r'\bibliographystyle')]
pattern = re.compile(r'\\(section|subsection)(\*?)\{|\\begin\{(table\*?|figure\*?|equation)\}|\n\s*\n')
sec_n, sub_n, pos, pending = 0, 0, 0, ''


def flush_text(text, indent=True):
    for chunk in re.split(r'\n\s*\n', text):
        if chunk.strip():
            para(chunk, indent=indent)


while True:
    m = pattern.search(body, pos)
    if not m:
        flush_text(body[pos:])
        break
    text = body[pos:m.start()]
    if m.group(1):                                         # section or subsection
        flush_text(text)
        name, end = group(body, m.end() - 1)
        if m.group(1) == 'section':
            if m.group(2):
                heading(name.upper(), 1)
            else:
                sec_n += 1
                sub_n = 0
                heading(f'{ROMAN[sec_n - 1]}. {name.upper()}', 1)
        else:
            sub_n += 1
            heading(f'{chr(64 + sub_n)}. {name}', 2)
        pos = end
    elif m.group(3):                                       # environment
        env = m.group(3)
        end_tag = '\\end{' + env + '}'
        end = body.index(end_tag, m.end()) + len(end_tag)
        block = body[m.end():end - len(end_tag)]
        if env == 'equation':
            flush_text(text)
            equation(block)
            rest_start = end
            nxt = pattern.search(body, rest_start)
            stop = nxt.start() if nxt else len(body)
            flush_text(body[rest_start:stop], indent=False)
            pos = stop
            continue
        flush_text(text)
        if env.startswith('table'):
            table(block)
        else:
            figure(block, env.endswith('*'))
        pos = end
    else:                                                  # paragraph break
        flush_text(text)
        pos = m.end()

# ── references ───────────────────────────────────────────────────────────

heading('REFERENCES', 1)
for key, entry in re.findall(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})', bbl, re.S):
    entry = re.sub(r'\\BIBentry\w+', '', entry)
    entry = re.sub(r'\\newblock', '', entry)
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.4)
    add_runs(p, [{'text': f'[{CITE[key]}]\t'}] + inline(' '.join(entry.split())), 9.5)

doc.save(OUT)
print(f'{OUT}: {len(CITE)} references, {eq_count[0]} display equations')
