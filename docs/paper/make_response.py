"""Write the point-by-point response to the review of the first version.

The responses live in response.json (point, what was done, where), so the
PDF and the Word file always say the same thing:

  python docs/paper/make_response.py            # -> response-to-review.tex/.docx
  tectonic -X compile docs/paper/response-to-review.tex
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, 'response.json')


def tex_escape(s):
    """Plain text with *emphasis* and `code` marks to LaTeX."""
    s = s.replace('\\', r'\textbackslash{}')
    for a, b in (('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('$', r'\$')):
        s = s.replace(a, b)
    s = re.sub(r'`([^`]+)`', r'\\texttt{\1}', s)
    s = re.sub(r'\*([^*]+)\*', r'\\emph{\1}', s)
    for a, b in (('≈', r'$\approx$'), ('Δ', r'$\Delta$'), ('η²', r'$\eta^2$'), ('±', r'$\pm$'),
                 ('≤', r'$\leq$'), ('≥', r'$\geq$'), ('→', r'$\rightarrow$'), ('×', r'$\times$'),
                 ('−', '--'), ('²', r'$^2$'), ('³', r'$^3$'), ('ρ', r'$\rho$')):
        s = s.replace(a, b)
    return s


def write_tex(data, path):
    out = [r'\documentclass[11pt]{article}',
           r'\usepackage[a4paper,margin=2.2cm]{geometry}',
           r'\usepackage{booktabs,array,longtable,xcolor,iftex}',
           r'\ifXeTeX\usepackage{fontspec}\setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular,'
           r' BoldFont=*-bold, ItalicFont=*-italic, BoldItalicFont=*-bolditalic]\fi',
           r'\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}',
           r'\begin{document}',
           r'{\Large\bfseries ' + tex_escape(data['title']) + r'\par}\medskip',
           tex_escape(data['subtitle']) + r'\par',
           r'\textit{' + tex_escape(data['authors']) + r'}\par\medskip']
    for para in data['summary']:
        out.append(tex_escape(para) + r'\par')
    for section in data['sections']:
        out.append(r'\section*{' + tex_escape(section['heading']) + '}')
        out.append(r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.22\textwidth}'
                   r'>{\raggedright\arraybackslash}p{0.56\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}@{}}')
        out.append(r'\toprule Review point & What we did & Where \\ \midrule\endhead')
        for item in section['items']:
            out.append(r'\textbf{' + tex_escape(item['id']) + r'} ' + tex_escape(item['point']) + ' & '
                       + tex_escape(item['response']) + ' & ' + tex_escape(item['where']) + r' \\ \midrule')
        out[-1] = out[-1].replace(r'\midrule', r'\bottomrule')
        out.append(r'\end{longtable}')
    if data.get('not_done'):
        out.append(r'\section*{' + tex_escape(data['not_done']['heading']) + '}')
        out.append(r'\begin{itemize}')
        for x in data['not_done']['items']:
            out.append(r'\item ' + tex_escape(x))
        out.append(r'\end{itemize}')
    out.append(r'\end{document}')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out) + '\n')


def write_docx(data, path):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.shared import Cm, Pt

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.0)
        s.top_margin = s.bottom_margin = Cm(2.0)
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(10.5)

    def runs(p, text):
        for part in re.split(r'(\*[^*]+\*|`[^`]+`)', text):
            if not part:
                continue
            if part.startswith('*'):
                p.add_run(part[1:-1]).italic = True
            elif part.startswith('`'):
                r = p.add_run(part[1:-1])
                r.font.name = 'Consolas'
            else:
                p.add_run(part)

    doc.add_heading(data['title'], level=1)
    doc.add_paragraph(data['subtitle'])
    doc.add_paragraph().add_run(data['authors']).italic = True
    for para in data['summary']:
        runs(doc.add_paragraph(), para)
    widths = (Cm(4.0), Cm(10.2), Cm(2.8))
    for section in data['sections']:
        doc.add_heading(section['heading'], level=2)
        t = doc.add_table(rows=1, cols=3)
        t.style = 'Table Grid'
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, label, w in zip(t.rows[0].cells, ('Review point', 'What we did', 'Where'), widths):
            cell.width = w
            cell.paragraphs[0].add_run(label).bold = True
        for item in section['items']:
            cells = t.add_row().cells
            p = cells[0].paragraphs[0]
            p.add_run(item['id'] + ' ').bold = True
            runs(p, item['point'])
            runs(cells[1].paragraphs[0], item['response'])
            runs(cells[2].paragraphs[0], item['where'])
            for c, w in zip(cells, widths):
                c.width = w
    if data.get('not_done'):
        doc.add_heading(data['not_done']['heading'], level=2)
        for x in data['not_done']['items']:
            runs(doc.add_paragraph(style='List Bullet'), x)
    doc.save(path)


def main():
    with open(SOURCE, encoding='utf-8') as fh:
        data = json.load(fh)
    write_tex(data, os.path.join(HERE, 'response-to-review.tex'))
    write_docx(data, os.path.join(HERE, 'response-to-review.docx'))
    print('response-to-review.tex and .docx ->', HERE)


if __name__ == '__main__':
    main()
