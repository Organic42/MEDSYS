"""Draw Fig. 1 of the paper (study overview) as figures/overview.pdf.

Run from the repository root: python docs/paper/make_overview.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
INK, MUTED, FILL, ACCENT = '#1F2933', '#667085', '#F2F4F7', '#FFE6D1'

plt.rcParams.update({'font.size': 6.6, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(7.16, 1.95))
ax.set_xlim(-0.5, 100.5)
ax.set_ylim(0, 29)
ax.axis('off')


def box(x, y, w, h, text, fill=FILL, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8',
                                linewidth=0.8, edgecolor=INK, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', color=INK, linespacing=1.25,
            fontweight='bold' if bold else 'normal')


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=8,
                                 linewidth=0.8, color=INK))


def row(y, label, boxes, compare):
    ax.text(0.8, y + 8.9, label, ha='left', va='bottom', color=INK, fontweight='bold')
    x, h, gap = 0.8, 8.0, 2.6
    for i, (w, text, fill) in enumerate(boxes):
        box(x, y, w, h, text, fill)
        if i < len(boxes) - 1:
            arrow(x + w + 0.35, x + w + gap - 0.35, y + h / 2)
        x += w + gap
    arrow(x - gap + 0.35, x - 0.35, y + h / 2)
    box(x, y, 100 - x - 0.4, h, compare, '#FFFFFF', bold=False)


recon = 'Reconstruct (shared code):\nblur, marching cubes,\nTaubin, decimation'
row(17.5, 'Tier 1: digital phantoms (exact truth)', [
    (19.0, 'Analytic shape with\nexact signed\ndistance function', FILL),
    (19.5, 'Voxelise at h =\n0.5, 1, 2 mm or\n0.8 × 0.8 × 5 mm', ACCENT),
    (24.0, recon, FILL),
], 'Score against the\nexact surface')
row(4.0, 'Tier 2: real anatomy (expert reference)', [
    (19.0, 'CT scan and\nexpert spleen mask\n(41 scans)', FILL),
    (19.5, 'Resample: native,\n1.5 mm, 3 mm or\n5 mm slices', ACCENT),
    (24.0, 'Segment (expert mask\nor TotalSegmentator),\nthen reconstruct', FILL),
], 'Score against the\nreference surface\n(expert mesh, native)')
for x, w, text in ((22.4, 19.5, 'sampling error'), (44.5, 24.0, 'segmentation and reconstruction error')):
    ax.text(x + w / 2, 0.4, text, ha='center', va='bottom', color=MUTED, fontsize=6.4, style='italic')

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'overview.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'overview.png'), bbox_inches='tight', dpi=200)
print('overview ->', OUT)
