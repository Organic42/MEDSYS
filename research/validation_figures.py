"""Figures for validation tests 1 and 2 (PNG and PDF).

    python -m research.validation_figures <card_validation run> [<repeatability run>] [--out DIR]

Colours: blue for the reference or earlier version, orange for the held-out
organ, the corrected card or the app's default path (checked for colour-vision deficiency with the
dataviz validator: CVD dE 25, normal-vision dE 33). Every series is also
labelled in the legend, so identity never rests on colour alone.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.tier2 import results_file  # noqa: E402

BLUE, ORANGE = '#3D74C9', '#E8650A'
INK, MUTED, GRID = '#1c1c1e', '#6b6b70', '#e6e3df'


def _style(plt):
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9,
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                         'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def _save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'))


def coverage_figure(card_run, out, plt, headline=True):
    """Before/after: share of scans inside the card's middle half and 90% range."""
    import pandas as pd
    from research.card_validation import coverage
    df = pd.read_csv(results_file(card_run))
    ok = df[(df.source != 'gt') & (df.status == 'ok')]
    rows = []
    for card in ('v1', 'v2'):
        c = coverage(ok, card)
        c = c[c.what == 'true_assd_mm'].groupby(['organ', 'mode', 'level'], sort=False)[['mid', 'p90']].mean()
        rows.append(c.add_prefix(card + '_'))
    cov = pd.concat(rows, axis=1).reset_index()
    order = {'native': 0, 'iso1.5': 1, 'iso3': 2, 'slice5': 3}
    names = {'native': 'native', 'iso1.5': '1.5 mm', 'iso3': '3 mm', 'slice5': '5 mm slices'}
    cov = cov.sort_values(['organ', 'mode', 'level'], key=lambda s: s.map(order) if s.name == 'level' else s,
                          ascending=[False, True, True]).reset_index(drop=True)
    labels = [f"{r.organ.capitalize()} · {r['mode']} · {names[r.level]}" for _, r in cov.iterrows()]
    y = np.arange(len(cov))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.4), sharey=True)
    for ax, key, target, title in ((axes[0], 'mid', 50, 'Inside the middle half (target 50%)'),
                                   (axes[1], 'p90', 90, 'Inside the 90% range (target 90%)')):
        a, b = 100 * cov[f'v1_{key}'], 100 * cov[f'v2_{key}']
        ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color=GRID, linewidth=2, zorder=1)
        ax.scatter(a, y, s=42, color=BLUE, edgecolor='white', linewidth=1.5, zorder=3, label='Card v1 (as shipped)')
        ax.scatter(b, y, s=42, color=ORANGE, edgecolor='white', linewidth=1.5, zorder=4,
                   label='Card v2 (grid-matched)')
        ax.axvline(target, color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Share of scans (%), leave-one-scan-out')
        ax.set_title(title, loc='left', fontsize=9.5, color=INK, fontweight='semibold')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(y, labels)
    worst = cov.loc[cov.v1_mid.idxmin()]
    axes[0].annotate(f"{100 * worst.v1_mid:.0f}% → {100 * worst.v2_mid:.0f}%",
                     (100 * worst.v1_mid, y[cov.v1_mid.idxmin()]), xytext=(8, -12), textcoords='offset points',
                     fontsize=8, color=INK)
    axes[1].legend(loc='lower left', fontsize=8)
    if headline:                        # slides and the report; a paper figure has a caption instead
        fig.suptitle('Is the accuracy card honest? True error of the delivered mesh against the card\'s range',
                     x=0.0, ha='left', fontsize=11, fontweight='semibold', color=INK)
        fig.text(0.0, -0.02, 'TotalSegmentator on 41 spleen and 41 pancreas CT scans with expert labels (MSD), '
                 f'meshed by the app\'s own code; {len(ok)} meshes. Dashed: target. Where only orange shows, '
                 'both versions agree.', fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, out, 'card_coverage')
    plt.close(fig)


def resolution_figure(card_run, out, plt):
    """The resolution line: coarsened expert mask, observed vs 0.177 x Delta."""
    import pandas as pd
    df = pd.read_csv(results_file(card_run))
    gt = df[df.source == 'gt']
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    d = np.linspace(0, gt.delta_added_mm.max() * 1.05, 50)
    ax.fill_between(d, 0.177 * d - 0.111, 0.177 * d + 0.111, color=GRID, alpha=0.9, linewidth=0,
                    label='±0.111 mm prediction interval')
    ax.plot(d, 0.177 * d, color=INK, linewidth=2, label='Card: 0.177 × Δ')
    for organ, colour, label in (('spleen', BLUE, 'Spleen (fit)'), ('pancreas', ORANGE, 'Pancreas (held out)')):
        g = gt[gt.structure == organ]
        ax.scatter(g.delta_added_mm, g.canon_assd_mm, s=22, color=colour, edgecolor='white', linewidth=1,
                   zorder=3, label=f'{label}, {len(g)} grids')
    pa = gt[gt.structure == 'pancreas']
    inside = ((pa.canon_assd_mm - 0.177 * pa.delta_added_mm).abs() <= 0.111).mean()
    ax.set_xlabel('Sampling added by the coarser grid, Δ (mm)')
    ax.set_ylabel('Surface error of a perfect segmentation (ASSD, mm)')
    ax.set_xlim(0, d.max())
    ax.set_ylim(0, None)
    ax.set_title(f'Resolution limit: {100 * inside:.0f}% of held-out pancreas grids inside the interval',
                 loc='left', fontsize=9.5, color=INK, fontweight='semibold')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    _save(fig, out, 'resolution_limit')
    plt.close(fig)


def repeatability_figure(rep_run, out, plt):
    """How far a repeated run moves the mesh, next to the error being measured."""
    import pandas as pd
    df = pd.read_csv(results_file(rep_run))
    rep = df[~((df.path == 'worker') & (df.run == 0))]
    paths = [('worker', 'Same GPU worker (app default)'), ('fresh_gpu', 'New GPU process'), ('cpu', 'CPU')]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    for ax, mode in zip(axes, ('fast', 'full')):
        for i, (path, label) in enumerate(paths):
            g = rep[(rep['mode'] == mode) & (rep.path == path)]
            if not len(g):
                continue
            jitter = (np.random.default_rng(i).random(len(g)) - 0.5) * 0.3
            colour = ORANGE if path == 'worker' else BLUE
            ax.scatter(g.mesh_assd_vs_first_mm, np.full(len(g), len(paths) - 1 - i) + jitter, s=18,
                       color=colour, edgecolor='white', linewidth=0.8, zorder=3)
            top = g.mesh_assd_vs_first_mm.max()
            ax.text(top + 0.0015, len(paths) - 1 - i,
                    'identical every time' if top == 0 else f'max {top:.3f} mm', va='center', fontsize=7.5,
                    color=INK)
        ax.set_title(f'TotalSegmentator {mode} mode', loc='left', fontsize=9.5, color=INK, fontweight='semibold')
        ax.set_xlabel('Mesh shift from the first run (ASSD, mm)')
        ax.set_xlim(-0.001, max(0.03, rep.mesh_assd_vs_first_mm.max() * 1.6))
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(paths)), [p[1] for p in paths][::-1])
    fig.suptitle('Same scan, run again: how far does the model move?', x=0.0, ha='left', fontsize=11,
                 fontweight='semibold', color=INK)
    fig.text(0.0, -0.06, f'{df.case.nunique()} CT scans, {len(df)} segmentation runs. For scale: the typical '
             'surface error itself is 0.7-1.9 mm, and the resolution limit of a 1 mm scan is 0.18 mm.',
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, out, 'repeatability')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('card_run')
    ap.add_argument('rep_run', nargs='?')
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'validation', 'figures'))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    os.makedirs(args.out, exist_ok=True)
    coverage_figure(args.card_run, args.out, plt)
    resolution_figure(args.card_run, args.out, plt)
    if args.rep_run:
        repeatability_figure(args.rep_run, args.out, plt)
    print(f'figures -> {args.out}')


if __name__ == '__main__':
    main()
