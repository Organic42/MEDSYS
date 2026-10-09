"""
Figures and LaTeX tables for a Tier 2 run that includes the resolution factor.

Usage (from the repository root):
  python -m research.tier2_report output/research/tier2/<run> --out docs/tier2-report
  python -m research.tier2_report <run> --out <dir> --baseline output/research/tier2/<native-only run>

Writes <out>/figures/*.pdf (with .png previews), <out>/tables/*.tex and
<out>/numbers.json. Every number in the tables is computed from the run's
results.csv, so the report can be rebuilt from the data alone. --baseline
adds a reproducibility table comparing native rows with an earlier run.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.tier2 import RESOLUTIONS, eta_squared, results_file  # noqa: E402

# MEDSYS palette: dark text for native, anatomy colours for the coarser levels.
COLOURS = {'native': '#1F2933', 'iso1.5': '#4F91D9', 'iso3': '#FF7A18', 'slice5': '#2A9D8F'}
SOURCES = {'gt': 'Expert mask', 'ts_fast': 'TotalSegmentator fast', 'classical': 'Classical'}
TEXT, TEXT2, RULE = '#1F2933', '#667085', '#D0D5DD'
CONFIGS = (('canonical', 'Canonical'), ('production', 'Production'))


# ── data ─────────────────────────────────────────────────────────────────

def _bool(s):
    return s.astype(str).str.lower() == 'true'


def load(run_dir):
    df = pd.read_csv(results_file(run_dir))
    with open(os.path.join(run_dir, 'run.json'), encoding='utf-8') as fh:
        run = json.load(fh)
    for col in ('resampled', 'is_canonical', 'topology_ok'):
        if col in df:
            df[col] = _bool(df[col])
    grid = df['grid_spacing'].str.split('x', expand=True).astype(float).to_numpy()
    native = df['spacing'].str.split('x', expand=True).astype(float).to_numpy()
    # Sampling added by coarsening, in quadrature: sqrt(mean over axes of s'^2 - s^2).
    # Equals the spacing h when an isotropic grid h replaces an infinitely fine one.
    df['added_mm'] = np.sqrt(np.clip(grid ** 2 - native ** 2, 0, None).mean(axis=1))
    order = [r for r in RESOLUTIONS if r in set(df['resolution'])]
    df['resolution'] = pd.Categorical(df['resolution'], categories=order, ordered=True)
    df['cfg'] = np.where(df['config'] == 'production', 'production',
                         np.where(df['is_canonical'], 'canonical', df['config']))
    return df, run


def split(df):
    """Mask and mesh rows. Non-native levels keep only scans they actually
    resampled: elsewhere the level equals native and would dilute its effect."""
    ok = (df['status'] == 'ok') & ((df['resolution'] == 'native') | df['resampled'])
    return df[ok & (df['level'] == 'mask')], df[ok & (df['level'] == 'mesh')]


# ── formatting ───────────────────────────────────────────────────────────

def pm(s, d=3):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if s.empty:
        return '--'
    return f'{s.mean():.{d}f}' if len(s) == 1 else f'{s.mean():.{d}f} $\\pm$ {s.std():.{d}f}'


def pct(x):
    return '--' if pd.isna(x) else f'{100 * x:.0f}\\%'


def pval(p):
    return '--' if pd.isna(p) else ('$<$0.001' if p < 0.001 else f'{p:.3f}')


def code(name):
    return f'\\texttt{{{name}}}'


def write_table(path, header, rows, align, group=None):
    """A booktabs tabular; rows of None are \\midrule; group is an optional header line above."""
    out = [f'\\begin{{tabular}}{{{align}}}', '\\toprule'] + ([group] if group else []) + [
        ' & '.join(header) + ' \\\\', '\\midrule']
    for r in rows:
        out.append('\\midrule' if r is None else ' & '.join(str(c) for c in r) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}', '']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out))


def boot_ci(x, rng, n=5000):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── tables ───────────────────────────────────────────────────────────────

def tables(df, run, out, baseline=None):
    from scipy.stats import wilcoxon
    os.makedirs(out, exist_ok=True)
    mask, mesh = split(df)
    order = list(df['resolution'].cat.categories)
    sources = [s for s in SOURCES if s in set(df['source'])]
    nums = {'cases': int(df['case'].nunique()), 'rows': int(len(df)), 'mesh_rows': int(len(mesh)),
            'wall_time_min': round(run['wall_time_s'] / 60, 1), 'commit': run.get('commit')}

    # T1: resolution levels
    per = df.drop_duplicates(['case', 'resolution'])
    rows = []
    for r in order:
        g = per[per['resolution'] == r]
        sp = g['grid_spacing'].str.split('x', expand=True).astype(float)
        inplane, slice_mm = RESOLUTIONS[r]
        rule = 'as acquired' if r == 'native' else (
            f'$\\geq${inplane:g}\\,mm in-plane, $\\geq${slice_mm:g}\\,mm slice' if inplane
            else f'native in-plane, $\\geq${slice_mm:g}\\,mm slice')
        rows.append([code(r), rule, f"{int(g['resampled'].sum())}/{len(g)}",
                     f'{sp[0].median():.2f}', f'{sp[2].median():.1f}',
                     f'{sp[2].min():.1f}--{sp[2].max():.1f}'])
        nums[f'resampled_{r}'] = int(g['resampled'].sum())
    write_table(os.path.join(out, 'levels.tex'),
                ['Level', 'Rule', 'Resampled', 'In-plane (median, mm)', 'Slice (median, mm)', 'Slice range (mm)'],
                rows, 'llrrrr')

    # T2: mask level
    rows = []
    for src in sources:
        for r in order:
            g = mask[(mask['source'] == src) & (mask['resolution'] == r)]
            if not len(g):
                continue
            vol = (g['pred_cm3'] - g['gt_cm3']) / g['gt_cm3'] * 100
            rows.append([SOURCES[src] if src != 'gt' else 'Expert mask, resampled', code(r), len(g),
                         pm(g['dice']), pm(g['mask_hd95_mm'], 2), pm(g['mask_assd_mm'], 2), pm(vol, 1)])
            nums[f'dice_{src}_{r}'] = round(float(g['dice'].mean()), 4)
            nums[f'mask_hd95_{src}_{r}'] = round(float(g['mask_hd95_mm'].mean()), 2)
        rows.append(None)
    write_table(os.path.join(out, 'mask.tex'),
                ['Source', 'Level', '$n$', 'Dice', 'HD95 (mm)', 'ASSD (mm)', 'Volume error (\\%)'],
                rows[:-1], 'llrrrrr')

    # T3: mesh level, canonical and production
    rows = []
    for src in sources:
        for r in order:
            for cfg, label in CONFIGS:
                g = mesh[(mesh['source'] == src) & (mesh['resolution'] == r) & (mesh['cfg'] == cfg)]
                if not len(g):
                    continue
                rows.append([SOURCES[src], code(r), label, pm(g['assd_mm']), pm(g['hd95_mm'], 2),
                             pm(g['nsd'], 3), pm(g['mean_signed_mm']), pm(g['volume_err_pct'], 1),
                             pct(g['topology_ok'].mean())])
                for k in ('assd_mm', 'hd95_mm', 'mean_signed_mm', 'volume_err_pct', 'nsd'):
                    nums[f'{k}_{src}_{r}_{cfg}'] = round(float(g[k].mean()), 4)
                    nums[f'{k}_sd_{src}_{r}_{cfg}'] = round(float(g[k].std()), 4)
                nums[f'topo_{src}_{r}_{cfg}'] = round(float(g['topology_ok'].mean()), 3)
        rows.append(None)
    write_table(os.path.join(out, 'mesh.tex'),
                ['Source', 'Level', 'Config', 'ASSD (mm)', 'HD95 (mm)', 'NSD@1\\,mm', 'Signed (mm)',
                 'Volume (\\%)', 'Closed'], rows[:-1], 'lllrrrrrr')

    # T4: variance shares over the grid. ASSD in mm is primary: the canonical
    # expert mesh at native reproduces the reference exactly, so ASSD reaches 0
    # there and log-ASSD is ill-conditioned. Log-ASSD is shown where it is
    # well defined (expert mask below native; the engine everywhere).
    grid = grid_rows(mesh).copy()
    factors = ['source', 'resolution', 'blur_mm', 'taubin_iter', 'decimation', 'case']
    sel = {'all': grid}
    sel.update({s: grid[grid['source'] == s] for s in sources})
    shares = {k: eta_squared(g, 'assd_mm', [f for f in factors if not (k != 'all' and f == 'source')])
              for k, g in sel.items()}
    log_sel = {s: (g[g['resolution'] != 'native'] if s == 'gt' else g).copy()
               for s, g in sel.items() if s != 'all'}
    log_shares = {}
    for s, g in log_sel.items():
        g['log_assd'] = np.log(g['assd_mm'])
        log_shares[s] = eta_squared(g, 'log_assd', [f for f in factors if f != 'source'])
    names = {'source': 'Mask source', 'resolution': 'Resolution level', 'blur_mm': 'Blur',
             'taubin_iter': 'Taubin iterations', 'decimation': 'Decimation', 'case': 'Case (patient)'}
    cols = [shares[k] for k in sel] + [log_shares[s] for s in log_sel]
    rows = []
    for f in factors:
        rows.append([names[f]] + [('--' if f not in c else f'{100 * c[f]:.1f}\\%') for c in cols])
    rows.append(None)
    rows.append(['Interactions + residual'] + [f'{100 * (1 - sum(c.values())):.1f}\\%' for c in cols])
    short = {'gt': 'Expert', 'ts_fast': 'TotalSeg.', 'classical': 'Classical'}
    header = (['Factor', 'All'] + [short[s] for s in sources]
              + [short[s] + (' (coarse)' if s == 'gt' else '') for s in log_sel])
    n1, n2 = len(sel), len(log_sel)
    group = (f'& \\multicolumn{{{n1}}}{{c}}{{ASSD (mm)}} & \\multicolumn{{{n2}}}{{c}}{{log ASSD}} \\\\\n'
             f'\\cmidrule(lr){{2-{1 + n1}}}\\cmidrule(lr){{{2 + n1}-{1 + n1 + n2}}}')
    write_table(os.path.join(out, 'variance.tex'), header, rows, 'l' + 'r' * len(cols), group=group)
    nums['variance'] = {k: {f: round(v, 4) for f, v in s.items()} for k, s in shares.items()}
    nums['variance_log'] = {k: {f: round(v, 4) for f, v in s.items()} for k, s in log_shares.items()}
    nums['variance_rows'] = {k: int(len(g)) for k, g in sel.items()}

    # T5: paired effect of each level against native, resampled cases only
    rng = np.random.default_rng(2026)
    rows, paired = [], {}
    for src in sources:
        for cfg, label in CONFIGS:
            sub = mesh[(mesh['source'] == src) & (mesh['cfg'] == cfg)]
            base = sub[sub['resolution'] == 'native'].set_index('case')['assd_mm']
            for r in order[1:]:
                lvl = sub[(sub['resolution'] == r) & sub['resampled']].set_index('case')['assd_mm']
                both = pd.concat([base, lvl], axis=1, keys=['n', 'r']).dropna()
                if len(both) < 2:
                    continue
                d = both['r'] - both['n']
                lo, hi = boot_ci(d, rng)
                p = wilcoxon(both['r'], both['n']).pvalue if (d != 0).any() else np.nan
                # the canonical expert mask at native is the reference itself: no ratio
                ratio = (both['r'] / both['n']).median() if (both['n'] > 1e-9).all() else np.nan
                rows.append([SOURCES[src], label, code(r), len(both), f"{both['n'].mean():.3f}",
                             f"{both['r'].mean():.3f}", f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
                             '--' if pd.isna(ratio) else f'{ratio:.2f}$\\times$', pval(p)])
                paired[f'{src}_{cfg}_{r}'] = {'n': int(len(both)), 'native': float(both['n'].mean()),
                                              'level': float(both['r'].mean()), 'diff': float(d.mean()),
                                              'ci': [lo, hi], 'ratio_median': None if pd.isna(ratio) else float(ratio),
                                              'p': None if pd.isna(p) else float(p)}
        rows.append(None)
    write_table(os.path.join(out, 'paired.tex'),
                ['Source', 'Config', 'Level', '$n$', 'At native', 'At level', 'Difference [95\\% CI]',
                 'Median ratio', 'Wilcoxon $p$'], rows[:-1], 'lllrrrrrr')
    nums['paired'] = paired

    # T6: Taubin by resolution (no blur, no decimation)
    t = grid_rows(mesh)
    t = t[(t['blur_mm'] == 0) & (t['decimation'] == 0)]
    iters = sorted(t['taubin_iter'].dropna().unique())
    rows = []
    for src in sources:
        for r in order:
            g = t[(t['source'] == src) & (t['resolution'] == r)]
            if not len(g):
                continue
            means = g.groupby('taubin_iter')['assd_mm'].mean()
            rows.append([SOURCES[src], code(r)] + [f'{means.get(i, np.nan):.3f}' for i in iters])
            nums[f'taubin_{src}_{r}'] = {int(i): round(float(means.get(i, np.nan)), 4) for i in iters}
        rows.append(None)
    write_table(os.path.join(out, 'taubin.tex'), ['Source', 'Level'] + [f'{int(i)} it.' for i in iters],
                rows[:-1], 'll' + 'r' * len(iters))

    # T7: blur by resolution (no smoothing, no decimation)
    b = grid_rows(mesh)
    b = b[(b['taubin_iter'] == 0) & (b['decimation'] == 0)]
    blurs = sorted(b['blur_mm'].dropna().unique())
    rows = []
    for src in sources:
        for r in order:
            g = b[(b['source'] == src) & (b['resolution'] == r)]
            if not len(g):
                continue
            means = g.groupby('blur_mm')['assd_mm'].mean()
            rows.append([SOURCES[src], code(r)] + [f'{means.get(x, np.nan):.3f}' for x in blurs])
            nums[f'blur_{src}_{r}'] = {str(x): round(float(means.get(x, np.nan)), 4) for x in blurs}
        rows.append(None)
    write_table(os.path.join(out, 'blur.tex'), ['Source', 'Level'] + [f'{x:g}\\,mm' for x in blurs],
                rows[:-1], 'll' + 'r' * len(blurs))

    # T8: topology by decimation
    rows = []
    decs = sorted(grid_rows(mesh)['decimation'].unique())
    for src in sources:
        for r in order:
            g = grid_rows(mesh)
            g = g[(g['source'] == src) & (g['resolution'] == r)]
            if not len(g):
                continue
            rate = g.groupby('decimation')['topology_ok'].mean()
            rows.append([SOURCES[src], code(r)] + [pct(rate.get(x, np.nan)) for x in decs])
            nums[f'topo_dec_{src}_{r}'] = {str(x): round(float(rate.get(x, np.nan)), 3) for x in decs}
        rows.append(None)
    write_table(os.path.join(out, 'topology.tex'), ['Source', 'Level'] + [f'{x:g}' for x in decs],
                rows[:-1], 'll' + 'r' * len(decs))
    nums['open_canonical'] = {f'{s}_{r}': int((~mesh[(mesh['source'] == s) & (mesh['resolution'] == r)
                                                     & (mesh['cfg'] == 'canonical')]['topology_ok']).sum())
                              for s in sources for r in order}

    # Sampling floor: canonical expert-mask ASSD against the sampling added by coarsening
    floor = mesh[(mesh['source'] == 'gt') & (mesh['cfg'] == 'canonical') & mesh['resampled']]
    if len(floor):
        x, y = floor['added_mm'].to_numpy(), floor['assd_mm'].to_numpy()
        slope = float((x @ y) / (x @ x))
        nums['floor_slope'] = round(slope, 4)
        nums['floor_r2'] = round(float(1 - ((y - slope * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)
        nums['floor_r'] = round(float(np.corrcoef(x, y)[0, 1]), 3) if np.ptp(x) > 0 and np.ptp(y) > 0 else None
        nums['floor_ratio_by_level'] = {r: round(float((g['assd_mm'] / g['added_mm']).mean()), 3)
                                        for r, g in floor.groupby('resolution', observed=True)}
        nums['added_mm_by_level'] = {r: [round(float(g['added_mm'].min()), 2), round(float(g['added_mm'].max()), 2)]
                                     for r, g in floor.groupby('resolution', observed=True)}

    # T9: reproducibility against an earlier native-only run
    if baseline is not None:
        keys = ['case', 'source', 'level', 'config', 'blur_mm', 'blur_vox', 'taubin_iter', 'decimation']
        old = baseline[(baseline['level'] == 'mesh') & (baseline['status'] == 'ok')].copy()
        new = mesh[mesh['resolution'] == 'native'].copy()
        for frame in (old, new):
            frame['topology_ok'] = _bool(frame['topology_ok'])
        m = old.merge(new, on=keys, suffixes=('_o', '_n'))
        m['d'] = (m['assd_mm_o'] - m['assd_mm_n']).abs()
        m['flip'] = m['topology_ok_o'] != m['topology_ok_n']
        rows = []
        for src in sources:
            for label, sel in (('no decimation', m['decimation'] == 0), ('decimated', m['decimation'] > 0)):
                g = m[(m['source'] == src) & sel]
                if not len(g):
                    continue
                rows.append([SOURCES[src], label, len(g), f"{g['d'].max():.1e}", f"{g['d'].median():.1e}",
                             f"{int(g['flip'].sum())} ({100 * g['flip'].mean():.1f}\\%)"])
                nums[f'repro_{src}_{label.replace(" ", "_")}'] = {
                    'rows': int(len(g)), 'max': float(g['d'].max()), 'median': float(g['d'].median()),
                    'flips': int(g['flip'].sum())}
        write_table(os.path.join(out, 'reproducibility.tex'),
                    ['Source', 'Meshes', 'Rows', 'Max $|\\Delta|$ ASSD (mm)', 'Median $|\\Delta|$ (mm)',
                     'Topology flips'], rows, 'llrrrr')
    return nums


def grid_rows(mesh):
    return mesh[mesh['config'] == 'grid']


# ── figures ──────────────────────────────────────────────────────────────

def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': RULE, 'axes.labelcolor': TEXT,
        'xtick.color': TEXT2, 'ytick.color': TEXT2, 'text.color': TEXT, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#EEF0F3', 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'savefig.bbox': 'tight', 'savefig.dpi': 200, 'pdf.fonttype': 42,
    })
    return plt


def _save(fig, out, name):
    fig.savefig(os.path.join(out, f'{name}.pdf'))
    fig.savefig(os.path.join(out, f'{name}.png'))


def _strip(ax, data, positions, colour, rng, filled=True, width=0.32):
    bp = ax.boxplot(data, positions=positions, widths=width, showfliers=False, patch_artist=True,
                    medianprops={'color': TEXT, 'linewidth': 1},
                    whiskerprops={'color': TEXT2, 'linewidth': 0.8}, capprops={'color': TEXT2, 'linewidth': 0.8})
    for box, c in zip(bp['boxes'], colour):
        box.set(facecolor=c if filled else 'white', edgecolor=c if not filled else TEXT2, linewidth=0.9,
                alpha=0.85 if filled else 1)
    for p, d, c in zip(positions, data, colour):
        ax.scatter(p + rng.uniform(-width / 3, width / 3, len(d)), d, s=4, color=c if not filled else TEXT,
                   alpha=0.5, linewidths=0, zorder=3)


def figures(df, out):
    plt = _style()
    from matplotlib.patches import Patch
    os.makedirs(out, exist_ok=True)
    mask, mesh = split(df)
    order = list(df['resolution'].cat.categories)
    sources = [s for s in SOURCES if s in set(df['source'])]
    rng = np.random.default_rng(0)
    x = np.arange(len(order))

    # F1: ASSD by resolution, canonical vs production, one panel per source
    for metric, name, label in (('assd_mm', 'assd_by_resolution', 'ASSD to reference surface (mm)'),
                                ('mean_signed_mm', 'signed_by_resolution', 'Mean signed distance (mm)')):
        fig, axes = plt.subplots(1, len(sources), figsize=(6.3, 2.5), sharey=True)
        for ax, src in zip(np.atleast_1d(axes), sources):
            for off, (cfg, _), filled in ((-0.19, CONFIGS[0], False), (0.19, CONFIGS[1], True)):
                data, pos, col = [], [], []
                for i, r in enumerate(order):
                    v = mesh[(mesh['source'] == src) & (mesh['resolution'] == r) & (mesh['cfg'] == cfg)][metric]
                    if len(v):
                        data.append(v.to_numpy())
                        pos.append(i + off)
                        col.append(COLOURS[r])
                if data:
                    _strip(ax, data, pos, col, rng, filled=filled)
            if metric == 'mean_signed_mm':
                ax.axhline(0, color=TEXT2, linewidth=0.7)
            ax.set_xticks(x, order)
            ax.set_title(SOURCES[src], loc='left')
        np.atleast_1d(axes)[0].set_ylabel(label)
        fig.legend(handles=[Patch(facecolor='white', edgecolor=TEXT2, label='Canonical (no blur, smoothing or decimation)'),
                            Patch(facecolor=TEXT2, edgecolor=TEXT2, label='Production (settings the app ships)')],
                   loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        _save(fig, out, name)
        plt.close(fig)

    # F2: Dice by resolution, expert mask resampled vs engines
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    width = 0.8 / len(sources)
    for k, src in enumerate(sources):
        data, pos, col = [], [], []
        for i, r in enumerate(order):
            v = mask[(mask['source'] == src) & (mask['resolution'] == r)]['dice']
            if len(v):
                data.append(v.to_numpy())
                pos.append(i + (k - (len(sources) - 1) / 2) * width)
                col.append(COLOURS[r])
        if data:
            _strip(ax, data, pos, col, rng, filled=(src != 'gt'), width=width * 0.8)
    ax.set_xticks(x, order)
    ax.set_ylabel('Dice against expert mask')
    ax.legend(handles=[Patch(facecolor='white', edgecolor=TEXT2, label='Expert mask, resampled (sampling only)'),
                       Patch(facecolor=TEXT2, label='TotalSegmentator fast')],
              loc='lower left', frameon=False)
    fig.tight_layout()
    _save(fig, out, 'dice_by_resolution')
    plt.close(fig)

    # F3: sampling floor, canonical expert-mask ASSD against grid spacing
    floor = mesh[(mesh['source'] == 'gt') & (mesh['cfg'] == 'canonical') & mesh['resampled']]
    if len(floor):
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for r in order[1:]:
            g = floor[floor['resolution'] == r]
            if not len(g):
                continue
            ax.scatter(g['added_mm'], g['assd_mm'], s=10, color=COLOURS[r], label=r, alpha=0.8, linewidths=0)
        xs = np.linspace(0, floor['added_mm'].max() * 1.05, 50)
        xv, yv = floor['added_mm'].to_numpy(), floor['assd_mm'].to_numpy()
        slope = (xv @ yv) / (xv @ xv)
        r2 = 1 - ((yv - slope * xv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        ax.plot(xs, slope * xs, color=TEXT, linewidth=0.9, label=f'fit: {slope:.3f} $\\Delta$ ($R^2$ {r2:.2f})')
        ax.plot(xs, 0.14 * xs, color=TEXT2, linewidth=0.9, linestyle='--', label='Tier 1 floor: 0.14 $h$')
        ax.set_xlabel('Added sampling $\\Delta$ (mm)')
        ax.set_ylabel('ASSD, expert mask (mm)')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, loc='upper left')
        fig.tight_layout()
        _save(fig, out, 'sampling_floor')
        plt.close(fig)

    # F4: Taubin iterations by resolution (no blur, no decimation)
    t = grid_rows(mesh)
    t = t[(t['blur_mm'] == 0) & (t['decimation'] == 0)]
    fig, axes = plt.subplots(1, len(sources), figsize=(6.3, 2.4))
    for ax, src in zip(np.atleast_1d(axes), sources):
        for r in order:
            g = t[(t['source'] == src) & (t['resolution'] == r)]
            if not len(g):
                continue
            s = g.groupby('taubin_iter')['assd_mm'].agg(['mean', 'sem'])
            ax.errorbar(s.index, s['mean'], yerr=1.96 * s['sem'], color=COLOURS[r], marker='o', markersize=3,
                        linewidth=1.1, capsize=2, label=r)
        ax.set_title(SOURCES[src], loc='left')
        ax.set_xlabel('Taubin iterations')
        ax.set_xticks(sorted(t['taubin_iter'].unique()))
    np.atleast_1d(axes)[0].set_ylabel('Mean ASSD (mm), 95% CI')
    np.atleast_1d(axes)[-1].legend(frameon=False, title='Level', title_fontsize=7)
    fig.tight_layout()
    _save(fig, out, 'taubin_by_resolution')
    plt.close(fig)

    # F5: variance shares of ASSD (mm), one panel per mask source
    grid = grid_rows(mesh)
    factors = ['resolution', 'blur_mm', 'taubin_iter', 'decimation', 'case']
    names = ['Resolution', 'Blur', 'Taubin', 'Decimation', 'Case (patient)', 'Interactions\n+ residual']
    fig, axes = plt.subplots(1, len(sources), figsize=(6.3, 2.3), sharey=True)
    for ax, src in zip(np.atleast_1d(axes), sources):
        sh = eta_squared(grid[grid['source'] == src], 'assd_mm', factors)
        vals = [sh[f] for f in factors] + [1 - sum(sh.values())]
        ypos = np.arange(len(vals))[::-1]
        colours = ['#FF7A18', '#98A2B3', '#98A2B3', '#98A2B3', '#4F91D9', '#D0D5DD']
        ax.barh(ypos, [100 * v for v in vals], color=colours, height=0.65)
        for yp, v in zip(ypos, vals):
            ax.text(100 * v + 1.5, yp, f'{100 * v:.1f}%', va='center', fontsize=7, color=TEXT)
        ax.set_yticks(ypos, names)
        ax.set_xlim(0, 105)
        ax.set_xlabel('Share of ASSD variance (%)')
        ax.set_title(SOURCES[src], loc='left')
        ax.grid(axis='y', visible=False)
    fig.tight_layout()
    _save(fig, out, 'variance_shares')
    plt.close(fig)

    # F6: topology by decimation and resolution
    g0 = grid_rows(mesh)
    fig, axes = plt.subplots(1, len(sources), figsize=(6.3, 2.3), sharey=True)
    for ax, src in zip(np.atleast_1d(axes), sources):
        for r in order:
            g = g0[(g0['source'] == src) & (g0['resolution'] == r)]
            if not len(g):
                continue
            rate = g.groupby('decimation')['topology_ok'].mean() * 100
            ax.plot(rate.index, rate.values, marker='o', markersize=3, color=COLOURS[r], linewidth=1.1, label=r)
        ax.set_title(SOURCES[src], loc='left')
        ax.set_xlabel('Decimation (fraction of faces removed)')
        ax.set_xticks(sorted(g0['decimation'].unique()))
    np.atleast_1d(axes)[0].set_ylabel('Closed and manifold (%)')
    np.atleast_1d(axes)[0].legend(frameon=False, title='Level', title_fontsize=7, loc='lower left')
    fig.tight_layout()
    _save(fig, out, 'topology_by_decimation')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--baseline', default=None, help='earlier native-only run for the reproducibility table')
    args = ap.parse_args()
    df, run = load(args.run_dir)
    baseline = pd.read_csv(results_file(args.baseline)) if args.baseline else None
    nums = tables(df, run, os.path.join(args.out, 'tables'), baseline)
    figures(df, os.path.join(args.out, 'figures'))
    nums['run_dir'] = os.path.relpath(args.run_dir, ROOT).replace('\\', '/')
    with open(os.path.join(args.out, 'numbers.json'), 'w', encoding='utf-8') as fh:
        json.dump(nums, fh, indent=1, default=float)
    print(f'tables, figures and numbers.json -> {args.out}')


if __name__ == '__main__':
    main()
