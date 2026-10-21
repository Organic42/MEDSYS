"""
Statistics behind the paper (docs/paper/medsys-paper.tex) that go beyond the
Tier 2 report: scan-clustered uncertainty for the variance shares, a
random-intercept variance-component estimate, validation of the sampling
relationship, and a controlled comparison of meshing grids.

Usage (from the repository root):
  python -m research.paper_stats --tier1 output/research/phantoms/<run> \
         --tier2 output/research/tier2/<spleen run> \
         [--pancreas output/research/tier2/<pancreas run>] --out docs/paper

With --pancreas, the held-out pancreas run is analysed the same way and
the hypotheses fixed in research/heldout_hypotheses.md are evaluated.

Writes <out>/paper_numbers.json, <out>/tables/variance.tex,
<out>/tables/tier2.tex and <out>/figures/sampling_floor.pdf.
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

from research.tier2 import eta_squared, results_file  # noqa: E402
from research import tier2_report as rep  # noqa: E402

GRID = ['blur_mm', 'taubin_iter', 'decimation']


# ── variance shares with clustered bootstrap ─────────────────────────────

def clustered_shares(df, y, factors, cluster, rng, n_boot=1000, strata=None):
    """eta^2 per factor with a bootstrap over clusters (scans or placements).

    Resampled clusters are relabelled so a cluster drawn twice counts as two.
    With strata, clusters are resampled within each stratum.
    """
    point = eta_squared(df, y, factors)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    if strata is not None:
        by_stratum = {}
        for k in keys:
            by_stratum.setdefault(groups[k][strata].iloc[0], []).append(k)
    draws = []
    for _ in range(n_boot):
        if strata is None:
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            chosen = [keys[i] for i in pick]
        else:
            chosen = [ks[i] for ks in by_stratum.values()
                      for i in rng.choice(len(ks), size=len(ks), replace=True)]
        parts = []
        for j, k in enumerate(chosen):
            g = groups[k].copy()
            g[cluster] = f'{k}#{j}'
            parts.append(g)
        draws.append(eta_squared(pd.concat(parts, ignore_index=True), y, factors))
    out = {}
    for f in factors:
        v = np.array([d[f] for d in draws])
        out[f] = {'eta2': point[f], 'ci': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    rest = np.array([1 - sum(d.values()) for d in draws])
    out['interactions_residual'] = {'eta2': 1 - sum(point.values()),
                                    'ci': [float(np.percentile(rest, 2.5)), float(np.percentile(rest, 97.5))]}
    return out


def random_intercept_components(df, y, cluster, cells):
    """Variance components of a balanced cluster x cell layout with one
    observation per combination: y = cell mean + cluster intercept + error.

    With balanced data, the ANOVA estimators equal REML. Returns the share of
    total variance due to the fixed cells, the random cluster intercepts and
    the residual (cluster x cell interaction).
    """
    tab = df.pivot_table(index=cluster, columns=cells, values=y, aggfunc='mean')
    tab = tab.dropna(axis=1, how='any').dropna(axis=0, how='any')
    a, b = tab.shape                                  # clusters, cells
    grand = tab.values.mean()
    row = tab.values.mean(axis=1, keepdims=True)
    col = tab.values.mean(axis=0, keepdims=True)
    ms_row = b * ((row - grand) ** 2).sum() / (a - 1)
    ms_col = a * ((col - grand) ** 2).sum() / (b - 1)
    ms_res = ((tab.values - row - col + grand) ** 2).sum() / ((a - 1) * (b - 1))
    var_cluster = max(0.0, (ms_row - ms_res) / b)
    var_cells = max(0.0, (ms_col - ms_res) / a)
    total = var_cluster + var_cells + ms_res
    return {'clusters': int(a), 'cells': int(b), 'share_cluster': var_cluster / total,
            'share_cells': var_cells / total, 'share_residual': ms_res / total,
            'icc_conditional': var_cluster / (var_cluster + ms_res)}


# ── sampling relationship ────────────────────────────────────────────────

def slope0(x, y):
    return float(x @ y / (x @ x))


def sampling_relationship(floor, rng, n_boot=2000):
    x, y = floor['added_mm'].to_numpy(), floor['assd_mm'].to_numpy()
    b = slope0(x, y)
    pred = b * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    scans = floor['case'].unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(scans, size=len(scans), replace=True)
        s = pd.concat([floor[floor['case'] == c] for c in pick])
        boots.append(slope0(s['added_mm'].to_numpy(), s['assd_mm'].to_numpy()))
    a1, b1 = np.polyfit(x, y, 1)[::-1]                # free intercept: y = a1 + b1 x
    # leave one scan out: fit on the other scans, predict the held-out scan
    loso_pred = np.empty_like(y)
    for c in scans:
        m = (floor['case'] == c).to_numpy()
        loso_pred[m] = slope0(x[~m], y[~m]) * x[m]
    loso_err = y - loso_pred
    loso_r2 = 1 - (loso_err ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # leave one level out: fit on two levels, predict the third
    lolo = {}
    for lvl in floor['resolution'].unique():
        m = (floor['resolution'] == lvl).to_numpy()
        bb = slope0(x[~m], y[~m])
        e = y[m] - bb * x[m]
        lolo[str(lvl)] = {'slope_without': bb, 'mae': float(np.abs(e).mean()), 'mean_error': float(e.mean())}
    resid = y - pred
    by_level = {str(l): {'mean_residual': float(resid[(floor['resolution'] == l).to_numpy()].mean()),
                         'n': int((floor['resolution'] == l).sum())}
                for l in floor['resolution'].unique()}
    return {'n_points': int(len(y)), 'n_scans': int(len(scans)), 'slope': b,
            'slope_ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'r2': float(r2), 'rmse': float(np.sqrt((resid ** 2).mean())),
            'free_intercept': {'intercept': float(a1), 'slope': float(b1)},
            'loso': {'mae': float(np.abs(loso_err).mean()), 'rmse': float(np.sqrt((loso_err ** 2).mean())),
                     'r2': float(loso_r2)},
            'leave_one_level_out': lolo, 'residual_by_level': by_level,
            'boot_slopes': boots}


# ── controlled grid comparison ───────────────────────────────────────────

def grid_comparison(mesh, source, a='native', b='iso3'):
    """Paired change from grid a to grid b at identical physical settings
    (blur in mm, Taubin iterations, decimation): one value per configuration."""
    g = mesh[(mesh['source'] == source) & (mesh['config'] == 'grid')]
    keys = ['case'] + GRID
    left = g[g['resolution'] == a].set_index(keys)
    right = g[g['resolution'] == b].set_index(keys)
    both = left[['assd_mm', 'topology_ok', 'n_faces']].join(
        right[['assd_mm', 'topology_ok', 'n_faces']], lsuffix='_a', rsuffix='_b', how='inner')
    both['d_assd'] = both['assd_mm_b'] - both['assd_mm_a']
    per_cfg = both.groupby(level=GRID).agg(
        d_assd=('d_assd', 'mean'), closed_a=('topology_ok_a', 'mean'), closed_b=('topology_ok_b', 'mean'),
        faces_a=('n_faces_a', 'median'), faces_b=('n_faces_b', 'median'))
    return both, per_cfg


def grid_comparison_stats(both, rng, n_boot=5000):
    """Scan-level summary of a paired grid comparison.

    Each scan's 48 configuration-level differences are averaged first, so the
    scan is the unit; the mean over scans gets a bootstrap CI over scans and a
    Wilcoxon signed-rank test. Closure is compared the same way at decimation
    0.75 (share of a scan's 12 configurations that are closed).
    """
    from scipy.stats import wilcoxon

    def summary(per_scan):
        v = per_scan.to_numpy()
        boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
        return {'mean': float(v.mean()), 'ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                'p': float(wilcoxon(v).pvalue) if np.any(v != 0) else None, 'scans': int(len(v)),
                'scans_improved': int((v < 0).sum())}

    d_assd = both.groupby(level='case')['d_assd'].mean()
    heavy = both[both.index.get_level_values('decimation') == 0.75]
    closed = heavy.groupby(level='case')[['topology_ok_a', 'topology_ok_b']].mean()
    out = {'d_assd_per_scan': summary(d_assd),
           'closure_dec075': {'native': float(closed['topology_ok_a'].mean()),
                              'iso3': float(closed['topology_ok_b'].mean()),
                              'difference': summary(-(closed['topology_ok_b'] - closed['topology_ok_a']))}}
    # closure gain is reported positive: flip the sign used by summary()
    diff = out['closure_dec075']['difference']
    diff['mean'], diff['ci'] = -diff['mean'], [-diff['ci'][1], -diff['ci'][0]]
    diff['scans_improved'] = int(((closed['topology_ok_b'] - closed['topology_ok_a']) > 0).sum())
    return out


def variance_table(nums, path):
    """The paper's variance table: eta^2 [95% CI] in %, per tier, organ and
    mask source, plus the random-intercept share of variance between scans."""
    cols = [('T1 phantoms', nums['tier1_shares']['assd_mm'], None)]
    for organ, d in nums['tier2'].items():
        for src, label in (('gt', 'expert'), ('ts_fast', 'TS fast')):
            if src in d['shares']:
                cols.append((f'{organ.capitalize()} {label}', d['shares'][src]['assd_mm'],
                             d['shares'][src]['random_intercept']['share_cluster']))

    def cell(d, key):
        if key is None or key not in d:
            return '--'
        v = d[key]
        return f"{100 * v['eta2']:.1f} [{100 * v['ci'][0]:.1f}, {100 * v['ci'][1]:.1f}]"

    rows = [('Spacing / resolution', 'spacing', 'resolution'), ('Shape / scan', 'shape', 'case'),
            ('Blur', 'blur_mm', 'blur_mm'), ('Taubin iterations', 'taubin_iter', 'taubin_iter'),
            ('Decimation', 'decimation', 'decimation'), ('Isosurface variant', 'isosurface', None),
            ('Interactions + residual', 'interactions_residual', 'interactions_residual')]
    lines = []
    for label, k1, k2 in rows:
        lines.append(label + ''.join(f' & {cell(d, k1 if i == 0 else k2)}' for i, (_, d, _) in enumerate(cols))
                     + r' \\')
    lines += [r'\midrule', 'Scans, random intercept$^{a}$' + ''.join(
        ' & --' if ri is None else f' & {100 * ri:.1f}' for _, _, ri in cols) + r' \\']
    # The whole tabular is written here: rules inside an \input file break a tabular.
    # Footnote a (the random-intercept estimand) belongs in the table caption.
    head = ['% generated by research/paper_stats.py; do not edit by hand',
            r'\begin{tabular}{@{}l' + 'c' * len(cols) + '@{}}', r'\toprule',
            'Factor' + ''.join(f' & {name}' for name, _, _ in cols) + r' \\', r'\midrule']
    foot = [r'\bottomrule', r'\end{tabular}']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(head + lines + foot) + '\n')


# ── figure ───────────────────────────────────────────────────────────────

def floor_figure(floor, rel, path, held_out=None):
    """Expert-mask sampling error against added sampling; held_out adds the
    pancreas points (hollow), which the spleen fit did not see."""
    plt = rep._style()
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for r in [r for r in rep.RESOLUTIONS if r != 'native']:
        g = floor[floor['resolution'] == r]
        if len(g):
            ax.scatter(g['added_mm'], g['assd_mm'], s=10, color=rep.colour(r), alpha=0.8,
                       linewidths=0, label=f'{r} ($n$={g["case"].nunique()})')
    if held_out is not None and len(held_out):
        ax.scatter(held_out['added_mm'], held_out['assd_mm'], s=9, facecolors='none', edgecolors=rep.TEXT2,
                   linewidths=0.6, alpha=0.8, label=f'pancreas, held out ($n$={held_out["case"].nunique()})')
    top = max(floor['added_mm'].max(), held_out['added_mm'].max() if held_out is not None else 0)
    xs = np.linspace(0, top * 1.05, 60)
    lo, hi = rel['slope_ci']
    ax.fill_between(xs, lo * xs, hi * xs, color=rep.TEXT, alpha=0.12, linewidth=0, label='95% CI (scan bootstrap)')
    ax.plot(xs, rel['slope'] * xs, color=rep.TEXT, linewidth=1.0,
            label=f'ASSD = {rel["slope"]:.3f} $\\Delta$ ($R^2$ {rel["r2"]:.2f})')
    ax.plot(xs, 0.14 * xs, color=rep.TEXT2, linewidth=0.9, linestyle='--', label='Tier 1 floor, 0.14 $h$')
    ax.set_xlabel('Added sampling $\\Delta$ (mm)')
    ax.set_ylabel('ASSD, expert mask (mm)')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc='upper left', fontsize=6.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── per-dataset statistics ───────────────────────────────────────────────

def tier2_stats(run_dir, rng, n_boot):
    """Variance shares, sampling relationship, grid comparison and level
    effects of one Tier 2 run. Returns (stats, mesh rows, mask rows)."""
    df, run = rep.load(run_dir)
    mask, mesh = rep.split(df)
    mesh = mesh.copy()
    mesh['resolution'] = mesh['resolution'].astype(str)
    sources = [s for s in ('gt', 'ts_fast', 'ts_full') if s in set(mesh['source'])]
    grid = mesh[mesh['config'] == 'grid'].copy()
    f2 = ['resolution'] + GRID + ['case']
    out = {'run': os.path.relpath(run_dir, ROOT).replace('\\', '/'), 'scans': int(df['case'].nunique()),
           'sources': sources, 'shares': {}, 'grid_comparison': {}}
    for src in sources:
        g = grid[grid['source'] == src].copy()
        shares = {'assd_mm': clustered_shares(g, 'assd_mm', f2, 'case', rng, n_boot)}
        gl = g[g['resolution'] != 'native'].copy() if src == 'gt' else g.copy()
        gl['log_assd'] = np.log(gl['assd_mm'])
        shares['log_assd'] = clustered_shares(gl, 'log_assd', f2, 'case', rng, n_boot)
        # random-intercept components on the levels that resample every scan
        balanced = [r for r in ('native', 'iso1.5', 'iso3')
                    if g[g['resolution'] == r]['case'].nunique() == out['scans']]
        shares['random_intercept'] = random_intercept_components(
            g[g['resolution'].isin(balanced)], 'assd_mm', 'case', ['resolution'] + GRID)
        shares['random_intercept']['levels'] = balanced
        out['shares'][src] = shares
    floor = mesh[(mesh['source'] == 'gt') & (mesh['cfg'] == 'canonical') & mesh['resampled']].copy()
    out['floor'] = floor
    out['sampling_relationship'] = sampling_relationship(floor, rng)
    for src in sources:
        both, per_cfg = grid_comparison(mesh, src)
        dec = per_cfg.groupby(level='decimation').agg(
            d_assd=('d_assd', 'mean'), closed_native=('closed_a', 'mean'), closed_iso3=('closed_b', 'mean'))
        smooth = per_cfg[per_cfg.index.get_level_values('taubin_iter') == 30]
        out['grid_comparison'][src] = {
            'configs': int(len(per_cfg)),
            'mean_d_assd_all_configs': float(per_cfg['d_assd'].mean()),
            'share_configs_iso3_better': float((per_cfg['d_assd'] < 0).mean()),
            'by_decimation': {str(k): {c: float(v) for c, v in r.items()} for k, r in dec.iterrows()},
            'taubin30_by_blur_dec': {f'blur{k[0]:g}_dec{k[2]:g}': {'d_assd': float(r['d_assd']),
                                                                  'closed_native': float(r['closed_a']),
                                                                  'closed_iso3': float(r['closed_b'])}
                                     for k, r in smooth.iterrows()},
            'faces_median_native': float(both['n_faces_a'].median()),
            'faces_median_iso3': float(both['n_faces_b'].median()),
            'scan_level': grid_comparison_stats(both, rng),
        }
    out['paired'] = paired_changes(mesh, sources, rng)
    return out, mesh, mask


def paired_changes(mesh, sources, rng, n_boot=5000):
    """Canonical ASSD change of each level against native, paired by scan."""
    from scipy.stats import wilcoxon
    out = {}
    for src in sources:
        sub = mesh[(mesh['source'] == src) & (mesh['cfg'] == 'canonical')]
        base = sub[sub['resolution'] == 'native'].set_index('case')['assd_mm']
        for r in ('iso1.5', 'iso3', 'slice5'):
            lvl = sub[(sub['resolution'] == r) & sub['resampled']].set_index('case')['assd_mm']
            both = pd.concat([base, lvl], axis=1, keys=['n', 'r']).dropna()
            if len(both) < 2:
                continue
            d = (both['r'] - both['n']).to_numpy()
            boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
            out[f'{src}_{r}'] = {'n': int(len(d)), 'native': float(both['n'].mean()),
                                 'level': float(both['r'].mean()), 'diff': float(d.mean()),
                                 'ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                 'p': float(wilcoxon(d).pvalue) if np.any(d != 0) else None}
    return out


def results_table(runs, path):
    """The paper's Tier 2 table: one row per source and level, one column
    block per organ (n, Dice, canonical ASSD, its paired change against
    native with the scan-bootstrap CI of paired_changes, share of closed
    production meshes). runs maps organ -> (mesh rows, mask rows, paired)."""
    names = {'gt': 'Expert mask', 'ts_fast': 'TS fast', 'ts_full': 'TS full'}
    blocks = []
    for organ, (mesh, mask, paired) in runs.items():
        cells = {}
        for src in names:
            can = mesh[(mesh['source'] == src) & (mesh['cfg'] == 'canonical')]
            prod = mesh[(mesh['source'] == src) & (mesh['cfg'] == 'production')]
            for r in ('native', 'iso1.5', 'iso3', 'slice5'):
                c = can[can['resolution'] == r]
                if not len(c):
                    continue
                dice = mask[(mask['source'] == src) & (mask['resolution'] == r)]['dice']
                p = paired.get(f'{src}_{r}')
                change = '--' if p is None else (f"${p['diff']:+.3f}$ [${p['ci'][0]:+.3f}$, "
                                                 f"${p['ci'][1]:+.3f}$]")
                closed = prod[prod['resolution'] == r]['topology_ok'].mean()
                cells[(src, r)] = [str(c['case'].nunique()),
                                   '--' if dice.empty else f'{dice.mean():.3f}',
                                   f"{c['assd_mm'].mean():.3f}", change, f'{100 * closed:.0f}']
        blocks.append(cells)
    lines = ['% generated by research/paper_stats.py; do not edit by hand',
             '\\begin{tabular}{@{}ll' + 'rrrcr' * len(runs) + '@{}}', '\\toprule',
             ' & & ' + ' & '.join(f'\\multicolumn{{5}}{{c}}{{{o}}}' for o in runs) + ' \\\\',
             ''.join(f'\\cmidrule(lr){{{3 + 5 * i}-{7 + 5 * i}}}' for i in range(len(runs))),
             'Source & Level' + ' & $n$ & Dice & ASSD & Change vs native & Closed' * len(runs) + ' \\\\',
             '\\midrule']
    for src, label in names.items():
        if not any((src, r) in b for b in blocks for r in ('native',)):
            continue
        for i, r in enumerate(('native', 'iso1.5', 'iso3', 'slice5')):
            row = [label if i == 0 else '', f'\\lvl{{{r}}}']
            for b in blocks:
                row += b.get((src, r), ['--'] * 5)
            lines.append(' & '.join(row) + ' \\\\')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


# ── held-out confirmation (research/heldout_hypotheses.md) ───────────────

def heldout(spleen, pancreas):
    """Evaluate H1-H6, fixed before the pancreas data existed, on the pancreas run."""
    rel_s, floor_p = spleen['sampling_relationship'], pancreas['floor']
    err = floor_p['assd_mm'].to_numpy() - rel_s['slope'] * floor_p['added_mm'].to_numpy()
    rel_p = pancreas['sampling_relationship']
    h1_mae = float(np.abs(err).mean())

    def share(d, src, f):
        return d['shares'][src]['assd_mm'][f]['eta2']

    recon = ('blur_mm', 'taubin_iter', 'decimation')
    gt_res = share(pancreas, 'gt', 'resolution')
    h2 = gt_res > share(pancreas, 'gt', 'case') and all(gt_res > share(pancreas, 'gt', f) for f in recon)
    ts_case, ts_res = share(pancreas, 'ts_fast', 'case'), share(pancreas, 'ts_fast', 'resolution')
    ri = pancreas['shares']['ts_fast']['random_intercept']['share_cluster']
    p4 = pancreas['paired'].get('ts_fast_iso3')
    p5 = pancreas['paired'].get('ts_fast_slice5')
    g6 = pancreas['grid_comparison']['ts_fast']['scan_level']

    def verdict(*parts):
        return 'confirmed' if all(parts) else ('partly confirmed' if any(parts) else 'not confirmed')

    return {
        'H1': {'spleen_slope': rel_s['slope'], 'pancreas_slope': rel_p['slope'],
               'pancreas_slope_ci': rel_p['slope_ci'], 'pancreas_r2': rel_p['r2'],
               'prediction_mae': h1_mae, 'prediction_mean_error': float(err.mean()),
               'criterion_mae': 0.074, 'verdict': verdict(h1_mae <= 0.074)},
        'H2': {'eta2_resolution': gt_res, 'eta2_case': share(pancreas, 'gt', 'case'),
               'eta2_recon_max': max(share(pancreas, 'gt', f) for f in recon),
               'verdict': verdict(h2)},
        'H3': {'eta2_case': ts_case, 'eta2_resolution': ts_res, 'random_intercept': ri,
               'verdict': verdict(ts_case > ts_res and ri > 0.5)},
        'H4': {**(p4 or {}), 'margin': 0.10,
               'verdict': verdict(p4 is not None and -0.10 <= p4['ci'][0] and p4['ci'][1] <= 0.10)},
        'H5': {**(p5 or {}), 'verdict': verdict(p5 is not None and p5['diff'] > 0 and p5['ci'][0] > 0)},
        'H6': {'closure_gain': g6['closure_dec075']['difference'], 'd_assd': g6['d_assd_per_scan'],
               'verdict': verdict(g6['closure_dec075']['difference']['ci'][0] > 0,
                                  -0.05 <= g6['d_assd_per_scan']['mean'] <= 0.05)},
    }


def heldout_table(nums, path):
    """The paper's held-out table: each hypothesis with its spleen value, its
    pancreas value and the verdict under the criterion fixed beforehand."""
    sp, h = nums['tier2']['spleen'], nums['heldout']

    def share(d, src, f):
        return 100 * d['shares'][src]['assd_mm'][f]['eta2']

    def change(p):
        return f"${p['diff']:+.3f}$ [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}]".replace('[-', '[$-$').replace(', -', ', $-$')

    def grid6(g):
        c, a = g['closure_dec075']['difference'], g['d_assd_per_scan']
        return f"${100 * c['mean']:+.0f}$ pts; ${a['mean']:+.3f}$"

    rel = sp['sampling_relationship']
    rows = [
        ('H1', 'Spleen $0.177\\Delta$ fit predicts', f"MAE {rel['loso']['mae']:.3f}$^b$",
         f"MAE {h['H1']['prediction_mae']:.3f}"),
        ('H2', 'Spacing leads (expert)',
         f"{share(sp, 'gt', 'resolution'):.1f} vs {share(sp, 'gt', 'case'):.1f}",
         f"{100 * h['H2']['eta2_resolution']:.1f} vs {100 * h['H2']['eta2_case']:.1f}"),
        ('H3', 'Scans lead (engine)',
         f"{share(sp, 'ts_fast', 'case'):.1f}; RI {100 * sp['shares']['ts_fast']['random_intercept']['share_cluster']:.1f}",
         f"{100 * h['H3']['eta2_case']:.1f}; RI {100 * h['H3']['random_intercept']:.1f}"),
        ('H4', '\\lvl{iso3} leaves engine', change(sp['paired']['ts_fast_iso3']),
         change(h['H4']) if 'diff' in h['H4'] else '--'),
        ('H5', '\\lvl{slice5} raises engine', change(sp['paired']['ts_fast_slice5']),
         change(h['H5']) if 'diff' in h['H5'] else '--'),
        ('H6', '3\\mm grid: closure up, ASSD same', grid6(sp['grid_comparison']['ts_fast']['scan_level']),
         grid6(nums['tier2']['pancreas']['grid_comparison']['ts_fast']['scan_level'])),
    ]
    short = {'confirmed': 'yes', 'partly confirmed': 'partly', 'not confirmed': 'no'}
    lines = ['% generated by research/paper_stats.py; do not edit by hand',
             '\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{1.6cm}llc@{}}', '\\toprule',
             ' & Hypothesis$^a$ & Spleen & Pancreas & Held \\\\', '\\midrule']
    for key, text, s_val, p_val in rows:
        lines.append(f'{key} & {text} & {s_val} & {p_val} & {short[h[key]["verdict"]]} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--tier1', required=True)
    ap.add_argument('--tier2', required=True, help='spleen run')
    ap.add_argument('--pancreas', default=None, help='held-out pancreas run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {'seed': args.seed, 'bootstrap_replicates': args.boot}

    # Tier 1: placements are the replicate clusters, resampled within shape
    t1 = pd.read_csv(results_file(args.tier1))
    t1 = t1[(t1['config'] == 'grid') & (t1['status'] == 'ok')].copy()
    t1['placement_id'] = t1['shape'] + '/' + t1['placement'].astype(str)
    t1['log_assd'] = np.log(t1['assd_mm'])
    f1 = ['spacing', 'shape', 'blur_mm', 'isosurface', 'taubin_iter', 'decimation']
    out['tier1_shares'] = {y: clustered_shares(t1, y, f1, 'placement_id', rng, args.boot, strata='shape')
                           for y in ('assd_mm', 'log_assd')}

    # Tier 2: scans are the clusters
    spleen, s_mesh, s_mask = tier2_stats(args.tier2, rng, args.boot)
    tables = {'Spleen': (s_mesh, s_mask, spleen['paired'])}
    os.makedirs(os.path.join(args.out, 'figures'), exist_ok=True)
    datasets = {'spleen': spleen}
    if args.pancreas:
        datasets['pancreas'], p_mesh, p_mask = tier2_stats(args.pancreas, rng, args.boot)
        tables['Pancreas'] = (p_mesh, p_mask, datasets['pancreas']['paired'])
        out['heldout'] = heldout(spleen, datasets['pancreas'])
    floor_figure(spleen['floor'], spleen['sampling_relationship'],
                 os.path.join(args.out, 'figures', 'sampling_floor.pdf'),
                 datasets['pancreas']['floor'] if 'pancreas' in datasets else None)
    for d in datasets.values():
        d.pop('floor')
        d['sampling_relationship'].pop('boot_slopes')
    out['tier2'] = datasets
    # The first version's keys, kept so its numbers stay traceable.
    out['tier2_shares'] = spleen['shares']
    out['sampling_relationship'] = spleen['sampling_relationship']
    out['grid_comparison'] = spleen['grid_comparison']

    with open(os.path.join(args.out, 'paper_numbers.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    os.makedirs(os.path.join(args.out, 'tables'), exist_ok=True)
    variance_table(out, os.path.join(args.out, 'tables', 'variance.tex'))
    results_table(tables, os.path.join(args.out, 'tables', 'tier2.tex'))
    if 'heldout' in out:
        heldout_table(out, os.path.join(args.out, 'tables', 'heldout.tex'))
    print(f'paper_numbers.json, tables/variance.tex, tables/tier2.tex and figures/sampling_floor.pdf -> {args.out}')


if __name__ == '__main__':
    main()
