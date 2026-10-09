"""
Statistics behind the paper (docs/paper/medsys-paper.tex) that go beyond the
Tier 2 report: scan-clustered uncertainty for the variance shares, a
random-intercept variance-component estimate, validation of the sampling
relationship, and a controlled comparison of meshing grids.

Usage (from the repository root):
  python -m research.paper_stats --tier1 output/research/phantoms/<run> \
         --tier2 output/research/tier2/<run> --out docs/paper

Writes <out>/paper_numbers.json and <out>/figures/sampling_floor.pdf.
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
    """LaTeX rows of the paper's variance table: eta^2 [95% CI] in %, plus the
    random-intercept share of variance between scans."""
    t1 = nums['tier1_shares']['assd_mm']
    gt = nums['tier2_shares']['gt']['assd_mm']
    ts = nums['tier2_shares']['ts_fast']['assd_mm']

    def cell(d, key):
        if key not in d:
            return '--'
        v = d[key]
        return f"{100 * v['eta2']:.1f} [{100 * v['ci'][0]:.1f}, {100 * v['ci'][1]:.1f}]"

    rows = [('Spacing / resolution', 'spacing', 'resolution'), ('Shape / scan', 'shape', 'case'),
            ('Blur', 'blur_mm', 'blur_mm'), ('Taubin iterations', 'taubin_iter', 'taubin_iter'),
            ('Decimation', 'decimation', 'decimation'), ('Isosurface variant', 'isosurface', None),
            ('Interactions + residual', 'interactions_residual', 'interactions_residual')]
    lines = []
    for label, k1, k2 in rows:
        lines.append(f'{label} & {cell(t1, k1)} & {cell(gt, k2) if k2 else "--"} & {cell(ts, k2) if k2 else "--"} \\\\')
    ri = {s: nums['tier2_shares'][s]['random_intercept']['share_cluster'] for s in ('gt', 'ts_fast')}
    lines += ['\\midrule', f"Scans, random intercept$^{{a}}$ & -- & {100 * ri['gt']:.1f} & {100 * ri['ts_fast']:.1f} \\\\"]
    # The whole tabular is written here: rules inside an \input file break a tabular.
    head = ['% generated by research/paper_stats.py; do not edit by hand',
            '\\begin{tabular}{@{}lccc@{}}', '\\toprule',
            'Factor & T1 phantoms & T2 expert mask & T2 TS \\\\', '\\midrule']
    foot = ['\\bottomrule',
            '\\multicolumn{4}{@{}p{0.97\\columnwidth}@{}}{\\scriptsize $^{a}$Variance between scans in a '
            'random-intercept model on the balanced levels (\\texttt{native}, \\texttt{iso1.5}, \\texttt{iso3}): '
            'a different estimand from $\\eta^2$ (Section~V).}',
            '\\end{tabular}']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(head + lines + foot) + '\n')


# ── figure ───────────────────────────────────────────────────────────────

def floor_figure(floor, rel, path):
    plt = rep._style()
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for r in [r for r in rep.RESOLUTIONS if r != 'native']:
        g = floor[floor['resolution'] == r]
        if len(g):
            ax.scatter(g['added_mm'], g['assd_mm'], s=10, color=rep.COLOURS[r], alpha=0.8,
                       linewidths=0, label=f'{r} ($n$={g["case"].nunique()})')
    xs = np.linspace(0, floor['added_mm'].max() * 1.05, 60)
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


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--tier1', required=True)
    ap.add_argument('--tier2', required=True)
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
    df, run = rep.load(args.tier2)
    mask, mesh = rep.split(df)
    grid = mesh[mesh['config'] == 'grid'].copy()
    grid['resolution'] = grid['resolution'].astype(str)
    f2 = ['resolution'] + GRID + ['case']
    out['tier2_shares'] = {}
    for src in ('gt', 'ts_fast'):
        g = grid[grid['source'] == src].copy()
        shares = {'assd_mm': clustered_shares(g, 'assd_mm', f2, 'case', rng, args.boot)}
        gl = g[g['resolution'] != 'native'].copy() if src == 'gt' else g.copy()
        gl['log_assd'] = np.log(gl['assd_mm'])
        shares['log_assd'] = clustered_shares(gl, 'log_assd', f2, 'case', rng, args.boot)
        # random-intercept components on the balanced levels (every scan resampled)
        bal = g[g['resolution'].isin(['native', 'iso1.5', 'iso3'])]
        shares['random_intercept'] = random_intercept_components(
            bal, 'assd_mm', 'case', ['resolution'] + GRID)
        out['tier2_shares'][src] = shares

    # Sampling relationship
    floor = mesh[(mesh['source'] == 'gt') & (mesh['cfg'] == 'canonical') & mesh['resampled']].copy()
    floor['resolution'] = floor['resolution'].astype(str)
    rel = sampling_relationship(floor, rng)
    os.makedirs(os.path.join(args.out, 'figures'), exist_ok=True)
    floor_figure(floor, rel, os.path.join(args.out, 'figures', 'sampling_floor.pdf'))
    rel.pop('boot_slopes')
    out['sampling_relationship'] = rel

    # Controlled grid comparison: same physical blur, smoothing and decimation
    mesh = mesh.copy()
    mesh['resolution'] = mesh['resolution'].astype(str)
    out['grid_comparison'] = {}
    for src in ('ts_fast', 'gt'):
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

    with open(os.path.join(args.out, 'paper_numbers.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    os.makedirs(os.path.join(args.out, 'tables'), exist_ok=True)
    variance_table(out, os.path.join(args.out, 'tables', 'variance.tex'))
    print(f'paper_numbers.json, tables/variance.tex and figures/sampling_floor.pdf -> {args.out}')


if __name__ == '__main__':
    main()
