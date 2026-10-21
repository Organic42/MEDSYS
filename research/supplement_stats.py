"""
Supplementary analyses for the revised paper. Each section answers a point
of the review of the first version:

  S1  design and counts                     S7  blur in voxels and in millimetres
  S2  Tier 1 floor; smoothing by organ      S8  local and task-specific metrics
  S3  per-scan distributions, failures      S9  coarsening-model sensitivity
  S4  sampling models, prediction intervals S10 engine modes (fast and full)
  S5  the slice5 subsets                    S11 scan properties behind the case effect
  S6  topology without the cut scan         S12 sensitivity across configurations

Usage (from the repository root):
  python -m research.supplement_stats --tier1 <run> --spleen <run> --pancreas <run> \
         --sensitivity <run> --spleen-data data/msd/Task09_Spleen \
         --pancreas-data data/msd/Task07_Pancreas --out docs/paper/supplement

Writes <out>/tables/*.tex, <out>/figures/*.pdf and <out>/supplement_numbers.json.
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

from research import tier2_report as rep  # noqa: E402
from research.tier2 import eta_squared, results_file  # noqa: E402

GRID = ['blur_mm', 'taubin_iter', 'decimation']
LEVELS = ('native', 'iso1.5', 'iso3', 'slice5')
SHORT = {'gt': 'Expert mask', 'ts_fast': 'TS fast', 'ts_full': 'TS full'}
ORGAN = {'msd_spleen': 'Spleen', 'msd_pancreas': 'Pancreas'}


# ── helpers ──────────────────────────────────────────────────────────────

def load(run_dir):
    df, run = rep.load(run_dir)
    mask, mesh = rep.split(df)
    mask, mesh = mask.copy(), mesh.copy()
    for frame in (df, mask, mesh):
        frame['resolution'] = frame['resolution'].astype(str)
    return {'df': df, 'mask': mask, 'mesh': mesh, 'run': run, 'organ': ORGAN.get(run['dataset'], run['dataset'])}


def code(name):
    return f'\\texttt{{{name}}}'


def ci_text(mean, ci, d=3, sign=True):
    f = f'{{:{"+" if sign else ""}.{d}f}}'
    return f'{f.format(mean)} [{f.format(ci[0])}, {f.format(ci[1])}]'


def boot_mean(x, rng, n=5000):
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired(mesh, src, cfg, level, rng, base_level='native', metric='assd_mm'):
    """Per-scan change of a metric from base_level to level (scans the level resampled)."""
    sub = mesh[(mesh['source'] == src) & (mesh['cfg'] == cfg)]
    base = sub[sub['resolution'] == base_level].set_index('case')[metric]
    lvl = sub[(sub['resolution'] == level) & sub['resampled']].set_index('case')[metric]
    both = pd.concat([base, lvl], axis=1, keys=['a', 'b']).dropna()
    return both['b'] - both['a']


def sources_of(d):
    return [s for s in ('gt', 'ts_fast', 'ts_full') if s in set(d['mesh']['source'])]


def table(path, header, rows, align, group=None, dedupe=2):
    """A booktabs table; a label in the first `dedupe` columns that repeats the
    row above (within a block between rules) is left blank."""
    out, last = [], None
    for r in rows:
        if r is None:
            out.append(None)
            last = None
            continue
        shown = list(r)
        if last is not None:
            for i in range(min(dedupe, len(r))):
                if r[i] != last[i]:
                    break
                shown[i] = ''
        out.append(shown)
        last = list(r)
    rep.write_table(path, header, out, align, group)


# ── S1 design and counts ─────────────────────────────────────────────────

def s1_counts(t1, runs, out, nums):
    rows = [['Tier 1', 'phantoms', '4 shapes $\\times$ 5 placements', 'exact surface', '--',
             str(t1['spacing'].nunique()), str(int(t1[t1['config'] == 'grid'].groupby(
                 ['shape', 'spacing', 'placement']).size().iloc[0]) + 1), '--', f'{len(t1):,}']]
    nums['counts'] = {'tier1_meshes': int(len(t1))}
    for key, d in runs.items():
        df, mask = d['df'], d['mask']
        mesh_all = df[(df['level'] == 'mesh') & (df['status'] == 'ok')]
        calls = {s: int(len(mask[mask['source'] == s][['case', 'resolution']].drop_duplicates()))
                 for s in sources_of(d) if s != 'gt'}
        configs = int(mesh_all.groupby(['case', 'resolution', 'source']).size().max())
        levels = sorted(set(df['resolution']), key=lambda r: rep.level_index(r))
        rows.append(['Tier 2' if key != 'sensitivity' else 'Sensitivity', d['organ'],
                     f"{df['case'].nunique()} scans", 'expert mask',
                     ', '.join(SHORT[s] for s in sources_of(d)), str(len(levels)), str(configs),
                     ', '.join(f'{v}' for v in calls.values()), f'{len(mesh_all):,}'])
        nums['counts'][key] = {'scans': int(df['case'].nunique()), 'levels': levels, 'configs_per_mask': configs,
                               'engine_calls': calls, 'meshes': int(len(mesh_all)),
                               'mesh_rows_in_level_effects': int(len(d['mesh']))}
    table(os.path.join(out, 'counts.tex'),
          ['Tier', 'Data', 'Units', 'Reference', 'Segmentation', 'Levels', 'Configs', 'Engine calls', 'Meshes'],
          rows, 'lllllrrrr')


# ── S2 Tier 1 floor and smoothing by organ ───────────────────────────────

def s2_floor_and_smoothing(t1, runs, out, nums):
    can = t1[(t1['config'] == 'grid') & (t1['blur_mm'] == 0) & (t1['isosurface'] == 'lewiner')
             & (t1['taubin_iter'] == 0) & (t1['decimation'] == 0)]
    spacings = ['0.5', '1', '2', '0.8x0.8x5']
    rows, floor = [], {}
    for shape in ('sphere', 'capsule', 'torus', 'rounded_box'):
        g = can[can['shape'] == shape].groupby('spacing')[['assd_mm', 'hd95_mm']].mean()
        rows.append([shape.replace('_', ' ').capitalize()] +
                    [f"{g.loc[s, 'assd_mm']:.3f} ({g.loc[s, 'hd95_mm']:.2f})" for s in spacings])
        floor[shape] = {s: float(g.loc[s, 'assd_mm']) for s in spacings}
    table(os.path.join(out, 'tier1_floor.tex'), ['Phantom', '0.5\\,mm', '1\\,mm', '2\\,mm', '$0.8\\times0.8\\times5$\\,mm'],
          rows, 'lcccc')
    nums['tier1_floor'] = floor

    # Taubin iterations without blur or decimation: phantoms and both organs
    rows, smooth = [], {}
    t = t1[(t1['config'] == 'grid') & (t1['blur_mm'] == 0) & (t1['isosurface'] == 'lewiner') & (t1['decimation'] == 0)]
    for shape in ('sphere', 'capsule', 'torus', 'rounded_box'):
        for sp in ('1', '2'):
            m = t[(t['shape'] == shape) & (t['spacing'] == sp)].groupby('taubin_iter')['assd_mm'].mean()
            rows.append([f'Phantom: {shape.replace("_", " ")}', f'{sp}\\,mm (exact)'] + [f'{m.get(i, np.nan):.3f}' for i in (0, 10, 30, 60)])
            smooth[f'{shape}_{sp}'] = {int(i): float(v) for i, v in m.items()}
    rows.append(None)
    for key, d in runs.items():
        if key == 'sensitivity':
            continue
        g = d['mesh'][(d['mesh']['config'] == 'grid') & (d['mesh']['blur_mm'] == 0) & (d['mesh']['decimation'] == 0)]
        for src in sources_of(d):
            for r in ('native', 'iso3'):
                m = g[(g['source'] == src) & (g['resolution'] == r)].groupby('taubin_iter')['assd_mm'].mean()
                rows.append([f'{d["organ"]}: {SHORT[src]}', code(r)] + [f'{m.get(i, np.nan):.3f}' for i in (0, 10, 30, 60)])
                smooth[f'{key}_{src}_{r}'] = {int(i): float(v) for i, v in m.items()}
    table(os.path.join(out, 'smoothing.tex'), ['Surface', 'Grid', '0 it.', '10 it.', '30 it.', '60 it.'], rows, 'llrrrr')
    nums['smoothing'] = smooth


# ── S3 per-scan distributions and failure cases ──────────────────────────

def fov_cut_cases(data_dir, dataset, cases):
    """Scans whose expert organ mask touches the edge of the image."""
    import nibabel as nib
    from research.tier2 import DATASETS, list_cases
    labels = DATASETS[dataset]['structures']
    values = next(iter(labels.values()))
    cut = {}
    for cid, _, _, lbl in list_cases(dataset, data_dir):
        if cid not in cases:
            continue
        m = np.isin(np.asanyarray(nib.load(lbl).dataobj), values)
        cut[cid] = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
                        or m[:, :, 0].any() or m[:, :, -1].any())
    return cut


def s3_per_scan(runs, out, nums, figs):
    plt = rep._style()
    keys = [k for k in runs if k != 'sensitivity']
    srcs = ['gt', 'ts_fast', 'ts_full']
    fig, axes = plt.subplots(len(keys), 3, figsize=(7.0, 2.3 * len(keys)), squeeze=False)
    rows = []
    nums['per_scan'] = {}
    for i, key in enumerate(keys):
        d = runs[key]
        can = d['mesh'][d['mesh']['cfg'] == 'canonical']
        for j, src in enumerate(srcs):
            ax = axes[i][j]
            g = can[can['source'] == src]
            if not len(g):
                ax.set_visible(False)
                continue
            piv = g.pivot_table(index='case', columns='resolution', values='assd_mm')
            piv = piv[[r for r in LEVELS if r in piv.columns]]
            x = np.arange(piv.shape[1])
            # engines: cap the axis just above the bulk of the scans and mark the few above it
            values = piv.to_numpy()[~np.isnan(piv.to_numpy())]
            top = (float(values.max()) * 1.08 if src == 'gt'
                   else max(3.0, float(np.ceil(np.percentile(values, 97.5) * 1.15))))
            worst = list(piv.max(axis=1).nlargest(3).index) if src != 'gt' else []
            off_scale = {}
            for case, v in piv.iterrows():
                hot = case in worst
                ax.plot(x, np.minimum(v.values, top), color='#FF7A18' if hot else rep.TEXT2,
                        alpha=0.9 if hot else 0.35, linewidth=1.0 if hot else 0.6, marker='o', markersize=1.8)
                last = int(np.flatnonzero(~np.isnan(v.values))[-1])
                for xi, val in zip(x, v.values):
                    if val > top and not (hot and xi == last):       # highlighted ends carry their value
                        k = off_scale.get(xi, 0)
                        off_scale[xi] = k + 1
                        ax.annotate(f'{val:.1f}', (xi, top), xytext=(4, -2 - 8 * k), textcoords='offset points',
                                    fontsize=6, color='#C2410C', va='top')
            # label each highlighted scan where its line ends, spaced so labels do not overlap
            ends = []
            for c in worst:
                vals = piv.loc[c].to_numpy()
                last = int(np.flatnonzero(~np.isnan(vals))[-1])
                ends.append((min(float(vals[last]), top), last, c, float(vals[last])))
            placed = []
            for yv, xi, c, val in sorted(ends, reverse=True):
                for py, px in placed:
                    if px == xi and py - yv < 0.07 * top:
                        yv = py - 0.07 * top
                text = c.split('_')[-1] + (f' ({val:.1f})' if val > top else '')
                ax.annotate(text, (x[xi], yv), xytext=(4, 0), textcoords='offset points',
                            fontsize=6, color='#C2410C', va='center', annotation_clip=False)
                placed.append((yv, xi))
            ax.set_ylim(0, top)
            ax.set_xticks(x, list(piv.columns))
            ax.set_title(f'{d["organ"]}: {SHORT[src]}', loc='left')
            if j == 0:
                ax.set_ylabel('Canonical ASSD (mm)')
            stats = {r: {'median': float(piv[r].median()), 'q1': float(piv[r].quantile(0.25)),
                         'q3': float(piv[r].quantile(0.75)), 'max': float(piv[r].max()),
                         'worst': str(piv[r].idxmax())} for r in piv.columns}
            nums['per_scan'][f'{key}_{src}'] = stats
            for r in piv.columns:
                if src == 'gt' and r == 'native':
                    continue                       # the reference itself: zero by construction
                s = stats[r]
                rows.append([d['organ'], SHORT[src], code(r), int(piv[r].notna().sum()),
                             f"{s['median']:.3f} [{s['q1']:.3f}, {s['q3']:.3f}]",
                             f"{s['max']:.3f} ({s['worst'].split('_')[-1]})"])
        rows.append(None)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, 'per_scan.pdf'))
    plt.close(fig)
    table(os.path.join(out, 'per_scan.tex'), ['Organ', 'Source', 'Level', '$n$', 'Median [IQR] (mm)', 'Worst (scan)'],
          rows[:-1], 'lllrrr')


def s3_failures(runs, data_dirs, out, nums):
    rows = []
    nums['failures'] = {}
    for key, d in runs.items():
        if key == 'sensitivity':
            continue
        cut = fov_cut_cases(data_dirs[key], d['run']['dataset'], set(d['df']['case']))
        nums['failures'][f'{key}_fov_cut'] = sorted(c for c, v in cut.items() if v)
        can = d['mesh'][(d['mesh']['cfg'] == 'canonical') & (d['mesh']['resolution'] == 'native')]
        dice = d['mask'][d['mask']['resolution'] == 'native'].set_index(['source', 'case'])['dice']
        for src in [s for s in sources_of(d) if s != 'gt']:
            g = can[can['source'] == src].set_index('case')
            for case in g['assd_mm'].nlargest(3).index:
                r = g.loc[case]
                missed, extra = largest_disagreement(d['run']['dataset'], data_dirs[key], case, src)
                rows.append([d['organ'], SHORT[src], case.split('_')[-1], r['spacing'].split('x')[-1][:4],
                             f"{dice.get((src, case), np.nan):.3f}", f"{r['assd_mm']:.2f}", f"{r['hd95_mm']:.1f}",
                             f'{missed:.1f}', f'{extra:.1f}', f"{r['volume_err_pct']:+.1f}",
                             'cut by FOV' if cut.get(case) else ''])
                nums['failures'][f'{key}_{src}_{case}'] = {
                    'assd': float(r['assd_mm']), 'hd95': float(r['hd95_mm']), 'dice': float(dice.get((src, case), np.nan)),
                    'patch_mm2': float(r['defect_patch_mm2']), 'volume_err_pct': float(r['volume_err_pct']),
                    'largest_missed_pct': missed, 'largest_extra_pct': extra, 'fov_cut': bool(cut.get(case))}
            rows.append(None)
    header = ['Organ', 'Engine', 'Scan', 'Slice (mm)', 'Dice', 'ASSD (mm)', 'HD95 (mm)', 'Missed (\\%)',
              'Extra (\\%)', 'Volume (\\%)', 'Note']
    align = 'lllrrrrrrrl'
    if not any(r and r[-1] for r in rows):            # no scan needs a note: drop the column
        rows = [r[:-1] if r else r for r in rows]
        header, align = header[:-1], align[:-1]
    table(os.path.join(out, 'failures.tex'), header, rows[:-1], align)


def largest_disagreement(dataset, data_dir, case, src):
    """Largest connected region the engine missed and added at native
    resolution, each as a percentage of the expert organ volume."""
    import nibabel as nib
    from scipy import ndimage
    from research.tier2 import DATASETS, cached_mask_path, list_cases
    structure, values = next(iter(DATASETS[dataset]['structures'].items()))
    lbl = next(c[3] for c in list_cases(dataset, data_dir) if c[0] == case)
    gt = np.isin(np.asanyarray(nib.load(lbl).dataobj), values)
    seg = np.load(cached_mask_path(dataset, case, src, structure))['mask']

    def largest(m):
        lab, n = ndimage.label(m)
        return 0.0 if n == 0 else float(np.bincount(lab.ravel())[1:].max() / gt.sum() * 100)
    return largest(gt & ~seg), largest(seg & ~gt)


# ── S4 sampling models and prediction intervals ──────────────────────────

def added_components(floor):
    """Sampling added in-plane and along the slice axis (quadrature, as Delta)."""
    grid = floor['grid_spacing'].str.split('x', expand=True).astype(float).to_numpy()
    nat = floor['spacing'].str.split('x', expand=True).astype(float).to_numpy()
    add2 = np.clip(grid ** 2 - nat ** 2, 0, None)
    slice_axis = np.where(np.ptp(nat, axis=1) > 1e-6, np.argmax(nat, axis=1), 2)
    sl = add2[np.arange(len(add2)), slice_axis]
    return np.sqrt((add2.sum(axis=1) - sl) / 3), np.sqrt(sl / 3)


def _linear(design):
    def fit_(f, y):
        return np.linalg.lstsq(design(f), y, rcond=None)[0]

    def predict(f, coef):
        return design(f) @ coef
    return fit_, predict


def _quadrature():
    """ASSD = sqrt(c1 d_in^2 + c2 d_sl^2): axis-specific weights inside the quadrature sum."""
    from scipy.optimize import least_squares

    def predict(f, c):
        return np.sqrt(np.clip(c[0] * f['d_in'].to_numpy() ** 2 + c[1] * f['d_sl'].to_numpy() ** 2, 0, None))

    def fit_(f, y):
        return least_squares(lambda c: predict(f, c) - y, x0=[0.03, 0.03], bounds=(0, np.inf)).x
    return fit_, predict


MODELS = {
    r'M0: $b\Delta$': _linear(lambda f: np.c_[f['added_mm']]),
    r'M1: $a+b\Delta$': _linear(lambda f: np.c_[np.ones(len(f)), f['added_mm']]),
    r'M2: $b_1\Delta_{\mathrm{in}}+b_2\Delta_{\mathrm{sl}}$': _linear(lambda f: np.c_[f['d_in'], f['d_sl']]),
    r'M3: $\sqrt{c_1\Delta_{\mathrm{in}}^2+c_2\Delta_{\mathrm{sl}}^2}$': _quadrature(),
}


def s4_sampling_models(runs, out, nums):
    floors = {}
    for key, d in runs.items():
        m = d['mesh']
        f = m[(m['source'] == 'gt') & (m['cfg'] == 'canonical') & m['resampled']].copy()
        if key == 'sensitivity':                 # noise touches only the CT: keep the box profile
            f = f[f['resolution'].str.endswith('~box')]
        f['d_in'], f['d_sl'] = added_components(f)
        floors[key] = f
    sp = floors['spleen']
    y = sp['assd_mm'].to_numpy()
    rows, res = [], {}
    for name, (fit_, predict) in MODELS.items():
        coef = fit_(sp, y)
        pred = predict(sp, coef)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        loso = np.empty_like(y)
        for c in sp['case'].unique():
            k = (sp['case'] == c).to_numpy()
            loso[k] = predict(sp[k], fit_(sp[~k], y[~k]))
        e = y - loso
        half = float(np.quantile(np.abs(e), 0.95))        # jackknife 95% prediction interval
        lolo = {}
        for lvl in sp['resolution'].unique():
            k = (sp['resolution'] == lvl).to_numpy()
            lolo[lvl] = float(np.abs(y[k] - predict(sp[k], fit_(sp[~k], y[~k]))).mean())
        entry = {'coef': [float(c) for c in coef], 'r2': float(r2), 'loso_mae': float(np.abs(e).mean()),
                 'loso_r2': float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), 'pi95_half': half,
                 'lolo_mae': lolo}
        for key in [k for k in floors if k != 'spleen']:
            f = floors[key]
            ep = f['assd_mm'].to_numpy() - predict(f, coef)
            entry[f'{key}_mae'] = float(np.abs(ep).mean())
            entry[f'{key}_bias'] = float(ep.mean())
            entry[f'{key}_coverage'] = float((np.abs(ep) <= half).mean())
            entry[f'{key}_n'] = int(len(ep))
        res[name] = entry
        coef_txt = ', '.join(f'{c:.3f}' for c in coef)
        rows.append([name, coef_txt, f'{r2:.2f}', f"{entry['loso_mae']:.3f}", f'$\\pm${half:.3f}',
                     f"{lolo.get('slice5', np.nan):.3f}",
                     f"{entry.get('pancreas_mae', np.nan):.3f} ({100 * entry.get('pancreas_coverage', np.nan):.0f}\\%)"
                     if 'pancreas' in floors else '--',
                     f"{entry.get('sensitivity_mae', np.nan):.3f}" if 'sensitivity' in floors else '--'])
    table(os.path.join(out, 'sampling_models.tex'),
          ['Model', 'Coefficients', '$R^2$', 'LOSO MAE', '95\\% PI', 'slice5 held out', 'Pancreas MAE (PI cov.)',
           'Box profile MAE'], rows, 'lrrrrrrr')
    # Per-organ and per-profile slopes of M0, with scan-bootstrap CIs
    rng = np.random.default_rng(2026)
    slopes = {}
    for key, f in floors.items():
        x, yy = f['added_mm'].to_numpy(), f['assd_mm'].to_numpy()
        cases = f['case'].unique()
        idx = {c: np.flatnonzero((f['case'] == c).to_numpy()) for c in cases}
        boots = []
        for _ in range(2000):
            k = np.concatenate([idx[c] for c in rng.choice(cases, len(cases))])
            boots.append(float(x[k] @ yy[k] / (x[k] @ x[k])))
        slopes[key] = {'slope': float(x @ yy / (x @ x)), 'ci': [float(np.percentile(boots, 2.5)),
                                                                 float(np.percentile(boots, 97.5))],
                       'n': int(len(x)), 'scans': int(len(cases))}
    nums['sampling_models'] = res
    nums['sampling_slopes'] = slopes
    return floors


# ── S5 slice5 subsets ────────────────────────────────────────────────────

def s5_slice5(runs, out, nums, paper=None):
    """Paired slice5 effects; CIs are taken from the paper's numbers when given,
    so the supplement and the paper quote the same interval."""
    from scipy.stats import t as student
    rng = np.random.default_rng(2026)
    rows = []
    nums['slice5'] = {}
    for key, d in runs.items():
        if key == 'sensitivity':
            continue
        for src in sources_of(d):
            dlt = paired(d['mesh'], src, 'canonical', 'slice5', rng)
            n = len(dlt)
            if n < 3:
                continue
            ci = boot_mean(dlt, rng)
            shared = (paper or {}).get('tier2', {}).get(key, {}).get('paired', {}).get(f'{src}_slice5')
            if shared and abs(shared['diff'] - dlt.mean()) < 1e-9:
                ci = shared['ci']
            loo = [float(dlt.drop(c).mean()) for c in dlt.index]
            sd = float(dlt.std())
            mde = float((student.ppf(0.975, n - 1) + student.ppf(0.80, n - 1)) * sd / np.sqrt(n))
            nums['slice5'][f'{key}_{src}'] = {'n': n, 'mean': float(dlt.mean()), 'ci': ci, 'loo': [min(loo), max(loo)],
                                              'up': int((dlt > 0).sum()), 'sd': sd, 'mde80': mde,
                                              'per_scan': {c: float(v) for c, v in dlt.items()}}
            rows.append([d['organ'], SHORT[src], n, ci_text(dlt.mean(), ci), f'{min(loo):+.3f} to {max(loo):+.3f}',
                         f'{int((dlt > 0).sum())}/{n}', f'{sd:.3f}', f'{mde:.3f}'])
        rows.append(None)
    table(os.path.join(out, 'slice5.tex'),
          ['Organ', 'Source', '$n$', 'Change [95\\% CI] (mm)', 'Leave-one-out range', 'Scans up', 'SD', 'MDE$_{80\\%}$'],
          rows[:-1], 'llrrrrrr')


# ── S6 topology without the field-of-view-cut scans ──────────────────────

def s6_topology(runs, out, nums):
    rows = []
    nums['topology'] = {}
    for key, d in runs.items():
        if key == 'sensitivity':
            continue
        cut = set(nums['failures'].get(f'{key}_fov_cut', []))
        g = d['mesh'][d['mesh']['config'] == 'grid']
        for src in sources_of(d):
            for r in LEVELS:
                s = g[(g['source'] == src) & (g['resolution'] == r)]
                if not len(s):
                    continue
                allr = s.groupby('decimation')['topology_ok'].mean()
                kept = s[~s['case'].isin(cut)].groupby('decimation')['topology_ok'].mean()
                nums['topology'][f'{key}_{src}_{r}'] = {'all': {str(k): float(v) for k, v in allr.items()},
                                                        'without_cut': {str(k): float(v) for k, v in kept.items()}}
                rows.append([d['organ'], SHORT[src], code(r)] +
                            [f'{100 * allr[x]:.0f} / {100 * kept[x]:.0f}' for x in (0.0, 0.25, 0.5, 0.75)])
            rows.append(None)
    table(os.path.join(out, 'topology.tex'),
          ['Organ', 'Source', 'Level', '0', '0.25', '0.5', '0.75'], rows[:-1], 'lllrrrr',
          group='& & & \\multicolumn{4}{c}{Closed (\\%), all scans / without FOV-cut scans, by decimation} \\\\\n'
                '\\cmidrule(lr){4-7}')


# ── S7 blur units ────────────────────────────────────────────────────────

def s7_blur_units(runs, out, nums):
    rng = np.random.default_rng(2026)
    rows = []
    nums['blur_units'] = {}
    for key, d in runs.items():
        if key == 'sensitivity':
            continue
        m = d['mesh']
        for src in sources_of(d):
            for r in LEVELS:
                a = m[(m['source'] == src) & (m['resolution'] == r) & (m['cfg'] == 'production')].set_index('case')
                b = m[(m['source'] == src) & (m['resolution'] == r) & (m['cfg'] == 'production_mm')].set_index('case')
                both = a[['assd_mm', 'topology_ok']].join(b[['assd_mm', 'topology_ok']], lsuffix='_v', rsuffix='_m',
                                                           how='inner')
                if len(both) < 3:
                    continue
                dl = both['assd_mm_m'] - both['assd_mm_v']
                ci = boot_mean(dl, rng)
                nums['blur_units'][f'{key}_{src}_{r}'] = {
                    'vox': float(both['assd_mm_v'].mean()), 'mm': float(both['assd_mm_m'].mean()),
                    'diff': float(dl.mean()), 'ci': ci, 'closed_vox': float(both['topology_ok_v'].mean()),
                    'closed_mm': float(both['topology_ok_m'].mean()), 'n': int(len(both))}
                rows.append([d['organ'], SHORT[src], code(r), len(both), f"{both['assd_mm_v'].mean():.3f}",
                             f"{both['assd_mm_m'].mean():.3f}", ci_text(dl.mean(), ci),
                             f"{100 * both['topology_ok_v'].mean():.0f} / {100 * both['topology_ok_m'].mean():.0f}"])
            rows.append(None)
    table(os.path.join(out, 'blur_units.tex'),
          ['Organ', 'Source', 'Level', '$n$', 'Voxel blur', 'mm blur', 'mm $-$ voxel [95\\% CI]', 'Closed (\\%)'],
          rows[:-1], 'lllrrrrr')


# ── S8 local and task-specific metrics ───────────────────────────────────

def s8_local(runs, out, nums):
    rows = []
    nums['local'] = {}
    for key, d in runs.items():
        if key == 'sensitivity':
            continue
        m = d['mesh']
        for src in sources_of(d):
            for r in LEVELS:
                for cfg in ('canonical', 'production'):
                    g = m[(m['source'] == src) & (m['resolution'] == r) & (m['cfg'] == cfg)]
                    if not len(g) or (src == 'gt' and r == 'native' and cfg == 'canonical'):
                        continue
                    v = {'nsd05': g['nsd_0.5mm'].mean(), 'nsd1': g['nsd'].mean(), 'nsd2': g['nsd_2mm'].mean(),
                         'p99': g['p99_mm'].mean(), 'defect_pct': g['defect_area_pct'].mean(),
                         'patch_median': g['defect_patch_mm2'].median(), 'patch_max': g['defect_patch_mm2'].max(),
                         'missed_pct': g['missed_ref_pct'].mean(), 'abs_vol': g['volume_err_pct'].abs().mean(),
                         'hd95_le1': (g['hd95_mm'] <= 1).mean(), 'hd95_le2': (g['hd95_mm'] <= 2).mean(),
                         'vol_le5': (g['volume_err_pct'].abs() <= 5).mean(), 'n': len(g)}
                    nums['local'][f'{key}_{src}_{r}_{cfg}'] = {k: float(x) for k, x in v.items()}
                    rows.append([d['organ'], SHORT[src], code(r), cfg[:4] + '.', f"{v['nsd05']:.2f}", f"{v['nsd1']:.2f}",
                                 f"{v['nsd2']:.2f}", f"{v['p99']:.2f}", f"{v['defect_pct']:.1f}",
                                 f"{v['patch_max']:.0f}", f"{v['abs_vol']:.1f}",
                                 f"{100 * v['hd95_le2']:.0f}", f"{100 * v['vol_le5']:.0f}"])
            rows.append(None)
    table(os.path.join(out, 'local.tex'),
          ['Organ', 'Source', 'Level', 'Cfg', 'NSD$_{0.5}$', 'NSD$_1$', 'NSD$_2$', 'P99', 'Defect (\\%)',
           'Max patch', '$|$Vol$|$ (\\%)', 'HD95$\\leq$2', '$|$Vol$|\\leq$5\\%'],
          rows[:-1], 'llll' + 'r' * 9)


def error_map_figure(d, data_dir, cases, src, path):
    """Local error of the engine on chosen scans, at native resolution, in a coronal view.

    Top row: the delivered (production) mesh coloured by its distance to the
    reference surface, where the engine adds or displaces surface. Bottom
    row: the reference surface coloured by its distance to the engine's
    mesh, where the engine misses anatomy. The reference is shown lightly
    smoothed for legibility; its colours are distances from the exact
    canonical surface.
    """
    import nibabel as nib
    import pyvista as pv
    from reconstruction import ReconParams, extract_isosurface, mesh_arrays, postprocess_surface
    from research.surface_metrics import point_to_mesh_distance
    from research.tier2 import (DATASETS, _params, cached_mask_path, configs_for, crop_to,
                                list_cases)
    dataset = d['run']['dataset']
    structure, labels = next(iter(DATASETS[dataset]['structures'].items()))
    by_id = {c[0]: c for c in list_cases(dataset, data_dir)}
    prod = next(c for c in configs_for('canonical', src, structure) if c['config'] == 'production')
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(shape=(2, len(cases)), off_screen=True, window_size=(520 * len(cases), 1000), border=False)

    def polydata(v, f):
        return pv.PolyData(np.asarray(v, float), np.hstack([np.full((len(f), 1), 3), f]).ravel())

    bar = {'title': 'Distance (mm)', 'n_labels': 5, 'fmt': '%.0f', 'color': 'black',
           'position_x': 0.25, 'position_y': 0.04, 'width': 0.5}
    for i, case in enumerate(cases):
        lab = nib.load(by_id[case][3])
        sp = tuple(float(z) for z in lab.header.get_zooms()[:3])
        gt = np.isin(np.asanyarray(lab.dataobj), labels)
        test = np.load(cached_mask_path(dataset, case, src, structure))['mask']
        (gc,), g0 = crop_to([gt], 6)
        (tc,), t0 = crop_to([test], 6)
        rv, rf = extract_isosurface(gc, sp, ReconParams(), tuple(float(a) * s for a, s in zip(g0, sp)))
        params = _params(prod, sp)
        v, f = extract_isosurface(tc, sp, params, tuple(float(a) * s for a, s in zip(t0, sp)))
        pv_mesh = postprocess_surface(v, f, params)
        tv, tf = mesh_arrays(pv_mesh)
        delivered = polydata(tv, tf)
        delivered['d'] = np.minimum(point_to_mesh_distance(tv, rv, rf)[0], 4.0)
        reference = polydata(rv, rf)
        reference['d'] = np.minimum(point_to_mesh_distance(rv, tv, tf)[0], 4.0)
        reference = reference.smooth_taubin(n_iter=20, pass_band=0.1)
        row = d['mesh'].query("case == @case and source == @src and resolution == 'native'")
        assd = row[row['cfg'] == 'canonical']['assd_mm'].iloc[0]
        for r, (mesh, title) in enumerate(((delivered, f'{case} (ASSD {assd:.2f} mm): delivered mesh'),
                                           (reference, f'{case}: reference, distance to engine'))):
            plotter.subplot(r, i)
            plotter.add_mesh(mesh, scalars='d', cmap='inferno', clim=(0, 4), smooth_shading=True,
                             show_scalar_bar=(r == 1 and i == len(cases) - 1), scalar_bar_args=bar)
            plotter.add_text(title, font_size=9, color='black')
            plotter.view_xz()
            plotter.camera.elevation = 10
            plotter.reset_camera()
            plotter.camera.zoom(1.15)
    plotter.set_background('white')
    plotter.screenshot(path)
    plotter.close()


# ── S9 coarsening-model sensitivity ──────────────────────────────────────

def s9_coarsening(runs, out, nums):
    if 'sensitivity' not in runs:
        return
    rng = np.random.default_rng(2026)
    base, sens = runs['spleen']['mesh'], runs['sensitivity']['mesh']
    bm, sm = runs['spleen']['mask'], runs['sensitivity']['mask']
    rows = []
    nums['coarsening'] = {}
    for src in [s for s in ('gt', 'ts_fast') if s in set(sens['source'])]:
        for lvl in ('iso3', 'slice5'):
            for variant in ('box', 'noise'):
                if src == 'gt' and variant == 'noise':
                    continue                       # noise is added to the CT only
                for cfg in ('canonical', 'production'):
                    a = base[(base['source'] == src) & (base['resolution'] == lvl) & (base['cfg'] == cfg)].set_index('case')
                    b = sens[(sens['source'] == src) & (sens['resolution'] == f'{lvl}~{variant}')
                             & (sens['cfg'] == cfg)].set_index('case')
                    both = a[['assd_mm']].join(b[['assd_mm']], lsuffix='_g', rsuffix='_v', how='inner')
                    if len(both) < 3:
                        continue
                    dl = both['assd_mm_v'] - both['assd_mm_g']
                    ci = boot_mean(dl, rng)
                    da = bm[(bm['source'] == src) & (bm['resolution'] == lvl)].set_index('case')['dice']
                    db = sm[(sm['source'] == src) & (sm['resolution'] == f'{lvl}~{variant}')].set_index('case')['dice']
                    nums['coarsening'][f'{src}_{lvl}_{variant}_{cfg}'] = {
                        'gaussian': float(both['assd_mm_g'].mean()), 'variant': float(both['assd_mm_v'].mean()),
                        'diff': float(dl.mean()), 'ci': ci, 'n': int(len(both)),
                        'dice_gaussian': float(da.reindex(both.index).mean()),
                        'dice_variant': float(db.reindex(both.index).mean())}
                    rows.append([SHORT[src], code(lvl), variant, cfg, len(both), f"{both['assd_mm_g'].mean():.3f}",
                                 f"{both['assd_mm_v'].mean():.3f}", ci_text(dl.mean(), ci),
                                 f"{da.reindex(both.index).mean():.3f} / {db.reindex(both.index).mean():.3f}"])
        rows.append(None)
    table(os.path.join(out, 'coarsening.tex'),
          ['Source', 'Level', 'Variant', 'Config', '$n$', 'Gaussian', 'Variant', 'Difference [95\\% CI]', 'Dice G / V'],
          rows[:-1], 'llllrrrrr')


# ── S10 engine modes ─────────────────────────────────────────────────────

def s10_engines(runs, out, nums):
    rng = np.random.default_rng(2026)
    rows = []
    nums['engines'] = {}
    for key, d in runs.items():
        if key == 'sensitivity' or 'ts_full' not in sources_of(d):
            continue
        m, k = d['mesh'], d['mask']
        for r in LEVELS:
            for cfg in ('canonical', 'production'):
                a = m[(m['source'] == 'ts_fast') & (m['resolution'] == r) & (m['cfg'] == cfg)].set_index('case')
                b = m[(m['source'] == 'ts_full') & (m['resolution'] == r) & (m['cfg'] == cfg)].set_index('case')
                both = a[['assd_mm', 'topology_ok']].join(b[['assd_mm', 'topology_ok']], lsuffix='_f', rsuffix='_u',
                                                           how='inner')
                if len(both) < 3:
                    continue
                dl = both['assd_mm_u'] - both['assd_mm_f']
                ci = boot_mean(dl, rng)
                dice = k[k['resolution'] == r].pivot_table(index='case', columns='source', values='dice')
                nums['engines'][f'{key}_{r}_{cfg}'] = {
                    'fast': float(both['assd_mm_f'].mean()), 'full': float(both['assd_mm_u'].mean()),
                    'diff': float(dl.mean()), 'ci': ci, 'full_better': int((dl < 0).sum()), 'n': int(len(both)),
                    'dice_fast': float(dice['ts_fast'].mean()), 'dice_full': float(dice['ts_full'].mean()),
                    'closed_fast': float(both['topology_ok_f'].mean()), 'closed_full': float(both['topology_ok_u'].mean())}
                rows.append([d['organ'], code(r), cfg, len(both), f"{dice['ts_fast'].mean():.3f} / {dice['ts_full'].mean():.3f}",
                             f"{both['assd_mm_f'].mean():.3f}", f"{both['assd_mm_u'].mean():.3f}", ci_text(dl.mean(), ci),
                             f'{int((dl < 0).sum())}/{len(both)}'])
        rows.append(None)
    table(os.path.join(out, 'engines.tex'),
          ['Organ', 'Level', 'Config', '$n$', 'Dice fast / full', 'ASSD fast', 'ASSD full', 'Full $-$ fast [95\\% CI]',
           'Full better'], rows[:-1], 'lllrrrrrr')


# ── S11 scan properties behind the case effect ───────────────────────────

def scan_features(dataset, data_dir, cases):
    import nibabel as nib
    from scipy import ndimage
    from research.tier2 import DATASETS, list_cases
    values = next(iter(DATASETS[dataset]['structures'].values()))
    feats = {}
    for cid, _, img_path, lbl in list_cases(dataset, data_dir):
        if cid not in cases:
            continue
        lab = nib.load(lbl)
        sp = np.array(lab.header.get_zooms()[:3], dtype=float)
        m = np.isin(np.asanyarray(lab.dataobj), values)
        hu = np.asanyarray(nib.load(img_path).dataobj).astype(np.float32)
        inner = ndimage.binary_erosion(m, iterations=1)
        shell = ndimage.binary_dilation(m, iterations=max(1, int(round(3 / sp[:2].min())))) & ~m
        shell &= hu > -200                                    # soft tissue around the organ, not air
        vol = m.sum() * sp.prod() / 1000
        faces = ndimage.binary_dilation(m) & ~m               # crude surface: outer boundary voxels
        feats[cid] = {'slice_mm': float(sp.max()), 'inplane_mm': float(sp[:2].min()), 'volume_cm3': float(vol),
                      'organ_hu': float(hu[inner].mean()), 'organ_hu_sd': float(hu[inner].std()),
                      'contrast_hu': float(hu[inner].mean() - hu[shell].mean()),
                      'boundary_ratio': float(faces.sum() / max(m.sum(), 1)),
                      'fov_cut': float(m[..., 0].any() or m[..., -1].any() or m[0].any() or m[-1].any()
                                       or m[:, 0].any() or m[:, -1].any())}
    return pd.DataFrame(feats).T


def s11_covariates(runs, data_dirs, out, nums):
    from scipy.stats import spearmanr
    rows = []
    nums['covariates'] = {}
    names = {'slice_mm': 'Slice spacing', 'inplane_mm': 'In-plane spacing', 'volume_cm3': 'Organ volume',
             'organ_hu': 'Organ HU', 'organ_hu_sd': 'Organ HU SD', 'contrast_hu': 'Organ-to-surround contrast',
             'boundary_ratio': 'Boundary-to-volume ratio', 'fov_cut': 'Cut by field of view'}
    for key, d in runs.items():
        if key == 'sensitivity':
            continue
        feats = scan_features(d['run']['dataset'], data_dirs[key], set(d['df']['case']))
        nums['covariates'][f'{key}_features'] = feats.round(4).to_dict(orient='index')
        for src in [s for s in sources_of(d) if s != 'gt']:
            y = d['mesh'][(d['mesh']['source'] == src) & (d['mesh']['cfg'] == 'canonical')
                          & (d['mesh']['resolution'] == 'native')].set_index('case')['assd_mm']
            f = feats.reindex(y.index)
            corr = {}
            for c in names:
                if f[c].nunique() > 1:
                    rho, p = spearmanr(f[c], y)
                    corr[c] = {'rho': float(rho), 'p': float(p)}
            # leave-one-scan-out linear model on the continuous properties
            cols = ['slice_mm', 'volume_cm3', 'contrast_hu', 'organ_hu_sd', 'boundary_ratio']
            X = np.c_[np.ones(len(f)), (f[cols] - f[cols].mean()) / f[cols].std()]
            yy = y.to_numpy()
            pred = np.array([X[i] @ np.linalg.lstsq(np.delete(X, i, 0), np.delete(yy, i), rcond=None)[0]
                             for i in range(len(yy))])
            loso_r2 = float(1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
            nums['covariates'][f'{key}_{src}'] = {'spearman': corr, 'loso_r2_linear': loso_r2, 'n': int(len(yy))}
            rows.append([d['organ'], SHORT[src]] + [
                f"{corr[c]['rho']:+.2f}{'$^*$' if corr[c]['p'] < 0.05 else ''}" if c in corr else '--'
                for c in names] + [f'{loso_r2:.2f}'])
    table(os.path.join(out, 'covariates.tex'),
          ['Organ', 'Engine'] + [names[c] for c in names] + ['LOSO $R^2$'], rows, 'll' + 'r' * (len(names) + 1))


# ── S12 sensitivity across configurations ────────────────────────────────

def s12_configurations(runs, out, nums, figs):
    plt = rep._style()
    nums['configurations'] = {}
    keys = [k for k in runs if k != 'sensitivity']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    labels, data_iso, data_sl, rows = [], [], [], []
    for key in keys:
        d = runs[key]
        m = d['mesh'][d['mesh']['config'] != 'production_mm']
        for src in [s for s in sources_of(d) if s != 'gt']:
            s = m[m['source'] == src].copy()
            s['cfg_id'] = s['config'].astype(str) + '/' + s[GRID].apply(
                lambda row: '/'.join(f'{v}' for v in row), axis=1)
            iso, sl, case_dominant = [], [], []
            for cid, g in s.groupby('cfg_id'):
                base = g[g['resolution'] == 'native'].set_index('case')['assd_mm']
                for lvl, store in (('iso3', iso), ('slice5', sl)):
                    v = g[(g['resolution'] == lvl) & g['resampled']].set_index('case')['assd_mm']
                    store.append(float((v - base.reindex(v.index)).mean()))
                sh = eta_squared(g, 'assd_mm', ['resolution', 'case'])
                case_dominant.append(sh['case'] > sh['resolution'])
            iso, sl = np.array(iso), np.array(sl)
            nums['configurations'][f'{key}_{src}'] = {
                'configs': int(len(iso)), 'iso3_change_range': [float(iso.min()), float(iso.max())],
                'iso3_within_005': float((np.abs(iso) <= 0.05).mean()),
                'slice5_change_range': [float(sl.min()), float(sl.max())], 'slice5_positive': float((sl > 0).mean()),
                'case_dominant': float(np.mean(case_dominant))}
            labels.append(f'{d["organ"]}\n{SHORT[src]}')
            data_iso.append(iso)
            data_sl.append(sl)
            rows.append([d['organ'], SHORT[src], len(iso), f'{iso.min():+.3f} to {iso.max():+.3f}',
                         f'{100 * (np.abs(iso) <= 0.05).mean():.0f}\\%', f'{sl.min():+.3f} to {sl.max():+.3f}',
                         f'{100 * (sl > 0).mean():.0f}\\%', f'{100 * np.mean(case_dominant):.0f}\\%'])
    for ax, data, title in ((axes[0], data_iso, 'iso3 $-$ native'), (axes[1], data_sl, 'slice5 $-$ native')):
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops={'color': rep.TEXT})
        rng = np.random.default_rng(0)
        for i, v in enumerate(data, 1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color='#FF7A18', alpha=0.6, linewidths=0)
        ax.axhline(0, color=rep.TEXT2, linewidth=0.7)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=6.5)
        ax.set_title(f'Engine ASSD change, {title}', loc='left')
        ax.set_ylabel('Mean change over scans (mm)')
    fig.tight_layout()
    fig.savefig(os.path.join(figs, 'configurations.pdf'))
    plt.close(fig)
    table(os.path.join(out, 'configurations.tex'),
          ['Organ', 'Engine', 'Configs', 'iso3 change range', '$|$iso3$|\\leq$0.05', 'slice5 change range',
           'slice5 $>$ 0', 'Case $>$ resolution'], rows, 'llrrrrrr')


# ── main ─────────────────────────────────────────────────────────────────

# ── S13 engine repeatability; S14 the application's accuracy estimate ────
# From validation tests 2 and 1 (research/repeatability.py, research/card_validation.py).

REP_PATHS = {'worker': 'one persistent process', 'fresh_gpu': 'new GPU process', 'cpu': 'CPU'}
LEVEL_NAMES = {'native': 'native', 'iso1.5': 'iso1.5', 'iso3': 'iso3', 'slice5': 'slice5'}


def s13_repeatability(rep_dir, out, nums):
    rep = pd.read_csv(results_file(rep_dir))
    first = rep[(rep['path'] == 'worker') & (rep['run'] == 0)]
    again = rep.drop(first.index)
    rows, res = [], {}
    for (mode, path), g in again.groupby(['mode', 'path'], sort=False):
        ref = first[first['mode'] == mode].set_index('case')['true_assd_mm']
        runs_ = pd.concat([rep[(rep['mode'] == mode) & (rep['path'] == path)].set_index('case')['true_assd_mm'], ref])
        spread = float((runs_.groupby(level=0).max() - runs_.groupby(level=0).min()).max())
        res[f'{mode}_{path}'] = {'runs': int(len(g)), 'organ_voxels_max': int(g['organ_diff_voxels'].max()),
                                 'dice_min': float(g['organ_dice'].min()),
                                 'shift_max_mm': float(g['mesh_assd_vs_first_mm'].max()), 'assd_range_mm': spread,
                                 'seconds': float(rep[(rep['mode'] == mode) & (rep['path'] == path)]['seconds'].median())}
        r = res[f'{mode}_{path}']
        rows.append([mode, REP_PATHS[path], r['runs'], r['organ_voxels_max'], f"{r['dice_min']:.4f}",
                     f"{r['shift_max_mm']:.3f}", f"{r['assd_range_mm']:.3f}", f"{r['seconds']:.0f}"])
    nums['repeatability'] = {'scans': int(rep['case'].nunique()), 'runs': int(len(rep)), 'cells': res,
                             'shift_max_mm': float(again['mesh_assd_vs_first_mm'].max()),
                             'organ_voxels_max': int(again['organ_diff_voxels'].max())}
    table(os.path.join(out, 'repeatability.tex'),
          ['Mode', 'Path', 'Runs', 'Organ voxels', 'Dice', 'Mesh shift', 'ASSD range', 'Seconds'], rows, 'llrrrrrr')


def s14_accuracy_estimate(card_dir, out, figs, nums):
    import accuracy
    from research.card_validation import coverage
    df = pd.read_csv(results_file(card_dir))
    ok = df[(df['source'] != 'gt') & (df['status'] == 'ok')]
    gt = df[df['source'] == 'gt']
    cov = {k: coverage(ok, k) for k in ('v1', 'v2')}
    true = {k: c[c['what'] == 'true_assd_mm'] for k, c in cov.items()}
    cells = pd.concat([true[k].groupby(['organ', 'mode', 'level'])[['mid', 'p90']].mean().add_prefix(k + '_')
                       for k in ('v1', 'v2')], axis=1).reset_index()
    order = {lv: i for i, lv in enumerate(LEVEL_NAMES)}
    cells = cells.sort_values(['organ', 'mode', 'level'], ascending=[False, True, True],
                              key=lambda c: c.map(order) if c.name == 'level' else c)
    rows, last = [], None
    for _, r in cells.iterrows():
        if last is not None and r['organ'] != last:
            rows.append(None)
        last = r['organ']
        rows.append([r['organ'], r['mode'], code(r['level']), f"{100 * r['v1_mid']:.0f}", f"{100 * r['v2_mid']:.0f}",
                     f"{100 * r['v1_p90']:.0f}", f"{100 * r['v2_p90']:.0f}"])
    table(os.path.join(out, 'accuracy_estimate.tex'),
          ['Organ', 'Mode', 'Level', 'v1 IQR', 'v2 IQR', 'v1 90\\%', 'v2 90\\%'], rows, 'lllrrrr')
    pa = gt[gt['structure'] == 'pancreas']
    miss = (pa['canon_assd_mm'] - accuracy.SLOPE * pa['delta_added_mm']).abs()
    worst = cells.loc[cells['v1_mid'].idxmin()]
    nums['accuracy_estimate'] = {
        'meshes': int(len(ok)), 'scans': int(df['case'].nunique()),
        'v1_mid': float(true['v1']['mid'].mean()), 'v1_p90': float(true['v1']['p90'].mean()),
        'v2_mid': float(true['v2']['mid'].mean()), 'v2_p90': float(true['v2']['p90'].mean()),
        'worst': {'organ': worst['organ'], 'mode': worst['mode'], 'level': worst['level'],
                  'v1_mid': float(worst['v1_mid']), 'v2_mid': float(worst['v2_mid'])},
        'resolution_pancreas_inside': float((miss <= accuracy.PREDICTION_INTERVAL_MM).mean()),
        'resolution_pancreas_mae': float(miss.mean()),
        'at_least_resolution': float((ok['true_assd_mm'] >= accuracy.SLOPE * ok['rms_spacing_mm']).mean()),
        'smoothing_signed_match_mm': float((ok['card_recon_signed_mm']
                                            - (ok['true_signed_mm'] - ok['canon_signed_mm'])).abs().median()),
        'watertight_agree': int((ok['card_watertight'] == ok['independent_manifold']).sum()),
        'rebuild_identical': int(ok['rebuild_identical'].sum())}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from research import validation_figures as vf
    vf._style(plt)
    vf.coverage_figure(card_dir, figs, plt, headline=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--tier1', required=True)
    ap.add_argument('--spleen', required=True)
    ap.add_argument('--pancreas', default=None)
    ap.add_argument('--sensitivity', default=None)
    ap.add_argument('--spleen-data', required=True)
    ap.add_argument('--pancreas-data', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-images', action='store_true', help='skip work that reads the CT volumes')
    ap.add_argument('--repeatability', default=None, help='validation test 2 run (S13)')
    ap.add_argument('--card-validation', default=None, help='validation test 1 run (S14)')
    ap.add_argument('--paper-numbers', default='docs/paper/paper_numbers.json',
                    help='paper_numbers.json, for intervals shared with the paper')
    args = ap.parse_args()
    tabs, figs = os.path.join(args.out, 'tables'), os.path.join(args.out, 'figures')
    os.makedirs(tabs, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    t1 = pd.read_csv(results_file(args.tier1))
    t1['spacing'] = t1['spacing'].astype(str)
    runs = {'spleen': load(args.spleen)}
    data_dirs = {'spleen': args.spleen_data}
    if args.pancreas:
        runs['pancreas'] = load(args.pancreas)
        data_dirs['pancreas'] = args.pancreas_data
    if args.sensitivity:
        runs['sensitivity'] = load(args.sensitivity)
    nums = {}
    s1_counts(t1, runs, tabs, nums)
    s2_floor_and_smoothing(t1, runs, tabs, nums)
    s3_per_scan(runs, tabs, nums, figs)
    if not args.skip_images:
        s3_failures(runs, data_dirs, tabs, nums)
    s4_sampling_models(runs, tabs, nums)
    paper = None
    if args.paper_numbers and os.path.exists(args.paper_numbers):
        with open(args.paper_numbers, encoding='utf-8') as fh:
            paper = json.load(fh)
    s5_slice5(runs, tabs, nums, paper)
    if 'failures' in nums:
        s6_topology(runs, tabs, nums)
    s7_blur_units(runs, tabs, nums)
    s8_local(runs, tabs, nums)
    s9_coarsening(runs, tabs, nums)
    s10_engines(runs, tabs, nums)
    if not args.skip_images:
        s11_covariates(runs, data_dirs, tabs, nums)
        worst = nums['per_scan']['spleen_ts_fast']['native']['worst']
        median_case = (runs['spleen']['mesh'].query("source == 'ts_fast' and cfg == 'canonical' and resolution == 'native'")
                       .set_index('case')['assd_mm'].sort_values().index[20])
        error_map_figure(runs['spleen'], args.spleen_data, [median_case, worst], 'ts_fast',
                         os.path.join(figs, 'error_maps.png'))
    s12_configurations(runs, tabs, nums, figs)
    if args.repeatability:
        s13_repeatability(args.repeatability, tabs, nums)
    if args.card_validation:
        s14_accuracy_estimate(args.card_validation, tabs, figs, nums)
    with open(os.path.join(args.out, 'supplement_numbers.json'), 'w', encoding='utf-8') as fh:
        json.dump(nums, fh, indent=1, default=float)
    print(f'supplement tables, figures and numbers -> {args.out}')


if __name__ == '__main__':
    main()
