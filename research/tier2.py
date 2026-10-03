"""
Tier 2: reconstruction error on real anatomy against expert annotations.

For each case and structure, the expert mask is turned into the reference
surface by the canonical procedure (no blur, marching cubes, no smoothing,
no decimation; the same procedure whose error floor Tier 1 measured). Test
meshes are then built from

  * the expert mask itself, under every reconstruction setting, which
    isolates reconstruction error on real anatomy (Exp. B); and
  * each segmentation engine's mask, under the same settings, which gives
    the end-to-end error (Exp. C),

and scored against the reference surface. Engine masks are also scored at
mask level (Dice, IoU, HD95, ASSD; Exp. A).

Usage (from the repository root):
  python -m research.tier2 --dataset msd_spleen --data data/msd/Task09_Spleen --preset pilot
  python -m research.tier2 --dataset ctorg --data data/ctorg --preset full --workers 5
  python -m research.tier2 --resummarise output/research/tier2/<run>

Outputs go to output/research/tier2/<timestamp>_<dataset>_<preset>/
(results.csv, run.json, summary.md). Engine masks are cached under
output/research/tier2/cache/ so re-runs skip segmentation.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reconstruction import (ReconParams, blur_mm_to_vox, extract_isosurface,  # noqa: E402
                            mesh_arrays, postprocess_surface)

# ── datasets ─────────────────────────────────────────────────────────────

DATASETS = {
    'msd_spleen': {
        'title': 'Medical Segmentation Decathlon, Task09 Spleen',
        'licence': 'CC-BY-SA 4.0',
        'labels': 'expert manual segmentation of the spleen in every training case',
        'structures': {'spleen': (1,)},
        'manual_cases': {},
    },
    'ctorg': {
        'title': 'CT-ORG (The Cancer Imaging Archive)',
        'licence': 'CC BY 3.0',
        'labels': ('manual liver in all 140 cases; manual lungs and bone only in cases 0-20 '
                   '(the training split lungs and bone are morphological, so they are excluded)'),
        'structures': {'liver': (1,), 'lungs': (3,), 'bone': (5,)},
        'manual_cases': {'lungs': range(0, 21), 'bone': range(0, 21)},
    },
}


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def list_cases(dataset, root):
    """[(case_id, case_number, image_path, label_path)] for a dataset folder."""
    if dataset == 'msd_spleen':
        with open(os.path.join(root, 'dataset.json'), encoding='utf-8') as fh:
            meta = json.load(fh)
        out = []
        for item in meta['training']:
            img = os.path.normpath(os.path.join(root, item['image']))
            lbl = os.path.normpath(os.path.join(root, item['label']))
            cid = os.path.basename(img).replace('.nii.gz', '')
            out.append((cid, int(re.findall(r'\d+', cid)[-1]), img, lbl))
        return sorted(out, key=lambda c: _natural(c[0]))
    if dataset == 'ctorg':
        out = []
        for img in glob.glob(os.path.join(root, '**', 'volume-*.nii.gz'), recursive=True):
            n = int(re.findall(r'volume-(\d+)', os.path.basename(img))[0])
            lbl = os.path.join(os.path.dirname(img), f'labels-{n}.nii.gz')
            if not os.path.exists(lbl):
                hits = glob.glob(os.path.join(root, '**', f'labels-{n}.nii.gz'), recursive=True)
                lbl = hits[0] if hits else None
            if lbl:
                out.append((f'ctorg_{n:03d}', n, img, lbl))
        return sorted(out, key=lambda c: c[1])
    raise ValueError(f'unknown dataset {dataset!r}')


def structure_allowed(dataset, structure, case_number):
    """False where a structure's label is not an expert annotation for this case."""
    allowed = DATASETS[dataset]['manual_cases'].get(structure)
    return True if allowed is None else case_number in allowed


# ── segmentation engines ─────────────────────────────────────────────────

LUNG_LOBES = ('lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
              'lung_middle_lobe_right', 'lung_lower_lobe_right')
BONE_PREFIXES = ('rib_', 'vertebrae_')
BONE_NAMES = ('sternum', 'costal_cartilages', 'clavicula_left', 'clavicula_right',
              'scapula_left', 'scapula_right', 'humerus_left', 'humerus_right',
              'femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum', 'skull')


def ts_files_for(structure):
    """Predicate on TotalSegmentator output names for a study structure."""
    if structure == 'lungs':
        return lambda n: n in LUNG_LOBES
    if structure == 'bone':
        return lambda n: n.startswith(BONE_PREFIXES) or n in BONE_NAMES
    return lambda n: n == structure


ENGINES = {
    # name: (structures it can segment, or None for any TotalSegmentator class)
    'ts_fast': None,
    'classical': ('lungs', 'bone'),
}


def engine_supports(engine, structure):
    s = ENGINES[engine]
    return True if s is None else structure in s


def cache_dir(dataset, case_id):
    out_root = os.environ.get('VRSEG_OUTPUT') or os.path.join(ROOT, 'output')
    return os.path.join(out_root, 'research', 'tier2', 'cache', dataset, case_id)


def cached_mask_path(dataset, case_id, engine, structure):
    return os.path.join(cache_dir(dataset, case_id), f'{engine}_{structure}.npz')


def segment_case(dataset, case, structures, engines, device, log=print):
    """Run each engine once per case and cache one mask per structure."""
    import nibabel as nib
    cid, _, img_path, _ = case
    need = [(e, s) for e in engines for s in structures
            if engine_supports(e, s) and not os.path.exists(cached_mask_path(dataset, cid, e, s))]
    if not need:
        return {}
    os.makedirs(cache_dir(dataset, cid), exist_ok=True)
    timings = {}
    img = nib.load(img_path)
    shape = img.shape[:3]
    if any(e == 'ts_fast' for e, _ in need):
        seg_dir = os.path.join(cache_dir(dataset, cid), 'ts_fast_seg')
        if not glob.glob(os.path.join(seg_dir, '*.nii.gz')):
            from totalsegmentator.python_api import totalsegmentator
            t = time.perf_counter()
            totalsegmentator(img_path, seg_dir, fast=True, ml=False, device=device,
                             task='total', quiet=True)
            timings['ts_fast_s'] = time.perf_counter() - t
            log(f'  {cid}: TotalSegmentator (3 mm) {timings["ts_fast_s"]:.0f} s on {device}')
        for e, s in need:
            if e != 'ts_fast':
                continue
            keep = ts_files_for(s)
            mask = np.zeros(shape, dtype=bool)
            for f in glob.glob(os.path.join(seg_dir, '*.nii.gz')):
                if keep(os.path.basename(f).replace('.nii.gz', '')):
                    m = np.asanyarray(nib.load(f).dataobj) > 0
                    if m.shape != shape:
                        raise RuntimeError(f'{cid}: TotalSegmentator grid {m.shape} != image {shape}')
                    mask |= m
            np.savez_compressed(cached_mask_path(dataset, cid, e, s), mask=mask)
    if any(e == 'classical' for e, _ in need):
        from validate import heuristic_bone, heuristic_lungs
        hu = np.asanyarray(img.dataobj).astype(np.float32)
        for e, s in need:
            if e != 'classical':
                continue
            t = time.perf_counter()
            mask = heuristic_lungs(hu) if s == 'lungs' else heuristic_bone(hu)
            timings[f'classical_{s}_s'] = time.perf_counter() - t
            np.savez_compressed(cached_mask_path(dataset, cid, e, s), mask=mask)
    return timings


# ── reconstruction configurations ────────────────────────────────────────

# Tier 1 found the isosurface variant explained 0.0% of variance, so Tier 2
# fixes it (Lewiner) and spends the budget on the factors that matter.
GRID = {'blur_mm': (0.0, 0.5, 1.0), 'taubin_iter': (0, 10, 30, 60),
        'decimation': (0.0, 0.25, 0.5, 0.75)}
PILOT_GRID = {'blur_mm': (0.0, 1.0), 'taubin_iter': (0, 30), 'decimation': (0.0, 0.5)}
CANONICAL = {'blur_mm': 0.0, 'taubin_iter': 0, 'decimation': 0.0}

# What the app actually ships for each engine and structure (blur in voxels,
# Taubin iterations, decimation), read from segmentation_pipeline.py call sites.
PRODUCTION = {
    ('ts_fast', None): (0.6, 30, 0.6),      # run_totalseg: build_mesh(sigma=0.6, reduction=0.6)
    ('classical', 'lungs'): (0.7, 30, 0.5),  # run_ct_chest
    ('classical', 'bone'): (0.5, 30, 0.4),   # run_ct_chest
    ('gt', None): (0.6, 30, 0.5),            # build_mesh defaults
}


def production_for(source, structure):
    return PRODUCTION.get((source, structure)) or PRODUCTION[(source if source != 'classical' else 'gt', None)]


def configs_for(grid_name, source, structure):
    grid = {'full': GRID, 'pilot': PILOT_GRID, 'canonical': None}[grid_name]
    combos = ([dict(zip(grid, v)) for v in product(*grid.values())] if grid
              else [dict(CANONICAL)])
    out = [dict(c, config='grid', blur_vox=None) for c in combos]
    b, t, d = production_for(source, structure)
    out.append({'config': 'production', 'blur_mm': None, 'blur_vox': b, 'taubin_iter': t, 'decimation': d})
    return out


def _params(cfg, spacing):
    blur = ((cfg['blur_vox'],) * 3 if cfg['blur_vox'] is not None
            else blur_mm_to_vox(cfg['blur_mm'], spacing))
    return ReconParams(blur_sigma_vox=blur, isosurface='lewiner',
                       taubin_iter=cfg['taubin_iter'], decimation=cfg['decimation'])


def crop_to(masks, margin):
    """Crop masks to their joint bounding box plus margin voxels; returns (crops, start)."""
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= m
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, union.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return [m[sl] for m in masks], lo


# ── one unit of work: a case, a structure, a mask source ────────────────

def run_unit(task):
    import nibabel as nib
    from validate import compute_metrics
    from research.surface_metrics import compare_meshes
    (dataset, cid, label_path, structure, label_values, source, grid_name,
     seed, n_samples, tol_mm, case_i) = task
    lab = nib.load(label_path)
    spacing = tuple(float(z) for z in lab.header.get_zooms()[:3])
    gt = np.isin(np.asanyarray(lab.dataobj).astype(np.int16), label_values)
    base = {'dataset': dataset, 'case': cid, 'structure': structure, 'source': source,
            'spacing': 'x'.join(f'{s:g}' for s in spacing), 'slice_mm': max(spacing)}
    if source == 'gt':
        test = gt
    else:
        test = np.load(cached_mask_path(dataset, cid, source, structure))['mask']
    rows = []
    if test.sum() == 0:
        return [dict(base, level='mask', status='empty_mask')]
    if source != 'gt':
        m = compute_metrics(test, gt, spacing)
        rows.append(dict(base, level='mask', status='ok', config='mask', dice=m['dice'], iou=m['iou'],
                         mask_hd95_mm=m['hd95_mm'], mask_assd_mm=m['assd_mm'],
                         pred_cm3=m['pred_cm3'], gt_cm3=m['gt_cm3']))

    margin = int(np.ceil(4 + 3 * 1.0 / min(spacing)))       # room for 1 mm blur
    (gt_c, test_c), start = crop_to([gt, test], margin)
    origin = tuple(float(a) * s for a, s in zip(start, spacing))
    rv, rf = extract_isosurface(gt_c, spacing, ReconParams(), origin)   # reference surface
    iso_cache = {}
    for cfg_i, cfg in enumerate(configs_for(grid_name, source, structure)):
        row = dict(base, level='mesh', config=cfg['config'],
                   is_canonical=(cfg['config'] == 'grid' and all(cfg[k] == v for k, v in CANONICAL.items())),
                   blur_mm=cfg['blur_mm'], blur_vox=cfg['blur_vox'],
                   taubin_iter=cfg['taubin_iter'], decimation=cfg['decimation'])
        params = _params(cfg, spacing)
        try:
            t0 = time.perf_counter()
            if params.blur_sigma_vox not in iso_cache:
                iso_cache[params.blur_sigma_vox] = extract_isosurface(test_c, spacing, params, origin)
            v, f = iso_cache[params.blur_sigma_vox]
            tv, tf = mesh_arrays(postprocess_surface(v, f, params))
            t1 = time.perf_counter()
            metrics = compare_meshes(tv, tf, rv, rf,
                                     np.random.default_rng([seed, case_i, cfg_i, len(source)]),
                                     n_samples=n_samples, tol_mm=tol_mm)
            row.update(status='ok', **metrics, t_post_s=t1 - t0, t_metric_s=time.perf_counter() - t1)
        except (ValueError, RuntimeError) as exc:
            row.update(status='no_surface', error=str(exc)[:120])
        rows.append(row)
    return rows


# ── run ──────────────────────────────────────────────────────────────────

PRESETS = {
    'smoke': {'cases': 1, 'grid': 'canonical', 'samples': 4000},
    'pilot': {'cases': 5, 'grid': 'pilot', 'samples': 20000},
    'full': {'cases': None, 'grid': 'full', 'samples': 20000},
}


def run_experiment(dataset, data_root, preset='pilot', engines=('ts_fast', 'classical'),
                   workers=None, seed=2026, tol_mm=1.0, outdir=None, max_cases=None, log=print):
    from research.phantom_experiment import _provenance
    meta, cfg = DATASETS[dataset], PRESETS[preset]
    cases = list_cases(dataset, data_root)
    limit = max_cases or cfg['cases']
    cases = cases[:limit] if limit else cases
    structures = list(meta['structures'])
    engines = [e for e in engines if any(engine_supports(e, s) for s in structures)]

    try:
        import torch
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    t_start = time.perf_counter()
    seg_timings = {}
    log(f'Segmenting {len(cases)} case(s) with {", ".join(engines) or "no engines"} ({device})...')
    for case in cases:
        seg_timings[case[0]] = segment_case(dataset, case, structures, engines, device, log)

    tasks = []
    for case_i, (cid, num, _, lbl) in enumerate(cases):
        for s in structures:
            if not structure_allowed(dataset, s, num):
                continue
            for source in ['gt'] + [e for e in engines if engine_supports(e, s)]:
                tasks.append((dataset, cid, lbl, s, meta['structures'][s], source, cfg['grid'],
                              seed, cfg['samples'], tol_mm, case_i))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows = []
    log(f'Reconstructing and scoring: {len(tasks)} units on {workers} worker(s)...')
    if workers == 1:
        for t in tasks:
            rows.extend(run_unit(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_unit, t) for t in tasks]
            for done, fut in enumerate(as_completed(futures), 1):
                rows.extend(fut.result())
                log(f'  unit {done}/{len(tasks)} done')
    wall = time.perf_counter() - t_start

    run_id = time.strftime('%Y%m%d-%H%M%S') + f'_{dataset}_{preset}'
    out_root = os.environ.get('VRSEG_OUTPUT') or os.path.join(ROOT, 'output')
    outdir = outdir or os.path.join(out_root, 'research', 'tier2', run_id)
    os.makedirs(outdir, exist_ok=True)
    rows.sort(key=lambda r: (r['structure'], r['case'], r['source'], str(r.get('config'))))
    fields = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with open(os.path.join(outdir, 'results.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    try:
        import totalsegmentator
        ts_version = getattr(totalsegmentator, '__version__', None)
    except ImportError:
        ts_version = None
    run = {
        'tier': 2, 'dataset': dataset, 'dataset_info': meta, 'preset': preset, 'seed': seed,
        'cases': [c[0] for c in cases], 'structures': structures, 'engines': engines,
        'grid': cfg['grid'], 'grid_levels': GRID if cfg['grid'] == 'full' else PILOT_GRID,
        'isosurface': 'lewiner (fixed: Tier 1 found it explained 0.0% of variance)',
        'canonical': CANONICAL, 'production': {f'{k[0]}/{k[1] or "default"}': v for k, v in PRODUCTION.items()},
        'reference_surface': 'canonical reconstruction of the expert mask at native spacing',
        'samples_per_side': cfg['samples'], 'nsd_tolerance_mm': tol_mm, 'segmentation_device': device,
        'segmentation_timings_s': seg_timings, 'totalsegmentator_version': ts_version,
        'rows': len(rows), 'units': len(tasks), 'workers': workers, 'wall_time_s': round(wall, 1),
        **_provenance(),
    }
    with open(os.path.join(outdir, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump(run, fh, indent=2, default=lambda o: list(o) if isinstance(o, range) else str(o))
    try:
        summary = summarise(rows, run)
    except ImportError:
        summary = 'pandas is not installed; see results.csv.\n'
    with open(os.path.join(outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(summary)
    return outdir, rows


# ── summary ──────────────────────────────────────────────────────────────

def summarise(rows, run):
    """Descriptive markdown summary of one Tier 2 run. Requires pandas."""
    import pandas as pd
    df = pd.DataFrame(rows)
    info = run['dataset_info']
    lines = [f"# Tier 2: {info['title']} ({run['preset']})", '',
             f"Commit {run['commit']} | {len(run['cases'])} cases | {run['rows']} rows | "
             f"wall time {run['wall_time_s'] / 60:.1f} min | seed {run['seed']}", '',
             f"Licence {info['licence']}. Labels: {info['labels']}.", '',
             'Reference surface: canonical reconstruction of the expert mask. Descriptive only; '
             'distances in mm; mean ± SD across cases.', '']
    if 'slice_mm' in df:
        lines += [f"Slice spacing across cases: {df['slice_mm'].min():g}–{df['slice_mm'].max():g} mm.", '']

    def msd(s, d=3):
        s = pd.to_numeric(s, errors='coerce').dropna()
        return '—' if s.empty else (f'{s.mean():.{d}f} ± {s.std():.{d}f}' if len(s) > 1 else f'{s.mean():.{d}f}')

    mask = df[(df.get('level') == 'mask') & (df.get('status') == 'ok')] if 'level' in df else df.iloc[0:0]
    if len(mask):
        lines += ['## Segmentation (mask level, Exp. A)', '',
                  '| structure | engine | cases | Dice | IoU | HD95 | ASSD |', '|---|---|---|---|---|---|---|']
        for (s, e), g in mask.groupby(['structure', 'source']):
            lines.append(f'| {s} | {e} | {len(g)} | {msd(g.dice)} | {msd(g.iou)} | '
                         f'{msd(g.mask_hd95_mm, 2)} | {msd(g.mask_assd_mm, 2)} |')
        lines.append('')

    mesh = df[(df.get('level') == 'mesh') & (df.get('status') == 'ok')] if 'level' in df else df.iloc[0:0]
    if len(mesh):
        lines += ['## End to end (mesh against reference surface, Exp. B and C)', '',
                  'Source `gt` is the expert mask itself: its error is reconstruction alone. '
                  'Canonical on `gt` reproduces the reference, so it should read 0.', '',
                  '| structure | source | config | ASSD | HD95 | mean signed | volume error % | closed + manifold |',
                  '|---|---|---|---|---|---|---|---|']
        for (s, src), g in mesh.groupby(['structure', 'source']):
            for name, sub in (('canonical', g[g.is_canonical == True]),  # noqa: E712
                              ('production', g[g.config == 'production'])):
                if len(sub):
                    lines.append(f'| {s} | {src} | {name} | {msd(sub.assd_mm)} | {msd(sub.hd95_mm, 2)} | '
                                 f'{msd(sub.mean_signed_mm)} | {msd(sub.volume_err_pct, 1)} | '
                                 f'{sub.topology_ok.astype(bool).mean():.0%} |')
        lines.append('')

        grid = mesh[mesh.config == 'grid'].copy()
        if len(grid) and grid['blur_mm'].nunique() > 1:
            import numpy as np
            factors = ['source', 'blur_mm', 'taubin_iter', 'decimation', 'case']
            grid = grid[grid['assd_mm'] > 0]
            grid['log_assd'] = np.log(grid['assd_mm'])
            lines += ['## Share of log-ASSD variance (grid configurations, preview)', '',
                      'Main effects (eta squared); "case" is patient-to-patient variation. '
                      'Exact rows with zero error (canonical on the expert mask) are excluded. '
                      'Formal attribution uses mixed-effects models and Sobol indices.', '']
            for s, g in grid.groupby('structure'):
                y = g['log_assd']
                tot = float(((y - y.mean()) ** 2).sum())
                shares = {}
                for f in factors:
                    m = g.groupby(f)['log_assd'].agg(['mean', 'count'])
                    shares[f] = float((m['count'] * (m['mean'] - y.mean()) ** 2).sum()) / tot if tot else float('nan')
                rest = 1 - sum(shares.values())
                lines.append(f'- **{s}**: ' + ', '.join(f'{k} {v:.1%}' for k, v in shares.items())
                             + f', interactions + residual {rest:.1%}')
            lines.append('')
            lines += ['## Mean ASSD by setting (grid)', '']
            for f in ['blur_mm', 'taubin_iter', 'decimation']:
                piv = grid.pivot_table(index='source', columns=f, values='assd_mm', aggfunc='mean')
                lines.append(f'- **{f}**: ' + '; '.join(
                    f'{src}: ' + ', '.join(f'{c}→{piv.loc[src, c]:.3f}' for c in piv.columns)
                    for src in piv.index))
            lines.append('')

        lines += ['## Topology (closed and manifold, by decimation)', '']
        topo = mesh.groupby(['source', 'decimation'])['topology_ok'].apply(lambda x: x.astype(bool).mean()).unstack()
        lines += ['| source | ' + ' | '.join(str(c) for c in topo.columns) + ' |', '|---' * (len(topo.columns) + 1) + '|']
        for src, r in topo.iterrows():
            lines.append(f'| {src} | ' + ' | '.join('—' if pd.isna(v) else f'{v:.0%}' for v in r) + ' |')
        lines.append('')

    bad = df[df.get('status').isin(['empty_mask', 'no_surface'])] if 'status' in df else df.iloc[0:0]
    if len(bad):
        lines += [f'{len(bad)} rows had no usable surface: '
                  + ', '.join(f'{k}: {v}' for k, v in bad.groupby(['structure', 'source', 'status']).size().items()), '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dataset', choices=sorted(DATASETS))
    ap.add_argument('--data', help='dataset folder')
    ap.add_argument('--preset', choices=sorted(PRESETS), default='pilot')
    ap.add_argument('--engines', default='ts_fast,classical')
    ap.add_argument('--cases', type=int, default=None, help='use only the first N cases')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--resummarise', metavar='RUN_DIR', default=None)
    args = ap.parse_args()
    if args.resummarise:
        import pandas as pd
        rows = pd.read_csv(os.path.join(args.resummarise, 'results.csv')).to_dict('records')
        with open(os.path.join(args.resummarise, 'run.json'), encoding='utf-8') as fh:
            run = json.load(fh)
        with open(os.path.join(args.resummarise, 'summary.md'), 'w', encoding='utf-8') as fh:
            fh.write(summarise(rows, run))
        print(f'summary rebuilt -> {args.resummarise}')
        return
    if not (args.dataset and args.data):
        ap.error('--dataset and --data are required')
    outdir, rows = run_experiment(args.dataset, args.data, args.preset,
                                  tuple(e for e in args.engines.split(',') if e),
                                  args.workers, args.seed, args.tol, args.outdir, args.cases)
    print(f'{len(rows)} rows -> {outdir}')


if __name__ == '__main__':
    main()
