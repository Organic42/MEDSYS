"""Validation test 1: is the in-app accuracy estimate honest?

Runs the application's own meshing code (segmentation_pipeline.build_mesh,
with the accuracy estimate on) on the cached TotalSegmentator masks of every
Tier 2 scan (41 spleen, 41 pancreas; fast and full modes; native, 1.5 mm,
3 mm and 5 mm-slice grids), then compares each line of the accuracy card with
the delivered mesh's true error against the expert surface:

  resolution limit   0.177 x Delta on the expert mask itself (pancreas held
                     out of the fit), and the app's 0.14-0.21 x RMS spacing
                     range on the Tier 1 phantoms' exact surfaces;
                     also the card's "expect at least the resolution limit"
  smoothing          the shift the card measures on the mesh against the
                     change it actually caused in the error to the expert
                     surface
  segmentation       leave-one-scan-out: the card's median and middle half,
                     rebuilt without the scan, against the scan's true error
                     (and a 90% range, which the card does not show yet)
  watertight         the card's flag against an independent VTK check

Needs the engine masks that research/tier2.py cached under
output/research/tier2/cache/. Writes results.csv, run.json and summary.md to
output/research/card_validation/<run>/.

    python -m research.card_validation --workers 8
    python -m research.card_validation --resummarise output/research/card_validation/<run>
"""
import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research import tier2  # noqa: E402

DATA = {'msd_spleen': os.path.join(ROOT, 'data', 'msd', 'Task09_Spleen'),
        'msd_pancreas': os.path.join(ROOT, 'data', 'msd', 'Task07_Pancreas')}
TIER1 = os.path.join(ROOT, 'research', 'results', 'tier1_phantoms_20261002-222422_full', 'results.csv.gz')
SOURCES = ('ts_fast', 'ts_full')
LEVELS = ('native', 'iso1.5', 'iso3', 'slice5')
N_SAMPLES = 20_000                      # per side, as in the paper


def delta_added(spacing, grid):
    """Sampling a coarser grid adds to the native one (the paper's Delta)."""
    s, g = np.asarray(spacing, float), np.asarray(grid, float)
    return float(np.sqrt(np.mean(g ** 2 - s ** 2)))


def _vtk_closed(mesh):
    """Closed by VTK's own edge classification (no boundary, no non-manifold
    edges), independent of the edge counting in research/surface_metrics.py."""
    edges = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=True,
                                       feature_edges=False, manifold_edges=False)
    return edges.n_cells == 0


def run_case(task):
    import nibabel as nib

    import accuracy
    import config
    from reconstruction import ReconParams, extract_isosurface, mesh_arrays
    from research.surface_metrics import compare_meshes
    from segmentation_pipeline import build_mesh
    config.ACCURACY = True
    dataset, cid, case_i, label_path, seed = task
    structure, values = next(iter(tier2.DATASETS[dataset]['structures'].items()))
    lab = nib.load(label_path)
    spacing = tuple(float(z) for z in lab.header.get_zooms()[:3])
    gt = np.isin(np.asanyarray(lab.dataobj).astype(np.int16), values)
    (gt_c,), gt_start = tier2.crop_to([gt], 6)
    rv, rf = extract_isosurface(gt_c, spacing, ReconParams(),          # the reference surface
                                tuple(float(a) * s for a, s in zip(gt_start, spacing)))
    rows = []
    for li, level in enumerate(LEVELS):
        grid = tier2.grid_spacing(spacing, level)
        resampled = tier2.effective_resolution(spacing, level) != 'native'
        if level != 'native' and not resampled:
            continue                    # the scan is already this coarse: same as native
        base = {'dataset': dataset, 'structure': structure, 'case': cid, 'resolution': level,
                'spacing': 'x'.join(f'{s:g}' for s in spacing), 'grid': 'x'.join(f'{s:g}' for s in grid),
                'grid_max_mm': max(grid), 'rms_spacing_mm': accuracy.added_sampling(grid),
                'delta_added_mm': delta_added(spacing, grid)}
        if resampled:
            # Line 1 on real anatomy: the expert mask itself, coarsened
            m = tier2.resample_mask(gt, spacing, grid, tier2.profile_for(level))
            (mc,), st = tier2.crop_to([m], 2)
            v, f = extract_isosurface(mc, grid, ReconParams(), tuple(float(a) * g for a, g in zip(st, grid)))
            r = compare_meshes(v, f, rv, rf, np.random.default_rng([seed, case_i, li, 0]),
                               n_samples=N_SAMPLES, local=False)
            rows.append(dict(base, source='gt', canon_assd_mm=r['assd_mm']))
        for si, source in enumerate(SOURCES):
            path = tier2.cached_mask_path(dataset, cid, source, structure, level)
            mask = np.load(path)['mask'].astype(bool)
            row = dict(base, source=source)
            t0 = time.perf_counter()
            mesh = build_mesh(mask, grid, sigma=0.6, reduction=0.6, min_voxels=200)   # the app's call
            row['t_build_s'] = time.perf_counter() - t0
            if mesh is None or 'medsys_recon_assd_mm' not in mesh.field_data:
                rows.append(dict(row, status='no_mesh'))
                continue
            card = {k: float(mesh.field_data[f'medsys_{k}'][0]) for k in accuracy.FIELDS}
            again = build_mesh(mask, grid, sigma=0.6, reduction=0.6, min_voxels=200)      # same mask again
            card2 = {k: float(again.field_data[f'medsys_{k}'][0]) for k in accuracy.FIELDS}
            (av, af), (bv, bf) = mesh_arrays(mesh), mesh_arrays(again)
            row['rebuild_identical'] = bool(np.array_equal(av, bv) and np.array_equal(af, bf) and all(
                np.array_equal(card[k], card2[k], equal_nan=True) for k in card))
            lim = accuracy.resolution_limit(grid)
            ref = accuracy.engine_reference(source[3:], structure, grid)
            tv, tf = mesh_arrays(mesh)
            true = compare_meshes(tv, tf, rv, rf, np.random.default_rng([seed, case_i, li, 1 + si]),
                                  n_samples=N_SAMPLES, local=False)
            (mc,), st = tier2.crop_to([mask], 2)
            cv, cf = extract_isosurface(mc, grid, ReconParams(), tuple(float(a) * g for a, g in zip(st, grid)))
            canon = compare_meshes(cv, cf, rv, rf, np.random.default_rng([seed, case_i, li, 3 + si]),
                                   n_samples=N_SAMPLES, local=False)
            row.update(
                status='ok',
                card_resolution_mm=lim['estimate_mm'], card_resolution_lo_mm=lim['range_mm'][0],
                card_resolution_hi_mm=lim['range_mm'][1],
                card_recon_assd_mm=card['recon_assd_mm'], card_recon_p95_mm=card['recon_p95_mm'],
                card_recon_signed_mm=card['recon_signed_mm'], card_volume_change_pct=card['volume_change_pct'],
                card_watertight=bool(card['watertight']), card_pieces=int(card['pieces']),
                card_engine_level=ref['level'], card_engine_median_mm=ref['median_mm'],
                card_engine_q1_mm=ref['iqr_mm'][0], card_engine_q3_mm=ref['iqr_mm'][1],
                independent_manifold=_vtk_closed(mesh),
                true_assd_mm=true['assd_mm'], true_hd95_mm=true['hd95_mm'],
                true_signed_mm=true['mean_signed_mm'], true_volume_err_pct=true['volume_err_pct'],
                true_nsd_1mm=true['nsd'],
                canon_assd_mm=canon['assd_mm'], canon_signed_mm=canon['mean_signed_mm'],
                canon_volume_err_pct=canon['volume_err_pct'])
            rows.append(row)
    return rows


# ── analysis ─────────────────────────────────────────────────────────────

def _pct(x):
    return f'{100 * np.mean(x):.0f}%'


# The card as first shipped: one native reference per organ and mode, plus the
# change to 5 mm slices from the paper's Table II on 5 mm grids.
V1_THICK = {('fast', 'spleen'): 0.188, ('full', 'spleen'): 0.212, ('fast', 'pancreas'): 0.143,
            ('full', 'pancreas'): 0.232}
V1_SLOPE_LOW = 0.14
QS = ('p5', 'q1', 'median', 'q3', 'p95')


def level_values(g, level):
    """Canonical engine ASSD of every scan on one study grid. A scan already
    as coarse as the grid was not resampled, so its native value stands in."""
    native = g[g.resolution == 'native'].set_index('case').canon_assd_mm
    at = g[g.resolution == level].set_index('case').canon_assd_mm
    return at.combine_first(native) if level != 'native' else native


def loso(values, cid):
    """Quartiles and 5-95% range of the other scans."""
    return dict(zip(QS, np.percentile(values.drop(cid).to_numpy(), [5, 25, 50, 75, 95])))


def coverage(ok, card):
    """Per delivered mesh: the card's segmentation range, rebuilt without the
    scan, and whether the mesh's true error fell inside it."""
    import pandas as pd

    import accuracy
    rows = []
    for (organ, source), g in ok.groupby(['structure', 'source']):
        mode = source[3:]
        vals = {lv: level_values(g, lv) for lv in LEVELS}
        for _, r in g.iterrows():
            if card == 'v1':
                ref = loso(vals['native'], r.case)
                if r.resolution == 'slice5':
                    ref = {k: v + V1_THICK[(mode, organ)] for k, v in ref.items()}
            else:
                spacing = [float(x) for x in r.grid.split('x')]
                ref = loso(vals[accuracy.reference_level(spacing)], r.case)
            for what in ('true_assd_mm', 'canon_assd_mm'):
                y = r[what]
                rows.append({'organ': organ, 'mode': mode, 'level': r.resolution, 'what': what,
                             'mid': ref['q1'] <= y <= ref['q3'], 'p90': ref['p5'] <= y <= ref['p95'],
                             'above': y > ref['q3'], 'abs_err': abs(y - ref['median'])})
    return pd.DataFrame(rows)


def summarise(df, run):
    import pandas as pd
    ok = df[(df.source != 'gt') & (df.status == 'ok')].copy()
    gt = df[df.source == 'gt'].copy()
    L = [f"# Validation test 1: the in-app accuracy estimate\n",
         f"Run {run['started']}, commit {run.get('commit', '?')}. Application code: "
         f"`segmentation_pipeline.build_mesh` with the accuracy estimate on, applied to the "
         f"cached TotalSegmentator masks of {df.case.nunique()} expert-labelled scans "
         f"({', '.join(sorted(df.dataset.unique()))}); {len(ok)} delivered meshes. "
         f"True error = ASSD of the delivered mesh against the expert surface "
         f"({N_SAMPLES:,} points per side). Coverage is the share of scans inside a stated range.\n"]

    # 1. Resolution limit
    L.append('## 1. Resolution limit\n')
    L.append('**On real anatomy.** The expert mask coarsened to each grid, against its native surface; '
             'predicted 0.177 x Delta (Delta = the sampling the coarser grid adds). The spleen is the fit; '
             'the pancreas never entered it.\n')
    import accuracy
    lo, hi = accuracy.SLOPE_LOW, accuracy.SLOPE_HIGH
    L.append(f'| Organ | Grids | Mean abs. error (mm) | Inside +/-0.111 mm | Inside {lo}-{hi} x Delta |')
    L.append('|---|---|---|---|---|')
    gt['pred'] = 0.177 * gt.delta_added_mm
    for organ, g in gt.groupby('structure'):
        err = (g.canon_assd_mm - g.pred).abs()
        inside = (g.canon_assd_mm >= lo * g.delta_added_mm) & (g.canon_assd_mm <= hi * g.delta_added_mm)
        L.append(f"| {organ}{' (held out)' if organ == 'pancreas' else ' (fit)'} | {len(g)} | "
                 f"{err.mean():.3f} | {_pct(err <= 0.111)} | {_pct(inside)} |")
    if os.path.exists(TIER1):
        t1 = pd.read_csv(TIER1)
        t1 = t1[(t1.is_canonical == True) & (t1.status == 'ok')].copy()  # noqa: E712
        sp = t1.spacing.astype(str).str.split('x', expand=True).astype(float)
        t1['rms'] = np.sqrt((sp ** 2).mean(axis=1))
        inside = (t1.assd_mm >= lo * t1.rms) & (t1.assd_mm <= hi * t1.rms)
        v1 = (t1.assd_mm >= V1_SLOPE_LOW * t1.rms) & (t1.assd_mm <= hi * t1.rms)
        L.append(f"\n**On exact surfaces (Tier 1 phantoms).** The card's range, {lo}-{hi} x RMS spacing, "
                 f"held for {_pct(inside)} of {len(t1)} canonical meshes; by shape: "
                 + ', '.join(f"{s} {_pct(g)}" for s, g in inside.groupby(t1['shape'])) + '. '
                 f"As first shipped ({V1_SLOPE_LOW}-{hi}) it held {_pct(v1)}: isotropic grids sit right at "
                 f"0.14 x spacing, half of them just below. (The range comes from these data, so this is "
                 f"consistency, not prediction.)\n")
    below = ok.true_assd_mm >= ok.card_resolution_mm
    L.append(f"**\"Expect at least the resolution limit.\"** The delivered mesh's true error was at "
             f"least the card's resolution limit for {_pct(below)} of {len(ok)} meshes "
             f"(lowest ratio {np.min(ok.true_assd_mm / ok.card_resolution_mm):.2f}x).\n")

    # 2. Smoothing and simplification
    L.append('## 2. Smoothing and simplification\n')
    change = ok.true_assd_mm - ok.canon_assd_mm
    shift_signed = ok.true_signed_mm - ok.canon_signed_mm
    L.append(f"The card measures each mesh's shift from its own raw surface: median "
             f"{ok.card_recon_assd_mm.median():.3f} mm (range {ok.card_recon_assd_mm.min():.3f}-"
             f"{ok.card_recon_assd_mm.max():.3f}). Smoothing changed the true error by "
             f"{change.median():+.3f} mm (median; range {change.min():+.3f} to {change.max():+.3f}); the "
             f"change stayed within the card's shift for {_pct(change.abs() <= ok.card_recon_assd_mm)} of meshes. "
             f"The card's signed shift (shrinkage) matched the change in signed distance to the expert "
             f"surface within {(ok.card_recon_signed_mm - shift_signed).abs().median():.3f} mm "
             f"(median; 95th percentile {np.percentile((ok.card_recon_signed_mm - shift_signed).abs(), 95):.3f}).\n")

    # 3. Segmentation
    L.append('## 3. Segmentation (leave one scan out)\n')
    L.append("For each delivered mesh, the card's reference (canonical engine ASSD of the *other* 40 scans) "
             "is rebuilt and the mesh's true error is checked against it. The card shows the median and the "
             "middle half, so about 50% is the honest target there, and about 90% for the 5th-95th percentile "
             "range. **v1** is the card as first shipped: the native reference, plus the paper's thick-slice "
             "change on 5 mm grids. **v2** matches the reference to the scan's grid "
             "(`accuracy.reference_level`) and adds the 90% range.\n")
    v1, v2 = coverage(ok, 'v1'), coverage(ok, 'v2')
    L.append('| Organ | Mode | Grid | Scans | v1 middle half | v1 90% range | v2 middle half | v2 90% range | '
             'v2 median abs. error (mm) |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    a1, a2 = v1[v1.what == 'true_assd_mm'], v2[v2.what == 'true_assd_mm']
    cov_rows = []
    for (organ, mode, level), a in a1.groupby(['organ', 'mode', 'level'], sort=False):
        b = a2[(a2.organ == organ) & (a2['mode'] == mode) & (a2.level == level)]
        cov_rows.append({'organ': organ, 'mode': mode, 'level': level, 'n': len(a), 'v1_mid': a.mid.mean(),
                         'v1_p90': a.p90.mean(), 'v2_mid': b.mid.mean(), 'v2_p90': b.p90.mean()})
        L.append(f"| {organ} | {mode} | {level} | {len(a)} | {_pct(a.mid)} | {_pct(a.p90)} | {_pct(b.mid)} | "
                 f"{_pct(b.p90)} | {b.abs_err.median():.3f} |")
    cov = pd.DataFrame(cov_rows)
    w1, w2 = cov.sort_values('v1_mid').iloc[0], cov.sort_values('v2_mid').iloc[0]
    c2 = v2[v2.what == 'canon_assd_mm']
    L.append(f"\nAll {len(a1)} meshes: v1 middle half {_pct(a1.mid)} (target 50%), 90% range {_pct(a1.p90)} "
             f"(target 90%), {_pct(a1.above)} above the middle half; v2 {_pct(a2.mid)}, {_pct(a2.p90)}, "
             f"{_pct(a2.above)} above. Worst cell: v1 {100 * w1.v1_mid:.0f}% ({w1.organ}, {w1['mode']}, "
             f"{w1.level}); v2 {100 * w2.v2_mid:.0f}% ({w2.organ}, {w2['mode']}, {w2.level}). "
             f"Against the segmentation alone (canonical meshes, the line's exact claim), v2 covers "
             f"{_pct(c2.mid)} and {_pct(c2.p90)}.\n")
    # The shipped reference applied to the wrong organ: why the card says "not validated" elsewhere
    sp_ref = ok[(ok.structure == 'spleen') & (ok.resolution == 'native')]
    pa = ok[(ok.structure == 'pancreas') & (ok.resolution == 'native')]
    if len(sp_ref) and len(pa):
        lo, hi = np.percentile(sp_ref.canon_assd_mm, [5, 95])
        L.append(f"**Why other organs say \"not validated\".** The spleen's 90% range "
                 f"({lo:.2f}-{hi:.2f} mm) would have covered only {_pct((pa.true_assd_mm >= lo) & (pa.true_assd_mm <= hi))} "
                 f"of pancreas meshes.\n")

    # 4. Watertight
    agree = ok.card_watertight == ok.independent_manifold
    L.append('## 4. Watertight flag\n')
    L.append(f"The card's flag agreed with VTK's independent manifold check for {_pct(agree)} of {len(ok)} meshes "
             f"({ok.card_watertight.sum()} watertight).\n")

    if 'rebuild_identical' in ok:
        L.append('## 5. Same mask, same mesh\n')
        L.append(f"Meshing each mask a second time gave an identical mesh (vertices and faces) and identical card "
                 f"values for {int(ok.rebuild_identical.sum())} of {len(ok)} meshes.\n")

    L.append('## Timing\n')
    L.append(f"Median build time with the estimate: {ok.t_build_s.median():.2f} s per mesh.\n")
    return '\n'.join(L), cov


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cases', type=int, default=None, help='first N cases per organ (smoke test)')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--resummarise', metavar='RUN_DIR', default=None)
    args = ap.parse_args()
    import pandas as pd
    if args.resummarise:
        df = pd.read_csv(tier2.results_file(args.resummarise))
        with open(os.path.join(args.resummarise, 'run.json'), encoding='utf-8') as fh:
            run = json.load(fh)
        text, cov = summarise(df, run)
        with open(os.path.join(args.resummarise, 'summary.md'), 'w', encoding='utf-8') as fh:
            fh.write(text)
        cov.to_csv(os.path.join(args.resummarise, 'coverage.csv'), index=False)
        print(text)
        return
    import subprocess
    if os.name == 'nt':                 # keep the desktop responsive; workers inherit it
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    outdir = args.outdir or os.path.join(ROOT, 'output', 'research', 'card_validation', stamp)
    os.makedirs(outdir, exist_ok=True)
    tasks = []
    for dataset, root in DATA.items():
        with open(os.path.join(ROOT, 'research', 'results', 'tier2_pancreas_heldout_20261018-223006' if
                               dataset == 'msd_pancreas' else 'tier2_spleen_two_modes_20261018-210930',
                               'run.json'), encoding='utf-8') as fh:
            used = set(json.load(fh)['cases'])
        cases = [c for c in tier2.list_cases(dataset, root) if c[0] in used][:args.cases]
        tasks += [(dataset, cid, n, lbl, args.seed) for cid, n, _, lbl in cases]
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = '?'
    run = {'test': 'card_validation', 'started': stamp, 'commit': commit + ' (+ uncommitted changes)',
           'cases': len(tasks), 'sources': SOURCES, 'levels': LEVELS, 'n_samples': N_SAMPLES,
           'seed': args.seed, 'python': platform.python_version()}
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows += fut.result()
            print(f'{i}/{len(tasks)} scans  {time.time() - t0:.0f} s', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'results.csv'), index=False)
    run['seconds'] = round(time.time() - t0)
    with open(os.path.join(outdir, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump(run, fh, indent=1)
    text, cov = summarise(df, run)
    with open(os.path.join(outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    cov.to_csv(os.path.join(outdir, 'coverage.csv'), index=False)
    print(text)
    print(f'-> {outdir}')


if __name__ == '__main__':
    main()
