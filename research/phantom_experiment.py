"""
Tier 1 phantom experiment: reconstruction error against exact surfaces.

Voxelises each phantom at each spacing, reconstructs it under every research
configuration (blur in mm x isosurface x Taubin iterations x decimation), plus
the production defaults, and scores every mesh against the exact surface.
The canonical configuration (no blur, Lewiner marching cubes, no smoothing,
no decimation) is the reference-surface procedure; its error is the floor
below which Tier 2 differences cannot be interpreted.

Usage (from the repository root):
  python -m research.phantom_experiment --preset smoke
  python -m research.phantom_experiment --preset pilot
  python -m research.phantom_experiment --preset full --workers 5

Outputs go to output/research/phantoms/<timestamp>_<preset>/:
  results.csv   one row per mesh
  run.json      provenance: commit, versions, grid, timing
  summary.md    descriptive summary (not the formal RQ1 analysis)

Accuracy only: timings are recorded for planning, not as a cost benchmark.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
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
from research.phantoms import default_phantoms, random_placement, sample_surface, voxelise  # noqa: E402
from research.surface_metrics import compare_to_phantom  # noqa: E402

# Research grid, fixed before analysis (proposal v3.2, Exp. B).
GRID = {
    'blur_mm': (0.0, 0.5, 1.0),
    'isosurface': ('lewiner', 'lorensen'),
    'taubin_iter': (0, 10, 30, 60),
    'decimation': (0.0, 0.25, 0.5, 0.75),
}
CANONICAL = {'blur_mm': 0.0, 'isosurface': 'lewiner', 'taubin_iter': 0, 'decimation': 0.0}
# build_mesh() defaults in segmentation_pipeline.py. Blur is in voxels there;
# call sites override these per structure.
PRODUCTION = {'blur_vox': 0.6, 'isosurface': 'lewiner', 'taubin_iter': 30, 'decimation': 0.5}

SPACINGS = ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (0.8, 0.8, 5.0))
PLACEMENT_OFFSET_MM = 5.0   # >= the coarsest spacing, so offsets cover every sub-voxel position

PRESETS = {
    'smoke': {'shapes': ('sphere',), 'spacings': ((1.0, 1.0, 1.0),), 'placements': 1,
              'grid': 'canonical', 'samples': 3000},
    'pilot': {'shapes': None, 'spacings': SPACINGS, 'placements': 1,
              'grid': 'full', 'samples': 20000},
    'full': {'shapes': None, 'spacings': SPACINGS, 'placements': 5,
             'grid': 'full', 'samples': 20000},
}


def spacing_label(spacing):
    if len(set(spacing)) == 1:
        return f"{spacing[0]:g}"
    return 'x'.join(f"{s:g}" for s in spacing)


def research_configs(grid):
    """Configurations to run for one (shape, spacing, placement) unit."""
    if grid == 'full':
        combos = [dict(zip(GRID, values)) for values in product(*GRID.values())]
    elif grid == 'canonical':
        combos = [dict(CANONICAL)]
    else:
        raise ValueError(f"unknown grid {grid!r}")
    configs = [dict(c, config='grid', blur_vox=None) for c in combos]
    configs.append(dict(PRODUCTION, config='production', blur_mm=None))
    return configs


def _params(cfg, spacing):
    blur = (cfg['blur_vox'],) * 3 if cfg['blur_vox'] is not None else blur_mm_to_vox(cfg['blur_mm'], spacing)
    return ReconParams(blur_sigma_vox=blur, isosurface=cfg['isosurface'],
                       taubin_iter=cfg['taubin_iter'], decimation=cfg['decimation'])


def _is_canonical(cfg):
    return cfg['config'] == 'grid' and all(cfg[k] == v for k, v in CANONICAL.items())


def run_unit(task):
    """Reconstruct and score every configuration for one shape, spacing and placement."""
    seed, shape_i, spacing_i, place_i, spacing, grid, n_samples, tol_mm = task
    phantom = default_phantoms()[shape_i]
    placement = random_placement(np.random.default_rng([seed, shape_i, place_i]),
                                 max_offset_mm=PLACEMENT_OFFSET_MM)
    truth = sample_surface(phantom, placement, n_samples,
                           np.random.default_rng([seed, shape_i, place_i, 1]))
    mask, origin = voxelise(phantom, placement, spacing)

    configs = research_configs(grid)
    # Isosurface extraction depends only on blur and method: do it once per pair.
    iso_cache = {}
    rows = []
    for cfg_i, cfg in enumerate(configs):
        params = _params(cfg, spacing)
        base = {
            'shape': phantom.name, 'spacing': spacing_label(spacing),
            'spacing_max_mm': max(spacing), 'anisotropic': len(set(spacing)) > 1,
            'placement': place_i, 'config': cfg['config'], 'is_canonical': _is_canonical(cfg),
            'blur_mm': cfg['blur_mm'], 'blur_vox': cfg['blur_vox'],
            'isosurface': cfg['isosurface'], 'taubin_iter': cfg['taubin_iter'],
            'decimation': cfg['decimation'],
        }
        key = (params.blur_sigma_vox, params.isosurface)
        t0 = time.perf_counter()
        if key not in iso_cache:
            try:
                iso_cache[key] = extract_isosurface(mask, spacing, params, origin)
            except (ValueError, RuntimeError) as exc:
                iso_cache[key] = exc
        iso = iso_cache[key]
        t_iso = time.perf_counter() - t0
        if isinstance(iso, Exception):
            # e.g. heavy blur lifts a thin structure's peak below the 0.5 level
            rows.append(dict(base, status='no_surface', t_iso_s=t_iso))
            continue
        t1 = time.perf_counter()
        verts, faces = mesh_arrays(postprocess_surface(iso[0], iso[1], params))
        t_post = time.perf_counter() - t1
        t2 = time.perf_counter()
        metrics = compare_to_phantom(
            verts, faces, phantom, placement, truth,
            np.random.default_rng([seed, shape_i, place_i, spacing_i, cfg_i, 2]),
            n_mesh_samples=n_samples, tol_mm=tol_mm)
        t_metric = time.perf_counter() - t2
        rows.append(dict(base, status='ok', **metrics, t_iso_s=t_iso,
                         t_post_s=t_post, t_metric_s=t_metric))
    return rows


def _provenance():
    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ''
    versions = {'python': platform.python_version()}
    for mod in ('numpy', 'scipy', 'skimage', 'pyvista', 'vtk'):
        try:
            m = __import__(mod)
            versions[mod] = (m.vtkVersion.GetVTKVersion() if mod == 'vtk'
                             else getattr(m, '__version__', '?'))
        except Exception:
            versions[mod] = None
    commit = git('rev-parse', '--short', 'HEAD')
    return {'commit': commit + ('+dirty' if git('status', '--porcelain') else ''),
            'versions': versions, 'platform': platform.platform(),
            'cpu_count': os.cpu_count()}


def run_experiment(preset='pilot', workers=None, seed=2026, tol_mm=1.0, outdir=None,
                   samples=None):
    cfg = PRESETS[preset]
    phantoms = default_phantoms()
    shape_idx = [i for i, p in enumerate(phantoms)
                 if cfg['shapes'] is None or p.name in cfg['shapes']]
    n_samples = samples or cfg['samples']
    tasks = [(seed, si, pi, pl, cfg['spacings'][pi], cfg['grid'], n_samples, tol_mm)
             for si in shape_idx for pi in range(len(cfg['spacings']))
             for pl in range(cfg['placements'])]

    run_id = time.strftime('%Y%m%d-%H%M%S') + f"_{preset}"
    out_root = os.environ.get('VRSEG_OUTPUT') or os.path.join(ROOT, 'output')
    outdir = outdir or os.path.join(out_root, 'research', 'phantoms', run_id)
    os.makedirs(outdir, exist_ok=True)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t_start = time.perf_counter()
    rows = []
    if workers == 1:
        for t in tasks:
            rows.extend(run_unit(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_unit, t) for t in tasks]
            for done, fut in enumerate(as_completed(futures), 1):
                rows.extend(fut.result())
                print(f"  unit {done}/{len(tasks)} done", flush=True)
    wall = time.perf_counter() - t_start

    rows.sort(key=lambda r: (r['shape'], r['spacing_max_mm'], r['spacing'], r['placement']))
    fields = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with open(os.path.join(outdir, 'results.csv'), 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    run = {
        'preset': preset, 'seed': seed, 'nsd_tolerance_mm': tol_mm, 'samples_per_side': n_samples,
        'grid': GRID if cfg['grid'] == 'full' else {'canonical_only': CANONICAL},
        'canonical': CANONICAL, 'production': PRODUCTION,
        'spacings_mm': [list(s) for s in cfg['spacings']], 'placements': cfg['placements'],
        'placement_offset_mm': PLACEMENT_OFFSET_MM,
        'phantoms': [{'name': phantoms[i].name, 'class': type(phantoms[i]).__name__,
                      'params': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in vars(phantoms[i]).items()},
                      'volume_mm3': phantoms[i].volume, 'area_mm2': phantoms[i].area,
                      'genus': phantoms[i].genus} for i in shape_idx],
        'meshes': len(rows), 'units': len(tasks), 'workers': workers,
        'wall_time_s': round(wall, 1), **_provenance(),
    }
    with open(os.path.join(outdir, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump(run, fh, indent=2, default=str)

    try:
        summary = summarise(rows, run)
    except ImportError:
        summary = "pandas is not installed; see results.csv.\n"
    with open(os.path.join(outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(summary)
    return outdir, rows


def summarise(rows, run):
    """Descriptive markdown summary of one run. Requires pandas."""
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df['status'] == 'ok']
    order = [spacing_label(tuple(s)) for s in run['spacings_mm']]
    lines = [
        f"# Tier 1 phantom experiment: {run['preset']}",
        "",
        f"Commit {run['commit']} | {run['meshes']} meshes | {run['units']} units | "
        f"{run['workers']} workers | wall time {run['wall_time_s'] / 60:.1f} min | seed {run['seed']}",
        "",
        "Descriptive summary only. The formal RQ1 attribution (mixed-effects model and "
        "Sobol indices) is a later step. Distances in mm; surface samples per side: "
        f"{run['samples_per_side']}; NSD tolerance {run['nsd_tolerance_mm']} mm.",
        "",
    ]

    canon = ok[ok['is_canonical']]
    prod = ok[ok['config'] == 'production']
    lines += ["## Reference floor", "",
              "Canonical configuration (no blur, Lewiner marching cubes, no smoothing, no "
              "decimation): the reference-surface procedure. Cells: ASSD (HD95), mean over "
              "placements; columns are voxel spacings in mm.", ""]
    if len(canon):
        assd = canon.pivot_table(index='shape', columns='spacing', values='assd_mm', aggfunc='mean')
        hd = canon.pivot_table(index='shape', columns='spacing', values='hd95_mm', aggfunc='mean')
        cols = [c for c in order if c in assd.columns]
        lines += ["| shape | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for shape in assd.index:
            lines.append(f"| {shape} | " + " | ".join(
                f"{assd.loc[shape, c]:.3f} ({hd.loc[shape, c]:.2f})" for c in cols) + " |")
        lines.append("")

    if len(prod) and len(canon):
        lines += ["## Production defaults versus canonical", "",
                  "Production = `build_mesh` defaults (blur 0.6 voxels, 30 Taubin iterations, "
                  "0.5 decimation). Cells: ASSD canonical -> production; signed = production "
                  "mean signed distance (negative = shrinkage).", ""]
        c_assd = canon.pivot_table(index='shape', columns='spacing', values='assd_mm', aggfunc='mean')
        p_assd = prod.pivot_table(index='shape', columns='spacing', values='assd_mm', aggfunc='mean')
        p_sign = prod.pivot_table(index='shape', columns='spacing', values='mean_signed_mm', aggfunc='mean')
        cols = [c for c in order if c in c_assd.columns and c in p_assd.columns]
        lines += ["| shape | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for shape in c_assd.index:
            cells = []
            for c in cols:
                cells.append(f"{c_assd.loc[shape, c]:.3f} -> {p_assd.loc[shape, c]:.3f} "
                             f"(signed {p_sign.loc[shape, c]:+.2f})")
            lines.append(f"| {shape} | " + " | ".join(cells) + " |")
        lines.append("")

    grid = ok[ok['config'] == 'grid'].copy()
    factors = ['spacing', 'shape', 'blur_mm', 'isosurface', 'taubin_iter', 'decimation']
    if grid['blur_mm'].nunique() > 1:
        import numpy as np
        grid['log_assd'] = np.log(grid['assd_mm'])
        replicated = run['placements'] > 1

        def shares(col):
            y = grid[col]
            ss_total = float(((y - y.mean()) ** 2).sum())
            out = {}
            for f in factors:
                means = grid.groupby(f)[col].agg(['mean', 'count'])
                out[f] = float((means['count'] * (means['mean'] - y.mean()) ** 2).sum()) / ss_total
            cell = grid.groupby(factors)[col].transform('mean')
            out['replication'] = float(((y - cell) ** 2).sum()) / ss_total
            out['interactions'] = 1 - sum(out.values())
            return out

        raw, logged = shares('assd_mm'), shares('log_assd')
        failed = int((df['config'] == 'grid').sum() - len(grid))
        lines += ["## Share of ASSD variance by source (preview)", "",
                  "Main effects are eta squared; replication is placement-to-placement "
                  "variation within identical settings; interactions are the rest. ASSD "
                  "scales roughly multiplicatively with voxel size, so the log column is "
                  "the fairer view of interactions. Shares depend on the factor ranges "
                  "chosen, not on the shapes alone. Exact only for a balanced grid"
                  + (f"; {failed} grid meshes failed, so treat as approximate." if failed else "; this grid is balanced."),
                  "", "| source | raw ASSD | log ASSD |", "|---|---|---|"]
        for f in factors + ['replication', 'interactions']:
            if f == 'replication' and not replicated:
                lines.append("| replication | n/a (1 placement) | n/a |")
                continue
            lines.append(f"| {f} | {raw[f]:.1%} | {logged[f]:.1%} |")
        lines.append("")

        lines += ["## Mean ASSD by factor level (grid configurations)", ""]
        for f in factors[2:]:
            means = grid.groupby(f)['assd_mm'].mean()
            lines.append(f"- **{f}**: " + ", ".join(f"{k} -> {v:.3f}" for k, v in means.items()))
        lines.append("")

    lines += ["## Topology", "",
              "Share of meshes that are closed, a single component and of the expected genus.", ""]
    topo = ok.groupby(['isosurface', 'spacing'])['topology_ok'].mean().unstack()
    cols = [c for c in order if c in topo.columns]
    lines += ["| isosurface | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for iso, row in topo.iterrows():
        lines.append(f"| {iso} | " + " | ".join(f"{row[c]:.0%}" for c in cols) + " |")
    bad = ok[~ok['topology_ok'].astype(bool)]
    if len(bad):
        open_mesh = bad[~bad['closed'].astype(bool)]
        lines += ["", f"Failure modes among {len(bad)} meshes (one mesh can have several): "
                  f"{len(open_mesh)} open (boundary edges), "
                  f"{int((bad['components'] > 1).sum())} split into several components, "
                  f"{int((bad['closed'].astype(bool) & (bad['components'] == 1)).sum())} closed "
                  f"but wrong genus. Open meshes with no decimation: "
                  f"{int((open_mesh['decimation'] == 0).sum())}."]
    no_surf = df[df['status'] == 'no_surface']
    lines.append("")
    if len(no_surf):
        counts = no_surf.groupby(['shape', 'spacing']).size()
        lines.append("Configurations that produced no surface at all: "
                     + ", ".join(f"{s} @ {sp}: {n}" for (s, sp), n in counts.items()) + ".")
    else:
        lines.append("Every configuration produced a surface.")
    lines.append("")

    lines += ["## Timing (planning only, not a cost benchmark)", "",
              f"Mean per mesh: isosurface {ok['t_iso_s'].mean():.3f} s, smoothing and decimation "
              f"{ok['t_post_s'].mean():.3f} s, metrics {ok['t_metric_s'].mean():.3f} s.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--preset', choices=sorted(PRESETS), default='pilot')
    ap.add_argument('--workers', type=int, default=None,
                    help='processes (default: CPU count - 1; 1 = run in-process)')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--tol', type=float, default=1.0, help='NSD tolerance in mm')
    ap.add_argument('--samples', type=int, default=None, help='surface samples per side')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--resummarise', metavar='RUN_DIR', default=None,
                    help='rebuild summary.md from an existing run folder and exit')
    args = ap.parse_args()
    if args.resummarise:
        import pandas as pd
        rows = pd.read_csv(os.path.join(args.resummarise, 'results.csv')).to_dict('records')
        with open(os.path.join(args.resummarise, 'run.json'), encoding='utf-8') as fh:
            run = json.load(fh)
        with open(os.path.join(args.resummarise, 'summary.md'), 'w', encoding='utf-8') as fh:
            fh.write(summarise(rows, run))
        print(f"summary rebuilt -> {args.resummarise}")
        return
    outdir, rows = run_experiment(args.preset, args.workers, args.seed, args.tol,
                                  args.outdir, args.samples)
    print(f"{len(rows)} meshes -> {outdir}")


if __name__ == '__main__':
    main()
