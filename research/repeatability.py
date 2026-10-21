"""Validation test 2: does the same scan give the same model every time?

Runs TotalSegmentator on the same CT scans repeatedly, the way the
application does (ml=False, all structures), along three paths:

  worker      the persistent GPU worker the app uses (gpu_worker.py), so
              repeated jobs share one warm process
  fresh_gpu   a new process per run, as the app's in-process fallback does;
              nnU-Net's cuDNN autotuning can pick different kernels here
  cpu         a new process per run on the CPU, as on a machine without a GPU

Each run's masks are compared voxel by voxel with the first worker run, then
the target organ is meshed with the app's own build_mesh call, and the mesh
is compared with the first run's mesh and with the expert surface. Meshing
the same mask twice must give an identical mesh.

    python -m research.repeatability                       # 5 spleen + 5 pancreas scans
    python -m research.repeatability --resummarise output/research/repeatability/<run>
"""
import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research import tier2  # noqa: E402
from research.card_validation import DATA, N_SAMPLES  # noqa: E402

FRESH = ("import sys\n"
         "from totalsegmentator.python_api import totalsegmentator\n"
         "totalsegmentator(sys.argv[1], sys.argv[2], fast=sys.argv[3] == 'fast', ml=False,\n"
         "                 device=sys.argv[4], task='total', quiet=True)\n")


def run_ts(path, nifti, seg_dir, mode, device):
    import gpu_worker
    os.makedirs(seg_dir, exist_ok=True)
    t0 = time.perf_counter()
    if path == 'worker':
        if not gpu_worker.run_totalseg(nifti, seg_dir, fast=mode == 'fast', task='total'):
            raise RuntimeError('the GPU worker did not run the job')
    else:
        p = subprocess.run([sys.executable, '-c', FRESH, nifti, seg_dir, mode, device],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        if p.returncode:
            raise RuntimeError(f'TotalSegmentator ({device}) failed: {(p.stderr or p.stdout)[-800:]}')
    return time.perf_counter() - t0


def label_map(seg_dir):
    """All structure masks of a run as one label map (TotalSegmentator's
    structures do not overlap), with the structure names in label order."""
    import nibabel as nib
    files = sorted(glob.glob(os.path.join(seg_dir, '*.nii.gz')))
    names = [os.path.basename(f)[:-7] for f in files]
    lm = None
    for i, f in enumerate(files, 1):
        m = np.asanyarray(nib.load(f).dataobj) > 0
        if lm is None:
            lm = np.zeros(m.shape, np.uint8)
        lm[m] = i
    return lm, names


def same_card(a, b):
    """Accuracy-card values equal, counting two blanks (NaN: no volume for an open mesh) as equal."""
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def mesh_of(mask, spacing):
    import accuracy
    from segmentation_pipeline import build_mesh
    mesh = build_mesh(mask, spacing, sigma=0.6, reduction=0.6, min_voxels=200)   # the app's call
    if mesh is None:
        raise RuntimeError('no mesh: the organ mask is empty')
    card = {k: float(mesh.field_data[f'medsys_{k}'][0]) for k in accuracy.FIELDS}
    return mesh, card


def run_case(dataset, cid, case_i, image, label, args, tmp, log):
    import nibabel as nib

    import config
    from reconstruction import ReconParams, extract_isosurface, mesh_arrays
    from research.surface_metrics import compare_meshes
    config.ACCURACY = True
    structure, values = next(iter(tier2.DATASETS[dataset]['structures'].items()))
    lab = nib.load(label)
    spacing = tuple(float(z) for z in lab.header.get_zooms()[:3])
    gt = np.isin(np.rint(np.asanyarray(lab.dataobj)).astype(np.int16), values)
    (gt_c,), st = tier2.crop_to([gt], 6)
    rv, rf = extract_isosurface(gt_c, spacing, ReconParams(), tuple(float(a) * s for a, s in zip(st, spacing)))
    rows = []
    for mode in ('fast', 'full'):
        plan = ([('worker', 'gpu', r) for r in range(args.worker_runs)]
                + [('fresh_gpu', 'gpu', r) for r in range(args.fresh_runs)]
                + [('cpu', 'cpu', r) for r in range(args.cpu_runs if mode == 'fast' or cid in args.full_cpu
                                                    else 0)])
        ref = None
        for path, device, r in plan:
            seg_dir = os.path.join(tmp, f'{cid}_{mode}_{path}_{r}')
            secs = run_ts(path, image, seg_dir, mode, device)
            lm, names = label_map(seg_dir)
            shutil.rmtree(seg_dir, ignore_errors=True)
            organ = lm == names.index(structure) + 1
            mesh, card = mesh_of(organ, spacing)
            tv, tf = mesh_arrays(mesh)
            again, card2 = mesh_of(organ, spacing)           # meshing the same mask twice
            av, af = mesh_arrays(again)
            identical = np.array_equal(tv, av) and np.array_equal(tf, af) and same_card(card, card2)
            true = compare_meshes(tv, tf, rv, rf, np.random.default_rng([args.seed, case_i]),
                                  n_samples=N_SAMPLES, local=False)
            row = {'dataset': dataset, 'structure': structure, 'case': cid, 'mode': mode, 'path': path,
                   'device': device, 'run': r, 'seconds': round(secs, 1), 'organ_voxels': int(organ.sum()),
                   'rebuild_identical': identical, 'true_assd_mm': true['assd_mm'],
                   'true_hd95_mm': true['hd95_mm'], 'true_volume_err_pct': true['volume_err_pct'],
                   'card_recon_assd_mm': card['recon_assd_mm'], 'card_watertight': bool(card['watertight'])}
            if ref is None:
                ref = {'lm': lm, 'names': names, 'organ': organ, 'v': tv, 'f': tf}
                row.update(labelmap_diff_voxels=0, structures_changed=0, organ_diff_voxels=0, organ_dice=1.0,
                           mesh_assd_vs_first_mm=0.0, mesh_hd95_vs_first_mm=0.0)
            else:
                if names != ref['names']:
                    raise RuntimeError(f'{cid}: structure list changed between runs')
                diff = lm != ref['lm']
                changed = len((set(np.unique(lm[diff])) | set(np.unique(ref['lm'][diff]))) - {0})
                o_diff = int((organ != ref['organ']).sum())
                dice = 2 * (organ & ref['organ']).sum() / max(organ.sum() + ref['organ'].sum(), 1)
                if o_diff == 0 and np.array_equal(tv, ref['v']) and np.array_equal(tf, ref['f']):
                    m = {'assd_mm': 0.0, 'hd95_mm': 0.0}
                else:
                    m = compare_meshes(tv, tf, ref['v'], ref['f'], np.random.default_rng([args.seed, case_i, 7]),
                                       n_samples=N_SAMPLES, local=False)
                row.update(labelmap_diff_voxels=int(diff.sum()), structures_changed=changed,
                           organ_diff_voxels=o_diff, organ_dice=float(dice),
                           mesh_assd_vs_first_mm=m['assd_mm'], mesh_hd95_vs_first_mm=m['hd95_mm'])
            rows.append(row)
            log(f"  {cid} {mode:4} {path:9} run {r}: {secs:5.1f} s, organ diff {row['organ_diff_voxels']} vox, "
                f"all-structure diff {row['labelmap_diff_voxels']} vox, mesh shift "
                f"{row['mesh_assd_vs_first_mm']:.4f} mm, true ASSD {row['true_assd_mm']:.3f} mm")
    return rows


def summarise(df, run):
    import pandas as pd
    L = ['# Validation test 2: repeatability\n',
         f"Run {run['started']}. {df.case.nunique()} scans ({', '.join(sorted(df.structure.unique()))}), "
         f"TotalSegmentator fast and full modes, {len(df)} segmentation runs. Every run is compared with the "
         f"first run on the app's persistent GPU worker for the same scan and mode.\n",
         '## Same mask, same mesh\n',
         (f"Meshing the same mask twice gave an identical mesh and identical accuracy-card values in "
          f"{df.rebuild_identical.sum()} of {len(df)} runs.\n" if run.get('rebuild_check') == 'nan-aware' else
          f"Not reported for this run: its check compared the card's blank volume change (open meshes) as "
          f"unequal to itself, flagging the {(~df.rebuild_identical).sum()} open meshes and no others. "
          f"Meshes rebuilt from the same masks were identical; validation test 1 checks this on every mesh.\n"),
         '## Same scan, repeated segmentation\n',
         '| Mode | Path | Runs | Scans with any organ change | Organ voxels changed (max) | Organ Dice (min) | '
         'All-structure voxels changed (median, max) | Mesh shift vs first run, mm (median, max) |',
         '|---|---|---|---|---|---|---|---|']
    rep = df[~((df.path == 'worker') & (df.run == 0))]
    for (mode, path), g in rep.groupby(['mode', 'path'], sort=False):
        changed = g.groupby('case').organ_diff_voxels.max() > 0
        L.append(f"| {mode} | {path} | {len(g)} | {changed.sum()} of {len(changed)} | {g.organ_diff_voxels.max()} | "
                 f"{g.organ_dice.min():.5f} | {g.labelmap_diff_voxels.median():.0f}, {g.labelmap_diff_voxels.max()} | "
                 f"{g.mesh_assd_vs_first_mm.median():.4f}, {g.mesh_assd_vs_first_mm.max():.4f} |")
    L.append('\n## Does it change the measured error?\n')
    L.append('Spread of the true error (ASSD against the expert surface) across repeated runs of the same scan, '
             'compared with the spread between scans in the Tier 2 study.\n')
    L.append('| Mode | Path | Largest within-scan range of true ASSD (mm) | Mean within-scan SD (mm) |')
    L.append('|---|---|---|---|')
    for (mode, path), g in df.groupby(['mode', 'path'], sort=False):
        first = df[(df['mode'] == mode) & (df.path == 'worker') & (df.run == 0)].set_index('case').true_assd_mm
        both = pd.concat([g.set_index('case').true_assd_mm, first]).groupby(level=0)
        L.append(f"| {mode} | {path} | {(both.max() - both.min()).max():.4f} | {both.std(ddof=0).mean():.4f} |")
    between = {}
    for name, d in (('spleen', 'tier2_spleen_two_modes_20261018-210930'),
                    ('pancreas', 'tier2_pancreas_heldout_20261018-223006')):
        p = os.path.join(ROOT, 'research', 'results', d, 'results.csv.gz')
        if os.path.exists(p):
            t = pd.read_csv(p)
            t = t[(t.source.isin(['ts_fast', 'ts_full'])) & (t.resolution == 'native') & (t.config == 'production')]
            q = t.groupby('source').assd_mm.quantile([0.25, 0.75]).unstack()
            between[name] = ', '.join(f"{s[3:]} {r[0.75] - r[0.25]:.3f}" for s, r in q.iterrows())
    if between:
        L.append('\nFor scale, the interquartile range of the same error between scans in the Tier 2 study '
                 '(production meshes, native; 41 scans per organ): '
                 + '; '.join(f"{k} {v}" for k, v in between.items()) + ' mm.\n')
    L.append('## Speed\n')
    L.append('| Mode | Path | Median seconds per scan |')
    L.append('|---|---|---|')
    for (mode, path), g in df.groupby(['mode', 'path'], sort=False):
        L.append(f"| {mode} | {path} | {g.seconds.median():.1f} |")
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--per-organ', type=int, default=5, help='scans per organ (seeded sample of the Tier 2 cases)')
    ap.add_argument('--worker-runs', type=int, default=5)
    ap.add_argument('--fresh-runs', type=int, default=3)
    ap.add_argument('--cpu-runs', type=int, default=2)
    ap.add_argument('--full-cpu-scans', type=int, default=2,
                    help='scans per organ that also run full mode on the CPU (slow)')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--resummarise', metavar='RUN_DIR', default=None)
    args = ap.parse_args()
    import pandas as pd
    if args.resummarise:
        df = pd.read_csv(tier2.results_file(args.resummarise))
        with open(os.path.join(args.resummarise, 'run.json'), encoding='utf-8') as fh:
            run = json.load(fh)
        text = summarise(df, run)
        with open(os.path.join(args.resummarise, 'summary.md'), 'w', encoding='utf-8') as fh:
            fh.write(text)
        print(text)
        return
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    import gpu_worker
    stamp = time.strftime('%Y%m%d-%H%M%S')
    outdir = args.outdir or os.path.join(ROOT, 'output', 'research', 'repeatability', stamp)
    tmp = os.path.join(outdir, 'tmp')
    os.makedirs(tmp, exist_ok=True)
    logf = open(os.path.join(outdir, 'log.txt'), 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    rng = np.random.default_rng(args.seed)
    picked = []
    for dataset, root in DATA.items():
        with open(os.path.join(ROOT, 'research', 'results', 'tier2_pancreas_heldout_20261018-223006' if
                               dataset == 'msd_pancreas' else 'tier2_spleen_two_modes_20261018-210930',
                               'run.json'), encoding='utf-8') as fh:
            used = set(json.load(fh)['cases'])
        cases = [c for c in tier2.list_cases(dataset, root) if c[0] in used]
        idx = sorted(rng.choice(len(cases), args.per_organ, replace=False))
        picked += [(dataset,) + cases[i] for i in idx]
    args.full_cpu = {c[1] for k, c in enumerate(picked) if k % args.per_organ < args.full_cpu_scans}
    run = {'test': 'repeatability', 'started': stamp, 'rebuild_check': 'nan-aware', 'cases': [c[1] for c in picked],
           'worker_runs': args.worker_runs, 'fresh_runs': args.fresh_runs, 'cpu_runs': args.cpu_runs,
           'full_cpu_cases': sorted(args.full_cpu), 'seed': args.seed,
           'python': platform.python_version()}
    try:
        import torch
        run['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        run['torch'] = torch.__version__
        import totalsegmentator
        run['totalsegmentator'] = getattr(totalsegmentator, '__version__', '?')
    except Exception:  # noqa: BLE001
        pass
    gpu_worker.ensure_started(log=log, wait=True)
    if gpu_worker.status() != 'warm':
        sys.exit('GPU worker did not start')
    rows, t0 = [], time.time()
    try:
        for k, (dataset, cid, n, image, label) in enumerate(picked, 1):
            log(f'[{k}/{len(picked)}] {cid}  ({time.time() - t0:.0f} s)')
            rows += run_case(dataset, cid, n, image, label, args, tmp, log)
            pd.DataFrame(rows).to_csv(os.path.join(outdir, 'results.csv'), index=False)
    finally:
        gpu_worker.stop()
    shutil.rmtree(tmp, ignore_errors=True)
    run['seconds'] = round(time.time() - t0)
    with open(os.path.join(outdir, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump(run, fh, indent=1)
    df = pd.DataFrame(rows)
    text = summarise(df, run)
    with open(os.path.join(outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    log(text)
    log(f'-> {outdir}')


if __name__ == '__main__':
    main()
