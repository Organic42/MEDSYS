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

Acquisition resolution is the whole-plot factor alongside the engine
(Exp. D): the CT and the expert mask are resampled to a coarser grid before
segmentation and meshing, while the reference surface always comes from the
expert mask at native resolution. Every grid shares one physical frame
(voxel i sits at i * spacing), so meshes from any grid line up with it.

Usage (from the repository root):
  python -m research.tier2 --dataset msd_spleen --data data/msd/Task09_Spleen --preset pilot
  python -m research.tier2 --dataset ctorg --data data/ctorg --preset full --workers 5
  python -m research.tier2 --dataset msd_spleen --data ... --resolutions native,iso3
  python -m research.tier2 --dataset msd_pancreas --data data/msd/Task07_Pancreas --sample 41 --preset full
  python -m research.tier2 --dataset msd_spleen --data ... --preset sensitivity --engines ts_fast
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
    'msd_pancreas': {
        'title': 'Medical Segmentation Decathlon, Task07 Pancreas',
        'licence': 'CC-BY-SA 4.0',
        'labels': ('expert manual segmentation of the pancreas and of pancreatic tumour in every '
                   'training case; the two labels are merged into the organ'),
        'structures': {'pancreas': (1, 2)},
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
    'amos': {
        'title': 'AMOS 2022, CT scans of the labelled training and validation splits',
        'licence': 'CC BY 4.0 (Zenodo record 7262581)',
        'labels': ('manual annotation of 15 abdominal organs in every case, from multiple centres and vendors; '
                   'only the CT scans (ids below 500) are used'),
        # study structure = TotalSegmentator class name: label value(s) in AMOS
        'structures': {'spleen': (1,), 'pancreas': (10,), 'liver': (6,), 'aorta': (8,),
                       'inferior_vena_cava': (9,)},
        'manual_cases': {},
    },
}


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def list_cases(dataset, root):
    """[(case_id, case_number, image_path, label_path)] for a dataset folder."""
    if dataset.startswith('msd_'):
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
    if dataset == 'amos':
        with open(os.path.join(root, 'dataset.json'), encoding='utf-8') as fh:
            meta = json.load(fh)
        out = []
        for item in meta['training'] + meta.get('validation', []):
            img = os.path.normpath(os.path.join(root, item['image']))
            lbl = os.path.normpath(os.path.join(root, item['label']))
            n = int(re.findall(r'\d+', os.path.basename(img))[-1])
            if n < 500 and os.path.exists(img) and os.path.exists(lbl):     # 500 and above are MRI
                out.append((f'amos_{n:04d}', n, img, lbl))
        return sorted(out, key=lambda c: c[1])
    raise ValueError(f'unknown dataset {dataset!r}')


def structure_allowed(dataset, structure, case_number):
    """False where a structure's label is not an expert annotation for this case."""
    allowed = DATASETS[dataset]['manual_cases'].get(structure)
    return True if allowed is None else case_number in allowed


# ── acquisition resolution ───────────────────────────────────────────────

# Minimum spacing (in-plane mm, slice mm) per level. Scans are only ever
# coarsened, never upsampled: an axis already coarser than the level keeps
# its native spacing, and a scan coarser than a level on every axis is left
# as it is (that level then equals native for that case).
RESOLUTIONS = {
    'native': (0.0, 0.0),
    'iso1.5': (1.5, 1.5),
    'iso3': (3.0, 3.0),          # TotalSegmentator fast's working resolution
    'slice5': (0.0, 5.0),        # thick-slice acquisition, native in-plane
}


# Alternative coarsening models for the sensitivity analysis, written as
# '<level>~<variant>'. 'box' replaces the Gaussian anti-aliasing with a
# rectangular slice profile as wide as the coarse voxel (a slab average);
# 'noise' keeps the Gaussian and adds white noise of NOISE_HU to the coarse
# CT, so only the engine sees it.
VARIANTS = ('', 'box', 'noise')
NOISE_HU = 20.0


def split_level(resolution):
    """'iso3~box' -> ('iso3', 'box'); 'iso3' -> ('iso3', '')."""
    base, _, variant = resolution.partition('~')
    if base not in RESOLUTIONS or variant not in VARIANTS:
        raise ValueError(f'unknown resolution level {resolution!r}')
    return base, variant


def level_index(resolution):
    """Sort key and seed component: the base level, then the variant."""
    base, variant = split_level(resolution)
    return list(RESOLUTIONS).index(base) + 10 * VARIANTS.index(variant)


def grid_spacing(spacing, resolution):
    """Voxel spacing of a scan after coarsening it to a resolution level.

    The slice axis is the one with the largest native spacing (axis 2 when
    the scan is isotropic).
    """
    inplane, slice_mm = RESOLUTIONS[split_level(resolution)[0]]
    spacing = np.asarray(spacing, dtype=float)
    floor = np.full(3, inplane)
    floor[int(np.argmax(spacing)) if np.ptp(spacing) > 1e-6 else 2] = slice_mm
    return tuple(float(s) for s in np.maximum(spacing, floor))


def grid_shape(shape, spacing, grid):
    """Shape of the coarse grid: voxel j at j * grid, inside the native extent."""
    return tuple(int(np.floor((n - 1) * s / g + 1e-9)) + 1 for n, s, g in zip(shape, spacing, grid))


def box_kernel(width):
    """Weights of a rectangular profile `width` voxels wide on the integer voxel grid."""
    half = width / 2
    k = np.arange(-int(np.ceil(half)), int(np.ceil(half)) + 1)
    w = np.clip(np.minimum(k + 0.5, half) - np.maximum(k - 0.5, -half), 0, None)
    return w / w.sum()


def resample(volume, spacing, grid, order=1, profile='gaussian'):
    """Anti-aliased resampling onto a coarser grid in the same physical frame.

    Each coarsened axis is filtered first, approximating the wider sampling
    aperture of a coarser acquisition: by default a Gaussian with
    sigma = (factor - 1) / 2 voxels (scikit-image's rescale convention), or
    with profile='box' a rectangular profile one coarse voxel wide. Voxel j
    of the output is then interpolated at j * grid. Returns float32.
    """
    from scipy import ndimage
    vol = np.asarray(volume, dtype=np.float32)
    factor = np.asarray(grid, dtype=float) / np.asarray(spacing, dtype=float)
    if np.allclose(factor, 1.0):
        return vol
    if profile == 'box':
        for axis, f in enumerate(factor):
            if f > 1 + 1e-6:
                vol = ndimage.correlate1d(vol, box_kernel(f), axis=axis, mode='nearest')
    else:
        sigma = np.maximum(0.0, (factor - 1) / 2)
        if np.any(sigma > 0):
            vol = ndimage.gaussian_filter(vol, sigma, mode='nearest')
    return ndimage.affine_transform(vol, factor, output_shape=grid_shape(vol.shape, spacing, grid),
                                    order=order, mode='nearest')


def profile_for(resolution):
    return 'box' if split_level(resolution)[1] == 'box' else 'gaussian'


def resample_mask(mask, spacing, grid, profile='gaussian'):
    """A mask as it would appear on a coarser grid: partial-volume fraction >= 0.5."""
    if np.allclose(grid, spacing):
        return np.asarray(mask, dtype=bool)
    return resample(mask, spacing, grid, profile=profile) >= 0.5


def to_native(mask, spacing, grid, native_shape):
    """Nearest-neighbour map of a coarse-grid mask back onto the native grid."""
    idx = [np.minimum(np.rint(np.arange(n) * s / g).astype(int), m - 1)
           for n, s, g, m in zip(native_shape, spacing, grid, mask.shape)]
    return mask[np.ix_(*idx)]


def resampled_image(img, grid, profile='gaussian', noise_rng=None):
    """A NIfTI image resampled to a coarser grid, with the affine to match.

    Each voxel axis of the affine is stretched by its factor and the
    translation is kept, so coarse voxel j lands on the world point of
    native voxel j * factor. With noise_rng, white noise of NOISE_HU is
    added to the coarse image.
    """
    import nibabel as nib
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    affine = img.affine.copy()
    affine[:3, :3] = affine[:3, :3] * (np.asarray(grid) / np.asarray(spacing))[None, :]
    vol = resample(np.asanyarray(img.dataobj), spacing, grid, profile=profile)
    if noise_rng is not None:
        vol += noise_rng.normal(0.0, NOISE_HU, vol.shape).astype(np.float32)
    return nib.Nifti1Image(vol, affine)


def effective_resolution(spacing, resolution):
    """'native' when a level leaves this scan's spacing unchanged."""
    return 'native' if np.allclose(grid_spacing(spacing, resolution), spacing) else resolution


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
    'ts_full': None,
    'classical': ('lungs', 'bone'),
}

# TotalSegmentator modes. Fast predicts every class with one model at 3 mm.
# Full predicts at 1.5 mm; with roi_subset it runs only the sub-model that
# holds the study's structures, on a crop found by the 3 mm model
# (robust_crop), which is how a single organ is segmented in practice.
TS_MODES = {
    'ts_fast': {'fast': True, 'label': '3 mm'},
    'ts_full': {'fast': False, 'robust_crop': True, 'label': '1.5 mm'},
}


def ts_roi(structures):
    """TotalSegmentator class names covering the study structures, or None for all."""
    names = []
    for s in structures:
        if s == 'bone':
            return None
        names.extend(LUNG_LOBES if s == 'lungs' else (s,))
    return names


def engine_supports(engine, structure):
    s = ENGINES[engine]
    return True if s is None else structure in s


def cache_dir(dataset, case_id):
    out_root = os.environ.get('VRSEG_OUTPUT') or os.path.join(ROOT, 'output')
    return os.path.join(out_root, 'research', 'tier2', 'cache', dataset, case_id)


def _suffix(resolution):
    return '' if resolution == 'native' else f'@{resolution}'


def cached_mask_path(dataset, case_id, engine, structure, resolution='native'):
    return os.path.join(cache_dir(dataset, case_id), f'{engine}_{structure}{_suffix(resolution)}.npz')


def segment_case(dataset, case, structures, engines, device, resolutions=('native',), log=print,
                 seed=2026):
    """Run each engine once per case and resolution; cache one mask per structure.

    Coarser levels segment a resampled copy of the CT, so the engine sees
    what a coarser acquisition would have given it; its mask stays on that
    coarse grid. Levels that leave a scan unchanged reuse the native masks.
    """
    import nibabel as nib
    cid, number, img_path, _ = case
    img = nib.load(img_path)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    levels = list(dict.fromkeys(effective_resolution(spacing, r) for r in resolutions))
    timings = {}
    for res in levels:
        need = [(e, s) for e in engines for s in structures if engine_supports(e, s)
                and not os.path.exists(cached_mask_path(dataset, cid, e, s, res))]
        if not need:
            continue
        os.makedirs(cache_dir(dataset, cid), exist_ok=True)
        grid = grid_spacing(spacing, res)
        hu = None
        if res == 'native':
            src = img_path
        else:
            t = time.perf_counter()
            noise = (np.random.default_rng([seed, number, level_index(res)])
                     if split_level(res)[1] == 'noise' else None)
            coarse = resampled_image(img, grid, profile_for(res), noise)
            hu = np.asanyarray(coarse.dataobj)
            # per process, so engines can run in parallel on the same case
            src = os.path.join(cache_dir(dataset, cid), f'image{_suffix(res)}.{os.getpid()}.nii.gz')
            nib.save(coarse, src)
            timings[f'resample{_suffix(res)}_s'] = time.perf_counter() - t
        shape = grid_shape(img.shape[:3], spacing, grid)
        for engine, mode in TS_MODES.items():
            if not any(e == engine for e, _ in need):
                continue
            seg_dir = os.path.join(cache_dir(dataset, cid), f'{engine}_seg{_suffix(res)}')
            if not glob.glob(os.path.join(seg_dir, '*.nii.gz')):
                from totalsegmentator.python_api import totalsegmentator
                options = {k: v for k, v in mode.items() if k != 'label'}
                if not mode['fast']:
                    options['roi_subset'] = ts_roi(structures)
                t = time.perf_counter()
                totalsegmentator(src, seg_dir, ml=False, device=device, task='total', quiet=True,
                                 **options)
                timings[f'{engine}{_suffix(res)}_s'] = time.perf_counter() - t
                log(f'  {cid} [{res}]: TotalSegmentator ({mode["label"]}) '
                    f'{timings[f"{engine}{_suffix(res)}_s"]:.0f} s on {device}')
            for e, s in need:
                if e != engine:
                    continue
                keep = ts_files_for(s)
                mask = np.zeros(shape, dtype=bool)
                for f in glob.glob(os.path.join(seg_dir, '*.nii.gz')):
                    if keep(os.path.basename(f).replace('.nii.gz', '')):
                        m = np.asanyarray(nib.load(f).dataobj) > 0
                        if m.shape != shape:
                            raise RuntimeError(f'{cid}: TotalSegmentator grid {m.shape} != image {shape}')
                        mask |= m
                np.savez_compressed(cached_mask_path(dataset, cid, e, s, res), mask=mask,
                                    spacing=np.asarray(grid))
        if any(e == 'classical' for e, _ in need):
            from validate import heuristic_bone, heuristic_lungs
            if hu is None:
                hu = np.asanyarray(img.dataobj).astype(np.float32)
            for e, s in need:
                if e != 'classical':
                    continue
                t = time.perf_counter()
                mask = heuristic_lungs(hu) if s == 'lungs' else heuristic_bone(hu)
                timings[f'classical_{s}{_suffix(res)}_s'] = time.perf_counter() - t
                np.savez_compressed(cached_mask_path(dataset, cid, e, s, res), mask=mask,
                                    spacing=np.asarray(grid))
        if res != 'native' and os.path.exists(src):
            os.remove(src)                    # resampled CT: cheap to rebuild, large to keep
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
    ('ts_full', None): (0.6, 30, 0.6),      # same call site; only the engine mode differs
    ('classical', 'lungs'): (0.7, 30, 0.5),  # run_ct_chest
    ('classical', 'bone'): (0.5, 30, 0.4),   # run_ct_chest
    ('gt', None): (0.6, 30, 0.5),            # build_mesh defaults
}


PRODUCTION_BLUR_MM = 0.5


def production_for(source, structure):
    return PRODUCTION.get((source, structure)) or PRODUCTION[(source if source != 'classical' else 'gt', None)]


NEW_SOURCES = ('ts_full',)


def configs_for(grid_name, source, structure):
    grid = {'full': GRID, 'pilot': PILOT_GRID, 'canonical': None}[grid_name]
    combos = ([dict(zip(grid, v)) for v in product(*grid.values())] if grid
              else [dict(CANONICAL)])
    out = [dict(c, config='grid', blur_vox=None) for c in combos]
    b, t, d = production_for(source, structure)
    out.append({'config': 'production', 'blur_mm': None, 'blur_vox': b, 'taubin_iter': t, 'decimation': d})
    # Production with its blur held in millimetres: the application's 0.6-voxel
    # blur is about 0.5 mm in-plane at native spacing, but grows with the grid.
    out.append({'config': 'production_mm', 'blur_mm': PRODUCTION_BLUR_MM, 'blur_vox': None,
                'taubin_iter': t, 'decimation': d})
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
     seed, n_samples, tol_mm, case_i, resolution) = task
    lab = nib.load(label_path)
    spacing = tuple(float(z) for z in lab.header.get_zooms()[:3])
    gt = np.isin(np.rint(np.asanyarray(lab.dataobj)).astype(np.int16), label_values)
    grid = grid_spacing(spacing, resolution)
    resampled = effective_resolution(spacing, resolution) != 'native'
    base = {'dataset': dataset, 'case': cid, 'structure': structure, 'source': source,
            'spacing': 'x'.join(f'{s:g}' for s in spacing), 'slice_mm': max(spacing),
            'resolution': resolution, 'resampled': resampled,
            'grid_spacing': 'x'.join(f'{s:g}' for s in grid), 'grid_max_mm': max(grid)}
    if not gt.any():
        # The structure is not in this scan's field of view (e.g. no lungs on an
        # abdominal CT), so there is nothing to score against.
        return [dict(base, level='mask', status='no_reference')]
    if source == 'gt':
        test = resample_mask(gt, spacing, grid, profile_for(resolution))
    else:
        res = 'native' if not resampled else resolution
        test = np.load(cached_mask_path(dataset, cid, source, structure, res))['mask']
    if test.shape != grid_shape(gt.shape, spacing, grid):
        raise RuntimeError(f'{cid} {source} [{resolution}]: mask grid {test.shape} does not match '
                           f'{grid_shape(gt.shape, spacing, grid)}')
    rows = []
    if test.sum() == 0:
        return [dict(base, level='mask', status='empty_mask')]
    if source != 'gt' or resampled:
        # Mask-level metrics on the native grid; for the expert mask this is
        # the overlap lost to sampling alone.
        native = to_native(test, spacing, grid, gt.shape) if resampled else test
        m = compute_metrics(native, gt, spacing)
        rows.append(dict(base, level='mask', status='ok', config='mask', dice=m['dice'], iou=m['iou'],
                         mask_hd95_mm=m['hd95_mm'], mask_assd_mm=m['assd_mm'],
                         pred_cm3=m['pred_cm3'], gt_cm3=m['gt_cm3']))

    # Each mask is cropped on its own grid; origins put both in the native frame.
    (gt_c,), gt_start = crop_to([gt], int(np.ceil(4 + 3 * 1.0 / min(spacing))))
    rv, rf = extract_isosurface(gt_c, spacing, ReconParams(),            # reference surface
                                tuple(float(a) * s for a, s in zip(gt_start, spacing)))
    (test_c,), start = crop_to([test], int(np.ceil(4 + 3 * 1.0 / min(grid))))   # room for 1 mm blur
    origin = tuple(float(a) * s for a, s in zip(start, grid))
    # Native keeps the seed sequence of the native-only runs, so they reproduce;
    # sources added later get their own stream.
    res_i = level_index(resolution)
    stream = ([seed, case_i, 0, len(source)] + ([res_i] if res_i else [])
              + ([100 + NEW_SOURCES.index(source)] if source in NEW_SOURCES else []))
    iso_cache = {}
    for cfg_i, cfg in enumerate(configs_for(grid_name, source, structure)):
        row = dict(base, level='mesh', config=cfg['config'],
                   is_canonical=(cfg['config'] == 'grid' and all(cfg[k] == v for k, v in CANONICAL.items())),
                   blur_mm=cfg['blur_mm'], blur_vox=cfg['blur_vox'],
                   taubin_iter=cfg['taubin_iter'], decimation=cfg['decimation'])
        params = _params(cfg, grid)
        stream[2] = cfg_i
        try:
            t0 = time.perf_counter()
            if params.blur_sigma_vox not in iso_cache:
                iso_cache[params.blur_sigma_vox] = extract_isosurface(test_c, grid, params, origin)
            v, f = iso_cache[params.blur_sigma_vox]
            tv, tf = mesh_arrays(postprocess_surface(v, f, params))
            t1 = time.perf_counter()
            metrics = compare_meshes(tv, tf, rv, rf, np.random.default_rng(stream),
                                     n_samples=n_samples, tol_mm=tol_mm)
            row.update(status='ok', **metrics, t_post_s=t1 - t0, t_metric_s=time.perf_counter() - t1)
        except (ValueError, RuntimeError) as exc:
            row.update(status='no_surface', error=str(exc)[:120])
        rows.append(row)
    return rows


# ── run ──────────────────────────────────────────────────────────────────

PRESETS = {
    'smoke': {'cases': 1, 'grid': 'canonical', 'samples': 4000, 'resolutions': ('native', 'iso3')},
    'pilot': {'cases': 5, 'grid': 'pilot', 'samples': 20000, 'resolutions': tuple(RESOLUTIONS)},
    'full': {'cases': None, 'grid': 'full', 'samples': 20000, 'resolutions': tuple(RESOLUTIONS)},
    # Coarsening-model sensitivity: canonical and production meshes only.
    'sensitivity': {'cases': None, 'grid': 'canonical', 'samples': 20000,
                    'resolutions': ('iso3~box', 'slice5~box', 'iso3~noise', 'slice5~noise')},
}


def resume_rows(path, grid):
    """Rows of the complete units in an interrupted run's partial.jsonl.

    A unit (case, structure, source, level) is complete when all its mesh
    rows were written; partial units and a cut-off last line are dropped,
    and the file is rewritten with the complete units only. Returns
    (rows, set of complete unit keys).
    """
    if not os.path.exists(path):
        return [], set()
    units = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            units.setdefault((r['case'], r['structure'], r['source'], r['resolution']), []).append(r)
    done = {}
    for key, rs in units.items():
        meshes = sum(r.get('level') == 'mesh' for r in rs)
        if meshes == len(configs_for(grid, key[2], key[1])) or any(r.get('status') in ('empty_mask', 'no_reference') for r in rs):
            done[key] = rs
    rows = [r for rs in done.values() for r in rs]
    with open(path, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + '\n')
    return rows, set(done)


def run_experiment(dataset, data_root, preset='pilot', engines=('ts_fast', 'classical'),
                   workers=None, seed=2026, tol_mm=1.0, outdir=None, max_cases=None,
                   resolutions=None, log=print, sample=None, segment_only=False):
    from research.phantom_experiment import _provenance
    meta, cfg = DATASETS[dataset], PRESETS[preset]
    cases = list_cases(dataset, data_root)
    limit = max_cases or cfg['cases']
    cases = cases[:limit] if limit else cases
    if sample and sample < len(cases):
        # A seeded random subset, drawn before anything is measured.
        keep = np.sort(np.random.default_rng(seed).choice(len(cases), sample, replace=False))
        cases = [cases[i] for i in keep]
    structures = list(meta['structures'])
    engines = [e for e in engines if any(engine_supports(e, s) for s in structures)]
    resolutions = sorted(dict.fromkeys(resolutions or cfg['resolutions']), key=level_index)

    try:
        import torch
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    t_start = time.perf_counter()
    seg_timings = {}
    log(f'Segmenting {len(cases)} case(s) with {", ".join(engines) or "no engines"} '
        f'at {", ".join(resolutions)} ({device})...')
    for case in cases:
        seg_timings[case[0]] = segment_case(dataset, case, structures, engines, device, resolutions, log,
                                            seed)
    if segment_only:
        return None, []

    tasks = []
    for case_i, (cid, num, _, lbl) in enumerate(cases):
        for s in structures:
            if not structure_allowed(dataset, s, num):
                continue
            for res in resolutions:
                for source in ['gt'] + [e for e in engines if engine_supports(e, s)]:
                    tasks.append((dataset, cid, lbl, s, meta['structures'][s], source, cfg['grid'],
                                  seed, cfg['samples'], tol_mm, case_i, res))
    run_id = time.strftime('%Y%m%d-%H%M%S') + f'_{dataset}_{preset}'
    out_root = os.environ.get('VRSEG_OUTPUT') or os.path.join(ROOT, 'output')
    outdir = outdir or os.path.join(out_root, 'research', 'tier2', run_id)
    os.makedirs(outdir, exist_ok=True)
    # Rows are also appended unit by unit, so an interrupted run keeps its work;
    # passing its folder as outdir resumes it from the units already complete.
    rows, resumed = resume_rows(os.path.join(outdir, 'partial.jsonl'), grid=cfg['grid'])
    if resumed:
        tasks = [t for t in tasks if (t[1], t[3], t[5], t[11]) not in resumed]
        log(f'Resuming: {len(resumed)} units already scored')
    partial = open(os.path.join(outdir, 'partial.jsonl'), 'a', encoding='utf-8')

    def collect(unit_rows):
        rows.extend(unit_rows)
        for r in unit_rows:
            partial.write(json.dumps(r, default=str) + '\n')
        partial.flush()

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    log(f'Reconstructing and scoring: {len(tasks)} units on {workers} worker(s) -> {outdir}')
    with partial:
        if workers == 1:
            for t in tasks:
                collect(run_unit(t))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_unit, t) for t in tasks]
                for done, fut in enumerate(as_completed(futures), 1):
                    collect(fut.result())
                    log(f'  unit {done}/{len(tasks)} done')
    wall = time.perf_counter() - t_start

    rows.sort(key=lambda r: (r['structure'], r['case'], level_index(r['resolution']),
                             r['source'], str(r.get('config'))))
    fields = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with open(os.path.join(outdir, 'results.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.remove(os.path.join(outdir, 'partial.jsonl'))       # results.csv now holds every row
    try:
        from importlib.metadata import version
        ts_version = version('TotalSegmentator')
    except Exception:  # not installed
        ts_version = None
    run = {
        'tier': 2, 'dataset': dataset, 'dataset_info': meta, 'preset': preset, 'seed': seed,
        'cases': [c[0] for c in cases], 'structures': structures, 'engines': engines,
        'grid': cfg['grid'], 'grid_levels': GRID if cfg['grid'] == 'full' else PILOT_GRID,
        'isosurface': 'lewiner (fixed: Tier 1 found it explained 0.0% of variance)',
        'canonical': CANONICAL, 'production': {f'{k[0]}/{k[1] or "default"}': v for k, v in PRODUCTION.items()},
        'reference_surface': 'canonical reconstruction of the expert mask at native spacing',
        'resolutions': {r: RESOLUTIONS[split_level(r)[0]] for r in resolutions},
        'coarsening_variants': {'box': 'rectangular slice profile one coarse voxel wide',
                                'noise': f'Gaussian profile plus white noise, SD {NOISE_HU:g} HU, on the CT'},
        'case_sample': sample,
        'ts_modes': {k: {a: b for a, b in v.items() if a != 'label'} for k, v in TS_MODES.items()},
        'resolution_rule': ('minimum spacing (in-plane mm, slice mm); coarsen only; CT and expert '
                            'mask resampled with Gaussian anti-aliasing (sigma = (factor - 1) / 2 '
                            'voxels) and linear interpolation, mask thresholded at 0.5; engine '
                            'masks scored at mask level after nearest-neighbour mapping to native'),
        'samples_per_side': cfg['samples'], 'nsd_tolerance_mm': tol_mm, 'segmentation_device': device,
        'segmentation_timings_s': seg_timings, 'totalsegmentator_version': ts_version,
        'rows': len(rows), 'units': len(tasks) + len(resumed), 'resumed_units': len(resumed),
        'workers': workers, 'wall_time_s': round(wall, 1),
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

def results_file(run_dir):
    """results.csv of a run, or the compressed copy kept in research/results/."""
    path = os.path.join(run_dir, 'results.csv')
    return path if os.path.exists(path) else path + '.gz'


def eta_squared(df, y, factors):
    """Main-effect share of the variance of column y for each factor (descriptive)."""
    v = df[y]
    tot = float(((v - v.mean()) ** 2).sum())
    shares = {}
    for f in factors:
        m = df.groupby(f, observed=True)[y].agg(['mean', 'count'])
        shares[f] = float((m['count'] * (m['mean'] - v.mean()) ** 2).sum()) / tot if tot else float('nan')
    return shares


def summarise(rows, run):
    """Descriptive markdown summary of one Tier 2 run. Requires pandas."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if 'resolution' not in df:                       # runs from before Exp. D
        df['resolution'], df['resampled'] = 'native', False
    order = sorted(set(df['resolution']), key=level_index)
    df['resolution'] = pd.Categorical(df['resolution'], categories=order, ordered=True)
    df['resampled'] = df['resampled'].astype(str).str.lower() == 'true'
    info = run['dataset_info']
    lines = [f"# Tier 2: {info['title']} ({run['preset']})", '',
             f"Commit {run['commit']} | {len(run['cases'])} cases | {run['rows']} rows | "
             f"wall time {run['wall_time_s'] / 60:.1f} min | seed {run['seed']}", '',
             f"Licence {info['licence']}. Labels: {info['labels']}.", '',
             'Reference surface: canonical reconstruction of the expert mask. Descriptive only; '
             'distances in mm; mean ± SD across cases.', '']
    if 'slice_mm' in df:
        lines += [f"Slice spacing across cases: {df['slice_mm'].min():g}–{df['slice_mm'].max():g} mm.", '']
    if len(order) > 1:
        per_case = df.drop_duplicates(['case', 'resolution'])
        lines += ['## Acquisition resolution (Exp. D)', '',
                  'Each level is a minimum spacing (in-plane, slice): the CT and the expert mask are '
                  'resampled with Gaussian anti-aliasing before segmentation and meshing, and the '
                  'reference surface stays at native resolution. Scans already coarser than a level '
                  'keep their native spacing on that axis.', '',
                  '| resolution | minimum spacing (mm) | cases resampled | largest grid spacing (median, mm) |',
                  '|---|---|---|---|']
        for r, g in per_case.groupby('resolution', observed=True):
            inplane, slice_mm = RESOLUTIONS[split_level(r)[0]]
            rule = 'as acquired' if r == 'native' else (
                f'{inplane:g} in-plane, {slice_mm:g} slice' if inplane else f'native in-plane, {slice_mm:g} slice')
            lines.append(f"| {r} | {rule} | {int(g['resampled'].sum())}/{len(g)} | {g['grid_max_mm'].median():g} |")
        lines.append('')

    def msd(s, d=3):
        s = pd.to_numeric(s, errors='coerce').dropna()
        return '—' if s.empty else (f'{s.mean():.{d}f} ± {s.std():.{d}f}' if len(s) > 1 else f'{s.mean():.{d}f}')

    mask = df[(df.get('level') == 'mask') & (df.get('status') == 'ok')] if 'level' in df else df.iloc[0:0]
    if len(mask):
        lines += ['## Segmentation (mask level, Exp. A)', '',
                  'Coarse masks are mapped back to the native grid (nearest neighbour) and scored '
                  'against the expert mask. Source `gt` is the expert mask resampled: the overlap '
                  'lost to sampling alone.', '',
                  '| structure | source | resolution | cases | Dice | IoU | HD95 | ASSD |',
                  '|---|---|---|---|---|---|---|---|']
        for (s, e, r), g in mask.groupby(['structure', 'source', 'resolution'], observed=True):
            lines.append(f'| {s} | {e} | {r} | {len(g)} | {msd(g.dice)} | {msd(g.iou)} | '
                         f'{msd(g.mask_hd95_mm, 2)} | {msd(g.mask_assd_mm, 2)} |')
        lines.append('')

    mesh = df[(df.get('level') == 'mesh') & (df.get('status') == 'ok')] if 'level' in df else df.iloc[0:0]
    if len(mesh):
        lines += ['## End to end (mesh against reference surface, Exp. B and C)', '',
                  'Source `gt` is the expert mask itself: its error is reconstruction (and, below '
                  'native, sampling) alone. Canonical on `gt` at native reproduces the reference, '
                  'so it should read 0.', '',
                  '| structure | source | resolution | config | ASSD | HD95 | mean signed | volume error % '
                  '| closed + manifold |',
                  '|---|---|---|---|---|---|---|---|---|']
        for (s, src, r), g in mesh.groupby(['structure', 'source', 'resolution'], observed=True):
            for name, sub in (('canonical', g[g.is_canonical == True]),  # noqa: E712
                              ('production', g[g.config == 'production'])):
                if len(sub):
                    lines.append(f'| {s} | {src} | {r} | {name} | {msd(sub.assd_mm)} | {msd(sub.hd95_mm, 2)} | '
                                 f'{msd(sub.mean_signed_mm)} | {msd(sub.volume_err_pct, 1)} | '
                                 f'{sub.topology_ok.astype(bool).mean():.0%} |')
        lines.append('')

        grid = mesh[mesh.config == 'grid'].copy()
        if len(grid) and grid['blur_mm'].nunique() > 1:
            import numpy as np
            factors = ['source'] + (['resolution'] if len(order) > 1 else []) + [
                'blur_mm', 'taubin_iter', 'decimation', 'case']
            grid = grid[grid['assd_mm'] > 0]
            grid['log_assd'] = np.log(grid['assd_mm'])
            lines += ['## Share of log-ASSD variance (grid configurations, preview)', '',
                      'Main effects (eta squared); "case" is patient-to-patient variation. '
                      'Exact rows with zero error (canonical on the expert mask) are excluded. '
                      'Formal attribution uses mixed-effects models and Sobol indices.', '']
            for s, g in grid.groupby('structure'):
                shares = eta_squared(g, 'log_assd', factors)
                rest = 1 - sum(shares.values())
                lines.append(f'- **{s}**: ' + ', '.join(f'{k} {v:.1%}' for k, v in shares.items())
                             + f', interactions + residual {rest:.1%}')
            lines.append('')
            lines += ['## Mean ASSD by setting (grid)', '']
            for f in [f for f in factors if f not in ('source', 'case')]:
                piv = grid.pivot_table(index='source', columns=f, values='assd_mm', aggfunc='mean',
                                       observed=True)
                lines.append(f'- **{f}**: ' + '; '.join(
                    f'{src}: ' + ', '.join(f'{c}→{piv.loc[src, c]:.3f}' for c in piv.columns)
                    for src in piv.index))
            lines.append('')

        lines += ['## Topology (closed and manifold, by decimation)', '']
        topo = (mesh.groupby(['source', 'resolution', 'decimation'], observed=True)['topology_ok']
                .apply(lambda x: x.astype(bool).mean()).unstack())
        lines += ['| source | resolution | ' + ' | '.join(str(c) for c in topo.columns) + ' |',
                  '|---' * (len(topo.columns) + 2) + '|']
        for (src, r), vals in topo.iterrows():
            lines.append(f'| {src} | {r} | ' + ' | '.join('—' if pd.isna(v) else f'{v:.0%}' for v in vals) + ' |')
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
    ap.add_argument('--sample', type=int, default=None, help='use a seeded random subset of N cases')
    ap.add_argument('--segment-only', action='store_true', help='run and cache the engines, then stop')
    ap.add_argument('--resolutions', default=None,
                    help=f'comma-separated subset of {",".join(RESOLUTIONS)} (default: the preset\'s)')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--resummarise', metavar='RUN_DIR', default=None)
    args = ap.parse_args()
    if args.resummarise:
        import pandas as pd
        rows = pd.read_csv(results_file(args.resummarise)).to_dict('records')
        with open(os.path.join(args.resummarise, 'run.json'), encoding='utf-8') as fh:
            run = json.load(fh)
        with open(os.path.join(args.resummarise, 'summary.md'), 'w', encoding='utf-8') as fh:
            fh.write(summarise(rows, run))
        print(f'summary rebuilt -> {args.resummarise}')
        return
    if not (args.dataset and args.data):
        ap.error('--dataset and --data are required')
    resolutions = tuple(r for r in (args.resolutions or '').split(',') if r) or None
    for r in resolutions or ():
        try:
            split_level(r)
        except ValueError as exc:
            ap.error(str(exc))
    outdir, rows = run_experiment(args.dataset, args.data, args.preset,
                                  tuple(e for e in args.engines.split(',') if e),
                                  args.workers, args.seed, args.tol, args.outdir, args.cases,
                                  resolutions, sample=args.sample, segment_only=args.segment_only)
    print('engines cached' if args.segment_only else f'{len(rows)} rows -> {outdir}')


if __name__ == '__main__':
    main()
