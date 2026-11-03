"""
Accuracy estimate for every mesh MEDSYS builds, grounded in the MEDSYS study
(Morya and Wage, "Stage-Wise Attribution of Geometric Error in Automatic
Anatomical Surface Reconstruction From Computed Tomography", 2026; the paper
and its data are in docs/paper/ and research/results/).

Three parts, kept separate because the study showed that stage errors do not
simply add up:

  resolution limit   the surface error the scan's voxel spacing alone causes,
                     even with a perfect segmentation. Estimated, not measured
                     on this scan: about 0.177 x Delta, with Delta the RMS
                     voxel spacing (paper, Sec. VI-A/B; range from the phantom
                     floor, 0.14-0.21 x Delta).
  reconstruction     how far smoothing and simplification moved this mesh from
                     the raw segmentation surface. Measured on this mesh with
                     the paper's own metric code.
  segmentation       the engine's error against expert labels, where the study
                     measured it (TotalSegmentator on spleen and pancreas), on
                     the study grid closest to the scan's. Elsewhere it is
                     reported as not validated.

Each line was checked against the true error of the app's own meshes on 82
expert-labelled scans (research/card_validation.py, validation test 1).

None of this is a guarantee for an individual scan, and MEDSYS is not a
medical device.
"""
import json
import os

import numpy as np

# Resolution limit: canonical ASSD per mm of RMS spacing. 0.177 is the study's
# fit on resampled expert masks (95% CI 0.171-0.184, held-out pancreas 0.180);
# the range is the central 95% of the exact-surface error on the Tier 1
# phantoms (0.127-0.212 x RMS spacing over 80 canonical meshes): isotropic
# grids sit near 0.14 h, the thin capsule at 0.8 x 0.8 x 5 mm near 0.21.
# Validation test 1 (research/card_validation.py) found the earlier lower
# bound, 0.14, held only 60% of the phantoms.
SLOPE, SLOPE_LOW, SLOPE_HIGH = 0.177, 0.13, 0.21
PREDICTION_INTERVAL_MM = 0.111      # 95% leave-one-scan-out interval of that fit

# Canonical ASSD (mm) of TotalSegmentator against expert surfaces, over 41
# scans per organ (MSD Task09 spleen, MSD Task07 pancreas; Tier 2 results),
# at each grid the study tested. A scan is compared with the grid closest to
# its own (reference_level): validation test 1 found a native-only reference
# held as few as 15% of coarse full-mode scans in its middle half, against
# 46-51% once matched to the grid.
GRIDS = {'native': 'at the resolution they were acquired',
         'iso1.5': 'resampled to 1.5 mm in-plane',
         'iso3': 'resampled to 3 mm voxels',
         'slice5': 'with 5 mm slices'}
_DATA = {'spleen': 'MSD Task09 Spleen', 'pancreas': 'MSD Task07 Pancreas (tumours included in the organ)'}


def _ref(median, q1, q3, p5, p95, scans):
    return {'median_mm': median, 'iqr_mm': [q1, q3], 'range90_mm': [p5, p95], 'scans': scans}


ENGINE_REFERENCE = {
    ('fast', 'spleen', 'native'): _ref(0.854, 0.81, 0.939, 0.741, 1.174, 41),
    ('fast', 'spleen', 'iso1.5'): _ref(0.848, 0.808, 0.976, 0.713, 1.114, 41),
    ('fast', 'spleen', 'iso3'): _ref(0.858, 0.781, 0.962, 0.724, 1.171, 41),
    ('fast', 'spleen', 'slice5'): _ref(0.935, 0.828, 1.027, 0.754, 1.291, 41),
    ('full', 'spleen', 'native'): _ref(0.699, 0.627, 0.797, 0.565, 1.046, 41),
    ('full', 'spleen', 'iso1.5'): _ref(0.72, 0.634, 0.819, 0.593, 1.075, 41),
    ('full', 'spleen', 'iso3'): _ref(0.95, 0.79, 1.173, 0.707, 1.664, 41),
    ('full', 'spleen', 'slice5'): _ref(0.8, 0.674, 0.886, 0.591, 1.186, 41),
    ('fast', 'pancreas', 'native'): _ref(1.881, 1.532, 2.549, 1.237, 4.04, 41),
    ('fast', 'pancreas', 'iso1.5'): _ref(1.89, 1.574, 2.554, 1.319, 4.006, 41),
    ('fast', 'pancreas', 'iso3'): _ref(1.934, 1.572, 2.679, 1.273, 4.316, 41),
    ('fast', 'pancreas', 'slice5'): _ref(1.947, 1.686, 2.548, 1.439, 4.175, 41),
    ('full', 'pancreas', 'native'): _ref(1.87, 1.476, 3.203, 1.039, 5.376, 41),
    ('full', 'pancreas', 'iso1.5'): _ref(1.953, 1.492, 3.264, 1.058, 5.275, 41),
    ('full', 'pancreas', 'iso3'): _ref(2.345, 1.869, 3.599, 1.151, 6.19, 41),
    ('full', 'pancreas', 'slice5'): _ref(2.128, 1.651, 3.203, 1.115, 5.617, 41),
}


def reference_level(spacing):
    """The study grid closest to a scan's spacing. The slice axis is the
    coarsest; in-plane spacing decides first, as the study's grids do."""
    s = np.sort(np.asarray(spacing, dtype=float))
    inplane, slice_mm = s[1], s[2]
    if inplane >= 2.25:
        return 'iso3'
    if inplane >= 1.25:
        return 'iso1.5'
    return 'slice5' if slice_mm >= 4 else 'native'


def engine_reference(mode, structure, spacing):
    """The engine's measured error for a structure on scans like this one, or None."""
    level = reference_level(spacing)
    ref = ENGINE_REFERENCE.get((mode, structure, level))
    if ref is None:
        return None
    return dict(ref, level=level, grid=GRIDS[level], data=_DATA[structure])

# What each line of the card rests on, sent with every estimate so no client
# shows a study statistic as if it were a prediction for this model:
#   measured   computed on this mesh
#   estimated  predicted for this scan from its voxel spacing with the study's fit
#   study      the engine's observed error on other expert-labelled scans; a
#              benchmark for scans like this one, not a prediction for this mesh
#              (predicting each mesh's own error is RQ2, not built yet)
BASIS = {'resolution_limit': 'estimated', 'reconstruction': 'measured',
         'watertight': 'measured', 'segmentation': 'study'}

FILE = 'accuracy.json'
SAMPLES = 3000                      # surface points per side; enough for a mean at ~0.01 mm


def added_sampling(spacing):
    """RMS voxel spacing: the sampling a grid adds to an infinitely fine one (paper, eq. 2)."""
    s = np.asarray(spacing, dtype=float)
    return float(np.sqrt(np.mean(s ** 2)))


def resolution_limit(spacing):
    delta = added_sampling(spacing)
    return {'estimate_mm': round(SLOPE * delta, 3),
            'range_mm': [round(SLOPE_LOW * delta, 3), round(SLOPE_HIGH * delta, 3)],
            'rms_spacing_mm': round(delta, 3)}


def assess(mask, spacing, mesh, seed=0):
    """Compare a delivered mesh with the raw surface of the same mask.

    The raw (canonical) surface is marching cubes at level 0.5 without blur,
    smoothing or decimation, the procedure the study uses for its references.
    Returns ASSD, the larger directed 95th percentile, the mean signed
    distance (negative: the mesh shrank), the volume change and whether the
    delivered mesh is watertight.
    """
    from reconstruction import ReconParams, extract_isosurface, mesh_arrays
    from research.surface_metrics import compare_meshes
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - 2, 0)
    hi = np.minimum(idx.max(axis=0) + 3, mask.shape)
    crop = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
    origin = tuple(float(a) * float(s) for a, s in zip(lo, spacing))
    rv, rf = extract_isosurface(crop, spacing, ReconParams(), origin)
    tv, tf = mesh_arrays(mesh)
    m = compare_meshes(tv, tf, rv, rf, np.random.default_rng(seed), n_samples=SAMPLES, local=False)
    closed = bool(m['closed'])
    return {'recon_assd_mm': round(m['assd_mm'], 3), 'recon_p95_mm': round(m['hd95_mm'], 3),
            'recon_signed_mm': round(m['mean_signed_mm'], 3),
            # an enclosed volume only means something for a closed mesh
            'volume_change_pct': round(m['volume_err_pct'], 2) if closed else None,
            'watertight': closed, 'pieces': int(m['components'])}


# ── storage: one accuracy.json per dataset, next to report.json ─────────

FIELDS = ('recon_assd_mm', 'recon_p95_mm', 'recon_signed_mm', 'volume_change_pct', 'watertight', 'pieces')


def attach(mesh, result, spacing):
    """Store an assessment on the mesh as numeric field data until it is saved."""
    for k in FIELDS:
        mesh.field_data[f'medsys_{k}'] = np.array([np.nan if result[k] is None else float(result[k])])
    mesh.field_data['medsys_spacing_mm'] = np.asarray(spacing, dtype=float)


def record(mesh, outdir, name):
    """Write a saved mesh's assessment into <outdir>/accuracy.json."""
    if 'medsys_recon_assd_mm' not in mesh.field_data:
        return
    entry = {k: float(mesh.field_data[f'medsys_{k}'][0]) for k in FIELDS}
    entry = {k: None if np.isnan(v) else v for k, v in entry.items()}
    entry['watertight'], entry['pieces'] = bool(entry['watertight']), int(entry['pieces'])
    spacing = [float(s) for s in mesh.field_data['medsys_spacing_mm']]
    path = os.path.join(outdir, FILE)
    data = {}
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
    data.setdefault('structures', {})[name] = entry
    data['spacing_mm'] = spacing
    data['resolution_limit'] = resolution_limit(spacing)
    data['prediction_interval_mm'] = PREDICTION_INTERVAL_MM
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=1)


def for_dataset(ds_dir, report):
    """accuracy.json of a dataset, with the engine's validated error added
    for the structures the study measured. None when there is none."""
    path = os.path.join(ds_dir, FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    engine = (report or {}).get('engine')
    data['engine'] = engine or 'classical'
    data['basis'] = dict(BASIS)
    data['validated'] = {}
    if data.get('spacing_mm'):              # from the current constants, not those at meshing time
        data['resolution_limit'] = resolution_limit(data['spacing_mm'])
    if engine == 'TotalSegmentator' and data.get('spacing_mm'):
        mode = 'fast' if (report or {}).get('fast_mode', True) else 'full'
        data['engine_mode'] = mode
        for name in data.get('structures', {}):
            ref = engine_reference(mode, name, data['spacing_mm'])
            if ref:
                data['validated'][name] = ref
    return data
