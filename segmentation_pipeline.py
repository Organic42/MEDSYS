"""
Modality-aware medical image segmentation pipeline.

Supports multiple datasets — each run writes all images and 3D models into its
own folder under output/<dataset_name>/.

  MRI brain : N4 bias -> NLM denoise -> BET skull strip -> MedSAM refine
              -> GM/WM/CSF tissue classification -> hybrid 3D surface
  CT chest  : HU conversion -> windowing -> lungs / skeleton / body / soft-tissue
              segmentation -> colored 3D meshes

Usage:
  python segmentation_pipeline.py --input "C:\\path\\to\\dicom_dir" --name chest_ct
  python segmentation_pipeline.py --input "C:\\path\\to\\brain_dir" --name brain_mri
  python segmentation_pipeline.py --input <dir> --name <ds> --modality CT
"""

import os
import sys
import glob
import json
import shutil
import argparse
import warnings
import collections

import numpy as np
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology

warnings.filterwarnings('ignore')

import time

# Path resolution lives in config.py so this module and app.py/tasks.py always
# agree on where data goes — critical when frozen into a .exe, where a plain
# `os.path.dirname(__file__)` would resolve inside the bundle's internal
# resource dir instead of next to the executable.
import config

PROJECT_DIR = config.RUNTIME_DIR
OUTPUT_ROOT = config.OUTPUT_ROOT
MEDSAM_CHECKPOINT = os.path.join(config.RUNTIME_DIR, 'medsam_vit_b.pth')


def get_device():
    """Return ('cuda'|'cpu', human-readable name)."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda', torch.cuda.get_device_name(0)
    except Exception:
        pass
    return 'cpu', 'CPU'


class Stopwatch:
    """Prints per-step wall time so the UI/log shows real progress."""
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0

    def lap(self, label):
        now = time.time()
        print(f"    [{label}: {now - self.last:.1f}s "
              f"(total {now - self.t0:.0f}s)]", flush=True)
        self.last = now

    def total(self):
        return time.time() - self.t0


# ════════════════════════════════════════════════════════════════════════
# SHARED: DICOM loading
# ════════════════════════════════════════════════════════════════════════

def find_dicom_files(input_dir):
    """Recursively collect DICOM files (.dcm, .ima, or extensionless)."""
    files = []
    for root, _, names in os.walk(input_dir):
        for nm in names:
            fp = os.path.join(root, nm)
            low = nm.lower()
            if low.endswith(('.dcm', '.ima')) or '.' not in nm:
                files.append(fp)
    return files


def load_largest_series(input_dir):
    """Load the largest single DICOM series (by SeriesInstanceUID) as a volume."""
    files = find_dicom_files(input_dir)
    by_uid = collections.defaultdict(list)
    meta = {}
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            continue
        uid = str(getattr(ds, 'SeriesInstanceUID', 'unknown'))
        by_uid[uid].append(fp)
        if uid not in meta:
            meta[uid] = ds
    if not by_uid:
        raise RuntimeError(f"No DICOM files found in {input_dir}")

    best_uid = max(by_uid, key=lambda u: len(by_uid[u]))
    print(f"  Found {len(by_uid)} series — using largest "
          f"('{getattr(meta[best_uid], 'SeriesDescription', '?')}', "
          f"{len(by_uid[best_uid])} slices)")

    slices = [pydicom.dcmread(fp) for fp in by_uid[best_uid]]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda s: int(getattr(s, 'InstanceNumber', 0)))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    # Spacing in ARRAY-axis order (slice, row, col) to match volume[z, y, x].
    # DICOM PixelSpacing = [row_spacing, col_spacing]; SliceThickness = z.
    try:
        ps = slices[0].PixelSpacing
        st = float(getattr(slices[0], 'SliceThickness', 1.0))
        spacing = (st, float(ps[0]), float(ps[1]))
    except Exception:
        spacing = (1.0, 1.0, 1.0)

    return slices, volume, spacing


def detect_modality(ds, override=None):
    if override:
        return override.upper()
    return str(getattr(ds, 'Modality', 'MR')).upper()


# ════════════════════════════════════════════════════════════════════════
# SHARED: mesh building / rendering
# ════════════════════════════════════════════════════════════════════════

def build_mesh(mask, spacing, sigma=0.6, taubin_iter=30, reduction=0.5,
               min_voxels=200):
    """Binary mask -> smoothed, decimated PyVista surface mesh."""
    import pyvista as pv
    if mask.sum() < min_voxels:
        return None
    sm = gaussian_filter(mask.astype(np.float32),
                         sigma=(sigma, sigma, sigma) if np.isscalar(sigma) else sigma)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            sm, level=0.5, spacing=spacing,
            gradient_direction='descent', allow_degenerate=False)
    except Exception:
        return None
    fa = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    mesh = pv.PolyData(verts, fa)
    mesh = mesh.smooth_taubin(n_iter=taubin_iter, pass_band=0.05,
                              normalize_coordinates=True)
    if reduction > 0:
        mesh = mesh.decimate_pro(reduction=reduction, feature_angle=45.0,
                                 preserve_topology=True)
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True,
                                auto_orient_normals=True, consistent_normals=True)
    return mesh


def save_mesh(mesh, outdir, name):
    import pyvista as pv
    for ext in ('vtk', 'stl', 'ply'):
        mesh.save(os.path.join(outdir, f'{name}.{ext}'))
    pv.save_meshio(os.path.join(outdir, f'{name}.obj'), mesh)


def render_meshes(mesh_specs, outdir, filename, title):
    """Render a list of (mesh, color, opacity) in 4 anatomical views."""
    import pyvista as pv
    meshes = [m for m, _, _ in mesh_specs if m is not None]
    if not meshes:
        return
    combined = meshes[0].copy()
    for m in meshes[1:]:
        combined += m
    center = np.array(combined.center)
    b = combined.bounds
    dist = np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]]) * 1.4

    views = {
        'Anterior':  (center + np.array([0, -dist, 0]), (0, 0, 1)),
        'Lateral':   (center + np.array([dist, 0, 0]),  (0, 0, 1)),
        'Posterior': (center + np.array([0, dist, 0]),  (0, 0, 1)),
        'Superior':  (center + np.array([0, 0, dist]),  (0, 1, 0)),
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (label, (cam_pos, up)) in zip(axes, views.items()):
        pl = pv.Plotter(off_screen=True, window_size=[600, 600], lighting='light_kit')
        for mesh, color, opacity in mesh_specs:
            if mesh is None:
                continue
            pl.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True,
                        specular=0.4, specular_power=18, ambient=0.3, diffuse=0.8)
        pl.background_color = '#0d1117'
        pl.camera_position = [cam_pos.tolist(), center.tolist(), up]
        pl.reset_camera()
        pl.camera.zoom(1.1)
        img = pl.screenshot(return_img=True)
        pl.close()
        ax.imshow(img)
        ax.set_title(label, fontsize=13, fontweight='bold', color='#333')
        ax.axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(outdir)}/{filename}")


def panel(rows, idxs, outdir, filename, suptitle):
    """rows = list of (label, volume-or-overlay-list, cmap-or-None)."""
    nr = len(rows)
    fig, axes = plt.subplots(nr, len(idxs), figsize=(3.6*len(idxs), 3.4*nr))
    if nr == 1:
        axes = axes[None, :]
    for r, (label, data, cmap) in enumerate(rows):
        for c, i in enumerate(idxs):
            ax = axes[r, c]
            if cmap is None:
                ax.imshow(np.clip(data[i], 0, 1))
            else:
                ax.imshow(data[i], cmap=cmap)
            ax.set_title(f'{label} — slice {i}', fontsize=9)
            ax.axis('off')
    plt.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(outdir)}/{filename}")


def overlay(base_slice, mask_slice, rgb):
    o = np.stack([base_slice]*3, axis=-1)
    o[mask_slice] = 0.4*o[mask_slice] + 0.6*np.array(rgb)
    return np.clip(o, 0, 1)


# ════════════════════════════════════════════════════════════════════════
# TOTALSEGMENTATOR ENGINE (deep-learning, ~100 structures, CT & MR)
# ════════════════════════════════════════════════════════════════════════

# Distinct colors cycled across the produced structures for the 3D viewer
_TS_PALETTE = [
    '#ff6b6b', '#5bb6ff', '#7CFC9B', '#f5e98c', '#c792ea', '#ff9ec4',
    '#ffa94d', '#74e0d8', '#9ccc65', '#ef9a9a', '#90caf9', '#fff176',
    '#ce93d8', '#80cbc4', '#ffab91', '#a5d6a7', '#f48fb1', '#b39ddb',
]


def _dicom_to_nifti(input_dir, work_dir):
    """Convert a DICOM directory to a single NIfTI for TotalSegmentator."""
    import dicom2nifti
    import dicom2nifti.settings as dset
    dset.disable_validate_slice_increment()
    os.makedirs(work_dir, exist_ok=True)
    dicom2nifti.convert_directory(input_dir, work_dir, compression=True, reorient=True)
    niis = glob.glob(os.path.join(work_dir, '*.nii.gz'))
    if not niis:
        raise RuntimeError("DICOM->NIfTI conversion produced no output")
    # largest file = the main series
    return max(niis, key=os.path.getsize)


def run_totalseg(input_dir, outdir, modality='CT', fast=True):
    """Deep-learning whole-body segmentation via TotalSegmentator (nnU-Net).
    Produces one labeled mask per anatomical structure + a mesh for each."""
    print("\n[TOTALSEGMENTATOR ENGINE]")
    sw = Stopwatch()
    dev, dev_name = get_device()
    ts_device = 'gpu' if dev == 'cuda' else 'cpu'
    print(f"  Device: {dev_name} | mode: {'fast (3mm)' if fast else 'full (1.5mm)'}")

    # 1. DICOM -> NIfTI
    print("[1/4] Converting DICOM to NIfTI...")
    work = os.path.join(outdir, '_work')
    nifti = _dicom_to_nifti(input_dir, work)
    sw.lap('DICOM->NIfTI')

    # 2. Run TotalSegmentator (one mask file per structure)
    print(f"[2/4] Running TotalSegmentator ({modality})... this can take a while on CPU")
    seg_dir = os.path.join(outdir, 'segmentations')
    os.makedirs(seg_dir, exist_ok=True)
    from totalsegmentator.python_api import totalsegmentator
    task = 'total' if modality == 'CT' else 'total_mr'
    totalsegmentator(nifti, seg_dir, fast=fast, ml=False, device=ts_device,
                     task=task, quiet=True)
    sw.lap('segmentation')

    # 3. Mesh every non-empty structure
    print("[3/4] Meshing structures...")
    import nibabel as nib
    seg_files = sorted(glob.glob(os.path.join(seg_dir, '*.nii.gz')))
    structures = {}
    mesh_specs = []
    color_i = 0
    for f in seg_files:
        name = os.path.basename(f).replace('.nii.gz', '')
        img = nib.load(f)
        mask = np.asanyarray(img.dataobj) > 0
        if mask.sum() < 200:
            continue
        zoom = tuple(float(z) for z in img.header.get_zooms()[:3])
        mesh = build_mesh(mask, zoom, sigma=0.6, reduction=0.6, min_voxels=200)
        if mesh is None:
            continue
        save_mesh(mesh, outdir, name)
        vol_cm3 = round(mask.sum() * float(np.prod(zoom)) / 1000.0, 1)
        structures[name] = {'volume_cm3': vol_cm3, 'faces': int(mesh.n_cells)}
        color = _TS_PALETTE[color_i % len(_TS_PALETTE)]
        mesh_specs.append((mesh, color, 1.0))
        color_i += 1
        print(f"    {name}: {vol_cm3} cm3", flush=True)
    sw.lap('meshing')
    print(f"  {len(structures)} structures meshed")

    # 4. Combined anatomy render + report
    print("[4/4] Rendering + report...")
    if mesh_specs:
        render_meshes(mesh_specs[:40], outdir, 'step1_anatomy.png',
                      f'TotalSegmentator — {len(structures)} structures')
    report = {'modality': modality, 'engine': 'TotalSegmentator',
              'device': dev_name, 'fast_mode': fast,
              'num_structures': len(structures),
              'runtime_sec': round(sw.total(), 1),
              'structures': structures}
    with open(os.path.join(outdir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # cleanup intermediate NIfTI working dir
    try:
        shutil.rmtree(work)
    except Exception:
        pass
    print(f"  Total runtime: {sw.total():.0f}s on {dev_name}")


# ════════════════════════════════════════════════════════════════════════
# CT CHEST BRANCH (fast heuristic engine)
# ════════════════════════════════════════════════════════════════════════

def run_ct_chest(slices, raw_volume, spacing, outdir):
    print("\n[CT CHEST PIPELINE]")

    # 1. HU conversion
    print("[1/5] HU conversion...")
    slope = float(getattr(slices[0], 'RescaleSlope', 1))
    intercept = float(getattr(slices[0], 'RescaleIntercept', -1024))
    hu = raw_volume * slope + intercept
    print(f"  HU range: [{hu.min():.0f}, {hu.max():.0f}]")

    n = hu.shape[0]
    idxs = sorted({max(0, n//6), n//4, n//2, 3*n//4, min(n-1, 5*n//6)})

    # 2. Windowed display volume + normalization for viewing
    win_lo, win_hi = -1000, 400
    disp = np.clip((hu - win_lo) / (win_hi - win_lo), 0, 1)
    panel([('CT (windowed)', disp, 'gray')], idxs, outdir,
          'step1_windowed.png', 'CT Chest — HU Windowed [-1000, 400]')

    # 3. Body mask — per-slice fill (robust to airway/trachea connection to air)
    print("[2/5] Body / skin surface...")
    from skimage.segmentation import clear_border
    body = np.zeros_like(hu, dtype=bool)
    for i in range(n):
        b = ndimage.binary_fill_holes(hu[i] > -500)
        lbl = measure.label(b)
        if lbl.max() > 0:
            b = lbl == max(measure.regionprops(lbl), key=lambda r: r.area).label
        body[i] = ndimage.binary_fill_holes(b)

    # 4. Lungs — internal air (not connected to scan border), 2 largest components
    print("[3/5] Lung segmentation...")
    lungs = np.zeros_like(body)
    for i in range(n):
        a = (hu[i] < -320) & body[i]      # air-density voxels inside the body
        a = clear_border(a)               # drop external air touching slice edge
        lungs[i] = a
    lungs = morphology.binary_opening(lungs, morphology.ball(2))
    lbl_air = measure.label(lungs)
    regions = sorted(measure.regionprops(lbl_air), key=lambda r: r.area, reverse=True)
    lungs = np.zeros_like(body)
    for r in regions[:2]:
        if r.area > 2000:
            lungs |= (lbl_air == r.label)
    lungs = ndimage.binary_fill_holes(lungs)
    lungs = morphology.binary_closing(lungs, morphology.ball(2))
    print(f"  Lung voxels: {lungs.sum():,}")

    # 5. Skeleton: bone HU, inside body
    print("[4/5] Skeleton / bone...")
    bone = (hu > 200) & body
    bone = morphology.binary_opening(bone, morphology.ball(1))
    bone = morphology.remove_small_objects(bone, min_size=200)
    print(f"  Bone voxels: {bone.sum():,}")

    # 6. Soft tissue: inside body, not lung, not bone, mid HU
    soft = body & ~lungs & ~bone & (hu > -200) & (hu < 200)
    soft = morphology.binary_opening(soft, morphology.ball(1))
    soft = morphology.remove_small_objects(soft, min_size=500)
    print(f"  Soft-tissue voxels: {soft.sum():,}")

    # Tissue overlay visualization
    colors = {'lung': (0.3, 0.7, 1.0), 'bone': (0.95, 0.95, 0.85),
              'soft': (1.0, 0.4, 0.4)}
    over_rows = []
    comp = []
    for i in range(n):
        o = np.stack([disp[i]]*3, axis=-1) * 0.7
        o[soft[i]]  = 0.5*o[soft[i]]  + 0.5*np.array(colors['soft'])
        o[bone[i]]  = 0.3*o[bone[i]]  + 0.7*np.array(colors['bone'])
        o[lungs[i]] = 0.4*o[lungs[i]] + 0.6*np.array(colors['lung'])
        comp.append(np.clip(o, 0, 1))
    panel([('CT', disp, 'gray'), ('Segmented', comp, None)], idxs, outdir,
          'step2_segmentation.png',
          'CT Chest Segmentation — Lungs (blue) / Bone (white) / Soft tissue (red)')

    # Save masks + volume report
    voxel_cm3 = spacing[0]*spacing[1]*spacing[2] / 1000.0
    vols = {
        'lungs_cm3': round(lungs.sum()*voxel_cm3, 1),
        'skeleton_cm3': round(bone.sum()*voxel_cm3, 1),
        'soft_tissue_cm3': round(soft.sum()*voxel_cm3, 1),
        'body_cm3': round(body.sum()*voxel_cm3, 1),
    }
    np.save(os.path.join(outdir, 'mask_lungs.npy'), lungs)
    np.save(os.path.join(outdir, 'mask_skeleton.npy'), bone)
    np.save(os.path.join(outdir, 'mask_body.npy'), body)
    np.save(os.path.join(outdir, 'mask_soft_tissue.npy'), soft)
    report = {
        'modality': 'CT', 'body_part': str(getattr(slices[0], 'StudyDescription', '?')),
        'voxel_spacing_mm': [round(s, 4) for s in spacing],
        'num_slices': n, 'hu_range': [float(hu.min()), float(hu.max())],
        'volumes': vols,
    }
    with open(os.path.join(outdir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Volumes (cm3): {vols}")

    # 7. 3D meshes
    print("[5/5] 3D mesh reconstruction...")
    m_lungs = build_mesh(lungs, spacing, sigma=0.7)
    m_bone  = build_mesh(bone, spacing, sigma=0.5, reduction=0.4)
    m_body  = build_mesh(body, spacing, sigma=1.0, reduction=0.6)
    m_soft  = build_mesh(soft, spacing, sigma=0.8)

    for mesh, name in [(m_lungs, 'lungs'), (m_bone, 'skeleton'),
                       (m_body, 'body'), (m_soft, 'soft_tissue')]:
        if mesh is not None:
            save_mesh(mesh, outdir, name)
            print(f"  Saved mesh: {name}.{{vtk,stl,ply,obj}} ({mesh.n_cells:,} faces)")

    # Combined cutaway render: translucent body + bones + lungs
    render_meshes(
        [(m_body, '#e8c9a0', 0.18), (m_bone, '#f5f0dc', 1.0),
         (m_lungs, '#5bb6ff', 0.85)],
        outdir, 'step3_3d_anatomy.png',
        'CT Chest 3D — Body (translucent) + Skeleton + Lungs')
    # Lungs-only hero render
    render_meshes([(m_lungs, '#ff9ec4', 1.0)], outdir,
                  'step3_lungs.png', 'CT Chest — Lung 3D Reconstruction')


# ════════════════════════════════════════════════════════════════════════
# MRI BRAIN BRANCH
# ════════════════════════════════════════════════════════════════════════

def run_mri_brain(slices, raw_volume, spacing, outdir, use_medsam=True):
    print("\n[MRI BRAIN PIPELINE]")
    import SimpleITK as sitk
    from skimage.restoration import denoise_nl_means, estimate_sigma

    sdesc = str(getattr(slices[0], 'SeriesDescription', '')).lower()
    sequence = 'T2' if 't2' in sdesc else 'T1'
    dev, dev_name = get_device()
    print(f"  Sequence: {sequence} (from '{sdesc}') | Device: {dev_name}")
    sw = Stopwatch()

    n = raw_volume.shape[0]
    idxs = sorted({max(0, n//6), n//4, n//2, 3*n//4, min(n-1, 5*n//6)})

    # 1. Preprocessing: N4 -> NLM -> normalize
    print("[1/6] Preprocessing (N4 -> NLM -> normalize)...")
    img = sitk.GetImageFromArray(raw_volume.astype(np.float32))
    img.SetSpacing(spacing[::-1])
    mask_sitk = sitk.OtsuThreshold(img, 0, 1, 200)
    img_s = sitk.Shrink(img, [4]*3)
    mask_s = sitk.Shrink(mask_sitk, [4]*3)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corr.Execute(img_s, mask_s)
    bias = corr.GetLogBiasFieldAsImage(img)
    bias_corr = sitk.GetArrayFromImage(img / sitk.Exp(bias))

    vn = (bias_corr - bias_corr.min()) / (bias_corr.max() - bias_corr.min() + 1e-8)
    sig = np.mean(estimate_sigma(vn, channel_axis=None))
    den = denoise_nl_means(vn, h=0.8*sig, sigma=sig, fast_mode=True,
                           patch_size=3, patch_distance=5, channel_axis=None)
    plo, phi = np.percentile(den, 1), np.percentile(den, 99)
    norm = np.clip((den - plo) / (phi - plo + 1e-8), 0, 1)
    panel([('Raw', raw_volume, 'gray'), ('N4+NLM+norm', norm, 'gray')],
          idxs, outdir, 'step1_preprocessing.png',
          'MRI Brain Preprocessing (N4 bias + NLM denoise + normalize)')
    sw.lap('preprocessing')

    # 2. BET skull strip
    print("[2/6] BET skull stripping...")
    bet_mask = _bet(norm, spacing)
    sw.lap('skull strip')

    # 3. MedSAM refine (optional)
    work_mask = bet_mask
    if use_medsam and os.path.exists(MEDSAM_CHECKPOINT):
        print(f"[3/6] MedSAM refinement (device: {dev_name})...")
        work_mask = _medsam_refine(norm, bet_mask)
        sw.lap('MedSAM')
    else:
        print("[3/6] MedSAM skipped — using BET mask")

    # 4. Post-process
    print("[4/6] Post-processing...")
    final_mask = _postprocess3d(work_mask)
    mesh_mask = _postprocess3d(bet_mask)   # BET for smooth 3D surface (hybrid)
    panel([('Brain mask',
            [overlay(norm[i], final_mask[i], (0.2, 1.0, 0.3)) for i in range(n)],
            None)], idxs, outdir, 'step2_brain_mask.png',
          'Brain Mask (MedSAM-refined)')

    # 5. Tissue classification GM/WM/CSF
    print("[5/6] Tissue classification (GM/WM/CSF)...")
    tissue, vols = _classify_tissues(norm, final_mask, spacing, sequence=sequence)
    comp = []
    tc = {1: (0.2, 0.6, 1.0), 2: (1.0, 0.3, 0.3), 3: (0.95, 0.95, 0.4)}
    for i in range(n):
        o = np.stack([norm[i]]*3, axis=-1)*0.6
        for c in (1, 2, 3):
            m = tissue[i] == c
            o[m] = 0.4*o[m] + 0.6*np.array(tc[c])
        comp.append(np.clip(o, 0, 1))
    panel([('T1', norm, 'gray'), ('CSF/GM/WM', comp, None)], idxs, outdir,
          'step3_tissue.png',
          f"Tissue Classification — CSF {vols['CSF']} / GM {vols['GM']} / WM {vols['WM']} cm3")
    np.save(os.path.join(outdir, 'tissue_map.npy'), tissue)
    np.save(os.path.join(outdir, 'mask_brain.npy'), final_mask)

    sw.lap('tissue classification')
    report = {'modality': 'MR', 'body_part': 'brain', 'sequence': sequence,
              'device': dev_name,
              'voxel_spacing_mm': [round(s, 4) for s in spacing],
              'num_slices': n, 'tissue_volumes_cm3': vols,
              'total_brain_volume_cm3': round(sum(vols.values()), 1)}
    with open(os.path.join(outdir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Volumes (cm3): {vols}")

    # 6. 3D surface (hybrid: BET mask)
    print("[6/6] 3D surface reconstruction...")
    # Anisotropy-aware smoothing: heavier along the slice axis for thick/few-slice
    # volumes to reduce the 5 mm staircase; light in-plane to keep detail.
    aniso = spacing[0] / (spacing[1] + 1e-8)
    zsig = float(np.clip(0.5 * aniso, 0.6, 2.0))
    m_brain = build_mesh(mesh_mask, spacing, sigma=(zsig, 0.6, 0.6))
    m_gm = build_mesh(tissue == 2, spacing, sigma=0.6)
    m_wm = build_mesh(tissue == 3, spacing, sigma=0.6)
    if m_brain is not None:
        save_mesh(m_brain, outdir, 'brain_surface')
        print(f"  Saved mesh: brain_surface ({m_brain.n_cells:,} faces)")
    for mesh, name in [(m_gm, 'gray_matter'), (m_wm, 'white_matter')]:
        if mesh is not None:
            save_mesh(mesh, outdir, name)
    render_meshes([(m_brain, '#f7c9a8', 1.0)], outdir,
                  'step4_3d_brain.png', 'Brain 3D Surface (Hybrid BET surface)')
    sw.lap('3D reconstruction')
    print(f"  Total runtime: {sw.total():.0f}s on {dev_name}")


# ── brain helpers ───────────────────────────────────────────────────────

def _bet(norm, spacing):
    # BET's surface-deformation model is unstable (can segfault) on thin or
    # highly anisotropic volumes — guard and use an intensity fallback there.
    n_slices = norm.shape[0]
    anisotropy = max(spacing) / (min(spacing) + 1e-8)
    if n_slices < 40 or anisotropy > 3.0:
        print(f"  Thin/anisotropic volume ({n_slices} slices, "
              f"anisotropy {anisotropy:.1f}) — using intensity skull-strip")
        return _skullstrip_fallback(norm, spacing)
    try:
        from brainextractor import BrainExtractor
        import nibabel as nib
        v = np.transpose(norm, (2, 1, 0)).astype(np.float32)
        aff = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
        bet = BrainExtractor(img=nib.Nifti1Image(v, aff))
        bet.run(iterations=500)
        m = np.asarray(bet.compute_mask()).astype(bool)
        return np.transpose(m, (2, 1, 0))
    except Exception as e:
        print(f"  BET failed ({e}) — intensity fallback")
        return _skullstrip_fallback(norm, spacing)


def _skullstrip_fallback(norm, spacing):
    """Intensity + morphology skull strip for thin/anisotropic MRI."""
    from skimage.filters import threshold_otsu
    thr = threshold_otsu(norm[norm > 0])
    fg = norm > (0.5 * thr)
    # per-slice fill + largest in-plane component (robust on thin volumes)
    clean = np.zeros_like(fg)
    for i in range(fg.shape[0]):
        s = ndimage.binary_fill_holes(fg[i])
        lbl = measure.label(s)
        if lbl.max() == 0:
            continue
        clean[i] = lbl == max(measure.regionprops(lbl), key=lambda r: r.area).label
    # in-plane erosion (mm-based) to peel skull/scalp, keep largest 3D component
    rad = max(2, int(round(6.0 / spacing[1])))
    eroded = np.zeros_like(clean)
    for i in range(clean.shape[0]):
        eroded[i] = morphology.binary_erosion(clean[i], morphology.disk(rad))
    eroded &= norm > (0.4 * thr)
    lbl3 = measure.label(eroded)
    if lbl3.max() == 0:
        return clean
    brain = lbl3 == max(measure.regionprops(lbl3), key=lambda r: r.area).label
    # dilate back the eroded margin within the head
    brain = morphology.binary_dilation(
        brain, morphology.ball(1)) & clean
    for i in range(brain.shape[0]):
        brain[i] = ndimage.binary_fill_holes(brain[i])
    return brain


def _medsam_refine(norm, bet_mask):
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry['vit_b'](checkpoint=None)
    state = torch.load(MEDSAM_CHECKPOINT, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    sam.load_state_dict(state, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device).eval()
    pred = SamPredictor(sam)
    out = np.zeros_like(bet_mask)
    H, W = norm.shape[1], norm.shape[2]
    for i in range(norm.shape[0]):
        if bet_mask[i].sum() < 50:
            continue
        sl = (norm[i]*255).astype(np.uint8)
        rgb = np.stack([sl]*3, axis=-1)
        rows = np.any(bet_mask[i], axis=1); cols = np.any(bet_mask[i], axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
        box = np.array([max(0, cmin-3), max(0, rmin-3),
                        min(W-1, cmax+3), min(H-1, rmax+3)])
        pred.set_image(rgb)
        masks, _, _ = pred.predict(box=box[None, :], multimask_output=False)
        out[i] = masks[0]
    return out & ndimage.binary_dilation(bet_mask, iterations=2)


def _postprocess3d(masks):
    refined = np.zeros_like(masks)
    for i in range(masks.shape[0]):
        if masks[i].sum() == 0:
            continue
        m = morphology.closing(masks[i], morphology.disk(4))
        m = ndimage.binary_fill_holes(m)
        m = morphology.opening(m, morphology.disk(2))
        lbl = measure.label(m)
        if lbl.max() == 0:
            continue
        refined[i] = lbl == max(measure.regionprops(lbl), key=lambda r: r.area).label
    lbl3 = measure.label(refined)
    if lbl3.max() == 0:
        return refined
    return lbl3 == max(measure.regionprops(lbl3), key=lambda r: r.area).label


def _classify_tissues(norm, brain_mask, spacing, sequence='T1'):
    from sklearn.mixture import GaussianMixture
    vox = norm[brain_mask].reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, covariance_type='full',
                          max_iter=200, n_init=3, random_state=42)
    labels = gmm.fit_predict(vox)
    order = np.argsort(gmm.means_.ravel())   # dark -> bright clusters
    if sequence == 'T2':
        # T2: WM darkest, GM mid, CSF brightest
        cmap = {order[0]: 3, order[1]: 2, order[2]: 1}   # WM, GM, CSF
    else:
        # T1: CSF darkest, GM mid, WM brightest
        cmap = {order[0]: 1, order[1]: 2, order[2]: 3}   # CSF, GM, WM
    tissue = np.zeros(norm.shape, dtype=np.uint8)
    tissue[brain_mask] = np.array([cmap[l] for l in labels], dtype=np.uint8)
    for c in (1, 2, 3):
        cm = tissue == c
        lbl = measure.label(cm)
        if lbl.max() == 0:
            continue
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        tiny = np.isin(lbl, np.where(sizes < 20)[0])
        tissue[(tissue == c) & tiny] = 0
    vcm3 = spacing[0]*spacing[1]*spacing[2]/1000.0
    vols = {'CSF': round((tissue == 1).sum()*vcm3, 1),
            'GM': round((tissue == 2).sum()*vcm3, 1),
            'WM': round((tissue == 3).sum()*vcm3, 1)}
    return tissue, vols


# ════════════════════════════════════════════════════════════════════════
# PET BRANCH (functional tracer uptake — e.g. FET for brain tumor)
# ════════════════════════════════════════════════════════════════════════

def run_pet(slices, raw_volume, spacing, outdir, tbr=1.6):
    """FET/FDG PET: delineate the metabolically active tumor by
    tumor-to-background ratio (TBR), render MIPs and 3D hotspot."""
    print("\n[PET PIPELINE — tumor uptake]")

    # 1. Apply rescale to get activity, normalize for display
    slope = float(getattr(slices[0], 'RescaleSlope', 1) or 1)
    inter = float(getattr(slices[0], 'RescaleIntercept', 0) or 0)
    activity = raw_volume * slope + inter
    activity = np.clip(activity, 0, None)
    disp = activity / (np.percentile(activity, 99.5) + 1e-8)
    disp = np.clip(disp, 0, 1)
    n = activity.shape[0]
    idxs = sorted({max(0, n//6), n//4, n//2, 3*n//4, min(n-1, 5*n//6)})

    # 2. Brain/head region = where there is any uptake (largest 3D component)
    print("[1/4] Locating cerebral uptake region...")
    head = activity > (0.10 * activity.max())
    head = ndimage.binary_fill_holes(head)
    lbl = measure.label(head)
    if lbl.max() > 0:
        head = lbl == max(measure.regionprops(lbl), key=lambda r: r.area).label

    # 3. Tumor hotspot via tumor-to-background ratio
    print("[2/4] Tumor hotspot (TBR threshold)...")
    # Erode the head ~4 mm to drop bright scalp/skull-margin uptake
    rim = max(2, int(round(4.0 / spacing[1])))
    brain_core = morphology.binary_erosion(head, morphology.ball(rim))
    if not brain_core.any():
        brain_core = head

    bg_mean = float(activity[brain_core].mean()) if brain_core.any() else 0.0
    # Robust peak (99.5th percentile, ignores single-voxel noise spikes)
    core_peak = float(np.percentile(activity[brain_core], 99.5)) if brain_core.any() else 0.0
    # Tumor = above TBR background AND in the top intensity band
    thresh = max(tbr * bg_mean, 0.55 * core_peak)
    hotspot = (activity > thresh) & brain_core
    hotspot = morphology.binary_opening(hotspot, morphology.ball(1))
    hotspot = morphology.remove_small_objects(hotspot, min_size=50)

    # Keep the dominant tumor focus (+ any comparable secondary focus ≥ 40% its size)
    lblh = measure.label(hotspot)
    if lblh.max() > 0:
        regs = sorted(measure.regionprops(lblh), key=lambda r: r.area, reverse=True)
        biggest = regs[0].area
        keep = {r.label for r in regs if r.area >= 0.4 * biggest}
        hotspot = np.isin(lblh, list(keep))
    lblh = measure.label(hotspot)
    n_foci = int(lblh.max())
    print(f"  Background mean: {bg_mean:.1f} | peak: {core_peak:.1f} | threshold: {thresh:.1f}")
    print(f"  Hotspot voxels: {hotspot.sum():,} across {n_foci} focus/foci")

    # 4. Overlay panel: PET (hot colormap) + hotspot contour
    comp = []
    for i in range(n):
        rgb = plt.cm.inferno(disp[i])[..., :3]
        m = hotspot[i]
        if m.any():
            edge = m ^ ndimage.binary_erosion(m)
            rgb[edge] = [0.2, 1.0, 0.2]      # green tumor contour
        comp.append(rgb)
    panel([('PET FET (inferno)', [plt.cm.inferno(disp[i])[..., :3] for i in range(n)], None),
           ('Tumor hotspot', comp, None)],
          idxs, outdir, 'step1_pet_hotspot.png',
          f'PET FET — Tumor Hotspot (TBR ≥ {tbr})')

    # 5. Maximum-Intensity Projections (axial/coronal/sagittal)
    print("[3/4] Maximum-intensity projections...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (axis, name) in zip(axes, [(0, 'Axial'), (1, 'Coronal'), (2, 'Sagittal')]):
        mip = activity.max(axis=axis)
        ax.imshow(mip, cmap='inferno', aspect='auto')
        ax.set_title(f'{name} MIP', fontsize=12, fontweight='bold')
        ax.axis('off')
    plt.suptitle('PET FET — Maximum Intensity Projections', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'step2_mip.png'), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(outdir)}/step2_mip.png")

    # 6. Report + masks + meshes
    vcm3 = spacing[0]*spacing[1]*spacing[2]/1000.0
    report = {'modality': 'PT',
              'study': str(getattr(slices[0], 'StudyDescription', '?')),
              'voxel_spacing_mm': [round(s, 4) for s in spacing], 'num_slices': n,
              'tbr_threshold': tbr, 'background_mean': round(bg_mean, 2),
              'tumor_foci': int(n_foci),
              'tumor_volume_cm3': round(hotspot.sum()*vcm3, 2),
              'max_uptake': round(float(activity.max()), 2)}
    with open(os.path.join(outdir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(outdir, 'mask_tumor_hotspot.npy'), hotspot)
    np.save(os.path.join(outdir, 'mask_uptake_region.npy'), head)
    print(f"  Tumor volume: {report['tumor_volume_cm3']} cm3")

    print("[4/4] 3D reconstruction...")
    m_head = build_mesh(head, spacing, sigma=1.0, reduction=0.6)
    m_tumor = build_mesh(hotspot, spacing, sigma=0.6, reduction=0.3, min_voxels=30)
    if m_head is not None:
        save_mesh(m_head, outdir, 'uptake_region')
    if m_tumor is not None:
        save_mesh(m_tumor, outdir, 'tumor_hotspot')
        print(f"  Saved tumor mesh ({m_tumor.n_cells:,} faces)")
    render_meshes([(m_head, '#8899aa', 0.15), (m_tumor, '#ff3b3b', 1.0)],
                  outdir, 'step3_3d_tumor.png',
                  'PET — Tumor Hotspot (red) in Cerebral Uptake Region')


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Modality-aware DICOM segmentation pipeline")
    ap.add_argument('--input', required=True, help="Directory containing DICOM files")
    ap.add_argument('--name', required=True, help="Dataset name (output subfolder)")
    ap.add_argument('--modality', choices=['CT', 'MR', 'PT'], default=None,
                    help="Override auto-detected modality")
    ap.add_argument('--no-medsam', action='store_true', help="Skip MedSAM (brain only)")
    ap.add_argument('--engine', choices=['heuristic', 'totalseg'], default='heuristic',
                    help="Segmentation engine: fast heuristic or TotalSegmentator (DL)")
    ap.add_argument('--no-fast', action='store_true',
                    help="TotalSegmentator full resolution (1.5mm) instead of fast (3mm)")
    args = ap.parse_args()

    outdir = os.path.join(OUTPUT_ROOT, args.name)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== Dataset: {args.name} ===")
    print(f"Input:  {args.input}")
    print(f"Output: {outdir}")

    print("\n[LOAD] Reading DICOM series...")
    slices, volume, spacing = load_largest_series(args.input)
    modality = detect_modality(slices[0], args.modality)
    body = str(getattr(slices[0], 'StudyDescription', '?'))
    print(f"  Modality: {modality} | Study: {body} | "
          f"Volume: {volume.shape} | Spacing: {tuple(round(s,3) for s in spacing)}")

    if args.engine == 'totalseg':
        # Deep-learning whole-body engine (CT or MR). Takes the DICOM dir directly.
        run_totalseg(args.input, outdir,
                     modality='MR' if modality in ('MR', 'PT') else 'CT',
                     fast=not args.no_fast)
    elif modality == 'CT':
        run_ct_chest(slices, volume, spacing, outdir)
    elif modality in ('PT', 'PET'):
        run_pet(slices, volume, spacing, outdir)
    else:
        run_mri_brain(slices, volume, spacing, outdir,
                      use_medsam=not args.no_medsam)

    print(f"\n[OK] Done. All images and 3D models in: {outdir}")


if __name__ == '__main__':
    main()
