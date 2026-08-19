"""
Segmentation accuracy validation — Dice / IoU / Hausdorff-95 / ASSD.

Reusable metrics + a demo that scores the fast heuristic engine against
TotalSegmentator (treated as the validated reference) on the same CT volume,
in identical NIfTI geometry so the comparison is voxel-exact.

Usage:
  python validate.py --dicom "C:\\path\\to\\ct_dicoms" \\
                     --totalseg output/chest_ct_ai/segmentations \\
                     --out output/chest_ct_ai/validation
"""
import os
import glob
import json
import argparse
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from skimage.segmentation import clear_border
# matplotlib imported lazily in main() so compute_metrics stays import-light for CI


# ── metrics ──────────────────────────────────────────────────────────────

def _surface(mask):
    return mask ^ ndimage.binary_erosion(mask)


def compute_metrics(pred, gt, spacing):
    """Overlap + surface-distance metrics between two boolean volumes.
    spacing is voxel size (mm) in the array's axis order."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    psum, gsum = pred.sum(), gt.sum()
    dice = 2.0 * inter / (psum + gsum) if (psum + gsum) else 1.0
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    vox_ml = float(np.prod(spacing)) / 1000.0

    out = {'dice': round(float(dice), 4), 'iou': round(float(iou), 4),
           'pred_cm3': round(psum * vox_ml, 1), 'gt_cm3': round(gsum * vox_ml, 1)}

    if psum == 0 or gsum == 0:
        out.update({'hd95_mm': None, 'assd_mm': None})
        return out

    ps, gs = _surface(pred), _surface(gt)
    dt_to_gt = distance_transform_edt(~gs, sampling=spacing)
    dt_to_pred = distance_transform_edt(~ps, sampling=spacing)
    d_pg = dt_to_gt[ps]      # pred surface -> gt
    d_gp = dt_to_pred[gs]    # gt surface -> pred
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    assd = (d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp))
    out.update({'hd95_mm': round(float(hd95), 2), 'assd_mm': round(float(assd), 2)})
    return out


# ── heuristic masks on a NIfTI HU volume (geometry-matched to reference) ──

def heuristic_lungs(hu):
    """Per-axial-slice (z = axis 2) body fill + internal-air isolation."""
    nz = hu.shape[2]
    lungs = np.zeros(hu.shape, dtype=bool)
    for k in range(nz):
        sl = hu[:, :, k]
        body = ndimage.binary_fill_holes(sl > -500)
        lbl = measure.label(body)
        if lbl.max() > 0:
            body = lbl == max(measure.regionprops(lbl), key=lambda r: r.area).label
            body = ndimage.binary_fill_holes(body)
        a = (sl < -320) & body
        lungs[:, :, k] = clear_border(a)
    lungs = morphology.opening(lungs, morphology.ball(2))
    lbl = measure.label(lungs)
    regs = sorted(measure.regionprops(lbl), key=lambda r: r.area, reverse=True)
    out = np.zeros_like(lungs)
    for r in regs[:2]:
        if r.area > 2000:
            out |= (lbl == r.label)
    return ndimage.binary_fill_holes(out)


def heuristic_bone(hu):
    bone = hu > 200
    bone = morphology.opening(bone, morphology.ball(1))
    return morphology.remove_small_objects(bone, min_size=200)


# ── reference unions from TotalSegmentator outputs ───────────────────────

LUNG_PARTS = ['lung_upper_lobe_left', 'lung_lower_lobe_left',
              'lung_upper_lobe_right', 'lung_middle_lobe_right',
              'lung_lower_lobe_right']
# The HU>200 heuristic targets the thoracic skeleton (cortical bone) — compare
# like-with-like against ribs + vertebrae + sternum (exclude scapula/humerus/
# clavicle/costal-cartilage which the heuristic doesn't attempt).
BONE_PREFIXES = ('rib_', 'vertebrae_', 'sternum')


def _union(seg_dir, names_or_prefixes, ref_shape, prefix=False):
    import nibabel as nib
    out = np.zeros(ref_shape, dtype=bool)
    for f in glob.glob(os.path.join(seg_dir, '*.nii.gz')):
        name = os.path.basename(f).replace('.nii.gz', '')
        hit = (any(name.startswith(p) for p in names_or_prefixes) if prefix
               else name in names_or_prefixes)
        if hit:
            out |= np.asanyarray(nib.load(f).dataobj) > 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dicom', required=True)
    ap.add_argument('--totalseg', required=True, help="TotalSegmentator segmentations dir")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import nibabel as nib
    import dicom2nifti
    import dicom2nifti.settings as dset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dset.disable_validate_slice_increment()

    print("Converting DICOM -> NIfTI (matching TotalSegmentator geometry)...")
    work = os.path.join(args.out, '_work')
    os.makedirs(work, exist_ok=True)
    dicom2nifti.convert_directory(args.dicom, work, compression=True, reorient=True)
    nii = max(glob.glob(os.path.join(work, '*.nii.gz')), key=os.path.getsize)
    img = nib.load(nii)
    hu = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    print(f"  Volume {hu.shape}, spacing {spacing} mm, HU [{hu.min():.0f},{hu.max():.0f}]")

    print("Building heuristic masks + reference unions...")
    results = {}
    pred_lung = heuristic_lungs(hu)
    ref_lung = _union(args.totalseg, LUNG_PARTS, hu.shape)
    results['lungs'] = compute_metrics(pred_lung, ref_lung, spacing)

    pred_bone = heuristic_bone(hu)
    ref_bone = _union(args.totalseg, BONE_PREFIXES, hu.shape, prefix=True)
    results['skeleton'] = compute_metrics(pred_bone, ref_bone, spacing)

    report = {
        'reference': 'TotalSegmentator (nnU-Net)',
        'candidate': 'fast heuristic engine',
        'spacing_mm': [round(s, 3) for s in spacing],
        'metrics': results,
        'interpretation': {
            'dice': 'overlap, 1.0 = perfect (>0.9 excellent, 0.7-0.9 good)',
            'hd95_mm': 'boundary error, lower is better',
            'assd_mm': 'mean surface distance, lower is better',
        },
    }
    with open(os.path.join(args.out, 'validation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print("\n=== Validation vs TotalSegmentator ===")
    for k, v in results.items():
        print(f"  {k:9s} Dice={v['dice']}  IoU={v['iou']}  "
              f"HD95={v['hd95_mm']}mm  ASSD={v['assd_mm']}mm  "
              f"(pred {v['pred_cm3']} vs ref {v['gt_cm3']} cm3)")

    # overlap visualization on 5 axial slices (last axis assumed z after reorient)
    nz = hu.shape[2]
    idxs = [nz//6, nz//3, nz//2, 2*nz//3, 5*nz//6]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.4))
    win = np.clip((hu + 1000) / 1400, 0, 1)
    for col, i in enumerate(idxs):
        for row, (pred, ref, title) in enumerate(
                [(pred_lung, ref_lung, 'Lungs'), (pred_bone, ref_bone, 'Skeleton')]):
            ax = axes[row, col]
            base = np.stack([win[:, :, i]] * 3, axis=-1)
            p, g = pred[:, :, i], ref[:, :, i]
            base[np.logical_and(p, g)] = [0.2, 1.0, 0.2]   # agreement = green
            base[np.logical_and(p, ~g)] = [1.0, 0.3, 0.3]  # heuristic only = red
            base[np.logical_and(~p, g)] = [0.3, 0.5, 1.0]  # reference only = blue
            ax.imshow(np.clip(base, 0, 1)); ax.axis('off')
            ax.set_title(f'{title} z{i}', fontsize=9)
    plt.suptitle('Heuristic vs TotalSegmentator — green=agree, red=heuristic-only, blue=ref-only',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'validation_overlap.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {args.out}/validation_report.json + validation_overlap.png")

    import shutil
    try:
        shutil.rmtree(work)
    except Exception:
        pass


if __name__ == '__main__':
    main()
