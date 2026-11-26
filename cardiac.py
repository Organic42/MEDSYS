"""Cardiac measurements from TotalSegmentator masks.

    measure(seg_dir, outdir) -> dict, also written to outdir/cardiac.json

* Chamber volumes in ml, when the heart-chamber masks exist
  (TotalSegmentator task heartchambers_highres), plus the whole heart.
* Aorta diameter along its centreline. The centreline is the 3-D skeleton of
  the aorta mask; the diameter at each centreline point is the largest sphere
  that fits inside the mask there (2 x distance to the wall, less one voxel,
  because the distance runs from voxel centre to the first outside voxel
  centre). That is the vessel's smallest width at that point, so it can read a
  little lower than a hand measurement of an oval cross-section.

Points are in mesh coordinates (array index * spacing), the same frame as the
STL files, so the viewer can mark them on the model. Research measurement,
not for diagnosis; the scan is not ECG-gated, so chamber volumes are at an
unknown point in the heartbeat.
"""
import json
import os

import numpy as np

CHAMBERS = [
    ('heart_ventricle_left', 'Left ventricle'),
    ('heart_ventricle_right', 'Right ventricle'),
    ('heart_atrium_left', 'Left atrium'),
    ('heart_atrium_right', 'Right atrium'),
    ('heart_myocardium', 'Myocardium (LV wall)'),
    ('atrial_appendage_left', 'Left atrial appendage'),
]
CHAMBER_NAMES = {n for n, _ in CHAMBERS} - {'atrial_appendage_left'}


def _load(path):
    import nibabel as nib
    img = nib.load(path)
    mask = np.asanyarray(img.dataobj) > 0
    return mask, tuple(float(z) for z in img.header.get_zooms()[:3]), nib.aff2axcodes(img.affine)


def _axis(axcodes, pos, neg):
    """(axis index, +1/-1) of the array axis that runs toward `pos`."""
    for i, c in enumerate(axcodes):
        if c == pos:
            return i, 1
        if c == neg:
            return i, -1
    return None, 1


def aorta_diameters(mask, spacing, axcodes):
    """Centreline diameters of the aorta, split into ascending, arch and
    descending where the anatomy allows it."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    lbl, n = ndimage.label(mask)
    if n > 1:                                        # keep the aorta, drop specks
        sizes = ndimage.sum(mask, lbl, range(1, n + 1))
        mask = lbl == (int(np.argmax(sizes)) + 1)
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    m = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    m = np.pad(m, 1)
    edt = ndimage.distance_transform_edt(m, sampling=spacing)
    skel = skeletonize(m)
    pts = np.argwhere(skel)
    if len(pts) == 0:
        return None
    vox = float(min(spacing))
    diam = np.maximum(2.0 * edt[skel] - vox, 0.0)
    world = (pts - 1 + lo) * np.asarray(spacing)      # mesh coordinates

    # Regions: the arch is above the highest axial slice that still cuts the
    # aorta twice; below it, the more anterior cut is the ascending aorta.
    si_ax, si_sign = _axis(axcodes, 'S', 'I')
    ap_ax, ap_sign = _axis(axcodes, 'A', 'P')
    region = np.full(len(pts), 'aorta', dtype=object)
    if si_ax is not None and ap_ax is not None:
        moved = np.moveaxis(m, si_ax, 0)
        levels = []
        for k in range(moved.shape[0]):
            ll, cnt = ndimage.label(moved[k])
            if cnt >= 2:
                sizes = ndimage.sum(moved[k], ll, range(1, cnt + 1))
                if np.sort(sizes)[-2] >= 20:          # two real cuts, not a speck
                    levels.append(k)
        if levels:
            top = max(levels) if si_sign > 0 else min(levels)
            for j, p in enumerate(pts):
                k = p[si_ax]
                above = k > top if si_sign > 0 else k < top
                if above:
                    region[j] = 'arch'
                    continue
                sl = np.moveaxis(m, si_ax, 0)[k]
                ll, cnt = ndimage.label(sl)
                if cnt < 2:
                    region[j] = 'descending'
                    continue
                sizes = ndimage.sum(sl, ll, range(1, cnt + 1))
                two = np.argsort(sizes)[-2:] + 1
                ap_in_slice = ap_ax if ap_ax < si_ax else ap_ax - 1
                cents = {int(t): ndimage.center_of_mass(ll == t)[ap_in_slice] for t in two}
                q = [c for c in range(3) if c != si_ax]
                mine = int(ll[p[q[0]], p[q[1]]])
                if mine not in cents:
                    region[j] = 'descending'
                    continue
                other = [t for t in cents if t != mine][0]
                anterior = (cents[mine] - cents[other]) * ap_sign > 0
                region[j] = 'ascending' if anterior else 'descending'

    sp = np.asarray(spacing, dtype=float)

    def cross_section(j):
        """Cut perpendicular to the centreline at skeleton point j, as a
        surgeon measures on a double-oblique view: largest width across the
        cut and the width of a circle of the same area."""
        from scipy.spatial import ConvexHull
        near = world[np.linalg.norm(world - world[j], axis=1) < 12.0]
        if len(near) < 5:
            return None
        t = np.linalg.svd(near - near.mean(0), full_matrices=False)[2][0]   # centreline direction
        u = np.cross(t, [1.0, 0, 0] if abs(t[0]) < .9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
        v = np.cross(t, u)
        step, R = 0.25, 45.0
        g = np.arange(-R, R + step, step)
        a, b = np.meshgrid(g, g, indexing='ij')
        P = world[j] + a[..., None] * u + b[..., None] * v                # mm, mesh frame
        ijk = P / sp - lo + 1                                              # index into padded crop
        # Linear interpolation, threshold at half: follows the wall between
        # voxels instead of their staircase, which would add up to a voxel.
        inside = ndimage.map_coordinates(m.astype(np.float32), ijk.reshape(-1, 3).T, order=1,
                                         mode='constant').reshape(a.shape) >= 0.5
        ll, _ = ndimage.label(inside)
        c = ll[len(g) // 2, len(g) // 2]
        if c == 0:
            return None
        sel2 = ll == c
        if sel2[0].any() or sel2[-1].any() or sel2[:, 0].any() or sel2[:, -1].any():
            return None                                                    # cut ran along the vessel
        pts2 = np.c_[a[sel2], b[sel2]]
        hull = pts2[ConvexHull(pts2).vertices]
        dmax = float(np.max(np.linalg.norm(hull[:, None] - hull[None], axis=2))) + step
        area = sel2.sum() * step * step
        return {'max_mm': round(dmax, 1), 'equiv_mm': round(2 * float(np.sqrt(area / np.pi)), 1)}

    # Points near an open end of the centreline (the scan's edge, the valve, a
    # branch tip) have neighbours on one side only, and a cut there is not a
    # clean cross-section. Prefer points with centreline on both sides.
    from scipy.spatial import cKDTree
    tree = cKDTree(world)
    near_end = np.array([np.linalg.norm(world[nb].mean(0) - world[i]) > 4.0
                         for i, nb in enumerate(tree.query_ball_point(world, 12.0))])

    def best(sel):
        if not sel.any():
            return None
        if (sel & ~near_end).any():
            sel = sel & ~near_end
        j = int(np.argmax(np.where(sel, diam, -1)))
        out = {'max_mm': round(float(diam[j]), 1), 'point_mm': [round(float(v), 1) for v in world[j]]}
        try:
            cs = cross_section(j)
        except Exception:  # noqa: BLE001  (an extra figure must never cost the run)
            cs = None
        if cs:
            out['cross_section'] = cs
        return out

    out = {'max': best(np.ones(len(pts), bool)), 'regions': {}}
    for r in ('ascending', 'arch', 'descending'):
        b = best(region == r)
        if b:
            out['regions'][r] = b
    order = np.argsort(world[:, si_ax if si_ax is not None else 2])
    out['profile_mm'] = [round(float(v), 1) for v in diam[order][::max(1, len(order) // 200)]]
    out['method'] = ('largest inscribed sphere along the mask skeleton '
                     '(smallest width at each point)')
    return out


def measure(seg_dir, outdir):
    res = {'note': 'Research measurement, not for diagnosis. Non-gated scan: chamber '
                   'volumes are at an unknown point in the heartbeat.'}
    chambers = []
    spacing = axcodes = None
    for name, label in CHAMBERS:
        p = os.path.join(seg_dir, f'{name}.nii.gz')
        if not os.path.exists(p):
            continue
        mask, spacing, axcodes = _load(p)
        n = int(mask.sum())
        if n:
            chambers.append({'name': name, 'label': label,
                             'volume_ml': round(n * float(np.prod(spacing)) / 1000.0, 1)})
    res['chambers'] = chambers
    hp = os.path.join(seg_dir, 'heart.nii.gz')
    if os.path.exists(hp):
        mask, spacing, axcodes = _load(hp)
        res['heart_volume_ml'] = round(int(mask.sum()) * float(np.prod(spacing)) / 1000.0, 1)
    ap = os.path.join(seg_dir, 'aorta.nii.gz')
    if os.path.exists(ap):
        mask, spacing, axcodes = _load(ap)
        if mask.any():
            res['aorta'] = aorta_diameters(mask, spacing, axcodes)
    if axcodes:
        res['axcodes'] = list(axcodes)
    if not chambers and 'aorta' not in res and 'heart_volume_ml' not in res:
        return None
    with open(os.path.join(outdir, 'cardiac.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == '__main__':
    import sys
    for ds in sys.argv[1:]:
        r = measure(os.path.join(ds, 'segmentations'), ds)
        a = (r or {}).get('aorta') or {}
        print(ds, 'heart', (r or {}).get('heart_volume_ml'), 'ml;',
              'aorta max', (a.get('max') or {}).get('max_mm'), 'mm;',
              {k: v['max_mm'] for k, v in a.get('regions', {}).items()},
              'chambers', [(c['label'], c['volume_ml']) for c in (r or {}).get('chambers', [])])
