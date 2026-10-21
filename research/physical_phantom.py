"""Validation test 3: printed phantoms in a real CT scanner.

The Tier 1 shapes made physical. Each object has an exact signed distance
function (research/phantoms.py), so a CT scan of the printed kit can be scored
against the true surface, after a rigid fit, with no reference mesh involved:

    design     STL files to print, kit.json and a sheet of caliper checks
    simulate   a synthetic CT of the kit, written as DICOM, to test the analysis
               before the real scan (partial volume, blur, noise, random poses)
    analyse    DICOM (or NIfTI) of the scanned kit -> per object and per
               reconstruction: canonical and production meshes from the app's
               own code, scored against the exact surface; the accuracy card's
               resolution limit is checked against the canonical error

    python -m research.physical_phantom design --out docs/validation/phantom
    python -m research.physical_phantom simulate --out output/research/phantom_sim --spacing 0.7,0.7,1.25
    python -m research.physical_phantom analyse <dicom dir or .nii.gz> [...] --out output/research/physical_phantom/<run>

Registration minimises the exact distance of the scanned surface to the true
one over a rigid pose (6 parameters), as phantom studies align a surface scan
to its CAD model. It absorbs any rigid offset, so the errors are shape errors:
what segmentation and reconstruction did to the geometry, not where the
object lay. A physical object has no other true pose.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.phantoms import (Capsule, Phantom, Placement, RoundedBox, Sphere, Torus,  # noqa: E402
                               _sdf_gradient_local, sample_surface)

N_SAMPLES = 20_000


# ── the kit ──────────────────────────────────────────────────────────────

def _segment_sdf(p, a, b, r):
    """Exact SDF of a capsule from a to b with radius r."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) - r


class Branch(Phantom):
    """A Y-shaped vessel: a trunk splitting into two thinner branches in the
    local xz plane. The union of capsules; its SDF is exact on the surface and
    away from the junction, which is enough for surface scoring. Volume and
    area have no closed form and are measured once on a fine grid."""
    name = 'branch'

    def __init__(self, trunk_r=3.0, trunk_len=15.0, branch_r=2.0, branch_len=18.0, angle_deg=35.0):
        a = np.radians(angle_deg)
        self.segs = [(np.array([0.0, 0.0, -trunk_len]), np.zeros(3), float(trunk_r)),
                     (np.zeros(3), np.array([branch_len * np.sin(a), 0.0, branch_len * np.cos(a)]), float(branch_r)),
                     (np.zeros(3), np.array([-branch_len * np.sin(a), 0.0, branch_len * np.cos(a)]), float(branch_r))]
        pts = np.array([p for s in self.segs for p in s[:2]])
        r = max(s[2] for s in self.segs)
        self._lo, self._hi = pts.min(axis=0) - r, pts.max(axis=0) + r
        self._centre = (self._lo + self._hi) / 2
        self._vol = self._area = None

    def sdf_local(self, p):
        p = np.asarray(p, dtype=float) + self._centre        # local origin at the bounding-box centre
        return np.min([_segment_sdf(p, a, b, r) for a, b, r in self.segs], axis=0)

    def _measure(self, h=0.08):
        verts, faces = exact_mesh(self, h)
        from research.surface_metrics import mesh_area, mesh_volume
        self._vol, self._area = float(mesh_volume(verts, faces)), float(mesh_area(verts, faces))

    @property
    def volume(self):
        if self._vol is None:
            self._measure()
        return self._vol

    @property
    def area(self):
        if self._area is None:
            self._measure()
        return self._area

    @property
    def half_extents(self):
        return (self._hi - self._lo) / 2

    @property
    def bounding_radius(self):
        return float(np.linalg.norm(self.half_extents))


def kit():
    """The objects to print, by name. Sizes follow Tier 1 and span vessel calibres."""
    return {
        'sphere_r15': Sphere(15.0),
        'rounded_box': RoundedBox((14.0, 10.0, 6.0), 2.0),
        'torus': Torus(18.0, 6.0),
        'capsule_r1': Capsule(1.0, 20.0),
        'capsule_r2': Capsule(2.0, 20.0),
        'capsule_r3': Capsule(3.0, 20.0),
        'branch': Branch(),
    }


def caliper_checks(name, ph):
    """What to measure on the printed object, with the design value (mm)."""
    if isinstance(ph, Sphere):
        return [('diameter', 2 * ph.r)]
    if isinstance(ph, RoundedBox):
        return [(f'edge {a}', 2 * (b + ph.rr)) for a, b in zip('xyz', ph.b)]
    if isinstance(ph, Torus):
        return [('outer diameter', 2 * (ph.R + ph.r)), ('tube thickness', 2 * ph.r)]
    if isinstance(ph, Capsule):
        return [('diameter', 2 * ph.r), ('length', 2 * (ph.h + ph.r))]
    if isinstance(ph, Branch):
        return [('trunk diameter', 2 * ph.segs[0][2]), ('branch diameter', 2 * ph.segs[1][2]),
                ('overall length', float(ph._hi[2] - ph._lo[2]))]
    return []


def exact_mesh(ph, h):
    """A mesh of the exact surface: marching cubes on the SDF at spacing h, then
    every vertex projected onto the surface along the SDF gradient."""
    from skimage.measure import marching_cubes
    he = ph.half_extents + 3 * h
    axes = [np.arange(-e, e + h, h) for e in he]
    g = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    sdf = ph.sdf_local(g.reshape(-1, 3)).reshape(g.shape[:3])
    v, f, _, _ = marching_cubes(sdf, level=0.0, spacing=(h, h, h))
    v = v - he
    for _ in range(2):
        v = v - ph.sdf_local(v)[:, None] * _sdf_gradient_local(ph, v)
    return v, f


def principal_frame(points):
    """Centroid, principal axes (columns, right-handed) and standard deviations."""
    c = points.mean(axis=0)
    w, vec = np.linalg.eigh(np.cov((points - c).T))
    order = np.argsort(w)[::-1]
    vec = vec[:, order]
    if np.linalg.det(vec) < 0:
        vec[:, 2] *= -1
    return c, vec, np.sqrt(np.maximum(w[order], 0))


def shape_signature(ph, h=0.3):
    """Volume and principal standard deviations of the solid, for matching."""
    he = ph.half_extents + h
    axes = [np.arange(-e, e + h, h) for e in he]
    g = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    inside = g[ph.sdf_local(g) < 0]
    _, axes_, sd = principal_frame(inside)
    return {'volume': float(len(inside) * h ** 3), 'sd': sd, 'axes': axes_, 'centroid': inside.mean(axis=0)}


# ── design ───────────────────────────────────────────────────────────────

def design(out):
    import pyvista as pv
    os.makedirs(out, exist_ok=True)
    meta = {'units': 'mm', 'objects': {}}
    for name, ph in kit().items():
        h = 0.1 if getattr(ph, 'r', 3) <= 2 else 0.25         # chord error < 0.005 mm, far below a printer's
        v, f = exact_mesh(ph, h)
        mesh = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64))
        mesh.save(os.path.join(out, f'{name}.stl'), binary=True)
        meta['objects'][name] = {'shape': type(ph).__name__, 'volume_mm3': round(float(ph.volume), 2),
                                 'area_mm2': round(float(ph.area), 2), 'triangles': int(len(f)),
                                 'checks': [{'what': w, 'design_mm': round(float(d), 2)} for w, d in caliper_checks(name, ph)]}
        print(f'{name}: {len(f):,} triangles, volume {ph.volume:.1f} mm3')
    with open(os.path.join(out, 'kit.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1)
    rows = ['object,what,design_mm,measured_mm']
    for name, m in meta['objects'].items():
        rows += [f"{name},{c['what']},{c['design_mm']}," for c in m['checks']]
    with open(os.path.join(out, 'calipers.csv'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(rows) + '\n')
    print(f'kit -> {out}')


# ── a synthetic scan of the kit ─────────────────────────────────────────

def render(objects, spacing, shape, background=-1000.0, material=150.0, psf_mm=0.5, noise_hu=12.0,
           supersample=3, rng=None):
    """CT-like volume: partial-volume occupancy by supersampling each voxel,
    a Gaussian point spread and white noise. objects: [(phantom, placement)]."""
    from scipy.ndimage import gaussian_filter
    from research.phantoms import sdf_world
    rng = rng or np.random.default_rng(0)
    spacing = np.asarray(spacing, float)
    occ = np.zeros(shape, np.float32)
    sub = (np.arange(supersample) + 0.5) / supersample - 0.5
    offs = np.stack(np.meshgrid(sub, sub, sub, indexing='ij'), -1).reshape(-1, 3) * spacing
    for ph, pl in objects:
        r = ph.bounding_radius + 2 * spacing.max()
        lo = np.maximum(np.floor((pl.center - r) / spacing).astype(int), 0)
        hi = np.minimum(np.ceil((pl.center + r) / spacing).astype(int) + 1, shape)
        idx = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing='ij'), -1)
        centres = idx.reshape(-1, 3) * spacing
        frac = np.zeros(len(centres), np.float32)
        for o in offs:
            frac += sdf_world(ph, centres + o, pl) < 0
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        occ[sl] = np.maximum(occ[sl], (frac / len(offs)).reshape(idx.shape[:3]))
    vol = background + (material - background) * occ
    vol = gaussian_filter(vol, sigma=psf_mm / spacing)
    return (vol + rng.normal(0, noise_hu, vol.shape)).astype(np.float32)


def write_dicom(vol, spacing, out):
    """Axial CT slices (array axes x, y, z; z = slice), 16-bit with a rescale."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    os.makedirs(out, exist_ok=True)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    for k in range(vol.shape[2]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.PatientName, ds.PatientID = 'CT', 'PHANTOM^SIMULATED', 'PHANTOM'
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.SeriesDescription = f'simulated phantom {spacing[0]:g}x{spacing[1]:g}x{spacing[2]:g}'
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(k * spacing[2])]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]     # row spacing (y), column spacing (x)
        ds.SliceThickness = float(spacing[2])
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[0]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.PixelData = np.round(vol[:, :, k].T).astype(np.int16).tobytes()
        ds.save_as(os.path.join(out, f'slice_{k:04d}.dcm'), enforce_file_format=True)


def simulate(out, spacing, seed=2026, medium='air'):
    """Place the kit at random, non-overlapping poses and write a DICOM series."""
    rng = np.random.default_rng(seed)
    objs, centres = [], []
    spacing = np.asarray(spacing, float)
    box = np.array([260.0, 260.0, 80.0])
    for name, ph in kit().items():
        for _ in range(500):
            c = rng.uniform(ph.bounding_radius + 6, box - ph.bounding_radius - 6)
            if all(np.linalg.norm(c - c2) > ph.bounding_radius + r2 + 20 for c2, r2 in centres):
                break
        q = rng.normal(size=4)
        objs.append((name, ph, Placement(Rotation.from_quat(q / np.linalg.norm(q)), c)))
        centres.append((c, ph.bounding_radius))
    shape = np.ceil(box / spacing).astype(int)
    bg = -1000.0 if medium == 'air' else 0.0
    vol = render([(ph, pl) for _, ph, pl in objs], spacing, tuple(shape), background=bg, rng=rng)
    write_dicom(vol, spacing, out)
    truth = {n: {'rotvec': pl.rotation.as_rotvec().tolist(), 'center': pl.center.tolist()} for n, _, pl in objs}
    with open(os.path.join(out, 'truth.json'), 'w', encoding='utf-8') as fh:
        json.dump({'spacing': spacing.tolist(), 'medium': medium, 'objects': truth}, fh, indent=1)
    print(f'simulated scan {shape} at {spacing} mm -> {out}')


# ── analysis ─────────────────────────────────────────────────────────────

def load_volume(path):
    """HU volume with array axes (x, y, z) in mm spacing, from a DICOM folder or NIfTI."""
    import SimpleITK as sitk
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(path)
        if not series:
            raise RuntimeError(f'no DICOM series in {path}')
        files = max((reader.GetGDCMSeriesFileNames(path, s) for s in series), key=len)
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32).transpose(2, 1, 0)
    return vol, tuple(float(s) for s in img.GetSpacing())


def threshold(vol):
    """Half-way between background and material: the classical 50% surface."""
    from skimage.filters import threshold_otsu
    sample = vol[::2, ::2, ::2].ravel()
    t = threshold_otsu(sample)
    bg, mat = np.median(sample[sample < t]), np.median(sample[sample >= t])
    return float((bg + mat) / 2), float(bg), float(mat)


def components(mask, spacing, min_mm3=10.0, gap_mm=6.0):
    """Scanned objects: connected pieces, with pieces less than gap_mm apart
    grouped into one object (thick slices break thin tubes apart, as Tier 1
    found), and anything touching the scan border (the table, a container) or
    smaller than min_mm3 dropped. The scan protocol keeps objects at least
    20 mm apart."""
    from scipy.ndimage import binary_erosion, find_objects, label
    from scipy.spatial import cKDTree
    lab, _ = label(mask, structure=np.ones((3, 3, 3)))
    spacing = np.asarray(spacing, float)
    pieces = []
    for i, sl in enumerate(find_objects(lab), 1):
        if sl is None or any(s.start == 0 or s.stop == mask.shape[d] for d, s in enumerate(sl)):
            continue
        sub = lab[sl] == i
        edge = sub & ~binary_erosion(sub)
        pts = (np.argwhere(edge) + [s.start for s in sl]) * spacing
        pieces.append((i, np.array([s.start for s in sl]), np.array([s.stop for s in sl]), cKDTree(pts)))
    parent = list(range(len(pieces)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    pad = np.ceil(gap_mm / spacing).astype(int)
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            _, lo_a, hi_a, tree_a = pieces[a]
            _, lo_b, hi_b, tree_b = pieces[b]
            if not (np.all(lo_a - pad < hi_b) and np.all(lo_b - pad < hi_a)):
                continue                                       # boxes too far apart to be close
            d, _ = tree_b.query(tree_a.data, k=1, distance_upper_bound=gap_mm)
            if np.isfinite(d).any():
                parent[root(a)] = root(b)
    groups = {}
    for k, piece in enumerate(pieces):
        groups.setdefault(root(k), []).append(piece)
    out, vox = [], float(np.prod(spacing))
    for members in groups.values():
        lo = np.min([m[1] for m in members], axis=0)
        hi = np.max([m[2] for m in members], axis=0)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        sub = np.isin(lab[sl], [m[0] for m in members])
        if sub.sum() * vox >= min_mm3:
            out.append((sl, sub, len(members)))
    return out


def match(found, spacing, shapes):
    """Assign scanned objects to kit objects by volume and principal spreads."""
    from scipy.optimize import linear_sum_assignment
    names = list(shapes)
    sig = {n: shape_signature(ph) for n, ph in shapes.items()}
    feats = []
    for sl, sub, _ in found:
        pts = np.argwhere(sub) * spacing
        _, _, sd = principal_frame(pts)
        feats.append((sub.sum() * np.prod(spacing), sd))
    cost = np.zeros((len(found), len(names)))
    for i, (v, sd) in enumerate(feats):
        for j, n in enumerate(names):
            cost[i, j] = abs(np.log(v / sig[n]['volume'])) + np.abs(np.log((sd + 0.3) / (sig[n]['sd'] + 0.3))).sum()
    rows, cols = linear_sum_assignment(cost)
    return [(found[r], names[c], float(cost[r, c])) for r, c in zip(rows, cols)], sig


def register(ph, sig, points):
    """Rigid pose (Placement) minimising the exact SDF of the scanned surface points."""
    from scipy.optimize import least_squares
    c, axes, _ = principal_frame(points)
    best = None
    for flip in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        r0 = Rotation.from_matrix(axes @ np.diag(flip) @ sig['axes'].T)
        t0 = c - r0.apply(sig['centroid'])

        def resid(x):
            return ph.sdf_local(Rotation.from_rotvec(x[:3]).inv().apply(points - x[3:]))
        sol = least_squares(resid, np.concatenate([r0.as_rotvec(), t0]), loss='soft_l1', f_scale=0.5)
        if best is None or sol.cost < best.cost:
            best = sol
    return Placement(Rotation.from_rotvec(best.x[:3]), best.x[3:])


def analyse_one(path, rng, log=print):
    import accuracy
    from reconstruction import ReconParams, extract_isosurface, mesh_arrays, postprocess_surface
    from research.surface_metrics import compare_to_phantom, sample_mesh_surface
    vol, spacing = load_volume(path)
    thr, bg, mat = threshold(vol)
    shapes = kit()
    found = components(vol > thr, spacing)
    matched, sig = match(found, spacing, shapes)
    label = os.path.basename(os.path.normpath(path))
    log(f'{label}: spacing {spacing}, threshold {thr:.0f} HU (background {bg:.0f}, material {mat:.0f}), '
        f'{len(found)} objects')
    rows = []
    lim = accuracy.resolution_limit(spacing)
    configs = {'canonical': ReconParams(),
               'production': ReconParams(blur_sigma_vox=(0.6,) * 3, isosurface='lewiner', taubin_iter=30,
                                         decimation=0.5)}
    base = {'scan': label, 'spacing': 'x'.join(f'{s:g}' for s in spacing), 'max_spacing_mm': max(spacing),
            'rms_spacing_mm': accuracy.added_sampling(spacing), 'threshold_hu': thr, 'background_hu': bg,
            'material_hu': mat, 'card_resolution_mm': lim['estimate_mm'],
            'card_resolution_lo_mm': lim['range_mm'][0], 'card_resolution_hi_mm': lim['range_mm'][1]}
    for (sl, sub, pieces), name, cost in matched:
        ph = shapes[name]
        obj = dict(base, object=name, match_cost=cost, scanned_pieces=pieces)
        origin = tuple(float(s.start) * sp - 2 * sp for s, sp in zip(sl, spacing))
        pad = np.pad(sub, 2)
        try:
            cv, cf = extract_isosurface(pad, spacing, ReconParams(), origin)
            pts = sample_mesh_surface(cv, cf, 5000, rng)
            pl = register(ph, sig[name], pts)
        except (ValueError, RuntimeError) as exc:
            rows += [dict(obj, config=cfg, status=f'registration failed: {str(exc)[:80]}') for cfg in configs]
            log(f'  {name:12s} registration failed: {exc}')
            continue
        truth = sample_surface(ph, pl, N_SAMPLES, rng)
        obj.update(pose_rotvec=pl.rotation.as_rotvec().round(5).tolist(), pose_center=np.round(pl.center, 4).tolist())
        for cfg, params in configs.items():
            try:
                v, f = extract_isosurface(pad, spacing, params, origin)
                v, f = mesh_arrays(postprocess_surface(v, f, params))
                rows.append(dict(obj, config=cfg, status='ok',
                                 **compare_to_phantom(v, f, ph, pl, truth, rng, n_mesh_samples=N_SAMPLES)))
            except (ValueError, RuntimeError):
                # the blur left nothing above 0.5: the object vanished from this mesh
                rows.append(dict(obj, config=cfg, status='no surface'))
        r = rows[-2]
        log(f'  {name:12s} canonical ASSD {r.get("assd_mm", float("nan")):.3f} mm, '
            f'production: {rows[-1]["status"]} (match cost {cost:.2f}, {pieces} piece{"s" if pieces > 1 else ""})')
    seen = {name for _, name, _ in matched}
    for name in shapes:
        if name not in seen:
            rows += [dict(base, object=name, config=cfg, status='not found') for cfg in configs]
            log(f'  {name:12s} not found in the scan')
    return rows


def summarise(df, calipers=None):
    L = ['# Validation test 3: printed phantoms in CT\n',
         'Each scanned object is fitted rigidly to its exact surface, then its canonical mesh (raw marching '
         'cubes) and production mesh (the app\'s smoothing and decimation) are scored against it. The card\'s '
         'resolution limit (0.177 x RMS spacing, range 0.13-0.21) is checked against the canonical error.\n',
         '| Scan | Object | Pieces | Canonical ASSD (mm) | Production ASSD (mm) | HD95 (mm) | Volume (prod., %) | '
         'Card range (mm) | Inside |',
         '|---|---|---|---|---|---|---|---|---|']
    def val(r, key, fmt):
        return fmt.format(r[key]) if r['status'] == 'ok' else r['status']
    for (scan, obj), g in df.groupby(['scan', 'object'], sort=False):
        c, p = g[g.config == 'canonical'].iloc[0], g[g.config == 'production'].iloc[0]
        if c.status != 'ok':
            L.append(f'| {scan} | {obj} | - | {c.status} | {p.status} | - | - | '
                     f'{c.card_resolution_lo_mm:.2f}-{c.card_resolution_hi_mm:.2f} | - |')
            continue
        inside = c.card_resolution_lo_mm <= c.assd_mm <= c.card_resolution_hi_mm
        flag = ' (match uncertain)' if c.match_cost > 1.0 else ''
        L.append(f'| {scan} | {obj}{flag} | {int(c.scanned_pieces)} | {c.assd_mm:.3f} | {val(p, "assd_mm", "{:.3f}")} | '
                 f'{c.hd95_mm:.3f} | {val(p, "volume_err_pct", "{:+.1f}")} | '
                 f'{c.card_resolution_lo_mm:.2f}-{c.card_resolution_hi_mm:.2f} | {"yes" if inside else "no"} |')
    can = df[(df.config == 'canonical') & (df.status == 'ok')]
    inside = ((can.assd_mm >= can.card_resolution_lo_mm) & (can.assd_mm <= can.card_resolution_hi_mm)).mean()
    L.append(f'\nThe card\'s resolution range held for {100 * inside:.0f}% of {len(can)} canonical meshes.\n')
    if calipers is not None and calipers.measured_mm.notna().any():
        cal = calipers.dropna(subset=['measured_mm'])
        err = cal.measured_mm - cal.design_mm
        L.append(f'Printed objects against design (calipers, {len(cal)} measurements): mean {err.mean():+.3f} mm, '
                 f'largest {err.abs().max():.3f} mm. Errors below this are within printing tolerance.\n')
    return '\n'.join(L)


def analyse(paths, out, calipers=None, seed=2026):
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for p in paths:
        rows += analyse_one(p, rng)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'results.csv'), index=False)
    cal = pd.read_csv(calipers) if calipers else None
    text = summarise(df, cal)
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    with open(os.path.join(out, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump({'test': 'physical_phantom', 'scans': [os.path.abspath(p) for p in paths], 'seed': seed,
                   'n_samples': N_SAMPLES, 'seconds': round(time.time() - t0, 1)}, fh, indent=1)
    print(text)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('design')
    d.add_argument('--out', default=os.path.join(ROOT, 'docs', 'validation', 'phantom'))
    s = sub.add_parser('simulate')
    s.add_argument('--out', required=True)
    s.add_argument('--spacing', default='0.7,0.7,1.25')
    s.add_argument('--medium', choices=('air', 'water'), default='air')
    s.add_argument('--seed', type=int, default=2026)
    a = sub.add_parser('analyse')
    a.add_argument('scans', nargs='+', help='DICOM folders or NIfTI files, one per reconstruction')
    a.add_argument('--out', default=None)
    a.add_argument('--calipers', default=None, help='calipers.csv with the measured_mm column filled in')
    a.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    if args.cmd == 'design':
        design(args.out)
    elif args.cmd == 'simulate':
        simulate(args.out, [float(x) for x in args.spacing.split(',')], args.seed, args.medium)
    else:
        scans = [p for s in args.scans for p in (sorted(glob.glob(s)) or [s])]
        out = args.out or os.path.join(ROOT, 'output', 'research', 'physical_phantom', time.strftime('%Y%m%d-%H%M%S'))
        analyse(scans, out, args.calipers, args.seed)


if __name__ == '__main__':
    main()
