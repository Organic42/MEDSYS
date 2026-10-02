"""
Tests for the Tier 1 phantom experiment and the shared reconstruction module.

The first group needs only numpy/scipy/scikit-image (what CI installs). Tests
that need PyVista/VTK, or the full pipeline's imports, skip when those are
missing.
"""
import os
import sys

import numpy as np
import pytest
from skimage import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from reconstruction import ReconParams, blur_mm_to_vox, extract_isosurface  # noqa: E402
from research.phantoms import (Capsule, RoundedBox, Sphere, Torus, default_phantoms,  # noqa: E402
                               identity_placement, random_placement, sample_surface,
                               sdf_world, voxelise)
from research.surface_metrics import (compare_to_phantom, mesh_area, mesh_topology,  # noqa: E402
                                      mesh_volume)


def sdf_mesh(phantom, spacing=0.4, pad=2.0):
    """A smooth, accurate mesh of the unrotated phantom: marching cubes on its
    SDF field, sampled over the shape's own bounding box."""
    he = phantom.half_extents + pad
    axes = [np.arange(-h, h + spacing, spacing) for h in he]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    field = phantom.sdf_local(grid).reshape([len(a) for a in axes])
    verts, faces, _, _ = measure.marching_cubes(field, level=0.0, spacing=(spacing,) * 3)
    return verts - he, faces


# ── phantoms: exact distances and closed-form geometry ──────────────────────

@pytest.mark.parametrize('phantom, point, expected', [
    (Sphere(10), (10, 0, 0), 0.0),
    (Sphere(10), (0, 0, 0), -10.0),
    (Sphere(10), (0, 4, 3), -5.0),
    (Sphere(10), (13, 0, 0), 3.0),
    (Capsule(2, 20), (0, 0, 22), 0.0),
    (Capsule(2, 20), (2, 0, 0), 0.0),
    (Capsule(2, 20), (0, 0, 0), -2.0),
    (Capsule(2, 20), (5, 0, 10), 3.0),
    (Capsule(2, 20), (0, 0, 25), 3.0),
    (Torus(18, 6), (18, 0, 0), -6.0),
    (Torus(18, 6), (24, 0, 0), 0.0),
    (Torus(18, 6), (12, 0, 0), 0.0),
    (Torus(18, 6), (18, 0, 6), 0.0),
    (Torus(18, 6), (0, 0, 0), 12.0),
    (RoundedBox((14, 10, 6), 2), (16, 0, 0), 0.0),
    (RoundedBox((14, 10, 6), 2), (0, 0, 0), -8.0),
    (RoundedBox((14, 10, 6), 2), (0, 0, 10), 2.0),
    (RoundedBox((14, 10, 6), 2), (17, 10, 6), 1.0),
])
def test_sdf_values_are_exact(phantom, point, expected):
    assert phantom.sdf_local(np.array([point], dtype=float))[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('phantom', default_phantoms(), ids=lambda p: p.name)
def test_closed_form_volume_and_area_match_a_fine_mesh(phantom):
    # A piecewise-linear mesh underestimates a curved surface by about
    # (spacing / radius)^2, so the 2 mm capsule needs a finer grid.
    verts, faces = sdf_mesh(phantom, spacing=0.2 if phantom.name == 'capsule' else 0.4)
    assert mesh_volume(verts, faces) == pytest.approx(phantom.volume, rel=0.01)
    assert mesh_area(verts, faces) == pytest.approx(phantom.area, rel=0.015)


@pytest.mark.parametrize('phantom', default_phantoms(), ids=lambda p: p.name)
def test_voxelised_volume_matches_closed_form(phantom):
    placement = random_placement(np.random.default_rng(4), max_offset_mm=5.0)
    spacing = (0.4, 0.4, 0.4)
    mask, _ = voxelise(phantom, placement, spacing)
    voxel_volume = mask.sum() * np.prod(spacing)
    tol = 0.03 if phantom.name == 'capsule' else 0.01   # thin tube: coarser relative sampling
    assert voxel_volume == pytest.approx(phantom.volume, rel=tol)


def test_placement_round_trip_and_proper_rotation():
    placement = random_placement(np.random.default_rng(1), max_offset_mm=5.0)
    pts = np.random.default_rng(2).normal(size=(100, 3)) * 10
    assert np.allclose(placement.to_world(placement.to_local(pts)), pts)
    assert np.linalg.det(placement.rotation.as_matrix()) == pytest.approx(1.0)


@pytest.mark.parametrize('phantom', default_phantoms(), ids=lambda p: p.name)
def test_surface_samples_lie_on_the_exact_surface(phantom):
    placement = random_placement(np.random.default_rng(5), max_offset_mm=5.0)
    pts = sample_surface(phantom, placement, 5000, np.random.default_rng(6))
    assert pts.shape == (5000, 3)
    assert np.abs(sdf_world(phantom, pts, placement)).max() < 1e-9


def test_surface_samples_are_uniform_by_area():
    # On a torus the outer half (cos v > 0) holds more area than the inner half:
    # fraction = (pi R + 2 r) / (2 pi R). Uniform-by-area sampling must match it.
    torus = Torus(18, 6)
    pts = sample_surface(torus, identity_placement(), 20000, np.random.default_rng(7))
    outer = np.mean(np.hypot(pts[:, 0], pts[:, 1]) > torus.R)
    expected = (np.pi * torus.R + 2 * torus.r) / (2 * np.pi * torus.R)
    assert outer == pytest.approx(expected, abs=0.015)
    sphere_pts = sample_surface(Sphere(15), identity_placement(), 20000, np.random.default_rng(8))
    assert np.linalg.norm(sphere_pts.mean(axis=0)) < 0.3


# ── mesh topology and metrics ───────────────────────────────────────────────

def test_topology_of_sphere_torus_and_two_spheres():
    v, f = sdf_mesh(Sphere(8), spacing=0.5)
    topo = mesh_topology(f)
    assert (topo['components'], topo['closed'], topo['genus']) == (1, True, 0)
    v, f = sdf_mesh(Torus(12, 4), spacing=0.5)
    assert mesh_topology(f)['genus'] == 1
    v1, f1 = sdf_mesh(Sphere(5), spacing=0.5)
    two = np.concatenate([f1, f1 + len(v1)])
    topo = mesh_topology(two)
    assert (topo['components'], topo['genus']) == (2, 0)


def test_topology_flags_holes():
    v, f = sdf_mesh(Sphere(8), spacing=0.5)
    topo = mesh_topology(f[10:])          # remove faces -> boundary edges
    assert topo['boundary_edges'] > 0
    assert not topo['closed']
    assert np.isnan(topo['genus'])


def test_metrics_recover_a_known_1mm_offset():
    truth_shape = Sphere(10)
    placement = identity_placement()
    verts, faces = sdf_mesh(Sphere(11), spacing=0.25)
    truth = sample_surface(truth_shape, placement, 8000, np.random.default_rng(9))
    m = compare_to_phantom(verts, faces, truth_shape, placement, truth,
                           np.random.default_rng(10), n_mesh_samples=8000, tol_mm=0.5)
    assert m['mean_signed_mm'] == pytest.approx(1.0, abs=0.02)
    assert m['assd_mm'] == pytest.approx(1.0, abs=0.04)
    assert m['hd95_mm'] == pytest.approx(1.0, abs=0.06)
    assert m['nsd'] < 0.01                # nothing within 0.5 mm
    assert m['volume_err_pct'] == pytest.approx((1.1 ** 3 - 1) * 100, abs=1.0)
    assert m['topology_ok']


def test_metrics_near_zero_for_an_accurate_mesh():
    shape = Sphere(10)
    placement = identity_placement()
    verts, faces = sdf_mesh(shape, spacing=0.25)
    truth = sample_surface(shape, placement, 8000, np.random.default_rng(11))
    m = compare_to_phantom(verts, faces, shape, placement, truth,
                           np.random.default_rng(12), n_mesh_samples=8000, tol_mm=1.0)
    assert m['assd_mm'] < 0.03
    assert m['hd95_mm'] < 0.06
    assert m['nsd'] == 1.0
    assert abs(m['mean_signed_mm']) < 0.02


# ── reconstruction stages that need no VTK ──────────────────────────────────

def test_blur_is_converted_from_millimetres_per_axis():
    assert blur_mm_to_vox(1.0, (0.5, 1.0, 2.0)) == (2.0, 1.0, 0.5)


def test_isosurface_rejects_unknown_method():
    with pytest.raises(ValueError):
        extract_isosurface(np.ones((5, 5, 5), bool), (1, 1, 1), ReconParams(isosurface='nope'))


def test_isosurface_applies_world_origin():
    mask = np.zeros((20, 20, 20), bool)
    mask[5:15, 5:15, 5:15] = True
    v0, _ = extract_isosurface(mask, (1, 1, 1), ReconParams())
    v1, _ = extract_isosurface(mask, (1, 1, 1), ReconParams(), origin=(10.0, -3.0, 2.5))
    assert np.allclose(v1 - v0, (10.0, -3.0, 2.5))


@pytest.mark.parametrize('method', ['lewiner', 'lorensen'])
def test_reference_procedure_error_shrinks_with_finer_voxels(method):
    shape = Sphere(15)
    placement = random_placement(np.random.default_rng(13), max_offset_mm=5.0)
    truth = sample_surface(shape, placement, 6000, np.random.default_rng(14))
    assd = {}
    for s in (0.5, 2.0):
        mask, origin = voxelise(shape, placement, (s, s, s))
        v, f = extract_isosurface(mask, (s, s, s), ReconParams(isosurface=method), origin)
        assd[s] = compare_to_phantom(v, f, shape, placement, truth, np.random.default_rng(15),
                                     n_mesh_samples=6000)['assd_mm']
    assert assd[0.5] < assd[2.0]
    assert assd[0.5] < 0.5 * 0.5           # well under half a voxel at 0.5 mm


# ── tests that need PyVista / the full pipeline ─────────────────────────────

def _build_mesh_original(mask, spacing, sigma=0.6, taubin_iter=30, reduction=0.5,
                         min_voxels=200):
    """Frozen copy of build_mesh() as it was before the reconstruction refactor."""
    import pyvista as pv
    from scipy.ndimage import gaussian_filter
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


@pytest.mark.parametrize('kwargs', [
    {},                                                    # defaults
    {'sigma': (1.0, 0.6, 0.6), 'reduction': 0.0},          # anisotropic blur, no decimation
    {'sigma': 1.0, 'taubin_iter': 10, 'reduction': 0.6},   # TotalSegmentator-path style values
])
def test_build_mesh_output_unchanged_by_refactor(kwargs):
    pytest.importorskip('pyvista')
    pytest.importorskip('pydicom')
    pytest.importorskip('matplotlib')
    import segmentation_pipeline as sp
    placement = random_placement(np.random.default_rng(16), max_offset_mm=5.0)
    mask, _ = voxelise(Torus(18, 6), placement, (0.8, 0.8, 2.0))
    new = sp.build_mesh(mask, (0.8, 0.8, 2.0), **kwargs)
    old = _build_mesh_original(mask, (0.8, 0.8, 2.0), **kwargs)
    assert np.array_equal(np.asarray(new.points), np.asarray(old.points))
    assert np.array_equal(np.asarray(new.faces), np.asarray(old.faces))


def test_build_mesh_still_rejects_tiny_masks():
    pytest.importorskip('pyvista')
    pytest.importorskip('pydicom')
    pytest.importorskip('matplotlib')
    import segmentation_pipeline as sp
    mask = np.zeros((10, 10, 10), bool)
    mask[4:6, 4:6, 4:6] = True
    assert sp.build_mesh(mask, (1, 1, 1)) is None


def test_exact_point_to_mesh_distance():
    pytest.importorskip('pyvista')
    from research.surface_metrics import point_to_mesh_distance
    verts, faces = sdf_mesh(Sphere(10), spacing=0.25)
    dirs = np.random.default_rng(17).normal(size=(500, 3))
    pts = 13 * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    d, exact = point_to_mesh_distance(pts, verts, faces)
    assert exact
    assert np.allclose(d, 3.0, atol=0.01)


def test_postprocess_stages_are_independent():
    pytest.importorskip('pyvista')
    from reconstruction import mesh_arrays, postprocess_surface
    mask, origin = voxelise(Sphere(12), identity_placement(), (1, 1, 1))
    v, f = extract_isosurface(mask, (1, 1, 1), ReconParams(), origin)
    v0, f0 = mesh_arrays(postprocess_surface(v, f, ReconParams()))
    assert np.allclose(v0, v) and np.array_equal(f0, f)       # no smoothing, no decimation
    _, fd = mesh_arrays(postprocess_surface(v, f, ReconParams(decimation=0.5)))
    assert 0.4 < len(fd) / len(f) < 0.6


def test_summary_variance_shares_add_up():
    pd = pytest.importorskip('pandas')
    from itertools import product as iproduct
    from research.phantom_experiment import summarise
    rows = []
    rng = np.random.default_rng(18)
    levels = [('0.5', '2'), ('sphere', 'torus'), (0.0, 1.0), ('lewiner', 'lorensen'), (0, 30), (0.0, 0.5), (0, 1)]
    for sp, shape, blur, iso, tau, dec, place in iproduct(*levels):
        assd = (0.1 if sp == '0.5' else 0.3) * (1.5 if shape == 'torus' else 1.0) + 0.01 * rng.random()
        rows.append({'status': 'ok', 'config': 'grid', 'is_canonical': blur == 0 and tau == 0 and dec == 0
                     and iso == 'lewiner', 'spacing': sp, 'shape': shape, 'blur_mm': blur,
                     'isosurface': iso, 'taubin_iter': tau, 'decimation': dec, 'placement': place,
                     'assd_mm': assd, 'hd95_mm': 2 * assd, 'mean_signed_mm': -0.01,
                     'topology_ok': True, 'closed': True, 'components': 1,
                     't_iso_s': 0.0, 't_post_s': 0.0, 't_metric_s': 0.0})
    run = {'preset': 'test', 'commit': 'x', 'meshes': len(rows), 'units': 1, 'workers': 1,
           'wall_time_s': 1.0, 'seed': 0, 'samples_per_side': 10, 'nsd_tolerance_mm': 1.0,
           'spacings_mm': [[0.5] * 3, [2.0] * 3], 'placements': 2}
    text = summarise(rows, run)
    section = text.split('## Share of ASSD variance')[1].split('\n## ')[0]
    shares = {}
    for line in section.splitlines():
        parts = [p.strip() for p in line.strip('|').split('|')]
        if len(parts) == 3 and parts[1].endswith('%') and parts[0] != 'source':
            shares[parts[0]] = float(parts[1].rstrip('%'))
    assert set(shares) >= {'spacing', 'shape', 'replication', 'interactions'}
    assert sum(shares.values()) == pytest.approx(100.0, abs=0.5)
    assert shares['spacing'] > shares['blur_mm']      # the constructed signal dominates
    assert pd is not None


def test_smoke_experiment_runs_end_to_end(tmp_path):
    pytest.importorskip('pyvista')
    pytest.importorskip('pandas')
    from research.phantom_experiment import run_experiment
    outdir, rows = run_experiment('smoke', workers=1, outdir=str(tmp_path))
    assert {r['config'] for r in rows} == {'grid', 'production'}
    assert all(r['status'] == 'ok' for r in rows)
    canonical = [r for r in rows if r['is_canonical']]
    assert len(canonical) == 1 and 0.05 < canonical[0]['assd_mm'] < 0.3
    for name in ('results.csv', 'run.json', 'summary.md'):
        assert os.path.exists(os.path.join(outdir, name))
