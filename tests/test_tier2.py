"""
Tests for the Tier 2 (real anatomy) experiment and mesh-to-mesh metrics.

Dataset rules, structure mapping and configurations need only numpy. Tests
that build meshes need PyVista, and the end-to-end unit also needs nibabel;
those skip when the packages are missing (as in CI).
"""
import json
import os
import sys

import numpy as np
import pytest
from skimage import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from research import tier2  # noqa: E402
from research.phantoms import Sphere  # noqa: E402


def sphere_mesh(radius, spacing=0.25, center=(0.0, 0.0, 0.0)):
    """Accurate sphere mesh from marching cubes on its distance field."""
    half = radius + 2
    axis = np.arange(-half, half + spacing, spacing)
    g = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    field = Sphere(radius).sdf_local(g).reshape((len(axis),) * 3)
    v, f, _, _ = measure.marching_cubes(field, level=0.0, spacing=(spacing,) * 3)
    return v - half + np.asarray(center), f


# ── dataset rules ───────────────────────────────────────────────────────

def test_ctorg_lungs_and_bone_only_from_manual_cases():
    assert tier2.structure_allowed('ctorg', 'lungs', 0)
    assert tier2.structure_allowed('ctorg', 'bone', 20)
    assert not tier2.structure_allowed('ctorg', 'lungs', 21)    # morphological label
    assert not tier2.structure_allowed('ctorg', 'bone', 139)
    assert tier2.structure_allowed('ctorg', 'liver', 139)       # manual everywhere
    assert tier2.structure_allowed('msd_spleen', 'spleen', 63)


def test_list_cases_msd(tmp_path):
    (tmp_path / 'dataset.json').write_text(json.dumps({'training': [
        {'image': './imagesTr/spleen_10.nii.gz', 'label': './labelsTr/spleen_10.nii.gz'},
        {'image': './imagesTr/spleen_2.nii.gz', 'label': './labelsTr/spleen_2.nii.gz'},
    ]}))
    cases = tier2.list_cases('msd_spleen', str(tmp_path))
    assert [c[0] for c in cases] == ['spleen_2', 'spleen_10']          # natural order
    assert [c[1] for c in cases] == [2, 10]
    assert cases[0][3].endswith(os.path.join('labelsTr', 'spleen_2.nii.gz'))


def test_list_cases_ctorg_pairs_volumes_with_labels(tmp_path):
    (tmp_path / 'OrganSegmentations').mkdir()
    for n in (3, 12):
        (tmp_path / 'OrganSegmentations' / f'volume-{n}.nii.gz').write_bytes(b'')
        (tmp_path / 'OrganSegmentations' / f'labels-{n}.nii.gz').write_bytes(b'')
    (tmp_path / 'OrganSegmentations' / 'volume-99.nii.gz').write_bytes(b'')   # no label: skipped
    cases = tier2.list_cases('ctorg', str(tmp_path))
    assert [(c[0], c[1]) for c in cases] == [('ctorg_003', 3), ('ctorg_012', 12)]


# ── engines and structure mapping ───────────────────────────────────────

def test_totalsegmentator_structure_mapping():
    lungs, bone, spleen = (tier2.ts_files_for(s) for s in ('lungs', 'bone', 'spleen'))
    assert all(lungs(n) for n in tier2.LUNG_LOBES) and not lungs('trachea')
    assert all(bone(n) for n in ('rib_left_3', 'vertebrae_T5', 'sternum', 'hip_left', 'skull'))
    assert not bone('aorta') and not bone('spinal_cord')
    assert spleen('spleen') and not spleen('spleen_left')


def test_engine_support():
    assert tier2.engine_supports('ts_fast', 'spleen')
    assert tier2.engine_supports('classical', 'lungs')
    assert not tier2.engine_supports('classical', 'spleen')


# ── configurations ──────────────────────────────────────────────────────

def test_grid_sizes_and_canonical():
    full = tier2.configs_for('full', 'gt', 'spleen')
    assert len(full) == 3 * 4 * 4 + 1                  # isosurface fixed; plus production
    assert sum(c['config'] == 'production' for c in full) == 1
    assert len(tier2.configs_for('pilot', 'gt', 'spleen')) == 8 + 1
    canon = [c for c in tier2.configs_for('canonical', 'gt', 'spleen') if c['config'] == 'grid']
    assert canon == [dict(tier2.CANONICAL, config='grid', blur_vox=None)]


@pytest.mark.parametrize('source, structure, expected', [
    ('ts_fast', 'spleen', (0.6, 30, 0.6)),
    ('classical', 'lungs', (0.7, 30, 0.5)),
    ('classical', 'bone', (0.5, 30, 0.4)),
    ('gt', 'liver', (0.6, 30, 0.5)),
])
def test_production_settings_match_the_app(source, structure, expected):
    assert tier2.production_for(source, structure) == expected
    prod = [c for c in tier2.configs_for('pilot', source, structure) if c['config'] == 'production'][0]
    assert (prod['blur_vox'], prod['taubin_iter'], prod['decimation']) == expected


def test_crop_keeps_both_masks_and_reports_origin():
    a = np.zeros((40, 40, 40), bool)
    b = np.zeros_like(a)
    a[10:15, 10:15, 10:15] = True
    b[20:25, 12:14, 18:22] = True
    (ca, cb), start = tier2.crop_to([a, b], margin=2)
    assert tuple(start) == (8, 8, 8)
    assert ca.sum() == a.sum() and cb.sum() == b.sum()
    # joint extent: axis 0 voxels 10-24, axis 1 10-14, axis 2 10-21; plus 2 each side
    assert ca.shape == (24 + 2 + 1 - 8, 14 + 2 + 1 - 8, 21 + 2 + 1 - 8)


# ── acquisition resolution ──────────────────────────────────────────────

def test_grid_spacing_only_coarsens():
    assert tier2.grid_spacing((0.8, 0.8, 5.0), 'native') == (0.8, 0.8, 5.0)
    assert tier2.grid_spacing((0.8, 0.8, 5.0), 'iso3') == (3.0, 3.0, 5.0)       # slice already coarser
    assert tier2.grid_spacing((0.8, 0.8, 1.5), 'iso1.5') == (1.5, 1.5, 1.5)
    assert tier2.grid_spacing((0.8, 0.8, 1.5), 'slice5') == (0.8, 0.8, 5.0)
    assert tier2.grid_spacing((1.0, 1.0, 1.0), 'slice5') == (1.0, 1.0, 5.0)     # isotropic: axis 2
    assert tier2.grid_spacing((5.0, 0.7, 0.7), 'slice5') == (5.0, 0.7, 0.7)     # slice axis found by size
    assert tier2.effective_resolution((0.7, 0.7, 8.0), 'slice5') == 'native'
    assert tier2.effective_resolution((0.7, 0.7, 2.0), 'slice5') == 'slice5'


def test_grid_shape_stays_inside_the_native_extent():
    assert tier2.grid_shape((100, 100, 20), (1, 1, 5), (3, 3, 5)) == (34, 34, 20)
    assert tier2.grid_shape((10, 10, 10), (0.8, 0.8, 0.8), (1.5, 1.5, 1.5)) == (5, 5, 5)


def _sphere(shape, spacing, centre_mm, radius):
    idx = np.indices(shape).reshape(3, -1).T * np.asarray(spacing)
    return (np.linalg.norm(idx - np.asarray(centre_mm), axis=1) < radius).reshape(shape)


def test_resample_keeps_the_physical_frame_and_volume():
    spacing, grid = (0.8, 0.8, 2.0), (3.0, 3.0, 3.0)
    centre = np.array([31.0, 27.0, 33.0])
    native = _sphere((80, 80, 34), spacing, centre, 14)
    coarse = tier2.resample_mask(native, spacing, grid)
    assert coarse.shape == tier2.grid_shape(native.shape, spacing, grid)
    centroid = np.argwhere(coarse).mean(axis=0) * np.asarray(grid)
    assert np.abs(centroid - centre).max() < 0.5                              # same world position
    vol = coarse.sum() * np.prod(grid)
    assert vol == pytest.approx(native.sum() * np.prod(spacing), rel=0.05)
    assert np.array_equal(tier2.resample_mask(native, spacing, spacing), native)


def test_to_native_round_trip():
    spacing, grid = (0.8, 0.8, 2.0), (3.0, 3.0, 5.0)
    native = _sphere((80, 80, 34), spacing, (31, 27, 33), 14)
    back = tier2.to_native(tier2.resample_mask(native, spacing, grid), spacing, grid, native.shape)
    assert back.shape == native.shape
    dice = 2 * (back & native).sum() / (back.sum() + native.sum())
    assert 0.85 < dice < 1.0                                                  # sampling loses a little
    assert np.array_equal(tier2.to_native(native, spacing, spacing, native.shape), native)


def test_resampled_image_affine_maps_coarse_voxels_onto_native_world_points():
    nib = pytest.importorskip('nibabel')
    affine = np.array([[-0.8, 0, 0, 120.0], [0, -0.8, 0, 90.0], [0, 0, 2.0, -300.0], [0, 0, 0, 1]])
    img = nib.Nifti1Image(np.random.default_rng(0).normal(size=(40, 40, 12)).astype(np.float32), affine)
    coarse = tier2.resampled_image(img, (3.0, 3.0, 3.0))
    assert coarse.shape == (11, 11, 8)
    assert np.allclose(coarse.header.get_zooms()[:3], (3.0, 3.0, 3.0))
    j = np.array([4, 7, 5, 1.0])
    native_index = np.r_[j[:3] * np.array([3 / 0.8, 3 / 0.8, 3 / 2.0]), 1]
    assert np.allclose(coarse.affine @ j, affine @ native_index)


def test_segment_case_caches_coarse_engine_masks(tmp_path, monkeypatch):
    nib = pytest.importorskip('nibabel')
    monkeypatch.setenv('VRSEG_OUTPUT', str(tmp_path / 'out'))
    spacing = (0.8, 0.8, 2.0)
    hu = np.where(_sphere((60, 60, 30), spacing, (24, 24, 30), 14), 700.0, -50.0).astype(np.float32)
    img_path = str(tmp_path / 'ct.nii.gz')
    nib.save(nib.Nifti1Image(hu, np.diag(list(spacing) + [1.0])), img_path)
    case = ('case_1', 1, img_path, None)
    tier2.segment_case('msd_spleen', case, ['bone'], ['classical'], 'cpu',
                       ('native', 'iso3', 'slice5'), log=lambda *_: None)
    for res, grid in (('native', spacing), ('iso3', (3.0, 3.0, 3.0)), ('slice5', (0.8, 0.8, 5.0))):
        z = np.load(tier2.cached_mask_path('msd_spleen', 'case_1', 'classical', 'bone', res))
        assert z['mask'].shape == tier2.grid_shape(hu.shape, spacing, grid)
        assert z['mask'].sum() > 0
    assert sorted(os.listdir(tier2.cache_dir('msd_spleen', 'case_1'))) == [
        'classical_bone.npz', 'classical_bone@iso3.npz', 'classical_bone@slice5.npz']   # resampled CTs removed


def test_presets_cover_resolutions():
    assert tier2.PRESETS['full']['resolutions'] == tuple(tier2.RESOLUTIONS)
    assert 'native' in tier2.PRESETS['smoke']['resolutions']


# ── mesh-to-mesh metrics (PyVista) ──────────────────────────────────────

def test_compare_meshes_recovers_a_1mm_expansion():
    pytest.importorskip('pyvista')
    from research.surface_metrics import compare_meshes
    tv, tf = sphere_mesh(11)
    rv, rf = sphere_mesh(10)
    m = compare_meshes(tv, tf, rv, rf, np.random.default_rng(1), n_samples=6000, tol_mm=0.5)
    assert m['mean_signed_mm'] == pytest.approx(1.0, abs=0.03)
    assert m['assd_mm'] == pytest.approx(1.0, abs=0.03)
    assert m['volume_err_pct'] == pytest.approx((1.1 ** 3 - 1) * 100, abs=1.0)
    assert m['nsd'] < 0.01 and m['topology_ok']


def test_compare_meshes_sign_survives_flipped_reference_normals():
    pytest.importorskip('pyvista')
    from research.surface_metrics import compare_meshes
    tv, tf = sphere_mesh(9)                       # test lies inside: shrinkage
    rv, rf = sphere_mesh(10)
    a = compare_meshes(tv, tf, rv, rf, np.random.default_rng(2), n_samples=4000)
    b = compare_meshes(tv, tf, rv, rf[:, ::-1].copy(), np.random.default_rng(2), n_samples=4000)
    assert a['mean_signed_mm'] == pytest.approx(-1.0, abs=0.03)
    assert b['mean_signed_mm'] == pytest.approx(a['mean_signed_mm'], abs=1e-6)


def test_compare_meshes_identical_is_zero():
    pytest.importorskip('pyvista')
    from research.surface_metrics import compare_meshes
    v, f = sphere_mesh(8, spacing=0.4)
    m = compare_meshes(v, f, v, f, np.random.default_rng(3), n_samples=3000)
    assert m['assd_mm'] < 1e-9 and m['hd95_mm'] < 1e-9 and m['nsd'] == 1.0


# ── one end-to-end unit on a synthetic labelled volume ──────────────────

def test_run_unit_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip('pyvista')
    nib = pytest.importorskip('nibabel')
    monkeypatch.setenv('VRSEG_OUTPUT', str(tmp_path / 'out'))
    spacing = (0.8, 0.8, 2.5)
    shape = (64, 64, 32)
    idx = np.indices(shape).reshape(3, -1).T * np.asarray(spacing)
    centre = np.asarray(shape) * np.asarray(spacing) / 2
    gt = (np.linalg.norm(idx - centre, axis=1) < 15).reshape(shape)
    engine = (np.linalg.norm(idx - centre - [1.6, 0, 0], axis=1) < 14).reshape(shape)
    label = nib.Nifti1Image(gt.astype(np.int16), np.diag(list(spacing) + [1.0]))
    label_path = str(tmp_path / 'labels.nii.gz')
    nib.save(label, label_path)
    os.makedirs(tier2.cache_dir('msd_spleen', 'case_1'))
    np.savez_compressed(tier2.cached_mask_path('msd_spleen', 'case_1', 'ts_fast', 'spleen'), mask=engine)

    base = ('msd_spleen', 'case_1', label_path, 'spleen', (1,))
    gt_rows = tier2.run_unit(base + ('gt', 'canonical', 0, 3000, 1.0, 0, 'native'))
    assert not [r for r in gt_rows if r['level'] == 'mask']          # expert vs itself: no mask row
    canon = [r for r in gt_rows if r.get('is_canonical')][0]
    assert canon['status'] == 'ok' and canon['assd_mm'] < 1e-9      # reference reproduces itself

    eng_rows = tier2.run_unit(base + ('ts_fast', 'canonical', 0, 3000, 1.0, 0, 'native'))
    mask_row = [r for r in eng_rows if r['level'] == 'mask'][0]
    assert 0.5 < mask_row['dice'] < 1.0
    meshes = [r for r in eng_rows if r['level'] == 'mesh']
    assert {r['config'] for r in meshes} == {'grid', 'production'}
    assert all(r['status'] == 'ok' for r in meshes)
    canon = [r for r in meshes if r['is_canonical']][0]
    assert canon['mean_signed_mm'] < 0                               # smaller engine sphere: shrinkage

    # 3 mm: the expert mask resampled loses a little overlap and gains surface error
    coarse = tier2.run_unit(base + ('gt', 'canonical', 0, 3000, 1.0, 0, 'iso3'))
    mask_row = [r for r in coarse if r['level'] == 'mask'][0]
    assert mask_row['resampled'] and mask_row['grid_spacing'] == '3x3x3'
    assert 0.85 < mask_row['dice'] < 1.0
    canon = [r for r in coarse if r.get('is_canonical')][0]
    assert canon['status'] == 'ok' and 0.05 < canon['assd_mm'] < 1.5
    assert abs(canon['mean_signed_mm']) < 1.0                        # frames line up: no gross offset

    # an engine mask cached on the coarse grid is read and scored on that grid
    grid = tier2.grid_spacing(spacing, 'iso3')
    np.savez_compressed(tier2.cached_mask_path('msd_spleen', 'case_1', 'ts_fast', 'spleen', 'iso3'),
                        mask=tier2.resample_mask(engine, spacing, grid))
    eng3 = tier2.run_unit(base + ('ts_fast', 'canonical', 0, 3000, 1.0, 0, 'iso3'))
    assert all(r['status'] == 'ok' for r in eng3)
    assert [r for r in eng3 if r.get('is_canonical')][0]['mean_signed_mm'] < 0
