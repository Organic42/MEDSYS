"""Tests for the per-mesh accuracy estimate (accuracy.py)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import accuracy  # noqa: E402


def test_resolution_limit_follows_the_study():
    iso = accuracy.resolution_limit((1.0, 1.0, 1.0))
    assert iso['rms_spacing_mm'] == 1.0
    assert iso['estimate_mm'] == pytest.approx(0.177) and iso['range_mm'] == [0.13, 0.21]
    thick = accuracy.resolution_limit((0.84, 0.84, 5.0))        # Tier 1's 0.8 x 0.8 x 5 grid
    assert thick['rms_spacing_mm'] == pytest.approx(np.sqrt((0.84 ** 2 * 2 + 25) / 3), abs=1e-3)
    assert 0.40 < thick['estimate_mm'] < 0.60                  # phantoms measured 0.50-0.63 mm there


def _sphere(radius_mm=12.0, spacing=(1.0, 1.0, 1.0), n=40):
    idx = np.indices((n, n, n)).reshape(3, -1).T * np.asarray(spacing)
    centre = np.asarray((n, n, n)) * np.asarray(spacing) / 2
    return (np.linalg.norm(idx - centre, axis=1) < radius_mm).reshape(n, n, n)


def test_assess_measures_smoothing_on_a_sphere():
    pytest.importorskip('pyvista')
    from reconstruction import ReconParams, extract_isosurface, postprocess_surface
    mask, spacing = _sphere(), (1.0, 1.0, 1.0)
    params = ReconParams(blur_sigma_vox=(0.6,) * 3, taubin_iter=30, decimation=0.5)
    mesh = postprocess_surface(*extract_isosurface(mask, spacing, params), params)
    r = accuracy.assess(mask, spacing, mesh)
    assert r['watertight'] and r['pieces'] == 1
    assert 0 < r['recon_assd_mm'] < 0.5 and r['recon_p95_mm'] >= r['recon_assd_mm']
    assert r['volume_change_pct'] is not None and abs(r['volume_change_pct']) < 5
    # the raw surface of the same mask reproduces itself
    raw = postprocess_surface(*extract_isosurface(mask, spacing, ReconParams()), ReconParams())
    same = accuracy.assess(mask, spacing, raw)
    assert same['recon_assd_mm'] < 1e-6


def test_record_and_load_round_trip(tmp_path):
    pytest.importorskip('pyvista')
    import pyvista as pv
    mesh = pv.Sphere()
    result = {'recon_assd_mm': 0.2, 'recon_p95_mm': 0.6, 'recon_signed_mm': -0.05, 'volume_change_pct': None,
              'watertight': False, 'pieces': 3}
    accuracy.attach(mesh, result, (0.8, 0.8, 5.0))
    accuracy.record(mesh, str(tmp_path), 'spleen')
    accuracy.record(pv.Sphere(), str(tmp_path), 'no_estimate')       # a mesh without one is skipped
    data = json.loads((tmp_path / accuracy.FILE).read_text())
    assert data['structures'] == {'spleen': result}
    assert data['resolution_limit'] == accuracy.resolution_limit((0.8, 0.8, 5.0))


def test_validated_figures_only_for_measured_engine_and_organs(tmp_path):
    (tmp_path / accuracy.FILE).write_text(json.dumps({'structures': {'spleen': {}, 'liver': {}},
                                                      'spacing_mm': [1, 1, 1]}))
    ts = accuracy.for_dataset(str(tmp_path), {'engine': 'TotalSegmentator', 'fast_mode': True})
    assert set(ts['validated']) == {'spleen'} and ts['validated']['spleen']['median_mm'] == 0.854
    full = accuracy.for_dataset(str(tmp_path), {'engine': 'TotalSegmentator', 'fast_mode': False})
    assert full['engine_mode'] == 'full' and full['validated']['spleen']['median_mm'] == 0.699
    classical = accuracy.for_dataset(str(tmp_path), {'modality': 'CT'})
    assert classical['engine'] == 'classical' and classical['validated'] == {}
    assert accuracy.for_dataset(str(tmp_path / 'missing'), {}) is None


def test_engine_figures_follow_the_scan_grid(tmp_path):
    assert accuracy.reference_level((0.7, 0.7, 2.5)) == 'native'
    assert accuracy.reference_level((0.8, 0.8, 5.0)) == 'slice5'
    assert accuracy.reference_level((5.0, 0.8, 0.8)) == 'slice5'          # slice axis first
    assert accuracy.reference_level((1.5, 1.5, 5.0)) == 'iso1.5'
    assert accuracy.reference_level((3.0, 3.0, 3.0)) == 'iso3'
    (tmp_path / accuracy.FILE).write_text(json.dumps({'structures': {'spleen': {}}, 'spacing_mm': [3, 3, 5]}))
    ref = accuracy.for_dataset(str(tmp_path), {'engine': 'TotalSegmentator', 'fast_mode': False})['validated']['spleen']
    assert ref['level'] == 'iso3' and ref['median_mm'] == 0.95
    assert ref['range90_mm'][0] < ref['iqr_mm'][0] < ref['median_mm'] < ref['iqr_mm'][1] < ref['range90_mm'][1]
    for key, r in accuracy.ENGINE_REFERENCE.items():
        assert r['range90_mm'][0] <= r['iqr_mm'][0] <= r['median_mm'] <= r['iqr_mm'][1] <= r['range90_mm'][1], key


def test_build_mesh_attaches_an_estimate_unless_switched_off(monkeypatch):
    pytest.importorskip('pyvista')
    import segmentation_pipeline as sp
    mask = _sphere()
    mesh = sp.build_mesh(mask, (1.0, 1.0, 1.0))
    assert 'medsys_recon_assd_mm' in mesh.field_data
    monkeypatch.setattr(sp.config, 'ACCURACY', False)
    assert 'medsys_recon_assd_mm' not in sp.build_mesh(mask, (1.0, 1.0, 1.0)).field_data
