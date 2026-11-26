"""Tests for the cardiac measurements (cardiac.py) on synthetic vessels of known size."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cardiac  # noqa: E402

LAS = ('L', 'A', 'S')


def tube(shape, spacing, centre_mm, radius_mm):
    """Mask of every voxel within radius_mm[i] of centreline point centre_mm[i]."""
    from scipy.spatial import cKDTree
    idx = np.indices(shape).reshape(3, -1).T * np.asarray(spacing)
    d, i = cKDTree(centre_mm).query(idx)
    return (d <= np.asarray(radius_mm)[i]).reshape(shape)


def line(p0, p1, n=600):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(p0, float) * (1 - t) + np.asarray(p1, float) * t


def test_straight_tube_gives_its_diameter():
    sp = (0.8, 0.8, 1.0)
    mask = tube((80, 80, 70), sp, line((32, 32, 2), (32, 32, 66)), np.full(600, 15.0))
    a = cardiac.aorta_diameters(mask, sp, LAS)
    cs = a['max']['cross_section']
    assert cs['max_mm'] == pytest.approx(30, abs=1.5)
    assert cs['equiv_mm'] == pytest.approx(30, abs=1.5)
    assert a['max']['max_mm'] == pytest.approx(30, abs=2.0)      # inscribed sphere


def test_slanted_tube_is_cut_at_right_angles():
    # At 45 degrees an axial cut would read about 30 * sqrt(2) = 42 mm.
    sp = (1.0, 1.0, 1.0)
    mask = tube((100, 50, 100), sp, line((15, 25, 15), (85, 25, 85)), np.full(600, 10.0))
    cs = cardiac.aorta_diameters(mask, sp, LAS)['max']['cross_section']
    assert cs['max_mm'] == pytest.approx(20, abs=1.5)
    assert cs['equiv_mm'] == pytest.approx(20, abs=1.5)


def test_arch_is_split_into_ascending_arch_and_descending():
    # Ascending limb in front (higher y = anterior in LAS), 34 mm wide; descending
    # limb behind it, 24 mm wide and running lower; a half-circle arch joins them.
    asc = line((40, 85, 20), (40, 85, 100))
    th = np.linspace(0, np.pi, 600)
    arch = np.c_[np.full(600, 40), 57.5 + 27.5 * np.cos(th), 100 + 27.5 * np.sin(th)]
    desc = line((40, 30, 100), (40, 30, 5))
    centre = np.r_[asc, arch, desc]
    radius = np.r_[np.full(600, 17.0), np.linspace(17, 12, 600), np.full(600, 12.0)]
    mask = tube((80, 120, 150), (1.0, 1.0, 1.0), centre, radius)
    regions = cardiac.aorta_diameters(mask, (1.0, 1.0, 1.0), LAS)['regions']
    assert set(regions) == {'ascending', 'arch', 'descending'}
    assert regions['ascending']['cross_section']['max_mm'] == pytest.approx(34, abs=2.5)
    assert regions['descending']['cross_section']['max_mm'] == pytest.approx(24, abs=2.5)
    assert regions['ascending']['point_mm'][1] > regions['descending']['point_mm'][1]


def test_measure_writes_volumes(tmp_path):
    nib = pytest.importorskip('nibabel')     # not in the CI requirements
    seg = tmp_path / 'segmentations'
    seg.mkdir()
    aff = np.diag([-0.5, 0.5, 2.0, 1.0])               # LAS, 0.5 mm voxels
    lv = np.zeros((40, 40, 20), np.uint8)
    lv[:20, :20, :10] = 1                               # 4000 voxels * 0.5 mm3 = 2.0 ml
    nib.save(nib.Nifti1Image(lv, aff), str(seg / 'heart_ventricle_left.nii.gz'))
    res = cardiac.measure(str(seg), str(tmp_path))
    assert res['chambers'] == [{'name': 'heart_ventricle_left', 'label': 'Left ventricle', 'volume_ml': 2.0}]
    assert res['axcodes'] == ['L', 'A', 'S']
    assert json.loads((tmp_path / 'cardiac.json').read_text())['chambers'][0]['volume_ml'] == 2.0


def test_measure_returns_none_without_cardiac_masks(tmp_path):
    (tmp_path / 'segmentations').mkdir()
    assert cardiac.measure(str(tmp_path / 'segmentations'), str(tmp_path)) is None
    assert not (tmp_path / 'cardiac.json').exists()
