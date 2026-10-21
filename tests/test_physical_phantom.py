"""Tests for the printed-phantom analysis (research/physical_phantom.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research import physical_phantom as pp  # noqa: E402
from research.phantoms import Capsule, Sphere  # noqa: E402


def test_fragments_close_together_are_one_object():
    mask = np.zeros((60, 30, 30), bool)
    mask[10:20, 10:15, 10:15] = True
    mask[22:30, 10:15, 10:15] = True          # 2 voxels (2 mm) from the first piece
    mask[45:55, 10:15, 10:15] = True          # 15 mm away: a separate object
    found = pp.components(mask, (1.0, 1.0, 1.0), min_mm3=1, gap_mm=6.0)
    assert sorted(n for _, _, n in found) == [1, 2]


def test_branch_is_closed_form_where_it_matters():
    b = pp.Branch()
    tip = b.segs[1][1] - b._centre + np.array([0, 0, 0.0])
    assert abs(b.sdf_local(tip[None])[0] + b.segs[1][2]) < 1e-9     # a branch end lies one radius inside
    assert 900 < b.volume < 1000


def test_simulated_scan_is_recovered(tmp_path, monkeypatch):
    pytest.importorskip('SimpleITK')
    pytest.importorskip('pydicom')
    small = {'sphere_r8': Sphere(8.0), 'capsule_r2': Capsule(2.0, 8.0)}
    monkeypatch.setattr(pp, 'kit', lambda: small)
    pp.simulate(str(tmp_path / 'scan'), [1.0, 1.0, 1.0], seed=3)
    rows = pp.analyse_one(str(tmp_path / 'scan'), np.random.default_rng(0), log=lambda m: None)
    can = {r['object']: r for r in rows if r['config'] == 'canonical'}
    assert set(can) == set(small) and all(r['status'] == 'ok' for r in can.values())
    # about 0.14 x voxel on exact surfaces (Tier 1), plus partial volume and noise
    assert can['sphere_r8']['assd_mm'] < 0.3 and can['capsule_r2']['assd_mm'] < 0.4
    assert abs(can['sphere_r8']['volume_err_pct']) < 5
