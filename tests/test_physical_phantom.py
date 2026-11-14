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


def test_scanner_export_tree_is_split_into_reconstructions(tmp_path):
    # A USB export: every reconstruction in one tree, one series split over two folders
    # with file names in no particular order, and a scout that must be skipped.
    pytest.importorskip('SimpleITK')
    pydicom = pytest.importorskip('pydicom')
    rng = np.random.default_rng(1)
    thin = rng.normal(0, 50, (16, 12, 30)).astype(np.float32)
    thick = rng.normal(0, 50, (16, 12, 24)).astype(np.float32)
    root = tmp_path / 'usb' / 'DICOM'
    pp.write_dicom(thin, (0.5, 0.5, 1.25), str(root / 'A'))
    pp.write_dicom(thick, (0.8, 0.8, 5.0), str(root / 'B'))
    pp.write_dicom(thin[:, :, :2], (0.5, 0.5, 1.25), str(root / "scout"))
    for i, f in enumerate(sorted((root / 'A').iterdir())):            # split series A, scramble names
        dest = root / ('A2' if i % 2 else 'A')
        dest.mkdir(exist_ok=True)
        f.rename(dest / f'IM{(i * 7919) % 1000:04d}')
    for k, (sub, number) in enumerate((('A', 3), ('A2', 3), ('B', 5))):
        for f in (root / sub).iterdir():
            ds = pydicom.dcmread(f)
            ds.SeriesNumber, ds.ConvolutionKernel = number, 'B30f'
            ds.save_as(f)
    series = pp.find_series(str(tmp_path / 'usb'))
    assert [len(fs) for _, fs in series] == [30, 24]                   # scout skipped, A reassembled
    assert series[0][0].startswith('S3 ') and '1.25 mm' in series[0][0] and 'B30f' in series[0][0]
    vol, spacing = pp.load_volume(series[0][1])
    assert np.allclose(spacing, (0.5, 0.5, 1.25)) and np.allclose(vol, np.round(thin))   # slices back in order
    assert [label for label, _ in pp.scans_from([str(tmp_path / 'usb')])] == [label for label, _ in series]
