"""Parallel meshing (segmentation_pipeline.mesh_structures) must give exactly
what meshing one by one gives: same meshes, report entries and accuracy.json."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pv = pytest.importorskip('pyvista')
nib = pytest.importorskip('nibabel')
import accuracy  # noqa: E402
import segmentation_pipeline as sp  # noqa: E402


def _seg_dir(path):
    """Three TotalSegmentator-style mask files: a sphere, a box and one too small to mesh."""
    z, y, x = np.mgrid[:40, :48, :48]
    masks = {'spleen': (z - 20) ** 2 + (y - 24) ** 2 + (x - 24) ** 2 < 12 ** 2,
             'liver': (abs(z - 20) < 8) & (abs(y - 20) < 12) & (abs(x - 28) < 9),
             'tiny': (z == 5) & (y < 4) & (x < 4)}
    affine = np.diag([0.8, 0.8, 2.0, 1.0])
    path.mkdir()
    for name, m in masks.items():
        nib.save(nib.Nifti1Image(m.astype(np.uint8), affine), str(path / f'{name}.nii.gz'))
    return sorted(str(p) for p in path.glob('*.nii.gz'))


def _mesh(seg_files, outdir, workers, monkeypatch):
    monkeypatch.setattr(sp.config, 'MESH_WORKERS', workers)
    outdir.mkdir()
    return sp.mesh_structures(seg_files, str(outdir))


def test_parallel_meshing_matches_serial(tmp_path, monkeypatch, capsys):
    files = _seg_dir(tmp_path / 'seg')
    serial = _mesh(files, tmp_path / 'serial', 1, monkeypatch)
    parallel = _mesh(files, tmp_path / 'parallel', 2, monkeypatch)
    assert 'meshed in 2 processes' in capsys.readouterr().out
    assert [r['name'] for r in serial] == ['liver', 'spleen']          # sorted order; the tiny mask is skipped
    assert parallel == serial
    for name in ('liver', 'spleen'):
        a, b = (pv.read(str(tmp_path / d / f'{name}.vtk')) for d in ('serial', 'parallel'))
        assert np.array_equal(a.points, b.points) and np.array_equal(a.faces, b.faces)
        assert os.path.exists(tmp_path / 'parallel' / f'{name}.obj')
    acc = [json.loads((tmp_path / d / accuracy.FILE).read_text()) for d in ('serial', 'parallel')]
    assert acc[0] == acc[1] and set(acc[0]['structures']) == {'liver', 'spleen'}


def test_pool_failure_falls_back_to_meshing_one_by_one(tmp_path, monkeypatch, capsys):
    import concurrent.futures

    class Broken:
        def __init__(self, *a, **k):
            raise OSError('no processes here')

    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', Broken)
    out = _mesh(_seg_dir(tmp_path / 'seg'), tmp_path / 'out', 4, monkeypatch)
    assert 'meshing one by one' in capsys.readouterr().out
    assert [r['name'] for r in out] == ['liver', 'spleen']


def test_worker_count(monkeypatch):
    monkeypatch.setattr(sp.config, 'MESH_WORKERS', 0)
    assert 1 <= sp.mesh_workers(79) <= 12
    assert sp.mesh_workers(1) == 1
    monkeypatch.setattr(sp.config, 'MESH_WORKERS', 3)
    assert sp.mesh_workers(79) == 3
