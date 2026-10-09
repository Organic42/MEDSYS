"""
Tests for the statistics added for the paper (research/paper_stats.py).

Synthetic data with known structure: the random-intercept estimator must
recover known variance components, the sampling-relationship fit must recover
a known slope and validate it, and the grid comparison must aggregate per scan.
Needs pandas and scipy; skips when they are missing (as in CI).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_random_intercept_recovers_known_components():
    pd = pytest.importorskip('pandas')
    pytest.importorskip('scipy')
    from research.paper_stats import random_intercept_components
    rng = np.random.default_rng(0)
    clusters, cells = 300, 40
    u = rng.normal(0, 2.0, clusters)                  # between-cluster sd 2   -> variance 4
    c = rng.normal(0, 1.0, cells)                     # fixed cell effects     -> variance ~1
    rows = [(i, j, u[i] + c[j] + rng.normal(0, 0.5)) for i in range(clusters) for j in range(cells)]
    df = pd.DataFrame(rows, columns=['case', 'cell', 'y'])
    out = random_intercept_components(df, 'y', 'case', ['cell'])
    var_cells = c.var(ddof=1)
    expected = 4.0 / (4.0 + var_cells + 0.25)
    assert out['clusters'] == clusters and out['cells'] == cells
    assert out['share_cluster'] == pytest.approx(expected, abs=0.05)
    assert out['icc_conditional'] == pytest.approx(4.0 / 4.25, abs=0.03)


def test_sampling_relationship_recovers_slope_and_validates():
    pd = pytest.importorskip('pandas')
    pytest.importorskip('scipy')
    from research.paper_stats import sampling_relationship
    rng = np.random.default_rng(1)
    rows = []
    for k in range(30):
        for lvl, x in (('iso1.5', 1.0), ('iso3', 2.5)):
            rows.append({'case': f'c{k}', 'resolution': lvl, 'added_mm': x + rng.normal(0, 0.05),
                         'assd_mm': 0.2 * x + rng.normal(0, 0.01)})
    rel = sampling_relationship(pd.DataFrame(rows), rng, n_boot=300)
    assert rel['slope'] == pytest.approx(0.2, abs=0.005)
    assert rel['slope_ci'][0] < 0.2 < rel['slope_ci'][1]
    assert rel['n_scans'] == 30 and rel['n_points'] == 60
    assert rel['loso']['r2'] > 0.9 and rel['loso']['mae'] < 0.02
    assert set(rel['leave_one_level_out']) == {'iso1.5', 'iso3'}


def test_grid_comparison_averages_within_scan_first():
    pd = pytest.importorskip('pandas')
    pytest.importorskip('scipy')
    from research.paper_stats import grid_comparison, grid_comparison_stats
    rows = []
    for k in range(10):
        for res, offset in (('native', 0.0), ('iso3', -0.02)):
            for blur in (0.0, 1.0):
                for dec in (0.0, 0.75):
                    rows.append({'source': 'ts_fast', 'config': 'grid', 'case': f'c{k}', 'resolution': res,
                                 'blur_mm': blur, 'taubin_iter': 0, 'decimation': dec,
                                 'assd_mm': 1.0 + 0.01 * k + offset,
                                 'topology_ok': not (res == 'native' and dec == 0.75), 'n_faces': 100})
    both, per_cfg = grid_comparison(pd.DataFrame(rows), 'ts_fast')
    assert len(per_cfg) == 4 and len(both) == 40
    stats = grid_comparison_stats(both, np.random.default_rng(2), n_boot=200)
    assert stats['d_assd_per_scan']['mean'] == pytest.approx(-0.02)
    assert stats['d_assd_per_scan']['scans'] == 10 and stats['d_assd_per_scan']['scans_improved'] == 10
    closure = stats['closure_dec075']
    assert closure['native'] == 0.0 and closure['iso3'] == 1.0
    assert closure['difference']['mean'] == pytest.approx(1.0) and closure['difference']['scans_improved'] == 10
