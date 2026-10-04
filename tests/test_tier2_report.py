"""
Tests for the Tier 2 report builder on a small synthetic results table.

Needs pandas and scipy for the tables and matplotlib for the figures; the
test skips when they are missing (as in CI).
"""
import json
import os
import sys
from itertools import product

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def synthetic_run(path):
    """results.csv and run.json shaped like a real Tier 2 run."""
    pd = pytest.importorskip('pandas')
    rng = np.random.default_rng(0)
    rows = []
    grids = {'native': '0.8x0.8x5', 'iso3': '3x3x5'}
    for case, (res, grid), src in product(('c1', 'c2', 'c3'), grids.items(), ('gt', 'ts_fast')):
        base = dict(dataset='msd_spleen', case=case, structure='spleen', source=src, spacing='0.8x0.8x5',
                    slice_mm=5.0, resolution=res, resampled=res != 'native', grid_spacing=grid,
                    grid_max_mm=5.0, status='ok')
        if src != 'gt' or res != 'native':
            rows.append(dict(base, level='mask', config='mask', dice=0.9 + 0.05 * rng.random(),
                             mask_hd95_mm=2.0, mask_assd_mm=0.5, pred_cm3=100.0, gt_cm3=102.0))
        configs = [dict(config='grid', blur_mm=b, blur_vox=None, taubin_iter=t, decimation=d)
                   for b, t, d in product((0.0, 1.0), (0, 30), (0.0, 0.5))]
        configs.append(dict(config='production', blur_mm=None, blur_vox=0.6, taubin_iter=30, decimation=0.6))
        for c in configs:
            canonical = c['config'] == 'grid' and c['blur_mm'] == 0 and c['taubin_iter'] == 0 and c['decimation'] == 0
            exact = canonical and src == 'gt' and res == 'native'
            assd = 0.0 if exact else (0.9 if src == 'ts_fast' else 0.1) + (0.3 if res == 'iso3' else 0) + 0.05 * rng.random()
            rows.append(dict(base, level='mesh', is_canonical=canonical, **c, assd_mm=assd, hd95_mm=3 * assd,
                             nsd=1 - assd / 2, mean_signed_mm=-assd / 4, volume_err_pct=-assd,
                             topology_ok=c['decimation'] < 0.5 or rng.random() < 0.5))
    pd.DataFrame(rows).to_csv(path / 'results.csv', index=False)
    (path / 'run.json').write_text(json.dumps({'wall_time_s': 60.0, 'commit': 'abc1234'}))


def test_report_tables_and_figures(tmp_path):
    pytest.importorskip('scipy')
    pytest.importorskip('matplotlib')
    synthetic_run(tmp_path)
    from research import tier2_report
    df, run = tier2_report.load(str(tmp_path))
    nums = tier2_report.tables(df, run, str(tmp_path / 'tables'))
    tier2_report.figures(df, str(tmp_path / 'figures'))
    for t in ('levels', 'mask', 'mesh', 'variance', 'paired', 'taubin', 'blur', 'topology'):
        text = (tmp_path / 'tables' / f'{t}.tex').read_text(encoding='utf-8')
        assert text.startswith('\\begin{tabular}') and '\\bottomrule' in text
    for f in ('assd_by_resolution', 'signed_by_resolution', 'dice_by_resolution', 'sampling_floor',
              'taubin_by_resolution', 'variance_shares', 'topology_by_decimation'):
        assert (tmp_path / 'figures' / f'{f}.pdf').stat().st_size > 0
    # the canonical expert mask at native is the reference itself: no ratio, but a paired difference
    assert nums['paired']['gt_canonical_iso3']['ratio_median'] is None
    assert nums['paired']['gt_canonical_iso3']['diff'] > 0.3
    assert nums['resampled_iso3'] == 3 and nums['resampled_native'] == 0
    shares = nums['variance']['all']
    assert set(shares) == {'source', 'resolution', 'blur_mm', 'taubin_iter', 'decimation', 'case'}
    assert shares['source'] > shares['blur_mm']
