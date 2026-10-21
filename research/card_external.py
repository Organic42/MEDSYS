"""Validation test 4: the accuracy estimate on data it has never seen.

Test 1 checked the card's segmentation line leave one scan out on the 82 scans
its figures come from. Here the card is frozen as shipped (accuracy.py) and
applied to AMOS 2022 CT scans: other centres, vendors and annotators. For each
delivered mesh, does its true error against the AMOS expert surface fall inside
the card's stated middle half (target 50%) and 9-in-10 range (target 90%)?

Uses the cached TotalSegmentator masks from a Tier 2 run on AMOS, so run that
first (python -m research.tier2 --dataset amos ... ).

    python -m research.card_external [--cases N] [--workers 6]

AMOS labels the pancreas without tumour; the card's pancreas figures come from
MSD Task07, whose pancreas includes tumour. A difference there is a difference
of definition as much as of data, and is reported separately.
"""
import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research import tier2  # noqa: E402
from research.card_validation import SOURCES, run_case  # noqa: E402

DATASET = 'amos'
ORGANS = ('spleen', 'pancreas')          # the structures the card has figures for


def cached(cid, organ):
    return all(os.path.exists(tier2.cached_mask_path(DATASET, cid, s, organ)) for s in SOURCES)


def summarise(df, run):
    ok = df[(df.source != 'gt') & (df.status == 'ok')].copy()
    ok['mid'] = (ok.true_assd_mm >= ok.card_engine_q1_mm) & (ok.true_assd_mm <= ok.card_engine_q3_mm)
    ok['p90'] = (ok.true_assd_mm >= ok.card_engine_p5_mm) & (ok.true_assd_mm <= ok.card_engine_p95_mm)
    ok['above'] = ok.true_assd_mm > ok.card_engine_q3_mm
    ok['below'] = ok.true_assd_mm < ok.card_engine_q1_mm
    lim_ok = (ok.true_assd_mm >= ok.card_resolution_mm).mean()
    gt = df[df.source == 'gt']
    L = ['# Validation test 4: the accuracy estimate on unseen data (AMOS)\n',
         f"Run {run['started']}. The card frozen as shipped, applied to {df.case.nunique()} AMOS CT scans "
         f"({len(ok)} delivered meshes). Targets: 50% inside the middle half, 90% inside the 9-in-10 range."
         + (' **Pilot: too few scans for a verdict.**' if df.case.nunique() < 20 else '') + '\n',
         '| Organ | Mode | Meshes | Middle half | 9-in-10 range | Below middle half | Above middle half | '
         'Median true ASSD (mm) | Card median (mm) |',
         '|---|---|---|---|---|---|---|---|---|']
    for (organ, source), g in ok.groupby(['structure', 'source']):
        L.append(f'| {organ} | {source[3:]} | {len(g)} | {100 * g.mid.mean():.0f}% | {100 * g.p90.mean():.0f}% | '
                 f'{100 * g.below.mean():.0f}% | {100 * g.above.mean():.0f}% | {g.true_assd_mm.median():.3f} | '
                 f'{g.card_engine_median_mm.median():.3f} |')
    L.append(f'\nAll meshes: middle half {100 * ok.mid.mean():.0f}%, 9-in-10 range {100 * ok.p90.mean():.0f}%. '
             f'True error at least the resolution limit: {100 * lim_ok:.0f}%. '
             f'Watertight flag agrees with VTK: {int((ok.card_watertight == ok.independent_manifold).sum())} of {len(ok)}.\n')
    if len(gt):
        import accuracy
        miss = (gt.canon_assd_mm - accuracy.SLOPE * gt.delta_added_mm).abs()
        L.append(f'Resolution limit on AMOS expert masks coarsened to each grid: {100 * (miss <= accuracy.PREDICTION_INTERVAL_MM).mean():.0f}% '
                 f'of {len(gt)} grids inside +/-{accuracy.PREDICTION_INTERVAL_MM} mm (mean miss {miss.mean():.3f} mm).\n')
    L.append('AMOS labels the pancreas without tumour; the card\'s pancreas figures include tumour (MSD Task07).\n')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'amos', 'amos22'))
    ap.add_argument('--cases', type=int, default=None)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    import pandas as pd
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    cases = [c for c in tier2.list_cases(DATASET, args.data) if any(cached(c[0], o) for o in ORGANS)][:args.cases]
    tasks = [(DATASET, cid, n, lbl, args.seed, organ) for cid, n, _, lbl in cases for organ in ORGANS
             if cached(cid, organ)]
    if not tasks:
        sys.exit('no cached AMOS engine masks: run research.tier2 on AMOS first')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    outdir = args.outdir or os.path.join(ROOT, 'output', 'research', 'card_external', stamp)
    os.makedirs(outdir, exist_ok=True)
    run = {'test': 'card_external', 'started': stamp, 'dataset': DATASET, 'scans': len(cases), 'tasks': len(tasks),
           'sources': SOURCES, 'seed': args.seed, 'python': platform.python_version()}
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(run_case, t) for t in tasks]), 1):
            rows += fut.result()
            print(f'{i}/{len(tasks)}  {time.time() - t0:.0f} s', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'results.csv'), index=False)
    run['seconds'] = round(time.time() - t0)
    with open(os.path.join(outdir, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump(run, fh, indent=1)
    text = summarise(df, run)
    with open(os.path.join(outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(text)
    print(f'-> {outdir}')


if __name__ == '__main__':
    main()
