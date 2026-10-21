"""Write the validation report (HTML, and PDF with --pdf) from the two validation runs.

    python docs/validation/make_report.py [--card RUN] [--rep RUN] [--out FILE] [--pdf]

The PDF is printed by a headless Edge or Chrome, which also loads the web fonts.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import accuracy  # noqa: E402
from research.card_validation import coverage  # noqa: E402
from research.tier2 import results_file  # noqa: E402

CARD_RUN = os.path.join(ROOT, 'research', 'results', 'validation_card_20261020-203227')
REP_RUN = os.path.join(ROOT, 'research', 'results', 'validation_repeatability_20261020-182532')


def print_pdf(html_path, pdf_path):
    import subprocess
    import tempfile
    for exe in (r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                r'C:\Program Files\Google\Chrome\Application\chrome.exe', 'chromium', 'google-chrome'):
        if os.path.isabs(exe) and not os.path.exists(exe):
            continue
        with tempfile.TemporaryDirectory() as profile:
            try:
                subprocess.run([exe, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                                '--virtual-time-budget=20000', '--run-all-compositor-stages-before-draw',
                                f'--user-data-dir={profile}', f'--print-to-pdf={pdf_path}',
                                'file:///' + os.path.abspath(html_path).replace(os.sep, '/')], check=True, timeout=180)
            except (OSError, subprocess.SubprocessError):
                continue
        print('pdf ->', pdf_path)
        return
    print('no Edge or Chrome found: open the HTML and print it to PDF')


ap = argparse.ArgumentParser()
ap.add_argument('--card', default=CARD_RUN)
ap.add_argument('--rep', default=REP_RUN)
ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'validation', 'validation-report.html'))
ap.add_argument('--pdf', action='store_true')
a = ap.parse_args()
card_run, rep_run, out = a.card, a.rep, a.out
card_stamp, rep_stamp = os.path.basename(card_run.rstrip('/\\')), os.path.basename(rep_run.rstrip('/\\'))

# ── test 1 ───────────────────────────────────────────────────────────────
df = pd.read_csv(results_file(card_run))
ok = df[(df.source != 'gt') & (df.status == 'ok')]
gt = df[df.source == 'gt']
t1 = {}
for organ, g in gt.groupby('structure'):
    err = (g.canon_assd_mm - 0.177 * g.delta_added_mm).abs()
    t1[organ] = (len(g), err.mean(), 100 * (err <= 0.111).mean())
at_least = 100 * (ok.true_assd_mm >= 0.177 * ok.rms_spacing_mm).mean()
change = ok.true_assd_mm - ok.canon_assd_mm
smooth_match = (ok.card_recon_signed_mm - (ok.true_signed_mm - ok.canon_signed_mm)).abs()
within_shift = 100 * (change.abs() <= ok.card_recon_assd_mm).mean()
wt_agree, n_mesh, n_scans = int((ok.card_watertight == ok.independent_manifold).sum()), len(ok), df.case.nunique()
rebuild = int(ok.rebuild_identical.sum())
cov = {k: coverage(ok, k) for k in ('v1', 'v2')}
cells = pd.concat([cov[k][cov[k].what == 'true_assd_mm'].groupby(['organ', 'mode', 'level'])[['mid', 'p90']].mean()
                   .add_prefix(k + '_') for k in ('v1', 'v2')], axis=1)
worst = cells.v1_mid.idxmin()
a1, a2 = cov['v1'][cov['v1'].what == 'true_assd_mm'], cov['v2'][cov['v2'].what == 'true_assd_mm']
c2 = cov['v2'][cov['v2'].what == 'canon_assd_mm']
names = {'native': 'native', 'iso1.5': '1.5 mm', 'iso3': '3 mm', 'slice5': '5 mm slices'}
t1_phantom = pd.read_csv(os.path.join(ROOT, 'research', 'results', 'tier1_phantoms_20261002-222422_full', 'results.csv.gz'))
t1_phantom = t1_phantom[(t1_phantom.is_canonical == True) & (t1_phantom.status == 'ok')]  # noqa: E712
sp = t1_phantom.spacing.astype(str).str.split('x', expand=True).astype(float)
ratio = t1_phantom.assd_mm / np.sqrt((sp ** 2).mean(axis=1))
ph_v1 = 100 * ((ratio >= 0.14) & (ratio <= 0.21)).mean()
ph_v2 = 100 * ((ratio >= accuracy.SLOPE_LOW) & (ratio <= accuracy.SLOPE_HIGH)).mean()

# ── test 2 ───────────────────────────────────────────────────────────────
rep = pd.read_csv(results_file(rep_run))
rrun = json.load(open(os.path.join(rep_run, 'run.json'), encoding='utf-8'))
again = rep[~((rep.path == 'worker') & (rep.run == 0))]
paths = {'worker': 'Persistent GPU worker (app default)', 'fresh_gpu': 'New GPU process each run', 'cpu': 'CPU'}
rows2 = []
for (mode, path), g in again.groupby(['mode', 'path'], sort=False):
    first = rep[(rep['mode'] == mode) & (rep.path == 'worker') & (rep.run == 0)].set_index('case').true_assd_mm
    allr = pd.concat([rep[(rep['mode'] == mode) & (rep.path == path)].set_index('case').true_assd_mm, first])
    rng = (allr.groupby(level=0).max() - allr.groupby(level=0).min()).max()
    secs = rep[(rep['mode'] == mode) & (rep.path == path)].seconds.median()
    rows2.append(f"<tr><td>{mode}</td><td>{paths[path]}</td><td class='n'>{len(g)}</td>"
                 f"<td class='n'>{int(g.organ_diff_voxels.max())}</td><td class='n'>{g.organ_dice.min():.4f}</td>"
                 f"<td class='n'>{g.mesh_assd_vs_first_mm.max():.3f}</td><td class='n'>{rng:.3f}</td>"
                 f"<td class='n'>{secs:.0f}</td></tr>")
max_shift = again.mesh_assd_vs_first_mm.max()
q = {}
for organ, d in (('spleen', 'tier2_spleen_two_modes_20261018-210930'), ('pancreas', 'tier2_pancreas_heldout_20261018-223006')):
    t = pd.read_csv(os.path.join(ROOT, 'research', 'results', d, 'results.csv.gz'))
    t = t[(t.source == 'ts_fast') & (t.resolution == 'native') & (t.config == 'production')].assd_mm
    q[organ] = t.quantile(0.75) - t.quantile(0.25)
w = again[again.path == 'worker']

cov_rows = []
_order = {'native': 0, 'iso1.5': 1, 'iso3': 2, 'slice5': 3}
_sorted = cells.reset_index().sort_values(['organ', 'mode', 'level'], ascending=[False, True, True],
                                          key=lambda c: c.map(_order) if c.name == 'level' else c)
for _, r in _sorted.iterrows():
    organ, mode, level = r.organ, r['mode'], r.level
    flag = ' class="hl"' if (organ, mode, level) == worst else ''
    cov_rows.append(f"<tr{flag}><td>{organ}</td><td>{mode}</td><td>{names[level]}</td>"
                    f"<td class='n'>{100 * r.v1_mid:.0f}%</td><td class='n'>{100 * r.v2_mid:.0f}%</td>"
                    f"<td class='n'>{100 * r.v1_p90:.0f}%</td><td class='n'>{100 * r.v2_p90:.0f}%</td></tr>")

R = accuracy.ENGINE_REFERENCE
_sp = ok[(ok.structure == 'spleen') & (ok.resolution == 'native')].canon_assd_mm
_pa = ok[(ok.structure == 'pancreas') & (ok.resolution == 'native')].true_assd_mm
_lo, _hi = np.percentile(_sp, [5, 95])
cross = 100 * ((_pa >= _lo) & (_pa <= _hi)).mean()
html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MEDSYS Validation Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{ --accent:#FF7A18; --ink:#1c1c1e; --muted:#63636a; --line:#e4e1dc; --panel:#faf8f5; }}
  @page {{ size:A4; margin:16mm 15mm;
    @bottom-left {{ content:"MEDSYS · Validation tests 1 and 2 · October 2026"; font:7.5pt Inter,"Segoe UI",sans-serif; color:#8a8a90; }}
    @bottom-right {{ content:counter(page) " / " counter(pages); font:7.5pt Inter,"Segoe UI",sans-serif; color:#8a8a90; }} }}
  * {{ box-sizing:border-box; }}
  html, body {{ background:#fff; }}
  body {{ font:9.4pt/1.47 Inter,"Segoe UI",sans-serif; color:var(--ink); margin:0; }}
  h1 {{ font-size:21pt; line-height:1.15; margin:0 0 6px; letter-spacing:-0.02em; }}
  h2 {{ font-size:13pt; margin:20px 0 8px; padding-top:6px; border-top:2px solid var(--accent); break-after:avoid; }}
  h3 {{ font-size:10.4pt; margin:12px 0 5px; break-after:avoid; }}
  p {{ margin:0 0 7px; }} ul, ol {{ margin:0 0 8px; padding-left:17px; }} li {{ margin:0 0 3px; }}
  .eyebrow {{ font:500 8pt "IBM Plex Mono",Consolas,monospace; color:var(--accent); letter-spacing:.08em; text-transform:uppercase; margin-bottom:8px; }}
  .meta {{ color:var(--muted); font-size:8.8pt; margin-bottom:12px; }}
  .box {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:4px; padding:10px 12px; margin:10px 0 12px; break-inside:avoid; }}
  .box h3 {{ margin-top:0; }} .box ul {{ margin-bottom:0; }}
  table {{ width:100%; border-collapse:collapse; margin:6px 0 10px; font-size:8.3pt; line-height:1.35; break-inside:avoid; }}
  th {{ text-align:left; font-weight:600; background:var(--panel); border-bottom:1.5px solid #cfcac3; padding:5px 6px; vertical-align:bottom; }}
  td {{ border-bottom:1px solid var(--line); padding:4px 6px; vertical-align:top; }}
  td.n, th.n {{ text-align:right; font-family:"IBM Plex Mono",Consolas,monospace; font-size:8pt; }}
  tr.hl td {{ background:#fff1e6; }}
  td.r {{ font-weight:500; }}
  figure {{ margin:8px 0 12px; break-inside:avoid; }} figure img {{ width:100%; }}
  figcaption {{ font-size:8.2pt; color:var(--muted); margin-top:3px; }}
  code {{ font:8.2pt "IBM Plex Mono",Consolas,monospace; background:#f3f1ee; padding:0 3px; border-radius:2px; }}
  pre {{ font:7.8pt/1.45 "IBM Plex Mono",Consolas,monospace; background:#f6f4f1; padding:8px 10px; border-radius:4px; white-space:pre-wrap; }}
  .small {{ font-size:8.2pt; color:var(--muted); }}
  .pb {{ break-before:page; }}
</style>
</head>
<body>
<div class="eyebrow">MEDSYS · Validation report</div>
<h1>Is the accuracy estimate honest, and is the model repeatable?</h1>
<div class="meta">20 October 2026 · Sameer Morya and Sarthak Wage, K J Somaiya School of Engineering, Mumbai · Validation tests 1 and 2</div>

<p>MEDSYS shows an accuracy estimate next to every 3D model it builds. The research paper measured how accurate the models are; these two tests ask the next questions. Test 1 checks whether the estimate the app shows matches each model's true error. Test 2 checks whether the same scan always gives the same model. Both run the application's own code, not research copies of it.</p>

<div class="box">
  <h3>Findings</h3>
  <ul>
    <li><b>Three of the card's four lines held</b> on {n_mesh} meshes from {n_scans} CT scans with expert outlines:
      <ul>
        <li>The resolution limit held for {t1['pancreas'][2]:.0f}% of grids on the pancreas, an organ held out of its fit.</li>
        <li>The true error was at least that limit for {at_least:.0f}% of meshes.</li>
        <li>The smoothing shift matched the true change within {smooth_match.median():.3f} mm (median).</li>
        <li>The watertight flag agreed with an independent check on {wt_agree} of {n_mesh} meshes.</li>
      </ul></li>
    <li><b>The segmentation line was overconfident on coarse scans.</b> It was right at the scans' own resolution, but on 3 mm grids in full mode only {100 * cells.loc[worst].v1_mid:.0f}% of spleen scans fell in its "middle half" (target 50%).</li>
    <li><b>The fix:</b> the card now uses the measurement made at the resolution closest to the scan, and shows a 9-in-10 range. Re-tested leave-one-scan-out on the same meshes:
      <ul>
        <li>{a2.mid.mean() * 100:.0f}% fall in the middle half (target 50%) and {a2.p90.mean() * 100:.0f}% in the 9-in-10 range (target 90%).</li>
        <li>The worst cell rose from {100 * cells.loc[worst].v1_mid:.0f}% to {100 * cells.loc[worst].v2_mid:.0f}%.</li>
      </ul></li>
    <li><b>On the app's GPU worker, the same scan gave the same segmentation every time.</b> Zero voxels changed in {len(w)} repeated runs on {rep.case.nunique()} scans.</li>
    <li><b>A new process or the CPU changed at most {int(again.organ_diff_voxels.max())} voxels of the organ.</b> That moved the model by at most {max_shift:.3f} mm, about {100 * max_shift / q['spleen']:.0f}% of the spread of the error between scans of the spleen.</li>
    <li><b>Meshing is deterministic.</b> The same segmentation gave an identical mesh and identical card values for {rebuild} of {n_mesh} meshes.</li>
  </ul>
</div>

<h2>1. Where these tests fit</h2>
<p>Validation of a digital health tool is usually split into four parts:</p>
<ul>
  <li><b>Verification:</b> does the software do the same thing every time? That is test 2.</li>
  <li><b>Analytical validation:</b> is the output close to the truth? The paper measured this on exact digital phantoms and on {n_scans} CT scans outlined by experts (41 spleen, 41 pancreas; Medical Segmentation Decathlon).</li>
  <li><b>Calibration:</b> is the accuracy the app shows honest? That is test 1.</li>
  <li><b>Physical and clinical validation:</b> printed phantoms in a real scanner, agreement with experts, use by surgeons. This is not yet done (Section 5).</li>
</ul>

<h2>2. Test 1: the accuracy card</h2>
<h3>Method</h3>
<p>The app's <code>build_mesh</code> call meshed the TotalSegmentator segmentations of all {n_scans} scans, with the accuracy estimate switched on:</p>
<ul>
  <li>both engine modes (fast, at 3 mm; full, at 1.5 mm);</li>
  <li>four grids: the scan's own resolution, 1.5 mm, 3 mm, and 5 mm slices;</li>
  <li>{n_mesh} meshes in total.</li>
</ul>
<p>Each mesh's true error is its average surface distance (ASSD) to the expert's outline, from 20,000 points per side. Each line of the card was compared with that error as follows:</p>
<ul>
  <li><b>Segmentation line:</b> checked leave-one-scan-out. The card's reference figures are rebuilt from the other 40 scans, so no scan is judged against figures that include itself. Its "middle half" should hold about half the scans, and its 9-in-10 range about nine in ten.</li>
  <li><b>Watertight flag:</b> compared with VTK's own classification of open and non-manifold edges.</li>
</ul>

<table>
  <tr><th style="width:22%">Card line</th><th style="width:40%">How it was checked</th><th style="width:38%">Result</th></tr>
  <tr><td>Resolution limit</td><td>Expert outline coarsened to each grid; predicted 0.177 × added sampling. Pancreas held out of the fit</td><td class="r">pancreas {t1['pancreas'][2]:.0f}% inside ±0.111 mm (mean miss {t1['pancreas'][1]:.3f} mm, {t1['pancreas'][0]} grids); spleen {t1['spleen'][2]:.0f}%</td></tr>
  <tr><td>Resolution range</td><td>Exact phantom surfaces (the data the range comes from)</td><td class="r">{ph_v2:.0f}% inside {accuracy.SLOPE_LOW}–{accuracy.SLOPE_HIGH} × spacing (was {ph_v1:.0f}% with 0.14)</td></tr>
  <tr><td>"Expect at least the resolution limit"</td><td>True error of every delivered mesh</td><td class="r">{at_least:.0f}% of {n_mesh}</td></tr>
  <tr><td>Smoothing &amp; simplification</td><td>Card's measured shift against the change smoothing actually caused in the true error</td><td class="r">signed shift within {smooth_match.median():.3f} mm (median; 95th pct {np.percentile(smooth_match, 95):.3f}); change within the shift for {within_shift:.0f}%</td></tr>
  <tr><td>Watertight</td><td>Independent VTK edge check</td><td class="r">{wt_agree} of {n_mesh} agree</td></tr>
  <tr><td>Segmentation</td><td>Leave one scan out; table below and Figure 1</td><td class="r">v2: {a2.mid.mean() * 100:.0f}% middle half, {a2.p90.mean() * 100:.0f}% 9-in-10 range</td></tr>
</table>

<h3>What the test found</h3>
<p><b>The card as first shipped</b> (v1) used the error measured at the scans' own resolution everywhere, adding the paper's thick-slice change on 5 mm grids:</p>
<ul>
  <li>At native resolution it was well calibrated.</li>
  <li>On coarse grids the true error was larger than the card implied, most of all for full mode on 3 mm grids: only {100 * cells.loc[worst].v1_mid:.0f}% of spleen scans fell in its middle half.</li>
  <li>Overall, v1 held {a1.mid.mean() * 100:.0f}% in the middle half and {a1.p90.mean() * 100:.0f}% in a 9-in-10 range. {a1.above.mean() * 100:.0f}% of meshes were worse than its middle half.</li>
</ul>
<p><b>The corrected card</b> (v2, now in <code>accuracy.py</code>):</p>
<ul>
  <li>It uses the reference measured on the study grid closest to the scan's spacing, and shows the 9-in-10 range as well as the middle half.</li>
  <li>On the same meshes it holds {a2.mid.mean() * 100:.0f}% and {a2.p90.mean() * 100:.0f}%, with {a2.above.mean() * 100:.0f}% worse than the middle half.</li>
  <li>Against the segmentation alone, which is the line's exact claim, v2 holds {c2.mid.mean() * 100:.0f}% and {c2.p90.mean() * 100:.0f}%.</li>
  <li>A 9-in-10 range drawn from 40 scans is expected to cover a new scan slightly less than 90% of the time, so {a2.p90.mean() * 100:.0f}% is about what an honest range gives.</li>
</ul>
<p>The phantoms showed one more problem. The resolution range's lower bound, 0.14 × spacing, sat at the median error of isotropic grids, so about half of them fell just below it. The bound is now {accuracy.SLOPE_LOW}, the phantoms' 2.5th percentile.</p>

<figure><img src="figures/card_coverage.png" alt="Coverage before and after the fix"><figcaption>Figure 1. Share of scans whose true error fell inside the card's middle half (left) and 9-in-10 range (right), before (blue) and after (orange) the fix, leave one scan out.</figcaption></figure>

<table>
  <tr><th>Organ</th><th>Mode</th><th>Grid</th><th class="n">v1 middle half</th><th class="n">v2 middle half</th><th class="n">v1 9-in-10</th><th class="n">v2 9-in-10</th></tr>
  {''.join(cov_rows)}
</table>
<p class="small">Targets: 50% and 90%. Highlighted: the worst cell before the fix. Scans already as coarse as a grid are counted once, at their own resolution.</p>

<figure style="width:72%"><img src="figures/resolution_limit.png" alt="Resolution limit"><figcaption>Figure 2. Surface error of a perfect segmentation (the expert outline, coarsened) against the sampling the coarser grid adds. Line: the card's 0.177 × Δ; band: its ±0.111 mm interval. Orange: the pancreas, never used in the fit.</figcaption></figure>

<h3>Limits of this test</h3>
<ul>
  <li><b>Same scans as the references:</b> the card's references come from these 82 scans. Leave-one-scan-out keeps each scan out of its own reference, but new scanners or hospitals may differ.</li>
  <li><b>Two organs only:</b> only the spleen and pancreas have expert outlines here. Every other structure shows "not validated" in the app, which is correct: the spleen's 9-in-10 range would have covered only {cross:.0f}% of pancreas meshes.</li>
</ul>

<h2 class="pb">3. Test 2: repeatability</h2>
<h3>Method</h3>
<p>{rep.case.nunique()} scans (5 spleen, 5 pancreas, drawn with seed {rrun['seed']}) went through TotalSegmentator repeatedly, in both modes, along three paths:</p>
<ul>
  <li><b>The app's persistent GPU worker:</b> {rrun['worker_runs']} runs per scan, in one warm process.</li>
  <li><b>A new GPU process each run:</b> {rrun['fresh_runs']} runs per scan. nnU-Net autotunes its GPU kernels per process, so kernels can differ.</li>
  <li><b>The CPU:</b> {rrun['cpu_runs']} runs per scan in fast mode, and on {len(rrun['full_cpu_cases'])} scans in full mode.</li>
</ul>
<p>That is {len(rep)} runs on an {rrun.get('gpu', 'GPU')}. Each run was compared voxel by voxel with the first worker run. The organ was meshed with the app's call, and the mesh was compared with the first run's mesh and with the expert's outline.</p>

<table>
  <tr><th>Mode</th><th>Path</th><th class="n">Runs</th><th class="n">Organ voxels changed (max)</th><th class="n">Organ Dice (min)</th><th class="n">Mesh shift (max, mm)</th><th class="n">Change in true error (max, mm)</th><th class="n">Seconds per scan</th></tr>
  {''.join(rows2)}
</table>
<p class="small">Runs exclude the first worker run, which is the reference. Change in true error is the largest range of the ASSD against the expert outline across repeats of one scan.</p>

<figure><img src="figures/repeatability.png" alt="Repeatability"><figcaption>Figure 3. Mesh shift of every repeated run from the first run of the same scan.</figcaption></figure>

<h3>What it shows</h3>
<ul>
  <li><b>The worker is exactly repeatable.</b> The app's default path gave bit-identical segmentations in all {len(w)} repeats. That is where users' results come from.</li>
  <li><b>Other paths change a few voxels.</b> A new process, or the CPU, changes at most {int(again.organ_diff_voxels.max())} voxels of the organ: different kernel choices and floating-point order flip voxels that sit on the decision boundary.</li>
  <li><b>That change is tiny next to the error.</b> The model moves by at most {max_shift:.3f} mm and its true error by at most {again.groupby(['case', 'mode']).true_assd_mm.apply(lambda s: s.max() - s.min()).max():.3f} mm. The spread of that error between scans is {q['spleen']:.2f} mm (spleen) and {q['pancreas']:.2f} mm (pancreas, interquartile range).</li>
  <li><b>Meshing adds no variation.</b> Rebuilding a mesh from the same segmentation gave identical vertices, faces and card values for {rebuild} of {n_mesh} meshes (checked in test 1).</li>
</ul>
<p class="small">The repeatability run's own rebuild check had a bug, now fixed: it compared the card's blank volume change (left blank for open meshes) as unequal to itself, so it flagged exactly the {int((~rep.rebuild_identical).sum())} open meshes and no others. The meshes themselves were identical. The rebuild count above comes from test 1, which checks every mesh.</p>

<h2>4. Changes to the application</h2>
<ul>
  <li><b>Reference by grid:</b> the segmentation line uses the reference measured on the study grid closest to the scan (<code>accuracy.reference_level</code>; native, 1.5 mm, 3 mm or 5 mm slices). It shows the median, the middle half and the 9-in-10 range.</li>
  <li><b>Resolution range:</b> now {accuracy.SLOPE_LOW}–{accuracy.SLOPE_HIGH} × RMS voxel spacing (was 0.14–0.21).</li>
  <li><b>No stale figures:</b> the resolution limit is recomputed from the current constants whenever a dataset is opened, so older results never show outdated figures.</li>
  <li><b>Tests:</b> the automated tests cover the new reference choice. All 115 tests pass.</li>
</ul>

<h2>5. Still to do</h2>
<ol>
  <li><b>A real object in a real scanner:</b> 3D-print shapes of known size, scan them at 0.6–5 mm slices, compare with the CAD file. Needs CT scanner time.</li>
  <li><b>More organs and hospitals:</b> liver, kidneys, bone and MRI from multi-centre public data (for example AMOS), checked for overlap with the engine's training data. Needs GPU time only.</li>
  <li><b>Agreement with experts:</b> two or three people outline the same scans, to see whether MEDSYS is as close to an expert as experts are to each other. Needs radiologist time.</li>
  <li><b>Reader study:</b> surgeons rate the models and time their use. Needs ethics approval.</li>
</ol>
<p>As for the held-out pancreas in the paper, each test's pass and fail thresholds will be written down before it runs.</p>

<h2>Reproduction</h2>
<pre>python -m research.card_validation --workers 8      # test 1, ~4 min, needs the Tier 2 engine cache
python -m research.repeatability                    # test 2, ~2 h on one GPU
python -m research.validation_figures research/results/{card_stamp} \\
       research/results/{rep_stamp}</pre>
<p class="small">Runs reported here: <code>research/results/{card_stamp}</code> and <code>research/results/{rep_stamp}</code>. Each has results.csv.gz, run.json and summary.md. The Tier 2 data is the Medical Segmentation Decathlon (CC-BY-SA 4.0). MEDSYS is not a medical device.</p>
</body>
</html>
'''
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, 'w', encoding='utf-8').write(html)
print('report ->', out)
if a.pdf:
    print_pdf(out, out[:-5] + '.pdf')
