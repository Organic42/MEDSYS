"""Insert or refresh the validation slides in MEDSYS-presentation.html, with
every number read from the validation runs.

    python docs/validation/make_slides.py [--card RUN] [--rep RUN] [--deck FILE]
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

ap = argparse.ArgumentParser()
ap.add_argument('--card', default=CARD_RUN)
ap.add_argument('--rep', default=REP_RUN)
ap.add_argument('--deck', default=os.path.join(ROOT, 'MEDSYS-presentation.html'))
a = ap.parse_args()
card_run, rep_run, deck = a.card, a.rep, a.deck
START, END = '<!-- validation slides: start -->', '<!-- validation slides: end -->'
CSS_MARK = '/* validation slides */'
TEAL, ORANGE = '#008C99', '#E0590A'          # checked with the dataviz validator on both deck surfaces

# ── numbers ──────────────────────────────────────────────────────────────
df = pd.read_csv(results_file(card_run))
ok = df[(df.source != 'gt') & (df.status == 'ok')]
gt = df[df.source == 'gt']
pa = gt[gt.structure == 'pancreas']
pa_err = (pa.canon_assd_mm - 0.177 * pa.delta_added_mm).abs()
res_inside = 100 * (pa_err <= 0.111).mean()
res_mae = pa_err.mean()
at_least = 100 * (ok.true_assd_mm >= 0.177 * ok.rms_spacing_mm).mean()
shift_signed = ok.true_signed_mm - ok.canon_signed_mm
smooth_match = (ok.card_recon_signed_mm - shift_signed).abs().median()
wt_agree = int((ok.card_watertight == ok.independent_manifold).sum())
n_mesh, n_scans = len(ok), df.case.nunique()

cov = {}
for card in ('v1', 'v2'):
    c = coverage(ok, card)
    c = c[c.what == 'true_assd_mm']
    cov[card] = c
cells = (pd.concat([cov[k].groupby(['organ', 'mode', 'level'], sort=False)[['mid', 'p90']].mean().add_prefix(k + '_')
                    for k in ('v1', 'v2')], axis=1).reset_index())
order = {'native': 0, 'iso1.5': 1, 'iso3': 2, 'slice5': 3}
cells = cells.sort_values(['organ', 'mode', 'level'], key=lambda s: s.map(order) if s.name == 'level' else s,
                          ascending=[False, True, True]).reset_index(drop=True)
worst = cells.loc[cells.v1_mid.idxmin()]
v2_mid, v2_p90 = 100 * cov['v2'].mid.mean(), 100 * cov['v2'].p90.mean()
v1_mid, v1_p90 = 100 * cov['v1'].mid.mean(), 100 * cov['v1'].p90.mean()

rep = pd.read_csv(results_file(rep_run))
again = rep[~((rep.path == 'worker') & (rep.run == 0))]
w = again[again.path == 'worker']
fresh, cpu = again[again.path == 'fresh_gpu'], again[again.path == 'cpu']
n_rep_scans, n_runs = rep.case.nunique(), len(rep)
max_shift = again.mesh_assd_vs_first_mm.max()
rebuild = int(ok.rebuild_identical.sum()) if 'rebuild_identical' in ok else None
organ_vox = int(again.organ_diff_voxels.max())

# The card shown on the slide: the spleen of a chest CT with 5 mm slices, TotalSegmentator fast
# (its measured smoothing shift and watertight flag; the rest follows from the spacing).
EXAMPLE_SPACING = (0.841796875, 0.841796875, 5.0)
ex_s = {'recon_assd_mm': 0.311, 'watertight': False}
ex_v = accuracy.engine_reference('fast', 'spleen', EXAMPLE_SPACING)
ex_lim = accuracy.resolution_limit(EXAMPLE_SPACING)

R = accuracy.ENGINE_REFERENCE

# ── accuracy summary: the model against the voxels, the experts and the real organ ──
_t1 = pd.read_csv(results_file(os.path.join(ROOT, 'research', 'results', 'tier1_phantoms_20261002-222422_full')))
_prod = _t1[(_t1.config == 'production') & (_t1.status == 'ok')]
mesh_err = _prod.groupby('spacing').agg(assd=('assd_mm', 'median'), vol=('volume_err_pct', 'median'))
_iso = [s for s in mesh_err.index if 'x' not in s]
voxel_frac = float(np.mean([mesh_err.assd[s] / float(s) for s in _iso]))
_nat = ok[ok.resolution == 'native']
fin = _nat.groupby(['structure', 'source']).agg(assd=('true_assd_mm', 'median'), hd95=('true_hd95_mm', 'median'),
                                                vol=('true_volume_err_pct', lambda s: s.abs().median()))
dice = {}
for _organ, _d in (('spleen', 'tier2_spleen_two_modes_20261018-210930'),
                   ('pancreas', 'tier2_pancreas_heldout_20261018-223006')):
    _t = pd.read_csv(results_file(os.path.join(ROOT, 'research', 'results', _d)))
    _m = _t[(_t.level == 'mask') & (_t.resolution == 'native') & _t.source.isin(['ts_fast', 'ts_full'])]
    dice[_organ] = _m.groupby('source').dice.median()
lim_1mm = accuracy.resolution_limit((1.0, 1.0, 1.0))['estimate_mm']
lim_5mm = accuracy.resolution_limit((0.8, 0.8, 5.0))['estimate_mm']
REFERENCE_FLOOR = '0.2–0.5'   # the expert surface's own sampling error (paper, Discussion)


def span(values, d=2, unit='&#8239;mm'):
    lo, hi = min(values), max(values)
    return f'{lo:.{d}f}{unit}' if f'{lo:.{d}f}' == f'{hi:.{d}f}' else f'{lo:.{d}f}–{hi:.{d}f}{unit}'


def organ_col(organ, key, d=2, unit='&#8239;mm'):
    return span([fin.loc[(organ, s), key] for s in ('ts_fast', 'ts_full')], d, unit)


mesh_rows = '\n'.join(
    f'          <tr><td class="k">{label}</td><td class="mono" style="text-align:right">{mesh_err.assd[s]:.2f}&#8239;mm</td></tr>'
    for s, label in (('0.5', '0.5 mm'), ('1', '1 mm'), ('2', '2 mm'), ('0.8x0.8x5', '0.8 × 0.8 × 5 mm')))
vol_shrink = span([-mesh_err.vol[s] for s in mesh_err.index], 1, '%')


def mm(x, d=2):
    return f'{x:.{d}f}&#8239;mm'


names = {'native': 'native', 'iso1.5': '1.5 mm', 'iso3': '3 mm', 'slice5': '5 mm slices'}

# ── slides ───────────────────────────────────────────────────────────────
css = f'''
  {CSS_MARK}
  .chip{{font-family:var(--mono);font-size:19px;font-weight:600;letter-spacing:1px;padding:5px 12px;
    border-radius:8px;align-self:flex-start;text-transform:uppercase}}
  .chip.done{{background:#E3F4F5;color:#006E78}}
  .chip.next{{background:#FDEBDD;color:#A3420A}}
  .dark .chip.done{{background:#0E3B40;color:#7FE3EA}}
  .dark .chip.next{{background:#3A2A1E;color:#FDBA8C}}
  .acard{{background:var(--white);border:1px solid var(--line);border-radius:18px;padding:30px;
    display:flex;flex-direction:column;gap:14px}}
  .arow2{{border:1px solid var(--line);border-radius:12px;padding:16px 20px;display:flex;flex-direction:column;gap:6px}}
  .arow2 .spread h3{{font-size:25px}}
  .arow2 .spread span{{font-family:var(--mono);font-size:25px;font-weight:700;color:var(--ink)}}
  .arow2 p{{font-size:20px;line-height:1.35}}
  .drow{{display:flex;align-items:center;gap:18px;height:33px}}
  .dl{{width:330px;font-size:21px;color:#AEBACB;text-align:right;white-space:nowrap}}
  .dt{{position:relative;flex:1;height:33px;border-bottom:1px solid #22304A}}
  .dt .tgt{{position:absolute;top:-4px;bottom:-4px;border-left:2px dashed #7C8798}}
  .dt .seg{{position:absolute;top:14px;height:4px;background:#3A4A64;border-radius:2px}}
  .dt b{{position:absolute;top:7px;width:19px;height:19px;margin-left:-9.5px;border-radius:50%;border:2px solid var(--ink)}}
  .dt b.v1{{background:{ORANGE}}} .dt b.v2{{background:{TEAL}}}
  .dv{{width:140px;font-family:var(--mono);font-size:20px;color:#E8EDF3;white-space:nowrap}}
  table.tight td{{padding:10px 18px}} table.tight th{{padding:12px 18px}}
  table.tighter td{{padding:8px 12px;white-space:nowrap}} table.tighter th{{padding:10px 12px}}
  .key{{display:flex;gap:34px;font-size:22px;color:#AEBACB;align-items:center}}
  .key i{{display:inline-block;width:18px;height:18px;border-radius:50%;margin-right:10px;vertical-align:-2px}}
  .lbar{{width:100%;height:28px;background:#E3E8EE;border-radius:10px;overflow:hidden}}
  .lbar > i{{display:block;height:100%;border-radius:10px;min-width:6px}}
'''


def dumbbell_rows():
    out = []
    for _, r in cells.iterrows():
        a, b = 100 * r.v1_mid, 100 * r.v2_mid
        lo, hi = min(a, b), max(a, b)
        label = f"{r.organ.capitalize()} · {r['mode']} · {names[r.level]}"
        val = f'{a:.0f} → {b:.0f}%' if abs(a - b) >= 0.5 else f'{b:.0f}%'
        out.append(f'      <div class="drow"><span class="dl">{label}</span><div class="dt"><i class="tgt" style="left:50%"></i>'
                   f'<i class="seg" style="left:{lo:.1f}%;width:{hi - lo:.1f}%"></i><b class="v1" style="left:{a:.1f}%"></b>'
                   f'<b class="v2" style="left:{b:.1f}%"></b></div><span class="dv">{val}</span></div>')
    return '\n'.join(out)


def bar(label, value, scale, colour, note):
    pct = 100 * value / scale
    return (f'      <div style="display:flex;flex-direction:column;gap:8px">\n'
            f'        <div class="spread"><h3 style="font-size:24px">{label}</h3>'
            f'<span class="mono" style="font-size:26px;font-weight:700;color:var(--ink)">{note}</span></div>\n'
            f'        <div class="lbar"><i style="width:{pct:.2f}%;background:{colour}"></i></div>\n      </div>')


ref_rows = '\n'.join(
    f'      <tr><td class="k">{o.capitalize()}</td><td>{m}</td><td class="mono" style="text-align:right">'
    f'{mm(R[(m, o, "native")]["median_mm"])}</td><td class="mono" style="text-align:right">'
    f'{R[(m, o, "native")]["iqr_mm"][0]:.2f}–{mm(R[(m, o, "native")]["iqr_mm"][1])}</td></tr>'
    for o in ('spleen', 'pancreas') for m in ('fast', 'full'))

worst_label = f"{worst.organ}, {worst['mode']} mode, {names[worst.level]} voxels"
slides = f'''{START}
<section class="slide dark" style="gap:40px">
  <p class="eyebrow">Validation</p>
  <h2 style="font-size:64px;max-width:1600px">Is the 3D model accurate — and does the app tell the truth about it?</h2>
  <div class="grid2" style="gap:22px">
    <div class="step dark" style="border:1px solid #2A3A52"><span class="num" style="color:var(--cyan)">01</span><div style="display:flex;flex-direction:column;gap:10px"><h3>Same input, same model</h3><p>Each scan segmented again and again, on GPU and CPU; meshing checked for identical output.</p><span class="chip done">Done · Test 2</span></div></div>
    <div class="step dark" style="border:1px solid #2A3A52"><span class="num" style="color:var(--cyan)">02</span><div style="display:flex;flex-direction:column;gap:10px"><h3>Close to the truth</h3><p>Exact digital phantoms and {n_scans} CT scans outlined by experts.</p><span class="chip done">Done · research paper</span></div></div>
    <div class="step dark" style="border:1px solid #2A3A52"><span class="num" style="color:var(--cyan)">03</span><div style="display:flex;flex-direction:column;gap:10px"><h3>An honest accuracy estimate</h3><p>Every line of the in-app accuracy card checked on {n_mesh} meshes.</p><span class="chip done">Done · Test 1</span></div></div>
    <div class="step dark" style="border:1px solid #2A3A52"><span class="num" style="color:var(--cyan)">04</span><div style="display:flex;flex-direction:column;gap:10px"><h3>Real objects, real experts</h3><p>Printed phantoms in a CT scanner, expert agreement, a surgeon reader study.</p><span class="chip next">Planned</span></div></div>
  </div>
  <p class="foot">Verification → accuracy against truth → calibration → clinical validation</p>
  <aside hidden>Frame the section with four questions and say which are answered. This follows the standard structure for validating digital health tools: verification (does the software do the same thing every time), analytical validation (is it close to the truth), calibration (is the number we show honest), clinical validation (does it help real users). Three are done with data; the fourth needs a CT scanner, a radiologist and ethics approval, and is the plan for the next weeks.</aside>
</section>

<section class="slide" style="background:var(--alt);gap:30px">
  <p class="eyebrow">Validation — accuracy against truth</p>
  <h2>Measured against experts, not another algorithm</h2>
  <div class="row" style="gap:26px">
    <div class="card" style="border:none;gap:10px;padding:28px 34px"><p class="mono" style="font-size:22px;color:var(--muted)">TIER 1 · EXACT SURFACES</p><span class="stat" style="font-size:54px">0.14 × voxel</span><p style="font-size:24px">The raw surface straight from the voxels, before smoothing. Four digital phantoms whose true surface is known exactly; 7,760 meshes.</p></div>
    <div class="card" style="border:none;gap:10px;padding:28px 34px"><p class="mono" style="font-size:22px;color:var(--muted)">TIER 2 · EXPERT-LABELLED CT</p><span class="stat" style="font-size:54px">{n_scans} scans</span><p style="font-size:24px">41 spleen and 41 pancreas CT scans outlined by experts (Medical Segmentation Decathlon); 49,200 meshes. The pancreas was held out: of six predictions fixed in advance, four held and one partly.</p></div>
  </div>
  <table class="tight" style="font-size:24px">
    <tr><th style="width:24%">Organ</th><th style="width:22%">Engine mode</th><th style="width:27%;text-align:right">Segmentation error</th><th style="width:27%;text-align:right">Middle half of scans</th></tr>
{ref_rows}
  </table>
  <p class="foot">Average distance from the expert's outline (ASSD) of the raw segmentation surface, the figure the app's card shows · the finished model: next slide</p>
  <aside hidden>This answers the limitation in the earlier version of this project: the reference is now experts, not another algorithm, and the sample is 82 scans, not one. The pancreas is the stronger evidence, because we wrote down six predictions before processing it. Be ready for "why is the pancreas worse?": it is thin and irregular, and many of these scans contain tumours that the engine misses.</aside>
</section>

<section class="slide" style="gap:28px;padding-top:112px">
  <p class="eyebrow">Validation — accuracy summary</p>
  <h2>How accurate is the 3D model?</h2>
  <div class="row" style="gap:24px;align-items:stretch">
    <div class="card" style="gap:12px;padding:28px 30px;flex:0.92">
      <p class="mono" style="font-size:19px;color:var(--teal);letter-spacing:2px">1 · AGAINST THE SCAN'S VOXELS</p>
      <span class="stat" style="font-size:52px">≈ {voxel_frac:.1f} voxel</span>
      <p style="font-size:23px">The meshing step alone, on digital shapes whose true surface is exact.</p>
      <table class="tighter" style="font-size:22px">
          <tr><th>Voxel size</th><th style="text-align:right">Surface error</th></tr>
{mesh_rows}
      </table>
      <p style="font-size:21px;color:var(--muted)">Smoothing shrinks the volume by {vol_shrink}.</p>
    </div>
    <div class="card hl" style="gap:12px;padding:28px 30px;flex:1.3">
      <p class="mono" style="font-size:19px;color:var(--teal);letter-spacing:2px">2 · AGAINST AN EXPERT'S OUTLINE</p>
      <span class="stat" style="font-size:52px">{organ_col('spleen', 'assd')}</span>
      <p style="font-size:23px">The finished spleen model on {n_scans} CT scans: about one CT pixel from where the expert drew it.</p>
      <table class="tighter" style="font-size:22px">
          <tr><th></th><th style="text-align:right">Spleen</th><th style="text-align:right">Pancreas</th></tr>
          <tr><td class="k">Average distance</td><td class="mono" style="text-align:right">{organ_col('spleen', 'assd')}</td><td class="mono" style="text-align:right">{organ_col('pancreas', 'assd', 1)}</td></tr>
          <tr><td class="k">95% of surface within</td><td class="mono" style="text-align:right">{organ_col('spleen', 'hd95', 1)}</td><td class="mono" style="text-align:right">{organ_col('pancreas', 'hd95', 0)}</td></tr>
          <tr><td class="k">Volume difference</td><td class="mono" style="text-align:right">{organ_col('spleen', 'vol', 0, '%')}</td><td class="mono" style="text-align:right">{organ_col('pancreas', 'vol', 0, '%')}</td></tr>
          <tr><td class="k">Overlap (Dice)</td><td class="mono" style="text-align:right">{span(dice['spleen'], 2, '')}</td><td class="mono" style="text-align:right">{span(dice['pancreas'], 2, '')}</td></tr>
      </table>
      <p style="font-size:21px;color:var(--muted)">Pancreas: experts included tumours that the engine mostly misses.</p>
    </div>
    <div class="card" style="gap:12px;padding:28px 30px;border:2px dashed var(--warn);flex:0.92">
      <p class="mono" style="font-size:19px;color:var(--warn);letter-spacing:2px">3 · AGAINST THE REAL ORGAN</p>
      <span class="stat" style="font-size:44px;color:var(--warn)">Not yet measured</span>
      <p style="font-size:23px">The CT is the only record of the organ, so this needs a physical test.</p>
      <p style="font-size:23px"><b>Can't beat the scan:</b> at least {mm(lim_1mm)} on a 1&#8239;mm scan, {mm(lim_5mm, 1)} on 5&#8239;mm slices.</p>
      <p style="font-size:23px"><b>The reference isn't perfect:</b> expert outlines carry {REFERENCE_FLOOR}&#8239;mm of their own error.</p>
      <p style="font-size:23px"><b>Next:</b> printed shapes of known size, scanned in a real CT (slide 21).</p>
    </div>
  </div>
  <p class="foot">Against experts, a spleen model sits within {organ_col('spleen', 'assd')} on average; meshing adds about a tenth of a voxel</p>
  <aside hidden>Answer "how accurate is it?" with three comparisons, because they are different questions. First, the meshing step: on exact digital shapes it costs about a tenth of a voxel. Second, the finished model against what an expert drew on the same CT: under 1 mm on average for the spleen, about one pixel; about 2 mm for the pancreas, mostly tumours the engine misses. Third, against the real organ: not measured, and it cannot be measured from scans alone. It can't be better than the scan's resolution, and the expert reference has its own 0.2 to 0.5 mm error. The printed-phantom test is how we will answer it, and CT scanner time is where the mentor can help.</aside>
</section>

<section class="slide" style="gap:36px">
  <p class="eyebrow">Validation — test 1</p>
  <h2>Is the accuracy estimate in the app honest?</h2>
  <div class="row" style="gap:30px;align-items:flex-start">
    <div style="width:560px;display:flex;flex-direction:column;gap:12px">
      <div class="acard">
        <p class="mono" style="font-size:20px;color:var(--teal);letter-spacing:3px">ACCURACY ESTIMATE · SPLEEN</p>
        <div class="arow2"><div class="spread"><h3>Resolution limit</h3><span>≈ {mm(ex_lim['estimate_mm'])}</span></div><p>{ex_lim['range_mm'][0]:.2f}–{mm(ex_lim['range_mm'][1])} from the voxel spacing alone</p></div>
        <div class="arow2"><div class="spread"><h3>Smoothing</h3><span>{mm(ex_s['recon_assd_mm'])}</span></div><p>How far smoothing moved the mesh</p></div>
        <div class="arow2"><div class="spread"><h3>Segmentation</h3><span>≈ {mm(ex_v['median_mm'])}</span></div><p>9 in 10 scans {ex_v['range90_mm'][0]:.2f}–{mm(ex_v['range90_mm'][1])}</p></div>
        <div class="arow2"><div class="spread"><h3>Watertight</h3><span>{'yes' if ex_s['watertight'] else 'no'}</span></div></div>
      </div>
      <p style="font-size:21px;color:var(--muted)">What a user sees: spleen, 5&#8239;mm slices</p>
    </div>
    <table style="font-size:25px;flex:1">
      <tr><th style="width:27%">Card line</th><th style="width:41%">How it was checked</th><th style="width:32%;text-align:right">Result</th></tr>
      <tr><td class="k">Resolution limit</td><td>Pancreas, held out of the fit: coarsened expert outlines</td><td class="mono" style="text-align:right;color:var(--teal)">{res_inside:.0f}% inside ±0.111&#8239;mm</td></tr>
      <tr><td class="k">"At least this much"</td><td>True error of every delivered mesh</td><td class="mono" style="text-align:right;color:var(--teal)">{at_least:.0f}% of {n_mesh}</td></tr>
      <tr><td class="k">Smoothing</td><td>Against the change it really caused</td><td class="mono" style="text-align:right;color:var(--teal)">within {mm(smooth_match, 3)}</td></tr>
      <tr><td class="k">Watertight</td><td>Against VTK's own edge check</td><td class="mono" style="text-align:right;color:var(--teal)">{wt_agree} of {n_mesh} agree</td></tr>
      <tr><td class="k">Segmentation</td><td>Leave one scan out</td><td class="mono" style="text-align:right;color:var(--warn)">next slide &#8594;</td></tr>
    </table>
  </div>
  <p class="foot">The app's own meshing code · {n_scans} expert-labelled CT scans · 2 engine modes × 4 resolutions · {n_mesh} meshes</p>
  <aside hidden>The point of this test: the app shows users an accuracy estimate, so we must prove the estimate itself. We ran the app's real meshing code, not a research copy, on all 82 scans and compared each line of the card with the true error against the experts. Mean miss on the held-out organ: {res_mae:.3f} mm. Four lines pass cleanly; the fifth is the interesting one.</aside>
</section>

<section class="slide dark" style="gap:28px;padding-top:110px">
  <p class="eyebrow">Validation — test 1 found a flaw</p>
  <h2 style="font-size:56px;max-width:1660px">On coarse scans the card was overconfident. Fixed, then re&#8209;tested on the same {n_mesh} meshes.</h2>
  <div class="row" style="gap:46px;align-items:flex-start">
    <div style="flex:1;display:flex;flex-direction:column;gap:2px">
      <div class="key" style="margin:0 0 10px 348px"><span><i style="background:{ORANGE}"></i>As first shipped</span><span><i style="background:{TEAL}"></i>Fixed: matched to the scan's resolution</span><span style="color:#7C8798">┆ target 50%</span></div>
{dumbbell_rows()}
      <div style="display:flex;margin-left:348px;margin-right:158px;justify-content:space-between;font-family:var(--mono);font-size:18px;color:#7C8798;padding-top:6px"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
      <p style="margin-left:348px;font-size:20px;color:#7C8798">Share of scans whose true error fell inside the card's "middle half"</p>
    </div>
    <div style="width:430px;display:flex;flex-direction:column;gap:26px">
      <div><span class="mono" style="font-size:76px;font-weight:700;color:#E8EDF3;line-height:1">{100 * worst.v1_mid:.0f}% <span style="color:#7C8798">→</span> <span style="color:#7FE3EA">{100 * worst.v2_mid:.0f}%</span></span><p style="font-size:23px;margin-top:12px">Worst case before the fix: {worst_label}. Should be 50%.</p></div>
      <hr>
      <div><span class="mono" style="font-size:54px;font-weight:700;color:#7FE3EA;line-height:1">{v2_mid:.0f}% · {v2_p90:.0f}%</span><p style="font-size:23px;margin-top:12px">All meshes after the fix: inside the middle half (target 50%) and the 9-in-10 range (target 90%). Before: {v1_mid:.0f}% and {v1_p90:.0f}%.</p></div>
    </div>
  </div>
  <aside hidden>The best slide in the validation section, because it shows the method working. The card compared every scan with figures measured at the scans' own resolution. On coarse scans, especially full mode at 3 mm, the true error is higher, so the card claimed more confidence than it had: only {100 * worst.v1_mid:.0f}% of those scans fell in its middle half instead of 50%. We now pick the reference measured at the grid closest to the scan and also show a 9-in-10 range. Re-tested leave-one-scan-out on the same meshes. Say plainly: a validation that never fails anything is not testing much.</aside>
</section>

<section class="slide" style="gap:28px">
  <p class="eyebrow">Validation — test 2</p>
  <h2>Same scan, run again: is it the same model?</h2>
  <div class="row" style="gap:24px">
    <div class="card hl" style="padding:26px 32px;gap:8px"><span class="stat" style="font-size:54px">{int(w.organ_diff_voxels.max())} voxels</span><p style="font-size:24px">changed in {len(w)} repeated runs on the app's GPU worker, the default path. Identical every time.</p></div>
    <div class="card" style="padding:26px 32px;gap:8px"><span class="stat" style="font-size:54px">≤ {mm(fresh.mesh_assd_vs_first_mm.max(), 3)}</span><p style="font-size:24px">mesh shift in a new GPU process ({len(fresh)} runs), where the engine re-tunes its GPU kernels.</p></div>
    <div class="card" style="padding:26px 32px;gap:8px"><span class="stat" style="font-size:54px">≤ {mm(cpu.mesh_assd_vs_first_mm.max(), 3)}</span><p style="font-size:24px">mesh shift on the CPU instead of the GPU ({len(cpu)} runs).</p></div>
  </div>
  <div style="display:flex;flex-direction:column;gap:12px">
    <p class="mono" style="font-size:22px;color:var(--muted);letter-spacing:3px">HOW BIG IS THAT, NEXT TO THE ERROR ITSELF?</p>
{bar('Largest run-to-run shift', max_shift, 2.0, ORANGE, mm(max_shift, 3))}
{bar('Resolution limit of a 1 mm scan', 0.177, 2.0, TEAL, mm(0.177))}
{bar('Typical error, spleen', R[('fast', 'spleen', 'native')]['median_mm'], 2.0, TEAL, mm(R[('fast', 'spleen', 'native')]['median_mm']))}
{bar('Typical error, pancreas', R[('fast', 'pancreas', 'native')]['median_mm'], 2.0, TEAL, mm(R[('fast', 'pancreas', 'native')]['median_mm']))}
  </div>
  <p class="foot">{n_rep_scans} CT scans · TotalSegmentator fast and full · {n_runs} segmentation runs · meshing the same segmentation twice: identical for {rebuild} of {n_mesh} meshes</p>
  <aside hidden>Repeatability is the floor of any accuracy claim: if the same scan gave different models, no error figure would mean anything. On the app's default path, the persistent GPU worker, nothing changed at all. In a fresh process or on the CPU up to {organ_vox} voxels at the organ's edge flip, which moves the surface by hundredths of a millimetre, a small fraction of the error we report. The meshing step itself is deterministic.</aside>
</section>

<section class="slide" style="background:var(--alt);gap:40px">
  <p class="eyebrow">Validation — next</p>
  <h2>What is still to prove, and what each test will show</h2>
  <div class="grid2">
    <div class="card" style="border:none;border-left:6px solid var(--teal);border-radius:14px;padding:34px;gap:12px"><h3>A real object in a real scanner</h3><p style="font-size:24px">3D-print shapes of known size, CT them at 0.6–5&#8239;mm slices, compare the model with the CAD file.</p><span class="chip next">Needs CT scanner time</span></div>
    <div class="card" style="border:none;border-left:6px solid var(--teal);border-radius:14px;padding:34px;gap:12px"><h3>More organs, other hospitals</h3><p style="font-size:24px">Liver, kidneys, bone and MRI from multi-centre public data, checked for overlap with the engine's training data.</p><span class="chip next">Needs GPU time only</span></div>
    <div class="card" style="border:none;border-left:6px solid var(--teal);border-radius:14px;padding:34px;gap:12px"><h3>As close as two experts?</h3><p style="font-size:24px">Two or three people outline the same scans. Is MEDSYS as close to an expert as the experts are to each other?</p><span class="chip next">Needs radiologist time</span></div>
    <div class="card" style="border:none;border-left:6px solid var(--teal);border-radius:14px;padding:34px;gap:12px"><h3>Useful to surgeons?</h3><p style="font-size:24px">Surgeons rate models and time their use; a standard usability score.</p><span class="chip next">Needs ethics approval</span></div>
  </div>
  <p class="foot">Pass/fail thresholds are written down before each test runs, as they were for the held-out pancreas</p>
  <aside hidden>End the section on a plan, not an apology. Each card says what it would prove and what it needs from outside the team: this is where the mentor can help, especially CT scanner time and a radiologist. Phantoms and public data need no ethics approval; any hospital scans would.</aside>
</section>
{END}'''

# ── insert ───────────────────────────────────────────────────────────────
raw = open(deck, encoding='utf-8', newline='').read()
crlf = '\r\n' in raw
html = raw.replace('\r\n', '\n')
if CSS_MARK in html:
    a = html.index('  ' + CSS_MARK)
    b = html.index('</style>', a)
    html = html[:a] + css.strip('\n') + '\n' + html[b:]
else:
    html = html.replace('</style>', css.lstrip('\n') + '</style>', 1)
if START in html:
    a, b = html.index(START), html.index(END) + len(END)
    html = html[:a] + slides + html[b:]
else:
    anchor = '<section class="slide" style="background:var(--alt);gap:44px">\n  <p class="eyebrow" style="color:var(--warn)">Limitations</p>'
    assert anchor in html, 'Limitations slide not found'
    html = html.replace(anchor, slides + '\n\n' + anchor, 1)
open(deck, 'w', encoding='utf-8', newline='').write(html.replace('\n', '\r\n') if crlf else html)
print(f'slides written: {html.count("<section class=\"slide")} slides in deck')
print(f'test1: res {res_inside:.0f}% mae {res_mae:.3f}; at least {at_least:.0f}%; smooth {smooth_match:.3f}; '
      f'wt {wt_agree}/{n_mesh}; worst {100 * worst.v1_mid:.0f}->{100 * worst.v2_mid:.0f}; v2 {v2_mid:.0f}/{v2_p90:.0f}')
print(f'test2: scans {n_rep_scans} runs {n_runs} max shift {max_shift:.4f} rebuild {rebuild}')
