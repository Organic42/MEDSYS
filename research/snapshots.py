"""Capture review snapshots of datasets from the running workbench (localhost:8000).

    python research/snapshots.py OUT_DIR DATASET [options as environment variables]

  ANGLES  comma list of name=azimuth/elevation, e.g. "front=270/0,left=0/0"
  HIDE    structure groups to hide, e.g. "bone,other organs"
  TAG     text added to file names
  SLICE   slice level 0..1 to add a CT-slice snapshot (top view)
  URL     workbench address (default http://localhost:8000)
  GL      "gpu" renders on the graphics card (full-detail models); default software

Opens the dataset directly by name, so no other dataset is ever shown.
"""
import os
import sys

from playwright.sync_api import sync_playwright

out, ds = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
angles = [a.split('=') for a in os.environ.get('ANGLES', 'front=270/0').split(',')]
tag = os.environ.get('TAG', '')
with sync_playwright() as pw:
    gl = (['--use-angle=d3d11', '--ignore-gpu-blocklist'] if os.environ.get('GL') == 'gpu'
          else ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    b = pw.chromium.launch(channel='msedge', headless=True, args=gl)
    pg = b.new_page(viewport={'width': 1600, 'height': 1000}, device_scale_factor=2)
    pg.goto(f"{os.environ.get('URL', 'http://localhost:8000')}/#{ds}")
    pg.reload()                                   # a hash-only change does not reload
    for _ in range(180):                          # until this dataset's meshes have loaded
        pg.wait_for_timeout(1000)
        if (pg.evaluate("() => document.getElementById('stageTitle').textContent") == ds and
                pg.evaluate("() => getComputedStyle(document.getElementById('loading')).display") == 'none'):
            break
    pg.wait_for_timeout(1500)
    pg.add_style_tag(content='#viewpanel,#hint{display:none!important}')
    for cat in filter(None, os.environ.get('HIDE', '').split(',')):
        pg.evaluate(f"""() => {{ for (const r of document.querySelectorAll('.catrow')) {{
            const n = r.querySelector('.catname'), c = r.querySelector('input');
            if (n && c && c.checked && n.textContent.trim().toLowerCase() === '{cat.strip().lower()}') c.click(); }} }}""")
    stage = pg.locator('#stage')
    for name, ae in angles:
        az, el = (float(v) for v in ae.split('/'))
        pg.evaluate(f'() => window.medsysView({az}, {el})')
        pg.wait_for_timeout(700)
        path = os.path.join(out, f'{ds}{tag}_{name}.png')
        stage.screenshot(path=path)
        print('saved', path)
    if os.environ.get('SLICE'):
        pg.evaluate(f"""() => {{ const on = document.getElementById('sliceOn'); on.checked = true;
            on.dispatchEvent(new Event('change')); const s = document.getElementById('slicePos');
            s.value = Math.round(s.max * {float(os.environ['SLICE'])}); s.dispatchEvent(new Event('input')); }}""")
        pg.evaluate('() => window.medsysView(270, 89)')
        pg.wait_for_timeout(900)
        path = os.path.join(out, f'{ds}{tag}_ct_slice_overlay.png')
        stage.screenshot(path=path)
        print('saved', path)
    b.close()
