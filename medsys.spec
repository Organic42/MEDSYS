# -*- mode: python ; coding: utf-8 -*-
# Build: python -m PyInstaller medsys.spec --noconfirm
#
# Lite build (matches requirements-web.txt): no torch / TotalSegmentator /
# Redis. MedSAM refinement and the "AI organs" engine are unavailable —
# the UI hides that option automatically via /api/capabilities.

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [('web', 'web')]
binaries = []
hiddenimports = [
    'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto',
    'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan', 'uvicorn.lifespan.on',
    'multipart', 'multipart.multipart',
    'skimage.filters.rank', 'skimage.morphology',
    'sklearn.mixture', 'sklearn.utils._typedefs', 'sklearn.utils._heap',
    'sklearn.utils._sorting', 'sklearn.utils._vector_sentinel',
    'sklearn.neighbors._partition_nodes',
]

# Packages whose PyInstaller hooks are unreliable with plain hidden-imports —
# pull in everything (submodules + data files) to avoid missing-module
# crashes that only show up at runtime, not at build time.
for pkg in ('vtkmodules', 'pyvista', 'skimage', 'sklearn', 'SimpleITK',
           'numba', 'llvmlite', 'pydicom', 'nibabel', 'meshio',
           'brainextractor', 'pywt'):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

for pkg in ('scipy', 'matplotlib'):
    hiddenimports += collect_submodules(pkg)

block_cipher = None

a = Analysis(
    ['entry.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'segment_anything', 'totalsegmentator',
        'redis', 'rq', 'tkinter', 'PyQt5', 'PySide2', 'IPython', 'notebook',
        'pytest',
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MEDSYS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MEDSYS',
)
