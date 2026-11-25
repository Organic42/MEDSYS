"""Write lite/<name>.stl for every mesh of existing datasets (see save_lite).

    python research/make_lite.py output/DATASET [output/DATASET ...]
"""
import glob
import os
import sys

import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmentation_pipeline import save_lite  # noqa: E402

for ds in sys.argv[1:]:
    files = glob.glob(os.path.join(ds, '*.stl')) + glob.glob(os.path.join(ds, 'meshes', '*.stl'))
    for f in files:
        save_lite(pv.read(f), ds, os.path.splitext(os.path.basename(f))[0])
    print(f'{ds}: {len(files)} lite meshes')
