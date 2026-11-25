"""Write shift/<name>.bin (see accuracy.write_shift_map) for existing datasets
that kept their TotalSegmentator masks in segmentations/.

    python research/make_shift_maps.py output/DATASET [output/DATASET ...]

Each delivered STL is compared point by point with the raw surface of its mask.
An STL read back keeps its triangle order, so the map lines up with the file.
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def one(args):
    ds, mask_path = args
    import nibabel as nib
    import numpy as np
    import pyvista as pv
    import accuracy
    name = os.path.basename(mask_path).replace('.nii.gz', '')
    stl = os.path.join(ds, f'{name}.stl')
    if not os.path.exists(stl):
        return None
    img = nib.load(mask_path)
    mask = np.asanyarray(img.dataobj) > 0
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    mesh = pv.read(stl)
    shift = accuracy.point_shift(mask, spacing, mesh)
    accuracy.write_shift_map(mesh, shift, ds, name)
    return name, float(np.median(shift)), float(np.percentile(shift, 95))


if __name__ == '__main__':
    for ds in sys.argv[1:]:
        masks = sorted(glob.glob(os.path.join(ds, 'segmentations', '*.nii.gz')))
        with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as ex:
            done = [r for r in ex.map(one, [(ds, m) for m in masks]) if r]
        med = sorted(r[1] for r in done)
        print(f'{ds}: {len(done)} maps; median of per-model medians {med[len(med) // 2]:.3f} mm; '
              f'largest 95th percentile {max(r[2] for r in done):.2f} mm')
