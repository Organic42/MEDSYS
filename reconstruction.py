"""
Surface reconstruction: binary mask -> triangle mesh.

Every stage is an explicit, recordable setting (ReconParams), so research
experiments can vary blur, isosurface algorithm, smoothing and decimation
independently. Production uses the same code: build_mesh() in
segmentation_pipeline.py fills ReconParams with its hand-tuned per-structure
values, so a result measured here describes the meshes MEDSYS actually ships.

The isosurface stage needs only numpy/scipy/scikit-image. Smoothing and
decimation need PyVista (VTK), imported lazily.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure

ISOSURFACE_METHODS = ('lewiner', 'lorensen')


@dataclass(frozen=True)
class ReconParams:
    """One reconstruction configuration.

    blur_sigma_vox: Gaussian blur of the mask before isosurfacing, per array
        axis, in voxels (all zeros disables it). Research code states blur in
        millimetres and converts with blur_mm_to_vox(), so that blur is not
        confounded with voxel spacing.
    isosurface: marching-cubes variant passed to skimage ('lewiner' resolves
        ambiguous cube configurations; 'lorensen' is the classic algorithm).
    taubin_iter: Taubin smoothing iterations (0 disables it).
    decimation: target fraction of triangles to remove (0 disables it).
    """
    blur_sigma_vox: tuple = (0.0, 0.0, 0.0)
    isosurface: str = 'lewiner'
    taubin_iter: int = 0
    taubin_pass_band: float = 0.05
    decimation: float = 0.0
    decimation_feature_angle: float = 45.0


def blur_mm_to_vox(sigma_mm, spacing):
    """Convert an isotropic blur in millimetres to per-axis sigmas in voxels."""
    return tuple(float(sigma_mm) / float(s) for s in spacing)


def extract_isosurface(mask, spacing, params, origin=(0.0, 0.0, 0.0)):
    """Blur the mask and extract its 0.5 isosurface.

    Returns (verts, faces) as numpy arrays, verts in world millimetres
    (array index * spacing + origin). Raises if no surface exists.
    """
    if params.isosurface not in ISOSURFACE_METHODS:
        raise ValueError(f"unknown isosurface method: {params.isosurface!r}")
    vol = mask.astype(np.float32)
    if any(s > 0 for s in params.blur_sigma_vox):
        vol = gaussian_filter(vol, sigma=params.blur_sigma_vox)
    verts, faces, _, _ = measure.marching_cubes(
        vol, level=0.5, spacing=tuple(float(s) for s in spacing),
        gradient_direction='descent', allow_degenerate=False,
        method=params.isosurface)
    verts = verts + np.asarray(origin, dtype=verts.dtype)
    return verts, faces


def postprocess_surface(verts, faces, params):
    """Taubin smoothing then decimation, returning a PyVista PolyData."""
    import pyvista as pv
    fa = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    mesh = pv.PolyData(verts, fa)
    if params.taubin_iter > 0:
        mesh = mesh.smooth_taubin(n_iter=params.taubin_iter,
                                  pass_band=params.taubin_pass_band,
                                  normalize_coordinates=True)
    if params.decimation > 0:
        mesh = mesh.decimate_pro(reduction=params.decimation,
                                 feature_angle=params.decimation_feature_angle,
                                 preserve_topology=True)
    return mesh


def reconstruct_surface(mask, spacing, params, origin=(0.0, 0.0, 0.0)):
    """Full reconstruction: blur -> isosurface -> smoothing -> decimation."""
    verts, faces = extract_isosurface(mask, spacing, params, origin)
    return postprocess_surface(verts, faces, params)


def mesh_arrays(mesh):
    """PyVista triangle mesh -> (verts, faces) numpy arrays."""
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return np.asarray(mesh.points, dtype=np.float64), faces
