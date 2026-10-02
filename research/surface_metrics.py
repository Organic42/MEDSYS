"""
Geometric comparison of a triangle mesh with an exact phantom surface.

Distances are symmetric and area-weighted:
  mesh -> truth : uniform-by-area points on the mesh, exact |SDF| to the phantom
  truth -> mesh : uniform-by-area points on the phantom, point-to-triangle
                  distance to the mesh (exact via VTK when available)

Definitions (all in mm unless stated):
  ASSD       mean of all distances in both directions
  HD95       max of the two directed 95th percentiles (MONAI convention)
  Hausdorff  max distance in either direction
  NSD        surface Dice at tolerance tol: share of both samples within tol
  mean_signed  mean SDF over the mesh; < 0 means the mesh sits inside the
               true surface on average (shrinkage), > 0 means expansion

numpy/scipy only; PyVista is optional and used for exact distances.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from research.phantoms import sdf_world


def _corners(verts, faces):
    return verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]


def triangle_areas(verts, faces):
    a, b, c = _corners(verts, faces)
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def mesh_area(verts, faces):
    return float(triangle_areas(verts, faces).sum())


def mesh_volume(verts, faces):
    """Enclosed volume by the divergence theorem; meaningful for closed meshes."""
    a, b, c = _corners(verts, faces)
    return float(abs(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()) / 6.0)


def sample_mesh_surface(verts, faces, n, rng):
    """n points uniformly distributed by area over the mesh."""
    areas = triangle_areas(verts, faces)
    idx = rng.choice(len(faces), size=n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    a, b, c = _corners(verts, faces[idx])
    return a + u[:, None] * (b - a) + v[:, None] * (c - a)


def mesh_topology(faces):
    """Topology from connectivity alone: components, holes, Euler number, genus.

    Edges used by one face are boundary edges (holes); by more than two,
    non-manifold. For a closed manifold mesh with c components,
    V - E + F = 2c - 2g, which gives the total genus g.
    """
    used = np.unique(faces)
    n_v, n_f = len(used), len(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    edges, counts = np.unique(e, axis=0, return_counts=True)
    n_e = len(edges)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    remap = np.full(int(faces.max()) + 1, -1)
    remap[used] = np.arange(n_v)
    graph = coo_matrix((np.ones(n_e), (remap[edges[:, 0]], remap[edges[:, 1]])),
                       shape=(n_v, n_v))
    n_comp, _ = connected_components(graph, directed=False)
    euler = n_v - n_e + n_f
    closed = boundary == 0 and nonmanifold == 0
    genus = (2 * n_comp - euler) / 2 if closed else float('nan')
    return {'n_vertices': n_v, 'n_faces': n_f, 'components': int(n_comp),
            'boundary_edges': boundary, 'nonmanifold_edges': nonmanifold,
            'euler': int(euler), 'closed': bool(closed), 'genus': genus}


def point_to_mesh_distance(points, verts, faces, approx_samples=400_000, rng=None):
    """Unsigned distance from each point to the nearest triangle.

    Exact (VTK cell locator) when PyVista is installed. Otherwise approximated
    by the nearest of approx_samples area-uniform mesh samples, which can only
    overestimate the true distance. Returns (distances, exact_flag).
    """
    try:
        import pyvista as pv
    except ImportError:
        rng = rng if rng is not None else np.random.default_rng(0)
        dense = sample_mesh_surface(verts, faces, approx_samples, rng)
        d, _ = cKDTree(dense).query(points)
        return d, False
    fa = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    mesh = pv.PolyData(np.asarray(verts, dtype=float), fa)
    _, closest = mesh.find_closest_cell(points, return_closest_point=True)
    return np.linalg.norm(points - closest, axis=1), True


def compare_to_phantom(verts, faces, phantom, placement, truth_points, rng,
                       n_mesh_samples=20_000, tol_mm=1.0):
    """All Tier 1 metrics for one mesh against the exact phantom surface."""
    mesh_pts = sample_mesh_surface(verts, faces, n_mesh_samples, rng)
    signed = sdf_world(phantom, mesh_pts, placement)
    d_mesh = np.abs(signed)
    d_truth, exact = point_to_mesh_distance(truth_points, verts, faces, rng=rng)
    both = np.concatenate([d_mesh, d_truth])
    topo = mesh_topology(faces)
    vol = mesh_volume(verts, faces)
    area = mesh_area(verts, faces)
    return {
        'assd_mm': float(both.mean()),
        'hd95_mm': float(max(np.percentile(d_mesh, 95), np.percentile(d_truth, 95))),
        'hausdorff_mm': float(max(d_mesh.max(), d_truth.max())),
        'nsd': float((both <= tol_mm).mean()),
        'mean_signed_mm': float(signed.mean()),
        'volume_err_pct': float((vol - phantom.volume) / phantom.volume * 100),
        'area_err_pct': float((area - phantom.area) / phantom.area * 100),
        'genus_expected': phantom.genus,
        'topology_ok': bool(topo['closed'] and topo['components'] == 1
                            and topo['genus'] == phantom.genus),
        'distance_exact': bool(exact),
        **topo,
    }
