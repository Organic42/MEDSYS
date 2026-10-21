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
             (Tier 2 also reports 0.5 and 2 mm, and the 99th percentile)
  defect     on the test mesh, faces whose three vertices all lie farther
             than DEFECT_MM from the reference: their share of the area and
             the area of the largest connected patch (local errors that a
             global mean hides)
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


def signed_volume(verts, faces):
    """Signed enclosed volume: positive when face normals point outward."""
    a, b, c = _corners(verts, faces)
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def signed_distance_to_mesh(points, verts, faces):
    """Signed distance from points to a closed mesh: negative inside.

    Uses VTK's implicit distance, whose sign follows the face winding; the
    sign is normalised with the mesh's signed volume so an inward-facing
    mesh cannot silently invert shrinkage and expansion. Requires PyVista.
    """
    import pyvista as pv
    fa = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    ref = pv.PolyData(np.asarray(verts, dtype=float), fa)
    probe = pv.PolyData(np.asarray(points, dtype=float))
    d = np.asarray(probe.compute_implicit_distance(ref)['implicit_distance'], dtype=float)
    return d if signed_volume(verts, faces) >= 0 else -d


DEFECT_MM = 2.0


def defect_patches(verts, faces, ref_verts, ref_faces, tau=DEFECT_MM):
    """(largest patch area in mm2, area share) of faces farther than tau from the reference.

    A face is displaced when its three vertices are; displaced faces that
    share a vertex form one patch.
    """
    d, _ = point_to_mesh_distance(verts, ref_verts, ref_faces)
    bad = (d[faces] > tau).all(axis=1)
    if not bad.any():
        return 0.0, 0.0
    areas = triangle_areas(verts, faces)
    f = faces[bad]
    n = len(f)
    incidence = coo_matrix((np.ones(3 * n), (np.repeat(np.arange(n), 3), f.ravel())),
                           shape=(n, len(verts))).tocsr()
    _, label = connected_components(incidence @ incidence.T, directed=False)
    patch = np.bincount(label, weights=areas[bad])
    return float(patch.max()), float(areas[bad].sum() / areas.sum())


def compare_meshes(test_verts, test_faces, ref_verts, ref_faces, rng,
                   n_samples=20_000, tol_mm=1.0, local=True):
    """Tier 2 metrics for a test mesh against a reference mesh.

    Same definitions as compare_to_phantom, with the reference surface in
    place of the analytic one: distances in both directions are exact
    point-to-triangle distances (VTK). mean_signed_mm > 0 means the test
    mesh lies outside the reference on average (expansion). local=False
    skips the per-vertex defect patches (NaN), the costliest part, for
    quick checks such as the app's accuracy estimate.
    """
    test_pts = sample_mesh_surface(test_verts, test_faces, n_samples, rng)
    ref_pts = sample_mesh_surface(ref_verts, ref_faces, n_samples, rng)
    d_test, exact_a = point_to_mesh_distance(test_pts, ref_verts, ref_faces, rng=rng)
    d_ref, exact_b = point_to_mesh_distance(ref_pts, test_verts, test_faces, rng=rng)
    both = np.concatenate([d_test, d_ref])
    signed = signed_distance_to_mesh(test_pts, ref_verts, ref_faces)
    topo = mesh_topology(test_faces)
    ref_vol, ref_area = mesh_volume(ref_verts, ref_faces), mesh_area(ref_verts, ref_faces)
    vol, area = mesh_volume(test_verts, test_faces), mesh_area(test_verts, test_faces)
    patch_mm2, defect_share = (defect_patches(test_verts, test_faces, ref_verts, ref_faces) if local
                               else (float('nan'), float('nan')))
    return {
        'assd_mm': float(both.mean()),
        'hd95_mm': float(max(np.percentile(d_test, 95), np.percentile(d_ref, 95))),
        'hausdorff_mm': float(max(d_test.max(), d_ref.max())),
        'nsd': float((both <= tol_mm).mean()),
        'nsd_0.5mm': float((both <= 0.5).mean()),
        'nsd_2mm': float((both <= 2.0).mean()),
        'p99_mm': float(max(np.percentile(d_test, 99), np.percentile(d_ref, 99))),
        'defect_patch_mm2': patch_mm2,
        'defect_area_pct': 100 * defect_share,
        'missed_ref_pct': float(100 * (d_ref > DEFECT_MM).mean()),
        'mean_signed_mm': float(signed.mean()),
        'volume_err_pct': float((vol - ref_vol) / ref_vol * 100) if ref_vol else float('nan'),
        'area_err_pct': float((area - ref_area) / ref_area * 100) if ref_area else float('nan'),
        'ref_volume_cm3': ref_vol / 1000.0,
        'topology_ok': bool(topo['closed'] and topo['nonmanifold_edges'] == 0),
        'distance_exact': bool(exact_a and exact_b),
        **topo,
    }


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
