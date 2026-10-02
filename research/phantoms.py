"""
Tier 1 digital phantoms: shapes whose true surface is known exactly.

Each phantom has an exact signed distance function (SDF): negative inside,
zero on the surface, and equal to the true Euclidean distance to the surface.
That makes the distance from any mesh point to the true surface exact, with
no reference mesh involved, so reconstruction error can be measured without
the reference-surface confound present in Tier 2.

All lengths are millimetres. Only numpy/scipy are required.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


class Phantom:
    """Base class. Subclasses define the SDF in local coordinates."""
    name = 'phantom'
    genus = 0

    def sdf_local(self, p):
        raise NotImplementedError

    @property
    def volume(self):
        raise NotImplementedError

    @property
    def area(self):
        raise NotImplementedError

    @property
    def half_extents(self):
        """Half-extents of the local axis-aligned bounding box."""
        raise NotImplementedError

    @property
    def bounding_radius(self):
        """Radius of a sphere about the local origin enclosing the shape."""
        raise NotImplementedError


class Sphere(Phantom):
    """Smooth, convex, constant curvature."""
    name = 'sphere'

    def __init__(self, radius=15.0):
        self.r = float(radius)

    def sdf_local(self, p):
        return np.linalg.norm(p, axis=-1) - self.r

    @property
    def volume(self):
        return 4.0 / 3.0 * np.pi * self.r ** 3

    @property
    def area(self):
        return 4.0 * np.pi * self.r ** 2

    @property
    def half_extents(self):
        return np.array([self.r] * 3)

    @property
    def bounding_radius(self):
        return self.r


class Capsule(Phantom):
    """Thin tube with hemispherical caps, along local z: a vessel-like structure."""
    name = 'capsule'

    def __init__(self, radius=2.0, half_length=20.0):
        self.r = float(radius)
        self.h = float(half_length)

    def sdf_local(self, p):
        z = np.clip(p[..., 2], -self.h, self.h)
        q = p.copy()
        q[..., 2] = p[..., 2] - z
        return np.linalg.norm(q, axis=-1) - self.r

    @property
    def volume(self):
        return np.pi * self.r ** 2 * (2 * self.h) + 4.0 / 3.0 * np.pi * self.r ** 3

    @property
    def area(self):
        return 2 * np.pi * self.r * (2 * self.h) + 4 * np.pi * self.r ** 2

    @property
    def half_extents(self):
        return np.array([self.r, self.r, self.h + self.r])

    @property
    def bounding_radius(self):
        return self.h + self.r


class Torus(Phantom):
    """Ring in the local xy plane: genus 1, so it tests topology preservation."""
    name = 'torus'
    genus = 1

    def __init__(self, major=18.0, minor=6.0):
        if minor >= major:
            raise ValueError("torus needs minor < major")
        self.R = float(major)
        self.r = float(minor)

    def sdf_local(self, p):
        ring = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2) - self.R
        return np.sqrt(ring ** 2 + p[..., 2] ** 2) - self.r

    @property
    def volume(self):
        return 2 * np.pi ** 2 * self.R * self.r ** 2

    @property
    def area(self):
        return 4 * np.pi ** 2 * self.R * self.r

    @property
    def half_extents(self):
        return np.array([self.R + self.r, self.R + self.r, self.r])

    @property
    def bounding_radius(self):
        return self.R + self.r


class RoundedBox(Phantom):
    """Box with rounded edges (box Minkowski-summed with a sphere): flat faces
    and tight curvature, closer to bone than the smooth shapes are."""
    name = 'rounded_box'

    def __init__(self, half_extents=(14.0, 10.0, 6.0), rounding=2.0):
        self.b = np.asarray(half_extents, dtype=float)
        self.rr = float(rounding)

    def sdf_local(self, p):
        q = np.abs(p) - self.b
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside - self.rr

    @property
    def volume(self):
        bx, by, bz = self.b
        r = self.rr
        return (8 * bx * by * bz + 8 * (bx * by + by * bz + bx * bz) * r
                + 2 * np.pi * r ** 2 * (bx + by + bz) + 4.0 / 3.0 * np.pi * r ** 3)

    @property
    def area(self):
        bx, by, bz = self.b
        r = self.rr
        return (8 * (bx * by + by * bz + bx * bz) + 4 * np.pi * r * (bx + by + bz)
                + 4 * np.pi * r ** 2)

    @property
    def half_extents(self):
        return self.b + self.rr

    @property
    def bounding_radius(self):
        return float(np.linalg.norm(self.b)) + self.rr


def default_phantoms():
    """The four Tier 1 shapes: smooth, thin, genus-1 and flat-faced."""
    return [Sphere(), Capsule(), Torus(), RoundedBox()]


@dataclass(frozen=True)
class Placement:
    """Rigid placement of a phantom in world coordinates."""
    rotation: Rotation
    center: np.ndarray

    def to_local(self, points):
        return self.rotation.inv().apply(np.asarray(points, dtype=float) - self.center)

    def to_world(self, points):
        return self.rotation.apply(points) + self.center


def random_placement(rng, max_offset_mm=0.0):
    """Uniformly random rotation and a random centre offset in [0, max_offset_mm).

    The rotation comes from a normalised Gaussian quaternion, which is uniform
    over SO(3) and avoids version-specific scipy random APIs. The offset moves
    the shape relative to the voxel grid so results do not depend on alignment.
    """
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    center = rng.uniform(0.0, max_offset_mm, size=3) if max_offset_mm > 0 else np.zeros(3)
    return Placement(Rotation.from_quat(q), center)


def identity_placement():
    return Placement(Rotation.identity(), np.zeros(3))


def sdf_world(phantom, points, placement):
    """Exact signed distance (mm) from world points to the placed phantom."""
    return phantom.sdf_local(placement.to_local(points))


def voxelise(phantom, placement, spacing, margin_mm=4.0):
    """Sample the placed phantom on a voxel grid (voxel centre inside = 1).

    Returns (mask, origin) where origin is the world position of voxel (0,0,0)
    and array axis i has spacing spacing[i]. The grid encloses the shape's
    bounding sphere plus margin_mm, enough for blur and smoothing to act freely.
    """
    spacing = np.asarray(spacing, dtype=float)
    half = phantom.bounding_radius + margin_mm
    lo = placement.center - half
    n = np.ceil(2 * half / spacing).astype(int) + 1
    axes = [lo[d] + np.arange(n[d]) * spacing[d] for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    mask = (sdf_world(phantom, pts, placement) < 0).reshape(n)
    return mask, lo


def _sdf_gradient_local(phantom, p, h=1e-5):
    g = np.empty_like(p)
    for d in range(3):
        e = np.zeros(3)
        e[d] = h
        g[:, d] = (phantom.sdf_local(p + e) - phantom.sdf_local(p - e)) / (2 * h)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-12)


def sample_surface(phantom, placement, n, rng, shell_mm=0.02, batch=1_000_000,
                   max_batches=400):
    """Points spread uniformly by area over the exact surface, in world mm.

    Draw points uniformly in the local bounding box, keep those within
    shell_mm of the surface (their density is proportional to surface area up
    to a relative error of about shell_mm times curvature), then project each
    onto the surface along the SDF gradient. For an exact SDF one projection
    lands on the surface to numerical precision.
    """
    he = phantom.half_extents + shell_mm
    kept = []
    total = 0
    for _ in range(max_batches):
        p = rng.uniform(-he, he, size=(batch, 3))
        d = phantom.sdf_local(p)
        sel = np.abs(d) < shell_mm
        if sel.any():
            p, d = p[sel], d[sel]
            p = p - d[:, None] * _sdf_gradient_local(phantom, p)
            kept.append(p)
            total += len(p)
        if total >= n:
            break
    if total < n:
        raise RuntimeError(f"only {total} of {n} surface samples after {max_batches} batches")
    local = np.concatenate(kept)[:n]
    return placement.to_world(local)
