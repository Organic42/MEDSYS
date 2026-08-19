"""
Unit tests for the VR segmentation pipeline.
Uses synthetic numpy arrays — no DICOM files required.
"""

import os
import sys
import numpy as np
import pytest
from scipy import ndimage
from skimage import measure, morphology

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate import compute_metrics  # noqa: E402


# ── helpers mirrored from pipeline ──────────────────────────────────────────

def preprocess_mri(volume, low_p=1, high_p=99):
    p_low = np.percentile(volume, low_p)
    p_high = np.percentile(volume, high_p)
    clipped = np.clip(volume, p_low, p_high)
    return (clipped - p_low) / (p_high - p_low + 1e-8)


def postprocess_slice(mask_2d):
    m = morphology.closing(mask_2d, morphology.disk(4))
    m = ndimage.binary_fill_holes(m)
    m = morphology.opening(m, morphology.disk(2))
    labeled = measure.label(m)
    if labeled.max() == 0:
        return np.zeros_like(mask_2d, dtype=bool)
    regions = measure.regionprops(labeled)
    return labeled == max(regions, key=lambda r: r.area).label


def synthetic_volume(shape=(32, 64, 64), seed=42):
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, dtype=np.float32)
    cx, cy = shape[1] // 2, shape[2] // 2
    for i in range(shape[0]):
        yy, xx = np.ogrid[:shape[1], :shape[2]]
        brain = ((xx - cx) / 20) ** 2 + ((yy - cy) / 18) ** 2 <= 1
        vol[i][brain] = 500 + rng.normal(0, 30, brain.sum())
        vol[i][~brain] = rng.normal(0, 10, (~brain).sum())
    return vol.clip(0, None)


# ── tests ────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_output_range(self):
        vol = synthetic_volume()
        norm = preprocess_mri(vol)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_shape_preserved(self):
        vol = synthetic_volume((10, 32, 32))
        assert preprocess_mri(vol).shape == vol.shape

    def test_uniform_volume_does_not_crash(self):
        vol = np.ones((8, 16, 16), dtype=np.float32) * 500
        norm = preprocess_mri(vol)
        assert norm.shape == vol.shape

    def test_high_contrast_spreads_range(self):
        vol = np.zeros((8, 32, 32), dtype=np.float32)
        vol[:, 8:24, 8:24] = 1000.0
        norm = preprocess_mri(vol)
        assert norm.max() > 0.9


class TestPostprocessing:
    def _circle_mask(self, size=64, radius=20):
        yy, xx = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2

    def test_fills_holes(self):
        mask = self._circle_mask()
        mask[30:34, 30:34] = False          # punch a hole
        result = postprocess_slice(mask)
        assert result[32, 32]               # hole should be filled

    def test_removes_small_noise(self):
        mask = self._circle_mask()
        mask[2, 2] = True                   # isolated noise pixel
        result = postprocess_slice(mask)
        assert not result[2, 2]             # noise removed

    def test_keeps_largest_component(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True           # large region
        mask[50:54, 50:54] = True           # small region
        result = postprocess_slice(mask)
        assert result[20, 20]               # large region kept
        assert not result[52, 52]           # small region removed

    def test_empty_mask_returns_zeros(self):
        mask = np.zeros((64, 64), dtype=bool)
        result = postprocess_slice(mask)
        assert result.sum() == 0


class TestVolumeIntegrity:
    def test_brain_voxel_count_in_range(self):
        vol = synthetic_volume()
        norm = preprocess_mri(vol)
        brain_mask = norm > 0.4
        voxels = brain_mask.sum()
        total = np.prod(vol.shape)
        ratio = voxels / total
        assert 0.01 < ratio < 0.90, f"Unexpected brain coverage ratio: {ratio:.3f}"

    def test_normalization_is_deterministic(self):
        vol = synthetic_volume(seed=7)
        assert np.allclose(preprocess_mri(vol), preprocess_mri(vol))

    def test_3d_largest_cc_removes_fragments(self):
        vol = np.zeros((20, 64, 64), dtype=bool)
        cx, cy = 32, 32
        yy, xx = np.ogrid[:64, :64]
        brain = ((xx - cx) ** 2 + (yy - cy) ** 2) <= 15 ** 2
        vol[5:15] = brain                   # main brain region
        vol[0, 0, 0] = True                 # isolated fragment
        labeled = measure.label(vol)
        regions = measure.regionprops(labeled)
        main = labeled == max(regions, key=lambda r: r.area).label
        assert not main[0, 0, 0]            # fragment removed
        assert main[10, 32, 32]             # brain retained


class TestValidationMetrics:
    SP = (1.0, 1.0, 1.0)

    def _ball(self, shape=(40, 40, 40), c=(20, 20, 20), r=10):
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        return ((zz-c[0])**2 + (yy-c[1])**2 + (xx-c[2])**2) <= r**2

    def test_identical_masks_dice_one(self):
        m = self._ball()
        r = compute_metrics(m, m, self.SP)
        assert r['dice'] == 1.0 and r['iou'] == 1.0
        assert r['hd95_mm'] == 0.0 and r['assd_mm'] == 0.0

    def test_disjoint_masks_dice_zero(self):
        a = self._ball(c=(10, 10, 10), r=5)
        b = self._ball(c=(30, 30, 30), r=5)
        r = compute_metrics(a, b, self.SP)
        assert r['dice'] == 0.0 and r['iou'] == 0.0
        assert r['hd95_mm'] > 0

    def test_partial_overlap_between_zero_and_one(self):
        a = self._ball(c=(20, 20, 18), r=10)
        b = self._ball(c=(20, 20, 22), r=10)
        r = compute_metrics(a, b, self.SP)
        assert 0.0 < r['dice'] < 1.0
        assert r['assd_mm'] > 0

    def test_empty_vs_nonempty_is_safe(self):
        empty = np.zeros((20, 20, 20), dtype=bool)
        full = self._ball((20, 20, 20), (10, 10, 10), 5)
        r = compute_metrics(empty, full, self.SP)
        assert r['dice'] == 0.0
        assert r['hd95_mm'] is None      # undefined when one mask is empty

    def test_spacing_scales_distance(self):
        a = self._ball(c=(20, 20, 18), r=10)
        b = self._ball(c=(20, 20, 22), r=10)
        r1 = compute_metrics(a, b, (1.0, 1.0, 1.0))
        r2 = compute_metrics(a, b, (2.0, 2.0, 2.0))
        assert r2['assd_mm'] > r1['assd_mm']   # larger voxels -> larger mm distances
