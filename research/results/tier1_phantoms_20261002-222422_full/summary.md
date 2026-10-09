# Tier 1 phantom experiment: full

Commit 130686f+dirty | 7760 meshes | 80 units | 5 workers | wall time 2.8 min | seed 2026

Descriptive summary only. The formal RQ1 attribution (mixed-effects model and Sobol indices) is a later step. Distances in mm; surface samples per side: 20000; NSD tolerance 1.0 mm.

## Reference floor

Canonical configuration (no blur, Lewiner marching cubes, no smoothing, no decimation): the reference-surface procedure. Cells: ASSD (HD95), mean over placements; columns are voxel spacings in mm.

| shape | 0.5 | 1 | 2 | 0.8x0.8x5 |
|---|---|---|---|---|
| capsule | 0.069 (0.17) | 0.144 (0.35) | 0.331 (0.77) | 0.634 (1.95) |
| rounded_box | 0.072 (0.17) | 0.145 (0.33) | 0.297 (0.68) | 0.525 (1.72) |
| sphere | 0.070 (0.17) | 0.147 (0.34) | 0.271 (0.75) | 0.495 (1.62) |
| torus | 0.068 (0.16) | 0.138 (0.33) | 0.278 (0.66) | 0.499 (1.61) |

## Production defaults versus canonical

Production = `build_mesh` defaults (blur 0.6 voxels, 30 Taubin iterations, 0.5 decimation). Cells: ASSD canonical -> production; signed = production mean signed distance (negative = shrinkage).

| shape | 0.5 | 1 | 2 | 0.8x0.8x5 |
|---|---|---|---|---|
| capsule | 0.069 -> 0.051 (signed -0.05) | 0.144 -> 0.188 (signed -0.18) | 0.331 -> 1.192 (signed -1.13) | 0.634 -> 0.947 (signed -0.57) |
| rounded_box | 0.072 -> 0.027 (signed -0.01) | 0.145 -> 0.074 (signed -0.03) | 0.297 -> 0.236 (signed -0.13) | 0.525 -> 0.461 (signed -0.08) |
| sphere | 0.070 -> 0.029 (signed -0.02) | 0.147 -> 0.088 (signed -0.08) | 0.271 -> 0.252 (signed -0.25) | 0.495 -> 0.390 (signed -0.11) |
| torus | 0.068 -> 0.028 (signed -0.01) | 0.138 -> 0.076 (signed -0.05) | 0.278 -> 0.251 (signed -0.23) | 0.499 -> 0.433 (signed -0.13) |

## Share of ASSD variance by source (preview)

Main effects are eta squared; replication is placement-to-placement variation within identical settings; interactions are the rest. ASSD scales roughly multiplicatively with voxel size, so the log column is the fairer view of interactions. Shares depend on the factor ranges chosen, not on the shapes alone. Exact only for a balanced grid; this grid is balanced.

| source | raw ASSD | log ASSD |
|---|---|---|
| spacing | 38.8% | 68.6% |
| shape | 21.3% | 13.5% |
| blur_mm | 0.8% | 3.3% |
| isosurface | 0.0% | 0.0% |
| taubin_iter | 1.2% | 0.5% |
| decimation | 0.2% | 0.3% |
| replication | 2.2% | 1.1% |
| interactions | 35.5% | 12.7% |

## Mean ASSD by factor level (grid configurations)

- **blur_mm**: 0.0 -> 0.263, 0.5 -> 0.260, 1.0 -> 0.314
- **isosurface**: lewiner -> 0.278, lorensen -> 0.280
- **taubin_iter**: 0 -> 0.269, 10 -> 0.256, 30 -> 0.260, 60 -> 0.331
- **decimation**: 0.0 -> 0.267, 0.25 -> 0.271, 0.5 -> 0.279, 0.75 -> 0.299

## Topology

Share of meshes that are closed, a single component and of the expected genus.

| isosurface | 0.5 | 1 | 2 | 0.8x0.8x5 |
|---|---|---|---|---|
| lewiner | 99% | 97% | 94% | 82% |
| lorensen | 99% | 98% | 95% | 81% |

Failure modes among 544 meshes (one mesh can have several): 301 open (boundary edges), 275 split into several components, 0 closed but wrong genus. Open meshes with no decimation: 0.

Every configuration produced a surface.

## Timing (planning only, not a cost benchmark)

Mean per mesh: isosurface 0.001 s, smoothing and decimation 0.016 s, metrics 0.088 s.
