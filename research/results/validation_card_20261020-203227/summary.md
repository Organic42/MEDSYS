# Validation test 1: the in-app accuracy estimate

Run 20261020-203227, commit 6ed708d (+ uncommitted changes). Application code: `segmentation_pipeline.build_mesh` with the accuracy estimate on, applied to the cached TotalSegmentator masks of 82 expert-labelled scans (msd_pancreas, msd_spleen); 590 delivered meshes. True error = ASSD of the delivered mesh against the expert surface (20,000 points per side). Coverage is the share of scans inside a stated range.

## 1. Resolution limit

**On real anatomy.** The expert mask coarsened to each grid, against its native surface; predicted 0.177 x Delta (Delta = the sampling the coarser grid adds). The spleen is the fit; the pancreas never entered it.

| Organ | Grids | Mean abs. error (mm) | Inside +/-0.111 mm | Inside 0.13-0.21 x Delta |
|---|---|---|---|---|
| pancreas (held out) | 117 | 0.029 | 98% | 92% |
| spleen (fit) | 96 | 0.037 | 95% | 81% |

**On exact surfaces (Tier 1 phantoms).** The card's range, 0.13-0.21 x RMS spacing, held for 88% of 80 canonical meshes; by shape: capsule 80%, rounded_box 70%, sphere 100%, torus 100%. As first shipped (0.14-0.21) it held 60%: isotropic grids sit right at 0.14 x spacing, half of them just below. (The range comes from these data, so this is consistency, not prediction.)

**"Expect at least the resolution limit."** The delivered mesh's true error was at least the card's resolution limit for 100% of 590 meshes (lowest ratio 1.04x).

## 2. Smoothing and simplification

The card measures each mesh's shift from its own raw surface: median 0.344 mm (range 0.168-1.194). Smoothing changed the true error by +0.075 mm (median; range -2.310 to +1.342); the change stayed within the card's shift for 97% of meshes. The card's signed shift (shrinkage) matched the change in signed distance to the expert surface within 0.022 mm (median; 95th percentile 0.120).

## 3. Segmentation (leave one scan out)

For each delivered mesh, the card's reference (canonical engine ASSD of the *other* 40 scans) is rebuilt and the mesh's true error is checked against it. The card shows the median and the middle half, so about 50% is the honest target there, and about 90% for the 5th-95th percentile range. **v1** is the card as first shipped: the native reference, plus the paper's thick-slice change on 5 mm grids. **v2** matches the reference to the scan's grid (`accuracy.reference_level`) and adds the 90% range.

| Organ | Mode | Grid | Scans | v1 middle half | v1 90% range | v2 middle half | v2 90% range | v2 median abs. error (mm) |
|---|---|---|---|---|---|---|---|---|
| pancreas | fast | native | 41 | 46% | 88% | 46% | 88% | 0.401 |
| pancreas | fast | iso1.5 | 41 | 49% | 85% | 46% | 85% | 0.474 |
| pancreas | fast | iso3 | 41 | 49% | 83% | 49% | 83% | 0.508 |
| pancreas | fast | slice5 | 35 | 54% | 86% | 51% | 83% | 0.366 |
| pancreas | full | native | 41 | 49% | 85% | 49% | 85% | 0.598 |
| pancreas | full | iso1.5 | 41 | 51% | 88% | 51% | 85% | 0.640 |
| pancreas | full | iso3 | 41 | 49% | 80% | 49% | 83% | 0.953 |
| pancreas | full | slice5 | 35 | 46% | 86% | 51% | 89% | 0.578 |
| spleen | fast | native | 41 | 49% | 90% | 56% | 90% | 0.069 |
| spleen | fast | iso1.5 | 41 | 44% | 78% | 44% | 78% | 0.069 |
| spleen | fast | iso3 | 41 | 37% | 71% | 46% | 76% | 0.102 |
| spleen | fast | slice5 | 14 | 43% | 57% | 50% | 86% | 0.096 |
| spleen | full | native | 41 | 61% | 93% | 63% | 93% | 0.079 |
| spleen | full | iso1.5 | 41 | 54% | 93% | 54% | 93% | 0.084 |
| spleen | full | iso3 | 41 | 15% | 56% | 46% | 90% | 0.208 |
| spleen | full | slice5 | 14 | 36% | 79% | 50% | 93% | 0.089 |

All 590 meshes: v1 middle half 46% (target 50%), 90% range 82% (target 90%), 35% above the middle half; v2 50%, 86%, 31% above. Worst cell: v1 15% (spleen, full, iso3); v2 44% (spleen, fast, iso1.5). Against the segmentation alone (canonical meshes, the line's exact claim), v2 covers 48% and 86%.

**Why other organs say "not validated".** The spleen's 90% range (0.60-1.16 mm) would have covered only 5% of pancreas meshes.

## 4. Watertight flag

The card's flag agreed with VTK's independent manifold check for 100% of 590 meshes (480 watertight).

## 5. Same mask, same mesh

Meshing each mask a second time gave an identical mesh (vertices and faces) and identical card values for 590 of 590 meshes.

## Timing

Median build time with the estimate: 0.30 s per mesh.
