# Tier 2: Medical Segmentation Decathlon, Task07 Pancreas (full)

Commit 6ed708d+dirty | 41 cases | 25045 rows | wall time 46.9 min | seed 2026

Licence CC-BY-SA 4.0. Labels: expert manual segmentation of the pancreas and of pancreatic tumour in every training case; the two labels are merged into the organ.

Reference surface: canonical reconstruction of the expert mask. Descriptive only; distances in mm; mean ± SD across cases.

Slice spacing across cases: 1.25–5 mm.

## Acquisition resolution (Exp. D)

Each level is a minimum spacing (in-plane, slice): the CT and the expert mask are resampled with Gaussian anti-aliasing before segmentation and meshing, and the reference surface stays at native resolution. Scans already coarser than a level keep their native spacing on that axis.

| resolution | minimum spacing (mm) | cases resampled | largest grid spacing (median, mm) |
|---|---|---|---|
| native | as acquired | 0/41 | 2.5 |
| iso1.5 | 1.5 in-plane, 1.5 slice | 41/41 | 2.5 |
| iso3 | 3 in-plane, 3 slice | 41/41 | 3 |
| slice5 | native in-plane, 5 slice | 35/41 | 5 |

## Segmentation (mask level, Exp. A)

Coarse masks are mapped back to the native grid (nearest neighbour) and scored against the expert mask. Source `gt` is the expert mask resampled: the overlap lost to sampling alone.

| structure | source | resolution | cases | Dice | IoU | HD95 | ASSD |
|---|---|---|---|---|---|---|---|
| pancreas | gt | iso1.5 | 41 | 0.973 ± 0.007 | 0.947 ± 0.013 | 0.82 ± 0.11 | 0.15 ± 0.05 |
| pancreas | gt | iso3 | 41 | 0.936 ± 0.017 | 0.881 ± 0.030 | 1.72 ± 0.47 | 0.42 ± 0.13 |
| pancreas | gt | slice5 | 35 | 0.942 ± 0.014 | 0.891 ± 0.024 | 2.46 ± 0.36 | 0.63 ± 0.12 |
| pancreas | ts_fast | native | 41 | 0.753 ± 0.121 | 0.615 ± 0.126 | 11.44 ± 16.60 | 3.16 ± 6.68 |
| pancreas | ts_fast | iso1.5 | 41 | 0.751 ± 0.121 | 0.613 ± 0.126 | 11.32 ± 16.69 | 3.19 ± 6.76 |
| pancreas | ts_fast | iso3 | 41 | 0.753 ± 0.123 | 0.616 ± 0.128 | 11.46 ± 16.78 | 3.20 ± 6.79 |
| pancreas | ts_fast | slice5 | 41 | 0.732 ± 0.119 | 0.589 ± 0.122 | 11.75 ± 16.49 | 3.35 ± 6.64 |
| pancreas | ts_full | native | 41 | 0.738 ± 0.143 | 0.602 ± 0.154 | 13.89 ± 18.69 | 3.60 ± 7.68 |
| pancreas | ts_full | iso1.5 | 41 | 0.737 ± 0.144 | 0.600 ± 0.154 | 13.50 ± 18.72 | 3.64 ± 8.13 |
| pancreas | ts_full | iso3 | 41 | 0.698 ± 0.146 | 0.553 ± 0.150 | 15.25 ± 19.07 | 4.00 ± 8.16 |
| pancreas | ts_full | slice5 | 41 | 0.710 ± 0.141 | 0.565 ± 0.147 | 14.80 ± 18.52 | 3.86 ± 7.64 |

## End to end (mesh against reference surface, Exp. B and C)

Source `gt` is the expert mask itself: its error is reconstruction (and, below native, sampling) alone. Canonical on `gt` at native reproduces the reference, so it should read 0.

| structure | source | resolution | config | ASSD | HD95 | mean signed | volume error % | closed + manifold |
|---|---|---|---|---|---|---|---|---|
| pancreas | gt | native | canonical | 0.000 ± 0.000 | 0.00 ± 0.00 | -0.000 ± 0.000 | 0.0 ± 0.0 | 100% |
| pancreas | gt | native | production | 0.189 ± 0.049 | 0.57 ± 0.15 | -0.026 ± 0.011 | -0.7 ± 0.3 | 98% |
| pancreas | gt | iso1.5 | canonical | 0.199 ± 0.012 | 0.58 ± 0.03 | -0.013 ± 0.011 | -0.4 ± 0.3 | 100% |
| pancreas | gt | iso1.5 | production | 0.293 ± 0.072 | 0.83 ± 0.22 | -0.070 ± 0.020 | -1.7 ± 0.8 | 95% |
| pancreas | gt | iso3 | canonical | 0.456 ± 0.036 | 1.19 ± 0.07 | -0.077 ± 0.041 | -2.2 ± 1.3 | 100% |
| pancreas | gt | iso3 | production | 0.535 ± 0.073 | 1.53 ± 0.22 | -0.249 ± 0.073 | -6.2 ± 3.0 | 100% |
| pancreas | gt | slice5 | canonical | 0.377 ± 0.166 | 1.13 ± 0.52 | 0.020 ± 0.048 | -0.1 ± 1.2 | 100% |
| pancreas | gt | slice5 | production | 0.403 ± 0.085 | 1.28 ± 0.26 | -0.048 ± 0.055 | -1.5 ± 1.5 | 95% |
| pancreas | ts_fast | native | canonical | 2.722 ± 3.888 | 11.24 ± 16.49 | -0.143 ± 1.453 | -13.0 ± 29.3 | 100% |
| pancreas | ts_fast | native | production | 2.836 ± 3.944 | 11.52 ± 16.66 | -0.193 ± 1.451 | -13.7 ± 29.1 | 37% |
| pancreas | ts_fast | iso1.5 | canonical | 2.768 ± 3.958 | 11.18 ± 16.63 | -0.185 ± 1.520 | -13.3 ± 30.0 | 98% |
| pancreas | ts_fast | iso1.5 | production | 2.911 ± 4.023 | 11.63 ± 16.78 | -0.269 ± 1.503 | -14.6 ± 29.5 | 85% |
| pancreas | ts_fast | iso3 | canonical | 2.842 ± 4.012 | 11.47 ± 16.84 | -0.248 ± 1.473 | -14.3 ± 29.1 | 100% |
| pancreas | ts_fast | iso3 | production | 2.977 ± 4.102 | 11.80 ± 17.00 | -0.504 ± 1.408 | -18.9 ± 27.4 | 93% |
| pancreas | ts_fast | slice5 | canonical | 2.843 ± 3.866 | 11.32 ± 16.45 | -0.201 ± 1.474 | -14.5 ± 29.6 | 100% |
| pancreas | ts_fast | slice5 | production | 2.954 ± 3.955 | 11.56 ± 16.58 | -0.283 ± 1.460 | -16.0 ± 29.0 | 59% |
| pancreas | ts_full | native | canonical | 3.021 ± 4.350 | 13.69 ± 18.65 | -0.682 ± 1.401 | -24.7 ± 27.7 | 100% |
| pancreas | ts_full | native | production | 3.163 ± 4.426 | 14.03 ± 18.81 | -0.724 ± 1.381 | -25.3 ± 27.6 | 76% |
| pancreas | ts_full | iso1.5 | canonical | 3.092 ± 4.545 | 13.49 ± 18.71 | -0.750 ± 1.372 | -25.6 ± 26.9 | 100% |
| pancreas | ts_full | iso1.5 | production | 3.222 ± 4.616 | 13.89 ± 18.93 | -0.834 ± 1.349 | -27.1 ± 26.5 | 93% |
| pancreas | ts_full | iso3 | canonical | 3.507 ± 4.631 | 15.51 ± 19.12 | -0.944 ± 1.348 | -31.9 ± 25.7 | 100% |
| pancreas | ts_full | iso3 | production | 3.796 ± 4.837 | 16.61 ± 19.56 | -1.217 ± 1.287 | -36.3 ± 24.3 | 90% |
| pancreas | ts_full | slice5 | canonical | 3.225 ± 4.341 | 14.51 ± 18.56 | -0.725 ± 1.426 | -27.0 ± 27.8 | 100% |
| pancreas | ts_full | slice5 | production | 3.371 ± 4.384 | 15.07 ± 18.76 | -0.804 ± 1.412 | -28.3 ± 27.4 | 78% |

## Share of log-ASSD variance (grid configurations, preview)

Main effects (eta squared); "case" is patient-to-patient variation. Exact rows with zero error (canonical on the expert mask) are excluded. Formal attribution uses mixed-effects models and Sobol indices.

- **pancreas**: source 23.1%, resolution 2.1%, blur_mm 0.8%, taubin_iter 1.4%, decimation 0.2%, case 1.9%, interactions + residual 70.3%

## Mean ASSD by setting (grid)

- **resolution**: gt: native→0.154, iso1.5→0.263, iso3→0.494, slice5→0.383; ts_fast: native→2.820, iso1.5→2.858, iso3→2.905, slice5→2.931; ts_full: native→3.136, iso1.5→3.175, iso3→3.613, slice5→3.334
- **blur_mm**: gt: 0.0→0.315, 0.5→0.320, 1.0→0.336; ts_fast: 0.0→2.861, 0.5→2.871, 1.0→2.903; ts_full: 0.0→3.291, 0.5→3.306, 1.0→3.346
- **taubin_iter**: gt: 0.0→0.285, 10.0→0.288, 30.0→0.331, 60.0→0.389; ts_fast: 0.0→2.831, 10.0→2.847, 30.0→2.901, 60.0→2.934; ts_full: 0.0→3.258, 10.0→3.279, 30.0→3.342, 60.0→3.378
- **decimation**: gt: 0.0→0.311, 0.25→0.313, 0.5→0.321, 0.75→0.348; ts_fast: 0.0→2.877, 0.25→2.875, 0.5→2.876, 0.75→2.886; ts_full: 0.0→3.310, 0.25→3.309, 0.5→3.311, 0.75→3.327

## Topology (closed and manifold, by decimation)

| source | resolution | 0.0 | 0.25 | 0.5 | 0.6 | 0.75 |
|---|---|---|---|---|---|---|
| gt | native | 100% | 96% | 91% | — | 58% |
| gt | iso1.5 | 100% | 98% | 93% | — | 71% |
| gt | iso3 | 100% | 98% | 95% | — | 85% |
| gt | slice5 | 100% | 99% | 94% | — | 65% |
| ts_fast | native | 100% | 98% | 86% | 51% | 38% |
| ts_fast | iso1.5 | 99% | 95% | 86% | 82% | 51% |
| ts_fast | iso3 | 100% | 96% | 89% | 91% | 63% |
| ts_fast | slice5 | 100% | 95% | 89% | 56% | 50% |
| ts_full | native | 100% | 94% | 75% | 73% | 38% |
| ts_full | iso1.5 | 100% | 94% | 83% | 87% | 54% |
| ts_full | iso3 | 100% | 96% | 90% | 85% | 64% |
| ts_full | slice5 | 100% | 96% | 82% | 74% | 39% |
