# Tier 2: Medical Segmentation Decathlon, Task09 Spleen (full)

Commit 6ed708d+dirty | 41 cases | 25024 rows | wall time 62.7 min | seed 2026

Licence CC-BY-SA 4.0. Labels: expert manual segmentation of the spleen in every training case.

Reference surface: canonical reconstruction of the expert mask. Descriptive only; distances in mm; mean ± SD across cases.

Slice spacing across cases: 1.5–8 mm.

## Acquisition resolution (Exp. D)

Each level is a minimum spacing (in-plane, slice): the CT and the expert mask are resampled with Gaussian anti-aliasing before segmentation and meshing, and the reference surface stays at native resolution. Scans already coarser than a level keep their native spacing on that axis.

| resolution | minimum spacing (mm) | cases resampled | largest grid spacing (median, mm) |
|---|---|---|---|
| native | as acquired | 0/41 | 5 |
| iso1.5 | 1.5 in-plane, 1.5 slice | 41/41 | 5 |
| iso3 | 3 in-plane, 3 slice | 41/41 | 5 |
| slice5 | native in-plane, 5 slice | 14/41 | 5 |

## Segmentation (mask level, Exp. A)

Coarse masks are mapped back to the native grid (nearest neighbour) and scored against the expert mask. Source `gt` is the expert mask resampled: the overlap lost to sampling alone.

| structure | source | resolution | cases | Dice | IoU | HD95 | ASSD |
|---|---|---|---|---|---|---|---|
| spleen | gt | iso1.5 | 41 | 0.985 ± 0.003 | 0.970 ± 0.005 | 0.81 ± 0.10 | 0.12 ± 0.06 |
| spleen | gt | iso3 | 41 | 0.968 ± 0.006 | 0.938 ± 0.012 | 1.20 ± 0.35 | 0.27 ± 0.16 |
| spleen | gt | slice5 | 14 | 0.971 ± 0.011 | 0.944 ± 0.020 | 2.37 ± 0.74 | 0.53 ± 0.15 |
| spleen | ts_fast | native | 41 | 0.940 ± 0.018 | 0.887 ± 0.031 | 3.18 ± 1.43 | 0.69 ± 0.24 |
| spleen | ts_fast | iso1.5 | 41 | 0.940 ± 0.019 | 0.887 ± 0.032 | 3.07 ± 1.33 | 0.69 ± 0.26 |
| spleen | ts_fast | iso3 | 41 | 0.940 ± 0.018 | 0.887 ± 0.030 | 3.03 ± 1.14 | 0.67 ± 0.24 |
| spleen | ts_fast | slice5 | 41 | 0.935 ± 0.020 | 0.878 ± 0.034 | 3.44 ± 1.43 | 0.79 ± 0.32 |
| spleen | ts_full | native | 41 | 0.951 ± 0.021 | 0.907 ± 0.036 | 2.96 ± 2.80 | 0.58 ± 0.34 |
| spleen | ts_full | iso1.5 | 41 | 0.951 ± 0.020 | 0.907 ± 0.034 | 2.97 ± 2.44 | 0.58 ± 0.30 |
| spleen | ts_full | iso3 | 41 | 0.932 ± 0.031 | 0.875 ± 0.051 | 4.80 ± 8.34 | 0.90 ± 0.77 |
| spleen | ts_full | slice5 | 41 | 0.945 ± 0.022 | 0.896 ± 0.038 | 3.24 ± 2.77 | 0.69 ± 0.38 |

## End to end (mesh against reference surface, Exp. B and C)

Source `gt` is the expert mask itself: its error is reconstruction (and, below native, sampling) alone. Canonical on `gt` at native reproduces the reference, so it should read 0.

| structure | source | resolution | config | ASSD | HD95 | mean signed | volume error % | closed + manifold |
|---|---|---|---|---|---|---|---|---|
| spleen | gt | native | canonical | 0.000 ± 0.000 | 0.00 ± 0.00 | -0.000 ± 0.000 | -0.0 ± 0.0 | 98% |
| spleen | gt | native | production | 0.234 ± 0.056 | 0.70 ± 0.17 | -0.022 ± 0.013 | -0.3 ± 0.3 | 98% |
| spleen | gt | iso1.5 | canonical | 0.204 ± 0.017 | 0.61 ± 0.04 | -0.003 ± 0.013 | -0.1 ± 0.2 | 98% |
| spleen | gt | iso1.5 | production | 0.374 ± 0.098 | 1.10 ± 0.31 | -0.043 ± 0.025 | -0.7 ± 0.5 | 95% |
| spleen | gt | iso3 | canonical | 0.418 ± 0.030 | 1.18 ± 0.11 | -0.039 ± 0.021 | -0.6 ± 0.4 | 98% |
| spleen | gt | iso3 | production | 0.517 ± 0.126 | 1.50 ± 0.36 | -0.142 ± 0.039 | -2.0 ± 0.9 | 98% |
| spleen | gt | slice5 | canonical | 0.152 ± 0.219 | 0.50 ± 0.71 | 0.018 ± 0.060 | 0.2 ± 0.7 | 98% |
| spleen | gt | slice5 | production | 0.319 ± 0.108 | 0.99 ± 0.34 | -0.013 ± 0.059 | -0.3 ± 0.8 | 95% |
| spleen | ts_fast | native | canonical | 0.901 ± 0.197 | 3.15 ± 1.58 | 0.261 ± 0.263 | 1.8 ± 3.4 | 98% |
| spleen | ts_fast | native | production | 0.948 ± 0.218 | 3.03 ± 1.65 | 0.221 ± 0.251 | 1.4 ± 3.4 | 66% |
| spleen | ts_fast | iso1.5 | canonical | 0.905 ± 0.223 | 3.02 ± 1.65 | 0.157 ± 0.270 | 0.7 ± 3.7 | 98% |
| spleen | ts_fast | iso1.5 | production | 0.958 ± 0.255 | 3.03 ± 1.72 | 0.099 ± 0.255 | 0.2 ± 3.6 | 90% |
| spleen | ts_fast | iso3 | canonical | 0.907 ± 0.206 | 2.95 ± 1.56 | 0.180 ± 0.267 | 1.1 ± 3.5 | 98% |
| spleen | ts_fast | iso3 | production | 0.914 ± 0.247 | 2.84 ± 1.51 | 0.038 ± 0.248 | -0.5 ± 3.4 | 95% |
| spleen | ts_fast | slice5 | canonical | 0.964 ± 0.211 | 3.27 ± 1.58 | 0.224 ± 0.278 | 1.2 ± 3.6 | 98% |
| spleen | ts_fast | slice5 | production | 0.993 ± 0.217 | 3.14 ± 1.62 | 0.174 ± 0.270 | 0.8 ± 3.5 | 56% |
| spleen | ts_full | native | canonical | 0.790 ± 0.379 | 2.82 ± 2.66 | 0.153 ± 0.428 | -0.1 ± 4.0 | 98% |
| spleen | ts_full | native | production | 0.855 ± 0.344 | 2.93 ± 2.84 | 0.092 ± 0.298 | -0.4 ± 4.1 | 98% |
| spleen | ts_full | iso1.5 | canonical | 0.805 ± 0.341 | 2.83 ± 2.49 | 0.105 ± 0.354 | -0.4 ± 4.0 | 98% |
| spleen | ts_full | iso1.5 | production | 0.860 ± 0.290 | 2.98 ± 2.49 | 0.005 ± 0.253 | -1.0 ± 4.1 | 93% |
| spleen | ts_full | iso3 | canonical | 1.210 ± 1.156 | 4.85 ± 9.48 | 0.002 ± 1.944 | -5.6 ± 5.3 | 98% |
| spleen | ts_full | iso3 | production | 1.205 ± 0.825 | 4.72 ± 7.94 | -0.315 ± 1.073 | -7.2 ± 6.0 | 95% |
| spleen | ts_full | slice5 | canonical | 0.863 ± 0.378 | 2.97 ± 2.68 | 0.098 ± 0.456 | -0.8 ± 4.2 | 98% |
| spleen | ts_full | slice5 | production | 0.903 ± 0.332 | 3.04 ± 2.73 | 0.029 ± 0.326 | -1.2 ± 4.3 | 93% |

## Share of log-ASSD variance (grid configurations, preview)

Main effects (eta squared); "case" is patient-to-patient variation. Exact rows with zero error (canonical on the expert mask) are excluded. Formal attribution uses mixed-effects models and Sobol indices.

- **spleen**: source 9.4%, resolution 1.8%, blur_mm 1.8%, taubin_iter 3.0%, decimation 0.2%, case 0.3%, interactions + residual 83.5%

## Mean ASSD by setting (grid)

- **resolution**: gt: native→0.193, iso1.5→0.317, iso3→0.476, slice5→0.287; ts_fast: native→0.940, iso1.5→0.947, iso3→0.923, slice5→0.990; ts_full: native→0.845, iso1.5→0.852, iso3→1.218, slice5→0.898
- **blur_mm**: gt: 0.0→0.297, 0.5→0.313, 1.0→0.345; ts_fast: 0.0→0.949, 0.5→0.950, 1.0→0.951; ts_full: 0.0→0.952, 0.5→0.955, 1.0→0.952
- **taubin_iter**: gt: 0.0→0.242, 10.0→0.263, 30.0→0.350, 60.0→0.417; ts_fast: 0.0→0.929, 10.0→0.941, 30.0→0.963, 60.0→0.967; ts_full: 0.0→0.929, 10.0→0.942, 30.0→0.967, 60.0→0.974
- **decimation**: gt: 0.0→0.311, 0.25→0.311, 0.5→0.316, 0.75→0.335; ts_fast: 0.0→0.949, 0.25→0.949, 0.5→0.950, 0.75→0.951; ts_full: 0.0→0.951, 0.25→0.950, 0.5→0.953, 0.75→0.958

## Topology (closed and manifold, by decimation)

| source | resolution | 0.0 | 0.25 | 0.5 | 0.6 | 0.75 |
|---|---|---|---|---|---|---|
| gt | native | 98% | 96% | 91% | — | 68% |
| gt | iso1.5 | 98% | 95% | 93% | — | 75% |
| gt | iso3 | 98% | 97% | 94% | — | 79% |
| gt | slice5 | 98% | 97% | 91% | — | 67% |
| ts_fast | native | 98% | 97% | 93% | 59% | 57% |
| ts_fast | iso1.5 | 98% | 98% | 93% | 90% | 63% |
| ts_fast | iso3 | 98% | 95% | 94% | 96% | 83% |
| ts_fast | slice5 | 98% | 97% | 93% | 51% | 58% |
| ts_full | native | 98% | 96% | 89% | 95% | 58% |
| ts_full | iso1.5 | 98% | 93% | 89% | 90% | 67% |
| ts_full | iso3 | 98% | 94% | 92% | 90% | 82% |
| ts_full | slice5 | 98% | 97% | 91% | 91% | 56% |
