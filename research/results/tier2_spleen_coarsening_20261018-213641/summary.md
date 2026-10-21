# Tier 2: Medical Segmentation Decathlon, Task09 Spleen (sensitivity)

Commit 6ed708d+dirty | 41 cases | 1258 rows | wall time 30.9 min | seed 2026

Licence CC-BY-SA 4.0. Labels: expert manual segmentation of the spleen in every training case.

Reference surface: canonical reconstruction of the expert mask. Descriptive only; distances in mm; mean ± SD across cases.

Slice spacing across cases: 1.5–8 mm.

## Acquisition resolution (Exp. D)

Each level is a minimum spacing (in-plane, slice): the CT and the expert mask are resampled with Gaussian anti-aliasing before segmentation and meshing, and the reference surface stays at native resolution. Scans already coarser than a level keep their native spacing on that axis.

| resolution | minimum spacing (mm) | cases resampled | largest grid spacing (median, mm) |
|---|---|---|---|
| iso3~box | 3 in-plane, 3 slice | 41/41 | 5 |
| slice5~box | native in-plane, 5 slice | 14/41 | 5 |
| iso3~noise | 3 in-plane, 3 slice | 41/41 | 5 |
| slice5~noise | native in-plane, 5 slice | 14/41 | 5 |

## Segmentation (mask level, Exp. A)

Coarse masks are mapped back to the native grid (nearest neighbour) and scored against the expert mask. Source `gt` is the expert mask resampled: the overlap lost to sampling alone.

| structure | source | resolution | cases | Dice | IoU | HD95 | ASSD |
|---|---|---|---|---|---|---|---|
| spleen | gt | iso3~box | 41 | 0.968 ± 0.006 | 0.938 ± 0.012 | 1.20 ± 0.35 | 0.27 ± 0.17 |
| spleen | gt | slice5~box | 14 | 0.971 ± 0.011 | 0.944 ± 0.020 | 2.37 ± 0.73 | 0.53 ± 0.15 |
| spleen | gt | iso3~noise | 41 | 0.968 ± 0.006 | 0.938 ± 0.012 | 1.20 ± 0.35 | 0.27 ± 0.16 |
| spleen | gt | slice5~noise | 14 | 0.971 ± 0.011 | 0.944 ± 0.020 | 2.37 ± 0.74 | 0.53 ± 0.15 |
| spleen | ts_fast | iso3~box | 41 | 0.940 ± 0.018 | 0.887 ± 0.031 | 3.06 ± 1.16 | 0.68 ± 0.24 |
| spleen | ts_fast | slice5~box | 41 | 0.934 ± 0.021 | 0.877 ± 0.035 | 3.45 ± 1.43 | 0.80 ± 0.33 |
| spleen | ts_fast | iso3~noise | 41 | 0.939 ± 0.019 | 0.885 ± 0.032 | 3.12 ± 1.33 | 0.69 ± 0.24 |
| spleen | ts_fast | slice5~noise | 41 | 0.934 ± 0.020 | 0.878 ± 0.034 | 3.42 ± 1.37 | 0.79 ± 0.32 |

## End to end (mesh against reference surface, Exp. B and C)

Source `gt` is the expert mask itself: its error is reconstruction (and, below native, sampling) alone. Canonical on `gt` at native reproduces the reference, so it should read 0.

| structure | source | resolution | config | ASSD | HD95 | mean signed | volume error % | closed + manifold |
|---|---|---|---|---|---|---|---|---|
| spleen | gt | iso3~box | canonical | 0.417 ± 0.030 | 1.18 ± 0.11 | -0.036 ± 0.021 | -0.6 ± 0.4 | 98% |
| spleen | gt | iso3~box | production | 0.518 ± 0.124 | 1.50 ± 0.36 | -0.141 ± 0.038 | -2.0 ± 0.9 | 98% |
| spleen | gt | slice5~box | canonical | 0.152 ± 0.219 | 0.49 ± 0.71 | 0.017 ± 0.057 | 0.1 ± 0.7 | 98% |
| spleen | gt | slice5~box | production | 0.320 ± 0.110 | 0.99 ± 0.34 | -0.013 ± 0.059 | -0.3 ± 0.8 | 95% |
| spleen | gt | iso3~noise | canonical | 0.417 ± 0.030 | 1.18 ± 0.11 | -0.037 ± 0.021 | -0.6 ± 0.4 | 98% |
| spleen | gt | iso3~noise | production | 0.517 ± 0.126 | 1.50 ± 0.36 | -0.142 ± 0.039 | -2.0 ± 0.9 | 98% |
| spleen | gt | slice5~noise | canonical | 0.152 ± 0.218 | 0.50 ± 0.71 | 0.018 ± 0.062 | 0.2 ± 0.7 | 98% |
| spleen | gt | slice5~noise | production | 0.319 ± 0.108 | 0.99 ± 0.34 | -0.013 ± 0.060 | -0.3 ± 0.8 | 95% |
| spleen | ts_fast | iso3~box | canonical | 0.914 ± 0.208 | 3.00 ± 1.59 | 0.172 ± 0.270 | 1.0 ± 3.5 | 98% |
| spleen | ts_fast | iso3~box | production | 0.919 ± 0.249 | 2.86 ± 1.55 | 0.032 ± 0.249 | -0.6 ± 3.5 | 95% |
| spleen | ts_fast | slice5~box | canonical | 0.970 ± 0.214 | 3.28 ± 1.51 | 0.206 ± 0.280 | 1.0 ± 3.7 | 98% |
| spleen | ts_fast | slice5~box | production | 1.000 ± 0.225 | 3.19 ± 1.70 | 0.161 ± 0.277 | 0.6 ± 3.6 | 59% |
| spleen | ts_fast | iso3~noise | canonical | 0.926 ± 0.222 | 2.99 ± 1.75 | 0.212 ± 0.260 | 1.5 ± 3.5 | 98% |
| spleen | ts_fast | iso3~noise | production | 0.934 ± 0.273 | 2.93 ± 1.81 | 0.078 ± 0.250 | -0.1 ± 3.5 | 95% |
| spleen | ts_fast | slice5~noise | canonical | 0.965 ± 0.209 | 3.27 ± 1.60 | 0.234 ± 0.273 | 1.3 ± 3.5 | 98% |
| spleen | ts_fast | slice5~noise | production | 0.993 ± 0.220 | 3.16 ± 1.74 | 0.186 ± 0.274 | 0.9 ± 3.5 | 56% |

## Topology (closed and manifold, by decimation)

| source | resolution | 0.0 | 0.5 | 0.6 |
|---|---|---|---|---|
| gt | iso3~box | 98% | 98% | — |
| gt | slice5~box | 98% | 95% | — |
| gt | iso3~noise | 98% | 96% | — |
| gt | slice5~noise | 98% | 95% | — |
| ts_fast | iso3~box | 98% | — | 96% |
| ts_fast | slice5~box | 98% | — | 49% |
| ts_fast | iso3~noise | 98% | — | 93% |
| ts_fast | slice5~noise | 98% | — | 55% |
