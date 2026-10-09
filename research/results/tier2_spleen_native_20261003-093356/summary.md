# Tier 2: Medical Segmentation Decathlon, Task09 Spleen (full)

Commit abb82ed+dirty | 41 cases | 4059 rows | wall time 19.7 min | seed 2026

Licence CC-BY-SA 4.0. Labels: expert manual segmentation of the spleen in every training case.

Reference surface: canonical reconstruction of the expert mask. Descriptive only; distances in mm; mean ± SD across cases.

Slice spacing across cases: 1.5–8 mm.

## Segmentation (mask level, Exp. A)

| structure | engine | cases | Dice | IoU | HD95 | ASSD |
|---|---|---|---|---|---|---|
| spleen | ts_fast | 41 | 0.940 ± 0.018 | 0.887 ± 0.031 | 3.18 ± 1.43 | 0.69 ± 0.24 |

## End to end (mesh against reference surface, Exp. B and C)

Source `gt` is the expert mask itself: its error is reconstruction alone. Canonical on `gt` reproduces the reference, so it should read 0.

| structure | source | config | ASSD | HD95 | mean signed | volume error % | closed + manifold |
|---|---|---|---|---|---|---|---|
| spleen | gt | canonical | 0.000 ± 0.000 | 0.00 ± 0.00 | -0.000 ± 0.000 | -0.0 ± 0.0 | 98% |
| spleen | gt | production | 0.234 ± 0.056 | 0.70 ± 0.17 | -0.022 ± 0.013 | -0.3 ± 0.3 | 98% |
| spleen | ts_fast | canonical | 0.901 ± 0.197 | 3.15 ± 1.58 | 0.261 ± 0.263 | 1.8 ± 3.4 | 98% |
| spleen | ts_fast | production | 0.948 ± 0.217 | 3.03 ± 1.65 | 0.221 ± 0.249 | 1.4 ± 3.4 | 66% |

## Share of log-ASSD variance (grid configurations, preview)

Main effects (eta squared); "case" is patient-to-patient variation. Exact rows with zero error (canonical on the expert mask) are excluded. Formal attribution uses mixed-effects models and Sobol indices.

- **spleen**: source 10.4%, blur_mm 6.6%, taubin_iter 10.2%, decimation 0.7%, case 0.1%, interactions + residual 72.0%

## Mean ASSD by setting (grid)

- **blur_mm**: gt: 0.0→0.148, 0.5→0.186, 1.0→0.244; ts_fast: 0.0→0.937, 0.5→0.941, 1.0→0.942
- **taubin_iter**: gt: 0.0→0.098, 10.0→0.126, 30.0→0.234, 60.0→0.313; ts_fast: 0.0→0.916, 10.0→0.922, 30.0→0.954, 60.0→0.968
- **decimation**: gt: 0.0→0.186, 0.25→0.187, 0.5→0.191, 0.75→0.208; ts_fast: 0.0→0.941, 0.25→0.941, 0.5→0.940, 0.75→0.938

## Topology (closed and manifold, by decimation)

| source | 0.0 | 0.25 | 0.5 | 0.6 | 0.75 |
|---|---|---|---|---|---|
| gt | 98% | 96% | 91% | — | 68% |
| ts_fast | 98% | 97% | 93% | 66% | 57% |
