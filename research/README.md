# Research experiments

Code for the MEDSYS research proposal (risk-aware DICOM-to-3D reconstruction,
`docs/gap-analysis-two-pager.tex`). Nothing here is part of the served app.

## Tier 1: digital phantoms (`phantom_experiment.py`)

Measures reconstruction error against shapes whose true surface is known
exactly, so no reference mesh is involved:

| Phantom | Tests |
|---|---|
| sphere (r 15 mm) | smooth, convex surface |
| capsule (r 2 mm, 44 mm long) | thin, vessel-like structure |
| torus (18 / 6 mm) | genus 1: topology preservation |
| rounded box (28 × 20 × 12 mm, 2 mm rounding) | flat faces and tight edges |

Each phantom is placed with a random rotation and sub-voxel offset, voxelised
at 0.5, 1, 2 and 0.8 × 0.8 × 5 mm, and reconstructed through the same code
production uses (`reconstruction.py`) under:

- the research grid: blur 0 / 0.5 / 1 **mm** × Lewiner or Lorensen marching
  cubes × 0 / 10 / 30 / 60 Taubin iterations × 0 / 0.25 / 0.5 / 0.75 decimation;
- the **canonical** configuration (no blur, Lewiner, no smoothing, no
  decimation). It is the reference-surface procedure, so its error is the floor
  below which Tier 2 differences are not interpretable;
- the **production** defaults of `build_mesh` (blur 0.6 **voxels**, 30 Taubin
  iterations, 0.5 decimation).

```bash
python -m research.phantom_experiment --preset smoke   # seconds, 2 meshes
python -m research.phantom_experiment --preset pilot   # 1,552 meshes, ~1 min
python -m research.phantom_experiment --preset full    # 7,760 meshes, ~4 min
python -m research.phantom_experiment --resummarise output/research/phantoms/<run>
```

Each run writes `results.csv` (one row per mesh), `run.json` (commit, library
versions, grid, seed) and `summary.md` to `output/research/phantoms/<run>/`.

### Metrics

Symmetric and area-weighted. Mesh-to-truth distances use the phantom's exact
signed distance function; truth-to-mesh distances are exact point-to-triangle
distances (VTK). Reported per mesh: ASSD, HD95 (max of directed 95th
percentiles), Hausdorff, surface Dice at 1 mm, mean signed distance (negative =
shrinkage), volume and area error against closed-form values, and topology
(components, boundary and non-manifold edges, Euler number, genus).

### Reading the summary

It is descriptive. The formal RQ1 attribution (mixed-effects model and Sobol
indices) is a later step. Variance shares depend on the factor ranges chosen,
and timings are for planning only, not a cost benchmark.

## Tier 2: real anatomy (`tier2.py`)

Measures error on real CT against expert annotations. For each case and
structure, the expert mask becomes the **reference surface** through the
canonical procedure, the one Tier 1 measured. Test meshes come from

- the **expert mask itself** under every reconstruction setting, which
  isolates reconstruction error on real anatomy (Exp. B), and
- each **segmentation engine**'s mask under the same settings, which gives the
  end-to-end error (Exp. C). Engine masks are also scored at mask level with
  Dice, IoU, HD95 and ASSD (Exp. A).

The grid is blur (0 / 0.5 / 1 mm) × Taubin (0 / 10 / 30 / 60) × decimation
(0 / 0.25 / 0.5 / 0.75). The isosurface variant is fixed because it explained
0.0% of variance in Tier 1. "Production" means the settings the app actually
uses for that engine and structure.

**Acquisition resolution** is the other whole-plot factor besides the engine
(Exp. D). Spacing explained 68.6% of variance in Tier 1, so Tier 2 asks the
same question on real scans. Each case is resampled to coarser grids **before**
segmentation, so the engine sees what a coarser acquisition would have given
it. The reference surface always comes from the expert mask at native
resolution.

| Level | Minimum spacing | Meaning |
|---|---|---|
| `native` | none | the scan as acquired |
| `iso1.5` | 1.5 mm on every axis | a typical diagnostic-quality grid |
| `iso3` | 3 mm on every axis | TotalSegmentator fast's working resolution |
| `slice5` | 5 mm slices, native in-plane | thick-slice acquisition |

The rules:

- Levels only coarsen. An axis already coarser than a level keeps its native
  spacing. A level that leaves a scan unchanged is flagged `resampled = False`
  and reuses the native masks.
- The CT and the expert mask are Gaussian anti-aliased (σ = (factor − 1)/2
  voxels), then linearly interpolated. The mask is then thresholded at 0.5.
- Every grid puts voxel *i* at *i* × spacing, so meshes from any grid share
  the reference's frame.
- Mask-level scores map coarse masks back to the native grid by nearest
  neighbour. For the expert mask itself, this gives the overlap lost to
  sampling alone.

| Dataset | Structures | Licence | Label rule |
|---|---|---|---|
| `msd_spleen`: Medical Segmentation Decathlon Task09 | spleen | CC-BY-SA 4.0 | expert labels in all 41 training cases |
| `msd_pancreas`: Medical Segmentation Decathlon Task07 | pancreas | CC-BY-SA 4.0 | expert pancreas and tumour labels, merged into the organ (TotalSegmentator's pancreas includes lesions); `--sample 41` draws the paper's held-out set |
| `ctorg`: CT-ORG (TCIA) | liver, lungs, bone | CC BY 3.0 | liver: all 140 cases; lungs and bone: **cases 0–20 only**, since the training split's lungs and bone were labelled by morphological algorithms |

Datasets go in `data/`, which git and Docker ignore. TotalSegmentator was
trained only on University Hospital Basel scans, so both datasets are
external to it.

```bash
# MSD Spleen: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar (1.5 GB)
python -m research.tier2 --dataset msd_spleen --data data/msd/Task09_Spleen --preset pilot
python -m research.tier2 --dataset msd_spleen --data data/msd/Task09_Spleen --preset full
python -m research.tier2 --dataset msd_spleen --data data/msd/Task09_Spleen --preset full --resolutions native
# MSD Pancreas, the held-out organ: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar (12.3 GB)
python -m research.tier2 --dataset msd_pancreas --data data/msd/Task07_Pancreas --sample 41 --preset full --engines ts_fast,ts_full
# Coarsening-model sensitivity: box slice profile and added noise at iso3 and slice5
python -m research.tier2 --dataset msd_spleen --data data/msd/Task09_Spleen --preset sensitivity --engines ts_fast
# CT-ORG: download from TCIA, then point --data at the folder holding volume-N / labels-N
python -m research.tier2 --dataset ctorg --data data/ctorg --preset full
```

Engines: `ts_fast` (TotalSegmentator, 3 mm) and `ts_full` (TotalSegmentator
at 1.5 mm, running only the sub-model that holds the study structures, on a
crop found by the 3 mm model) for any structure, and `classical` (the
intensity-threshold methods in `validate.py`) for lungs and bone. Engine masks
are cached in `output/research/tier2/cache/`, so re-runs skip segmentation;
`--segment-only` runs the engines and stops. Each run writes `results.csv`
(mask-level and mesh-level rows), `run.json` (dataset, licence, label rules,
grid, timings, commit) and `summary.md`. Rows are also saved unit by unit to
`partial.jsonl` while the run is in progress; passing the run's folder as
`--outdir` resumes it from the units already complete.

Besides the grid and production, every mask is meshed with `production_mm`
(production smoothing and decimation, blur 0.5 mm), which separates blur units
from the grid. Coarsening variants are written `<level>~box` (a rectangular
slice profile one coarse voxel wide, on the CT and the expert mask) and
`<level>~noise` (Gaussian profile plus white noise of 20 HU on the CT). Mesh
metrics also include NSD at 0.5 and 2 mm, the 99th percentile distance, and
local defects: faces of the test mesh more than 2 mm from the reference, as an
area share and the area of the largest connected patch.

`paper_stats.py` adds the statistics the paper needs beyond the report:
scan-clustered bootstrap CIs for the variance shares, a random-intercept
variance-component estimate, leave-one-scan-out and leave-one-level-out
validation of the sampling relationship, a grid comparison at identical
physical settings, and the held-out hypotheses of
[`heldout_hypotheses.md`](heldout_hypotheses.md). `supplement_stats.py` writes
the supplement's analyses (S1–S12). The per-mesh results of the runs behind the
paper are kept, compressed, in [`results/`](results/); see
[`docs/paper/README.md`](../docs/paper/README.md) for the full reproduction.

`tier2_report.py` turns a run into the figures, LaTeX tables and
`numbers.json` behind `docs/tier2-report/tier2-resolution-report.tex`. That
way every number in the report comes from `results.csv`:

```bash
python -m research.tier2_report output/research/tier2/<run> --out docs/tier2-report \
       --baseline output/research/tier2/<earlier native-only run>
```

## Validation of the application

Two checks of what the app itself delivers, using the Tier 2 data. Both run
the application's own code, not research copies of it.

**Test 1, the accuracy card (`card_validation.py`).** The app's
`build_mesh` call, with the accuracy estimate on, meshes the cached
TotalSegmentator masks of all 82 Tier 2 scans (both modes, all four grids:
590 meshes). Each line of the card is then checked against the delivered
mesh's true error to the expert surface:

- the resolution limit, on the coarsened expert mask with the pancreas held
  out of the fit, and its range on the Tier 1 phantoms;
- the smoothing shift, against the change it actually caused;
- the engine's figure, leave one scan out: its middle half should hold about
  half the scans and its 90% range about nine in ten;
- the watertight flag, against VTK's own edge classification.

The first run found the card's engine figure overconfident on coarse grids:
it used the native reference everywhere, and its middle half held as few as
15% of full-mode spleen meshes on 3 mm grids. The card now uses the study grid
closest to the scan (`accuracy.reference_level`) and shows a 90% range. The
summary reports both versions on the same meshes. The phantoms also showed the
resolution range's lower bound, 0.14, sat at the isotropic median; it is now
0.13, the phantoms' 2.5th percentile.

**Test 2, repeatability (`repeatability.py`).** The same scans (5 spleen,
5 pancreas, drawn with seed 2026) go through TotalSegmentator again and
again, in both modes: on the app's persistent GPU worker, in a new GPU
process each time (nnU-Net's cuDNN autotuning can choose different kernels),
and on the CPU. Each run is compared voxel by voxel with the first, the organ
is meshed with the app's call (twice, which must give an identical mesh), and
the mesh is compared with the first run's and with the expert surface.

```bash
python -m research.card_validation --workers 8          # ~3 min, needs the Tier 2 engine cache
python -m research.repeatability                        # ~2 h: GPU, then CPU runs
python -m research.validation_figures output/research/card_validation/<run> output/research/repeatability/<run>
```

Both write `results.csv`, `run.json` and `summary.md` under
`output/research/`; the runs reported are kept in [`results/`](results/).
**Test 3, printed phantoms in CT (`physical_phantom.py`), ready, awaiting a
scan.** The Tier 1 shapes plus three vessel calibres and a bifurcation, made
physical: `design` writes the STL files and a caliper sheet, `simulate` writes
a synthetic CT of the kit as DICOM to test the analysis, and `analyse` scores
the app's meshes of each scanned object against its exact surface after a
rigid fit. The print and scan protocol is in
[`docs/validation/phantom/`](../docs/validation/phantom/README.md).

`docs/validation/make_report.py --pdf` writes the validation report from those
runs, and `docs/validation/make_slides.py` refreshes the validation slides in
`MEDSYS-presentation.html`; every number in both comes from the runs.

## Tests

`tests/test_phantoms.py` checks the exact distance functions, the closed-form
volumes and areas against fine meshes, uniform-by-area surface sampling, the
metrics against a known 1 mm offset, and that `build_mesh` output is unchanged
by the move to `reconstruction.py`. `tests/test_tier2.py` checks the dataset
label rules, the TotalSegmentator structure mapping, the per-engine production
settings, mesh-to-mesh metrics (including that flipped reference normals cannot
invert the shrinkage sign), the resolution rules (coarsen only, shared physical
frame, volume kept, NIfTI affine of the resampled CT, coarse masks cached per
level) and end-to-end units on a synthetic labelled volume at native and 3 mm.
Tests that need PyVista or nibabel skip in CI.
