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

## Tests

`tests/test_phantoms.py` checks the exact distance functions, the closed-form
volumes and areas against fine meshes, uniform-by-area surface sampling, the
metrics against a known 1 mm offset, and that `build_mesh` output is unchanged
by the move to `reconstruction.py`. Tests that need PyVista skip in CI.
