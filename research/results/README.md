# Saved results

Per-mesh results of the runs reported in the paper, compressed
(`results.csv.gz`), with each run's `run.json` (commit, library versions,
grid, seed, engine timings) and `summary.md`. The scripts read the `.csv.gz`
files directly.

| Folder | Run | Rows |
|---|---|---|
| `tier1_phantoms_20261002-222422_full` | Tier 1, digital phantoms | 7,760 meshes |
| `tier2_spleen_native_20261003-093356` | Tier 2, MSD Spleen, native resolution only (first version) | 4,059 rows |
| `tier2_spleen_resolution_20261004-214129` | Tier 2, MSD Spleen, four resolution levels, TotalSegmentator fast (first version) | 16,332 rows |
| `tier2_spleen_two_modes_20261018-210930` | Tier 2, MSD Spleen, four levels, TotalSegmentator fast and full, extended metrics (revision) | 25,024 rows |
| `tier2_pancreas_heldout_20261018-223006` | Tier 2, MSD Pancreas, 41 scans drawn at random (seed 2026), the held-out organ (revision) | 25,045 rows |
| `tier2_spleen_coarsening_20261018-213641` | MSD Spleen, coarsening-model sensitivity: box slice profile and added noise at `iso3` and `slice5` (revision) | 1,258 rows |
| `validation_card_20261020-203227` | Validation test 1: the app's accuracy card against the true error of its own meshes, all 82 Tier 2 scans (`card_validation.py`); `coverage.csv` per organ, mode and grid | 803 rows (590 meshes, 213 coarsened expert masks) |
| `validation_repeatability_20261020-182532` | Validation test 2: 10 scans segmented repeatedly on the GPU worker, in new GPU processes and on the CPU (`repeatability.py`) | 184 runs |

Each row is one mesh (or one mask-level score): its configuration, its
distances to the reference surface, and its topology. No images or masks are
included.

The revision's spleen run contains every row of the first version's
resolution run, with identical values (checked row by row), plus the
full-resolution engine, the `production_mm` configuration and the extended
metrics (`nsd_0.5mm`, `nsd_2mm`, `p99_mm`, `defect_patch_mm2`,
`defect_area_pct`, `missed_ref_pct`). Its `run.json` records commit
`6ed708d+dirty`: the runs were made from the working tree that became the
revision commit, and part of the spleen scoring was resumed after a reboot
(`resumed_units`).

## Licence

The Tier 1 results are covered by the repository's MIT licence. The Tier 2
and validation results are derived from the Medical Segmentation Decathlon Task09 Spleen and
Task07 Pancreas datasets (Antonelli et al., Nature Communications 13, 4128,
2022), which are licensed CC-BY-SA 4.0, so these derived results are shared
under CC-BY-SA 4.0.
