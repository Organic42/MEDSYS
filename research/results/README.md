# Saved results

Per-mesh results of the three runs reported in the paper, compressed
(`results.csv.gz`), with each run's `run.json` (commit, library versions,
grid, seed) and `summary.md`. The scripts read the `.csv.gz` files directly.

| Folder | Run | Rows |
|---|---|---|
| `tier1_phantoms_20261002-222422_full` | Tier 1, digital phantoms | 7,760 meshes |
| `tier2_spleen_native_20261003-093356` | Tier 2, MSD Spleen, native resolution only | 4,059 rows |
| `tier2_spleen_resolution_20261004-214129` | Tier 2, MSD Spleen, four resolution levels | 16,332 rows |

Each row is one mesh (or one mask-level score): its configuration, its
distances to the reference surface, and its topology. No images or masks are
included.

## Licence

The Tier 1 results are covered by the repository's MIT licence. The Tier 2
results are derived from the Medical Segmentation Decathlon Task09 Spleen
dataset (Antonelli et al., Nature Communications 13, 4128, 2022), which is
licensed CC-BY-SA 4.0, so these derived results are shared under CC-BY-SA 4.0.
