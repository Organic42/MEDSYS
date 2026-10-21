# Held-out confirmation: hypotheses fixed before the pancreas data

Written on 2026-10-18, while MSD Task07 Pancreas was still downloading and
before any pancreas scan had been segmented, meshed or scored. The spleen
results (MSD Task09, 41 scans) generated each hypothesis; the pancreas set
tests them once, with the criteria below, and the paper reports every
outcome, confirmed or not.

## Data and design

- MSD Task07 Pancreas, training set. Expert labels 1 (pancreas) and 2
  (tumour) are merged into the organ, since TotalSegmentator's pancreas
  includes lesions.
- 41 scans, drawn at random with seed 2026 (`--sample 41`), the same number
  as the spleen set. No scan is excluded after the draw; a scan whose mask
  or mesh fails is reported as a failure.
- Same pipeline, levels (native, iso1.5, iso3, slice5), reconstruction grid
  (48 configurations plus production and production in millimetres),
  metrics and statistics as the spleen experiment. Engines: TotalSegmentator
  fast (the deployed mode) and full resolution.
- Level effects use only the scans a level actually resampled.

## Hypotheses and criteria (TotalSegmentator fast unless stated)

| | Hypothesis (from the spleen) | Spleen value | Confirmed on the pancreas if |
|---|---|---|---|
| H1 | Simulated coarsening moves a correct surface by about 0.177 Δ | slope 0.177 [0.171, 0.184]; LOSO MAE 0.037 mm | the spleen-fitted relationship predicts the pancreas expert-mask canonical ASSD with MAE ≤ 0.074 mm (twice the spleen LOSO MAE). The pancreas slope and its CI are reported either way |
| H2 | Spacing dominates the error of a correct segmentation | η² resolution 50.5% vs case 10.3% | η² of resolution exceeds that of the case and of each reconstruction factor, for the expert mask |
| H3 | The patient dominates the engine's error | η² case 89.5%; random intercept 94.2% | η² of the case exceeds that of resolution and the random-intercept scan share exceeds 50% |
| H4 | In-plane coarsening to 3 mm leaves the engine's error unchanged | iso3 − native +0.006 [−0.006, +0.019] mm | the 95% CI of the paired canonical ASSD change iso3 − native lies within ±0.10 mm |
| H5 | 5 mm slices raise the engine's error | slice5 − native +0.188 [+0.150, +0.229] mm | the paired change slice5 − native is positive with a 95% CI excluding 0 |
| H6 | Meshing on the engine's 3 mm grid improves closure at identical physical settings without changing accuracy | closure at decimation 0.75: +26 points [20, 32]; ΔASSD −0.017 mm | the closure gain at decimation 0.75 has a 95% CI above 0 and the scan-level ΔASSD lies within ±0.05 mm |

"Partly confirmed" is used only where a hypothesis has two parts and one
holds. Statistics are those of the paper: scan bootstrap (5,000 resamples)
for paired changes, 1,000 for variance shares, Wilcoxon signed-rank tests,
seed 2026.

## Exploratory (no confirmation claimed)

Full-resolution engine results on both organs, the coarsening-model
sensitivity (box slice profile, added noise), local defect metrics, the
smoothing-by-organ interaction and case covariates are reported as
exploratory.
