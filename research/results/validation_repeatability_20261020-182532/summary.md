# Validation test 2: repeatability

Run 20261020-182532. 10 scans (pancreas, spleen), TotalSegmentator fast and full modes, 184 segmentation runs. Every run is compared with the first run on the app's persistent GPU worker for the same scan and mode.

## Same mask, same mesh

Not reported for this run: its check compared the card's blank volume change (open meshes) as unequal to itself, flagging the 29 open meshes and no others. Meshes rebuilt from the same masks were identical; validation test 1 checks this on every mesh.

## Same scan, repeated segmentation

| Mode | Path | Runs | Scans with any organ change | Organ voxels changed (max) | Organ Dice (min) | All-structure voxels changed (median, max) | Mesh shift vs first run, mm (median, max) |
|---|---|---|---|---|---|---|---|
| fast | worker | 40 | 0 of 10 | 0 | 1.00000 | 0, 0 | 0.0000, 0.0000 |
| fast | fresh_gpu | 30 | 8 of 10 | 96 | 0.99925 | 770, 2208 | 0.0031, 0.0171 |
| fast | cpu | 20 | 10 of 10 | 96 | 0.99899 | 3246, 9024 | 0.0137, 0.0252 |
| full | worker | 40 | 0 of 10 | 0 | 1.00000 | 0, 0 | 0.0000, 0.0000 |
| full | fresh_gpu | 30 | 5 of 10 | 16 | 0.99962 | 154, 468 | 0.0000, 0.0146 |
| full | cpu | 4 | 1 of 2 | 4 | 0.99997 | 309, 372 | 0.0012, 0.0024 |

## Does it change the measured error?

Spread of the true error (ASSD against the expert surface) across repeated runs of the same scan, compared with the spread between scans in the Tier 2 study.

| Mode | Path | Largest within-scan range of true ASSD (mm) | Mean within-scan SD (mm) |
|---|---|---|---|
| fast | worker | 0.0000 | 0.0000 |
| fast | fresh_gpu | 0.0135 | 0.0017 |
| fast | cpu | 0.0143 | 0.0024 |
| full | worker | 0.0000 | 0.0000 |
| full | fresh_gpu | 0.0112 | 0.0007 |
| full | cpu | 0.0059 | 0.0003 |

For scale, the interquartile range of the same error between scans in the Tier 2 study (production meshes, native; 41 scans per organ): spleen fast 0.174, full 0.193; pancreas fast 1.078, full 1.893 mm.

## Speed

| Mode | Path | Median seconds per scan |
|---|---|---|
| fast | worker | 6.0 |
| fast | fresh_gpu | 29.9 |
| fast | cpu | 33.6 |
| full | worker | 18.4 |
| full | fresh_gpu | 75.7 |
| full | cpu | 175.1 |