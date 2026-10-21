# Validation test 3: printed phantoms in a real CT scanner

Seven objects whose true surface is known exactly (the paper's Tier 1 shapes,
made physical). Print them, scan them once, and the analysis measures how far
MEDSYS's 3D models are from the real objects.

| Object | Size | Tests |
|---|---|---|
| `sphere_r15` | Ø 30 mm | smooth, convex surface |
| `rounded_box` | 32 × 24 × 16 mm, 2 mm rounding | flat faces and tight edges, like bone |
| `torus` | Ø 48 mm, tube Ø 12 mm | a hole (topology) |
| `capsule_r1` | Ø 2 mm × 42 mm | small vessel |
| `capsule_r2` | Ø 4 mm × 44 mm | medium vessel |
| `capsule_r3` | Ø 6 mm × 46 mm | large vessel |
| `branch` | trunk Ø 6 mm, branches Ø 4 mm | a vessel bifurcation |

The STL files are generated, not stored in git:

```bash
python -m research.physical_phantom design          # STL files, kit.json and calipers.csv here
```

## 1. Print

- **Material:** PLA or resin. A resin (SLA) printer is preferred, at about ±0.05 mm. An FDM printer is accurate to only about ±0.1–0.2 mm, which is close to the error being measured.
- **Infill 100%**, so the CT sees a solid object rather than a lattice. Layer height 0.1 mm or finer.
- **Orientation:** lay the tubes, torus and branch flat. The sphere needs supports; sand the support marks smooth.
- **No metal** anywhere: no screws, weights or wire. Metal causes CT streak artefacts.
- **Measure** each object with calipers and fill in the `measured_mm` column of `calipers.csv`. The analysis reports printing error separately.
- **Optional gold standard:** add 2–3 precision plastic balls (acrylic or Delrin, grade ±0.025 mm, e.g. 12.7 and 25.4 mm). They're cheap and more accurate than any print.

## 2. Scan (one session, about 30 minutes)

A note for the radiographer: this is a plastic object, not a patient, so no ethics approval is needed, only permission to use the scanner.

- **Arrangement:** place the objects on a sheet of polystyrene foam or in a cardboard box, **at least 20 mm apart** and not touching the box walls. Tilting some of them is good.
- **Protocol:** a standard abdominal CT protocol, 120 kVp, soft-tissue (standard) kernel.
- **Reconstruct one acquisition several times:**
  - slice thickness **0.6 (or 0.625), 1.25, 2.5 and 5 mm**;
  - field of view about 250 mm (pixels about 0.5 mm), and one at 400–500 mm (pixels about 0.8–1 mm) to mimic body scans.
- **Optional:** a bone kernel at 1.25 mm, and a second scan with the objects submerged in water. PLA in water has a soft-tissue-like contrast of about 150 HU, closer to real organs.
- **Export:** uncompressed DICOM, one folder per reconstruction, onto a USB stick.

## 3. Analyse

```bash
python -m research.physical_phantom analyse "D:/phantom/*" --calipers docs/validation/phantom/calipers.csv
```

For each reconstruction and object, the analysis:

1. thresholds at the half-way point between background and plastic, the classical 50% surface;
2. groups pieces broken apart by thick slices into one object, and matches each object to the kit by its size and shape;
3. fits the exact surface rigidly, minimising the true distance;
4. scores two meshes: the raw surface, and the app's production settings (smoothing and decimation);
5. checks the accuracy card's resolution limit against the raw surface error.

Results go to `output/research/physical_phantom/<run>/`: results.csv, summary.md and run.json.

**Tested before the real scan** on simulated scans with partial volume, blur and noise:

- At 0.7 × 0.7 × 1.25 mm, every object came out at 0.13–0.22 mm.
- At 5 mm slices, the 2 mm and 4 mm tubes break into pieces, and the app's smoothing erases the 2 mm tube completely. The paper's Tier 1 found the same, and this is the effect the real scan will measure.

The rigid fit absorbs any offset in where the object lay, so the errors describe shape: what thresholding and reconstruction did to the geometry.
