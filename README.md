---
title: MEDSYS
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
dockerfile: Dockerfile.web
app_port: 7860
pinned: false
license: mit
---

<div align="center">

# MEDSYS

**Risk-aware reconstruction of 3D anatomy from medical scans**

A research project and an open platform: it turns DICOM studies into per-structure 3D meshes,
and measures how far those meshes can be trusted.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements)
[![CI](https://github.com/Organic42/MEDSYS/actions/workflows/python-app.yml/badge.svg)](https://github.com/Organic42/MEDSYS/actions/workflows/python-app.yml)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#status)

[Research question](#research-question) · [Findings so far](#findings-so-far) · [Methods](#methods) · [Reproduce](#reproducing-the-results) · [Platform](#the-platform) · [Roadmap](#roadmap) · [Cite](#citation)

</div>

---

> **Not a medical device.** MEDSYS is for research and education. It is not cleared or certified
> for clinical use and must not inform diagnosis or treatment. See [Disclaimer](#disclaimer).

## Summary

Automated pipelines can turn a CT or MRI scan into a 3D model in minutes, but they do not say how
far that model's surface may be from the real anatomy. Error enters at every stage (segmentation,
voxel size, blurring, surface extraction, smoothing, simplification), and the stages interact.

**Thesis.** This project develops a risk-aware framework for automated DICOM-to-3D anatomical
reconstruction that quantifies stage-wise geometric error, predicts final-mesh reliability without
ground truth, and selects reconstruction pipelines subject to task-specific accuracy and
computational constraints. In short: **measure → predict → decide**.

MEDSYS, the platform in this repository, is both the system under study and the instrument used
to study it. Production and research run the same reconstruction code
([`reconstruction.py`](reconstruction.py)), so a result measured here describes the meshes the
app actually ships.

## Research question

| | Question | Role | Status |
|---|---|---|---|
| **RQ1 — Measure** | Which pipeline stages, and which interactions between them, contribute most to final anatomical geometric error? | Core thesis | Tier 1 (phantoms) done; Tier 2 (anatomy) next |
| **RQ2 — Predict** | Can final-mesh error and task suitability be predicted without ground truth at inference time, with empirically calibrated coverage? | Major extension | Planned |
| **RQ3 — Decide** | Can those predictions select the lowest-cost pipeline that satisfies a predefined task-specific reliability constraint? | Applied extension | Planned |

Each question builds on the one before it. RQ2 is only distinct from existing segmentation quality
control if RQ1 shows that reconstruction stages or topology matter. The full proposal, with its
literature basis, is in [`docs/gap-analysis-two-pager.tex`](docs/gap-analysis-two-pager.tex)
(plain text: [`.txt`](docs/gap-analysis-two-pager.txt)); the evidence log behind it is
[`docs/literature-gap-analysis.md`](docs/literature-gap-analysis.md).

## Findings so far

These are early, descriptive results. Phantom results describe reconstruction alone, not anatomy;
single-scan results are pilots, not evidence of generalisation. The formal attribution
(mixed-effects models and Sobol indices) has not been run yet.

### 1. Reconstruction error on exact surfaces (RQ1, Tier 1)

Four digital phantoms with exact signed distance functions (a sphere, a 2 mm-radius capsule, a
torus and a rounded box), each at 5 random placements, voxelised at four spacings and reconstructed
under 96 research configurations plus the production defaults: **7,760 meshes**, scored with exact
surface distances.

**Reference floor.** The canonical procedure (no blur, marching cubes, no smoothing, no decimation)
is the one used to build reference surfaces in Tier 2, so its error is the floor below which
anatomy results cannot be interpreted. ASSD in mm (HD95 in brackets), mean of 5 placements:

| Phantom | 0.5 mm | 1 mm | 2 mm | 0.8 × 0.8 × 5 mm |
|---|---|---|---|---|
| Sphere | 0.070 (0.17) | 0.147 (0.34) | 0.271 (0.75) | 0.495 (1.62) |
| Capsule, r 2 mm | 0.069 (0.17) | 0.144 (0.35) | 0.331 (0.77) | 0.634 (1.95) |
| Torus | 0.068 (0.16) | 0.138 (0.33) | 0.278 (0.66) | 0.499 (1.61) |
| Rounded box | 0.072 (0.17) | 0.145 (0.33) | 0.297 (0.68) | 0.525 (1.72) |

Roughly 0.14 × the voxel size at isotropic spacing. Placement-to-placement variation was small
(coefficient of variation at most 0.19).

**What drives error.** Share of variance in log ASSD across the balanced grid: voxel spacing
**68.6%**, shape 13.5%, blur 3.3%, Taubin smoothing 0.5%, decimation 0.3%, isosurface variant
0.0%, placement 1.1%, interactions **12.7%**. Shares depend on the factor ranges chosen.

**Stages interact, so errors cannot simply be added.** Sixty Taubin iterations *reduce* mean ASSD
at 0.5 mm voxels (0.090 → 0.076 mm) but nearly double it at 2 mm (0.327 → 0.612 mm).

**The production defaults help smooth shapes and hurt thin ones.** On the sphere at 1 mm, ASSD
falls from 0.147 to 0.088 mm. On the 2 mm-radius capsule at 2 mm voxels it rises from 0.331 to
1.192 mm and the tube shrinks by 1.13 mm, more than half its radius. Two causes: smoothing at
coarse spacing, and blur specified in voxels (at 5 mm slices the 0.6-voxel blur is 3 mm along the
slice axis). The research grid therefore specifies blur in millimetres.

**Topology.** No mesh had holes without decimation: every open mesh came from the decimation step.
Separately, at 0.8 × 0.8 × 5 mm the thin capsule failed topology checks in 64% of meshes, mostly
by splitting into separate pieces: slices thicker than the 4 mm tube cannot sample it
continuously, a resolution limit rather than a reconstruction one.

### 2. Structural checks on a production model

The workbench checks every loaded mesh in the browser. On a 79-structure chest CT segmented by
TotalSegmentator (fast mode) and meshed with production settings, **46 of 79** meshes are closed,
**43 of 79** are a single piece and **58 of 79** are manifold. That is consistent with the phantom
finding that decimation opens holes, though on real data the cause is not yet established;
multi-piece lungs may also reflect the segmentation itself.

### 3. Pilot: classical engine against TotalSegmentator

One chest CT, scored against TotalSegmentator in fast mode. This measures *agreement with an
algorithm*, not anatomical accuracy, and motivated the research problem rather than answering it.

| Structure | Dice | IoU | HD95 | ASSD | Volume error |
|---|---|---|---|---|---|
| Lungs | 0.952 | 0.908 | 22.2 mm | 2.51 mm | −4.2% |
| Skeleton | 0.114 | 0.061 | 112.7 mm | 37.8 mm | −53.0% |

### 4. Compute cost

One 192-slice 1 mm brain MRI on an RTX 5080 (16 GB) and a 6-core CPU, PyTorch 2.12 with CUDA
13.0. Times are for the stage named; "warm" excludes first-run CUDA start-up.

| Stage | CPU | GPU | Speed-up |
|---|---|---|---|
| MedSAM refinement | 432.8 s | 24.5 s | 17.7× |
| Whole MRI pipeline | 540 s | 132 s | 4.1× |
| TotalSegmentator MRI, fast (3 mm) | 24.2 s | 29.2–30.5 s warm, 45.4 s first run | none |
| TotalSegmentator MRI, full (1.5 mm) | 60.2 s | 38.7 s warm, 50.7 s first run | 1.6× |

In fast mode the GPU loses: each job is a fresh process and CUDA start-up outweighs the small
model. The cost argument for cheap classical engines is weaker than commonly assumed, which is
why the thesis targets reliability rather than speed.

## Methods

**Reference standard.** Three objects are kept distinct: the *ground truth* (an expert-annotated
mask, itself a reference standard with an inter-rater floor), the *reference surface* (a mesh
built from that mask by one fixed, documented procedure: native-resolution marching cubes with no
smoothing or decimation), and the *test mesh* (MEDSYS output under a given configuration). Error
is always test mesh against reference surface.

**Two tiers.** Tier 1 uses digital phantoms with analytic surfaces, which isolate reconstruction
error exactly and measure the reference procedure's own error. Tier 2 measures end-to-end error
on public CT datasets with expert labels, CT first (lung, bone, liver, optionally vessels), with
brain MRI as an extension. TotalSegmentator is evaluated only on held-out or external data.

**Design.** A split-plot factorial: segmentation engine (classical, TotalSegmentator 3 mm,
TotalSegmentator 1.5 mm) × voxel spacing as the expensive whole plot; blur (0 / 0.5 / 1 mm) ×
isosurface algorithm × Taubin iterations (0 / 10 / 30 / 60) × decimation (0 / 0.25 / 0.5 / 0.75)
as the cheap sub-plot. The canonical baseline is TotalSegmentator 1.5 mm with the reference
procedure; production defaults appear as one named configuration only.

**Metrics.** Symmetric, area-weighted surface distances (ASSD, HD95, Hausdorff, surface Dice),
signed mean distance (shrinkage or expansion), volume and surface-area error, and topology
(components, boundary and non-manifold edges, genus).

**Sample size.** Set by RQ2: split conformal prediction at 95% coverage needs at least 19
calibration cases per group, so the plan is roughly 60–100 cases per structure.

## Reproducing the results

```bash
pip install -r requirements-full.txt

# Tier 1 phantom experiment (output/research/phantoms/<run>/)
python -m research.phantom_experiment --preset smoke   # seconds
python -m research.phantom_experiment --preset pilot   # 1,552 meshes, about 1 min
python -m research.phantom_experiment --preset full    # 7,760 meshes, about 3 min on 6 cores, seed 2026
python -m research.phantom_experiment --resummarise output/research/phantoms/<run>

# Reference comparison for one scan
python validate.py --dicom <ct_dicoms> --totalseg output/<dataset>/segmentations \
                   --out output/<dataset>/validation

# Tests (65): phantom geometry, metrics, reconstruction, and the pipeline
pytest
```

Each phantom run writes one row per mesh (`results.csv`), its provenance (`run.json`: commit,
library versions, grid, seed) and a descriptive summary (`summary.md`). Details:
[`research/README.md`](research/README.md).

## The platform

MEDSYS runs end to end: upload a zipped DICOM study in the browser, and it detects the modality,
segments it, reconstructs per-structure meshes and opens them in a 3D workbench.

| | |
|---|---|
| **Modalities** | CT, MRI (brain) and PET, routed automatically from DICOM metadata |
| **Engines** | A fast classical engine (intensity thresholds, BET, Gaussian mixture), and TotalSegmentator (nnU-Net) for 100+ labelled structures |
| **Workbench** | Three columns: dataset, structures and pipeline; a 3D stage with surface, wireframe and X-ray modes; mesh checks, the reliability estimate and metrics against a reference |
| **Honest reporting** | Mesh checks and reference metrics show measured data only; the reliability estimate stays empty until RQ2 exists, rather than showing invented numbers |
| **Neuroplasticity Explorer** | Questions about behaviour and the brain, mapped onto a 3D brain from a curated evidence base |
| **Deployment** | Docker Compose (API + Redis + workers), a lightweight Hugging Face Spaces image, and a standalone Windows build |

<details>
<summary><strong>Pipelines</strong></summary>

<br>

- **CT (classical):** Hounsfield units → windowing → body mask → lungs (border clearing) →
  skeleton (HU > 200) → soft tissue → meshes.
- **MRI brain (classical):** N4 bias correction → non-local-means denoising → BET skull strip →
  optional box-prompted MedSAM refinement → grey/white matter/CSF by Gaussian mixture → meshes.
- **PET:** activity normalisation → hotspot by tumour-to-background ratio → maximum-intensity
  projections and meshes.
- **TotalSegmentator:** DICOM → NIfTI → nnU-Net (CT or MR task, 3 mm fast mode in the app) →
  one mesh per structure.

Every route meshes through [`reconstruction.py`](reconstruction.py): Gaussian blur of the mask,
marching cubes, Taubin smoothing and decimation, with explicit, recordable parameters.

</details>

<details>
<summary><strong>Architecture</strong></summary>

<br>

```
Browser ──upload──▶ FastAPI (app.py) ──▶ job queue (in-process, or Redis + RQ)
                                              │
                                              ▼
                      segmentation_pipeline.py (subprocess per job)
                      ├─ modality detection and routing
                      ├─ classical engine / TotalSegmentator
                      └─ reconstruction.py ──▶ meshes + report.json
                                              │
              SQLite job store ◀──────────────┤
                                              ▼
                      Workbench (three.js) ◀── /api/datasets, /output/<dataset>/*.stl
```

Job state lives in SQLite, so it survives restarts and is shared by the API and workers. Each
segmentation runs as a subprocess, so a crashing job cannot take the service down.

</details>

<details>
<summary><strong>Running and deploying</strong></summary>

<br>

```bash
# Local web app
pip install -r requirements-full.txt
python app.py                         # http://127.0.0.1:8000

# Full stack: API + Redis + workers
docker compose up --build
docker compose up --scale worker=3

# Command line
python segmentation_pipeline.py --input <dicom_dir> --name <dataset>
python segmentation_pipeline.py --input <dicom_dir> --name <dataset> --engine totalseg [--no-fast]
```

- **Hugging Face Spaces:** `Dockerfile.web` with `requirements-web.txt` builds a CPU-only image
  without PyTorch or TotalSegmentator; the front matter at the top of this file configures the
  Space. The UI detects missing engines through `/api/capabilities`.
- **Windows:** `python -m PyInstaller medsys.spec --noconfirm` produces `dist/MEDSYS/`, which runs
  without Python. The build re-invokes its own executable to run each job, since a frozen app has
  no separate interpreter.
- **GPU:** install a CUDA build of PyTorch that supports your card. RTX 50-series cards need
  CUDA 12.8 or newer. Jobs fall back to the CPU when no GPU is found.

</details>

## Repository structure

```
MEDSYS/
├── reconstruction.py          # Shared mesh reconstruction (production and research)
├── segmentation_pipeline.py   # Modality-aware segmentation pipelines
├── validate.py                # Reference comparison: Dice, IoU, HD95, ASSD
├── research/                  # Tier 1 phantom experiment, exact surface metrics
├── docs/                      # Research proposal, gap analysis, one-page summary
├── tests/                     # 65 tests: pipeline, phantoms, metrics, reconstruction
├── app.py, tasks.py, worker.py, jobstore.py, config.py   # Web service and job handling
├── web/                       # Workbench and Neuroplasticity Explorer
├── brain_knowledge.py         # Explorer evidence base
├── entry.py, launcher.py, medsys.spec                    # Standalone Windows build
└── Dockerfile, Dockerfile.web, docker-compose.yml, requirements*.txt
```

## Roadmap

**Research**
- [x] Tier 1 phantom experiment with exact surfaces (RQ1)
- [ ] Systematic, logged literature search to confirm novelty
- [ ] Tier 2: public expert-annotated CT datasets, with leakage and label-bias controls
- [ ] Formal attribution: mixed-effects models and Sobol indices
- [ ] RQ2: reliability model with split-conformal intervals and task tolerances fixed in advance
- [ ] RQ3: accuracy-constrained pipeline selection, with latency and energy as costs

**Platform**
- [x] Shared, parameterised reconstruction module
- [x] GPU inference, with measured CPU and GPU timings
- [x] Browser-side mesh checks
- [ ] Production and research configuration profiles (parameters are still set at each call site)
- [ ] Minimum-size thresholds in mm³ rather than voxels (the voxel cutoff is 8× stricter at 3 mm than at 1.5 mm)
- [ ] Verified de-identification and opaque dataset identifiers
- [ ] Benchmark the MRI route against HD-BET and SynthStrip
- [ ] DICOM-SEG export and a provenance record on every output

## Status

A working research prototype. The platform runs end to end and its tests pass. The research is at
the end of RQ1 Tier 1: phantom results exist, anatomy results do not. No claim here should be read
as clinical accuracy.

### Limitations

- Phantom results describe reconstruction only; they say nothing about segmentation accuracy.
- The pilot comparison uses one scan and an algorithmic reference, not expert annotations.
- Compute timings come from one machine and one study.
- The novelty of RQ1 has not yet been confirmed by a systematic literature search.

## Requirements

- Python 3.10+
- `requirements-full.txt` for all engines; `requirements-web.txt` for the lightweight image
- Optional: an NVIDIA GPU with a matching CUDA build of PyTorch (MedSAM and TotalSegmentator)
- Optional: Docker

## Contributing

Issues and pull requests are welcome. Please run `pytest` before submitting, keep research and
production on the shared reconstruction code, and never commit patient data, DICOM files, model
checkpoints or pipeline output (`.gitignore` excludes them).

## Citation

```bibtex
@software{medsys2026,
  author = {Organic42},
  title  = {MEDSYS: Risk-Aware Reconstruction of 3D Anatomy from Medical Scans},
  year   = {2026},
  url    = {https://github.com/Organic42/MEDSYS}
}
```

MEDSYS builds on TotalSegmentator (Wasserthal et al., *Radiology: Artificial Intelligence*, 2023),
nnU-Net (Isensee et al., *Nature Methods*, 2021), MedSAM, BET, N4ITK and marching cubes; please
cite them where you use them.

## Disclaimer

MEDSYS is provided for **research and education only**. It is not a medical device, has not been
evaluated by any regulatory body, and must not be used to diagnose, treat or otherwise make
clinical decisions. Outputs from any engine may contain errors. Always defer to qualified
clinicians and validated clinical software.

## License

MIT. See [LICENSE](LICENSE).

## Contributors

- **[Organic42](https://github.com/Organic42)** — project lead
