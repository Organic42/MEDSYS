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

**DICOM-to-3D medical imaging platform**

Modality-aware segmentation · AI-assisted organ mapping · Interactive 3D visualization · Neuroplasticity Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements)
[![CI](https://github.com/Organic42/MEDSYS/actions/workflows/python-app.yml/badge.svg)](https://github.com/Organic42/MEDSYS/actions/workflows/python-app.yml)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)

[Overview](#overview) · [Features](#features) · [Architecture](#architecture) · [Quick Start](#quick-start) · [Deployment](#deployment) · [Accuracy](#accuracy--validation) · [Contributing](#contributing)

</div>

---

## Overview

MEDSYS turns raw clinical imaging (DICOM) into browser-explorable 3D anatomy. Upload a scan,
and it auto-detects the modality, runs the appropriate segmentation pipeline, and renders the
result as rotatable, per-structure 3D meshes alongside the source imagery — no desktop imaging
software, no manual conversion steps.

It ships two segmentation engines, three imaging modalities, a production-grade web service,
a free-tier hosting path, a standalone desktop build, and a quantitative accuracy-validation
suite — built as a complete, deployable system rather than a research script.

> **Not a medical device.** MEDSYS is an educational and research tool. It is not FDA-cleared,
> not CE-marked, and must not be used for diagnosis, treatment planning, or any clinical
> decision. See [Disclaimer](#disclaimer).

---

## Features

| Capability | Detail |
|---|---|
| **Modality-aware routing** | Auto-detects CT, MRI (T1/T2), or PET from DICOM metadata and dispatches to the matching pipeline |
| **Dual segmentation engines** | Fast heuristic engine (HU thresholding, BET, GMM) for instant results; TotalSegmentator (nnU-Net) for 100+ individually-labeled anatomical structures |
| **CT pipeline** | HU conversion → windowing → lung / skeleton / body / soft-tissue segmentation → colored 3D meshes |
| **MRI brain pipeline** | N4 bias correction → NLM denoising → BET skull stripping → MedSAM refinement → GM/WM/CSF tissue classification → hybrid 3D surface |
| **PET pipeline** | Activity normalization → cerebral uptake mapping → tumor hotspot via tumor-to-background ratio → MIP views |
| **Web application** | Drag-and-drop upload, live job log, per-dataset gallery, interactive three.js 3D viewer with per-structure show/hide |
| **Neuroplasticity Explorer** | Natural-language questions ("what does chronic stress do to my brain?") mapped onto an anatomically-sculpted 3D brain, backed by a curated evidence base with a Claude fallback for open-ended queries |
| **Accuracy validation** | Dice, IoU, Hausdorff-95, and Average Surface Distance metrics, computed against a reference engine on identical voxel geometry |
| **Production hardening** | SQLite-backed job persistence, bounded/queued job dispatch (Redis + RQ optional), health/readiness endpoints, containerized deployment |
| **Multiple distribution paths** | Docker Compose (full stack), free-tier hosting (Hugging Face Spaces), standalone Windows `.exe` (no Python required) |

---

## Architecture

```
                         ┌─────────────────────────┐
   Browser  ── upload ──▶│   FastAPI (app.py)      │
   (drag & drop)         │   /api/upload            │
                         │   /api/jobs/{id}          │
                         │   /api/datasets           │
                         └──────────┬───────────────┘
                                    │ enqueue
                    ┌───────────────┴────────────────┐
                    │                                 │
            in-process pool                    Redis + RQ (prod)
            (default, single host)             (scale-out workers)
                    │                                 │
                    └───────────────┬────────────────┘
                                    ▼
                    segmentation_pipeline.py (subprocess)
                    ├─ modality detection
                    ├─ heuristic engine  ──┐
                    └─ TotalSegmentator  ──┴─▶ per-structure 3D meshes + report.json
                                    │
                                    ▼
                    SQLite job store  ◀──status──  shared output/ + uploads/ volume
                                    │
                                    ▼
                    three.js viewer  ◀── GET /output/<dataset>/*.stl
```

**Key design decision — hybrid engines, not a single model.** The heuristic engine is fast,
dependency-light, and validated to be near-equivalent to the AI engine for structures that are
easy to separate by intensity (e.g. lungs, Dice ≈ 0.95). For structures that require learned
priors (e.g. individual bones, organs), it is measurably worse (Dice ≈ 0.11) — see
[Accuracy & Validation](#accuracy--validation). Rather than pick one engine and accept its
weaknesses everywhere, MEDSYS exposes both and lets the deployment (and the data) decide.

---

## Quick Start

### Web UI (local)

```bash
git clone https://github.com/Organic42/MEDSYS.git
cd MEDSYS
pip install -r requirements-full.txt
python app.py
# open http://127.0.0.1:8000
```

### Docker Compose (full stack — API + Redis + worker)

```bash
docker compose up --build
docker compose up --scale worker=3      # more concurrent jobs
docker compose --profile gpu up         # GPU worker (NVIDIA Container Toolkit)
```

### Standalone Windows build

No Python, no install — see [Standalone Windows Build](#standalone-windows-build).

### Command line

```bash
python segmentation_pipeline.py --input <dicom_dir> --name <dataset_name>
python segmentation_pipeline.py --input <dicom_dir> --name <dataset_name> --engine totalseg
```

---

## Deployment

### Free-tier hosting (Hugging Face Spaces)

`Dockerfile.web` + `requirements-web.txt` build a lightweight image (no torch / TotalSegmentator
/ Redis) sized for Hugging Face's free CPU tier (2 vCPU, 16 GB RAM, no card required):

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) — SDK: **Docker**, hardware: **CPU basic**.
2. Link it to this repository (`Settings → Repository → Sync from GitHub`, branch `main`), or
   push directly: `git remote add space https://huggingface.co/spaces/<you>/medsys && git push space main`.
3. The Space reads the YAML block at the top of this README automatically and builds from
   `Dockerfile.web` on port `7860`.

On this deployment, MedSAM refinement and the TotalSegmentator engine are unavailable (no
checkpoint/torch shipped); the UI detects this via `GET /api/capabilities` and disables that
option automatically. CT, MRI brain, and PET pipelines run fully. Storage is ephemeral.

### Standalone Windows build

```bash
pip install -r requirements-web.txt -r requirements-build.txt
python -m PyInstaller medsys.spec --noconfirm
```

Output lands in `dist/MEDSYS/` — zip it and send it. The recipient unzips anywhere and
double-clicks `MEDSYS.exe`; it starts the server and opens the UI in their browser
automatically. No Python installation required. **Windows only** — PyInstaller builds are
platform-specific; a macOS or Linux build requires running PyInstaller on that OS.

<details>
<summary>Why a custom entry point instead of <code>pyinstaller app.py</code></summary>

<br>

The app normally runs each segmentation job as a `python segmentation_pipeline.py ...`
*subprocess* — but a frozen `.exe` has no separate interpreter to hand a script to
(`sys.executable` **is** the exe). `entry.py` is the actual PyInstaller entry point: it
re-invokes the exe itself with a `--run-pipeline` flag when a job needs to run, and dispatches
straight into `segmentation_pipeline.main()` instead of starting the server. `config.py`
resolves data paths relative to the exe's own folder when frozen, not PyInstaller's internal
bundle directory, so results always land somewhere the user can find them.

</details>

### Production hardening reference

| Concern | Solution |
|---|---|
| Jobs lost on restart | SQLite job store (`jobstore.py`), survives restarts, shared by API + workers |
| Unbounded concurrency | Bounded `ThreadPoolExecutor` (dev) or Redis + RQ queue (prod) |
| Scale-out | Stateless API + N worker containers sharing Redis and a `/data` volume |
| Config & secrets | Environment-driven (`config.py`) — `REDIS_URL`, storage paths, limits, timeouts |
| Liveness / readiness | `GET /health`, `GET /ready` |
| Upload safety | Streamed to disk with a configurable size cap; zip contents validated |

---

## Pipeline Stages

<details>
<summary><strong>MRI Brain</strong></summary>

<br>

| Stage | Operation |
|---|---|
| 1. Ingestion | Largest DICOM series by `SeriesInstanceUID`, sorted by slice position |
| 2. Preprocessing | N4 bias field correction (SimpleITK) → non-local means denoising → normalization |
| 3. Skull stripping | Brain Extraction Tool (Smith 2002); falls back to an intensity-based method on thin/anisotropic volumes where BET's surface model is unstable |
| 4. MedSAM refinement | Box-prompted per-slice refinement of the BET mask (optional — requires a checkpoint) |
| 5. Post-processing | Morphological cleanup, 3D largest-connected-component |
| 6. Tissue classification | 3-class Gaussian Mixture Model → GM / WM / CSF, sequence-aware (T1 vs. T2 intensity ordering) |
| 7. 3D reconstruction | Marching cubes → Taubin smoothing → decimation → per-tissue meshes |

</details>

<details>
<summary><strong>CT Chest</strong></summary>

<br>

CT carries calibrated Hounsfield Units, so tissues separate by density directly:

| Stage | Operation |
|---|---|
| 1 | `HU = pixel × RescaleSlope + RescaleIntercept` |
| 2 | Windowing to `[-1000, 400]` HU |
| 3 | Body/skin surface — per-slice fill, robust to airway-to-air connectivity |
| 4 | Lungs — internal air isolated via border-clearing |
| 5 | Skeleton — `HU > 200` inside the body mask |
| 6 | Soft tissue — remaining voxels in a mid-HU band |
| 7 | Colored 3D meshes with a cutaway anatomy render |

</details>

<details>
<summary><strong>TotalSegmentator Engine</strong></summary>

<br>

DICOM → NIfTI → nnU-Net inference → per-structure mesh, for CT or MR. Produces 100+
individually-labeled structures (every rib and vertebra, each lung lobe, major organs and
vessels) in a single pass. GPU recommended; degrades gracefully to CPU.

</details>

<details>
<summary><strong>PET (FET / FDG)</strong></summary>

<br>

Activity normalization → cerebral uptake region detection → tumor hotspot isolation via
tumor-to-background ratio with head-erosion to exclude scalp uptake → maximum-intensity
projections → 3D tumor mesh.

</details>

---

## Accuracy & Validation

`validate.py` scores any segmentation against a reference on an identical voxel grid, reporting
**Dice, IoU, Hausdorff-95 (mm), and Average Surface Distance (mm)**.

**Measured result — heuristic engine vs. TotalSegmentator (reference), same chest CT:**

| Structure | Dice | IoU | HD95 | ASSD | Interpretation |
|---|---|---|---|---|---|
| Lungs | **0.95** | 0.91 | 22 mm | 2.5 mm | Heuristic ≈ AI — air-filled regions separate cleanly by intensity |
| Skeleton | **0.11** | 0.06 | 113 mm | 38 mm | Heuristic ≪ AI — intensity thresholding alone cannot recover full bone structure |

This is the empirical basis for the dual-engine design: the fast path is trustworthy where the
physics makes the problem easy, and the deployment should route to the AI engine where it
doesn't. Metrics are unit-tested for correctness (identical-mask, disjoint-mask, and
voxel-spacing-scaling cases).

```bash
python validate.py --dicom <ct_dicoms> \
                   --totalseg output/<dataset>/segmentations \
                   --out output/<dataset>/validation
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| Job queue | In-process `ThreadPoolExecutor` (default) or Redis + RQ (production) |
| Persistence | SQLite (job state), filesystem (imaging outputs) |
| Imaging | pydicom, SimpleITK, scikit-image, scikit-learn, nibabel, brainextractor |
| AI segmentation | TotalSegmentator (nnU-Net), MedSAM (segment-anything), PyTorch |
| Meshing / rendering | PyVista, VTK, matplotlib |
| Frontend | Vanilla JS, three.js (WebGL 3D viewer) |
| Neuroplasticity Explorer | Anthropic Claude (structured JSON, schema-constrained) with a curated fallback knowledge base |
| Deployment | Docker, Docker Compose, Hugging Face Spaces, PyInstaller |
| Testing | pytest, GitHub Actions |

---

## Repository Structure

```
MEDSYS/
├── app.py                     # FastAPI web service
├── segmentation_pipeline.py   # Core modality-aware segmentation pipeline
├── brain_knowledge.py         # Neuroplasticity Explorer knowledge base + Claude integration
├── validate.py                # Accuracy validation (Dice / IoU / HD95 / ASSD)
├── config.py                  # Environment-driven configuration
├── jobstore.py                # SQLite job persistence
├── tasks.py                   # Job execution (shared by in-process pool and RQ worker)
├── worker.py                  # RQ worker entry point
├── entry.py / launcher.py     # Standalone .exe entry point and desktop launcher
├── medsys.spec                # PyInstaller build spec
├── web/                       # Frontend (index.html, brain.html)
├── tests/                     # Unit tests
├── Dockerfile / Dockerfile.web / docker-compose.yml
└── requirements*.txt          # Full / web / build dependency sets
```

---

## Requirements

- Python 3.10+
- See `requirements-full.txt` (complete pipeline, all engines) or `requirements-web.txt`
  (lightweight — no torch/TotalSegmentator/Redis, matches the hosted demo)
- Docker (optional, for containerized deployment)
- CUDA-capable GPU (optional — accelerates MedSAM and TotalSegmentator; both fall back to CPU)

---

## Testing

```bash
pip install -r requirements.txt pytest
pytest
```

Unit tests cover preprocessing, post-processing, 3D connected-component handling, and
validation-metric correctness using synthetic data — no DICOM files or model checkpoints
required. CI runs on every push via GitHub Actions.

---

## Roadmap

- [ ] Atlas-based priors to correct GM/WM classification on non-standard MRI contrasts
- [ ] Dice/IoU validation against public ground-truth datasets
- [ ] Multi-series fusion (T1 + T2 + DTI)
- [ ] Direct Unity/Unreal VR scene export
- [ ] macOS/Linux standalone builds

---

## Contributing

Issues and pull requests are welcome. Please:

1. Run `pytest` and ensure it passes before submitting
2. Follow the existing modality-routing pattern when adding a new imaging modality or engine
3. Never commit patient data, DICOM files, or model checkpoints (`.gitignore` already excludes these)

---

## Disclaimer

MEDSYS is provided for **educational and research purposes only**. It is not a medical device,
has not been evaluated by any regulatory body, and must not be used to diagnose, treat, or
otherwise make clinical decisions about any patient. Segmentation outputs — from either engine
— are not validated for clinical accuracy and may contain errors. Always defer to qualified
medical professionals and validated clinical software for any healthcare decision.

---

## License

MIT — see [LICENSE](LICENSE).

## Contributors

- **[Organic42](https://github.com/Organic42)** — Project maintainer

<div align="center">
<sub>Built with FastAPI, PyTorch, TotalSegmentator, PyVista, and three.js.</sub>
</div>
