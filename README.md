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

# MEDSYS

A **modality-aware** medical-image segmentation pipeline that turns raw DICOM scans into VR-ready 3D meshes and volumetric analysis. It auto-detects scan type and routes to the right workflow — each dataset writes all images and 3D models into its own `output/<dataset_name>/` folder.

> **Hosted demo:** this repo deploys as-is on [Hugging Face Spaces](https://huggingface.co/new-space)
> (Docker SDK) using `Dockerfile.web` — a lightweight build (no torch/TotalSegmentator/Redis) sized
> for the free CPU tier. See **Hosting** below. The YAML block above is Spaces config; it's ignored
> everywhere else.

Two engines:
- **Heuristic** (default, no GPU needed) — fast classical pipelines per modality
- **TotalSegmentator** (`--engine totalseg`) — nnU-Net deep learning, **100+ individually-labeled
  structures** across the whole body (CT & MR). On the sample chest CT it produced **79 structures
  in ~83 s on CPU** (every lung lobe, ribs, vertebrae, heart, liver, spleen, vessels…).

| Modality | Heuristic pipeline |
|----------|----------|
| **MRI brain** (T1/T2) | N4 bias → NLM denoise → BET skull strip → MedSAM refine → GM/WM/CSF classification → hybrid 3D surface + per-tissue meshes |
| **CT chest** | HU conversion → windowing → lung / skeleton / body / soft-tissue segmentation → colored 3D meshes |
| **PET brain** (FET/FDG) | Activity normalization → cerebral-region detection → tumor hotspot by tumor-to-background ratio → MIP views + 3D tumor mesh |
| **Any CT / MR** (`--engine totalseg`) | DICOM→NIfTI → TotalSegmentator (nnU-Net) → per-structure 3D meshes (GPU recommended) |

The MRI branch auto-detects **T1 vs T2** from the series description and orders GM/WM/CSF
accordingly (CSF is dark on T1, bright on T2). For thin or highly anisotropic volumes
(few slices / thick slices) it skips BET — whose surface-deformation model is unstable
there — and uses an intensity-based skull strip instead.

---

## Web UI (drag-and-drop)

A browser front-end for non-technical users — drag a DICOM `.zip`, watch the live
pipeline log, then browse the segmented images and an interactive **3D viewer**
(three.js) with per-structure toggles.

```bash
pip install fastapi uvicorn python-multipart
python app.py
# open http://127.0.0.1:8000
```

- **Backend** (`app.py`, FastAPI): `POST /api/upload` extracts the zip and runs the
  pipeline in a background thread; `GET /api/jobs/{id}` streams status/log;
  `GET /api/datasets` lists results. Output images and `.stl` meshes are served statically.
- **Frontend** (`web/index.html`): drag-drop upload, modality override, live job log,
  per-dataset gallery + rotatable 3D model viewer with structure show/hide.

---

## Deployment

The service is containerized and scales API and workers independently.

```bash
docker compose up --build           # API + Redis + worker
docker compose up --scale worker=3  # more concurrent jobs
docker compose --profile gpu up     # GPU worker (needs NVIDIA Container Toolkit)
```

**Architecture:**

```
Browser ── POST /api/upload ──▶ API (FastAPI) ──enqueue──▶ Redis queue
                                     │                          │
                                GET /api/jobs/{id}          Worker(s) pull job
                                     │                          │ run pipeline (GPU-capable)
                                     ▼                          ▼
                              SQLite job store ◀── status ── shared /data volume
                                                              (output + uploads)
```

**Hardening built in:**

| Concern | Solution |
|---------|----------|
| Jobs lost on restart | **SQLite job store** (`jobstore.py`) — survives restarts, shared by API + workers |
| Unbounded concurrency / OOM | Bounded **`ThreadPoolExecutor`** (dev) or **Redis + RQ** queue (prod); `VRSEG_MAX_WORKERS` |
| Scale-out | Stateless API + N worker containers sharing Redis and a `/data` volume |
| Config & secrets | All via env (`config.py`): `REDIS_URL`, `VRSEG_*` paths, limits, timeouts |
| Liveness / readiness | `GET /health` and `GET /ready` (checks DB + Redis) for orchestrators |
| Upload safety | Streamed to disk with a `VRSEG_MAX_UPLOAD_MB` cap; zip validated |
| Housekeeping | `VRSEG_JOB_RETENTION_DAYS` prunes old job rows |

**Local dev** (no Redis needed): `python app.py` uses the in-process bounded pool and the
same SQLite store. Setting `REDIS_URL` switches to the queue path automatically.

### Free-tier hosting (Hugging Face Spaces)

For a public demo anyone can open in a browser, this repo ships `Dockerfile.web` +
`requirements-web.txt` — a trimmed build (no torch / TotalSegmentator / Redis) sized for
Hugging Face's free CPU tier (2 vCPU, 16 GB RAM, no card required):

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**, sign in (or create a
   free account).
2. **SDK: Docker** → template **"Blank"**. Name it (e.g. `medsys`). Keep it **Public**.
3. In **Settings → Space hardware**, confirm it's on the free **CPU basic** tier.
4. Link it to this GitHub repo: **Settings → Repository → "Sync from GitHub"**, point it at
   `Organic42/MEDSYS`, branch `main`. (Or push directly: `git remote add space
   https://huggingface.co/spaces/<you>/medsys && git push space main` using a Space access
   token as the password.)
5. The Space reads the YAML block at the top of this README automatically — it already
   points the build at `Dockerfile.web` and port `7860`. Build takes a few minutes; the
   Space is live at `https://huggingface.co/spaces/<you>/medsys` once it turns green.

What's different on this deployment: MedSAM refinement and the TotalSegmentator ("AI organs")
engine are unavailable (no checkpoint/torch shipped) — the UI detects this via
`GET /api/capabilities` and disables that option automatically. CT, MRI brain (BET + tissue
classification), and PET tumor pipelines all work fully. Storage is ephemeral: uploads and
results reset if the Space restarts, which is expected for a public test instance.

### Standalone Windows build (send it to a friend)

For testing without any hosting at all, MEDSYS also builds into a self-contained Windows
folder — no Python install required on the machine running it.

```bash
pip install -r requirements-web.txt -r requirements-build.txt
python -m PyInstaller medsys.spec --noconfirm
```

Output lands in `dist/MEDSYS/`. Zip that folder and send it — the recipient unzips it
anywhere and double-clicks `MEDSYS.exe`; it starts the server and opens the UI in their
browser automatically. Same lite feature set as the hosted demo (no MedSAM/TotalSegmentator).

**Why a `medsys.spec` file and an `entry.py` instead of just `pyinstaller app.py`:** the app
normally runs each segmentation job as a `python segmentation_pipeline.py ...` *subprocess* —
but a frozen `.exe` has no separate interpreter to hand a script to (`sys.executable` **is**
the exe). `entry.py` is the actual PyInstaller entry point; it re-invokes the exe itself with
a `--run-pipeline` flag when a job needs to run, and dispatches straight into
`segmentation_pipeline.main()` instead of starting the server. `config.py` resolves data
paths (output/uploads/job DB) relative to the exe's own folder when frozen, not the
PyInstaller bundle's internal resource directory, so results always land somewhere the user
can actually find them.

---

## Multi-Dataset Usage

```bash
# Auto-detects modality from the DICOM and writes to output/<name>/
python segmentation_pipeline.py --input "path/to/chest_ct_dicoms"  --name chest_ct
python segmentation_pipeline.py --input "path/to/brain_mri_dicoms" --name brain_mri

# Optional overrides
python segmentation_pipeline.py --input <dir> --name <ds> --modality CT
python segmentation_pipeline.py --input <dir> --name <ds> --no-medsam
```

Every run creates a self-contained folder:

```
output/chest_ct/
├── step1_windowed.png  step2_segmentation.png  step3_3d_anatomy.png  step3_lungs.png
├── report.json                         # volumes (cm³), HU range, spacing
├── mask_{lungs,skeleton,body,soft_tissue}.npy
└── {lungs,skeleton,body,soft_tissue}.{stl,obj,ply,vtk}
```

---

## Why Hybrid?

MedSAM and BET each win at different things, so the pipeline uses **both**:

| Task | Method | Reason |
|------|--------|--------|
| 2D per-slice masks | **MedSAM** (ViT-B) | Sharpest, tightest in-plane boundaries |
| Tissue classification | **MedSAM** mask + GMM | Tight brain region → less non-brain contamination |
| 3D VR surface | **BET** | True 3D method → smooth, no inter-slice stripe artifacts |

MedSAM is a 2D model: it segments each axial slice independently, so its masks jitter between slices and produce a striped 3D surface. BET deforms a single 3D surface, so its mesh is coherent. The hybrid keeps each where it's strongest.

---

## Pipeline Stages

> Both branches share DICOM loading (largest series by `SeriesInstanceUID`,
> array-axis-ordered voxel spacing) and the mesh/render helpers
> (marching cubes → Taubin smoothing → decimation → STL/OBJ/PLY/VTK export).

### MRI Brain Branch

### 1. DICOM Ingestion
- Loads the largest DICOM series, de-duplicating by `SeriesInstanceUID`
- Sorts slices by `ImagePositionPatient[z]`
- Typical target: `t1_mprage_fs_TRA_p2_iso_1.0` (192 axial slices, ~1 mm isotropic)

### 2. Medical-Grade Preprocessing
| Step | Operation | Purpose |
|------|-----------|---------|
| 2a | **N4 bias field correction** (SimpleITK) | Removes coil-proximity intensity inhomogeneity |
| 2b | **Non-local means denoising** (scikit-image) | Suppresses Rician noise, preserves edges |
| 2c | **Percentile normalization** | Rescales to `[0, 1]` |

### 3. Skull Stripping — BET
- Brain Extraction Tool (Smith 2002), pure-Python `brainextractor`
- Surface-deformation model produces a 3D-coherent brain mask

### 4. MedSAM Refinement
- MedSAM ViT-B with tight per-slice **box prompts** derived from the BET mask
- Box-prompt only (`multimask_output=False`) — MedSAM was trained exclusively on boxes
- Output intersected with the dilated BET mask to suppress any leakage outside the skull

### 5. Post-Processing
- Per-slice morphological cleanup (closing → fill holes → opening)
- 3D largest-connected-component to remove disconnected fragments

### 6. Tissue Classification — GM / WM / CSF
- 3-class **Gaussian Mixture Model** on T1 intensities inside the brain mask
- Classes ordered by mean intensity (CSF < GM < WM on T1)
- Size-based speckle removal (preserves the thin cortical GM ribbon)
- Outputs per-tissue volumes (cm³) and a GM/WM ratio in `brain_volumes.json`

### 7. 3D Surface Reconstruction
- Marching cubes on the BET mask → Taubin smoothing (feature-preserving) → 50% decimation → vertex normals
- Exports the brain surface + separate GM/WM/CSF meshes for layered VR

### CT Chest Branch

CT carries calibrated **Hounsfield Units**, so tissues separate by density — no bias correction or skull stripping needed.

| Stage | Operation |
|-------|-----------|
| 1. HU conversion | `HU = pixel × RescaleSlope + RescaleIntercept` |
| 2. Windowing | Clip to `[-1000, 400]` HU for display/normalization |
| 3. Body / skin | Per-slice fill of `HU > -500` + largest component (robust to airway-to-air connection) |
| 4. Lungs | Internal air (`HU < -320` inside body, `clear_border` to drop external air) → 2 largest components |
| 5. Skeleton | Bone `HU > 200` inside body + small-object removal |
| 6. Soft tissue | Inside body, excluding lung & bone, `-200 < HU < 200` |
| 7. 3D meshes | Colored surfaces: lungs, skeleton, body (translucent), soft tissue + cutaway anatomy render |

> **Axis-order note:** voxel spacing is stored as `(slice, row, col)` to match the
> NumPy volume `[z, y, x]`. This matters for anisotropic CT (e.g. 0.84 × 0.84 × 5.0 mm) —
> getting it wrong squashes the 3D reconstruction flat.

---

## Outputs

| File | Description |
|------|-------------|
| `step2_preprocessing.png` | Raw → N4 → denoised → normalized comparison |
| `step3_baseline_masks.png` | BET brain mask overlay |
| `step5_postprocessed.png` | MedSAM mask before/after post-processing |
| `step6_tissue_classification.png` | Color-coded GM/WM/CSF overlay |
| `step7_3d_reconstruction.png` | 4-view 3D brain surface render |
| `brain_masks.npy` | Boolean brain mask `(192, 256, 256)` |
| `tissue_map.npy` | Labeled volume (0=bg, 1=CSF, 2=GM, 3=WM) |
| `brain_volumes.json` | Tissue volumes (cm³), GM/WM ratio, normative reference |
| `brain_surface.{vtk,stl,ply,obj}` | Brain surface mesh (Unity/Unreal/MeshLab/3D-print) |
| `gray_matter.{stl,obj}` · `white_matter.{stl,obj}` · `csf.{stl,obj}` | Per-tissue VR layers |

### Example Volume Report
```json
{
  "tissue_volumes_cm3": { "CSF": 176.2, "GM": 391.3, "WM": 730.1 },
  "total_brain_volume_cm3": 1297.6,
  "gm_wm_ratio": 0.536,
  "normative_reference_cm3": {
    "total_brain": "1130–1500 (adult)",
    "gm_wm_ratio": "~1.1–1.3 (adult)"
  }
}
```
> **Note:** This dataset is a *fat-suppressed* MPRAGE with a bright-shifted intensity histogram, so a global GMM under-splits GM vs WM (ratio below the normative range). The report flags this transparently against reference values. Atlas-based priors would correct it.

---

## Setup

### Test/CI dependencies (lightweight)
```bash
pip install -r requirements.txt        # numpy, scipy, scikit-image
```

### Full pipeline dependencies
```bash
pip install -r requirements-full.txt   # + SimpleITK, brainextractor, torch,
                                        #   scikit-learn, pyvista, nibabel, etc.
```

### MedSAM checkpoint (not in repo — 375 MB)
```bash
pip install gdown
gdown 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O medsam_vit_b.pth
```
If the checkpoint is absent, the pipeline automatically falls back to BET-only masks.

---

## Usage

1. Place a DICOM series in `dicom_data/`
2. Set `TARGET_SERIES` and paths near the top of `segmentation_pipeline.py`
3. Run:
   ```bash
   python segmentation_pipeline.py
   ```
4. Inspect results in `output/`

> On CPU, the MedSAM pass runs the ViT-B encoder per slice (~10–20 min for ~150 slices). A CUDA GPU reduces this to under a minute.

---

## Accuracy Validation

`validate.py` scores any segmentation against a reference using **Dice, IoU,
Hausdorff-95 (mm), and Average Surface Distance (mm)**. Geometry is matched
voxel-exact (same NIfTI grid), so metrics are meaningful.

```bash
python validate.py --dicom <ct_dicoms> \
                   --totalseg output/<ds>/segmentations \
                   --out output/<ds>/validation
```

**Example — fast heuristic engine vs TotalSegmentator (reference) on the chest CT:**

| Structure | Dice | IoU | HD95 | ASSD | Verdict |
|-----------|------|-----|------|------|---------|
| **Lungs** | **0.95** | 0.91 | 22 mm | 2.5 mm | Heuristic ≈ DL — air regions are easy |
| **Skeleton** | **0.11** | 0.06 | 113 mm | 38 mm | Heuristic ≪ DL — HU>200 catches only cortical shells & non-bone high-HU |

The validation quantitatively justifies the engine split: the fast heuristic is fine
for lungs, but **bone/organ work needs TotalSegmentator**. An overlap map
(`validation_overlap.png`) visualizes agreement (green) vs each-only (red/blue).

Metrics (`compute_metrics`) are unit-tested (identical→1.0, disjoint→0.0,
spacing-scaling, empty-mask safety).

## Testing

```bash
pytest
```
Unit tests (`tests/test_pipeline.py`) cover preprocessing, post-processing, and 3D
fragment removal using synthetic arrays — no DICOM or model checkpoint required.

---

## Repository Structure

```
VR-segmentation/
├── segmentation_pipeline.py     # Full 7-step pipeline
├── tests/test_pipeline.py       # Unit tests (CI)
├── requirements.txt             # CI-only deps
├── requirements-full.txt        # Full pipeline deps
├── .github/workflows/           # GitHub Actions (lint + pytest)
└── README.md
```

> Patient DICOM data, model checkpoints (`*.pth`), and large mesh/volume outputs
> are git-ignored. DICOM files contain PHI and must never be committed.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| DICOM I/O | pydicom |
| Bias correction | SimpleITK (N4ITK) |
| Denoising | scikit-image (NLM) |
| Skull stripping | brainextractor (BET) |
| Segmentation | MedSAM / segment-anything (PyTorch) |
| Tissue classification | scikit-learn (GMM) |
| Meshing & rendering | scikit-image (marching cubes) + PyVista |
| VR export | STL / OBJ / PLY / VTK |

---

## Roadmap

- [ ] Atlas-based priors to correct GM/WM split on fat-suppressed contrasts
- [ ] Dice / IoU validation against ground-truth labels
- [ ] Multi-series fusion (T1 + T2 + DTI)
- [ ] Direct Unity/Unreal VR scene integration

---

## License

MIT — see [LICENSE](LICENSE).
