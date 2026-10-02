# MEDSYS — literature & commercial gap analysis (first pass)

> **Superseded in part.** After external review, the thesis was restructured as risk-aware reconstruction (measure -> predict -> decide), with G1-G3 merged into one research problem and G4/G5 moved to methodology and engineering. See `gap-analysis-two-pager.tex` / `.txt` (v3.1, CT-first, RQ1 as the core thesis). This file stays as the evidence log.

Prepared 2026-10-02. **Method and limits:** ~30 web searches plus a handful of primary-source fetches.
This is a scoping scan, NOT a systematic review. Every claim carries a confidence tag:

- **[P]** read on a primary source this session
- **[S]** from a search-result summary or secondary source; treat as a lead, verify before citing
- **[U]** unverified / my inference

"Not found in this pass" never means "does not exist". Phase 0 below is the real novelty check.

## 1. Retraction of the earlier thesis

I earlier proposed a "cost-aware hybrid" paper (cheap classical engine vs deep learning, routed per structure).
The evidence weakens the *cost* argument:

- TotalSegmentator ships `--fast` (3 mm), `--fastest` and `--roi_subset` ("saves a lot of runtime and memory") **[P]**
  https://github.com/wasserth/TotalSegmentator . Reported runtimes (about 20-30 s in fast mode on GPU) are **[S]**; the README's runtime table is an image, so I could not confirm them. Measure on the RTX 5080.
- DL skull-stripping has largely closed the speed gap: mean times roughly BET 2 s, SynthStrip 7 s, HD-BET 132 s **[S]** https://arxiv.org/pdf/2308.07003 ; HD-BET beat BET on Dice across datasets (median +1.33 to +2.63 points) **[S]** https://github.com/MIC-DKFZ/HD-BET
- Prior art exists for routing by difficulty: cloud-edge routing for ultrasound https://www.nature.com/articles/s41598-026-59234-y **[S]**; AutoPath, image-specific 3D inference https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7393252/ **[S]**; DeferredSeg, learning-to-defer to human experts https://arxiv.org/pdf/2604.12411 **[S]**; selective prediction for segmentation https://arxiv.org/pdf/2402.10665 **[S]**.
- **Kill criterion:** if TS `--fast` on the GPU takes < ~15 s and the classical engine saves < 10x energy per scan, drop the cost framing.

## 2. Research gaps (ranked by fit: 3 months, public data, one GPU)

| # | Gap | Evidence it is open | Risk |
|---|-----|---------------------|------|
| G1 | **Stage-wise error budget for automatic DICOM -> 3D mesh**: split final mesh error into engine, voxel spacing / partial volume, isosurface, smoothing, decimation; report calibrated +/- mm per structure and engine | 3D-printing literature decomposes error into segmentation / digital-editing / printing and finds segmentation error dominates **[S]** https://www.researchgate.net/publication/379332555_Quality_assurance_of_3D-printed_patient_specific_anatomical_models_a_systematic_review ; threshold and smoothing change dimensions **[S]** https://link.springer.com/article/10.1007/s10278-024-00998-y ; smoothing is known to degrade accuracy **[S]**; uncertainty through marching cubes exists in visualisation work **[S]** https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14333 . Not found: a study attributing mesh error stage by stage for automatic DL-vs-classical pipelines over many structures on public ground truth **[U]** | Novelty unconfirmed; must pass Phase 0 |
| G2 | **Label-free reliability gate**: predict per-structure failure of a cheap engine from interpretable features (HU histogram, topology, volume plausibility), then defer or flag | QC without ground truth is active: https://arxiv.org/pdf/2407.13307 , https://arxiv.org/pdf/2503.04522 , https://arxiv.org/pdf/2508.01460 **[S]**; uncertainty proxies often correlate weakly with Dice **[S]**. Crowded | Overlaps heavily; fold into G1 as a second contribution |
| G3 | **Per-scan inference energy and cost** across classical, TS (full / fast / ROI) and one foundation model, CPU vs GPU | Training carbon studied https://arxiv.org/pdf/2203.02202 ; inference energy studied for small models https://doi.org/10.3390/jimaging11060174 **[S]**. Whole-pipeline per-scan energy for deployed whole-body tools not found **[U]** | Modest novelty; useful as supporting evidence |
| G4 | **Pathology failure atlas**: where thresholds and DL each fail (consolidation, effusion, implants) | My search returned mostly DL COVID papers, so evidence that thresholds fail is thin **[U]** | Hypothesis only |
| G5 | **Evaluation rigor**: choose metrics with Metrics Reloaded, report per-case results with CIs, not only mean Dice | https://www.nature.com/articles/s41592-023-02151-z **[S]**; Dice cannot separate over- from under-segmentation **[S]** | A requirement, not a contribution |
| G6 | Browser-side private inference with WebGPU for 3D segmentation | Precedent for lung-screening AI in the browser https://pmc.ncbi.nlm.nih.gov/articles/PMC10099365 **[S]** | Heavy build; park |
| G7 | Verified de-identification gate | Tag-only tools miss burned-in text and nested sequences **[S]**; synthetic evaluation data exists https://arxiv.org/pdf/2508.01889 and the MIDI-B challenge https://pith.science/paper/2509.00437 **[S]** | Engineering, small paper at best; needed regardless |
| G8 | Foundation models under compute budgets | VISTA3D / SAM-Med3D strong on 3D CT but heavier **[S]** https://arxiv.org/pdf/2602.07643 | Fast-moving and crowded; skip |

**Product gap, not research:** the MRI route is BET followed by box-prompted MedSAM refinement of the BET mask (default when the checkpoint is present; skipped on the hosted build), then GMM tissue classes. An earlier version of this line omitted MedSAM. The route has not been benchmarked against SynthStrip / HD-BET; do that before making any MRI claim.

**Baseline gap:** the current "validation" uses TotalSegmentator as ground truth. A paper needs expert-annotated datasets.

## 3. Commercial gaps (hypotheses only; none validated with customers)

| # | Observation | Confidence |
|---|-------------|-----------|
| C1 | Segmentation for 3D printing is held by cleared incumbents (Mimics, Simpleware, Axial3D, RICOH 3D); diagnostic-use models are expected to come from cleared software https://pmc.ncbi.nlm.nih.gov/articles/PMC10080800/ , https://www.synopsys.com/simpleware/clinical-applications/poc-3d-printing.html . An uncleared tool cannot enter that lane; teaching, research and non-diagnostic use remain | [S] |
| C2 | Orchestration is crowded by open source (MONAI Deploy, Mercure, Orthanc, Kaapana) https://monai.io/ ; integrate (DICOMweb, DICOM-SEG), do not rebuild | [S] |
| C3 | **A licence-clean stack is a differentiator.** TotalSegmentator `total` and `total_mr` are Apache-2.0; many subtasks (heart chambers, appendicular bones, tissue types, brain structures, face, muscles, coronary arteries) need a paid commercial licence **[P]**. Datasets: TS v2 CC-BY-4.0 https://zenodo.org/records/10047292 **[S]**; MSD CC-BY-SA **[S]**; AMOS and FLARE reported as non-commercial, sources conflict **[S]**. Audit `brainextractor`, BET lineage and other dependencies **[U]** | mixed |
| C4 | EU: high-risk AI Act duties for AI in regulated products (including MDR devices) moved to 2 Aug 2028; stand-alone Annex III to 2 Dec 2027 **[P-domain, via search snippet]** https://www.consilium.europa.eu/en/press/press-releases/2026/06/29/artificial-intelligence-council-gives-final-green-light-to-simplify-and-streamline-rules/ . MDR obligations are unchanged **[S]**. US: the FDA list of AI-enabled devices (entries through June 2026) is dominated by radiology **[P]**; the page loaded truncated, so I have no reliable total count, and an earlier "over 800" figure in this doc was withdrawn http://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices ; a Class II classification for radiological ML quantitative imaging software appears to be codified at 21 CFR 892.2055 **[U: could not load the rule]** https://www.federalregister.gov/documents/2026/06/17/2026-12166/medical-devices-radiology-devices-classification-of-the-radiological-machine-learning-based ; PCCP guidance exists. A student project can produce documentation, not clearance | mixed |
| C5 | Hospitals will want *verified* anonymisation (see G7); your own pipeline currently names outputs from the zip filename and scrubs nothing | [P] code audit |
| C6 | VR planning has clinical evidence and vendors https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11043584/ , https://pmc.ncbi.nlm.nih.gov/articles/PMC12560611/ ; an affordable automatic scan -> WebXR/glTF path for teaching is an untested niche | [U] |
| C7 | CPU-only / on-prem sites: plausible, but weakened by TS `--fast` on CPU; needs measured numbers | [U] |

## 4. Recommended direction

Lead with **G1** (stage-wise mesh error budget + calibrated mm uncertainty), with **G2** as the gate that uses it and **G3/G5** as the evaluation spine. It matches the platform's real output (meshes), is measurable on public data, and answers the original "accuracy and precision" question.

## 5. Phase 0 (week 1): systematic novelty check

- Databases: PubMed, IEEE Xplore, arXiv, Semantic Scholar, Google Scholar, MICCAI / ISBI / MIDL proceedings.
- Queries (combine): "mesh accuracy" / "geometric fidelity" x "automatic segmentation" x "marching cubes" / "smoothing" / "decimation"; "error propagation" x "segmentation to 3D model"; "failure prediction" x "cheap model" x "segmentation"; "TotalSegmentator" x "mesh" / "3D printing".
- Record every query, hit count and decision in a log. Citation-chase the 10 closest papers forward and backward.
- **Pivot rule:** if an existing paper already does an automatic-pipeline, stage-wise mesh error budget across many structures, switch to G2/G3.

## 6. Hospital data (you can approach hospitals)

Use it for (a) external validation and (b) 15-20 clinician interviews to test C1/C6. Ethics approval and data-use agreements take months, so do not put the 3-month paper on its critical path: start the paperwork now, treat the data as a bonus, and de-identify before it touches the platform.
