# FaceBench: A Reproducible Benchmarking Framework for Deep Face Recognition — Initial Validation on LFW with InsightFace Buffalo-L

**Manuscript status:** Draft for internal review (single-experiment validation)  
**Corresponding experiment:** `20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c`  
**Framework version:** FaceBench 0.1.0  
**Date:** 2026-08-06

> **Scope note.** This draft reports the FaceBench software contribution and a **single** locked empirical validation (LFW × Buffalo-L, Baseline A). Multi-model and multi-dataset comparisons required for the full research questions (RQ1–RQ5) are **not** claimed here. Claims below are cross-checked against the experiment manifest and artifacts under `paper/`.

---

## Abstract

Comparative evaluation of face recognition models is often performed with ad hoc scripts, accuracy-only reporting, and incomplete provenance, which limits reproducibility and fair comparison. We present **FaceBench**, a modular Python framework that unifies dataset adapters, model adapters, verification/identification protocols, recognition and computational metrics, and automated publication-oriented reporting under YAML-driven experiment management. As an initial scientific validation, we execute a locked LFW View-2 verification experiment with InsightFace **Buffalo-L** using cosine similarity at the framework default threshold of 0.40. On 5958 of 6000 protocol pairs (42 pairs skipped due to failed face detection), Buffalo-L achieved accuracy 0.9817 (95% Wilson CI 0.9780–0.9848), AUC 0.9892 (95% Hanley–McNeil CI 0.9865–0.9918), and EER 0.0238, with zero false accepts at the operating threshold. Mean embedding time on CPU was 0.5585 s/image (1.79 FPS). We document hardware, weight checksums, skipped detections, and all report/figure artifacts. The study demonstrates end-to-end reproducibility of FaceBench on a canonical public benchmark and establishes Baseline A prior to shared-alignment multi-model comparisons.

**Keywords:** face recognition, benchmarking, reproducibility, LFW, InsightFace, Buffalo-L, verification metrics

---

## 1. Introduction

Deep face recognition systems are routinely compared in the literature, yet experimental practice frequently suffers from (i) accuracy-only reporting, (ii) inconsistent preprocessing and matching protocols across models, (iii) missing hardware and software provenance, and (iv) one-off scripts that cannot be reused. FaceBench addresses these gaps by providing an open, YAML-configurable benchmarking library that is independent of any application stack (including surveillance systems such as VISTA).

This paper makes two contributions at the present stage of the project:

1. **Software:** FaceBench — a Strategy/Factory architecture covering eight public datasets, five recognizer adapters, multi-axis metrics, robustness/scalability protocols, CLI orchestration, and automated HTML/Markdown/CSV/JSON reports with figures.
2. **Empirical baseline:** A fully logged LFW verification run with Buffalo-L (Baseline A: vendor-bundled detection/alignment inside InsightFace), including confidence intervals and complete artifact provenance.

We explicitly do **not** yet answer which model is best across datasets (RQ1–RQ5 in the FaceBench design). Those questions require frozen multi-model, multi-dataset runs under a shared alignment protocol (Baseline B), which are planned next.

---

## 2. Related Work and Gap

Prior comparative studies and toolkits often emphasize verification accuracy or TAR@FAR on a narrow set of models, with limited computational profiling and weak reproducibility metadata. FaceBench targets the combination of:

- recognition metrics (accuracy, precision, recall, F1, ROC/AUC, FAR/FRR, EER);
- computational metrics (load time, embedding latency, throughput, CPU/RAM, model size);
- category-organized public datasets (pose, age, low resolution, occlusion, surveillance, video);
- experiment IDs, environment snapshots, and (in this study) paper manifests with weight checksums.

Dataset policy is public-only with local paths; FaceBench neither bundles nor auto-downloads copyrighted image collections.

---

## 3. FaceBench Framework (Methods — Software)

### 3.1 Architecture

FaceBench organizes evaluation as: **config → dataset integrity → pairs/gallery → embed → match → metrics → reports/figures**. Core abstractions are `BaseDataset` and `BaseRecognizer`, constructed via factories. Matching supports cosine and Euclidean similarity. Evaluation modes include verification and identification, with optional synthetic robustness and gallery-size scalability ladders.

### 3.2 Models and datasets (v1 capability)

| Models | FaceNet, Dlib, Buffalo-L, AdaFace, MagFace |
| Datasets | LFW, CFP-FP, CPLFW, AgeDB-30, TinyFace, AR Face, ChokePoint, YTF |

Optional heavy dependencies are isolated behind extras (`[buffalo]`, `[facenet]`, etc.). Deterministic stubs exist for CI only and are forbidden in paper runs.

### 3.3 Reporting

Each run emits Markdown/HTML reports, CSV/JSON metrics, ROC and confusion figures, and environment metadata. The paper suite under `paper/` stores locked configs, hardware profiles, weight checksums, and curated results.

---

## 4. Experimental Protocol (Methods — Validation Study)

### 4.1 Design

| Factor | Setting |
|--------|---------|
| Dataset | LFW (deepfunneled images) |
| Protocol | View-2 `pairs.txt` (10 folds; 3000 same + 3000 different) |
| Model | InsightFace Buffalo-L (`buffalo_l`) |
| Matching | Cosine similarity |
| Threshold | 0.40 (FaceBench default) |
| Device | CPU (ONNX Runtime `CPUExecutionProvider`) |
| Seed | 42 |
| Alignment | Baseline A — detector bundled in InsightFace `FaceAnalysis` |
| Stub backends | Disabled |

### 4.2 Data preparation

The local LFW layout used nested deepfunneled folders. The effective image root was:

`D:/datasets/lfw/lfw-deepfunneled/lfw-deepfunneled`

Official UMass hosting for `pairs.txt` was unreachable from the experimental host (DNS failure). The View-2 protocol file was obtained from a public facenet repository mirror and the header was normalized from `10\\t300` to `10` to match FaceBench’s LFW parser. Integrity validation passed; 6000 pairs resolved with existing image files.

### 4.3 Weights and provenance

Buffalo-L weights were loaded from `C:/Users/Nehal/.insightface` (InsightFace root containing `models/buffalo_l/`). Recognition ONNX checksum:

`w600k_r50.onnx` SHA-256 = `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`

### 4.4 Detection failures and pair skipping

InsightFace returned “no faces” on 25 unique images, causing 42 protocol pairs to be unscorable. Those pairs were **skipped and logged** (not imputed). All reported metrics are conditioned on the 5958 scored pairs. This policy is documented in the experiment notes and should be stated in any manuscript using these numbers.

### 4.5 Statistical analysis

For binomial metrics we report Wilson score 95% confidence intervals from confusion counts. For AUC we report a Hanley–McNeil approximate 95% CI. Fold-wise mean±SD, bootstrap CIs on raw scores, and McNemar tests against other models require additional locked runs and/or persisted per-pair score files; they are **out of scope** for this single-model draft.

### 4.6 Implementation note

An initial call through the stock orchestrator aborted on the first undetectable face. The accepted paper run used `paper/reproduction/run_lfw_buffalo_l.py`, which composes FaceBench components with embedding caching and skip-on-missed-detection. No new recognizers or datasets were added.

---

## 5. Results

### 5.1 Recognition performance

On 5958 scored LFW pairs (2985 positive, 2973 negative):

| Metric | Estimate | 95% CI |
|--------|---------:|--------|
| Accuracy | 0.9817 | 0.9780–0.9848 (Wilson) |
| Precision | 1.0000 | 0.9987–1.0000 (Wilson) |
| Recall | 0.9635 | 0.9561–0.9696 (Wilson) |
| F1 | 0.9814 | — |
| AUC | 0.9892 | 0.9865–0.9918 (Hanley–McNeil) |
| EER | 0.0238 | — |
| FAR @ 0.40 | 0.0000 | 0.0000–0.0013 (Wilson) |
| FRR @ 0.40 | 0.0365 | 0.0304–0.0439 (Wilson) |

At cosine threshold 0.40, the confusion matrix was TP=2876, FN=109, FP=0, TN=2973. The operating point is highly conservative on false accepts (FAR = 0) at the cost of 109 false rejects. The EER operating threshold estimated from the ROC was 0.1219, indicating that the default 0.40 threshold is substantially stricter than the equal-error point for this score distribution.

### 5.2 Computational performance (CPU)

| Metric | Value |
|--------|------:|
| Model load time | 2.95 s |
| Avg. embedding time | 0.5585 s |
| Throughput | 1.79 FPS |
| CPU utilization | 254.2% (multi-threaded ORT) |
| Peak RSS | 627.0 MiB |
| On-disk model pack size | 600.7 MiB |
| Wall-clock experiment time | 1.87 h |

### 5.3 Artifacts

All acceptance artifacts were generated and validated: config snapshot, `env.json`, `manifest.json`, summary JSON, HTML/Markdown reports, metrics CSV/JSON, ROC curve, confusion matrix, and PR curve. Figures are archived under `paper/figures/` (`fig2`–`fig6`).

---

## 6. Discussion

### 6.1 What this experiment shows

1. FaceBench can drive a real pretrained model on a real public protocol end-to-end, with metrics, figures, and provenance suitable for supplementary material.
2. Buffalo-L on deepfunneled LFW yields strong verification metrics under a strict cosine threshold, with no false accepts among scored negatives at 0.40.
3. Detection failures are non-negligible (0.7% of pairs) and must be reported; silent crashes are unacceptable for paper pipelines.

### 6.2 Limitations (critical)

1. **Single model, single dataset.** No comparative ranking is possible; RQ1–RQ5 remain open.
2. **Baseline A alignment.** Vendor detection inside Buffalo-L confounds recognition with detector quality; shared RetinaFace alignment (Baseline B) is required for fair multi-model claims.
3. **CPU-only runtime.** Latency/FPS are not representative of GPU deployment; RQ2/RQ5 need CUDA runs.
4. **Protocol acquisition.** `pairs.txt` came from a mirror after DNS failure to UMass; future runs should pin a checksummed protocol file in the paper suite.
5. **No fold-wise mean±SD / McNemar in this draft.** Per-pair scores with fold IDs were not retained in the accepted artifact set in NPZ form for this run; statistical comparison modules exist in FaceBench but await multi-model score matrices.
6. **Skipped pairs** may slightly bias metrics if failures correlate with hard cases.

### 6.3 Implications for the full FaceBench study

This Baseline A run is a gate: it validates tooling before expanding to FaceNet, Dlib, AdaFace, and MagFace, and before pose/age/low-res category datasets. Methods text for the eventual multi-model paper should separate Baseline A (vendor align) from Baseline B (shared align).

---

## 7. Reproducibility

To reproduce this experiment:

1. Install `pip install -e ".[buffalo,reports,dev]"`.
2. Place deepfunneled LFW images and View-2 `pairs.txt` as documented in `docs/datasets/lfw.md`.
3. Place InsightFace `buffalo_l` under an InsightFace root; verify SHA-256 of `w600k_r50.onnx`.
4. Copy `paper/configs/lfw_buffalo_l.yaml` to `paper/configs/local/` and set absolute paths.
5. Run `python paper/reproduction/run_lfw_buffalo_l.py`.
6. Confirm all items in `paper/reproduction/acceptance_checklist.md`.

Hardware for this run: Windows 10 host `DESKTOP-NHQRICJ`, Intel CPU (Family 6 Model 151), no GPU used, Python 3.10.11, insightface 1.0.1, onnxruntime 1.23.2. See `paper/hardware/desktop_nhqricj.md`.

---

## 8. Conclusions

FaceBench provides a reusable, YAML-driven infrastructure for multi-axis face recognition benchmarking. An initial locked LFW × Buffalo-L experiment demonstrates that the pipeline produces complete scientific artifacts with strong verification metrics (accuracy 0.9817, AUC 0.9892, EER 0.0238) under documented constraints (CPU, vendor alignment, 42 skipped pairs). The next steps toward a full comparative paper are: (i) shared RetinaFace alignment, (ii) five-model LFW comparison with paired significance tests, and (iii) multi-dataset category evaluation.

---

## Acknowledgments

Datasets and model weights remain the property of their respective providers. FaceBench does not redistribute LFW images or InsightFace weights.

---

## References (seed list — expand for submission)

1. Huang et al., Labeled Faces in the Wild, Tech Report, UMass Amherst.
2. Deng et al., ArcFace / InsightFace model zoo (Buffalo-L).
3. FaceBench Design Document, 2026 (`docs/FaceBench-Design.md`).

---

## Appendix A — Claim ↔ Evidence Cross-Check

| Claim in manuscript | Evidence | Status |
|---------------------|----------|--------|
| Accuracy 0.9817 on scored LFW pairs | `manifest.json` recognition_metrics | Verified |
| 5958/6000 pairs scored; 42 skipped | manifest dataset + execution summary | Verified |
| FAR=0, FRR=0.0365 at threshold 0.40 | confusion TP/TN/FP/FN | Verified |
| AUC 0.9892; EER 0.0238 | recognition_metrics | Verified |
| Wilson / Hanley–McNeil CIs | `paper/tables/publication_tables.md` | Verified |
| CPU-only; no GPU | env/manifest `gpu: null`, ORT providers | Verified |
| Weight checksum recorded | manifest + `paper/weights/checksums.sha256` | Verified |
| Reports/figures generated | acceptance artifact list | Verified |
| Multi-model superiority | — | **Not claimed** |
| Shared identical align across models | — | **Not claimed** (Baseline A) |
| Real-time suitability | 1.79 FPS CPU | Discussed as CPU-bound only |

---

## Appendix B — Figure captions

- **Figure 2.** ROC curve for Buffalo-L on LFW View-2 (scored pairs). File: `paper/figures/fig2_roc_curve.png`
- **Figure 3.** Confusion matrix at cosine threshold 0.40. File: `paper/figures/fig3_confusion_matrix.png`
- **Figure 4.** Precision–recall curve. File: `paper/figures/fig4_pr_curve.png`
- **Figure 5.** Resource usage summary. File: `paper/figures/fig5_resource_usage.png`
- **Figure 6.** Embedding-time bar. File: `paper/figures/fig6_inference_time.png`

---

*End of draft. Ready for co-author review before expansion to five-model / multi-dataset experiments.*
