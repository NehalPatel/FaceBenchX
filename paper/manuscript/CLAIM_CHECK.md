# Claim ↔ Evidence Cross-Check

Manuscript: `paper/manuscript/FaceBench-LFW-BuffaloL-Draft.md`  
Experiment: `20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c`

| # | Manuscript claim | Evidence artifact | Verdict |
|---|------------------|-------------------|---------|
| 1 | Accuracy = 0.9817 | `manifest.json` → recognition_metrics.accuracy | Pass |
| 2 | AUC = 0.9892; EER = 0.0238 | same | Pass |
| 3 | Precision = 1.0; Recall = 0.9635; F1 = 0.9814 | same | Pass |
| 4 | FAR = 0; FRR = 0.0365 at cosine 0.40 | confusion FP=0, FN=109 | Pass |
| 5 | TP=2876, FN=109, FP=0, TN=2973 | confusion + Table 3 | Pass |
| 6 | 5958 scored / 6000 protocol; 42 skipped | dataset block + summary | Pass |
| 7 | 25 unique failed images | detection.failed_images | Pass |
| 8 | Wilson / Hanley–McNeil 95% CIs | `paper/tables/publication_tables.md` | Pass |
| 9 | Avg embed 0.5585 s; 1.79 FPS; load 2.95 s | computational_metrics | Pass |
| 10 | CPU-only; GPU null | environment.gpu / ORT providers | Pass |
| 11 | Weight SHA-256 for w600k_r50.onnx | manifest + checksums.sha256 | Pass |
| 12 | HTML/MD/CSV/JSON/ROC/confusion exist | execution summary artifact list | Pass |
| 13 | Wall time ≈ 1.87 h | execution_time_s = 6732.4 | Pass |
| 14 | “Best model overall” | — | **Not claimed** |
| 15 | Shared identical align for all models | — | **Not claimed** (Baseline A) |
| 16 | Real-time on edge GPU | — | **Not claimed** |
| 17 | Robust across pose/age/etc. | — | **Not claimed** |

## Residual risks for submission

1. Protocol file obtained via mirror after UMass DNS failure — pin checksum before journal submission.
2. Skipped pairs may bias difficulty distribution.
3. Default threshold 0.40 ≠ EER threshold 0.1219 — discuss operating-point choice.
4. Expand references before external submission.
