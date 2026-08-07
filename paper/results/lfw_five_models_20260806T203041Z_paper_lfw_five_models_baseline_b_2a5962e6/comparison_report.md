# LFW × Five Models — Baseline B Comparison Report

- **Status:** SUCCESS
- **Experiment ID:** `20260806T203041Z_paper_lfw_five_models_baseline_b_2a5962e6`
- **Output dir:** `D:\ModelComparision\experiments\20260806T203041Z_paper_lfw_five_models_baseline_b_2a5962e6`
- **Started (UTC):** 2026-08-06T20:30:41.302777+00:00
- **Ended (UTC):** 2026-08-06T20:37:23.829172+00:00
- **Elapsed:** 402.5 s (0.11 h)
- **Device:** `cuda:0`
- **Matching:** cosine @ 0.4
- **Detection:** RetinaFace/SCRFD shared, `crop_mode=bbox_margin`, margin=0.35
- **Pairs:** 6000 (scored per model may differ when embeds fail)
- **Unique images aligned:** 7676 / 7701 (failed=25)
- **Stub used:** No

## Per-model metrics

| Model | Accuracy | Precision | Recall | F1 | AUC | EER | FAR | FRR | Scored | Skipped |
|-------|----------|-----------|--------|----|-----|-----|-----|-----|--------|---------|
| magface | 0.8864 | 0.9432 | 0.8228 | 0.8789 | 0.9515 | 0.1079 | 0.0498 | 0.1772 | 5958 | 42 |

## Artifacts

- Manifest: `D:\ModelComparision\experiments\20260806T203041Z_paper_lfw_five_models_baseline_b_2a5962e6\manifest.json`
- Aggregated MD: `D:\ModelComparision\experiments\20260806T203041Z_paper_lfw_five_models_baseline_b_2a5962e6\aggregated\comparison.md`
- Aggregated HTML: `D:\ModelComparision\experiments\20260806T203041Z_paper_lfw_five_models_baseline_b_2a5962e6\aggregated\comparison.html`
- Aggregated CSV: `D:\ModelComparision\experiments\20260806T203041Z_paper_lfw_five_models_baseline_b_2a5962e6\aggregated\comparison.csv`
- Weight checksums: `D:\ModelComparision\paper\weights\checksums.sha256`
- Curated copy: `D:\ModelComparision\paper\results\lfw_five_models_20260806T203041Z_paper_lfw_five_models_baseline_b_2a5962e6`

## Protocol notes

- Shared RetinaFace alignment ran once; crops reused across models.
- AdaFace/MagFace architectures injected via `paper/reproduction/arch` (framework adapters unchanged).
- AdaFace/MagFace inputs channel-flipped RGB->BGR to match official inference without changing FaceBench preprocess.
- FaceNet uses facenet-pytorch VGGFace2 pretrained weights.
- Dlib uses official 5-point predictor + ResNet recognition `.dat` files.
- Buffalo-L uses InsightFace pack with vendor re-detect on shared crop.

