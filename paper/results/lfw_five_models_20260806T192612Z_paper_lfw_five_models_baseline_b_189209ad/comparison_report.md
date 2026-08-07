# LFW × Five Models — Baseline B Comparison Report

- **Status:** SUCCESS
- **Experiment ID:** `20260806T192612Z_paper_lfw_five_models_baseline_b_189209ad`
- **Output dir:** `D:\ModelComparision\experiments\20260806T192612Z_paper_lfw_five_models_baseline_b_189209ad`
- **Started (UTC):** 2026-08-06T19:26:12.083558+00:00
- **Ended (UTC):** 2026-08-06T19:26:54.049195+00:00
- **Elapsed:** 42.0 s (0.01 h)
- **Device:** `cuda:0`
- **Matching:** cosine @ 0.4
- **Detection:** RetinaFace/SCRFD shared, `crop_mode=bbox_margin`, margin=0.35
- **Pairs:** 40 (scored per model may differ when embeds fail)
- **Unique images aligned:** 67 / 67 (failed=0)
- **Stub used:** No

## Per-model metrics

| Model | Accuracy | Precision | Recall | F1 | AUC | EER | FAR | FRR | Scored | Skipped |
|-------|----------|-----------|--------|----|-----|-----|-----|-----|--------|---------|
| facenet | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 40 | 0 |
| dlib | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 40 | 0 |
| buffalo_l | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 40 | 0 |
| adaface | 0.7000 | 1.0000 | 0.7000 | 0.8235 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 40 | 0 |
| magface | 0.8000 | 1.0000 | 0.8000 | 0.8889 | 0.0000 | 0.0000 | 0.0000 | 0.2000 | 40 | 0 |

## Artifacts

- Manifest: `D:\ModelComparision\experiments\20260806T192612Z_paper_lfw_five_models_baseline_b_189209ad\manifest.json`
- Aggregated MD: `D:\ModelComparision\experiments\20260806T192612Z_paper_lfw_five_models_baseline_b_189209ad\aggregated\comparison.md`
- Aggregated HTML: `D:\ModelComparision\experiments\20260806T192612Z_paper_lfw_five_models_baseline_b_189209ad\aggregated\comparison.html`
- Aggregated CSV: `D:\ModelComparision\experiments\20260806T192612Z_paper_lfw_five_models_baseline_b_189209ad\aggregated\comparison.csv`
- Weight checksums: `D:\ModelComparision\paper\weights\checksums.sha256`
- Curated copy: `D:\ModelComparision\paper\results\lfw_five_models_20260806T192612Z_paper_lfw_five_models_baseline_b_189209ad`

## Protocol notes

- Shared RetinaFace alignment ran once; crops reused across models.
- AdaFace/MagFace architectures injected via `paper/reproduction/arch` (framework adapters unchanged).
- AdaFace/MagFace inputs channel-flipped RGB→BGR to match official inference without changing FaceBench preprocess.
- FaceNet uses facenet-pytorch VGGFace2 pretrained weights.
- Dlib uses official 5-point predictor + ResNet recognition `.dat` files.
- Buffalo-L uses InsightFace pack with vendor re-detect on shared crop.

