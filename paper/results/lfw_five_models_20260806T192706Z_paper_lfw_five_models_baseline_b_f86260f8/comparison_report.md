# LFW x Five Models - Baseline B Comparison Report

- **Status:** SUCCESS
- **Experiment ID:** `20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8`
- **Output dir:** `D:\ModelComparision\experiments\20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8`
- **Device:** `cuda:0`
- **Matching:** cosine @ 0.40
- **Detection:** RetinaFace/SCRFD shared, `crop_mode=bbox_margin`, margin=0.35
- **Pairs:** 6000 View-2 (per-model scored count may drop on embed failures)
- **MagFace correction:** re-scored with official BGR `[0,1]` input adaptation (framework preprocess unchanged).

## Per-model metrics

| Model | Accuracy | Precision | Recall | F1 | AUC | EER | FAR | FRR | Scored | Skipped |
|-------|----------|-----------|--------|----|-----|-----|-----|-----|--------|---------|
| facenet | 0.9758 | 0.9886 | 0.9628 | 0.9756 | 0.9933 | 0.0275 | 0.0111 | 0.0372 | 5958 | 42 |
| dlib | 0.5009 | 0.5009 | 1.0000 | 0.6675 | 0.9922 | 0.0234 | 1.0000 | 0.0000 | 5893 | 107 |
| buffalo_l | 0.9828 | 1.0000 | 0.9657 | 0.9826 | 0.9899 | 0.0216 | 0.0000 | 0.0343 | 5936 | 64 |
| adaface | 0.8098 | 0.9995 | 0.6208 | 0.7659 | 0.9711 | 0.0730 | 0.0003 | 0.3792 | 5958 | 42 |
| magface | 0.8864 | 0.9432 | 0.8228 | 0.8789 | 0.9515 | 0.1079 | 0.0498 | 0.1772 | 5958 | 42 |

## Artifacts

- Manifest: `D:\ModelComparision\experiments\20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8\manifest.json`
- Aggregated MD: `D:\ModelComparision\experiments\20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8\aggregated\comparison.md`
- Aggregated HTML: `D:\ModelComparision\experiments\20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8\aggregated\comparison.html`
- Aggregated CSV: `D:\ModelComparision\experiments\20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8\aggregated\comparison.csv`
- Weight checksums: `D:\ModelComparision\paper\weights\checksums.sha256`
- Comparison report: `D:\ModelComparision\paper\results\lfw_five_models_baseline_b_report.md`

## Protocol notes

- Shared RetinaFace alignment once; crops reused across models.
- Fixed decision threshold 0.40 cosine for all models.
- Dlib AUC/EER are strong while accuracy@0.40 is ~0.50 because dlib cosine scores sit above 0.40 for both classes; prefer EER threshold operationally.
- AdaFace/MagFace architectures injected via `paper/reproduction/arch`.
- MagFace uses paper-local BGR `[0,1]` wrapper to match official MagFace inference.

