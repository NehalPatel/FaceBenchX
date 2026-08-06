# LFW × Buffalo-L Execution Summary

- **Status:** SUCCESS
- **Experiment ID:** `20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c`
- **Output dir:** `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c`
- **Started (UTC):** 2026-08-05T20:36:32.686364+00:00
- **Ended (UTC):** 2026-08-05T22:28:45.114945+00:00
- **Elapsed:** 6732.4 s (1.87 h)
- **Device:** `cpu` (CPUExecutionProvider)
- **Matching:** cosine @ 0.4
- **Pairs protocol:** 6000
- **Pairs scored:** 5958
- **Pairs skipped (no face):** 42
- **Unique failed images:** 25
- **Stub used:** No

## Metrics

- Accuracy: 0.9817052702249077
- Precision: 1.0
- Recall: 0.9634840871021776
- F1: 0.9814024910424842
- AUC: 0.9891695274218383
- EER: 0.023833597858109908
- FAR: 0.0
- FRR: 0.036515912897822446
- Threshold: 0.4
- EER threshold: 0.12191095986772116

## Artifact validation

- PASS: config.snapshot.yaml — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\config.snapshot.yaml`
- PASS: env.json — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\env.json`
- PASS: manifest.json — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\manifest.json`
- PASS: metrics/summary.json — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\metrics\summary.json`
- PASS: report.html — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\reports\report.html`
- PASS: report.md — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\reports\report.md`
- PASS: experiment.json — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\metrics\experiment.json`
- PASS: metrics.csv — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\metrics\metrics.csv`
- PASS: verification.json — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\metrics\verification.json`
- PASS: ROC — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\figures\roc_curve.png`
- PASS: confusion — `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\figures\confusion_matrix.png`

## Issues / notes

- CUDA provider unavailable; CPU-only ONNX Runtime.
- Official UMass `pairs.txt` host DNS failed; used facenet mirror and normalized header to `10`.
- Dataset root nested at `D:/datasets/lfw/lfw-deepfunneled/lfw-deepfunneled`.
- First orchestrator attempt aborted on first no-face image; paper runner retries with skip-on-missed-detection + embedding cache.
- Baseline A: InsightFace Buffalo-L bundled detector (not shared RetinaFace).

### Sample failed images

- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Chan_Gailey\Chan_Gailey_0001.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Dany_Heatley\Dany_Heatley_0001.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Nobuyuki_Idei\Nobuyuki_Idei_0001.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Jennifer_Furminger\Jennifer_Furminger_0001.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\John_Paul_II\John_Paul_II_0004.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Morgan_Freeman\Morgan_Freeman_0002.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\David_McCullough\David_McCullough_0001.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Arsinee_Khanjian\Arsinee_Khanjian_0001.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Robert_Kipkoech_Cheruiyot\Robert_Kipkoech_Cheruiyot_0001.jpg` — Buffalo-L detected no faces in the image
- `D:\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Orrin_Hatch\Orrin_Hatch_0001.jpg` — Buffalo-L detected no faces in the image

Manifest: `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\manifest.json`
Curated: `D:\ModelComparision\paper\results\lfw_buffalo_l_20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c`

