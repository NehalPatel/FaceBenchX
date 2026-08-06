# FaceBench Report: paper_lfw_buffalo_l

- **Experiment ID:** `20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c`
- **Dataset:** LFW
- **Model:** buffalo_l

## Recognition Metrics

| Metric | Value |
| --- | ---: |
| Threshold | 0.4000 |
| Accuracy | 0.9817 |
| Precision | 1.0000 |
| Recall | 0.9635 |
| F1 | 0.9814 |
| FAR | 0.0000 |
| FRR | 0.0365 |
| AUC | 0.9892 |
| EER | 0.0238 |
| EER threshold | 0.1219 |
| Pairs | 5958 |

### Confusion Matrix

```
[[2973    0]
 [ 109 2876]]
```

## Computational Metrics

| Metric | Value |
| --- | ---: |
| Model load (s) | 2.9465 |
| Avg embedding (s) | 0.5585 |
| Avg inference (s) | 0.5585 |
| Latency (s) | 0.5585 |
| Throughput (FPS) | 1.7905 |
| CPU % | 254.2281 |
| RAM RSS (MiB) | 626.9648 |
| GPU % | - |
| GPU memory (MiB) | - |
| Model size (MiB) | 600.7100 |

## Figures

- **confusion_matrix:** `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\figures\confusion_matrix.png`
- **roc_curve:** `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\figures\roc_curve.png`
- **pr_curve:** `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\figures\pr_curve.png`
- **resource_usage:** `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\figures\resource_usage.png`
- **inference_time:** `D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c\runs\LFW__buffalo_l\figures\inference_time.png`

## Notes

Baseline A vendor detect. Scored 5958/6000 pairs; skipped 42 due to detection failures (25 unique images).
