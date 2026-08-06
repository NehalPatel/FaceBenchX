# Publication Tables — LFW × Buffalo-L (Baseline A)

Experiment ID: `20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c`

## Table 1. Recognition metrics with 95% confidence intervals

| Metric | Estimate | 95% CI | Method |
| --- | ---: | ---: | --- |
| Accuracy | 0.9817 | 0.9780–0.9848 | Wilson |
| Precision | 1.0000 | 0.9987–1.0000 | Wilson |
| Recall | 0.9635 | 0.9561–0.9696 | Wilson |
| F1 | 0.9814 | — | — |
| AUC | 0.9892 | 0.9865–0.9918 | Hanley–McNeil |
| EER | 0.0238 | — | — |
| FAR @ 0.40 | 0.0000 | 0.0000–0.0013 | Wilson |
| FRR @ 0.40 | 0.0365 | 0.0304–0.0439 | Wilson |

n = 5958 scored pairs (protocol 6000; 42 skipped). Cosine threshold = 0.40.

## Table 2. Computational metrics (CPU)

| Metric | Value |
| --- | ---: |
| Model load time (s) | 2.9465 |
| Avg. embedding time (s) | 0.5585 |
| Throughput (FPS) | 1.7905 |
| CPU utilization (%) | 254.23 |
| Peak RSS (MiB) | 626.96 |
| On-disk model size (MiB) | 600.71 |

## Table 3. Confusion matrix at cosine threshold 0.40

|  | Pred. Accept | Pred. Reject |
| --- | ---: | ---: |
| Actual same | 2876 | 109 |
| Actual different | 0 | 2973 |
