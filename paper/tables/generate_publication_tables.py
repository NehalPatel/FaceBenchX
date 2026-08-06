"""Generate publication tables from the frozen LFW x Buffalo-L experiment."""

from __future__ import annotations

import json
from pathlib import Path

from facebench.stats import hanley_mcneil_auc_ci, proportion_ci_from_confusion

EXP = Path(
    r"D:\ModelComparision\experiments\20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c"
)
OUT = Path(r"D:\ModelComparision\paper\tables")


def main() -> None:
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    r = manifest["recognition_metrics"]
    c = r["confusion"]
    comp = manifest["computational_metrics"]
    cis = proportion_ci_from_confusion(
        c["true_positive"],
        c["true_negative"],
        c["false_positive"],
        c["false_negative"],
    )
    auc_ci = hanley_mcneil_auc_ci(r["auc"], r["num_positive"], r["num_negative"])

    csv1 = [
        "Metric,Estimate,CI95_low,CI95_high,Method",
        f"Accuracy,{r['accuracy']:.6f},{cis['accuracy'].low:.6f},{cis['accuracy'].high:.6f},Wilson",
        f"Precision,{r['precision']:.6f},{cis['precision'].low:.6f},{cis['precision'].high:.6f},Wilson",
        f"Recall,{r['recall']:.6f},{cis['recall'].low:.6f},{cis['recall'].high:.6f},Wilson",
        f"F1,{r['f1']:.6f},,,",
        f"AUC,{r['auc']:.6f},{auc_ci.low:.6f},{auc_ci.high:.6f},Hanley-McNeil",
        f"EER,{r['eer']:.6f},,,",
        f"FAR,{r['far']:.6f},{cis['far'].low:.6f},{cis['far'].high:.6f},Wilson",
        f"FRR,{r['frr']:.6f},{cis['frr'].low:.6f},{cis['frr'].high:.6f},Wilson",
    ]
    (OUT / "table1_recognition_metrics.csv").write_text(
        "\n".join(csv1) + "\n", encoding="utf-8"
    )

    csv2 = [
        "Metric,Value,Unit",
        f"model_load_time_s,{comp['model_load_time_s']:.6f},s",
        f"avg_embedding_time_s,{comp['avg_embedding_time_s']:.6f},s",
        f"throughput_fps,{comp['throughput_fps']:.6f},FPS",
        f"cpu_percent,{comp['cpu_percent']:.4f},%",
        f"ram_rss_mb,{comp['ram_rss_mb']:.4f},MiB",
        f"model_size_mb,{comp['model_size_mb']:.4f},MiB",
        f"num_samples,{comp['num_samples']},embeddings",
    ]
    (OUT / "table2_computational_metrics.csv").write_text(
        "\n".join(csv2) + "\n", encoding="utf-8"
    )

    csv3 = [
        "Actual,Pred_Accept,Pred_Reject",
        f"Same,{c['true_positive']},{c['false_negative']}",
        f"Different,{c['false_positive']},{c['true_negative']}",
    ]
    (OUT / "table3_confusion_matrix.csv").write_text(
        "\n".join(csv3) + "\n", encoding="utf-8"
    )

    md = f"""# Publication Tables — LFW × Buffalo-L (Baseline A)

Experiment ID: `20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c`

## Table 1. Recognition metrics with 95% confidence intervals

| Metric | Estimate | 95% CI | Method |
| --- | ---: | ---: | --- |
| Accuracy | {r['accuracy']:.4f} | {cis['accuracy'].low:.4f}–{cis['accuracy'].high:.4f} | Wilson |
| Precision | {r['precision']:.4f} | {cis['precision'].low:.4f}–{cis['precision'].high:.4f} | Wilson |
| Recall | {r['recall']:.4f} | {cis['recall'].low:.4f}–{cis['recall'].high:.4f} | Wilson |
| F1 | {r['f1']:.4f} | — | — |
| AUC | {r['auc']:.4f} | {auc_ci.low:.4f}–{auc_ci.high:.4f} | Hanley–McNeil |
| EER | {r['eer']:.4f} | — | — |
| FAR @ 0.40 | {r['far']:.4f} | {cis['far'].low:.4f}–{cis['far'].high:.4f} | Wilson |
| FRR @ 0.40 | {r['frr']:.4f} | {cis['frr'].low:.4f}–{cis['frr'].high:.4f} | Wilson |

n = {r['num_pairs']} scored pairs (protocol 6000; 42 skipped). Cosine threshold = 0.40.

## Table 2. Computational metrics (CPU)

| Metric | Value |
| --- | ---: |
| Model load time (s) | {comp['model_load_time_s']:.4f} |
| Avg. embedding time (s) | {comp['avg_embedding_time_s']:.4f} |
| Throughput (FPS) | {comp['throughput_fps']:.4f} |
| CPU utilization (%) | {comp['cpu_percent']:.2f} |
| Peak RSS (MiB) | {comp['ram_rss_mb']:.2f} |
| On-disk model size (MiB) | {comp['model_size_mb']:.2f} |

## Table 3. Confusion matrix at cosine threshold 0.40

|  | Pred. Accept | Pred. Reject |
| --- | ---: | ---: |
| Actual same | {c['true_positive']} | {c['false_negative']} |
| Actual different | {c['false_positive']} | {c['true_negative']} |
"""
    (OUT / "publication_tables.md").write_text(md, encoding="utf-8")
    print("Wrote publication tables to", OUT)


if __name__ == "__main__":
    main()
