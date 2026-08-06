"""Markdown experiment report writer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from facebench.reports.types import ComparisonRow, ExperimentReportData


def render_markdown_report(data: ExperimentReportData) -> str:
    """Render a Markdown report for a single experiment.

    Args:
        data: Experiment report payload.

    Returns:
        Markdown document string.
    """
    lines = [
        f"# FaceBench Report: {data.experiment_name}",
        "",
        f"- **Experiment ID:** `{data.experiment_id}`",
        f"- **Dataset:** {data.dataset_name}",
        f"- **Model:** {data.model_name}",
        "",
    ]
    if data.recognition is not None:
        r = data.recognition
        lines.extend(
            [
                "## Recognition Metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Threshold | {r.threshold:.4f} |",
                f"| Accuracy | {r.accuracy:.4f} |",
                f"| Precision | {r.precision:.4f} |",
                f"| Recall | {r.recall:.4f} |",
                f"| F1 | {r.f1:.4f} |",
                f"| FAR | {r.far:.4f} |",
                f"| FRR | {r.frr:.4f} |",
                f"| AUC | {r.auc:.4f} |",
                f"| EER | {r.eer:.4f} |",
                f"| EER threshold | {r.eer_threshold:.4f} |",
                f"| Pairs | {r.num_pairs} |",
                "",
                "### Confusion Matrix",
                "",
                "```",
                str(r.confusion.as_matrix()),
                "```",
                "",
            ]
        )
    if data.computational is not None:
        c = data.computational
        lines.extend(
            [
                "## Computational Metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Model load (s) | {_fmt(c.model_load_time_s)} |",
                f"| Avg embedding (s) | {_fmt(c.avg_embedding_time_s)} |",
                f"| Avg inference (s) | {_fmt(c.avg_inference_time_s)} |",
                f"| Latency (s) | {_fmt(c.recognition_latency_s)} |",
                f"| Throughput (FPS) | {_fmt(c.throughput_fps)} |",
                f"| CPU % | {_fmt(c.cpu_percent)} |",
                f"| RAM RSS (MiB) | {_fmt(c.ram_rss_mb)} |",
                f"| GPU % | {_fmt(c.gpu_percent)} |",
                f"| GPU memory (MiB) | {_fmt(c.gpu_memory_mb)} |",
                f"| Model size (MiB) | {_fmt(c.model_size_mb)} |",
                "",
            ]
        )
    if data.figure_paths:
        lines.extend(["## Figures", ""])
        for key, path in data.figure_paths.items():
            lines.append(f"- **{key}:** `{path}`")
        lines.append("")
    if data.notes:
        lines.extend(["## Notes", "", data.notes, ""])
    return "\n".join(lines)


def render_comparison_markdown(rows: Sequence[ComparisonRow]) -> str:
    """Render an aggregated comparison Markdown table.

    Args:
        rows: Comparison rows.

    Returns:
        Markdown document string.
    """
    lines = [
        "# FaceBench Aggregated Comparison",
        "",
        "| Dataset | Model | Accuracy | F1 | AUC | EER | Emb (s) | FPS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.dataset_name} | {row.model_name} | "
            f"{_fmt(row.accuracy)} | {_fmt(row.f1)} | {_fmt(row.auc)} | "
            f"{_fmt(row.eer)} | {_fmt(row.avg_embedding_time_s)} | "
            f"{_fmt(row.throughput_fps)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(data: ExperimentReportData, path: str | Path) -> Path:
    """Write a Markdown report to disk.

    Args:
        data: Experiment report payload.
        path: Destination path.

    Returns:
        Resolved path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown_report(data), encoding="utf-8")
    return target.resolve()


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
