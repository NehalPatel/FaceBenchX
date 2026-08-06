"""HTML experiment report writer."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path

from facebench.reports.types import ComparisonRow, ExperimentReportData


def render_html_report(data: ExperimentReportData) -> str:
    """Render an HTML report for a single experiment.

    Args:
        data: Experiment report payload.

    Returns:
        HTML document string.
    """
    sections: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        f"<title>FaceBench — {escape(data.experiment_name)}</title>",
        "<style>",
        "body{font-family:Georgia,serif;margin:2rem;color:#222;}",
        "table{border-collapse:collapse;margin:1rem 0;}",
        "th,td{border:1px solid #ccc;padding:0.4rem 0.7rem;}",
        "th{background:#f3f3f3;text-align:left;}",
        "td.num{text-align:right;font-variant-numeric:tabular-nums;}",
        "img{max-width:520px;border:1px solid #ddd;margin:0.5rem 0;}",
        "code{background:#f6f6f6;padding:0.1rem 0.3rem;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>FaceBench Report: {escape(data.experiment_name)}</h1>",
        "<ul>",
        (
            "<li><strong>Experiment ID:</strong> "
            f"<code>{escape(data.experiment_id)}</code></li>"
        ),
        f"<li><strong>Dataset:</strong> {escape(data.dataset_name)}</li>",
        f"<li><strong>Model:</strong> {escape(data.model_name)}</li>",
        "</ul>",
    ]

    if data.recognition is not None:
        r = data.recognition
        sections.extend(
            [
                "<h2>Recognition Metrics</h2>",
                "<table>",
                "<tr><th>Metric</th><th>Value</th></tr>",
                *_metric_rows(
                    [
                        ("Threshold", r.threshold),
                        ("Accuracy", r.accuracy),
                        ("Precision", r.precision),
                        ("Recall", r.recall),
                        ("F1", r.f1),
                        ("FAR", r.far),
                        ("FRR", r.frr),
                        ("AUC", r.auc),
                        ("EER", r.eer),
                        ("EER threshold", r.eer_threshold),
                        ("Pairs", float(r.num_pairs)),
                    ]
                ),
                "</table>",
            ]
        )

    if data.computational is not None:
        c = data.computational
        sections.extend(
            [
                "<h2>Computational Metrics</h2>",
                "<table>",
                "<tr><th>Metric</th><th>Value</th></tr>",
                *_metric_rows(
                    [
                        ("Model load (s)", c.model_load_time_s),
                        ("Avg embedding (s)", c.avg_embedding_time_s),
                        ("Avg inference (s)", c.avg_inference_time_s),
                        ("Latency (s)", c.recognition_latency_s),
                        ("Throughput (FPS)", c.throughput_fps),
                        ("CPU %", c.cpu_percent),
                        ("RAM RSS (MiB)", c.ram_rss_mb),
                        ("GPU %", c.gpu_percent),
                        ("GPU memory (MiB)", c.gpu_memory_mb),
                        ("Model size (MiB)", c.model_size_mb),
                    ]
                ),
                "</table>",
            ]
        )

    if data.figure_paths:
        sections.append("<h2>Figures</h2>")
        for key, path in data.figure_paths.items():
            sections.append(f"<h3>{escape(key)}</h3>")
            sections.append(f'<p><img src="{escape(path)}" alt="{escape(key)}"/></p>')
            sections.append(f"<p><code>{escape(path)}</code></p>")

    if data.notes:
        sections.extend(["<h2>Notes</h2>", f"<p>{escape(data.notes)}</p>"])

    sections.extend(["</body>", "</html>"])
    return "\n".join(sections)


def render_comparison_html(rows: Sequence[ComparisonRow]) -> str:
    """Render an aggregated comparison HTML table.

    Args:
        rows: Comparison rows.

    Returns:
        HTML document string.
    """
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"/>',
        "<title>FaceBench Comparison</title>",
        "<style>body{font-family:Georgia,serif;margin:2rem;}",
        "table{border-collapse:collapse;}th,td{border:1px solid #ccc;",
        "padding:0.4rem 0.7rem;}th{background:#f3f3f3;}",
        "td.num{text-align:right;}</style></head><body>",
        "<h1>FaceBench Aggregated Comparison</h1>",
        "<table><tr>",
        "<th>Dataset</th><th>Model</th><th>Accuracy</th><th>F1</th>",
        "<th>AUC</th><th>EER</th><th>Emb (s)</th><th>FPS</th>",
        "</tr>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            f"<td>{escape(row.dataset_name)}</td>"
            f"<td>{escape(row.model_name)}</td>"
            f"<td class='num'>{_fmt(row.accuracy)}</td>"
            f"<td class='num'>{_fmt(row.f1)}</td>"
            f"<td class='num'>{_fmt(row.auc)}</td>"
            f"<td class='num'>{_fmt(row.eer)}</td>"
            f"<td class='num'>{_fmt(row.avg_embedding_time_s)}</td>"
            f"<td class='num'>{_fmt(row.throughput_fps)}</td>"
            "</tr>"
        )
    lines.extend(["</table></body></html>"])
    return "\n".join(lines)


def write_html_report(data: ExperimentReportData, path: str | Path) -> Path:
    """Write an HTML report to disk.

    Args:
        data: Experiment report payload.
        path: Destination path.

    Returns:
        Resolved path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html_report(data), encoding="utf-8")
    return target.resolve()


def _metric_rows(items: list[tuple[str, float | None]]) -> list[str]:
    rows: list[str] = []
    for name, value in items:
        rows.append(
            f"<tr><td>{escape(name)}</td><td class='num'>{_fmt(value)}</td></tr>"
        )
    return rows


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer() and abs(value) >= 1:
        return str(int(value))
    return f"{value:.4f}"
