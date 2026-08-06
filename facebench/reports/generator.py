"""High-level report generation facade."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from facebench.metrics.computational import ComputationalMetrics
from facebench.metrics.recognition import RecognitionMetrics
from facebench.reports.exporters import (
    export_comparison_csv,
    export_json,
    export_metrics_csv,
)
from facebench.reports.html_report import render_comparison_html, write_html_report
from facebench.reports.markdown_report import (
    render_comparison_markdown,
    write_markdown_report,
)
from facebench.reports.types import (
    ComparisonRow,
    ExperimentReportData,
    comparison_row_from_report,
)
from facebench.visualization.figures import FigureGenerator


class ReportGenerator:
    """Write per-dataset and aggregated FaceBench report artifacts.

    Output layout under ``output_dir``::

        reports/
          report.md
          report.html
        metrics/
          metrics.csv
          experiment.json
        figures/
          *.png
        aggregated/   # only for write_aggregated
          comparison.md
          comparison.html
          comparison.csv
          comparison.json
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the generator.

        Args:
            output_dir: Experiment output root (typically an experiment_id dir).
        """
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.metrics_dir = self.output_dir / "metrics"
        self.figures_dir = self.output_dir / "figures"
        self.aggregated_dir = self.output_dir / "aggregated"
        for path in (
            self.reports_dir,
            self.metrics_dir,
            self.figures_dir,
            self.aggregated_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def write_per_dataset(
        self,
        *,
        experiment_id: str,
        experiment_name: str,
        dataset_name: str,
        model_name: str,
        recognition: RecognitionMetrics | None = None,
        computational: ComputationalMetrics | None = None,
        config: dict | None = None,
        y_true: Sequence[int] | None = None,
        y_score: Sequence[float] | None = None,
        notes: str = "",
        generate_figures: bool = True,
    ) -> ExperimentReportData:
        """Write HTML/Markdown/CSV/JSON (+ optional figures) for one run.

        Args:
            experiment_id: Experiment identifier.
            experiment_name: Human-readable name.
            dataset_name: Dataset name.
            model_name: Model name.
            recognition: Optional recognition metrics.
            computational: Optional computational metrics.
            config: Optional config snapshot.
            y_true: Optional labels for PR curve.
            y_score: Optional scores for PR curve.
            notes: Optional free-form notes.
            generate_figures: When ``True`` and recognition metrics exist,
                write the standard figure suite.

        Returns:
            Populated :class:`ExperimentReportData`.
        """
        figure_paths: dict[str, str] = {}
        if generate_figures and recognition is not None:
            figures = FigureGenerator(self.figures_dir)
            saved = figures.save_standard_suite(
                recognition,
                y_true=y_true,
                y_score=y_score,
            )
            if computational is not None:
                resource_values = {
                    k: v
                    for k, v in {
                        "CPU %": computational.cpu_percent,
                        "RAM MiB": computational.ram_rss_mb,
                        "GPU %": computational.gpu_percent,
                        "GPU MiB": computational.gpu_memory_mb,
                    }.items()
                    if v is not None
                }
                if resource_values:
                    saved["resource_usage"] = figures.save_metric_bar_chart(
                        resource_values,
                        title=f"Resource Usage — {model_name}",
                        ylabel="Value",
                        filename="resource_usage.png",
                    )
                if computational.avg_embedding_time_s is not None:
                    saved["inference_time"] = figures.save_metric_bar_chart(
                        {model_name: computational.avg_embedding_time_s},
                        title="Inference / Embedding Time",
                        ylabel="Seconds",
                        filename="inference_time.png",
                    )
            figure_paths = {key: str(path) for key, path in saved.items()}

        data = ExperimentReportData(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            model_name=model_name,
            config=dict(config or {}),
            recognition=recognition,
            computational=computational,
            figure_paths=figure_paths,
            notes=notes,
        )

        write_markdown_report(data, self.reports_dir / "report.md")
        write_html_report(data, self.reports_dir / "report.html")
        export_metrics_csv(data, self.metrics_dir / "metrics.csv")
        export_json(data, self.metrics_dir / "experiment.json")
        return data

    def write_aggregated(
        self,
        reports: Sequence[ExperimentReportData],
        *,
        also_write_comparison_figures: bool = True,
    ) -> list[ComparisonRow]:
        """Write aggregated comparison tables across multiple runs.

        Args:
            reports: Per-experiment report payloads.
            also_write_comparison_figures: When ``True``, emit accuracy/F1
                and latency comparison bar charts.

        Returns:
            Generated comparison rows.
        """
        rows = [comparison_row_from_report(item) for item in reports]
        export_comparison_csv(rows, self.aggregated_dir / "comparison.csv")
        export_json(
            {"rows": [row.to_dict() for row in rows]},
            self.aggregated_dir / "comparison.json",
        )
        (self.aggregated_dir / "comparison.md").write_text(
            render_comparison_markdown(rows),
            encoding="utf-8",
        )
        (self.aggregated_dir / "comparison.html").write_text(
            render_comparison_html(rows),
            encoding="utf-8",
        )

        if also_write_comparison_figures and rows:
            figures = FigureGenerator(self.aggregated_dir)
            accuracy = {
                f"{row.model_name}@{row.dataset_name}": row.accuracy
                for row in rows
                if row.accuracy is not None
            }
            if accuracy:
                figures.save_metric_bar_chart(
                    accuracy,
                    title="Accuracy Comparison",
                    ylabel="Accuracy",
                    filename="accuracy_comparison.png",
                )
            f1_values = {
                f"{row.model_name}@{row.dataset_name}": row.f1
                for row in rows
                if row.f1 is not None
            }
            if f1_values:
                figures.save_metric_bar_chart(
                    f1_values,
                    title="F1 Comparison",
                    ylabel="F1",
                    filename="f1_comparison.png",
                )
            latency = {
                f"{row.model_name}@{row.dataset_name}": row.avg_embedding_time_s
                for row in rows
                if row.avg_embedding_time_s is not None
            }
            if latency:
                figures.save_metric_bar_chart(
                    latency,
                    title="Embedding Time Comparison",
                    ylabel="Seconds",
                    filename="latency_comparison.png",
                )
            radar_series: dict[str, dict[str, float]] = {}
            for row in rows:
                if None in (row.accuracy, row.f1, row.auc):
                    continue
                key = f"{row.model_name}"
                radar_series[key] = {
                    "accuracy": float(row.accuracy),
                    "f1": float(row.f1),
                    "auc": float(row.auc),
                    "eer_inv": 1.0 - float(row.eer or 0.0),
                }
            if radar_series:
                figures.save_radar_chart(
                    radar_series,
                    filename="radar_comparison.png",
                )
        return rows
