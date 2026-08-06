"""HTML, Markdown, CSV, and JSON experiment reports (Phase 7)."""

from __future__ import annotations

from facebench.reports.exporters import (
    export_comparison_csv,
    export_json,
    export_metrics_csv,
)
from facebench.reports.generator import ReportGenerator
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

__all__ = [
    "ComparisonRow",
    "ExperimentReportData",
    "ReportGenerator",
    "comparison_row_from_report",
    "export_comparison_csv",
    "export_json",
    "export_metrics_csv",
    "render_comparison_html",
    "render_comparison_markdown",
    "write_html_report",
    "write_markdown_report",
]
