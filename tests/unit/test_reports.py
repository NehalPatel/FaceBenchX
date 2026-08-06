"""Unit tests for Phase 7 reports and figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from facebench.metrics import ComputationalMetrics, compute_recognition_metrics
from facebench.reports import ReportGenerator
from facebench.visualization import FigureGenerator


def _sample_recognition():
    y_true = np.asarray([1, 1, 1, 0, 0, 0])
    y_score = np.asarray([0.95, 0.85, 0.75, 0.2, 0.15, 0.05])
    return y_true, y_score, compute_recognition_metrics(y_true, y_score, threshold=0.5)


def test_figure_generator_writes_suite(tmp_path: Path) -> None:
    """Standard figure suite writes PNG files."""
    y_true, y_score, metrics = _sample_recognition()
    figures = FigureGenerator(tmp_path / "figures")
    paths = figures.save_standard_suite(metrics, y_true=y_true, y_score=y_score)
    assert paths["confusion_matrix"].is_file()
    assert paths["roc_curve"].is_file()
    assert paths["pr_curve"].is_file()
    figures.save_radar_chart(
        {"m1": {"accuracy": 0.9, "f1": 0.8, "auc": 0.95}},
        filename="radar.png",
    )
    assert (tmp_path / "figures" / "radar.png").is_file()
    figures.save_scalability_chart(
        [10, 100, 500],
        [0.99, 0.97, 0.95],
        latency_s=[0.01, 0.02, 0.05],
    )
    assert (tmp_path / "figures" / "scalability.png").is_file()


def test_report_generator_per_dataset_and_aggregated(tmp_path: Path) -> None:
    """ReportGenerator writes md/html/csv/json and aggregated tables."""
    y_true, y_score, recognition = _sample_recognition()
    computational = ComputationalMetrics(
        avg_embedding_time_s=0.01,
        throughput_fps=100.0,
        cpu_percent=20.0,
        ram_rss_mb=256.0,
    )
    root = tmp_path / "exp"
    generator = ReportGenerator(root)
    data = generator.write_per_dataset(
        experiment_id="exp001",
        experiment_name="demo",
        dataset_name="LFW",
        model_name="buffalo_l",
        recognition=recognition,
        computational=computational,
        config={"device": "cpu"},
        y_true=y_true.tolist(),
        y_score=y_score.tolist(),
    )
    assert (root / "reports" / "report.md").is_file()
    assert (root / "reports" / "report.html").is_file()
    assert (root / "metrics" / "metrics.csv").is_file()
    assert (root / "metrics" / "experiment.json").is_file()
    assert data.figure_paths
    assert "roc_curve" in data.figure_paths

    data2 = generator.write_per_dataset(
        experiment_id="exp002",
        experiment_name="demo2",
        dataset_name="LFW",
        model_name="facenet",
        recognition=recognition,
        computational=computational,
        generate_figures=False,
    )
    rows = generator.write_aggregated([data, data2])
    assert len(rows) == 2
    assert (root / "aggregated" / "comparison.csv").is_file()
    assert (root / "aggregated" / "comparison.md").is_file()
    assert (root / "aggregated" / "comparison.html").is_file()
    assert (root / "aggregated" / "comparison.json").is_file()
    assert (root / "aggregated" / "accuracy_comparison.png").is_file()
