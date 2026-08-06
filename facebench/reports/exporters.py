"""CSV and JSON metric exporters."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from facebench.reports.types import ComparisonRow, ExperimentReportData


def export_json(data: dict[str, Any] | ExperimentReportData, path: str | Path) -> Path:
    """Write a JSON experiment file.

    Args:
        data: Mapping or :class:`ExperimentReportData`.
        path: Destination path.

    Returns:
        Resolved output path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = data.to_dict() if isinstance(data, ExperimentReportData) else data
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    return target.resolve()


def export_metrics_csv(
    data: ExperimentReportData,
    path: str | Path,
) -> Path:
    """Write flat recognition/computational metrics as a one-row CSV.

    Args:
        data: Experiment report payload.
        path: Destination CSV path.

    Returns:
        Resolved output path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "experiment_id": data.experiment_id,
        "experiment_name": data.experiment_name,
        "dataset_name": data.dataset_name,
        "model_name": data.model_name,
    }
    if data.recognition is not None:
        recog = data.recognition.to_dict()
        for key in (
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "far",
            "frr",
            "auc",
            "eer",
            "eer_threshold",
            "num_pairs",
        ):
            row[key] = recog.get(key)
    if data.computational is not None:
        for key, value in data.computational.to_dict().items():
            if key == "extra":
                continue
            row[f"compute_{key}"] = value

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return target.resolve()


def export_comparison_csv(
    rows: Sequence[ComparisonRow],
    path: str | Path,
) -> Path:
    """Write an aggregated comparison table as CSV.

    Args:
        rows: Comparison rows.
        path: Destination CSV path.

    Returns:
        Resolved output path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "model_name",
        "accuracy",
        "f1",
        "auc",
        "eer",
        "avg_embedding_time_s",
        "throughput_fps",
        "cpu_percent",
        "ram_rss_mb",
        "gpu_percent",
        "gpu_memory_mb",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = row.to_dict()
            writer.writerow({key: payload.get(key) for key in fieldnames})
    return target.resolve()
