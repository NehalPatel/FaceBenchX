"""Publication-ready figure generators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from facebench.metrics.recognition import RecognitionMetrics
from facebench.visualization.style import apply_publication_style, use_agg_backend


class FigureGenerator:
    """Generate IEEE/Springer-oriented evaluation figures.

    Requires ``matplotlib`` (and optionally ``seaborn``). Callers should
    install ``facebench[reports]``.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the figure writer.

        Args:
            output_dir: Directory for saved figure files.
        """
        use_agg_backend()
        apply_publication_style()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_confusion_matrix(
        self,
        metrics: RecognitionMetrics,
        *,
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png",
    ) -> Path:
        """Save a confusion-matrix heatmap.

        Args:
            metrics: Recognition metrics containing confusion counts.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to the saved PNG.
        """
        import matplotlib.pyplot as plt

        matrix = metrics.confusion.as_matrix()
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Pred Neg", "Pred Pos"])
        ax.set_yticks([0, 1], labels=["True Neg", "True Pos"])
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(int(value)), ha="center", va="center", color="black")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        return self._save(fig, filename)

    def save_roc_curve(
        self,
        metrics: RecognitionMetrics,
        *,
        title: str = "ROC Curve",
        filename: str = "roc_curve.png",
    ) -> Path:
        """Save an ROC curve with AUC annotation.

        Args:
            metrics: Recognition metrics with ROC arrays.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to the saved PNG.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.plot(metrics.roc_fpr, metrics.roc_tpr, label=f"AUC = {metrics.auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
        ax.set_xlabel("False Accept Rate (FAR)")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        return self._save(fig, filename)

    def save_precision_recall_curve(
        self,
        y_true: Sequence[int] | np.ndarray,
        y_score: Sequence[float] | np.ndarray,
        *,
        title: str = "Precision-Recall Curve",
        filename: str = "pr_curve.png",
    ) -> Path:
        """Save a precision-recall curve from raw labels/scores.

        Args:
            y_true: Binary labels.
            y_score: Similarity scores.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to the saved PNG.
        """
        import matplotlib.pyplot as plt

        labels = np.asarray(y_true).astype(bool)
        scores = np.asarray(y_score, dtype=np.float64)
        order = np.argsort(-scores)
        labels = labels[order]
        tps = np.cumsum(labels)
        fps = np.cumsum(~labels)
        precision = tps / np.maximum(tps + fps, 1)
        recall = tps / max(int(labels.sum()), 1)
        precision = np.r_[1.0, precision]
        recall = np.r_[0.0, recall]

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        return self._save(fig, filename)

    def save_metric_bar_chart(
        self,
        values: Mapping[str, float],
        *,
        title: str,
        ylabel: str,
        filename: str,
    ) -> Path:
        """Save a labeled bar chart (accuracy, F1, latency, etc.).

        Args:
            values: Label → metric value.
            title: Figure title.
            ylabel: Y-axis label.
            filename: Output filename.

        Returns:
            Path to the saved PNG.
        """
        import matplotlib.pyplot as plt

        labels = list(values.keys())
        heights = [float(values[label]) for label in labels]
        fig, ax = plt.subplots(figsize=(max(4.5, 0.7 * len(labels) + 2), 4.0))
        ax.bar(labels, heights, color="#4C72B0")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        return self._save(fig, filename)

    def save_radar_chart(
        self,
        series: Mapping[str, Mapping[str, float]],
        *,
        title: str = "Model Capability Radar",
        filename: str = "radar_chart.png",
    ) -> Path:
        """Save a multi-model radar chart.

        Args:
            series: Model name → metric-name → value (preferably normalized
                to ``[0, 1]``).
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to the saved PNG.

        Raises:
            ValueError: If ``series`` is empty.
        """
        import matplotlib.pyplot as plt

        if not series:
            raise ValueError("series must contain at least one model")
        metric_names = list(next(iter(series.values())).keys())
        if not metric_names:
            raise ValueError("each series entry must contain metrics")

        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
        angles = np.concatenate([angles, angles[:1]])

        fig, ax = plt.subplots(figsize=(5.5, 5.0), subplot_kw={"polar": True})
        for model_name, metrics in series.items():
            values = [float(metrics[m]) for m in metric_names]
            values = values + values[:1]
            ax.plot(angles, values, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1], metric_names)
        ax.set_title(title)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        return self._save(fig, filename)

    def save_scalability_chart(
        self,
        identity_counts: Sequence[int],
        accuracy: Sequence[float],
        latency_s: Sequence[float] | None = None,
        *,
        title: str = "Scalability",
        filename: str = "scalability.png",
    ) -> Path:
        """Save accuracy (and optional latency) vs enrolled identities.

        Args:
            identity_counts: Gallery sizes.
            accuracy: Accuracy at each gallery size.
            latency_s: Optional latency series.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to the saved PNG.
        """
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(5.5, 4.0))
        ax1.plot(
            identity_counts, accuracy, marker="o", color="#4C72B0", label="Accuracy"
        )
        ax1.set_xlabel("Enrolled identities")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(title)
        if latency_s is not None:
            ax2 = ax1.twinx()
            ax2.plot(
                identity_counts,
                latency_s,
                marker="s",
                color="#C44E52",
                label="Latency (s)",
            )
            ax2.set_ylabel("Latency (s)")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax1.legend(loc="best")
        return self._save(fig, filename)

    def save_standard_suite(
        self,
        metrics: RecognitionMetrics,
        *,
        y_true: Sequence[int] | np.ndarray | None = None,
        y_score: Sequence[float] | np.ndarray | None = None,
        prefix: str = "",
    ) -> dict[str, Path]:
        """Save the default single-run figure suite.

        Args:
            metrics: Recognition metrics.
            y_true: Optional labels for PR curve.
            y_score: Optional scores for PR curve.
            prefix: Optional filename prefix.

        Returns:
            Mapping of figure key → saved path.
        """
        paths: dict[str, Path] = {
            "confusion_matrix": self.save_confusion_matrix(
                metrics, filename=f"{prefix}confusion_matrix.png"
            ),
            "roc_curve": self.save_roc_curve(
                metrics, filename=f"{prefix}roc_curve.png"
            ),
        }
        if y_true is not None and y_score is not None:
            paths["pr_curve"] = self.save_precision_recall_curve(
                y_true,
                y_score,
                filename=f"{prefix}pr_curve.png",
            )
        return paths

    def _save(self, fig: Any, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path)
        fig.clf()
        import matplotlib.pyplot as plt

        plt.close(fig)
        return path.resolve()
