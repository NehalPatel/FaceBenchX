"""Experiment orchestrator for FaceBench evaluation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from facebench.core.experiment_manager import ExperimentManager, ExperimentRecord
from facebench.datasets.factory import DatasetFactory
from facebench.evaluation import (
    run_identification,
    run_robustness_suite,
    run_scalability_ladder,
    run_verification,
)
from facebench.matcher import create_matcher
from facebench.metrics import MetricCalculator
from facebench.models.backends import DeterministicStubBackend
from facebench.models.factory import ModelFactory
from facebench.reports import ExperimentReportData, ReportGenerator
from facebench.utils.env_info import collect_environment_info
from facebench.utils.logging import get_logger


@dataclass(slots=True)
class RunResult:
    """Result of an orchestrated evaluation run.

    Attributes:
        experiment: Created experiment record.
        reports: Per dataset/model report payloads.
        output_dir: Experiment output directory.
        dry_run: Whether evaluation was skipped.
        extra: Extensible metadata.
    """

    experiment: ExperimentRecord
    reports: list[ExperimentReportData] = field(default_factory=list)
    output_dir: str = ""
    dry_run: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class ExperimentOrchestrator:
    """Coordinate config → dataset → model → metrics → report pipelines."""

    def __init__(
        self,
        *,
        allow_stub: bool = False,
        generate_figures: bool = True,
        max_pairs: int | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            allow_stub: When ``True``, construct recognizers with
                ``allow_stub=True`` / deterministic backends for smoke runs.
            generate_figures: When ``True``, emit publication figures.
            max_pairs: Optional cap on verification pairs per dataset.
        """
        self.allow_stub = allow_stub
        self.generate_figures = generate_figures
        self.max_pairs = max_pairs
        self._logger = get_logger("core.orchestrator")
        self._datasets = DatasetFactory()
        self._models = ModelFactory()
        self._metrics = MetricCalculator()

    def run(self, config: dict[str, Any]) -> RunResult:
        """Execute a configuration (single or multi dataset/model).

        Args:
            config: Normalized experiment configuration.

        Returns:
            :class:`RunResult` with experiment metadata and reports.

        Raises:
            ValueError: If datasets or models are missing from config.
            DatasetFactoryError: If a dataset adapter cannot be created.
            ModelFactoryError: If a model adapter cannot be created.
        """
        return self.run_batch(config)

    def run_batch(self, config: dict[str, Any]) -> RunResult:
        """Execute the full dataset × model evaluation matrix.

        Args:
            config: Normalized batch experiment configuration.

        Returns:
            Aggregated :class:`RunResult`.
        """
        datasets = config.get("datasets") or []
        models = config.get("models") or []
        if not datasets:
            raise ValueError("Configuration contains no datasets")
        if not models:
            raise ValueError("Configuration contains no models")

        output_root = Path(config["experiment"]["output_dir"])
        manager = ExperimentManager(root_dir=output_root)
        env = collect_environment_info().to_dict()
        record = manager.create_experiment(config, environment=env)
        output_dir = Path(record.output_dir)
        log_file = output_dir / "logs" / "run.log"
        self._logger.info(
            "Starting experiment %s (%d datasets × %d models)",
            record.experiment_id,
            len(datasets),
            len(models),
        )

        dry_run = bool(config.get("_cli", {}).get("dry_run", False))
        if dry_run:
            self._logger.info("Dry-run enabled; skipping evaluation")
            (output_dir / "logs" / "dry_run.txt").write_text(
                f"Dry-run for {record.experiment_id}\n",
                encoding="utf-8",
            )
            return RunResult(
                experiment=record,
                reports=[],
                output_dir=str(output_dir),
                dry_run=True,
                extra={"log_file": str(log_file)},
            )

        matching = config.get("matching", {})
        method = str(matching.get("method", "cosine"))
        threshold = matching.get("threshold")
        threshold_f = float(threshold) if threshold is not None else None
        device = str(config.get("device", "cpu"))
        matcher = create_matcher(method)
        max_pairs = self.max_pairs
        eval_cfg = config.get("evaluation", {})
        if max_pairs is None and isinstance(eval_cfg, dict):
            maybe = eval_cfg.get("max_pairs")
            if maybe is not None:
                max_pairs = int(maybe)
        mode = str(eval_cfg.get("mode", "verification")).lower()
        robustness_cfg = config.get("robustness") or {}
        scalability_cfg = config.get("scalability") or {}
        seed = int(config.get("experiment", {}).get("seed", 42))

        reports: list[ExperimentReportData] = []
        report_root = ReportGenerator(output_dir)

        for dataset_entry in datasets:
            dataset_name = dataset_entry["name"]
            root_path = dataset_entry.get("root_path")
            if not root_path:
                raise ValueError(
                    f"Dataset {dataset_name!r} is missing root_path in config"
                )
            dataset = self._datasets.create(dataset_name, root_path)
            integrity = dataset.validate_integrity()
            if not integrity.ok:
                messages = "; ".join(integrity.messages) or "integrity failed"
                prep = integrity.prep_doc or dataset.prep_doc
                raise FileNotFoundError(
                    f"Dataset {dataset_name} integrity check failed: {messages}. "
                    f"See {prep}"
                )

            pairs = dataset.load_identity_pairs()
            if max_pairs is not None:
                pairs = pairs[: max(0, max_pairs)]

            for model_entry in models:
                model_name = model_entry["name"]
                self._logger.info(
                    "Evaluating dataset=%s model=%s mode=%s",
                    dataset_name,
                    model_name,
                    mode,
                )
                recognizer = self._build_recognizer(model_name, device, model_entry)
                profiler = self._metrics.create_profiler(warmup=2)
                weights = model_entry.get("weights_path")
                if weights:
                    profiler.set_model_size_mb(self._metrics.model_size(weights))

                def _load(model: Any = recognizer) -> None:
                    model.load_model(device)

                profiler.time_model_load(_load)

                combo_dir = (
                    output_dir / "runs" / f"{_safe(dataset_name)}__{_safe(model_name)}"
                )
                combo_reports = ReportGenerator(combo_dir)
                extra: dict[str, Any] = {"mode": mode}
                recognition = None
                computational = profiler.summarize()
                labels: list[int] | None = None
                scores: list[float] | None = None
                notes = ""
                verification = None

                if mode == "verification":
                    if not pairs:
                        raise ValueError(
                            f"Dataset {dataset_name} produced no identity pairs"
                        )
                    verification = run_verification(
                        pairs,
                        recognizer,
                        matcher,
                        profiler,
                        threshold=threshold_f,
                        metrics=self._metrics,
                    )
                    recognition = verification.recognition
                    computational = verification.computational
                    labels = verification.labels
                    scores = verification.scores
                    extra["verification"] = verification.to_dict()
                    self._write_json(
                        combo_dir / "metrics" / "verification.json",
                        verification.to_dict(),
                    )
                else:
                    gallery = dataset.load_gallery()
                    probe = dataset.load_probe()
                    identification = run_identification(
                        gallery,
                        probe,
                        recognizer,
                        matcher,
                        profiler,
                    )
                    computational = profiler.summarize()
                    extra["identification"] = identification.to_dict()
                    notes = (
                        f"Identification rank-1={identification.rank1_accuracy:.4f} "
                        f"gallery={identification.num_gallery} "
                        f"probes={identification.num_probes}"
                    )
                    self._write_json(
                        combo_dir / "metrics" / "identification.json",
                        identification.to_dict(),
                    )

                if bool(robustness_cfg.get("enabled")):
                    if not pairs:
                        raise ValueError(
                            "Robustness evaluation requires identity pairs "
                            f"from dataset {dataset_name}"
                        )
                    baseline = verification if mode == "verification" else None
                    robustness = run_robustness_suite(
                        pairs,
                        recognizer,
                        matcher,
                        transforms=list(robustness_cfg.get("transforms") or []),
                        baseline=baseline,
                        threshold=threshold_f,
                        metrics=self._metrics,
                        base_dataset=dataset_name,
                        profiler_factory=lambda: self._metrics.create_profiler(
                            warmup=0
                        ),
                    )
                    extra["robustness"] = robustness.to_dict()
                    self._write_json(
                        combo_dir / "metrics" / "robustness.json",
                        robustness.to_dict(),
                    )
                    self._logger.info(
                        "Robustness complete for %s / %s (%d transforms)",
                        dataset_name,
                        recognizer.name,
                        len(robustness.conditions),
                    )

                if bool(scalability_cfg.get("enabled")):
                    ladder = run_scalability_ladder(
                        dataset,
                        recognizer,
                        matcher,
                        identity_counts=list(
                            scalability_cfg.get("identity_counts") or []
                        ),
                        seed=seed,
                        profiler=self._metrics.create_profiler(warmup=0),
                    )
                    extra["scalability"] = ladder.to_dict()
                    self._write_json(
                        combo_dir / "metrics" / "scalability.json",
                        ladder.to_dict(),
                    )
                    if self.generate_figures:
                        self._maybe_scalability_figure(combo_dir, ladder)
                    self._logger.info(
                        "Scalability complete for %s / %s (%d points)",
                        dataset_name,
                        recognizer.name,
                        len(ladder.points),
                    )

                report = combo_reports.write_per_dataset(
                    experiment_id=record.experiment_id,
                    experiment_name=config["experiment"]["name"],
                    dataset_name=dataset_name,
                    model_name=recognizer.name,
                    recognition=recognition,
                    computational=computational,
                    config=config,
                    y_true=labels,
                    y_score=scores,
                    generate_figures=self.generate_figures and recognition is not None,
                    notes=notes,
                )
                report.extra.update(extra)
                # Persist enriched summary beside standard report artifacts.
                self._write_json(
                    combo_dir / "metrics" / "run_extra.json",
                    extra,
                )
                reports.append(report)
                if recognition is not None:
                    self._logger.info(
                        "Finished %s / %s | AUC=%.4f EER=%.4f F1=%.4f",
                        dataset_name,
                        recognizer.name,
                        recognition.auc,
                        recognition.eer,
                        recognition.f1,
                    )
                else:
                    self._logger.info(
                        "Finished %s / %s | %s",
                        dataset_name,
                        recognizer.name,
                        notes or mode,
                    )

        if len(reports) > 1 or bool(eval_cfg.get("aggregate_report", False)):
            report_root.write_aggregated(reports)

        summary_path = output_dir / "metrics" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "experiment_id": record.experiment_id,
                    "runs": [item.to_dict() for item in reports],
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        return RunResult(
            experiment=record,
            reports=reports,
            output_dir=str(output_dir),
            dry_run=False,
            extra={"summary": str(summary_path)},
        )

    def _build_recognizer(
        self,
        model_name: str,
        device: str,
        model_entry: dict[str, Any],
    ) -> Any:
        """Construct a recognizer, optionally with stub backends.

        Args:
            model_name: Model config name.
            device: Device string.
            model_entry: Model config mapping.

        Returns:
            Loaded-capable recognizer instance.
        """
        kwargs = {key: value for key, value in model_entry.items() if key != "name"}
        if self.allow_stub:
            kwargs["allow_stub"] = True
            dim = 128 if model_name.lower() in {"dlib", "dlib_fr"} else 512
            kwargs["backend"] = DeterministicStubBackend(embedding_dim=dim)
        return self._models.create(model_name, device=device, **kwargs)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _maybe_scalability_figure(self, combo_dir: Path, ladder: Any) -> None:
        """Emit a scalability chart when matplotlib is available."""
        counts: list[int] = []
        accuracy: list[float] = []
        latency: list[float] = []
        for point in ladder.points:
            if point.identification is None:
                continue
            counts.append(point.identity_count)
            accuracy.append(point.identification.rank1_accuracy)
            latency.append(point.identification.mean_search_latency_s)
        if not counts:
            return
        try:
            from facebench.visualization.figures import FigureGenerator
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._logger.warning("Skipping scalability figure: %s", exc)
            return
        try:
            figures = FigureGenerator(combo_dir / "figures")
            path = figures.save_scalability_chart(
                counts,
                accuracy,
                latency,
                title=f"Scalability — {ladder.dataset_name}",
            )
            self._logger.info("Wrote scalability figure %s", path)
        except Exception as exc:  # pragma: no cover
            self._logger.warning("Failed to write scalability figure: %s", exc)


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
