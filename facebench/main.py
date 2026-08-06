"""FaceBench command-line interface (Phase 8)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from facebench import __version__
from facebench.core.config_loader import ConfigError, load_config
from facebench.core.experiment_manager import ExperimentManager
from facebench.core.orchestrator import ExperimentOrchestrator
from facebench.core.registry import get_default_registry
from facebench.datasets.factory import DatasetFactory
from facebench.metrics.computational import ComputationalMetrics
from facebench.metrics.recognition import ConfusionCounts, RecognitionMetrics
from facebench.models.factory import ModelFactory
from facebench.reports import ExperimentReportData, ReportGenerator
from facebench.utils.env_info import collect_environment_info
from facebench.utils.logging import get_logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="facebench",
        description=(
            "FaceBench - unified benchmarking framework for deep face "
            "recognition models."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"facebench {__version__}",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser(
        "validate-config",
        help="Load and validate a YAML experiment configuration.",
    )
    validate.add_argument("--config", required=True, type=Path)

    env_parser = subparsers.add_parser(
        "env",
        help="Print captured environment metadata as JSON.",
    )
    env_parser.add_argument("--pretty", action="store_true")

    init_parser = subparsers.add_parser(
        "init-experiment",
        help="Create an experiment directory and snapshot config/env.",
    )
    init_parser.add_argument("--config", required=True, type=Path)
    init_parser.add_argument("--output-root", type=Path, default=None)

    datasets = subparsers.add_parser(
        "list-datasets",
        help="List supported public datasets and categories.",
    )
    datasets.add_argument("--category", default=None)

    subparsers.add_parser(
        "list-models",
        help="List implemented recognition model adapters.",
    )

    run = subparsers.add_parser(
        "run",
        help="Run a FaceBench experiment from a YAML configuration.",
    )
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output-root", type=Path, default=None)
    run.add_argument(
        "--allow-stub",
        action="store_true",
        help="Use deterministic stub model backends (smoke / CI).",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Create experiment metadata without evaluating pairs.",
    )
    run.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip publication figure generation.",
    )
    run.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on verification pairs per dataset.",
    )

    history = subparsers.add_parser(
        "history",
        help="List experiment history under an experiments root.",
    )
    history.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments"),
        help="Experiments root directory (default: experiments).",
    )
    history.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent experiments to show.",
    )

    report = subparsers.add_parser(
        "report",
        help="Rebuild aggregated reports from an experiment summary JSON.",
    )
    report.add_argument(
        "--experiment-dir",
        required=True,
        type=Path,
        help="Path to an experiment output directory.",
    )
    report.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip comparison figure generation.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    setup_logging(level=getattr(args, "log_level", "INFO"))
    logger = get_logger("cli")

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "validate-config":
            return _cmd_validate_config(args.config, logger)
        if args.command == "env":
            return _cmd_env(pretty=args.pretty)
        if args.command == "init-experiment":
            return _cmd_init_experiment(args.config, args.output_root, logger)
        if args.command == "list-datasets":
            return _cmd_list_datasets(args.category)
        if args.command == "list-models":
            return _cmd_list_models()
        if args.command == "run":
            return _cmd_run(args, logger)
        if args.command == "history":
            return _cmd_history(args.output_root, args.limit)
        if args.command == "report":
            return _cmd_report(args.experiment_dir, not args.no_figures, logger)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        logger.exception("Unhandled error: %s", exc)
        return 1

    parser.print_help()
    return 0


def _cmd_validate_config(config_path: Path, logger: logging.Logger) -> int:
    """Validate a configuration file and print a short summary."""
    config = load_config(config_path)
    logger.info(
        "Config OK | experiment=%s | datasets=%d | models=%d | device=%s",
        config["experiment"]["name"],
        len(config["datasets"]),
        len(config["models"]),
        config["device"],
    )
    return 0


def _cmd_env(*, pretty: bool) -> int:
    """Print environment metadata."""
    info = collect_environment_info()
    indent = 2 if pretty else None
    json.dump(info.to_dict(), sys.stdout, indent=indent, default=str)
    sys.stdout.write("\n")
    return 0


def _cmd_init_experiment(
    config_path: Path,
    output_root: Path | None,
    logger: logging.Logger,
) -> int:
    """Create an experiment workspace from a config file."""
    config = load_config(config_path)
    root = output_root or Path(config["experiment"]["output_dir"])
    manager = ExperimentManager(root_dir=root)
    env = collect_environment_info().to_dict()
    record = manager.create_experiment(config, environment=env)
    logger.info(
        "Created experiment %s at %s",
        record.experiment_id,
        record.output_dir,
    )
    return 0


def _cmd_list_datasets(category: str | None) -> int:
    """List supported datasets."""
    registry = get_default_registry()
    implemented = set(DatasetFactory().available())
    if category:
        names = registry.list_by_category(category)
        print(f"Category: {category}")
        for name in names:
            mark = "*" if name in implemented else " "
            print(f"  {mark} {name}")
        return 0

    print("(* = adapter implemented)")
    for cat in registry.list_categories():
        print(f"{cat}:")
        for name in registry.list_by_category(cat):
            mark = "*" if name in implemented else " "
            print(f"  {mark} {name}")
    return 0


def _cmd_list_models() -> int:
    """List implemented recognition models."""
    print("Implemented models:")
    for name in ModelFactory().available():
        print(f"  - {name}")
    return 0


def _cmd_run(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run an experiment from YAML configuration."""
    config = load_config(args.config)
    if args.output_root is not None:
        config["experiment"]["output_dir"] = str(args.output_root)
    config.setdefault("_cli", {})
    config["_cli"]["dry_run"] = bool(args.dry_run)

    orchestrator = ExperimentOrchestrator(
        allow_stub=bool(args.allow_stub),
        generate_figures=not bool(args.no_figures),
        max_pairs=args.max_pairs,
    )
    result = orchestrator.run(config)
    logger.info(
        "Experiment complete | id=%s | runs=%d | dry_run=%s | out=%s",
        result.experiment.experiment_id,
        len(result.reports),
        result.dry_run,
        result.output_dir,
    )
    for report in result.reports:
        recog = report.recognition
        if recog is None:
            continue
        logger.info(
            "  %s / %s | acc=%.4f f1=%.4f auc=%.4f eer=%.4f",
            report.dataset_name,
            report.model_name,
            recog.accuracy,
            recog.f1,
            recog.auc,
            recog.eer,
        )
    return 0


def _cmd_history(output_root: Path, limit: int) -> int:
    """Print recent experiment history."""
    manager = ExperimentManager(root_dir=output_root)
    records = manager.list_history()
    if not records:
        print(f"No experiment history under {output_root.resolve()}")
        return 0
    shown = min(limit, len(records))
    print(f"Showing last {shown} of {len(records)} experiments:")
    for record in records[-limit:][::-1]:
        print(
            f"- {record.experiment_id} | {record.name} | "
            f"{record.created_at} | {record.output_dir}"
        )
    return 0


def _cmd_report(
    experiment_dir: Path,
    generate_figures: bool,
    logger: logging.Logger,
) -> int:
    """Rebuild aggregated comparison reports from summary.json."""
    summary_path = experiment_dir / "metrics" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing summary.json at {summary_path}. " "Run `facebench run` first."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    reports = [_rehydrate_report(item) for item in (payload.get("runs") or [])]
    generator = ReportGenerator(experiment_dir)
    generator.write_aggregated(
        reports,
        also_write_comparison_figures=generate_figures,
    )
    logger.info("Rebuilt aggregated reports under %s", experiment_dir / "aggregated")
    return 0


def _rehydrate_report(item: dict) -> ExperimentReportData:
    """Rebuild report dataclasses from serialized summary JSON."""
    recognition = None
    recog_raw = item.get("recognition")
    if isinstance(recog_raw, dict):
        confusion_raw = recog_raw.get("confusion") or {}
        recognition = RecognitionMetrics(
            threshold=float(recog_raw["threshold"]),
            accuracy=float(recog_raw["accuracy"]),
            precision=float(recog_raw["precision"]),
            recall=float(recog_raw["recall"]),
            f1=float(recog_raw["f1"]),
            far=float(recog_raw["far"]),
            frr=float(recog_raw["frr"]),
            auc=float(recog_raw["auc"]),
            eer=float(recog_raw["eer"]),
            eer_threshold=float(recog_raw["eer_threshold"]),
            confusion=ConfusionCounts(
                true_positive=int(confusion_raw.get("true_positive", 0)),
                true_negative=int(confusion_raw.get("true_negative", 0)),
                false_positive=int(confusion_raw.get("false_positive", 0)),
                false_negative=int(confusion_raw.get("false_negative", 0)),
            ),
            roc_fpr=np.asarray(recog_raw.get("roc_fpr") or [0.0, 1.0]),
            roc_tpr=np.asarray(recog_raw.get("roc_tpr") or [0.0, 1.0]),
            roc_thresholds=np.asarray(recog_raw.get("roc_thresholds") or [1.0, 0.0]),
            num_pairs=int(recog_raw.get("num_pairs", 0)),
            num_positive=int(recog_raw.get("num_positive", 0)),
            num_negative=int(recog_raw.get("num_negative", 0)),
        )

    computational = None
    comp_raw = item.get("computational")
    if isinstance(comp_raw, dict):
        computational = ComputationalMetrics(
            model_load_time_s=_maybe_float(comp_raw.get("model_load_time_s")),
            avg_inference_time_s=_maybe_float(comp_raw.get("avg_inference_time_s")),
            avg_embedding_time_s=_maybe_float(comp_raw.get("avg_embedding_time_s")),
            recognition_latency_s=_maybe_float(comp_raw.get("recognition_latency_s")),
            throughput_fps=_maybe_float(comp_raw.get("throughput_fps")),
            cpu_percent=_maybe_float(comp_raw.get("cpu_percent")),
            ram_rss_mb=_maybe_float(comp_raw.get("ram_rss_mb")),
            gpu_percent=_maybe_float(comp_raw.get("gpu_percent")),
            gpu_memory_mb=_maybe_float(comp_raw.get("gpu_memory_mb")),
            model_size_mb=_maybe_float(comp_raw.get("model_size_mb")),
            num_samples=int(comp_raw.get("num_samples") or 0),
            warmup_samples=int(comp_raw.get("warmup_samples") or 0),
            extra=dict(comp_raw.get("extra") or {}),
        )

    return ExperimentReportData(
        experiment_id=str(item.get("experiment_id", "")),
        experiment_name=str(item.get("experiment_name", "")),
        dataset_name=str(item.get("dataset_name", "")),
        model_name=str(item.get("model_name", "")),
        config=dict(item.get("config") or {}),
        recognition=recognition,
        computational=computational,
        figure_paths=dict(item.get("figure_paths") or {}),
        notes=str(item.get("notes") or ""),
    )


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
