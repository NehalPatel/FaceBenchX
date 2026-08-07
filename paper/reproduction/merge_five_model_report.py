"""Merge corrected MagFace metrics into the primary five-model report."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facebench.metrics.computational import ComputationalMetrics  # noqa: E402
from facebench.metrics.recognition import ConfusionCounts, RecognitionMetrics  # noqa: E402
from facebench.reports import ReportGenerator  # noqa: E402
from facebench.reports.types import ExperimentReportData  # noqa: E402

PRIMARY = (
    ROOT
    / "experiments"
    / "20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8"
)
SUMMARY = ROOT / "paper" / "results" / "lfw_five_models_baseline_b_report.md"


def _recognition_from_dict(data: dict) -> RecognitionMetrics:
    conf = data["confusion"]
    if isinstance(conf, dict):
        confusion = ConfusionCounts(**{k: int(conf[k]) for k in (
            "true_positive", "true_negative", "false_positive", "false_negative"
        )})
    else:
        raise ValueError("confusion missing")
    return RecognitionMetrics(
        threshold=float(data["threshold"]),
        accuracy=float(data["accuracy"]),
        precision=float(data["precision"]),
        recall=float(data["recall"]),
        f1=float(data["f1"]),
        far=float(data["far"]),
        frr=float(data["frr"]),
        auc=float(data["auc"]),
        eer=float(data["eer"]),
        eer_threshold=float(data["eer_threshold"]),
        confusion=confusion,
        roc_fpr=np.asarray(data.get("roc_fpr") or [], dtype=np.float64),
        roc_tpr=np.asarray(data.get("roc_tpr") or [], dtype=np.float64),
        roc_thresholds=np.asarray(data.get("roc_thresholds") or [], dtype=np.float64),
        num_pairs=int(data.get("num_pairs") or 0),
        num_positive=int(data.get("num_positive") or 0),
        num_negative=int(data.get("num_negative") or 0),
        extra=dict(data.get("extra") or {}),
    )


def _computational_from_dict(data: dict | None) -> ComputationalMetrics | None:
    if not data:
        return None
    # ComputationalMetrics fields vary; keep only known constructor keys via to_dict roundtrip
    known = {
        "model_load_time_s",
        "avg_embedding_time_s",
        "avg_match_time_s",
        "throughput_embeddings_per_s",
        "cpu_percent",
        "ram_rss_mb",
        "gpu_percent",
        "gpu_memory_mb",
        "model_size_mb",
        "peak_ram_mb",
        "peak_gpu_memory_mb",
        "num_embeddings_timed",
        "num_matches_timed",
        "extra",
    }
    kwargs = {k: data[k] for k in known if k in data}
    return ComputationalMetrics(**kwargs)


def _load_run(exp_dir: Path, model: str) -> tuple[ExperimentReportData, dict]:
    run_dir = exp_dir / "runs" / f"LFW__{model}"
    ver = json.loads((run_dir / "metrics" / "verification.json").read_text(encoding="utf-8"))
    exp_json = {}
    exp_path = run_dir / "metrics" / "experiment.json"
    if exp_path.is_file():
        exp_json = json.loads(exp_path.read_text(encoding="utf-8"))
    recog = _recognition_from_dict(ver["recognition"])
    comp = _computational_from_dict(ver.get("computational") or exp_json.get("computational"))
    report = ExperimentReportData(
        experiment_id=PRIMARY.name,
        experiment_name="paper_lfw_five_models_baseline_b",
        dataset_name="LFW",
        model_name=model,
        config=exp_json.get("config") or {},
        recognition=recog,
        computational=comp,
        figure_paths=exp_json.get("figure_paths") or {},
        notes=exp_json.get("notes") or "",
    )
    model_manifest = {}
    mm = run_dir / "model_manifest.json"
    if mm.is_file():
        model_manifest = json.loads(mm.read_text(encoding="utf-8"))
    return report, model_manifest


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: merge_five_model_report.py <magface_experiment_dir>", file=sys.stderr)
        return 2
    mag_dir = Path(sys.argv[1])
    src = mag_dir / "runs" / "LFW__magface"
    dst = PRIMARY / "runs" / "LFW__magface"
    if not src.is_dir():
        print(f"Missing MagFace run: {src}", file=sys.stderr)
        return 1
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    models = ["facenet", "dlib", "buffalo_l", "adaface", "magface"]
    reports = []
    manifests = []
    for model in models:
        report, mm = _load_run(PRIMARY, model)
        reports.append(report)
        manifests.append(mm)

    gen = ReportGenerator(PRIMARY)
    rows = gen.write_aggregated(reports, also_write_comparison_figures=True)

    summary_path = PRIMARY / "metrics" / "summary.json"
    summary = {
        "experiment_id": PRIMARY.name,
        "protocol": "Baseline B shared RetinaFace bbox_margin",
        "runs": [r.to_dict() for r in reports],
        "comparison_rows": [row.to_dict() for row in rows],
        "magface_correction": {
            "note": "MagFace re-scored with official BGR [0,1] input wrapper",
            "source_experiment": str(mag_dir),
            "corrected_at": datetime.now(timezone.utc).isoformat(),
            "metrics": manifests[-1].get("recognition_metrics"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    manifest_path = PRIMARY / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"] = manifests
    manifest["magface_correction"] = summary["magface_correction"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# LFW x Five Models - Baseline B Comparison Report",
        "",
        "- **Status:** SUCCESS",
        f"- **Experiment ID:** `{PRIMARY.name}`",
        f"- **Output dir:** `{PRIMARY}`",
        "- **Device:** `cuda:0`",
        "- **Matching:** cosine @ 0.40",
        "- **Detection:** RetinaFace/SCRFD shared, `crop_mode=bbox_margin`, margin=0.35",
        "- **Pairs:** 6000 View-2 (per-model scored count may drop on embed failures)",
        "- **MagFace correction:** re-scored with official BGR `[0,1]` input adaptation "
        "(framework preprocess unchanged).",
        "",
        "## Per-model metrics",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | AUC | EER | FAR | FRR | Scored | Skipped |",
        "|-------|----------|-----------|--------|----|-----|-----|-----|-----|--------|---------|",
    ]
    for report, mm in zip(reports, manifests):
        rec = report.recognition
        assert rec is not None
        scored = mm.get("pairs_scored", rec.num_pairs)
        skipped = mm.get("pairs_skipped", "")
        lines.append(
            f"| {report.model_name} | {rec.accuracy:.4f} | {rec.precision:.4f} | "
            f"{rec.recall:.4f} | {rec.f1:.4f} | {rec.auc:.4f} | {rec.eer:.4f} | "
            f"{rec.far:.4f} | {rec.frr:.4f} | {scored} | {skipped} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Aggregated MD: `{PRIMARY / 'aggregated' / 'comparison.md'}`",
            f"- Aggregated HTML: `{PRIMARY / 'aggregated' / 'comparison.html'}`",
            f"- Aggregated CSV: `{PRIMARY / 'aggregated' / 'comparison.csv'}`",
            f"- Weight checksums: `{ROOT / 'paper' / 'weights' / 'checksums.sha256'}`",
            f"- Comparison report: `{SUMMARY}`",
            "",
            "## Protocol notes",
            "",
            "- Shared RetinaFace alignment once; crops reused across models.",
            "- Fixed decision threshold 0.40 cosine for all models.",
            "- Dlib AUC/EER are strong while accuracy@0.40 is ~0.50 because dlib "
            "cosine scores sit above 0.40 for both classes; prefer EER threshold operationally.",
            "- AdaFace/MagFace architectures injected via `paper/reproduction/arch`.",
            "- MagFace uses paper-local BGR `[0,1]` wrapper to match official MagFace inference.",
            "",
        ]
    )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    curated = ROOT / "paper" / "results" / f"lfw_five_models_{PRIMARY.name}"
    curated.mkdir(parents=True, exist_ok=True)
    (curated / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(PRIMARY / "aggregated" / "comparison.md", curated / "comparison.md")
    shutil.copy2(PRIMARY / "aggregated" / "comparison.csv", curated / "comparison.csv")
    shutil.copy2(manifest_path, curated / "manifest.json")
    (curated / "MANIFEST_POINTER.txt").write_text(
        f"experiment_dir={PRIMARY}\nmanifest={manifest_path}\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
