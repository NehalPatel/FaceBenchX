# FaceBench Paper Experiment Suite

This directory locks **publication experiments**. It is separate from framework code under `facebench/`.

**Phase:** Scientific Validation (Phase 10)  
**Architecture:** Frozen — do not add models/datasets here; only configs, provenance, and curated outputs.

## Layout

| Path | Purpose |
|------|---------|
| `configs/` | Locked YAML experiment manifests |
| `hardware/` | Machine profiles used for reported runs |
| `weights/` | Weight acquisition notes + checksum ledger (**no binaries**) |
| `results/` | Curated result copies / pointers (bulk outputs gitignored) |
| `tables/` | Manuscript-ready tables |
| `figures/` | Manuscript-ready figures |
| `logs/` | Curated logs excerpts |
| `reproduction/` | Repro guides, acceptance tests, publication checklist |
| `manuscript/` | Paper drafts and claim↔evidence checks |

## Next campaign (deferred)

Full RQ1–RQ5 comparative campaign is **paused**. Resume from [docs/Experimental-Campaign-Roadmap.md](../docs/Experimental-Campaign-Roadmap.md).

## Primary experiment (complete)

**LFW × five models — Baseline B** (shared RetinaFace).

- Config: [configs/lfw_five_models_baseline_b.yaml](configs/lfw_five_models_baseline_b.yaml)
- Comparison report: [results/lfw_five_models_baseline_b_report.md](results/lfw_five_models_baseline_b_report.md)
- **Word paper:** [manuscript/FaceBench-LFW-FiveModels-BaselineB.docx](manuscript/FaceBench-LFW-FiveModels-BaselineB.docx)
- Markdown companion: [manuscript/FaceBench-LFW-FiveModels-BaselineB.md](manuscript/FaceBench-LFW-FiveModels-BaselineB.md)
- Figures (confusion + comparison): [figures/five_models_baseline_b/](figures/five_models_baseline_b/)
- Tables: [tables/five_models_baseline_b_metrics.csv](tables/five_models_baseline_b_metrics.csv), [tables/five_models_baseline_b_confusion.csv](tables/five_models_baseline_b_confusion.csv)
- Regenerate Word doc: `python paper/manuscript/generate_five_model_docx.py`

## Earlier experiment

**LFW × Buffalo-L** — Baseline A (single-model validation).

- Config template: [configs/lfw_buffalo_l.yaml](configs/lfw_buffalo_l.yaml)
- Baseline B template: [configs/lfw_buffalo_l_baseline_b.yaml](configs/lfw_buffalo_l_baseline_b.yaml)
- Local config: `configs/local/lfw_buffalo_l.yaml` (gitignored paths)
- Execution plan: [reproduction/lfw_buffalo_l_execution_plan.md](reproduction/lfw_buffalo_l_execution_plan.md)
- Execution summary: [results/lfw_buffalo_l_execution_summary.md](results/lfw_buffalo_l_execution_summary.md)
- Manuscript draft: [manuscript/FaceBench-LFW-BuffaloL-Draft.md](manuscript/FaceBench-LFW-BuffaloL-Draft.md)
- Publication tables: [tables/publication_tables.md](tables/publication_tables.md)
- Claim check: [manuscript/CLAIM_CHECK.md](manuscript/CLAIM_CHECK.md)
- Baseline B migration: [../docs/BaselineB-RetinaFace-Migration-Plan.md](../docs/BaselineB-RetinaFace-Migration-Plan.md)
- Baseline B tolerance check: `python paper/reproduction/verify_baseline_b_tolerance.py`

## Rules

1. Never commit dataset images or model weight files.
2. Never use `--allow-stub` for paper runs.
3. After each accepted run, copy `manifest.json` (when available), key metrics, and figure paths into `results/`.
4. Record hardware in `hardware/` before claiming numbers in the manuscript.

## Related docs

- [docs/Phase10-Scientific-Validation-Plan.md](../docs/Phase10-Scientific-Validation-Plan.md)
- [docs/FaceBench-Design.md](../docs/FaceBench-Design.md)
- [docs/datasets/lfw.md](../docs/datasets/lfw.md)
