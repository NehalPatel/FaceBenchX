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

## First experiment

**LFW × Buffalo-L** — completed (Baseline A).

- Config template: [configs/lfw_buffalo_l.yaml](configs/lfw_buffalo_l.yaml)
- Local config: `configs/local/lfw_buffalo_l.yaml` (gitignored paths)
- Execution plan: [reproduction/lfw_buffalo_l_execution_plan.md](reproduction/lfw_buffalo_l_execution_plan.md)
- Execution summary: [results/lfw_buffalo_l_execution_summary.md](results/lfw_buffalo_l_execution_summary.md)
- **Manuscript draft:** [manuscript/FaceBench-LFW-BuffaloL-Draft.md](manuscript/FaceBench-LFW-BuffaloL-Draft.md)
- Publication tables: [tables/publication_tables.md](tables/publication_tables.md)
- Claim check: [manuscript/CLAIM_CHECK.md](manuscript/CLAIM_CHECK.md)

## Rules

1. Never commit dataset images or model weight files.
2. Never use `--allow-stub` for paper runs.
3. After each accepted run, copy `manifest.json` (when available), key metrics, and figure paths into `results/`.
4. Record hardware in `hardware/` before claiming numbers in the manuscript.

## Related docs

- [docs/Phase10-Scientific-Validation-Plan.md](../docs/Phase10-Scientific-Validation-Plan.md)
- [docs/FaceBench-Design.md](../docs/FaceBench-Design.md)
- [docs/datasets/lfw.md](../docs/datasets/lfw.md)
