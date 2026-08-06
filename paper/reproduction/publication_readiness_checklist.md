# Publication Readiness Checklist

Mirror of Phase 10 Objective 9. Update as gates pass.

## Framework

- [x] Framework architecture frozen (Phase 10 rule)
- [x] Dataset adapters implemented (8 public)
- [x] Dataset adapters validated on real LFW
- [x] Recognition adapters implemented (5 models)
- [x] Recognition adapters validated with real weights (Buffalo-L)
- [ ] Detection/alignment finalized (shared RetinaFace = Baseline B)
- [x] Metrics verified (unit tests)
- [ ] Metrics verified vs published baselines
- [x] Reports verified (unit tests)
- [x] Reports verified on real run
- [x] Figures verified on real run

## Reproducibility

- [ ] Experiment reproducibility verified (two-run tolerance)
- [ ] Hardware documented (`paper/hardware/`) — template only
- [x] Environment recorded (`env.json` + `manifest.json`)
- [x] `manifest.json` produced for first paper run (via paper runner)
- [x] Weight checksums recorded (`w600k_r50.onnx`)

## Benchmarks

- [x] First benchmark completed (LFW × Buffalo-L)
- [ ] Five-model benchmark completed
- [ ] Multi-dataset benchmark completed
- [x] Manuscript tables generated (preliminary metrics CSV/MD)
- [ ] Supplementary material prepared

## Paper suite scaffolding (P0)

- [x] `paper/` tree created
- [x] Locked LFW × Buffalo-L config template
- [x] Execution plan written
- [x] Acceptance checklist template
- [x] Weights / hardware templates
