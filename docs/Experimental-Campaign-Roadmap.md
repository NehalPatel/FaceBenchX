# FaceBenchX Experimental Campaign Roadmap (Deferred)

**Status:** Saved for later — do not execute until resumed.  
**Saved:** 2026-08-08  
**Cursor plan:** `RQ Gap Analysis Roadmap` (deferred)

This document parks the gap analysis between the completed LFW × five-model Baseline B manuscript and the full RQ1–RQ5 comparative publication. Framework remains frozen; next work is scientific experimentation under paper-local configs/runners only.

**Completed so far**

- FaceBenchX feature-complete framework
- LFW × five models under shared RetinaFace Baseline B
- Manuscript / Word draft: `paper/manuscript/FaceBench-LFW-FiveModels-BaselineB.docx`
- Experiment ID: `20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8`

**Authority**

- `docs/FaceBench-Design.md` §4 / §16
- `docs/Phase10-Scientific-Validation-Plan.md`
- Constraint: no new framework features unless a verified bug affects experimental validity

---

## RQ answerability today

| RQ | Answerable? | Why |
|----|-------------|-----|
| RQ1 Highest recognition accuracy | Partial (LFW only) | Five-model metrics on LFW View-2 only |
| RQ2 Best computational efficiency | No | Need dedicated uncached profiling |
| RQ3 Most robust across categories | No | No category-dataset locked runs yet |
| RQ4 Gallery-size scaling | No | Scalability ladder not run |
| RQ5 Real-time suitability | No | Needs RQ1+RQ2 Pareto synthesis |

---

## Priority catalog (resume order)

| ID | Experiment | Serves | Effort |
|----|------------|--------|--------|
| P1 | CFP-FP, CPLFW, AgeDB-30 verification × 5 models | RQ1, RQ3 (pose/age) | M–L |
| P2 | Dedicated compute profiling (warm-up, uncached) | RQ2, RQ5 | S–M |
| P3 | McNemar / fold stats (LFW scores already on disk) | Claims defense | S–M |
| P4 | TinyFace identification | RQ3 low-res, RQ4 method | M–L |
| P5 | AR Face verification | RQ3 occlusion | M |
| P6 | ChokePoint verification | RQ3 surveillance | L |
| P7 | Scalability ladders 10→5000 | RQ4 | M |
| P8 | YTF (first-frame + caveat, or mean-pool if approved) | RQ3 video | M–L |
| P9 | Synthetic LFW robustness (appendix only) | Secondary RQ3 | S–M |
| P10 | Pareto + final comparative manuscript + CLAIM_CHECK | RQ5 + paper | L |

**Minimum credible RQ1–RQ5 path:** P1 + P2 + P3 + P4 + P7 + P10 (add P5/P6/P8 for full RQ3 coverage).

---

## Execution waves (when resumed)

### Wave 0
Acquire/validate CFP-FP, CPLFW, AgeDB-30; lock Baseline B config fragment; plan align/embed caches.

### Wave 1 (highest ROI)
P1 → P2 → P3 → interim results pack.

### Wave 2
P4 TinyFace ID → P5 AR Face → P6 ChokePoint → stats per dataset.

### Wave 3
P7 scalability → P8 YTF → P9 optional appendix → P10 final manuscript.

**Reuse rules:** shared RetinaFace once per image; disk crop/embed caches; never profile latency on cached accuracy loops; start AR/ChokePoint/TinyFace access requests early.

---

## How to resume

1. Open this file and the Cursor plan `RQ Gap Analysis Roadmap`.
2. Start Wave 0 (dataset roots + integrity).
3. Do not expand framework scope unless a validity bug is confirmed.
