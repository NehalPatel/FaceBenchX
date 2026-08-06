# Paper Acceptance Checklist — per experiment run

Experiment ID: ______________________  
Config: ______________________  
Date: ______________________  
Operator: ______________________

Mark each item only after evidence exists on disk.

| ID | Check | Pass? | Evidence path / note |
|----|-------|-------|----------------------|
| A01 | Dataset validated (integrity OK) | ☐ | |
| A02 | Model loaded (real backend, not stub) | ☐ | |
| A03 | Embeddings generated (finite, expected dim) | ☐ | |
| A04 | Recognition / scoring completed | ☐ | |
| A05 | Metrics calculated (AUC/EER/F1 present) | ☐ | |
| A06 | HTML report generated | ☐ | |
| A07 | Markdown report generated | ☐ | |
| A08 | CSV generated | ☐ | |
| A09 | JSON metrics generated | ☐ | |
| A10 | ROC figure generated | ☐ | |
| A11 | Confusion matrix figure generated | ☐ | |
| A12 | Hardware / env recorded (`env.json`) | ☐ | |
| A13 | Manifest generated (`manifest.json`) | ☐ | _required after P1_ |
| A14 | No `--allow-stub` / stub backend | ☐ | |
| A15 | Seed recorded | ☐ | |
| A16 | Results curated into `paper/results/` | ☐ | |

**Overall:** ☐ ACCEPT / ☐ REJECT  

Reviewer signature: ______________________
