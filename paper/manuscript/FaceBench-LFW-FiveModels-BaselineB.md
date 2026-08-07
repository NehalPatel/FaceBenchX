# FaceBench: A Comparative Evaluation of Five Face Recognition Models on LFW under a Shared RetinaFace Preprocessing Pipeline

**Manuscript status:** Complete Baseline B multi-model draft  
**Experiment ID:** `20260806T192706Z_paper_lfw_five_models_baseline_b_f86260f8`  
**Framework:** FaceBench 0.1.0  
**Device:** NVIDIA GeForce RTX 3050 (`cuda:0`)

## Abstract

Fair comparison of face recognition models is often undermined by inconsistent detection/alignment pipelines, accuracy-only reporting, and incomplete experimental provenance. We present FaceBench and a locked LFW View-2 verification study of FaceNet, Dlib, Buffalo-L, AdaFace, and MagFace under shared RetinaFace/SCRFD Baseline B preprocessing with cosine similarity at threshold 0.40. Buffalo-L achieves the highest Acc@0.40 (0.9828, FAR=0); FaceNet achieves the highest AUC (0.9933). Dlib shows strong AUC/EER but Acc@0.40 collapse due to score-scale mismatch. Confusion matrices and comparison charts are included.

**Keywords:** face recognition, benchmarking, LFW, RetinaFace, FaceNet, Dlib, Buffalo-L, AdaFace, MagFace

## Results summary (Table 2)

| Model | Acc | Prec | Rec | F1 | AUC | EER | FAR | FRR | TP | TN | FP | FN |
|-------|-----|------|-----|----|-----|-----|-----|-----|----|----|----|----|
| FaceNet | 0.9758 | 0.9886 | 0.9628 | 0.9756 | 0.9933 | 0.0275 | 0.0111 | 0.0372 | 2874 | 2940 | 33 | 111 |
| Dlib | 0.5009 | 0.5009 | 1.0000 | 0.6675 | 0.9922 | 0.0234 | 1.0000 | 0.0000 | 2952 | 0 | 2941 | 0 |
| Buffalo-L | 0.9828 | 1.0000 | 0.9657 | 0.9826 | 0.9899 | 0.0216 | 0.0000 | 0.0343 | 2873 | 2961 | 0 | 102 |
| AdaFace | 0.8098 | 0.9995 | 0.6208 | 0.7659 | 0.9711 | 0.0730 | 0.0003 | 0.3792 | 1853 | 2972 | 1 | 1132 |
| MagFace | 0.8864 | 0.9432 | 0.8228 | 0.8789 | 0.9515 | 0.1079 | 0.0498 | 0.1772 | 2456 | 2825 | 148 | 529 |

## Figures

- Comparison: `paper/figures/five_models_baseline_b/comparison/`
- Confusion matrices: `paper/figures/five_models_baseline_b/confusion/`
- Word manuscript: `paper/manuscript/FaceBench-LFW-FiveModels-BaselineB.docx`

