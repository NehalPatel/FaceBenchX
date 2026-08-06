# Cursor AI Prompt

You are an experienced AI Research Scientist, Computer Vision Engineer, and Software Architect.

Your goal is NOT to immediately generate code.

Your first responsibility is to help design a **publication-quality research framework** that can be used for benchmarking modern deep face recognition models.

Think like someone helping prepare work for a Scopus/Q1 journal publication.

---

# Project Title

**FaceBench: A Unified Benchmarking Framework for Comparative Evaluation of Modern Deep Face Recognition Models**

---

# Research Goal

Design and later implement a completely modular benchmarking framework named **FaceBench**.

FaceBench must be an independent research framework.

It should NOT depend on any surveillance application (such as VISTA).

Instead, future projects (including VISTA) should be able to use FaceBench as an external benchmarking library.

---

# Research Motivation

Many research papers compare face recognition models, but:

- they usually evaluate only accuracy
- they often compare only two or three models
- they rarely evaluate computational efficiency
- they rarely evaluate robustness
- they rarely provide reusable benchmarking software
- most comparisons are difficult to reproduce

FaceBench should solve these problems.

---

# Research Objectives

The framework should allow researchers to compare multiple deep face recognition models using an identical evaluation protocol.

It should answer questions such as:

- Which model produces the highest recognition accuracy?
- Which model is fastest?
- Which model requires the least computational resources?
- Which model is most robust to lighting changes?
- Which model handles blur best?
- Which model handles side poses best?
- Which model performs best on low-quality images?
- Which model scales well as enrolled identities increase?
- Which model is most suitable for real-time applications?

---

# Initial Models

Implement support for the following representative models.

## Classical Generation

- FaceNet

Reason:
Historical baseline using Triplet Loss.

---

- Dlib Face Recognition

Reason:
Widely used lightweight CPU baseline.

---

## Modern Generation

- Buffalo-L (InsightFace)

Reason:
Current production-quality recognition system based on ArcFace training.

---

- AdaFace

Reason:
Quality-adaptive face recognition for degraded images.

---

- MagFace

Reason:
Magnitude-aware representation learning that estimates embedding quality.

---

# Future Expansion

Design the architecture so future models can be added without changing existing code.

Examples:

- GhostFaceNet
- MobileFaceNet
- ElasticFace
- CurricularFace
- Partial FC
- SFace
- ArcFace implementations
- Future ONNX models

---

# Supported Datasets

Training datasets are NOT required because pretrained models will be used.

Evaluation datasets should include:

- LFW
- CFP-FP
- AgeDB-30
- Custom Dataset
- User Dataset

The framework must allow switching datasets through configuration files.

---

# Evaluation Pipeline

Dataset

↓

Image Loader

↓

Face Detection

↓

Face Alignment

↓

Recognition Model

↓

Embedding Generation

↓

Embedding Matcher

↓

Identity Prediction

↓

Metric Calculator

↓

Report Generator

---

# Matching Algorithms

Support multiple similarity methods.

Initially:

- Cosine Similarity
- Euclidean Distance

Thresholds should be configurable.

---

# Recognition Metrics

Generate:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC
- FAR
- FRR
- Equal Error Rate (EER)

---

# Computational Metrics

Measure:

- Average inference time
- Embedding generation time
- Recognition latency
- Throughput (FPS)
- CPU utilization
- GPU utilization
- RAM usage
- GPU memory usage
- Model loading time
- Model size

---

# Robustness Evaluation

Automatically evaluate performance under:

- Low illumination
- Bright illumination
- Blur
- Gaussian noise
- JPEG compression
- Rotation
- Side pose
- Occlusion
- Different resolutions

---

# Scalability Evaluation

Evaluate recognition performance for:

- 10 identities
- 100 identities
- 500 identities
- 1000 identities
- 5000 identities

Measure both:

Recognition accuracy

and

Recognition speed.

---

# Publication Figures

Automatically generate publication-ready visualizations.

Include:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- F1 Comparison
- Inference Time Comparison
- CPU Usage Comparison
- GPU Usage Comparison
- Memory Usage Comparison
- Accuracy Bar Charts
- Radar Charts
- Scalability Charts

Figures should be suitable for IEEE and Springer publications.

---

# Experiment Management

Support:

- Multiple datasets
- Multiple models
- Multiple thresholds
- Automatic experiment IDs
- Result versioning
- Batch execution
- Reproducible experiments
- Experiment history

---

# Configuration

All experiments must be configurable using YAML.

Configurable items:

Dataset

Recognition model

Threshold

Similarity method

CPU/GPU

Batch size

Output directory

Experiment name

---

# Logging

Log:

Hardware

Operating system

CUDA version

Python version

Library versions

Experiment metadata

Execution time

Errors

Metrics

Configuration used

---

# Folder Structure

Design a clean modular architecture.

Example:

FaceBench/

datasets/

models/

buffalo/

facenet/

dlib/

adaface/

magface/

matcher/

metrics/

evaluation/

visualization/

reports/

experiments/

configs/

utils/

core/

tests/

docs/

main.py

---

# Architecture Requirements

Use clean software engineering principles.

Follow:

- SOLID
- Strategy Pattern
- Factory Pattern
- Dependency Injection where appropriate

Every recognition model should implement a common interface.

Example:

load_model()

preprocess()

generate_embedding()

compare()

predict()

---

# Report Generation

Automatically generate:

HTML report

Markdown report

CSV metrics

JSON experiment file

Publication-ready graphs

Summary comparison tables

---

# Deliverables (Do NOT write code yet)

Produce a detailed design document containing:

1. Project Scope
2. Research Novelty
3. Literature Gap
4. Research Questions
5. Proposed Contributions
6. Functional Requirements
7. Non-Functional Requirements
8. System Architecture
9. Folder Structure
10. UML Class Diagram
11. Component Diagram
12. Data Flow Diagram
13. Sequence Diagram
14. Configuration Design
15. Experiment Workflow
16. Evaluation Methodology
17. Module Responsibilities
18. Technology Stack
19. Risk Analysis
20. Future Extensions
21. Git Branch Strategy
22. Testing Strategy
23. Documentation Plan
24. Milestone-Based Development Roadmap

---

# Important Constraints

- Keep FaceBench completely independent from VISTA.
- Build it as a reusable benchmarking framework.
- Make every module replaceable.
- Design for future extension with new models.
- Ensure experiments are fully reproducible.
- Prioritize clean architecture over quick implementation.
- Assume this framework will eventually be released as an open-source research project.

**Do not generate implementation code until the complete research architecture and design document have been reviewed and approved.**
