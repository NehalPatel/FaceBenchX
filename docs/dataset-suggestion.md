# Public Evaluation Datasets

Do NOT create or use any custom dataset.

This benchmarking framework should only support well-established public datasets that are widely accepted in the face recognition research community.

The benchmark architecture must allow researchers to easily add future datasets, but the first version should support the following datasets.

---

## 1. LFW (Labeled Faces in the Wild)

Purpose:
General face verification benchmark.

Evaluation Focus:

- Overall recognition accuracy
- Precision
- Recall
- F1 Score
- ROC
- AUC

Challenge:

- Unconstrained images
- Natural variations

---

## 2. CFP-FP (Celebrities in Frontal-Profile)

Purpose:
Evaluate robustness against pose variation.

Evaluation Focus:

- Frontal vs Profile matching
- Pose robustness

Challenge:

- Extreme head rotation

---

## 3. CPLFW (Cross Pose LFW)

Purpose:
Evaluate recognition under combined pose and expression changes.

Evaluation Focus:

- Pose
- Expression
- Lighting

Challenge:

More difficult than standard LFW.

---

## 4. AgeDB-30

Purpose:
Evaluate age-invariant face recognition.

Evaluation Focus:

- Recognition across age gaps

Challenge:

Large age differences between images of the same individual.

---

## 5. TinyFace

Purpose:
Evaluate recognition on very low-resolution faces.

Evaluation Focus:

- Small face recognition
- Recognition quality degradation

Challenge:

Tiny face images.

---

## 6. AR Face Database

Purpose:
Evaluate robustness against occlusion and illumination.

Evaluation Focus:

- Sunglasses
- Scarves
- Lighting changes

Challenge:

Partial face visibility.

---

## 7. ChokePoint Dataset

Purpose:
Evaluate performance in surveillance environments.

Evaluation Focus:

- Multi-camera recognition
- CCTV scenarios
- Walking subjects

Challenge:

Real surveillance conditions.

---

## 8. YouTube Faces (YTF)

Purpose:
Evaluate video-based face recognition.

Evaluation Focus:

- Recognition consistency across frames
- Video verification

Challenge:

Motion blur, pose variation, compression artifacts.

---

# Dataset Categories

The framework should automatically organize datasets into research categories.

General Benchmark

- LFW

Pose Evaluation

- CFP-FP
- CPLFW

Age Evaluation

- AgeDB-30

Low Resolution Evaluation

- TinyFace

Occlusion Evaluation

- AR Face

Surveillance Evaluation

- ChokePoint

Video Evaluation

- YouTube Faces

---

# Dataset Loader Requirements

Every dataset loader should expose a common interface.

Example:

load_dataset()

load_identity_pairs()

load_gallery()

load_probe()

preprocess()

Each dataset implementation should follow the same interface so that experiments can be executed without changing the evaluation code.

---

# Dataset Configuration

Dataset selection must be configurable using YAML.

Example:

dataset:
name: LFW

or

dataset:
name: CFP-FP

or

dataset:
name: AgeDB-30

Changing the dataset should require only modifying the configuration file without changing the source code.

---

# Dataset Evaluation Strategy

The benchmark should support running:

• Single dataset experiments

• Multiple dataset experiments

• Batch evaluation across all supported datasets

The experiment manager should automatically generate independent reports for each dataset and an aggregated comparison report summarizing performance across all datasets.

---

# Important Constraint

Do not download or bundle any dataset inside the project repository.

Instead, design a dataset management system where users specify the local dataset path through the configuration file.

The framework should validate dataset integrity, report missing files, and provide clear instructions for preparing each supported dataset.
