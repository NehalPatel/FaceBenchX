# Hardware profile — DESKTOP-NHQRICJ (LFW x Buffalo-L Baseline A)

profile_id: desktop_nhqricj
recorded_at: 2026-08-05
hostname: DESKTOP-NHQRICJ

cpu:
  model: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel
  cores_logical: 12

gpu:
  present: false
  model: null
  notes: onnxruntime CPUExecutionProvider only (CUDA EP unavailable)

cuda:
  version: None

os:
  name: Windows 10

python:
  version: |
    3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]

notes: |
  First paper run: experiments/20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c
  Elapsed 1.87h for 5958/6000 LFW pairs on CPU.
  Baseline A uses Buffalo-L vendor detector.
