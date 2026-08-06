# Paper weights ledger (no binaries in git)

FaceBench does **not** redistribute pretrained weights. Store files outside the repo and record checksums here after download.

## Buffalo-L (InsightFace) — first paper experiment

| Item | Value |
|------|-------|
| Pack name | `buffalo_l` |
| Typical root | `/data/weights/insightface` (InsightFace `root`) |
| Contains | detection + recognition ONNX models under `models/buffalo_l/` |
| Install hint | `pip install 'facebench[buffalo]'` then first `FaceAnalysis` prepare, **or** pre-copy pack |
| SHA256 | _fill after local download_ |

### Checklist

- [ ] Pack present locally
- [ ] SHA256 recorded in `checksums.sha256`
- [ ] Path matches `paper/configs/lfw_buffalo_l.yaml` → `model.weights_path`
- [ ] Paper run did **not** use stub backend

## Future models (do not download until scheduled)

| Model | Notes |
|-------|-------|
| FaceNet | `facenet-pytorch` VGGFace2 or local `.pt` |
| Dlib | `shape_predictor_*.dat` + `dlib_face_recognition_resnet_model_v1.dat` |
| AdaFace | Official checkpoint + architecture (see Phase 10 plan) |
| MagFace | Official checkpoint + architecture (see Phase 10 plan) |
