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

## Five-model Baseline B LFW run

Local weight roots (not in git). Checksums: `paper/weights/checksums.sha256`.

| Model | Local path / source |
|-------|---------------------|
| FaceNet | `facenet-pytorch` InceptionResnetV1 `pretrained=vggface2` (auto-download) |
| Dlib | `D:/datasets/weights/dlib/` — `shape_predictor_5_face_landmarks.dat`, `dlib_face_recognition_resnet_model_v1.dat` |
| Buffalo-L | `C:/Users/Nehal/.insightface` InsightFace pack |
| AdaFace | `D:/datasets/weights/adaface/adaface_ir50_ms1mv2.ckpt` (IR-50 MS1MV2) |
| MagFace | `D:/datasets/weights/magface/magface_iresnet50_ms1mv2.pth` (iResNet50 MS1MV2 DDP) |

Architecture constructors for AdaFace/MagFace live under `paper/reproduction/arch/` (framework adapters unchanged).
