# Model Doctor — Local Handoff

## What is included

This package contains the completed college-project scope:

- validated 100-model TinyCNN/MNIST research zoo
- trained Random Forest Model Doctor
- four controlled fault classes
- Grad-CAM evidence
- six Professor Demo proof checkpoints
- React/Vite frontend
- FastAPI live upload backend
- genuine live `TinyCNN + MNIST` upload-and-diagnose mode
- tests and professor-demo documentation

## Quick start

### Option A — two terminals

Backend, from project root:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn src.api:app --reload --port 8000
```

Frontend, second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

### Option B — Windows launcher

Double-click:

```text
START_MODEL_DOCTOR.bat
```

It opens backend and frontend terminals separately.

## Professor demo

### Controlled proof

Open **Professor Proof Library** and use the six ready cases. These have known experimental ground truth.

### Genuine live upload

Open **Live Upload & Diagnose** and upload one of these real trained checkpoints:

```text
proof_models/checkpoints/model_0004.pt  healthy
proof_models/checkpoints/model_0001.pt  class imbalance
proof_models/checkpoints/model_0002.pt  label noise
proof_models/checkpoints/model_0003.pt  shortcut
```

The backend actually loads the checkpoint, evaluates it on the bundled MNIST diagnostic set, extracts the diagnostic features, invokes the trained Random Forest, and returns Grad-CAM evidence. For live uploads, the UI does not pretend to know the ground truth.

## Validation

Run:

```powershell
python -m pytest -q
```

The project includes live-upload tests in `tests/test_live_diagnosis.py`.

For frontend validation on Windows:

```powershell
cd frontend
npm.cmd install
npm.cmd run build
npm.cmd run dev
```

## Scope statement for the viva

> Model Doctor is a controlled, model-specific image-classification diagnostic system. It is experimentally validated on TinyCNN/MNIST with healthy, class imbalance, label noise, and shortcut-learning conditions. It also provides a live upload path for compatible TinyCNN checkpoints, where the system runs diagnostic probes, predicts a likely fault, and provides evidence. It does not claim arbitrary architecture or arbitrary dataset support.
