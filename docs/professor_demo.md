# Professor Demo — 5 Minute Flow

## Goal

Show both sides of Model Doctor:

1. **Controlled proof:** known fault + real trained checkpoint + known experimental ground truth.
2. **Live diagnosis:** a real trained checkpoint is uploaded through the application and the same diagnostic pipeline runs on it without selecting a precomputed diagnosis JSON.

## Before the demo

From the project root, start the backend:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn src.api:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run build
npm.cmd run dev
```

Open the Vite URL.

## Part 1 — Controlled proof library

### 1. Healthy baseline
Select **Healthy CNN Baseline**.

Say:
> This is a clean TinyCNN trained on MNIST. It is a control model with known experimental ground truth.

Show:
- diagnosis
- confidence
- probability bars
- diagnostic evidence
- Grad-CAM

### 2. Class imbalance
Select **Class Imbalance Case**.

Say:
> The training data for selected classes was deliberately reduced. The Doctor detects the resulting behavioural signature.

### 3. Label noise
Select **Label Noise Case**.

Say:
> Some training labels were deliberately corrupted. The model is still trained normally, but its observable behaviour changes.

### 4. Shortcut learning
Select **Shortcut Learning Case**.

Say:
> A class-dependent visual cue was inserted during training. The model can exploit that shortcut instead of relying only on the intended digit signal.

Point to:
- patch sensitivity
- diagnostic features
- Grad-CAM
- suggested fix

## Part 2 — Genuine live upload

Open **Live Upload & Diagnose**.

Upload:

```text
proof_models/checkpoints/model_0004.pt
```

Say:
> This time I am not selecting a precomputed diagnosis result. I am uploading an actual trained PyTorch checkpoint. The backend loads the model, runs the diagnostic probes on the bundled MNIST diagnostic set, extracts the ten features, sends them to the trained Random Forest Doctor, and generates fresh Grad-CAM evidence.

Then show the returned:
- diagnosis
- probability distribution
- live evaluation summary
- diagnostic feature values
- Grad-CAM
- suggested fix

Repeat with `model_0001.pt`, `model_0002.pt`, or `model_0003.pt` if time allows.

## Important distinction

The six proof cases have known ground-truth fault labels because the project deliberately injected those faults. An uploaded professor model does **not** have a known ground-truth fault label. The live result is therefore presented as a Model Doctor prediction with evidence, not as a verified ground-truth result.

## What to say if asked about arbitrary internet models

> The current validated live scope is TinyCNN with the project's MNIST preprocessing and diagnostic probes. A model file alone does not tell the system what dataset, preprocessing, labels, or architecture-specific tests it should use. So I support a defined model scope rather than claiming that any arbitrary PyTorch model can already be diagnosed. The architecture-adapter approach can be extended later to models such as ResNet or MobileNet after validation.

Do not claim arbitrary PyTorch model support.
