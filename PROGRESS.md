# Model Doctor Progress

## Current status
**Completed — Professor Demo + Live Upload Scope**

The project now contains both a validated controlled research experiment and a genuine live upload path for the supported model family.

## Validated research core

- 100-model TinyCNN/MNIST zoo.
- 25 models each for healthy, class imbalance, label noise, and shortcut.
- 10 diagnostic features per model.
- Random Forest Model Doctor.
- 5-fold CV accuracy: **0.980 ± 0.0245**.
- Permuted-label CV baseline: **0.340 ± 0.1393**.
- Six local proof checkpoints for offline demonstration.
- Grad-CAM evidence and diagnosis exports.

## Live diagnosis — COMPLETE

A FastAPI endpoint now accepts a trained PyTorch checkpoint and runs the actual diagnostic pipeline:

```text
Upload TinyCNN checkpoint
        ↓
Safe state-dict validation
        ↓
Bundled MNIST diagnostic set
        ↓
10 behavioural/diagnostic features
        ↓
Random Forest Model Doctor
        ↓
Fault probabilities + likely diagnosis
        ↓
Grad-CAM + corner-attention evidence
        ↓
Suggested improvement
```

Supported live input:
- TinyCNN architecture.
- MNIST-compatible 1×28×28 input.
- Raw `state_dict` or a checkpoint containing `state_dict`.
- `.pt`, `.pth`, or `.bin` upload.

For uploaded models, ground truth is **not** assumed. The UI clearly distinguishes a live prediction from the controlled proof cases where the injected ground truth is known.

## Frontend — COMPLETE

- Professor Proof Library.
- Live Upload & Diagnose tab.
- Advanced 100-model browser.
- Diagnosis probabilities.
- Diagnostic feature table.
- Grad-CAM evidence.
- Suggested fix.
- Live evaluation summary.
- Vite proxy to FastAPI `/api`.

## Professor demonstration

Recommended sequence:

1. Show the six-case Proof Library and explain controlled ground truth.
2. Open **Live Upload & Diagnose**.
3. Upload `proof_models/checkpoints/model_0004.pt` and show the system loading a real checkpoint through the upload endpoint.
4. Repeat with `model_0001.pt`, `model_0002.pt`, or `model_0003.pt` if time allows.
5. Explain that the same pipeline can be extended to another architecture only after implementing and validating its adapter, preprocessing, probes, and doctor training data.

## Explicit limitations

The project does **not** claim arbitrary `.pt` support. A model file alone is not enough to safely infer its dataset, preprocessing, labels, or appropriate probes. The current live scope is deliberately TinyCNN + MNIST.

External ResNet/MobileNet references remain documented as future adapter targets; their binaries are not represented as bundled validated live models.
