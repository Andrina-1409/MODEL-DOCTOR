# Live Diagnosis

## What this adds

Model Doctor now has two presentation modes:

1. **Professor Proof Library** — six prevalidated local checkpoints with known experimental ground truth.
2. **Live Upload & Diagnose** — upload a compatible trained PyTorch checkpoint and run the diagnostic pipeline on it during the presentation.

## Supported live scope

The live upload path deliberately supports **TinyCNN + MNIST** only. This is a model-specific research prototype, not an arbitrary-model debugger.

The uploaded checkpoint may be either:

- a raw TinyCNN `state_dict`, or
- a checkpoint dictionary containing a `state_dict` key.

Full serialized Python model objects are rejected by the loader. This keeps the upload path tensor/state-dict based rather than executing arbitrary Python objects from the upload.

## Live pipeline

```text
Uploaded TinyCNN checkpoint
          |
          v
Safe state-dict validation
          |
          v
Bundled MNIST diagnostic set
          |
          +--> clean accuracy / per-class accuracy
          +--> confidence / entropy / high-confidence errors
          +--> train-vs-test behavioural gap
          +--> matched/mismatched shortcut probe
          |
          v
10 diagnostic features
          |
          v
Trained Random Forest Model Doctor
          |
          v
Likely failure mode + probabilities
          |
          +--> Grad-CAM
          +--> corner-attention evidence
          +--> suggested fix
```

## What is and is not known for a live upload

For the six proof models, the experiment knows the injected fault, so the UI can display ground truth.

For a professor's uploaded model, **ground truth is not assumed**. The system reports the model doctor's predicted failure mode and the evidence used to reach it. This is the honest distinction between experimental validation and live inference.

## Running the live mode

### Terminal 1 — backend

From the project root:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn src.api:app --reload --port 8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

### Terminal 2 — frontend

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open the Vite URL and select **Live Upload & Diagnose**.

## Best professor demonstration

Use one of the bundled checkpoints as a live-upload proof:

```text
proof_models/checkpoints/model_0004.pt  -> healthy
proof_models/checkpoints/model_0001.pt  -> class imbalance
proof_models/checkpoints/model_0002.pt  -> label noise
proof_models/checkpoints/model_0003.pt  -> shortcut
```

This demonstrates that the same upload endpoint can receive a real trained checkpoint rather than only selecting a precomputed JSON result.

## Recommended wording

> "The proof library gives me controlled models with known ground truth so I can validate the method. The live mode is different: I upload a compatible trained TinyCNN checkpoint, the system runs the same diagnostic feature extraction and Random Forest doctor, and it reports the most likely fault with evidence. Because the uploaded model has no known ground-truth label, I don't claim that the prediction is experimentally verified for that upload."

## Current limitation

The live path does not yet support arbitrary architectures or arbitrary datasets. Adding another architecture requires a model adapter, compatible preprocessing/dataset adapter, architecture-specific probes, and validation of the doctor on a new model zoo.
