# Offline Proof Model Library

These six checkpoints are the recommended live-demo set. They are self-contained with the project's MNIST dataset, preprocessing code, diagnosis JSON, and Grad-CAM assets.

Use the React **Professor Demo Library** rather than searching for models during a presentation.

## Cases

- model_0004 — Healthy CNN Baseline
- model_0001 — Class Imbalance Case
- model_0002 — Label Noise Case
- model_0003 — Shortcut Learning Case
- model_0008 — Healthy Reproducibility Case
- model_0007 — Shortcut Stress Case

The files in `checkpoints/` are actual PyTorch checkpoints. They are copies of the validated zoo checkpoints and are intentionally kept small.

The MNIST dataset used by these models is stored under `data/MNIST/`.
