from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from .data import load_mnist
from .extract_features import FEATURE_COLUMNS, _prediction_metrics
from .gradcam import corner_attention_score, gradcam, overlay
from .probes import patch_probe_accuracy
from .model import TinyCNN

SUGGESTED_FIXES = {
    "healthy": "No major fault signature detected. Keep monitoring validation performance and class-wise behaviour.",
    "class_imbalance": "Rebalance the training data or use a weighted loss so under-represented classes contribute more equally.",
    "label_noise": "Review suspicious labels and consider label cleaning, robust loss functions, or confidence-aware training.",
    "shortcut": "Remove or randomize the spurious cue and use augmentation or counterfactual examples so predictions depend on the intended signal.",
}

LIVE_NOTE = (
    "Live diagnosis is supported for TinyCNN models trained for the bundled MNIST setup. "
    "The uploaded checkpoint is evaluated on the project's fixed MNIST diagnostic set; "
    "ground-truth fault labels are not assumed for an uploaded model."
)


def _normalise_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("The checkpoint contains non-tensor state entries and is not a supported TinyCNN state dict.")
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def load_uploaded_tinycnn(path: str | Path) -> TinyCNN:
    """Safely load a Tensor-only TinyCNN checkpoint or state_dict.

    Full Python objects are intentionally rejected. The accepted formats are:
      1. a raw TinyCNN state_dict
      2. a dict containing a `state_dict` key
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint. Upload a TinyCNN state_dict or a checkpoint containing `state_dict`.")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("The checkpoint's `state_dict` must be a dictionary of tensors.")

    state_dict = _normalise_state_dict(state_dict)
    model = TinyCNN()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "This model is not a compatible TinyCNN checkpoint. "
            "The live demo currently supports TinyCNN + MNIST only."
        ) from exc
    model.eval()
    return model


def _metrics(model: TinyCNN, images: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return _prediction_metrics(model, images, labels)


def _png_data_uri(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def diagnose_uploaded_model(
    checkpoint_path: str | Path,
    *,
    doctor_path: str | Path = "doctor.pkl",
    data_dir: str | Path = "data",
) -> dict[str, Any]:
    """Run the actual Model Doctor pipeline on an uploaded compatible model."""
    model = load_uploaded_tinycnn(checkpoint_path)
    data = load_mnist(data_dir)

    clean = _metrics(model, data.test_images, data.test_labels)
    train_metrics = _metrics(model, data.train_images, data.train_labels)
    matched, mismatched = patch_probe_accuracy(model, data.test_images, data.test_labels)

    row = {
        "acc_clean": clean["accuracy"],
        "acc_min_class": float(np.min(clean["class_acc"])),
        "acc_std_class": float(np.std(clean["class_acc"])),
        "mean_confidence": clean["mean_confidence"],
        "mean_entropy": clean["mean_entropy"],
        "conf_wrong_rate": clean["conf_wrong_rate"],
        "train_val_loss_gap": train_metrics["loss"] - clean["loss"],
        "train_conf_minus_val_conf": train_metrics["mean_confidence"] - clean["mean_confidence"],
        "acc_patch_matched": matched,
        "acc_patch_mismatched": mismatched,
    }

    doctor = joblib.load(doctor_path)
    X = pd.DataFrame([[row[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    probabilities_array = doctor.predict_proba(X)[0]
    probabilities = {
        label: float(prob)
        for label, prob in zip(doctor.classes_, probabilities_array)
    }
    diagnosis = str(doctor.classes_[int(probabilities_array.argmax())])
    confidence = float(probabilities_array.max())

    image = data.test_images[0]
    heatmap = gradcam(model, image)
    attention = corner_attention_score(heatmap)
    image_uri = _png_data_uri(overlay(image, heatmap))

    class_acc = {
        str(index): float(value)
        for index, value in enumerate(clean["class_acc"])
    }

    return {
        "model_id": Path(checkpoint_path).name,
        "source": "live_upload",
        "architecture": "TinyCNN",
        "dataset": "MNIST",
        "diagnosis": diagnosis,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": {name: float(row[name]) for name in FEATURE_COLUMNS},
        "class_accuracy": class_acc,
        "test_accuracy": float(clean["accuracy"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "gradcam_images": [image_uri],
        "corner_attention": attention,
        "suggested_fix": SUGGESTED_FIXES[diagnosis],
        "ground_truth": None,
        "ground_truth_available": False,
        "supported_scope": "TinyCNN + MNIST",
        "note": LIVE_NOTE,
    }
