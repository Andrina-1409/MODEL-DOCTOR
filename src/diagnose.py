from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from .data import load_mnist
from .extract_features import FEATURE_COLUMNS, extract_one
from .gradcam import corner_attention_score, gradcam, overlay
from .fault_injection import patch_for_class
from .train import load_checkpoint

SUGGESTED_FIXES = {
    "healthy": "no fault detected",
    "class_imbalance": "rebalance the data or use a weighted loss",
    "label_noise": "clean the labels or use a robust loss",
    "shortcut": "remove or randomize the spurious cue and use augmentation",
}


def diagnose(
    model_id: str,
    *,
    model_dir: str | Path = "models/zoo",
    doctor_path: str | Path = "doctor.pkl",
    data_dir: str | Path = "data",
    output_dir: str | Path = "reports/gradcam",
) -> dict:
    """Diagnose one model using the trained Random Forest and Grad-CAM evidence."""
    checkpoint_path = Path(model_dir) / f"{model_id}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Unknown model: {model_id}")

    model, metadata = load_checkpoint(checkpoint_path)
    data = load_mnist(data_dir)
    row = extract_one(model_id, checkpoint_path, data)

    doctor = joblib.load(doctor_path)
    X = pd.DataFrame([[row[feature] for feature in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    probabilities_array = doctor.predict_proba(X)[0]
    probabilities = {
        label: float(prob)
        for label, prob in zip(doctor.classes_, probabilities_array)
    }
    diagnosis = str(doctor.classes_[int(probabilities_array.argmax())])
    confidence = float(probabilities_array.max())

    image = data.test_images[0]
    if diagnosis == "shortcut":
        image_for_cam = patch_for_class(
            image.unsqueeze(0),
            data.test_labels[:1],
        )[0]
    else:
        image_for_cam = image
    heatmap = gradcam(model, image_for_cam)
    attention = corner_attention_score(heatmap)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overlay_path = output / f"{model_id}.png"
    from matplotlib import pyplot as plt
    plt.imsave(overlay_path, overlay(image_for_cam, heatmap))

    result = {
        "model_id": model_id,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": {feature: float(row[feature]) for feature in FEATURE_COLUMNS},
        "gradcam_images": [str(overlay_path).replace("\\", "/")],
        "corner_attention": attention,
        "suggested_fix": SUGGESTED_FIXES[diagnosis],
        "ground_truth": metadata["fault"],
    }
    return result


def save_diagnosis(result: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = diagnose(args.model_id)
    print(json.dumps(result, indent=2))
    if args.output:
        save_diagnosis(result, args.output)
