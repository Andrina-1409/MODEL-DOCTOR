from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import load_mnist
from .fault_injection import InjectionConfig, inject_fault
from .probes import patch_probe_accuracy
from .train import load_checkpoint

FEATURE_COLUMNS = [
    "acc_clean",
    "acc_min_class",
    "acc_std_class",
    "mean_confidence",
    "mean_entropy",
    "conf_wrong_rate",
    "train_val_loss_gap",
    "train_conf_minus_val_conf",
    "acc_patch_matched",
    "acc_patch_mismatched",
]
META_COLUMNS = ["model_id", "fault", "severity", "seed"]


@torch.no_grad()
def _prediction_metrics(model, images, labels, batch_size=512):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total_loss = 0.0
    total_conf = 0.0
    total_entropy = 0.0
    high_conf_wrong = 0
    per_class_correct = torch.zeros(10, dtype=torch.float64)
    per_class_total = torch.zeros(10, dtype=torch.float64)

    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size]
        y = labels[start:start + batch_size]
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1)
        conf = probs.max(1).values
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(1)

        correct_mask = pred == y
        correct += int(correct_mask.sum())
        total_loss += float(criterion(logits, y))
        total_conf += float(conf.sum())
        total_entropy += float(entropy.sum())
        high_conf_wrong += int(((~correct_mask) & (conf > 0.9)).sum())

        for digit in range(10):
            mask = y == digit
            per_class_total[digit] += mask.sum()
            per_class_correct[digit] += (correct_mask & mask).sum()

    n = len(labels)
    class_acc = per_class_correct / per_class_total.clamp_min(1)
    return {
        "accuracy": correct / n,
        "loss": total_loss / n,
        "mean_confidence": total_conf / n,
        "mean_entropy": total_entropy / n,
        "conf_wrong_rate": high_conf_wrong / n,
        "class_acc": class_acc.numpy(),
    }


def extract_one(model_id: str, checkpoint_path: Path, data) -> dict:
    model, metadata = load_checkpoint(checkpoint_path)
    config = InjectionConfig(
        fault=metadata["fault"],
        severity=float(metadata["severity"]),
        seed=int(metadata["seed"]),
        selected_digits=tuple(metadata["selected_digits"]),
    )

    clean = _prediction_metrics(model, data.test_images, data.test_labels)
    train_images, train_labels = inject_fault(
        data.train_images, data.train_labels, config
    )
    train_metrics = _prediction_metrics(model, train_images, train_labels)
    val_metrics = _prediction_metrics(model, data.test_images[:2_000], data.test_labels[:2_000])
    matched, mismatched = patch_probe_accuracy(
        model, data.test_images, data.test_labels
    )

    row = {
        "model_id": model_id,
        "fault": metadata["fault"],
        "severity": float(metadata["severity"]),
        "seed": int(metadata["seed"]),
        "acc_clean": clean["accuracy"],
        "acc_min_class": float(np.min(clean["class_acc"])),
        "acc_std_class": float(np.std(clean["class_acc"])),
        "mean_confidence": clean["mean_confidence"],
        "mean_entropy": clean["mean_entropy"],
        "conf_wrong_rate": clean["conf_wrong_rate"],
        "train_val_loss_gap": train_metrics["loss"] - val_metrics["loss"],
        "train_conf_minus_val_conf": train_metrics["mean_confidence"] - val_metrics["mean_confidence"],
        "acc_patch_matched": matched,
        "acc_patch_mismatched": mismatched,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one diagnostic feature row per model.")
    parser.add_argument("--model-dir", default="models/zoo")
    parser.add_argument("--output", default="features/features.csv")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoints = sorted(model_dir.glob("model_*.pt"))
    if len(checkpoints) != 100:
        raise RuntimeError(f"Expected exactly 100 model checkpoints, found {len(checkpoints)}")

    data = load_mnist(args.data_dir)
    rows = []
    for checkpoint in checkpoints:
        model_id = checkpoint.stem
        print(f"[features] {model_id}")
        rows.append(extract_one(model_id, checkpoint, data))

    frame = pd.DataFrame(rows)
    required = META_COLUMNS + FEATURE_COLUMNS
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise AssertionError(f"Missing feature columns: {missing}")
    if frame.shape[0] != 100:
        raise AssertionError("Expected one row per model.")
    if frame[required].isna().any().any():
        raise AssertionError("Feature table contains NaN.")
    if np.isinf(frame[FEATURE_COLUMNS].to_numpy()).any():
        raise AssertionError("Feature table contains infinity.")
    if not frame["model_id"].is_unique:
        raise AssertionError("model_id values must be unique.")
    bounded_unit = [
        "acc_clean", "acc_min_class", "acc_std_class",
        "mean_confidence", "conf_wrong_rate",
        "acc_patch_matched", "acc_patch_mismatched",
    ]
    for col in bounded_unit:
        if not ((frame[col] >= 0).all() and (frame[col] <= 1).all()):
            raise AssertionError(f"{col} is outside expected range.")
    if not ((frame["mean_entropy"] >= 0).all() and (frame["mean_entropy"] <= np.log(10) + 1e-6).all()):
        raise AssertionError("mean_entropy is outside the 10-class entropy range.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = frame.groupby("fault")[FEATURE_COLUMNS].agg(["mean", "std"]).round(6)
    summary.to_csv("reports/feature_summary.csv")
    print(f"Wrote {output} with {len(frame)} rows.")


if __name__ == "__main__":
    main()
