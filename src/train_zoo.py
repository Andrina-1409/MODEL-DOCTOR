from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import load_mnist
from .fault_injection import InjectionConfig, inject_fault
from .model import TinyCNN
from .train import TrainConfig, save_checkpoint

MASTER_SEED = 42
ZOO_SIZE = 100
FAULTS = ("healthy", "class_imbalance", "label_noise", "shortcut")


def model_spec(index: int) -> InjectionConfig:
    """Return deterministic metadata for model_0001 .. model_0100."""
    fault = FAULTS[index % len(FAULTS)]
    severity = 0.0 if fault == "healthy" else 0.25 + 0.25 * ((index // 4) % 4)
    seed = MASTER_SEED + index
    # Vary affected digits deterministically while keeping the count small.
    d1 = (index * 3 + 1) % 10
    d2 = (d1 + 5) % 10
    return InjectionConfig(
        fault=fault,
        severity=severity,
        seed=seed,
        selected_digits=(d1, d2),
    )


def train_one(
    model_id: str,
    config: InjectionConfig,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    output_dir: Path,
) -> dict:
    faulty_images, faulty_labels = inject_fault(train_images, train_labels, config)
    model = TinyCNN()

    history = __import__("src.train", fromlist=["train_model"]).train_model(
        model,
        faulty_images,
        faulty_labels,
        config=TrainConfig(
            epochs=5,
            batch_size=256,
            learning_rate=1e-3,
            seed=config.seed,
        ),
        val_images=validation_images,
        val_labels=validation_labels,
    )

    metadata = {
        "model_id": model_id,
        "fault": config.fault,
        "severity": config.severity,
        "seed": config.seed,
        "selected_digits": list(config.selected_digits),
        "epochs": 5,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "patch_mapping": "src.fault_injection.PATCH_MAPPING",
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_confidence": history["train_confidence"][-1],
        "final_val_confidence": history["val_confidence"][-1],
        "train_size_after_injection": int(faulty_images.shape[0]),
    }
    save_checkpoint(model, output_dir / f"{model_id}.pt", metadata=metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Model Doctor model zoo.")
    parser.add_argument("--limit", type=int, default=ZOO_SIZE)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models/zoo")
    args = parser.parse_args()

    if not 1 <= args.limit <= ZOO_SIZE:
        raise ValueError(f"--limit must be between 1 and {ZOO_SIZE}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_mnist(args.data_dir)
    # A fixed clean validation sample is used for training diagnostics.
    validation_images = data.test_images[:2_000]
    validation_labels = data.test_labels[:2_000]

    manifest_path = output_dir / "manifest.jsonl"
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["model_id"]] = row

    rows = []
    for index in range(1, args.limit + 1):
        model_id = f"model_{index:04d}"
        checkpoint = output_dir / f"{model_id}.pt"
        if checkpoint.exists() and model_id in existing:
            rows.append(existing[model_id])
            print(f"[skip] {model_id}")
            continue

        config = model_spec(index)
        print(f"[train] {model_id} fault={config.fault} severity={config.severity:.2f}")
        row = train_one(
            model_id,
            config,
            data.train_images,
            data.train_labels,
            validation_images,
            validation_labels,
            output_dir,
        )
        existing[model_id] = row
        rows.append(row)

        # Persist after every model so an interrupted run is resumable.
        ordered = [
            existing[f"model_{i:04d}"]
            for i in sorted(
                int(key.split("_")[1])
                for key in existing
                if int(key.split("_")[1]) <= args.limit
            )
        ]
        manifest_path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in ordered) + "\n"
        )

    print(f"Completed requested zoo limit: {args.limit}")


if __name__ == "__main__":
    main()
