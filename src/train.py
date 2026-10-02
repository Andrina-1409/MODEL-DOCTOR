from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import MNISTData
from .model import TinyCNN


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 42
    device: str | None = None


def train_model(
    model: TinyCNN,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    config: TrainConfig = TrainConfig(),
    val_images: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
) -> dict[str, list[float]]:
    """Train TinyCNN deterministically and optionally record clean validation metrics."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_confidence": [],
        "val_confidence": [],
    }

    for _ in range(config.epochs):
        model.train()
        running_loss = 0.0
        n = 0
        conf_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_n = yb.numel()
            running_loss += loss.item() * batch_n
            n += batch_n
            conf_sum += torch.softmax(logits.detach(), dim=1).max(dim=1).values.sum().item()

        history["train_loss"].append(running_loss / n)
        history["train_confidence"].append(conf_sum / n)

        if val_images is not None and val_labels is not None:
            model.eval()
            with torch.no_grad():
                vx = val_images.to(device)
                vy = val_labels.to(device)
                vlogits = model(vx)
                vloss = criterion(vlogits, vy)
                vconf = torch.softmax(vlogits, dim=1).max(dim=1).values.mean()
            history["val_loss"].append(float(vloss.item()))
            history["val_confidence"].append(float(vconf.item()))

    model.to("cpu")
    return history


def train_from_data(
    data: MNISTData,
    *,
    config: TrainConfig = TrainConfig(),
) -> tuple[TinyCNN, dict[str, list[float]]]:
    """Convenience wrapper using the prepared MNIST training split."""
    model = TinyCNN()
    history = train_model(
        model,
        data.train_images,
        data.train_labels,
        config=config,
        val_images=data.test_images[:2_000],
        val_labels=data.test_labels[:2_000],
    )
    return model, history


def save_checkpoint(
    model: TinyCNN,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> None:
    """Save a model state dict together with reproducibility metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        destination,
    )


def load_checkpoint(path: str | Path) -> tuple[TinyCNN, dict]:
    """Load a TinyCNN checkpoint and its metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("metadata", {})
