from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE = 20_000
DEFAULT_TEST_SIZE = 10_000

IMAGE_SIZE = 28
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MNISTData:
    """Container holding the prepared MNIST tensors."""

    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set PyTorch's random seed for deterministic data selection."""

    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mnist(
    data_dir: str | Path = "data",
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> MNISTData:
    """
    Download/load MNIST and return deterministic train/test subsets.

    Training:
        Exactly `train_size` images are selected from MNIST's
        60,000-image training split.

    Testing:
        Exactly `test_size` images are selected from MNIST's
        10,000-image test split.

    Images are returned as float32 tensors in [0, 1] with shape:
        [N, 1, 28, 28]

    Labels are returned as int64 tensors with shape:
        [N]
    """

    if train_size <= 0:
        raise ValueError("train_size must be positive.")

    if test_size <= 0:
        raise ValueError("test_size must be positive.")

    if train_size > 60_000:
        raise ValueError("train_size cannot exceed the MNIST training set.")

    if test_size > 10_000:
        raise ValueError("test_size cannot exceed the MNIST test set.")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Download/load the standard MNIST datasets.
    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
    )

    # Convert the raw uint8 images to float32 tensors in [0, 1].
    train_images = train_dataset.data.float().div(255.0)
    train_labels = train_dataset.targets.long()

    test_images = test_dataset.data.float().div(255.0)
    test_labels = test_dataset.targets.long()

    # Add the channel dimension:
    #
    # [N, 28, 28] -> [N, 1, 28, 28]
    train_images = train_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)

    # Deterministically select the requested number of samples.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_indices = torch.randperm(
        train_images.shape[0],
        generator=generator,
    )[:train_size]

    # Use a separate deterministic generator for the test set.
    test_generator = torch.Generator()
    test_generator.manual_seed(seed + 1)

    test_indices = torch.randperm(
        test_images.shape[0],
        generator=test_generator,
    )[:test_size]

    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]

    test_images = test_images[test_indices]
    test_labels = test_labels[test_indices]

    return MNISTData(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------

def create_dataloaders(
    data: MNISTData,
    batch_size: int = 128,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for the prepared datasets."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    train_dataset = TensorDataset(
        data.train_images,
        data.train_labels,
    )

    test_dataset = TensorDataset(
        data.test_images,
        data.test_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_dataset(data: MNISTData) -> None:
    """Validate the basic invariants required by Model Doctor."""

    if data.train_images.shape != (
        DEFAULT_TRAIN_SIZE,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise AssertionError(
            f"Unexpected train image shape: {data.train_images.shape}"
        )

    if data.test_images.shape != (
        DEFAULT_TEST_SIZE,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise AssertionError(
            f"Unexpected test image shape: {data.test_images.shape}"
        )

    if data.train_labels.shape != (DEFAULT_TRAIN_SIZE,):
        raise AssertionError(
            f"Unexpected train label shape: {data.train_labels.shape}"
        )

    if data.test_labels.shape != (DEFAULT_TEST_SIZE,):
        raise AssertionError(
            f"Unexpected test label shape: {data.test_labels.shape}"
        )

    if data.train_images.dtype != torch.float32:
        raise AssertionError("Training images must be float32.")

    if data.test_images.dtype != torch.float32:
        raise AssertionError("Test images must be float32.")

    if data.train_labels.dtype != torch.int64:
        raise AssertionError("Training labels must be int64.")

    if data.test_labels.dtype != torch.int64:
        raise AssertionError("Test labels must be int64.")

    if torch.any(data.train_images < 0) or torch.any(data.train_images > 1):
        raise AssertionError("Training images must be in [0, 1].")

    if torch.any(data.test_images < 0) or torch.any(data.test_images > 1):
        raise AssertionError("Test images must be in [0, 1].")

    if torch.any(data.train_labels < 0) or torch.any(data.train_labels >= NUM_CLASSES):
        raise AssertionError("Invalid training labels.")

    if torch.any(data.test_labels < 0) or torch.any(data.test_labels >= NUM_CLASSES):
        raise AssertionError("Invalid test labels.")