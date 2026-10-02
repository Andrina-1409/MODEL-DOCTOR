from __future__ import annotations

from dataclasses import dataclass

import torch

FAULTS = ("healthy", "class_imbalance", "label_noise", "shortcut")

# Fixed and documented class -> corner mapping. Coordinates are (row, col).
PATCH_MAPPING = {
    0: (0, 0),
    1: (0, 22),
    2: (22, 0),
    3: (22, 22),
    4: (0, 0),
    5: (0, 22),
    6: (22, 0),
    7: (22, 22),
    8: (0, 0),
    9: (22, 22),
}
PATCH_SIZE = 6
PATCH_VALUE = 1.0


@dataclass(frozen=True)
class InjectionConfig:
    fault: str
    severity: float = 0.0
    seed: int = 42
    selected_digits: tuple[int, ...] = (1, 7)


def _validate(config: InjectionConfig) -> None:
    if config.fault not in FAULTS:
        raise ValueError(f"Unknown fault: {config.fault}")
    if not 0.0 <= config.severity <= 1.0:
        raise ValueError("severity must be in [0, 1].")
    if len(set(config.selected_digits)) != len(config.selected_digits):
        raise ValueError("selected_digits must be unique.")
    if any(d < 0 or d > 9 for d in config.selected_digits):
        raise ValueError("selected_digits must contain MNIST classes 0-9.")


def _class_imbalance_keep_fraction(severity: float) -> float:
    # Severity 0 keeps 20%; severity 1 keeps 2%.
    return 0.20 - 0.18 * severity


def inject_fault(
    images: torch.Tensor,
    labels: torch.Tensor,
    config: InjectionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one documented fault to a training tensor pair."""
    _validate(config)
    out_images = images.clone()
    out_labels = labels.clone()

    generator = torch.Generator(device=images.device)
    generator.manual_seed(config.seed)

    if config.fault == "healthy":
        return out_images, out_labels

    if config.fault == "class_imbalance":
        keep_fraction = _class_imbalance_keep_fraction(config.severity)
        keep_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
        for digit in config.selected_digits:
            indices = torch.where(labels == digit)[0]
            keep_n = max(1, int(round(len(indices) * keep_fraction)))
            perm = torch.randperm(len(indices), generator=generator, device=labels.device)
            keep_mask[indices[perm[keep_n:]]] = False
        return out_images[keep_mask], out_labels[keep_mask]

    if config.fault == "label_noise":
        noise_fraction = 0.10 + 0.40 * config.severity
        n = int(round(labels.shape[0] * noise_fraction))
        if n:
            indices = torch.randperm(labels.shape[0], generator=generator, device=labels.device)[:n]
            replacements = torch.randint(0, 9, (n,), generator=generator, device=labels.device)
            original = out_labels[indices]
            replacements = replacements + (replacements >= original).long()
            out_labels[indices] = replacements
        return out_images, out_labels

    # shortcut: labels remain unchanged; only training images get patches.
    return add_class_patches(out_images, out_labels), out_labels


def add_class_patches(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    mapping: dict[int, tuple[int, int]] = PATCH_MAPPING,
) -> torch.Tensor:
    """Add the fixed class-dependent shortcut patch."""
    out = images.clone()
    for digit, (row, col) in mapping.items():
        mask = labels == digit
        if mask.any():
            out[mask, 0, row : row + PATCH_SIZE, col : col + PATCH_SIZE] = PATCH_VALUE
    return out


def patch_for_class(images: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """Apply the correct class-dependent patch to a batch of images."""
    return add_class_patches(images, classes)


def patch_with_wrong_class(images: torch.Tensor, true_classes: torch.Tensor) -> torch.Tensor:
    """Apply a deterministic wrong class patch to each image."""
    wrong_classes = (true_classes + 1) % 10
    return add_class_patches(images, wrong_classes)
