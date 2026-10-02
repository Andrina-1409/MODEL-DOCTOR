from __future__ import annotations

import torch

from .fault_injection import patch_for_class, patch_with_wrong_class
from .model import TinyCNN


@torch.no_grad()
def patch_probe_accuracy(
    model: TinyCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str | None = None,
) -> tuple[float, float]:
    """Return matched-patch and mismatched-patch accuracy on the same clean images."""
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device).eval()

    matched = patch_for_class(images, labels).to(target_device)
    mismatched = patch_with_wrong_class(images, labels).to(target_device)
    y = labels.to(target_device)

    matched_acc = (model(matched).argmax(1) == y).float().mean().item()
    mismatched_acc = (model(mismatched).argmax(1) == y).float().mean().item()

    model.to("cpu")
    return matched_acc, mismatched_acc


def corner_attention_mass(heatmap: torch.Tensor, corner_size: int = 6) -> float:
    """Fraction of total Grad-CAM mass contained in any image corner."""
    if heatmap.ndim != 2 or heatmap.shape != (28, 28):
        raise ValueError("heatmap must have shape [28, 28].")
    if torch.isnan(heatmap).any() or torch.isinf(heatmap).any():
        raise ValueError("heatmap contains NaN or infinity.")
    total = heatmap.sum()
    if total <= 0:
        return 0.0
    s = corner_size
    corners = torch.stack([
        heatmap[:s, :s].sum(),
        heatmap[:s, -s:].sum(),
        heatmap[-s:, :s].sum(),
        heatmap[-s:, -s:].sum(),
    ])
    return float(corners.max().item() / total.item())
