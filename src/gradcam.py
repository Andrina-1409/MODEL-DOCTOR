from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import TinyCNN


def gradcam(
    model: TinyCNN,
    image: torch.Tensor,
    target_class: int | None = None,
) -> torch.Tensor:
    """Compute a normalized 28x28 Grad-CAM heatmap from TinyCNN.conv2."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1:] != (1, 28, 28):
        raise ValueError("image must have shape [1, 28, 28] or [N, 1, 28, 28].")

    model = model.to("cpu").eval()
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h1 = model.conv2.register_forward_hook(forward_hook)
    h2 = model.conv2.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image[:1])
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        logits[0, target_class].backward()

        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value
        return cam.detach()
    finally:
        h1.remove()
        h2.remove()


def overlay(image: torch.Tensor, heatmap: torch.Tensor) -> np.ndarray:
    """Return an RGB uint8 visualization blending a grayscale image and heatmap."""
    if image.ndim == 4:
        image = image[0]
    if image.shape != (1, 28, 28) or heatmap.shape != (28, 28):
        raise ValueError("image and heatmap shapes must be [1,28,28] and [28,28].")
    base = image.squeeze(0).detach().cpu().numpy()
    heat = heatmap.detach().cpu().numpy()
    rgb = np.stack([base, base, base], axis=-1)
    # Simple red-channel emphasis keeps this implementation dependency-light.
    rgb[..., 0] = np.maximum(rgb[..., 0], heat)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def corner_attention_score(heatmap: torch.Tensor, corner_size: int = 6) -> float:
    """Return the maximum fraction of Grad-CAM mass in a 6x6 corner."""
    if heatmap.shape != (28, 28):
        raise ValueError("heatmap must have shape [28, 28].")
    total = heatmap.sum()
    if total <= 0:
        return 0.0
    s = corner_size
    masses = [
        heatmap[:s, :s].sum(),
        heatmap[:s, -s:].sum(),
        heatmap[-s:, :s].sum(),
        heatmap[-s:, -s:].sum(),
    ]
    return float(max(masses).item() / total.item())
