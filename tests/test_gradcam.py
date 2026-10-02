import torch

from src.gradcam import corner_attention_score, gradcam, overlay
from src.model import TinyCNN


def test_gradcam_shape_and_range():
    model = TinyCNN()
    image = torch.rand(1, 28, 28)
    heatmap = gradcam(model, image)
    assert heatmap.shape == (28, 28)
    assert torch.isfinite(heatmap).all()
    assert float(heatmap.min()) >= 0
    assert float(heatmap.max()) <= 1


def test_overlay_is_rgb_uint8():
    image = torch.rand(1, 28, 28)
    heatmap = gradcam(TinyCNN(), image)
    result = overlay(image, heatmap)
    assert result.shape == (28, 28, 3)
    assert result.dtype.name == "uint8"


def test_corner_score_range():
    heatmap = torch.ones(28, 28)
    score = corner_attention_score(heatmap)
    assert 0 <= score <= 1
