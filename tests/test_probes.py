import torch

from src.model import TinyCNN
from src.probes import corner_attention_mass, patch_probe_accuracy


def test_patch_probe_returns_valid_accuracies():
    model = TinyCNN()
    x = torch.rand(20, 1, 28, 28)
    y = torch.arange(20) % 10
    matched, mismatched = patch_probe_accuracy(model, x, y)
    assert 0.0 <= matched <= 1.0
    assert 0.0 <= mismatched <= 1.0


def test_corner_attention_mass():
    heatmap = torch.zeros(28, 28)
    heatmap[:6, :6] = 1
    assert corner_attention_mass(heatmap) == 1.0


def test_corner_attention_mass_rejects_wrong_shape():
    try:
        corner_attention_mass(torch.zeros(10, 10))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
