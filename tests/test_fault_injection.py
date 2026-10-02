import torch

from src.fault_injection import InjectionConfig, add_class_patches, inject_fault, PATCH_MAPPING


def _sample():
    images = torch.zeros(1000, 1, 28, 28)
    labels = torch.arange(1000) % 10
    return images, labels


def test_healthy_does_not_change_data():
    x, y = _sample()
    xx, yy = inject_fault(x, y, InjectionConfig("healthy"))
    assert torch.equal(x, xx)
    assert torch.equal(y, yy)


def test_label_noise_is_deterministic_and_changes_requested_fraction():
    x, y = _sample()
    cfg = InjectionConfig("label_noise", severity=0.5, seed=7)
    xx1, yy1 = inject_fault(x, y, cfg)
    xx2, yy2 = inject_fault(x, y, cfg)
    assert torch.equal(xx1, xx2)
    assert torch.equal(yy1, yy2)
    assert (yy1 != y).sum().item() == 300


def test_class_imbalance_reduces_selected_classes():
    x, y = _sample()
    _, yy = inject_fault(x, y, InjectionConfig("class_imbalance", severity=1.0, seed=7, selected_digits=(1, 7)))
    assert (yy == 1).sum().item() < (y == 1).sum().item()
    assert (yy == 7).sum().item() < (y == 7).sum().item()
    assert all((yy == d).sum().item() > 0 for d in range(10))


def test_shortcut_changes_images_but_not_labels():
    x, y = _sample()
    xx, yy = inject_fault(x, y, InjectionConfig("shortcut", severity=1.0))
    assert torch.equal(y, yy)
    assert not torch.equal(x, xx)
    for digit, (r, c) in PATCH_MAPPING.items():
        assert xx[y == digit, 0, r:r+6, c:c+6].eq(1.0).all()


def test_patch_mapping_is_class_dependent():
    x, y = _sample()
    xx = add_class_patches(x, y)
    assert not torch.equal(xx[y == 0], xx[y == 1])
