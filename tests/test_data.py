import torch

from src.data import (
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    load_mnist,
    validate_dataset,
)


def test_mnist_dataset_shape_and_dtype():
    data = load_mnist()

    validate_dataset(data)

    assert data.train_images.shape == (
        DEFAULT_TRAIN_SIZE,
        1,
        28,
        28,
    )

    assert data.test_images.shape == (
        DEFAULT_TEST_SIZE,
        1,
        28,
        28,
    )

    assert data.train_images.dtype == torch.float32
    assert data.test_images.dtype == torch.float32

    assert data.train_labels.dtype == torch.int64
    assert data.test_labels.dtype == torch.int64


def test_mnist_values_are_normalized():
    data = load_mnist()

    assert torch.min(data.train_images) >= 0.0
    assert torch.max(data.train_images) <= 1.0

    assert torch.min(data.test_images) >= 0.0
    assert torch.max(data.test_images) <= 1.0


def test_mnist_selection_is_reproducible():
    data_a = load_mnist(seed=42)
    data_b = load_mnist(seed=42)

    assert torch.equal(data_a.train_images, data_b.train_images)
    assert torch.equal(data_a.train_labels, data_b.train_labels)

    assert torch.equal(data_a.test_images, data_b.test_images)
    assert torch.equal(data_a.test_labels, data_b.test_labels)


def test_different_seed_changes_training_selection():
    data_a = load_mnist(seed=42)
    data_b = load_mnist(seed=123)

    assert not torch.equal(
        data_a.train_images,
        data_b.train_images,
    )