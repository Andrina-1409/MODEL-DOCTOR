import torch

from src.model import TinyCNN


def test_model_output_shape():
    model = TinyCNN()

    x = torch.randn(8, 1, 28, 28)

    output = model(x)

    assert output.shape == (8, 10)


def test_model_feature_shape():
    model = TinyCNN()

    x = torch.randn(8, 1, 28, 28)

    features = model.forward_features(x)

    assert features.shape == (8, 16, 5, 5)


def test_model_penultimate_shape():
    model = TinyCNN()

    x = torch.randn(8, 1, 28, 28)

    features = model.forward_penultimate(x)

    assert features.shape == (8, 400)


def test_model_has_trainable_parameters():
    model = TinyCNN()

    parameters = list(model.parameters())

    assert len(parameters) > 0
    assert all(parameter.requires_grad for parameter in parameters)


def test_model_is_deterministic_in_eval_mode():
    model = TinyCNN()
    model.eval()

    x = torch.randn(4, 1, 28, 28)

    output_a = model(x)
    output_b = model(x)

    assert torch.equal(output_a, output_b)