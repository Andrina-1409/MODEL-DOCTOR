import torch

from src.model import TinyCNN
from src.train import TrainConfig, load_checkpoint, save_checkpoint, train_model


def test_train_model_updates_parameters(tmp_path):
    x = torch.rand(64, 1, 28, 28)
    y = torch.arange(64) % 10
    model = TinyCNN()
    before = [p.detach().clone() for p in model.parameters()]
    history = train_model(model, x, y, config=TrainConfig(epochs=1, batch_size=32, seed=1))
    assert len(history["train_loss"]) == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_checkpoint_round_trip(tmp_path):
    model = TinyCNN()
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, metadata={"model_id": "model_0001"})
    loaded, metadata = load_checkpoint(path)
    assert metadata["model_id"] == "model_0001"
    for a, b in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(a, b)
