from pathlib import Path

import pytest
import torch

from src.live_diagnosis import load_uploaded_tinycnn, diagnose_uploaded_model
from src.model import TinyCNN


ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "proof_models" / "checkpoints" / "model_0004.pt"


def test_live_loader_accepts_bundled_tinycnn_checkpoint():
    model = load_uploaded_tinycnn(PROOF)
    assert isinstance(model, TinyCNN)
    assert not model.training


def test_live_diagnosis_runs_on_a_real_uploaded_checkpoint():
    result = diagnose_uploaded_model(PROOF)
    assert result["source"] == "live_upload"
    assert result["architecture"] == "TinyCNN"
    assert result["dataset"] == "MNIST"
    assert result["ground_truth"] is None
    assert set(result["features"]) == {
        "acc_clean", "acc_min_class", "acc_std_class", "mean_confidence",
        "mean_entropy", "conf_wrong_rate", "train_val_loss_gap",
        "train_conf_minus_val_conf", "acc_patch_matched", "acc_patch_mismatched",
    }
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["gradcam_images"][0].startswith("data:image/png;base64,")


def test_live_loader_rejects_incompatible_state_dict(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": {"not_a_tinycnn_weight": torch.tensor([1.0])}}, path)
    with pytest.raises(ValueError, match="compatible TinyCNN"):
        load_uploaded_tinycnn(path)
