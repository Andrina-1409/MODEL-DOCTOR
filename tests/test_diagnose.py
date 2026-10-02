import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.diagnose import diagnose
from src.extract_features import FEATURE_COLUMNS
from src.model import TinyCNN
from src.train import save_checkpoint


def test_diagnose_contract_with_tiny_fixture(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    save_checkpoint(
        TinyCNN(),
        model_dir / "model_0001.pt",
        metadata={
            "model_id": "model_0001",
            "fault": "healthy",
            "severity": 0.0,
            "seed": 42,
            "selected_digits": [1, 7],
        },
    )

    class FakeData:
        train_images = __import__("torch").rand(16, 1, 28, 28)
        train_labels = __import__("torch").arange(16) % 10
        test_images = __import__("torch").rand(20, 1, 28, 28)
        test_labels = __import__("torch").arange(20) % 10

    import src.diagnose as diagnose_module
    import src.extract_features as ef

    monkeypatch.setattr(diagnose_module, "load_mnist", lambda *_args, **_kwargs: FakeData())
    monkeypatch.setattr(ef, "load_mnist", lambda *_args, **_kwargs: FakeData())

    frame = pd.DataFrame([
        {"model_id": "m1", "fault": "healthy", "severity": 0.0, **{f: 0.1 for f in FEATURE_COLUMNS}},
        {"model_id": "m2", "fault": "shortcut", "severity": 0.5, **{f: 0.9 for f in FEATURE_COLUMNS}},
    ])
    doctor = RandomForestClassifier(n_estimators=20, random_state=42).fit(frame[FEATURE_COLUMNS], frame["fault"])
    doctor_path = tmp_path / "doctor.pkl"
    joblib.dump(doctor, doctor_path)

    result = diagnose(
        "model_0001",
        model_dir=model_dir,
        doctor_path=doctor_path,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "gradcam",
    )
    assert set(result) == {
        "model_id", "diagnosis", "confidence", "probabilities",
        "features", "gradcam_images", "corner_attention",
        "suggested_fix", "ground_truth",
    }
    assert result["model_id"] == "model_0001"
    assert 0 <= result["confidence"] <= 1
    assert set(result["probabilities"]) == {"healthy", "shortcut"}
