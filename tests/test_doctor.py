from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.extract_features import FEATURE_COLUMNS


def test_doctor_can_train_without_metadata_features():
    rows = []
    for fault in ["healthy", "class_imbalance", "label_noise", "shortcut"]:
        for i in range(5):
            rows.append({
                "model_id": f"{fault}_{i}",
                "fault": fault,
                "severity": 0.5,
                **{feature: 0.1 + i * 0.01 for feature in FEATURE_COLUMNS},
            })
    frame = pd.DataFrame(rows)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(frame[FEATURE_COLUMNS], frame["fault"])
    assert list(model.feature_importances_) and len(model.feature_importances_) == 10


def test_leakage_columns_are_not_features():
    assert "model_id" not in FEATURE_COLUMNS
    assert "fault" not in FEATURE_COLUMNS
    assert "severity" not in FEATURE_COLUMNS
