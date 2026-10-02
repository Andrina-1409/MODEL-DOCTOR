import pandas as pd

from src.extract_features import FEATURE_COLUMNS


def test_required_feature_columns_are_unique():
    assert len(FEATURE_COLUMNS) == 10
    assert len(set(FEATURE_COLUMNS)) == 10


def test_feature_frame_contract():
    frame = pd.DataFrame([{
        "model_id": "model_0001",
        "fault": "healthy",
        "severity": 0.0,
        "seed": 42,
        **{feature: 0.5 for feature in FEATURE_COLUMNS},
    }])
    assert list(frame[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
