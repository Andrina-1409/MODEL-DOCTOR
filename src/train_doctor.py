from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .extract_features import FEATURE_COLUMNS

RANDOM_STATE = 42


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Model Doctor.")
    parser.add_argument("--features", default="features/features.csv")
    parser.add_argument("--output-model", default="doctor.pkl")
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    X = frame[FEATURE_COLUMNS]
    y = frame["fault"]
    model_ids = frame["model_id"]

    assert not X.isna().any().any()
    assert set(["model_id", "fault", "severity"]).isdisjoint(FEATURE_COLUMNS)

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        model_ids,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    assert set(id_train).isdisjoint(set(id_test))

    doctor = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
    )
    doctor.fit(X_train, y_train)
    pred = doctor.predict(X_test)

    logistic = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(doctor, X, y, cv=cv, scoring="accuracy")

    permutation_rng = np.random.default_rng(RANDOM_STATE)
    shuffled_y = y.to_numpy().copy()
    permutation_rng.shuffle(shuffled_y)
    shuffled_scores = cross_val_score(
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        X,
        shuffled_y,
        cv=cv,
        scoring="accuracy",
    )

    reports = {
        "random_forest": _metrics(y_test, pred),
        "logistic_regression": _metrics(y_test, logistic_pred),
        "majority_dummy": _metrics(y_test, dummy_pred),
        "random_forest_5fold_cv_accuracy_mean": float(cv_scores.mean()),
        "random_forest_5fold_cv_accuracy_std": float(cv_scores.std()),
        "permuted_label_5fold_cv_accuracy_mean": float(shuffled_scores.mean()),
        "permuted_label_5fold_cv_accuracy_std": float(shuffled_scores.std()),
        "train_model_ids": sorted(id_train.tolist()),
        "test_model_ids": sorted(id_test.tolist()),
        "feature_columns": FEATURE_COLUMNS,
        "leakage_columns_excluded": ["model_id", "fault", "severity"],
    }
    if reports["random_forest"]["accuracy"] >= 0.999:
        reports["suspicious_perfect_accuracy_note"] = (
            "Very high holdout accuracy was observed. This is not automatically leakage; "
            "inspect feature distributions and keep the model-level split intact."
        )

    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(doctor, args.output_model)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(reports, indent=2))

    report_text = classification_report(y_test, pred, zero_division=0)
    Path("reports/classification_report.txt").write_text(report_text)

    labels = list(doctor.classes_)
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Model Doctor — Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=160)
    plt.close(fig)

    order = np.argsort(doctor.feature_importances_)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(FEATURE_COLUMNS)[order], doctor.feature_importances_[order])
    ax.set_xlabel("Importance")
    ax.set_title("Model Doctor — Feature Importance")
    fig.tight_layout()
    fig.savefig("reports/feature_importance.png", dpi=160)
    plt.close(fig)

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
