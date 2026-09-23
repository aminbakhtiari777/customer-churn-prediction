"""Dataset validation and feature definitions."""

from __future__ import annotations

import pandas as pd

TARGET = "Churn"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the training frame and return model features and a binary target."""
    required = set(FEATURES + [TARGET])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    clean = frame.copy()
    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    labels = clean[TARGET].astype(str).str.strip().str.lower()
    unknown = sorted(set(labels).difference({"yes", "no"}))
    if unknown:
        raise ValueError(f"Unsupported Churn labels: {', '.join(unknown)}")
    return clean[FEATURES], labels.map({"no": 0, "yes": 1}).astype(int)
