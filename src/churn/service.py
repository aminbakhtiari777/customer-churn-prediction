"""Model artifact loading and prediction service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class ModelService:
    def __init__(self, model: Any, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.model = model
        self.threshold = threshold

    @classmethod
    def from_path(cls, path: str | Path, threshold: float = 0.5) -> "ModelService":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        return cls(joblib.load(model_path), threshold=threshold)

    def predict(self, record: dict[str, Any]) -> dict[str, float | bool]:
        probability = float(self.model.predict_proba(pd.DataFrame([record]))[0, 1])
        return {
            "churn_probability": round(probability, 6),
            "will_churn": probability >= self.threshold,
            "threshold": self.threshold,
        }
