from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.features import FEATURES, split_features_target
from churn.modeling import build_pipeline, evaluate
from churn.service import ModelService


def sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(24):
        churn = index % 2 == 0
        rows.append({
            "customerID": f"C-{index}",
            "gender": "Female" if index % 3 else "Male",
            "SeniorCitizen": int(index % 4 == 0),
            "Partner": "No" if churn else "Yes",
            "Dependents": "No" if churn else "Yes",
            "tenure": 2 + index if churn else 30 + index,
            "PhoneService": "Yes",
            "MultipleLines": "No" if churn else "Yes",
            "InternetService": "Fiber optic" if churn else "DSL",
            "OnlineSecurity": "No" if churn else "Yes",
            "OnlineBackup": "No" if churn else "Yes",
            "DeviceProtection": "No" if churn else "Yes",
            "TechSupport": "No" if churn else "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month" if churn else "Two year",
            "PaperlessBilling": "Yes" if churn else "No",
            "PaymentMethod": "Electronic check" if churn else "Bank transfer",
            "MonthlyCharges": 90 + index if churn else 40 + index,
            "TotalCharges": " " if index == 0 else str((index + 1) * 100),
            "Churn": "Yes" if churn else "No",
        })
    return pd.DataFrame(rows)


class ChurnCoreTests(unittest.TestCase):
    def test_split_validates_and_normalizes_dataset(self) -> None:
        features, labels = split_features_target(sample_frame())
        self.assertEqual(list(features.columns), FEATURES)
        self.assertEqual(set(labels.unique()), {0, 1})
        self.assertTrue(pd.isna(features.iloc[0]["TotalCharges"]))

    def test_missing_column_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            split_features_target(sample_frame().drop(columns=["Contract"]))

    def test_pipeline_trains_evaluates_and_serves(self) -> None:
        features, labels = split_features_target(sample_frame())
        model = build_pipeline()
        model.fit(features, labels)
        metrics = evaluate(model, features, labels)
        self.assertGreaterEqual(metrics.recall, 0.5)
        prediction = ModelService(model).predict(features.iloc[1].to_dict())
        self.assertGreaterEqual(prediction["churn_probability"], 0)
        self.assertLessEqual(prediction["churn_probability"], 1)
        self.assertIsInstance(prediction["will_churn"], bool)

    def test_service_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            ModelService(object(), threshold=1.0)


if __name__ == "__main__":
    unittest.main()
