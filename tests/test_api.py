from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "FastAPI development dependencies are not installed",
)
class ApiTests(unittest.TestCase):
    def test_health_and_prediction(self) -> None:
        from fastapi.testclient import TestClient
        from churn.api import create_app

        class FakeService:
            def predict(self, record):
                self.last_record = record
                return {"churn_probability": 0.72, "will_churn": True, "threshold": 0.5}

        payload = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
            "Dependents": "No", "tenure": 4, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5,
            "TotalCharges": 358.0,
        }
        with TestClient(create_app(FakeService())) as client:
            self.assertTrue(client.get("/health").json()["model_loaded"])
            response = client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["will_churn"])
