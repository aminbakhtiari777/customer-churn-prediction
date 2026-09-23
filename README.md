# Customer Churn MLOps API

[![CI](https://github.com/aminbakhtiari777/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/aminbakhtiari777/customer-churn-prediction/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end machine-learning project that turns the IBM Telco Customer Churn dataset into a reproducible training pipeline and a tested prediction API.

The original exploratory notebook is preserved to show the learning path. The production path separates preprocessing, training, evaluation, model persistence, and serving into testable modules.

## Why this project matters

Churn is an imbalanced business problem. Accuracy alone can hide missed customers who are likely to leave, so this project reports precision, recall, F1, ROC-AUC, and PR-AUC and exposes a configurable decision threshold.

## Architecture

```mermaid
flowchart LR
    D[Telco data] --> P[Validated preprocessing]
    P --> T[Balanced classifier]
    T --> A[Versioned artifact]
    A --> F[FastAPI service]
    F --> R[Probability and decision]
```

## Features

- Reproducible scikit-learn `Pipeline`
- Missing-value handling inside the model pipeline
- One-hot encoding with safe unseen-category support
- Class-balanced logistic-regression baseline
- Business-focused evaluation metrics
- Configurable churn decision threshold
- FastAPI health and prediction endpoints
- Pydantic request validation
- Docker packaging
- Unit and API tests
- GitHub Actions continuous integration

## Repository structure

```text
.
├── Customer__Churn__Prediction.ipynb  Original exploration
├── src/churn/                         Production package
├── scripts/train.py                   Reproducible training entry point
├── tests/                             Behavior and API tests
├── models/                            Generated artifacts (not committed)
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train.py
uvicorn churn.api:app --app-dir src --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive API documentation.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 4,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 358.0
  }'
```

The response includes the churn probability, the configured threshold, and the final decision.

## Training and metrics

`scripts/train.py` downloads the public IBM dataset by default, creates a stratified train/test split, fits the complete pipeline, and writes:

- `models/churn_pipeline.joblib`
- `models/metrics.json`

Use a local CSV when network access is unavailable:

```bash
python scripts/train.py --data-file path/to/telco.csv
```

## Tests

```bash
python -m unittest discover -s tests -v
```

API integration tests run when the optional FastAPI development dependencies are installed.

## Docker

Train the model before building, then run:

```bash
docker build -t churn-mlops-api .
docker run --rm -p 8000:8000 churn-mlops-api
```

## Limitations and next steps

- The current model is a transparent production baseline, not a claim of causal churn prediction.
- Threshold selection should be tied to retention cost and campaign capacity.
- Real deployment needs drift monitoring, authentication, observability, and scheduled retraining.
- Fairness checks should be agreed with domain owners before customer-facing use.

## Author

**Amin Bakhtiari** — AI/ML engineering portfolio project.
