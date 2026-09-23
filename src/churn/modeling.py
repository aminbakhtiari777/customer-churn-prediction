"""Model construction, training, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold: float


def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer(transformers=[
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1_000, random_state=random_state
    )
    return Pipeline(steps=[("preprocess", preprocessing), ("classifier", classifier)])


def evaluate(model: Pipeline, features: pd.DataFrame, labels: pd.Series,
             threshold: float = 0.5) -> Metrics:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return Metrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        pr_auc=float(average_precision_score(labels, probabilities)),
        threshold=threshold,
    )


def metrics_dict(metrics: Metrics) -> dict[str, float]:
    return asdict(metrics)
