"""Train and persist the customer churn model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.features import split_features_target  # noqa: E402
from churn.modeling import build_pipeline, evaluate, metrics_dict  # noqa: E402

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-file", type=Path)
    source.add_argument("--data-url", default=DEFAULT_DATA_URL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data_file if args.data_file else args.data_url)
    features, labels = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test, threshold=args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "churn_pipeline.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_dict(metrics), indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics_dict(metrics), indent=2))


if __name__ == "__main__":
    main()
