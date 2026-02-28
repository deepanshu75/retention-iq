"""
Train logistic regression churn model and save pipeline to app/model.joblib.
Run after generating data/subscription_data.csv.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "subscription_data.csv"
MODEL_DIR = Path(__file__).resolve().parent / "app"
MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_COLUMNS = [
    "tenure_days",
    "sessions_last_30d",
    "payment_failures",
    "support_tickets",
    "feature_usage_score",
]
TARGET = "churned"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run generate_data.py first.")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X, y)
    accuracy = pipeline.score(X, y)
    print(f"Accuracy (on full dataset): {accuracy:.4f}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"pipeline": pipeline, "feature_names": FEATURE_COLUMNS}
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
