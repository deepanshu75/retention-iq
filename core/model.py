"""Load saved pipeline and compute churn probability and top drivers."""
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "app" / "model.joblib"

_artifact: dict | None = None


def load_model() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train.py first.")
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def predict_proba_and_drivers(feature_vector: np.ndarray) -> tuple[float, list[dict]]:
    artifact = load_model()
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    X = feature_vector.reshape(1, -1)
    proba = float(pipeline.predict_proba(X)[0, 1])
    scaler = pipeline.named_steps["scaler"]
    coef = pipeline.named_steps["clf"].coef_.ravel()
    X_scaled = scaler.transform(X).ravel()
    contributions = coef * X_scaled
    drivers = [
        {"feature": name, "impact": round(float(impact), 2)}
        for name, impact in zip(feature_names, contributions)
    ]
    drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)
    top_2 = drivers[:2]
    return proba, top_2


def risk_level(probability: float) -> str:
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "medium"
    return "high"
