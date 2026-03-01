"""FastAPI app: health and churn-score endpoints."""
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from app.features import get_user_features, load_data
from core.model import load_model, predict_proba_and_drivers, risk_level

_df = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _df
    load_model()
    _df = load_data()
    yield
    _df = None


app = FastAPI(title="Retention-IQ", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/churn-score/{user_id}")
def churn_score(user_id: str) -> dict:
    if _df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    features = get_user_features(_df, user_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")
    vec = np.array(features.values, dtype=float)
    probability, top_drivers = predict_proba_and_drivers(vec)
    return {
        "user_id": user_id,
        "churn_probability": round(probability, 2),
        "risk_level": risk_level(probability),
        "top_drivers": top_drivers,
    }
