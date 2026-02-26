"""
Inference server — loads Production model from MLflow registry at startup.
Exposes /predict and /health endpoints.
"""

import json
import logging
import os
import time

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/telco-churn/Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="telco-churn-inference")

# Loaded once at startup — not per request
_model = None
_fitted_features = None


@app.on_event("startup")
def load_model() -> None:
    global _model, _fitted_features
    logger.info("Loading model from registry: %s", MODEL_URI)
    _model = mlflow.sklearn.load_model(MODEL_URI)

    # Load the frozen training statistics stored alongside the model
    client = mlflow.tracking.MlflowClient()
    version = client.get_latest_versions("telco-churn", stages=["Production"])[0]
    artifact_path = client.download_artifacts(version.run_id, "fitted_features.json")
    with open(artifact_path) as f:
        _fitted_features = json.load(f)

    logger.info("Model v%s loaded — ready to serve", version.version)


class PredictRequest(BaseModel):
    MonthlyCharges: float
    tenure: int
    TotalCharges: float
    InternetService: str  # needed to compute HighValueFiber


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    model_version: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Reproduce the same feature logic as src/features.py — using FROZEN training stats
    monthly_charges_median = _fitted_features["monthly_charges_median"]
    high_value_fiber = int(
        req.InternetService == "Fiber optic"
        and req.MonthlyCharges > monthly_charges_median
    )

    row = pd.DataFrame(
        [
            {
                "MonthlyCharges": req.MonthlyCharges,
                "tenure": req.tenure,
                "TotalCharges": req.TotalCharges,
                "HighValueFiber": high_value_fiber,
            }
        ]
    )

    # Apply frozen scaler — same as training, no refitting
    scaler_mean = _fitted_features["scaler_mean"]
    scaler_scale = _fitted_features["scaler_scale"]
    row_scaled = (row.values - scaler_mean) / scaler_scale

    proba = float(_model.predict_proba(row_scaled)[0][1])
    pred = int(proba >= 0.5)

    latency_ms = (time.time() - start) * 1000
    logger.info("predict latency=%.1fms churn=%d proba=%.3f", latency_ms, pred, proba)

    return PredictResponse(
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        model_version=_fitted_features.get("model_version", "unknown"),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}
