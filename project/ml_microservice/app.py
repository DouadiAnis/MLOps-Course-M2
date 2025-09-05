from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="MLOps Microservice - Churn")


FEATURE_ORDER = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "InternetService", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    InternetService: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
_model = None


def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: CustomerFeatures):
    model = get_model()
    if model is None:
        return {"error": "Model not found. Train and place at artifacts/model.joblib"}

    # Build a single-row DataFrame with correct column order
    data = pd.DataFrame([[getattr(features, col) for col in FEATURE_ORDER]], columns=FEATURE_ORDER)

    pred = model.predict(data)[0]
    proba = float(model.predict_proba(data)[0, 1])
    return {"prediction": int(pred), "probability": proba}