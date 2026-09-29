"""Churn prediction API: send one customer's details, get their churn risk back."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# load the saved pipeline and threshold once, when the server starts, not on every request
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
saved = joblib.load(MODEL_PATH)
model, threshold = saved["model"], saved["threshold"]

app = FastAPI(title="Churn Prediction API")


# the 19 inputs the model was trained on; FastAPI rejects requests with missing or wrong-typed fields
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    # a real customer from the data, shown as a ready-made example on the /docs page
    model_config = {"json_schema_extra": {"examples": [{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    }]}}


@app.get("/health")
def health():
    # quick check that the server is up and the model is loaded
    return {"status": "ok", "threshold": threshold}


@app.post("/predict")
def predict(customer: Customer):
    # one-row table with the same column names the pipeline saw in training
    row = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(row)[:, 1][0])
    return {
        "churn_probability": round(proba, 3),
        "flag_for_offer": proba >= threshold,
        "threshold": threshold,
    }
