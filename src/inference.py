from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from joblib import load

from .config import MODELS_DIR
from .utils import load_json


def load_artifacts() -> Tuple[object, Dict]:
    model_path = MODELS_DIR / "telco_churn_model.joblib"
    metadata_path = MODELS_DIR / "metadata.json"
    model = load(model_path)
    metadata = load_json(metadata_path)
    return model, metadata


def build_input_frame(form_values: Dict) -> pd.DataFrame:
    return pd.DataFrame([form_values])


def predict_single(model, form_values: Dict) -> Tuple[int, float]:
    df = build_input_frame(form_values)
    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(probability >= 0.5)
    return prediction, probability


def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def recommendation_text(form_values: Dict, probability: float) -> list[str]:
    recs = []
    if form_values.get("Contract") == "Month-to-month":
        recs.append("This customer is on a month-to-month plan. That is a churn magnet. Offer a 12-month plan with a discount or bonus value.")
    if form_values.get("InternetService") == "Fiber optic":
        recs.append("Fiber optic accounts need extra attention. Check for support issues, speed complaints, or pricing pressure.")
    if form_values.get("PaymentMethod") == "Electronic check":
        recs.append("Electronic check users are often less sticky. Push autopay or card-based billing if the offer is acceptable.")
    if float(form_values.get("tenure", 0)) <= 6:
        recs.append("This is an early-life customer. The first 6 months need onboarding, proactive check-ins, and habit-building.")
    if float(form_values.get("MonthlyCharges", 0)) >= 80:
        recs.append("Monthly charges are high. High-value customers should get a review of plan value before they decide to leave.")
    if probability >= 0.70:
        recs.append("Treat this as an urgent save case. A tailored retention offer is justified.")
    elif probability >= 0.40:
        recs.append("This is a watch-list customer. Put them into a retention workflow instead of waiting.")
    else:
        recs.append("Risk looks manageable, but keep an eye on billing and support experience.")
    return recs
