from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import ALL_FEATURES, MODELS_DIR  # noqa: E402
from src.inference import load_artifacts, recommendation_text, risk_level  # noqa: E402
from src.pipeline import run_full_pipeline  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

st.title("Customer Churn Prediction")
st.caption("Production-style churn scoring dashboard for retention teams.")

model_path = MODELS_DIR / "telco_churn_model.joblib"
metadata_path = MODELS_DIR / "metadata.json"

if not model_path.exists() or not metadata_path.exists():
    with st.spinner("Training model for the first run..."):
        run_full_pipeline()

if not model_path.exists():
    st.error("Model still not found after training. Check the terminal logs for the failing step.")
    st.stop()

model, metadata = load_artifacts()

st.sidebar.header("Customer Details")

with st.sidebar.form("churn_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=float(tenure) * float(monthly_charges),
        step=1.0,
        help="If you do not know this exactly, the app will still work with a realistic estimated value.",
    )

    submit = st.form_submit_button("Predict churn")

if submit:
    payload = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
    }

    input_df = pd.DataFrame([payload], columns=ALL_FEATURES)
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = 1 if probability >= 0.5 else 0
    risk = risk_level(probability)
    recs = recommendation_text(payload, probability)

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{probability:.1%}")
    col2.metric("Risk Level", risk)
    col3.metric("Prediction", "Churn" if prediction == 1 else "Stay")

    st.subheader("Assessment")
    if risk == "High":
        st.error("High churn risk. This customer should be reviewed immediately.")
    elif risk == "Medium":
        st.warning("Medium churn risk. Put this customer into a retention workflow.")
    else:
        st.success("Low churn risk. Keep monitoring, but no urgent action needed.")

    st.subheader("Recommendations")
    for rec in recs:
        st.write(f"- {rec}")

    st.subheader("Saved Model Snapshot")
    st.write(f"Best model: **{metadata.get('best_model', 'Unknown')}**")
    st.write("Risk thresholds:", metadata.get("risk_thresholds", {}))

st.divider()
st.subheader("What this app does")
st.write(
    "This dashboard is built for retention teams, account managers, and business owners who need a quick churn score, "
    "not a notebook full of code."
)
