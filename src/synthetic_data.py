from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


def generate_demo_telco_data(n_rows: int = 7043) -> pd.DataFrame:
    """Generate a Telco-like dataset for offline demo/runtime fallback.

    The schema matches the Telco churn project, so the pipeline and Streamlit app
    can still run when the real CSV cannot be downloaded.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    customer_id = [f"{i:04d}-{rng.integers(1000, 9999)}" for i in range(n_rows)]

    gender = rng.choice(["Female", "Male"], size=n_rows)
    senior = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, size=n_rows)

    phone_service = rng.choice(["Yes", "No"], size=n_rows, p=[0.91, 0.09])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_rows, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.34, 0.44, 0.22])
    no_internet = internet_service == "No"

    def service_choice(yes_p: float, no_p: float):
        return np.where(
            no_internet,
            "No internet service",
            rng.choice(["Yes", "No"], size=n_rows, p=[yes_p, no_p]),
        )

    online_security = service_choice(0.34, 0.66)
    online_backup = service_choice(0.40, 0.60)
    device_protection = service_choice(0.41, 0.59)
    tech_support = service_choice(0.30, 0.70)
    streaming_tv = service_choice(0.42, 0.58)
    streaming_movies = service_choice(0.43, 0.57)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], size=n_rows, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    monthly_charges = (
        18
        + (internet_service == "Fiber optic") * rng.normal(50, 9, size=n_rows)
        + (internet_service == "DSL") * rng.normal(25, 7, size=n_rows)
        + (phone_service == "Yes") * rng.normal(12, 3, size=n_rows)
        + (streaming_tv == "Yes") * rng.normal(8, 2, size=n_rows)
        + (streaming_movies == "Yes") * rng.normal(8, 2, size=n_rows)
        + (online_security == "Yes") * rng.normal(4, 1, size=n_rows)
    )
    monthly_charges = np.clip(monthly_charges, 18.25, 120)

    total_charges = np.clip(monthly_charges * tenure + rng.normal(0, 80, size=n_rows), 0, None)
    total_charges = np.where(tenure == 0, np.nan, total_charges)

    # churn probability heuristic to create realistic structure
    logit = (
        -1.25
        + 0.045 * (monthly_charges - 70)
        - 0.055 * tenure
        + 0.8 * (contract == "Month-to-month")
        - 0.6 * (contract == "Two year")
        + 0.35 * (internet_service == "Fiber optic")
        + 0.45 * (payment_method == "Electronic check")
        + 0.35 * (paperless == "Yes")
        - 0.45 * (online_security == "Yes")
        - 0.30 * (tech_support == "Yes")
        + 0.20 * senior
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob, size=n_rows)
    churn = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
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
            "MonthlyCharges": np.round(monthly_charges, 2),
            "TotalCharges": np.round(total_charges, 2),
            "Churn": churn,
        }
    )
    return df
