from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .config import RAW_DATA_DIR, RAW_DATA_URL
from .synthetic_data import generate_demo_telco_data
from .utils import ensure_dir


def download_dataset(force: bool = False) -> Path:
    ensure_dir(RAW_DATA_DIR)
    destination = RAW_DATA_DIR / "telco_customer_churn.csv"
    if destination.exists() and not force:
        return destination

    try:
        response = requests.get(RAW_DATA_URL, timeout=30)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return destination
    except Exception:
        # Offline fallback so the project still runs end-to-end in restricted environments.
        fallback_df = generate_demo_telco_data()
        fallback_df.to_csv(destination, index=False)
        return destination


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    csv_path = download_dataset(force=force_download)
    df = pd.read_csv(csv_path)
    return df
