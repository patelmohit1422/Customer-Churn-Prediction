from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, ID_COLUMN, TARGET_COLUMN


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [c.strip() for c in cleaned.columns]

    if ID_COLUMN in cleaned.columns:
        cleaned = cleaned.drop(columns=[ID_COLUMN])

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype("int64")

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    # Real-world fix: when TotalCharges is blank, reconstruct it using monthly charges and tenure.
    missing_total = cleaned["TotalCharges"].isna()
    reconstructed = cleaned.loc[missing_total, "MonthlyCharges"] * cleaned.loc[missing_total, "tenure"]
    cleaned.loc[missing_total, "TotalCharges"] = reconstructed

    # Remaining missing values are handled by the pipeline, but we also make sure
    # the numeric column is sane before any downstream grouping.
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["TotalCharges"].median())

    return cleaned


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
