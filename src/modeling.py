from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

from .config import RANDOM_STATE, TEST_SIZE, MODELS_DIR, ALL_FEATURES
from .preprocessing import build_preprocessor, split_features_target
from .utils import ensure_dir, save_dataframe, save_json


def _predict_scores(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        return 1 / (1 + np.exp(-raw))
    return model.predict(X_test).astype(float)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    probas = _predict_scores(model, X_test)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probas)),
    }


def build_candidate_models() -> Dict[str, BaseEstimator]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_and_compare_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    candidate_models = build_candidate_models()
    fitted_pipelines: Dict[str, Pipeline] = {}
    records = []

    for name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["model"] = name
        records.append(metrics)
        fitted_pipelines[name] = pipeline

    results = pd.DataFrame(records).sort_values(by=["f1", "roc_auc"], ascending=False).reset_index(drop=True)
    return results, fitted_pipelines, (X_train, X_test, y_train, y_test)


def select_best_model(results: pd.DataFrame, fitted_pipelines: Dict[str, Pipeline]) -> Tuple[str, Pipeline]:
    best_row = results.iloc[0]
    best_name = str(best_row["model"])
    return best_name, fitted_pipelines[best_name]


def save_metrics(results: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    results.to_csv(path, index=False)


def save_best_model(model: Pipeline, path: Path) -> None:
    ensure_dir(path.parent)
    dump(model, path)


def get_feature_importance(best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    preprocessor = best_model.named_steps["preprocessor"]
    model = best_model.named_steps["model"]

    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        importances = coef[0]
    else:
        perm = permutation_importance(best_model, X_test, y_test, scoring="f1", n_repeats=10, random_state=RANDOM_STATE)
        feature_names = list(X_test.columns)
        importances = perm.importances_mean

    importance_df = pd.DataFrame(
        {
            "feature": feature_names[: len(importances)],
            "importance": importances[: len(feature_names)],
        }
    ).sort_values("importance", ascending=False)

    return importance_df.reset_index(drop=True)
