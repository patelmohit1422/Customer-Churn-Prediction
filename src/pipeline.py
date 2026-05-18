from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ARTIFACTS_DIR, EDA_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR
from .data_loader import load_raw_data
from .eda import generate_eda_plots, save_eda_summary
from .insights import generate_business_insights
from .modeling import (
    get_feature_importance,
    save_best_model,
    save_metrics,
    select_best_model,
    train_and_compare_models,
)
from .preprocessing import clean_telco_data
from .utils import ensure_dir, save_dataframe, save_json, save_text


def run_full_pipeline() -> None:
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(EDA_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(MODELS_DIR)

    print("Loading dataset...")
    raw_df = load_raw_data()
    print(f"Raw data shape: {raw_df.shape}")

    print("Cleaning dataset...")
    df = clean_telco_data(raw_df)
    save_dataframe(df, ARTIFACTS_DIR / "cleaned_dataset.csv")
    save_eda_summary(df)

    print("Generating EDA plots...")
    generate_eda_plots(df)

    print("Generating business insights...")
    report = generate_business_insights(df)

    print("Training models...")
    results, pipelines, split_data = train_and_compare_models(df)
    X_train, X_test, y_train, y_test = split_data

    save_metrics(results, ARTIFACTS_DIR / "model_metrics.csv")
    save_json(results.to_dict(orient="records"), ARTIFACTS_DIR / "model_metrics.json")

    best_name, best_model = select_best_model(results, pipelines)
    print(f"Best model selected: {best_name}")

    save_best_model(best_model, MODELS_DIR / "telco_churn_model.joblib")

    metadata = {
        "best_model": best_name,
        "feature_columns": list(X_train.columns),
        "target_column": "Churn",
        "positive_class_label": 1,
        "risk_thresholds": {"high": 0.70, "medium": 0.40},
        "performance_summary": results.to_dict(orient="records"),
    }
    save_json(metadata, MODELS_DIR / "metadata.json")

    print("Computing feature importance...")
    importance_df = get_feature_importance(best_model, X_test, y_test)
    save_dataframe(importance_df, ARTIFACTS_DIR / "feature_importance.csv")
    importance_df.head(15).to_csv(ARTIFACTS_DIR / "feature_importance_top15.csv", index=False)

    # Save a human-readable summary file for GitHub visitors.
    summary_lines = [
        "# Training Summary",
        "",
        f"Best model: {best_name}",
        "",
        "Model comparison:",
        results.to_string(index=False),
        "",
        "Top feature importance:",
        importance_df.head(10).to_string(index=False),
        "",
        "Business insights were saved to artifacts/reports/business_insights.md",
    ]
    save_text("\n".join(summary_lines), ARTIFACTS_DIR / "training_summary.md")

    print("\nTraining complete.")
    print(results.to_string(index=False))
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")
