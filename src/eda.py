from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import EDA_DIR, PLOTS_DIR
from .utils import ensure_dir, save_text


def _save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_eda_plots(df: pd.DataFrame) -> None:
    ensure_dir(EDA_DIR)
    ensure_dir(PLOTS_DIR)

    # 1) Churn distribution
    plt.figure(figsize=(6, 4))
    churn_counts = df["Churn"].map({0: "No Churn", 1: "Churn"}).value_counts().reindex(["No Churn", "Churn"])
    plt.bar(churn_counts.index.astype(str), churn_counts.values)
    plt.title("Churn Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Count")
    _save_fig(EDA_DIR / "churn_distribution.png")

    # 2) Churn vs contract
    contract_rates = (
        df.groupby("Contract")["Churn"]
        .mean()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(8, 4))
    plt.bar(contract_rates.index.astype(str), contract_rates.values)
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=15)
    _save_fig(EDA_DIR / "churn_vs_contract.png")

    # 3) Churn vs tenure
    plt.figure(figsize=(8, 4))
    churn0 = df.loc[df["Churn"] == 0, "tenure"]
    churn1 = df.loc[df["Churn"] == 1, "tenure"]
    plt.boxplot([churn0, churn1], labels=["No Churn", "Churn"])
    plt.title("Tenure vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (months)")
    _save_fig(EDA_DIR / "churn_vs_tenure.png")

    # 4) Churn vs monthly charges
    plt.figure(figsize=(8, 4))
    churn0 = df.loc[df["Churn"] == 0, "MonthlyCharges"]
    churn1 = df.loc[df["Churn"] == 1, "MonthlyCharges"]
    plt.boxplot([churn0, churn1], labels=["No Churn", "Churn"])
    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    _save_fig(EDA_DIR / "churn_vs_monthly_charges.png")

    # 5) Correlation heatmap
    corr_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(7, 5))
    im = plt.imshow(corr.values, interpolation="nearest")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.colorbar(im)
    plt.title("Correlation Heatmap")
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    _save_fig(EDA_DIR / "correlation_heatmap.png")


def save_eda_summary(df: pd.DataFrame) -> None:
    summary = []
    summary.append(f"Rows: {len(df):,}")
    summary.append(f"Churn rate: {df['Churn'].mean():.2%}")
    summary.append(f"Average tenure: {df['tenure'].mean():.1f} months")
    summary.append(f"Average monthly charges: {df['MonthlyCharges'].mean():.2f}")
    summary.append(f"Average total charges: {df['TotalCharges'].mean():.2f}")
    save_text("\n".join(summary), EDA_DIR / "eda_summary.txt")


def generate_eda_dashboard(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Churn distribution
    sns.countplot(x="Churn", data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Churn Distribution")

    # 2. Contract
    sns.barplot(x="Contract", y="Churn", data=df, ax=axes[0, 1])
    axes[0, 1].set_title("Churn vs Contract")

    # 3. Tenure
    sns.boxplot(x="Churn", y="tenure", data=df, ax=axes[0, 2])
    axes[0, 2].set_title("Tenure vs Churn")

    # 4. Monthly Charges
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=axes[1, 0])
    axes[1, 0].set_title("Monthly Charges vs Churn")

    # 5. Heatmap
    corr = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].corr()
    sns.heatmap(corr, annot=True, ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Heatmap")

    # Hide last empty plot
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("artifacts/eda/eda_dashboard.png")
    plt.close()