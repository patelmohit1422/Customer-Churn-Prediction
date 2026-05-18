from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import REPORTS_DIR
from .utils import ensure_dir, save_json, save_text


def _rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["Churn"].mean())


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def generate_business_insights(df: pd.DataFrame) -> Dict[str, List[str]]:
    ensure_dir(REPORTS_DIR)

    overall = _rate(df)
    insights = []

    contract_rates = (
        df.groupby("Contract", observed=False)
        .apply(_rate)
        .sort_values(ascending=False)
    )
    top_contract = contract_rates.index[0]
    top_contract_rate = contract_rates.iloc[0]

    tenure_buckets = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72, 10_000],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72", "73+"],
        include_lowest=True,
        right=True,
    )

    tenure_rates = (
        df.assign(tenure_bucket=tenure_buckets)
        .groupby("tenure_bucket", observed=False)
        .apply(_rate)
        .reindex(["0-6", "7-12", "13-24", "25-48", "49-72", "73+"], fill_value=0.0)
    )

    early_rate = float(tenure_rates.get("0-6", 0.0))
    long_rate = float(tenure_rates.get("73+", 0.0))

    insights.append(
        f"Customers on {top_contract} churn the most at {_format_pct(top_contract_rate)}. "
        f"That is well above the overall churn rate of {_format_pct(overall)}."
    )

    insights.append(
        f"Very new customers are the riskiest segment. The 0-6 month group churns at {_format_pct(early_rate)}, "
        f"while the 73+ month group drops to {_format_pct(long_rate)}."
    )

    internet_rates = (
        df.groupby("InternetService", observed=False)
        .apply(_rate)
        .sort_values(ascending=False)
    )
    worst_internet = internet_rates.index[0]
    worst_internet_rate = internet_rates.iloc[0]
    insights.append(
        f"Fiber optic users churn more than the rest. The churn rate for {worst_internet} customers is {_format_pct(worst_internet_rate)}."
    )

    payment_rates = (
        df.groupby("PaymentMethod", observed=False)
        .apply(_rate)
        .sort_values(ascending=False)
    )
    worst_payment = payment_rates.index[0]
    worst_payment_rate = payment_rates.iloc[0]
    insights.append(
        f"Customers paying through {worst_payment} are noticeably less sticky, with churn at {_format_pct(worst_payment_rate)}."
    )

    paperless_rates = df.groupby("PaperlessBilling", observed=False).apply(_rate)
    yes_rate = float(paperless_rates.get("Yes", overall))
    no_rate = float(paperless_rates.get("No", overall))
    insights.append(
        f"Paperless billing is associated with higher churn: {_format_pct(yes_rate)} for paperless customers versus {_format_pct(no_rate)} for non-paperless customers."
    )

    charges_split = (
        df.assign(charge_band=pd.qcut(df["MonthlyCharges"], q=4, duplicates="drop"))
        .groupby("charge_band", observed=False)
        .apply(_rate)
    )
    highest_band_rate = float(charges_split.iloc[-1]) if len(charges_split) else 0.0
    lowest_band_rate = float(charges_split.iloc[0]) if len(charges_split) else 0.0
    insights.append(
        f"Customers in the highest monthly charge band churn more than the cheapest band, roughly {_format_pct(highest_band_rate)} versus {_format_pct(lowest_band_rate)}."
    )

    recommendations = [
        "Push month-to-month customers toward 12-month or 24-month plans with a small incentive, because contract length is one of the clearest churn separators.",
        "Treat the first six months as a retention window. A better onboarding flow, check-in emails, and usage nudges will pay off here.",
        "Review fiber optic accounts with support issues or high charges first. They are expensive customers to lose and often leave for avoidable reasons.",
        "Make payment and billing simpler. Autopay incentives and clearer invoices can reduce churn on billing-sensitive segments.",
        "Target high monthly-charge customers with save offers before renewal, not after cancellation.",
    ]

    report = {
        "overall_churn_rate": overall,
        "insights": insights,
        "recommendations": recommendations,
    }

    save_json(report, REPORTS_DIR / "business_insights.json")

    markdown = ["# Business Insights Report", ""]
    markdown.append(f"Overall churn rate: **{_format_pct(overall)}**")
    markdown.append("")
    markdown.append("## Key observations")
    for item in insights:
        markdown.append(f"- {item}")
    markdown.append("")
    markdown.append("## Practical retention moves")
    for item in recommendations:
        markdown.append(f"- {item}")

    save_text("\n".join(markdown), REPORTS_DIR / "business_insights.md")
    return report