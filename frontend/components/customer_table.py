"""
Reusable Streamlit table renderers for customer-related data.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def render_discovered(rows: List[Dict[str, Any]]) -> None:
    """Render the output of the discovery agent."""
    if not rows:
        st.info("No discovered customers yet. Run the workflow to populate this view.")
        return

    df = pd.DataFrame(
        [
            {
                "ID": r.get("id"),
                "Code": r.get("customer_code"),
                "Name": r.get("full_name"),
                "Segment": r.get("customer_segment"),
                "Income (₹)": r.get("annual_income"),
                "Credit Score": r.get("credit_score"),
                "Recent Inquiry": r.get("has_recent_inquiry"),
                "Has Loan": r.get("has_existing_loan"),
                "Reason": r.get("discovery_reason"),
            }
            for r in rows
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_scored(rows: List[Dict[str, Any]]) -> None:
    """Render the output of the scoring agent."""
    if not rows:
        st.info("No scored customers yet.")
        return

    df = pd.DataFrame(
        [
            {
                "ID": r["customer"]["id"],
                "Name": r["customer"]["full_name"],
                "Segment": r["customer"]["customer_segment"],
                "Conversion Probability": r.get("conversion_probability"),
                "Score": r.get("score"),
                "Above Threshold": r.get("above_threshold"),
                "Txn Consistency": r.get("txn_consistency"),
                "Rationale": r.get("rationale"),
            }
            for r in rows
        ]
    )
    df = df.sort_values(by="Conversion Probability", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_recommendations(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.info("No recommendations available yet.")
        return
    df = pd.DataFrame(
        [
            {
                "ID": r["customer"]["id"],
                "Name": r["customer"]["full_name"],
                "Segment": r["customer"]["customer_segment"],
                "Probability": r.get("conversion_probability"),
                "Product": r.get("recommendation", {}).get("product_name"),
                "Amount (₹)": r.get("recommendation", {}).get("recommended_amount"),
                "Rate %": r.get("recommendation", {}).get("interest_rate"),
                "Tenure (m)": r.get("recommendation", {}).get("tenure_months"),
                "Confidence": r.get("recommendation", {}).get("confidence"),
                "Rationale": r.get("recommendation", {}).get("rationale"),
            }
            for r in rows
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
