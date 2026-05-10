"""
Customer Insights page - browse the customer database and 360° profiles.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api_client


def render() -> None:
    st.title("Customer Insights")
    st.caption("Browse the customer master, filter by segment/credit, and inspect 360° profiles.")

    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        segment = c1.selectbox("Segment", options=["", "HIGH", "MEDIUM", "LOW"], index=0)
        min_credit = c2.number_input("Min credit score", min_value=0, max_value=900, value=0, step=10)
        min_income = c3.number_input("Min annual income (₹)", min_value=0, value=0, step=100_000)
        limit = c4.number_input("Limit", min_value=1, max_value=500, value=50)

    try:
        result = api_client.list_customers(
            segment=segment or None,
            min_credit_score=int(min_credit),
            min_income=float(min_income),
            limit=int(limit),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load customers: {exc}")
        return

    rows = result.get("customers") or []
    if not rows:
        st.info("No customers match the filters.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("360° Customer Profile")
    if rows:
        options = {f"{r['customer_code']} – {r['full_name']}": r["id"] for r in rows}
        choice = st.selectbox("Pick a customer", options=list(options.keys()))
        if choice:
            try:
                profile = api_client.get_customer_profile(options[choice])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load profile: {exc}")
                return

            st.json(profile.get("customer", {}))

            t1, t2 = st.tabs(["Recent Loan Inquiries", "Recent CRM Interactions"])
            with t1:
                inquiries = profile.get("recent_inquiries") or []
                if inquiries:
                    st.dataframe(pd.DataFrame(inquiries), use_container_width=True, hide_index=True)
                else:
                    st.caption("No recent inquiries.")
            with t2:
                interactions = profile.get("recent_interactions") or []
                if interactions:
                    st.dataframe(pd.DataFrame(interactions), use_container_width=True, hide_index=True)
                else:
                    st.caption("No recent interactions.")

            st.metric("Transactions in last 90 days", profile.get("transaction_count_90d", 0))
