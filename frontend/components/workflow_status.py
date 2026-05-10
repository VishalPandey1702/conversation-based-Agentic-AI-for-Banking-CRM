"""
UI component that shows the current workflow execution status.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

WORKFLOW_STEPS = [
    ("customer_discovery", "Customer Discovery"),
    ("customer_scoring", "Scoring"),
    ("product_recommendation", "Recommendation"),
    ("outreach_generation", "Outreach"),
    ("campaign_execution", "Campaign"),
]


def render(state: Dict[str, Any]) -> None:
    """Render the step-by-step status badges + summary block."""
    completed: List[str] = state.get("completed_steps") or []
    current = state.get("current_step")
    errors: List[str] = state.get("errors") or []

    cols = st.columns(len(WORKFLOW_STEPS))
    for col, (step_id, label) in zip(cols, WORKFLOW_STEPS):
        if step_id in completed:
            status = "DONE"
            colour = "green"
        elif step_id == current:
            status = "RUNNING"
            colour = "orange"
        else:
            status = "PENDING"
            colour = "gray"
        with col:
            st.markdown(
                f"""
                <div style="border-radius:10px; padding:12px; background:#0f172a; text-align:center; border:1px solid #1e293b;">
                    <div style="font-size:12px; color:#94a3b8;">{label}</div>
                    <div style="font-weight:700; color:{colour}; margin-top:4px;">{status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if state.get("summary"):
        st.info(state["summary"])

    if errors:
        with st.expander(f"{len(errors)} error(s) recorded during this run", expanded=False):
            for e in errors:
                st.error(e)
