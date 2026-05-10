"""
Dashboard page - run the workflow and view step-level outputs.
"""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from frontend import api_client
from frontend.components import customer_table, workflow_status


_DEMO_PROMPTS = [
    "Find high-value customers likely to convert for a personal loan this month and generate personalized WhatsApp messages.",
    "Identify medium-segment customers with recent loan inquiries and recommend the most suitable product.",
    "Generate personalized outreach for HIGH-segment customers above a 0.7 conversion probability.",
]

_STEP_LABELS = {
    "customer_discovery": "Customer Discovery",
    "customer_scoring": "Scoring",
    "product_recommendation": "Recommendation",
    "outreach_generation": "Outreach",
    "campaign_execution": "Campaign",
}


def _metric_block(state: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Discovered", len(state.get("discovered_customers") or []))
    c2.metric("Scored", len(state.get("scored_customers") or []))
    c3.metric("Recommended", len(state.get("recommendations") or []))
    c4.metric("Messages", len(state.get("generated_messages") or []))
    c5.metric("Campaigns", len(state.get("campaign_results") or []))


def _trigger_form() -> None:
    st.subheader("Trigger a new workflow run")
    with st.form("run_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_area(
                "RM request",
                value=_DEMO_PROMPTS[0],
                height=100,
            )
            preset = st.selectbox(
                "Demo scenarios",
                options=["(custom)"] + _DEMO_PROMPTS,
                index=0,
            )
            if preset != "(custom)":
                user_query = preset
        with col2:
            top_n = st.number_input("Top N customers", min_value=1, max_value=200, value=10)
            threshold = st.slider("Conversion threshold", 0.0, 1.0, 0.55, 0.05)
            loan_type = st.selectbox("Loan type", ["PERSONAL", "HOME", "CAR", "EDUCATION"], index=0)
            rm_name = st.text_input("RM name", value="Priya Sharma")

        submitted = st.form_submit_button("Run Workflow", type="primary")
        if submitted:
            with st.spinner("Executing multi-agent workflow..."):
                try:
                    state = api_client.run_workflow(
                        user_query=user_query,
                        rm_name=rm_name,
                        top_n_customers=int(top_n),
                        min_conversion_threshold=float(threshold),
                        loan_type=loan_type,
                    )
                    st.session_state["active_run_id"] = state.get("run_id")
                    st.success(f"Workflow complete - run_id: {state.get('run_id')}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Run failed: {exc}")


def _step_outputs(state: Dict[str, Any]) -> None:
    st.subheader("Step outputs")
    tab_disc, tab_score, tab_reco = st.tabs(["Discovery", "Scoring", "Recommendation"])
    with tab_disc:
        customer_table.render_discovered(state.get("discovered_customers") or [])
    with tab_score:
        customer_table.render_scored(state.get("scored_customers") or [])
    with tab_reco:
        customer_table.render_recommendations(state.get("recommendations") or [])


def _rerun_controls(state: Dict[str, Any]) -> None:
    st.subheader("Rerun a single step")
    run_id = state.get("run_id")
    if not run_id:
        st.caption("No active run.")
        return

    cols = st.columns(len(_STEP_LABELS))
    for col, (step_id, label) in zip(cols, _STEP_LABELS.items()):
        with col:
            if st.button(f"↻ {label}", key=f"rerun-{step_id}"):
                with st.spinner(f"Rerunning {label}..."):
                    try:
                        new_state = api_client.rerun_step(run_id, step_id)
                        st.session_state["active_run_id"] = new_state.get("run_id")
                        st.success(f"{label} rerun complete.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Rerun failed: {exc}")


def render() -> None:
    st.title("Multi-Agent Workflow Dashboard")
    st.caption(
        "Supervisor → Discovery → Scoring → Recommendation → Outreach → Campaign. "
        "Each agent calls only the MCP tools it is permitted to use."
    )

    _trigger_form()
    st.divider()

    run_id = st.session_state.get("active_run_id")
    if not run_id:
        st.info("Trigger a run above to view results.")
        return

    try:
        state = api_client.get_run(run_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load run {run_id}: {exc}")
        return

    st.markdown(f"**Active run:** `{run_id}`")
    workflow_status.render(state)
    st.divider()
    _metric_block(state)
    st.divider()
    _step_outputs(state)
    st.divider()
    _rerun_controls(state)
