"""
Execution Logs page - inspect the agent_logs audit trail and MCP catalog.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api_client


_AGENTS = [
    "",
    "SupervisorAgent",
    "CustomerDiscoveryAgent",
    "ScoringAgent",
    "RecommendationAgent",
    "OutreachAgent",
    "CampaignAgent",
]


def _logs_panel() -> None:
    st.subheader("Audit Trail")
    c1, c2, c3 = st.columns(3)
    run_id = c1.text_input(
        "Filter by run_id",
        value=st.session_state.get("active_run_id") or st.session_state.get("last_run_id") or "",
    )
    agent_name = c2.selectbox("Filter by agent", options=_AGENTS, index=0)
    limit = c3.number_input("Limit", min_value=10, max_value=2000, value=200, step=50)

    try:
        result = api_client.get_logs(
            run_id=run_id or None,
            agent_name=agent_name or None,
            limit=int(limit),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load logs: {exc}")
        return

    rows = result.get("logs") or []
    if not rows:
        st.info("No log rows match the filters.")
        return

    df = pd.DataFrame(rows)
    df = df[
        [
            "timestamp",
            "run_id",
            "agent_name",
            "tool_name",
            "step_name",
            "status",
            "duration_ms",
            "reasoning",
            "error_message",
        ]
    ]
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Detailed log entry")
    log_ids = [str(r["id"]) for r in rows]
    chosen = st.selectbox("Pick a log id", options=[""] + log_ids)
    if chosen:
        row = next((r for r in rows if str(r["id"]) == chosen), None)
        if row:
            st.json(row)


def _tools_panel() -> None:
    st.subheader("MCP Tool Catalog (Role-Based)")
    try:
        all_tools = api_client.list_tools().get("tools") or []
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load tools: {exc}")
        return

    st.markdown(f"**Registered tools:** {len(all_tools)}")
    st.dataframe(pd.DataFrame(all_tools), use_container_width=True, hide_index=True)

    st.markdown("**Tools per role**")
    roles = [
        "supervisor_agent",
        "discovery_agent",
        "scoring_agent",
        "recommendation_agent",
        "outreach_agent",
        "campaign_agent",
    ]
    for role in roles:
        try:
            payload = api_client.list_tools_for_role(role)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{role}: {exc}")
            continue
        with st.expander(f"{role} – {len(payload.get('tools', []))} tool(s)"):
            st.dataframe(pd.DataFrame(payload.get("tools", [])), use_container_width=True, hide_index=True)


def render() -> None:
    st.title("Execution Logs & MCP Catalog")
    st.caption("Every agent step and tool invocation is recorded here for full explainability.")

    t1, t2 = st.tabs(["Audit Trail", "Tool Catalog"])
    with t1:
        _logs_panel()
    with t2:
        _tools_panel()
