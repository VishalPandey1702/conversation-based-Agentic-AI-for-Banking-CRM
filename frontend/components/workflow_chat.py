"""
Conversational UI for the Banking CRM agentic system.

A single Conversational Orchestrator agent decides — per turn — which
specialist tools to call (Discovery, Scoring, Recommendation, Outreach,
Campaign, etc.). The UI renders:
- the assistant's natural-language reply,
- a step-by-step "tool trace" showing every agent/tool the orchestrator invoked.

Default settings (Top N, threshold, loan type, RM name) sit above the chat
as suggestions; the orchestrator can override them from natural-language
instructions like "top 5", "above 0.7 probability", "for HOME loan".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from frontend import api_client
from frontend.components.orchestration_viz import orchestration_html


_TOOL_LABELS: Dict[str, str] = {
    "list_customers": "Filter customers",
    "get_customer": "Customer 360°",
    "discover_loan_candidates": "Discovery agent",
    "score_customers": "Scoring agent",
    "recommend_product": "Recommendation agent",
    "generate_outreach": "Outreach agent",
    "send_whatsapp": "Campaign agent (1)",
    "send_all_pending": "Campaign agent (batch)",
    "run_full_workflow": "Full pipeline",
    "explain_customer": "Explainability",
    "get_session_summary": "Session snapshot",
}


def _init_session() -> None:
    if "chat_log" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str, "extra": dict}
        st.session_state.chat_log: List[Dict[str, Any]] = []
    if "conv_id" not in st.session_state:
        st.session_state.conv_id = None


def _history_for_api() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for e in st.session_state.chat_log[-10:]:
        out.append({"role": e["role"], "content": e.get("content", "")})
    return out


def _render_tool_trace(trace: List[Dict[str, Any]]) -> None:
    if not trace:
        return
    summary = " → ".join(_TOOL_LABELS.get(t.get("tool"), t.get("tool", "?")) for t in trace)
    with st.expander(f"Reasoning trace · {len(trace)} tool call(s) · {summary}", expanded=False):
        for i, t in enumerate(trace, start=1):
            ok = t.get("ok", True)
            tool = t.get("tool", "?")
            label = _TOOL_LABELS.get(tool, tool)
            st.markdown(f"**{i}. {label}** _(tool: `{tool}`)_  " + ("✓" if ok else "✗"))
            args = t.get("args") or {}
            if args:
                st.code(args, language="json")
            brief = t.get("result_brief") or {}
            if brief:
                st.caption("Result")
                st.code(brief, language="json")


def _render_session_summary(summary: Dict[str, Any]) -> None:
    if not summary:
        return
    cols = st.columns(5)
    cols[0].metric("Discovered", summary.get("discovered", 0))
    cols[1].metric("Scored", summary.get("scored", 0))
    cols[2].metric("Reco'd", summary.get("recommendations", 0))
    cols[3].metric("Messages", summary.get("messages", 0))
    cols[4].metric("Sent", summary.get("campaigns", 0))


def _render_assistant(entry: Dict[str, Any]) -> None:
    extra = entry.get("extra") or {}
    with st.chat_message("assistant"):
        st.markdown(entry.get("content") or "_(no reply)_")
        # Pull any "data" the orchestrator returned via tools that produced tables
        if customers := extra.get("customers"):
            df = pd.DataFrame(
                [
                    {
                        "Code": x.get("customer_code"),
                        "Name": x.get("full_name"),
                        "Segment": x.get("customer_segment"),
                        "Income (₹)": x.get("annual_income"),
                        "Credit": x.get("credit_score"),
                    }
                    for x in customers
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        # Tool trace
        _render_tool_trace(extra.get("tool_trace") or [])
        # Working set summary
        if extra.get("session_summary"):
            with st.expander("Working memory (this conversation)"):
                _render_session_summary(extra["session_summary"])


def _render_user(entry: Dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(entry.get("content") or "")


def _flatten_table_data(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """If certain tools were called, surface their tabular output for the bubble."""
    out: Dict[str, Any] = {}
    for t in trace:
        if not t.get("ok"):
            continue
        tool = t.get("tool")
        brief = t.get("result_brief") or {}
        if tool in {"list_customers", "discover_loan_candidates"}:
            sample = brief.get("customers_first") or brief.get("candidates_first") or []
            if sample:
                out["customers"] = sample[:1]  # token-cheap; the full table lives in trace JSON
    return out


def render() -> None:
    _init_session()

    st.title("RM Banking CRM — talk to your agents")
    st.caption(
        "One **Conversational Orchestrator** agent. Specialist agents are exposed as tools "
        "(Discovery / Scoring / Recommendation / Outreach / Campaign). It decides which tools "
        "to call dynamically — no fixed pipeline."
    )

    # ----- Default settings -----
    with st.container():
        st.markdown("##### Defaults the orchestrator may use")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            top_n = st.number_input("Top N customers", min_value=1, max_value=200, value=10, step=1, key="cfg_top_n")
        with c2:
            threshold = st.slider("Conversion threshold", 0.0, 1.0, 0.55, 0.05, key="cfg_threshold")
        with c3:
            loan_type = st.selectbox("Loan type", ["PERSONAL", "HOME", "CAR", "EDUCATION"], index=0, key="cfg_loan_type")
        with c4:
            rm_name = st.text_input("RM name", value="Priya Sharma", key="cfg_rm_name")
        st.caption(
            "Override anytime by saying things like *top 5*, *above 0.7*, *HOME loan*, or *HIGH segment*."
        )

    st.divider()
    st.markdown("##### Conversation")

    # Suggested prompts (only when chat is empty)
    if not st.session_state.chat_log:
        st.caption("Try a quick prompt to get started:")
        cols = st.columns(3)
        suggestions = [
            "Find HIGH segment customers with credit > 750.",
            "Score the top 5 candidates and show their probabilities.",
            "Generate a personalized WhatsApp message for the top customer.",
        ]
        for col, text in zip(cols, suggestions):
            with col:
                if st.button(text, key=f"sug-{hash(text)}", use_container_width=True):
                    st.session_state["_pending_user_message"] = text
                    st.rerun()

    # Reset
    if st.session_state.chat_log:
        if st.button("Reset conversation", use_container_width=False):
            try:
                if st.session_state.conv_id:
                    api_client.reset_chat(st.session_state.conv_id)
            except Exception:  # noqa: BLE001
                pass
            st.session_state.chat_log = []
            st.session_state.conv_id = None
            st.rerun()

    # Render history
    for entry in st.session_state.chat_log:
        if entry["role"] == "user":
            _render_user(entry)
        else:
            _render_assistant(entry)

    # ----- Chat input -----
    typed = st.chat_input(
        "Tell me what to do — e.g. 'find HIGH segment customers, score them, outreach to the top one'."
    )
    queued: Optional[str] = st.session_state.pop("_pending_user_message", None)
    user_msg: Optional[str] = typed or queued

    if not user_msg:
        return

    # Append user bubble + render immediately
    st.session_state.chat_log.append({"role": "user", "content": user_msg, "extra": {}})

    try:
        with st.status("**Agents working together** — orchestration in progress", expanded=True):
            st.markdown(orchestration_html(), unsafe_allow_html=True)
            st.caption(
                "The orchestrator selects specialist agents via LLM tool-calling. "
                "Their full reasoning trace appears below the reply."
            )
            resp = api_client.chat(
                message=user_msg,
                history=_history_for_api(),
                rm_name=rm_name,
                conv_id=st.session_state.conv_id,
                top_n_customers=int(top_n),
                min_conversion_threshold=float(threshold),
                loan_type=loan_type,
            )
    except Exception as exc:  # noqa: BLE001
        st.session_state.chat_log.append(
            {
                "role": "assistant",
                "content": f"Could not reach the orchestrator: `{exc}`",
                "extra": {},
            }
        )
        st.rerun()
        return

    st.session_state.conv_id = resp.get("conv_id") or st.session_state.conv_id
    trace = resp.get("tool_trace") or []
    extra = {
        "tool_trace": trace,
        "session_summary": resp.get("session_summary") or {},
        **_flatten_table_data(trace),
    }
    if resp.get("error"):
        extra["error"] = resp["error"]

    st.session_state.chat_log.append(
        {
            "role": "assistant",
            "content": resp.get("reply") or "_(no reply)_",
            "extra": extra,
        }
    )
    st.rerun()
