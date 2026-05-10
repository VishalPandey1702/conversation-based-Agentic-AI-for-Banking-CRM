"""
Generated Messages page - WhatsApp-style preview of outreach messages.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api_client
from frontend.components import message_preview


def render() -> None:
    st.title("Generated Outreach Messages")
    st.caption("Personalized WhatsApp messages produced by the Outreach Agent.")

    run_id = st.session_state.get("active_run_id")
    if not run_id:
        st.info("Trigger a workflow run from the Dashboard to populate this view.")
        return

    try:
        msg_payload = api_client.get_messages(run_id)
        camp_payload = api_client.get_campaigns(run_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load messages: {exc}")
        return

    messages = msg_payload.get("messages") or []
    campaigns = camp_payload.get("campaigns") or []

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("WhatsApp preview")
        message_preview.render(messages)
    with c2:
        st.subheader("Campaign dispatch status")
        if campaigns:
            df = pd.DataFrame(campaigns)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No campaign records yet.")
