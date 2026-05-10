"""
Streamlit entry-point for the Agentic Banking CRM.

Run with:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the script directory first on sys.path, so `frontend` is not
# importable as a package unless the repo root is also on the path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from frontend import api_client
from frontend.components.workflow_chat import render as render_workflow_chat
from frontend.pages import customers as customers_page, logs as logs_page


st.set_page_config(
    page_title="Agentic Banking CRM",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _sidebar() -> str:
    st.sidebar.markdown("## RM Banking CRM")
    st.sidebar.caption("Agentic workflow — chat-style run log")

    page = st.sidebar.radio(
        "Go to",
        options=["Workflow chat", "Customers", "Logs"],
        index=0,
        label_visibility="visible",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**API**")
    try:
        h = api_client.health()
        st.sidebar.success(f"Connected · LLM {'on' if h.get('llm_configured') else 'off'}")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"API offline: {exc}")

    if st.session_state.get("last_run_id"):
        st.sidebar.caption(f"Last run: `{st.session_state.last_run_id}`")

    return page


def main() -> None:
    if "last_run_id" not in st.session_state:
        st.session_state.last_run_id = None

    page = _sidebar()

    if page == "Workflow chat":
        render_workflow_chat()
    elif page == "Customers":
        customers_page.render()
    else:
        logs_page.render()


if __name__ == "__main__":
    main()
