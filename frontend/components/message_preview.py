"""
WhatsApp-style message preview component.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


_BUBBLE_CSS = """
<style>
.wa-bubble {
    background: #0b3d2e;
    color: #e2e8f0;
    padding: 14px 16px;
    border-radius: 14px 14px 14px 4px;
    margin-bottom: 6px;
    max-width: 720px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    line-height: 1.45;
    font-size: 14px;
    white-space: pre-wrap;
}
.wa-meta {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 10px;
}
</style>
"""


def render(messages: List[Dict[str, Any]]) -> None:
    """Render WhatsApp-style bubbles with metadata."""
    if not messages:
        st.info("No outreach messages were generated for this run.")
        return

    st.markdown(_BUBBLE_CSS, unsafe_allow_html=True)
    for m in messages:
        st.markdown(
            f"<div class='wa-meta'><b>{m.get('full_name','-')}</b> "
            f"({m.get('customer_code','-')}) · "
            f"{m.get('phone','-')} · "
            f"<i>{m.get('product_name','-')}</i></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='wa-bubble'>{m.get('message','')}</div>", unsafe_allow_html=True)
        with st.expander("Why this message?"):
            st.write(m.get("rationale") or "No rationale recorded.")
