"""UI helpers for ParlayDesk Streamlit app."""
from __future__ import annotations
from typing import Any, Dict, Optional
import streamlit as st
from parlaydesk_init import make_key


def render_debug_panel(api_config: Dict[str, bool], analyzer_path: str, last_exception: Optional[str]):
    with st.expander("Debug info", expanded=False):
        st.write("API configuration flags:")
        st.json(api_config)
        st.write(f"Analyzer path: {analyzer_path}")
        if last_exception:
            st.error("Last exception (most recent run):")
            st.code(last_exception)
        else:
            st.success("No recent exceptions recorded.")


__all__ = ["render_debug_panel", "make_key"]
