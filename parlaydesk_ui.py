"""UI helpers for ParlayDesk Streamlit app."""
from __future__ import annotations
from typing import Any, Dict, Optional, List
import streamlit as st
from parlaydesk_init import make_key


def render_debug_panel(
    api_config: Dict[str, bool],
    analyzer_path: str,
    last_exception: Optional[str],
    master_stats: Optional[Dict[str, Any]] = None,
    game_sample: Optional[List[Dict[str, Any]]] = None,
    normalization_stats: Optional[Dict[str, Any]] = None,
):
    with st.expander("Debug info", expanded=False):
        st.write("API configuration flags:")
        st.json(api_config)
        st.write(f"Analyzer path: {analyzer_path}")
        if normalization_stats:
            st.write("Game normalization stats:")
            st.json(normalization_stats)
        if master_stats:
            st.write("Vertex Master Analyzer stats:")
            st.json(master_stats)
        if game_sample:
            st.write("Sample games processed:")
            st.json(game_sample)
        if last_exception:
            st.error("Last exception (most recent run):")
            st.code(last_exception)
        else:
            st.success("No recent exceptions recorded.")


__all__ = ["render_debug_panel", "make_key"]
