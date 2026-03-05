from typing import Dict, List

import streamlit as st


DEFAULT_SPORTS = ["NBA", "NFL", "NHL", "MLB", "NCAAB"]


def render_sidebar() -> Dict[str, object]:
    st.sidebar.header("Controls")
    sports: List[str] = st.sidebar.multiselect("Select Sports", DEFAULT_SPORTS, default=["NBA", "NCAAB"])
    max_rows = st.sidebar.slider("Max rows", min_value=10, max_value=300, value=75, step=5)
    include_realtime = st.sidebar.checkbox("Show Real-Time Edges", value=True)
    include_portfolio = st.sidebar.checkbox("Show Portfolio Allocation", value=True)
    run_analysis = st.sidebar.button("Run Master Analysis", type="primary")

    return {
        "sports": sports,
        "max_rows": max_rows,
        "include_realtime": include_realtime,
        "include_portfolio": include_portfolio,
        "run_analysis": run_analysis,
    }
