import pandas as pd
import streamlit as st


ANALYSIS_COLUMNS = [
    "league",
    "away_team",
    "home_team",
    "consensus_prob",
    "market_probability",
    "ml_probability",
    "ai_probability",
    "edge",
    "expected_value",
    "best_pick",
]

DEBUG_COLUMNS = [
    "home_team",
    "away_team",
    "odds_american",
    "market_probability",
    "ml_probability",
    "consensus_prob",
    "expected_value",
    "edge",
]


def render_analysis(df: pd.DataFrame) -> None:
    st.subheader("Model Analysis")
    available_cols = [col for col in ANALYSIS_COLUMNS if col in df.columns]
    if available_cols:
        st.dataframe(df[available_cols], width="stretch")
    else:
        st.dataframe(df, width="stretch")

    debug_cols = [col for col in DEBUG_COLUMNS if col in df.columns]
    if debug_cols:
        st.caption("Debug probability/EV table")
        st.dataframe(df[debug_cols], width="stretch")
