import pandas as pd
import streamlit as st


ANALYSIS_COLUMNS = [
    "league",
    "away_team",
    "home_team",
    "consensus_prob",
    "expected_value",
    "best_pick",
]


def render_analysis(df: pd.DataFrame) -> None:
    st.subheader("Model Analysis")
    available_cols = [col for col in ANALYSIS_COLUMNS if col in df.columns]
    if available_cols:
        st.dataframe(df[available_cols], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
