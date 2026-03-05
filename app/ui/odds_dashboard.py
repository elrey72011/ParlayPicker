import pandas as pd
import streamlit as st


def render_odds_table(df: pd.DataFrame) -> None:
    st.subheader("Live Odds")
    st.dataframe(df, use_container_width=True)
