import pandas as pd
import streamlit as st


def render_odds_table(df: pd.DataFrame) -> None:
    st.subheader("Live Odds")
    display_df = df.copy()
    if isinstance(display_df, pd.DataFrame) and not display_df.empty:
        ordered = [c for c in ["game_date", "game_time_est"] if c in display_df.columns]
        trailing = [c for c in display_df.columns if c not in ordered]
        display_df = display_df[ordered + trailing]
    st.dataframe(display_df, width="stretch")
