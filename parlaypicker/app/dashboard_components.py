from __future__ import annotations

import streamlit as st
import pandas as pd


def render_top_bets(df: pd.DataFrame):
    st.subheader("Top Bets")
    st.dataframe(df.head(20), width="stretch")
