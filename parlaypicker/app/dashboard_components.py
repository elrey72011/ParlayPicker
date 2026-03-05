from __future__ import annotations

import streamlit as st
import polars as pl


def render_top_bets(df: pl.DataFrame):
    st.subheader("Top Bets")
    st.dataframe(df.head(20).to_pandas(), width="stretch")
