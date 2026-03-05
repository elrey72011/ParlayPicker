import pandas as pd
import streamlit as st


def render_debug_panel(df: pd.DataFrame):
    with st.expander("Debug Info"):
        st.write("Rows:", len(df))
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(20), use_container_width=True)
