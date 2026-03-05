import pandas as pd
import streamlit as st


def render_debug(df: pd.DataFrame) -> None:
    st.subheader("Pipeline Diagnostics")
    st.write("Rows:", len(df))
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(25), use_container_width=True)


def render_debug_panel(df: pd.DataFrame) -> None:
    with st.expander("Debug Info"):
        render_debug(df)
