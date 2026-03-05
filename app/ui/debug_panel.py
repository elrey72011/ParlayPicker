import pandas as pd
import streamlit as st


def show_pipeline_debug(df: pd.DataFrame) -> None:
    st.subheader("Pipeline Debug")
    st.write("Rows:", len(df))
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(20), use_container_width=True)


def render_debug(df: pd.DataFrame) -> None:
    show_pipeline_debug(df)


def render_debug_panel(df: pd.DataFrame) -> None:
    with st.expander("Debug Info"):
        show_pipeline_debug(df)
