import pandas as pd
import streamlit as st


def render_analysis(df: pd.DataFrame) -> None:
    st.subheader("Model Analysis")
    st.dataframe(df, use_container_width=True)
