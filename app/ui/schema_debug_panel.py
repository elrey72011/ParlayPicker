import pandas as pd
import streamlit as st


def render_schema_debug(df: pd.DataFrame):
    with st.expander("Schema Debug"):
        st.write("Columns:")
        st.write(list(df.columns))
        st.write("Row count:", len(df))
