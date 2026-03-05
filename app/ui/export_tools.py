import pandas as pd
import streamlit as st


def render_exports(df: pd.DataFrame, filename: str = "analysis_export.csv") -> None:
    st.download_button(
        "Export Analysis CSV",
        df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
    )
