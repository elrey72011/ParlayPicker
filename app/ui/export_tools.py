import pandas as pd
import streamlit as st


def render_exports(df: pd.DataFrame, parlays: pd.DataFrame | None = None) -> None:
    if parlays is None:
        parlays = pd.DataFrame()

    st.download_button(
        "Export Analysis",
        df.to_csv(index=False),
        "analysis_export.csv",
    )

    st.download_button(
        "Export Parlays",
        parlays.to_csv(index=False),
        "parlays_export.csv",
    )
