import pandas as pd
import streamlit as st


def render_exports(df: pd.DataFrame, parlays: pd.DataFrame | None = None) -> None:
    if parlays is None:
        parlays = pd.DataFrame()

    analysis_csv = df.to_csv(index=False)
    st.download_button(
        "Export Analysis",
        analysis_csv,
        "analysis_export.csv",
        mime="text/csv",
    )

    parlays_csv = parlays.to_csv(index=False)
    st.download_button(
        "Export Parlays",
        parlays_csv,
        "parlays_export.csv",
        mime="text/csv",
    )
