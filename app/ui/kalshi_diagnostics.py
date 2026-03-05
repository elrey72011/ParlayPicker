import pandas as pd
import streamlit as st


KALSHI_COLUMNS = [
    "haskalshimarket",
    "kalshi_status",
    "kalshi_prob_for_pick",
    "kalshi_yes_side",
    "kalshi_ticker",
]


def render_kalshi_diagnostics(df: pd.DataFrame) -> None:
    st.subheader("Kalshi Diagnostics")
    normalized = {c.lower(): c for c in df.columns}
    selected = [normalized[c] for c in KALSHI_COLUMNS if c in normalized]

    if not selected:
        st.info("No Kalshi diagnostic columns found in analysis output.")
        return

    st.dataframe(df[selected].head(50), use_container_width=True)
