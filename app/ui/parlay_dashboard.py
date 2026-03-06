import pandas as pd
import streamlit as st


def render_parlays(parlays: pd.DataFrame) -> None:
    st.dataframe(parlays, width="stretch")
