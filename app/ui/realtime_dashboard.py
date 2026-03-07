import pandas as pd
import streamlit as st


def render_realtime_edges(edges: pd.DataFrame) -> None:
    st.subheader("Real-Time Betting Edges")
    st.dataframe(edges, width="stretch")
