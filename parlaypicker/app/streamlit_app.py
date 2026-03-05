from __future__ import annotations

import streamlit as st
import pandas as pd

try:
    from parlaypicker.models.model_loader import load_model
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from parlaypicker.models.model_loader import load_model
from parlaypicker.data_pipeline.sports_data_pipeline import run_pipeline
from parlaypicker.app.dashboard_components import render_top_bets


@st.cache_data(ttl=300)
def load_data(sport: str, date_key: str) -> pd.DataFrame:
    return run_pipeline(sport, date_key)


@st.cache_resource
def load_prediction_model(path: str):
    return load_model(path)


def main():
    st.title("ParlayPicker")
    sport = st.selectbox("Sport", ["NBA", "NFL", "NHL", "NCAAB", "NCAAF"])
    date_key = st.text_input("Date", "2026-01-01")
    df = load_data(sport, date_key)
    render_top_bets(df)


if __name__ == "__main__":
    main()
