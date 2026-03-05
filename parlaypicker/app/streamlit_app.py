from __future__ import annotations

from pathlib import Path

import streamlit as st
import polars as pl

try:
    from parlaypicker.models.model_loader import load_model
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[2]))
    from parlaypicker.models.model_loader import load_model
from parlaypicker.data_pipeline.sports_data_pipeline import run_pipeline
from parlaypicker.app.dashboard_components import render_top_bets


LATEST_ODDS_PATH = Path("data/parquet/latest_odds.parquet")


@st.cache_data(ttl=300)
def load_data(sport: str, date_key: str) -> pl.DataFrame:
    if LATEST_ODDS_PATH.exists():
        return pl.read_parquet(LATEST_ODDS_PATH)
    return run_pipeline(sport, date_key)


@st.cache_resource
def load_prediction_model(path: str):
    return load_model(path)


def main():
    st.title("ParlayPicker")
    sport = st.selectbox("Sport", ["NBA", "NFL", "NHL", "NCAAB", "NCAAF"])
    date_key = st.text_input("Date", "2026-01-01")
    bankroll = st.number_input("Bankroll", min_value=100.0, value=1000.0, step=50.0)
    df = load_data(sport, date_key)
    render_top_bets(df, bankroll=float(bankroll))


if __name__ == "__main__":
    main()
