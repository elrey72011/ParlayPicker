import pandas as pd
from app.ui.results_dashboard import render_results_dashboard

# Ensure session state allows API fetching without restriction
import streamlit as st

def main():
    st.set_page_config(page_title="Test Dashboard", layout="wide")

    # Mock some basic data
    mock_data = pd.DataFrame({
        "league": ["NBA", "NBA"],
        "home_team": ["Lakers", "Celtics"],
        "away_team": ["Warriors", "Heat"],
        "best_pick": ["Lakers ML", "Heat +5.5"],
        "Pick_Status": ["Actionable", "Actionable"],
        "odds_american": [-110, -110],
    })

    # Simulate session state
    if "perf_edited_picks" not in st.session_state:
        st.session_state["perf_edited_picks"] = mock_data
    if "restricted_leagues" not in st.session_state:
        st.session_state["restricted_leagues"] = set()

    render_results_dashboard(mock_data)

if __name__ == "__main__":
    main()
