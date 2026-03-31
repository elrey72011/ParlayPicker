import pandas as pd
from core.streamlit_pipeline import run_analysis_pipeline
import logging
import os

os.environ["ODDS_API_KEY"] = "mock_key"
os.environ["CFBD_API_KEY"] = "mock_key"
import streamlit as st
import unittest.mock as mock

logging.basicConfig(level=logging.INFO)

# Make streamlit secrets silent
st.secrets = {}

# Generate some dummy data that models real problems
spreads_df = pd.DataFrame([
    {"league": "NBA", "home_team": "Boston Celtics", "away_team": "Los Angeles Lakers", "market_type": "spread_home", "spread_line": -5.5, "odds_american": -110, "theover_probability": 0.55},
    {"league": "NBA", "home_team": "Miami Heat", "away_team": "Orlando Magic", "market_type": "spread_home", "spread_line": -3.5, "odds_american": -110, "theover_probability": 0.52},
    {"league": "NBA", "home_team": "Chicago Bulls", "away_team": "Detroit Pistons", "market_type": "spread_home", "spread_line": -1.5, "odds_american": -110, "theover_probability": 0.51},
])

totals_df = pd.DataFrame([
    {"league": "NBA", "home_team": "Boston Celtics", "away_team": "Los Angeles Lakers", "market_type": "total_over", "total_line": 220.5, "odds_american": -110, "theover_probability": 0.55},
])

import app_core.odds_api
app_core.odds_api.TheOddsAPIClient.get_odds = lambda self, *args, **kwargs: []

analysis_df, best_picks_df, diagnostics = run_analysis_pipeline(
    sports=["NBA"], max_rows=10, use_ml=True, spreads_df=spreads_df, totals_df=totals_df
)

print(analysis_df[['home_team', 'away_team', 'market_probability', 'ml_probability']])
