import pandas as pd
from core.streamlit_pipeline import compute_blended_probability
from app.ui.results_dashboard import render_results_dashboard
from app_core.results_ingestion import attach_results
import streamlit as st

def test_kalshi_probability_orientation_no_flip():
    """
    Test that compute_blended_probability DOES NOT flip the Kalshi
    probability. Kalshi probability should be treated as pick-side oriented.
    """
    p_market = pd.Series([0.5, 0.5])
    p_kalshi = pd.Series([0.6, 0.6]) # Kalshi probability for the pick
    p_ml = pd.Series([0.5, 0.5])
    p_theover = pd.Series([0.5, 0.5])
    p_sentiment = pd.Series([0.5, 0.5])
    league = pd.Series(["NBA", "NBA"])

    # One home/over market, one away/under market
    market_type = pd.Series(["spread_home", "total_under"])

    blended = compute_blended_probability(
        p_market=p_market,
        p_kalshi=p_kalshi,
        p_ml=p_ml,
        p_theover=p_theover,
        p_sentiment=p_sentiment,
        league=league,
        market_type=market_type
    )

    # Since p_kalshi=0.6 for both, and other probabilities are 0.5,
    # the blended probability should be identical for both rows and > 0.5.
    assert blended[0] == blended[1], f"Expected equal blend for both orientations, got {blended[0]} vs {blended[1]}"
    assert blended[0] > 0.5, f"Expected blend > 0.5 due to kalshi_prob=0.6, got {blended[0]}"

def test_multi_word_team_spread_grading():
    """
    Test that multi-word team names are parsed correctly for spread picks.
    """
    master_df = pd.DataFrame([{
        "league": "NBA",
        "home_team": "Los Angeles Lakers",
        "away_team": "Golden State Warriors",
        "spread_pick_line": -5.5,
        "spread_pick_team": "Los Angeles Lakers -5.5"
    }])

    results_df = pd.DataFrame([{
        "league": "NBA",
        "home_team": "Los Angeles Lakers",
        "away_team": "Golden State Warriors",
        "home_score": 110,
        "away_score": 100,
        "date": "2023-10-24" # Dummy date to satisfy column requirements
    }])

    # With a spread of -5.5, Lakers win by 10 (110-100), so the pick covers.
    attached = attach_results(master_df, results_df)

    assert "spread_result" in attached.columns
    assert attached["spread_result"].iloc[0] == "WIN", f"Expected spread_result='WIN', got {attached['spread_result'].iloc[0]}"

def test_multi_word_team_spread_grading_away():
    """
    Test that multi-word team names are parsed correctly for away spread picks.
    """
    master_df = pd.DataFrame([{
        "league": "NBA",
        "home_team": "New York Knicks",
        "away_team": "New Orleans Pelicans",
        "spread_pick_line": +2.5,
        "spread_pick_team": "New Orleans Pelicans +2.5"
    }])

    results_df = pd.DataFrame([{
        "league": "NBA",
        "home_team": "New York Knicks",
        "away_team": "New Orleans Pelicans",
        "home_score": 105,
        "away_score": 104,
        "date": "2023-10-24"
    }])

    # With a spread of +2.5, Pelicans lose by 1 (105-104), so the pick covers.
    attached = attach_results(master_df, results_df)

    assert "spread_result" in attached.columns
    assert attached["spread_result"].iloc[0] == "WIN", f"Expected spread_result='WIN', got {attached['spread_result'].iloc[0]}"

class MockUploadedFile:
    def __init__(self, name, size, content):
        self.name = name
        self.size = size
        self.content = content
        self._pos = 0

    def seek(self, pos):
        self._pos = pos

    def read(self):
        return self.content

    def __iter__(self):
        yield self.content.encode('utf-8')

# The session state test is slightly tricky due to Streamlit context,
# but we can test the `perf_picks_file_identifier` logic through a mock.
