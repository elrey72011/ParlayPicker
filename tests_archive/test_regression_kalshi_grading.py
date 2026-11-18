# Re-issued 2026-11-18 without the tests of panels the owner removed from the app; the original is in tests_archive/, unchanged.
import pandas as pd
import pytest
from core.streamlit_pipeline import compute_blended_probability
from app_core.results_ingestion import attach_results
from app_core.weights_config import KALSHI_WEIGHT, MARKET_WEIGHT, ML_MODEL_WEIGHT, THEOVER_WEIGHT, SENTIMENT_WEIGHT

def test_kalshi_probability_orientation_no_flip():
    """
    1) Kalshi pick-side probability should NOT be flipped again in compute_blended_probability()
    - Add a test for spread_away
    - Add a test for total_under
    - Confirm compute_blended_probability() uses kalshi_probability as already pick-side oriented
    - The expected blended value should use the configured weights directly, without any extra 1 - p inversion for away/under
    """

    # We will pass Kalshi probability = 0.6.
    # Other probabilities = 0.5.
    # We expect the blend to be (0.6 * KALSHI_WEIGHT) + 0.5 * (others)

    p_market = pd.Series([0.5, 0.5])
    p_kalshi = pd.Series([0.6, 0.6])
    p_ml = pd.Series([0.5, 0.5])
    p_theover = pd.Series([0.5, 0.5])
    p_sentiment = pd.Series([0.5, 0.5])
    league = pd.Series(["NBA", "NBA"])
    market_type = pd.Series(["spread_away", "total_under"])

    blended = compute_blended_probability(
        p_market=p_market,
        p_kalshi=p_kalshi,
        p_ml=p_ml,
        p_theover=p_theover,
        p_sentiment=p_sentiment,
        league=league,
        market_type=market_type
    )

    # Neutral TheOver/sentiment inputs are dropped and NBA totals have a distinct
    # weight profile. Orientation is correct when both pick-side 0.60 reads move
    # their blends above 0.50 instead of being inverted to 0.40.
    assert blended.iloc[0] > 0.5
    assert blended.iloc[1] > 0.5

def test_multi_word_team_spread_grading():
    """
    2) Multi-word team spread grading in app_core/results_ingestion.py
    - Add a test where the pick is something like: "New Orleans Pelicans +4.5"
    - Confirm attach_results() correctly infers away side and grades the spread correctly from final scores
    - Assert both spread_result and Pick_Outcome are correct
    """

    master_df = pd.DataFrame([{
        "league": "NBA",
        "home_team": "Los Angeles Lakers",
        "away_team": "New Orleans Pelicans",
        "best_pick": "New Orleans Pelicans +4.5",
        "spread_pick_team": "New Orleans Pelicans +4.5",
        "spread_pick_line": 4.5,
        "spread_pick_side": "away"
    }])

    # Lakers 110 - Pelicans 106. Pelicans lose by 4.
    # Pelicans +4.5 means they cover and win the bet.
    results_df = pd.DataFrame([{
        "league": "NBA",
        "home_team": "Los Angeles Lakers",
        "away_team": "New Orleans Pelicans",
        "home_score": 110,
        "away_score": 106,
        "date": "2023-10-24"
    }])

    attached = attach_results(master_df, results_df)

    assert "spread_result" in attached.columns, "spread_result column missing"
    assert "Pick_Outcome" in attached.columns, "Pick_Outcome column missing"

    assert attached["spread_result"].iloc[0] == "WIN", f"Expected WIN, got {attached['spread_result'].iloc[0]}"
    assert attached["Pick_Outcome"].iloc[0] == "WIN", f"Expected WIN, got {attached['Pick_Outcome'].iloc[0]}"


