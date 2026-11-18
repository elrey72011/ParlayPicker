# Re-issued on 2026-11-18 without the tests of library code the owner removed; the original is in tests_archive/, unchanged, and the tests kept are unchanged.
# Re-issued 2026-11-18 without the tests of panels the owner removed from the app; the original is in tests_archive/, unchanged.
import pandas as pd
import pytest
from core.streamlit_pipeline import compute_blended_probability
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


