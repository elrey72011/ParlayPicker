# Re-issued on 2026-11-18 without the tests of library code the owner removed; the original is in tests_archive/, unchanged, and the tests kept are unchanged.
import pandas as pd
from core.streamlit_pipeline import compute_blended_probability
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

    # NBA totals intentionally use different weights than sides, so the two
    # numeric blends need not be identical. Both must stay above neutral: an
    # accidental away/under inversion would turn the 0.60 input into 0.40.
    assert blended[0] > 0.5, f"Expected spread blend > 0.5, got {blended[0]}"
    assert blended[1] > 0.5, f"Expected total blend > 0.5, got {blended[1]}"


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
