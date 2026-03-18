import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core import kalshi_integrator as ki
from streamlit_app import _merge_kalshi_into_analysis
from core.streamlit_pipeline import build_best_picks_df


def test_clean_team_name_strips_noise_and_mascots():
    cleaned = ki.clean_team_name("Wichita St Shockers University")
    assert cleaned == "wichita"


def test_market_type_matches_with_keyword_inference():
    assert ki.market_type_matches("", "Yale wins by over 6.5 points", "")
    assert ki.market_type_matches("", "Over 154.5 points scored", "")
    assert ki.market_type_matches("spread_home", "Yale wins by over 6.5 points", "")
    assert ki.market_type_matches("total_over", "Over 154.5 points scored", "")


def test_merge_kalshi_allows_plus_minus_one_day_date_drift():
    analysis_df = pd.DataFrame(
        {
            "league": ["NCAAB"],
            "home_team": ["Duke"],
            "away_team": ["UNC"],
            "game_date": ["2026-03-07T23:30:00-05:00"],
        }
    )
    # Stored on next UTC day in best picks source
    best_picks_df = pd.DataFrame(
        {
            "league": ["NCAAB"],
            "home_team": ["Duke"],
            "away_team": ["UNC"],
            "game_date": ["2026-03-08T04:30:00Z"],
            "kalshi_match_status": ["matched"],
            "kalshi_probability": [0.61],
        }
    )

    merged = _merge_kalshi_into_analysis(analysis_df, best_picks_df)

    assert merged.loc[0, "kalshi_match_status"] == "matched"
    assert merged.loc[0, "kalshi_probability"] == 0.61


def test_best_picks_prefers_highest_expected_value_for_kalshi_spread_total_pair():
    analysis_df = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "league": ["NBA", "NBA"],
            "home_team": ["Boston Celtics", "Boston Celtics"],
            "away_team": ["Los Angeles Lakers", "Los Angeles Lakers"],
            "game_date": ["2026-03-10T00:00:00Z", "2026-03-10T00:00:00Z"],
            "market_type": ["spread_home", "total_over"],
            "spread_line": [-4.5, pd.NA],
            "total_line": [pd.NA, 229.5],
            "calibrated_probability": [0.54, 0.55],
            "expected_value": [0.03, -0.08],
            "edge": [0.02, 0.04],
            "kalshi_match_status": ["matched", "matched"],
            "odds_american": [-110, -110],
            "odds_source": ["test", "test"],
            "market_probability": [0.5, 0.5],
            "ml_probability": [0.55, 0.55],
            "consensus_agreement": ["⚖️ Neutral", "⚖️ Neutral"],
            "used_stale_features": [False, False],
        }
    )

    best = build_best_picks_df(analysis_df)

    assert len(best) == 1
    assert best.loc[0, "market_type"] == "spread_home"
    assert float(best.loc[0, "expected_value"]) == 0.03
