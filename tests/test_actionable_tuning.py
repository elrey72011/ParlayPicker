import pytest
import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

def test_agrees_row_can_become_actionable():
    # An Agrees side row that meets baseline thresholds
    analysis_df = pd.DataFrame([{
        "market_type": "spread_home",
        "expected_value": 0.02, # meets baseline
        "edge": 0.03, # meets baseline
        "calibrated_probability": 0.52, # above side floor
        "kalshi_probability": 0.45, # kalshi lower, so gap is 0.52 - 0.45 = 0.07 >= 0.03 -> Agrees
        "league": "NBA",
        "home_team": "Team A",
        "away_team": "Team B",
        "best_pick": "Team A -5",
        "game_date": "2024-01-01",
        "model_status": "OK",
        "odds_source": "live",
        "is_live_data": True,
        "used_stale_features": False,
        "matchup_id": "game1",
        "spread_line": -5,
        "total_line": 220,
    }])
    best_df = build_best_picks_df(analysis_df)
    assert not best_df.empty
    row = best_df.iloc[0]
    assert row["consensus_agreement"] == "Agrees"
    assert row["Pick_Status"] == "Actionable"


def test_neutral_row_downgraded_if_misses_stricter_thresholds():
    # A Neutral side row that meets baseline but fails stricter Neutral thresholds (ev=0.03, edge=0.04, prob=0.58)
    analysis_df = pd.DataFrame([{
        "market_type": "spread_home",
        "expected_value": 0.02, # meets baseline (0.01), but fails neutral (0.03)
        "edge": 0.03, # meets baseline (0.02), but fails neutral (0.04)
        "calibrated_probability": 0.55, # fails neutral prob (0.58)
        "kalshi_probability": 0.54, # gap = 0.01 -> Neutral
        "league": "NBA",
        "home_team": "Team A",
        "away_team": "Team B",
        "best_pick": "Team A -5",
        "game_date": "2024-01-01",
        "model_status": "OK",
        "odds_source": "live",
        "is_live_data": True,
        "used_stale_features": False,
        "matchup_id": "game1",
        "spread_line": -5,
        "total_line": 220,
    }])
    best_df = build_best_picks_df(analysis_df)
    row = best_df.iloc[0]
    assert row["consensus_agreement"] == "Neutral"
    assert row["Pick_Status"] == "Below Threshold"
    assert "Neutral overlay" in row["Status_Reason"]


def test_disagrees_row_downgraded_to_high_variance():
    # A Disagrees side row that meets baseline and neutral, but fails strict Disagrees (ev=0.04, edge=0.05, prob=0.60)
    analysis_df = pd.DataFrame([{
        "market_type": "spread_home",
        "expected_value": 0.035, # fails disagree (0.04)
        "edge": 0.045, # fails disagree (0.05)
        "calibrated_probability": 0.59, # fails disagree prob (0.60)
        "kalshi_probability": 0.65, # gap = -0.06 -> Disagrees
        "league": "NBA",
        "home_team": "Team A",
        "away_team": "Team B",
        "best_pick": "Team A -5",
        "game_date": "2024-01-01",
        "model_status": "OK",
        "odds_source": "live",
        "is_live_data": True,
        "used_stale_features": False,
        "matchup_id": "game1",
        "spread_line": -5,
        "total_line": 220,
    }])
    best_df = build_best_picks_df(analysis_df)
    row = best_df.iloc[0]
    assert row["consensus_agreement"] == "Disagrees"
    assert row["Pick_Status"] == "High Variance/Speculative"
    assert "Disagrees overlay" in row["Status_Reason"]


def test_side_row_below_side_prob_floor_downgraded():
    # Side row that meets EV and Edge but fails SIDE_MIN_WIN_PROB (0.50)
    analysis_df = pd.DataFrame([{
        "market_type": "spread_home",
        "expected_value": 0.05,
        "edge": 0.06,
        "calibrated_probability": 0.48, # fails side min prob (0.50), even though it is >0.40 baseline
        "kalshi_probability": 0.40, # gap = 0.08 -> Agrees
        "league": "NBA",
        "home_team": "Team A",
        "away_team": "Team B",
        "best_pick": "Team A -5",
        "game_date": "2024-01-01",
        "model_status": "OK",
        "odds_source": "live",
        "is_live_data": True,
        "used_stale_features": False,
        "matchup_id": "game1",
        "spread_line": -5,
        "total_line": 220,
    }])
    best_df = build_best_picks_df(analysis_df)
    row = best_df.iloc[0]
    assert row["Pick_Status"] == "Below Threshold"
    assert "Side Win Probability" in row["Status_Reason"]


def test_totals_rules_unaffected():
    # Total over must still clear 0.03 / 0.04 and its own prob floor.
    # We will give it Neutral consensus to ensure it overlays cleanly.
    analysis_df = pd.DataFrame([{
        "market_type": "total_over",
        "expected_value": 0.035, # passes over rules, passes Neutral rules
        "edge": 0.045, # passes over rules, passes Neutral rules
        "calibrated_probability": 0.59, # passes total prob (0.56) and neutral prob (0.58)
        "kalshi_probability": 0.58, # Neutral
        "league": "NBA",
        "home_team": "Team A",
        "away_team": "Team B",
        "best_pick": "Over 220",
        "game_date": "2024-01-01",
        "model_status": "OK",
        "odds_source": "live",
        "is_live_data": True,
        "used_stale_features": False,
        "matchup_id": "game1",
        "spread_line": -5,
        "total_line": 220,
    }])
    best_df = build_best_picks_df(analysis_df)
    row = best_df.iloc[0]
    assert row["consensus_agreement"] == "Neutral"
    assert row["Pick_Status"] == "Actionable"
