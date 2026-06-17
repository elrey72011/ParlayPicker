"""Corrupt-odds guard: a totals/spread market with implied prob outside [0.05, 0.95]
is bad feed data (17 Jun: Dodgers Over 9.5 came in at +1983 / 4.8% implied, +850% EV).
Block it as No Play before the garbage EV reaches the card."""
import pandas as pd

from core.streamlit_pipeline import build_best_picks_df


def _row(**over):
    base = {
        "league": "MLB",
        "home_team": "Los Angeles Dodgers",
        "away_team": "Tampa Bay",
        "game_date": "2026-06-17",
        "matchup_id": "2026-06-17|Los Angeles Dodgers|Tampa Bay",
        "market_type": "total_over",
        "expected_value": 8.5,
        "edge": 0.4,
        "calibrated_probability": 0.46,
        "ml_probability": 0.6,
        "model_probability": 0.6,
        "odds_american": 1983,
        "market_probability": 0.048,
        "total_line": 9.5,
        "spread_line": pd.NA,
        "line_source": "live",
        "live_total_line": 9.5,
        "is_live_data": True,
        "used_stale_features": False,
        "odds_source": "odds_api",
        "kalshi_probability": 0.295,
    }
    base.update(over)
    return base


def test_corrupt_odds_total_is_no_play():
    best = build_best_picks_df(pd.DataFrame([_row()]))
    row = best.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert "corrupt odds" in str(row["Status_Reason"]).lower()


def test_sane_odds_total_is_not_corrupt_flagged():
    best = build_best_picks_df(pd.DataFrame([_row(
        odds_american=-110, market_probability=0.476,
        expected_value=0.02, edge=0.02, calibrated_probability=0.50,
    )]))
    assert "corrupt odds" not in str(best.iloc[0]["Status_Reason"]).lower()
