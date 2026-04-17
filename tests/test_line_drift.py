import pandas as pd
import pytest
import numpy as np
from core.streamlit_pipeline import _expand_live_odds_to_bet_rows
import app_core.weights_config as config

def test_upload_matched_preserves_lines_and_delta_calculated(monkeypatch):
    monkeypatch.setattr(config, "LOCK_UPLOAD_LINES_FOR_MATCHED_ROWS", False)

    live_odds_df = pd.DataFrame([
        {
            "league": "NBA",
            "home_team": "Lakers",
            "away_team": "Bulls",
            "game_date": "2023-10-25",
            "matchup_id": "nba_lakers_bulls",
            "commence_time_raw": "2023-10-25T00:00:00Z",
            "novig_over_point": 215.5,
            "novig_under_point": 215.5,
            "novig_home_point": -5.5,
            "novig_away_point": 5.5,
        }
    ])

    theover_rows = pd.DataFrame([
        {
            "league": "NBA",
            "home_team": "Lakers",
            "away_team": "Bulls",
            "game_date": "2023-10-25",
            "matchup_id": "nba_lakers_bulls",
            "market_type": "total_over",
            "total_line": 214.5,
        },
        {
            "league": "NBA",
            "home_team": "Lakers",
            "away_team": "Bulls",
            "game_date": "2023-10-25",
            "matchup_id": "nba_lakers_bulls",
            "market_type": "spread_home",
            "spread_line": -4.5,
        }
    ])

    expanded, diag = _expand_live_odds_to_bet_rows(live_odds_df, theover_rows)

    over_row = expanded[expanded["market_type"] == "total_over"].iloc[0]
    assert float(over_row["uploaded_total_line"]) == 214.5
    assert float(over_row["live_total_line"]) == 215.5
    assert float(over_row["line_delta"]) == 1.0 # 215.5 - 214.5
    assert over_row["line_source"] == "live_odds"
    assert float(over_row["total_line"]) == 215.5

    spread_row = expanded[expanded["market_type"] == "spread_home"].iloc[0]
    assert float(spread_row["uploaded_spread_line"]) == -4.5
    assert float(spread_row["live_spread_line"]) == -5.5
    assert float(spread_row["line_delta"]) == -1.0
    assert spread_row["line_source"] == "live_odds"
    assert float(spread_row["spread_line"]) == -5.5

def test_lock_upload_lines(monkeypatch):
    monkeypatch.setattr(config, "LOCK_UPLOAD_LINES_FOR_MATCHED_ROWS", True)

    live_odds_df = pd.DataFrame([
        {
            "league": "NBA",
            "home_team": "Lakers",
            "away_team": "Bulls",
            "game_date": "2023-10-25",
            "matchup_id": "nba_lakers_bulls",
            "commence_time_raw": "2023-10-25T00:00:00Z",
            "novig_over_point": 215.5,
            "novig_under_point": 215.5,
        }
    ])

    theover_rows = pd.DataFrame([
        {
            "league": "NBA",
            "home_team": "Lakers",
            "away_team": "Bulls",
            "game_date": "2023-10-25",
            "matchup_id": "nba_lakers_bulls",
            "market_type": "total_over",
            "total_line": 214.5,
        }
    ])

    expanded, diag = _expand_live_odds_to_bet_rows(live_odds_df, theover_rows)
    over_row = expanded[expanded["market_type"] == "total_over"].iloc[0]

    assert float(over_row["uploaded_total_line"]) == 214.5
    assert float(over_row["live_total_line"]) == 215.5
    assert float(over_row["total_line"]) == 214.5
    assert over_row["line_source"] == "uploaded_theover"
