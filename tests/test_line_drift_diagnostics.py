import pandas as pd
from core.streamlit_pipeline import _expand_live_odds_to_bet_rows
import app_core.weights_config as config

def test_upload_matched_drift_diagnostics(monkeypatch):
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
            "total_line": 214.5, # Delta of 1.0
        },
        {
            "league": "NBA",
            "home_team": "Lakers",
            "away_team": "Bulls",
            "game_date": "2023-10-25",
            "matchup_id": "nba_lakers_bulls",
            "market_type": "spread_home",
            "spread_line": -4.5, # Delta of -1.0
        }
    ])

    expanded, diag = _expand_live_odds_to_bet_rows(live_odds_df, theover_rows)

    assert diag["upload_matched_rows"] == 2
    assert diag["upload_matched_drifted_rows"] == 2
    assert diag["drift_max"] == 1.0
    assert diag["drift_mean"] == 1.0
    assert diag["drift_breakdown"]["total_over"] == 1
    assert diag["drift_breakdown"]["spread_home"] == 1
