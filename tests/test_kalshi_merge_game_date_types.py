import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app import _merge_kalshi_into_analysis


def test_merge_kalshi_into_analysis_coerces_game_date_types():
    analysis_df = pd.DataFrame(
        {
            "league": ["NCAAB"],
            "home_team": ["Duke"],
            "away_team": ["UNC"],
            "game_date": [pd.Timestamp("2026-03-07T00:00:00Z")],
            "best_pick": ["Over 150.5"],
        }
    )
    best_picks_df = pd.DataFrame(
        {
            "league": ["NCAAB"],
            "home_team": ["Duke"],
            "away_team": ["UNC"],
            "game_date": ["2026-03-07"],
            "best_pick": ["Over 150.5"],
            "kalshi_match_status": ["matched"],
            "kalshi_probability": [0.57],
        }
    )

    merged = _merge_kalshi_into_analysis(analysis_df, best_picks_df)

    assert merged.loc[0, "kalshi_match_status"] == "matched"
    assert merged.loc[0, "kalshi_probability"] == 0.57

