import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _preprocess_bet_rows_for_league_bridge
from app_core import kalshi_integrator as ki


def test_preprocess_restores_ncaab_for_blank_league_with_college_keywords():
    df = pd.DataFrame(
        {
            "league": [pd.NA, "", "NBA"],
            "home_team": ["Oklahoma St Cowboys", "St Thomas Tommies", "Boston Celtics"],
            "away_team": ["Wichita State", "Gonzaga Bulldogs", "Miami Heat"],
        }
    )

    out = _preprocess_bet_rows_for_league_bridge(df)

    assert out.loc[0, "league"] == "ncaab"
    assert out.loc[1, "league"] == "ncaab"
    assert out.loc[2, "league"] == "nba"


def test_ncaab_and_nba_use_wider_line_tolerance_while_nhl_is_one_point():
    assert ki.MAX_LINE_TOLERANCE["NCAAB"] == 1.5
    assert ki.MAX_LINE_TOLERANCE["NBA"] == 1.5
    assert ki.MAX_LINE_TOLERANCE["NHL"] == 0.5

