import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import generate_parlays


def test_generate_parlays_advances_past_duplicate_game_slices():
    # First two rows are the same game, which previously could trap ranked parlay loop.
    df = pd.DataFrame(
        {
            "league": ["NBA", "NBA", "NBA"],
            "home_team": ["Lakers", "Lakers", "Bulls"],
            "away_team": ["Celtics", "Celtics", "Heat"],
            "best_pick": ["Over 220.5", "Lakers -3.5", "Under 210.5"],
            "calibrated_probability": [0.60, 0.58, 0.57],
            "expected_value": [0.1, 0.08, 0.06],
            "decimal_odds": [1.91, 1.91, 1.91],
            "odds_american": [-110, -110, -110],
        }
    )

    out = generate_parlays(df)

    assert isinstance(out, pd.DataFrame)
    # Should return promptly and be able to form at least one valid 2-leg parlay.
    assert not out.empty
    assert out["legs"].astype(int).ge(2).any()
