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
            "calibrated_probability": [0.60, 0.58, 0.59],
            "expected_value": [0.1, 0.08, 0.06],
            "edge": [0.08, 0.06, 0.05],
            "decimal_odds": [1.91, 1.91, 1.91],
            "odds_american": [-110, -110, -110],
        }
    )

    out = generate_parlays(df)

    assert isinstance(out, pd.DataFrame)
    # Should return promptly and be able to form at least one valid 2-leg parlay.
    assert not out.empty
    assert out["legs"].astype(int).ge(2).any()


def test_generate_parlays_populates_probability_ranked_research_fallback():
    df = pd.DataFrame(
        {
            "league": ["MLB", "MLB", "MLB"],
            "matchup_id": ["g1", "g2", "g3"],
            "home_team": ["B", "D", "F"],
            "away_team": ["A", "C", "E"],
            "best_pick": ["A +1.5", "C +1.5", "E +1.5"],
            "Pick_Status": ["Best Available / Pass"] * 3,
            "production_eligible": [False] * 3,
            "effective_win_probability": [0.66, 0.64, 0.62],
            "calibrated_probability": [0.66, 0.64, 0.62],
            "market_probability": [0.52, 0.52, 0.52],
            "decimal_odds": [1.90, 1.91, 1.92],
            "odds_american": [-111, -110, -109],
            "edge": [0.01, 0.00, -0.01],
            "market_line_source": ["live"] * 3,
            "final_pick_valid": [True] * 3,
            "best_available_selection_verified": [True] * 3,
            "best_available_ranking_verified": [True] * 3,
            "line_consistency_flag": [True] * 3,
            "line_event_identity_match_flag": [True] * 3,
        }
    )

    out = generate_parlays(df)

    assert len(out) == 3
    assert out["combined_probability"].is_monotonic_decreasing
    assert out["parlay_source"].eq("probability_ranked_fallback").all()
    assert out["recommended_bet"].eq(0.0).all()
    assert out["one_leg_per_game"].eq(True).all()
