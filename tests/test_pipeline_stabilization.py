import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from core.probability_engine import american_odds_to_prob
from core.streamlit_pipeline import _apply_analysis_calculations, generate_parlays_table
from core.team_normalizer import normalize_team


def test_team_normalizer_maps_common_abbreviations():
    assert normalize_team("LA Clippers") == "Los Angeles Clippers"
    assert normalize_team("Boston Celtics") == "Boston Celtics"


def test_american_odds_to_prob_positive_and_negative():
    assert round(american_odds_to_prob(150), 4) == 0.4
    assert round(american_odds_to_prob(-110), 4) == 0.5238


def test_analysis_calculations_create_consensus_ev_and_best_pick():
    df = pd.DataFrame(
        {
            "away_team": ["LA Clippers"],
            "home_team": ["NY Knicks"],
            "ai_probability": [0.60],
            "ml_probability": [0.54],
            "odds_american": [-110],
            "spread_edge": [0.05],
            "total_edge": [0.01],
        }
    )

    out = _apply_analysis_calculations(df)

    assert "market_probability" in out.columns
    assert "consensus_prob" in out.columns
    assert "expected_value" in out.columns
    assert out.loc[0, "best_pick"] == "LA Clippers spread"
    assert out.loc[0, "consensus_prob"] > 0.55


def test_parlay_generation_uses_ev_threshold_and_sorts_desc():
    df = pd.DataFrame(
        {
            "away_team": ["A", "B", "C"],
            "expected_value": [0.015, 0.08, 0.03],
        }
    )
    parlays = generate_parlays_table(df)
    assert list(parlays["away_team"]) == ["B", "C"]
