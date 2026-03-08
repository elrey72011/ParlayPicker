import numpy as np
import pandas as pd

from core.ev_engine import calculate_ev
from core.parlay_engine import generate_parlays
from core.probability_engine import american_to_prob, remove_vig
from core.streamlit_pipeline import _apply_analysis_calculations


def test_remove_vig_normalizes():
    probs = remove_vig([0.55, 0.55])
    assert round(sum(probs), 6) == 1.0


def test_vectorized_ev():
    probs = np.array([0.55, 0.5])
    odds = np.array([-110, 120])
    ev = calculate_ev(probs, odds)
    assert ev.shape == (2,)


def test_parlay_correlation_filter():
    df = pd.DataFrame(
        [
            {"away_team": "A", "home_team": "B", "expected_value": 0.10},
            {"away_team": "A", "home_team": "C", "expected_value": 0.09},
            {"away_team": "D", "home_team": "E", "expected_value": 0.08},
        ]
    )
    out = generate_parlays(df)
    assert len(out) == 2


def test_pipeline_adds_best_pick_and_consensus():
    df = pd.DataFrame(
        [
            {
                "away_team": "LAL",
                "home_team": "BOS",
                "ai_probability": 0.56,
                "ml_probability": 0.55,
                "odds_american": -110,
                "kalshi_probability": 0.57,
            }
        ]
    )
    out = _apply_analysis_calculations(df)
    assert "calibrated_probability" in out.columns
    assert "expected_value" in out.columns
    assert "best_pick" in out.columns
    assert 0 < american_to_prob(-110) < 1
