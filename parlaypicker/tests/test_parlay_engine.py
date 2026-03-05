import pandas as pd
from parlaypicker.core.parlay_engine import parlay_probability, build_parlays


def test_parlay_probability():
    assert parlay_probability([0.5, 0.5]) == 0.25


def test_build_parlays():
    df = pd.DataFrame([
        {"game_id": "1", "odds": -110, "true_probability": 0.55},
        {"game_id": "2", "odds": -105, "true_probability": 0.52},
    ])
    out = build_parlays(df, leg_count=2)
    assert len(out) == 1
