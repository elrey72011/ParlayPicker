import pandas as pd

from core.parlay_engine import generate_parlays
from shotgun_mode import is_correlated


def test_fallback_parlays_reject_duplicate_game_ids():
    frame = pd.DataFrame(
        [
            {
                "game_id": "same",
                "away_team": "Alias A",
                "home_team": "Alias B",
                "best_pick": "Same spread",
                "expected_value": 0.10,
            },
            {
                "game_id": "same",
                "away_team": "Different A",
                "home_team": "Different B",
                "best_pick": "Same total",
                "expected_value": 0.09,
            },
            {
                "game_id": "other",
                "away_team": "C",
                "home_team": "D",
                "best_pick": "Other spread",
                "expected_value": 0.08,
            },
        ]
    )

    out = generate_parlays(frame)

    assert len(out) == 2
    assert out["one_leg_per_game"].eq(True).all()
    assert out["unique_game_count"].eq(out["legs"]).all()
    assert not out["parlay_legs"].str.contains(
        r"Same spread.*Same total|Same total.*Same spread", regex=True
    ).any()


def test_shotgun_correlation_uses_shared_game_identity():
    assert is_correlated(
        {"Game": "event-1", "Home": "A", "Away": "B"},
        {"game_id": "event-1", "Home": "Different", "Away": "Aliases"},
    )
    assert not is_correlated(
        {"Game": "event-1", "Home": "A", "Away": "B"},
        {"Game": "event-2", "Home": "C", "Away": "D"},
    )
