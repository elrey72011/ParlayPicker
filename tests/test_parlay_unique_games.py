import pandas as pd

from core.parlay_engine import generate_parlays
from core.smart_parlay_engine import generate_probability_ranked_parlays
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


def _best_available_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "league": ["MLB"] * 4,
            "matchup_id": ["g1", "g2", "g3", "g4"],
            "away_team": ["A", "C", "E", "G"],
            "home_team": ["B", "D", "F", "H"],
            "best_pick": ["A +1.5", "C +1.5", "E +1.5", "G +1.5"],
            "Pick_Status": ["Best Available / Pass"] * 4,
            "effective_win_probability": [0.70, 0.68, 0.65, 0.62],
            "market_probability": [0.52, 0.52, 0.52, 0.52],
            "decimal_odds": [1.90, 1.91, 1.92, 1.93],
            "odds_source": ["novig"] * 4,
            "final_pick_valid": [True] * 4,
            "best_available_selection_verified": [True] * 4,
            "best_available_ranking_verified": [True] * 4,
            "line_consistency_flag": [True] * 4,
            "line_event_identity_match_flag": [True] * 4,
            "market_line_source": ["live"] * 4,
        }
    )


def test_probability_fallback_populates_and_sorts_high_to_low():
    out = generate_probability_ranked_parlays(
        _best_available_frame(), max_parlays=6
    )

    assert len(out) == 6
    assert out["combined_probability"].is_monotonic_decreasing
    assert "A @ B: A +1.5" in out.iloc[0]["parlay_legs"]
    assert "C @ D: C +1.5" in out.iloc[0]["parlay_legs"]
    assert out["one_leg_per_game"].eq(True).all()
    assert out["unique_game_count"].eq(out["legs"]).all()
    assert out["parlay_class"].eq("Research / Recreational").all()
    assert out["recommended_bet"].eq(0.0).all()
    assert out["kelly_fraction"].eq(0.0).all()


def test_probability_fallback_rejects_invalid_and_duplicate_games():
    frame = _best_available_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["best_pick"] = "A total"
    duplicate["effective_win_probability"] = 0.71
    frame = pd.concat([frame, duplicate], ignore_index=True)
    frame.loc[frame["matchup_id"].eq("g4"), "line_consistency_flag"] = False

    out = generate_probability_ranked_parlays(frame, max_parlays=20)

    assert not out["parlay_legs"].str.contains(
        r"A \+1\.5.*A total|A total.*A \+1\.5", regex=True
    ).any()
    assert not out["parlay_legs"].str.contains("G @ H", regex=False).any()
