import pandas as pd

from core.streamlit_pipeline import _build_best_picks


def test_build_best_picks_selects_highest_ev_market_per_game_and_formats_pick():
    df = pd.DataFrame(
        [
            {
                "league": "NBA",
                "home_team": "Celtics",
                "away_team": "Warriors",
                "game_date": "2026-01-01",
                "market_type": "spread_home",
                "spread": -4.5,
                "expected_value": 0.04,
                "model_probability": 0.53,
                "decimal_odds": 1.91,
            },
            {
                "league": "NBA",
                "home_team": "Celtics",
                "away_team": "Warriors",
                "game_date": "2026-01-01",
                "market_type": "total_over",
                "total": 221.5,
                "expected_value": 0.11,
                "model_probability": 0.58,
                "decimal_odds": 1.95,
            },
            {
                "league": "NBA",
                "home_team": "Lakers",
                "away_team": "Suns",
                "game_date": "2026-01-02",
                "market_type": "moneyline_home",
                "expected_value": 0.09,
                "model_probability": 0.56,
                "decimal_odds": 1.87,
            },
        ]
    )

    out = _build_best_picks(df)

    assert list(out.columns) == [
        "league",
        "home_team",
        "away_team",
        "best_pick",
        "market_type",
        "expected_value",
        "model_probability",
        "decimal_odds",
    ]

    game_one = out[(out["home_team"] == "Celtics") & (out["away_team"] == "Warriors")].iloc[0]
    assert game_one["market_type"] == "total_over"
    assert game_one["best_pick"] == "Over 221.5"

    game_two = out[(out["home_team"] == "Lakers") & (out["away_team"] == "Suns")].iloc[0]
    assert game_two["best_pick"] == "Lakers ML"
