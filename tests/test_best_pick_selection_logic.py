import pandas as pd

from core.streamlit_pipeline import build_best_picks_df


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
                "total_line": 221.5,
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

    out = build_best_picks_df(df)

    assert list(out.columns) == [
        "parlay_rank",
        "league",
        "home_team",
        "away_team",
        "game_date",
        "game_time_est",
        "market_type",
        "best_pick",
        "calibrated_probability",
        "expected_value",
        "edge",
        "consensus_agreement",
        "odds_american",
        "market_probability",
        "ml_probability",
        "kalshi_probability",
        "kalshi_match_status",
        "kalshi_match_reason",
    ]

    game_one = out[(out["home_team"] == "Celtics") & (out["away_team"] == "Warriors")].iloc[0]
    assert game_one["best_pick"] == "Over 221.5"

    assert out[(out["home_team"] == "Lakers") & (out["away_team"] == "Suns")].empty


def test_consensus_uses_edge_threshold_not_raw_probability():
    df = pd.DataFrame([
        {
            "league": "NBA",
            "home_team": "A",
            "away_team": "B",
            "game_date": "2026-01-01",
            "market_type": "total_over",
            "total_line": 220.5,
            "expected_value": 0.02,
            "edge": 0.01,
            "calibrated_probability": 0.90,
        }
    ])
    out = build_best_picks_df(df)
    assert out.loc[0, "consensus_agreement"] == "⚖️ Neutral"
