import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from core.streamlit_pipeline import build_theover_bet_rows


EXPECTED_COLUMNS = [
    "league",
    "home_team",
    "away_team",
    "game_date",
    "market_type",
    "spread_line",
    "total_line",
    "theover_probability",
    "odds_american",
    "market_probability",
    "expected_value",
    "edge",
    "best_pick",
]


def test_build_theover_bet_rows_missing_odds_defaults_to_pd_na():
    totals_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "pick": ["Over"],
            "line": [221.5],
            "winprobability": [0.57],
        }
    )

    out = build_theover_bet_rows(None, totals_df, ["NBA"])

    assert not out.empty
    assert pd.isna(out.loc[0, "odds_american"])


def test_build_theover_bet_rows_missing_probability_column_does_not_crash():
    totals_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "pick": ["Under"],
            "line": [219.5],
        }
    )

    out = build_theover_bet_rows(None, totals_df, ["NBA"])

    assert not out.empty
    assert "theover_probability" in out.columns
    assert pd.isna(out.loc[0, "theover_probability"])


def test_build_theover_bet_rows_only_totals_upload():
    totals_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "selection": ["Over"],
            "points": [220.5],
            "probability": [58],
            "american_odds": [-105],
        }
    )

    out = build_theover_bet_rows(None, totals_df, ["NBA"])

    assert len(out) == 2
    assert set(out["market_type"].tolist()) == {"total_over", "total_under"}
    assert out["total_line"].iloc[0] == 220.5
    assert out["odds_american"].iloc[0] == -105


def test_build_theover_bet_rows_only_spreads_upload():
    spreads_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "pick_team": ["Boston Celtics"],
            "spread_line": [-3.5],
            "win_probability": [0.54],
        }
    )

    out = build_theover_bet_rows(spreads_df, None, ["NBA"])

    assert len(out) == 2
    assert set(out["market_type"].tolist()) == {"spread_home", "spread_away"}
    assert sorted(out["spread_line"].tolist()) == [-3.5, 3.5]


def test_build_theover_bet_rows_empty_uploads_returns_stable_schema():
    out = build_theover_bet_rows(pd.DataFrame(), pd.DataFrame(), ["NBA"])

    assert out.empty
    for col in EXPECTED_COLUMNS:
        assert col in out.columns
