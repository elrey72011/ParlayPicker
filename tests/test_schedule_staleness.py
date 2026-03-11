import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_is_stale_schedule_handles_naive_base_and_aware_now():
    base_df = pd.DataFrame({"game_date": ["2026-01-05 19:00:00", "2026-01-06 20:00:00"]})
    bet_rows_df = pd.DataFrame()

    stale = sp.is_stale_schedule(base_df, bet_rows_df)

    assert stale is False


def test_is_stale_schedule_handles_string_dates_without_crash():
    base_df = pd.DataFrame({"game_date": ["2023-01-01", "2023-01-02"]})
    bet_rows_df = pd.DataFrame()

    stale = sp.is_stale_schedule(base_df, bet_rows_df)

    assert stale is False


def test_is_stale_schedule_empty_base_with_populated_bets_is_stale():
    base_df = pd.DataFrame(columns=["game_date"])
    bet_rows_df = pd.DataFrame({"game_date": ["2026-02-01T00:00:00Z"]})

    stale = sp.is_stale_schedule(base_df, bet_rows_df)

    assert stale is False


def test_is_stale_schedule_mixed_invalid_dates_does_not_crash():
    base_df = pd.DataFrame({"game_date": ["not-a-date", None, "2026-03-05"]})
    bet_rows_df = pd.DataFrame({"game_date": ["2026-03-06"]})

    stale = sp.is_stale_schedule(base_df, bet_rows_df)

    assert isinstance(stale, bool)


def test_run_analysis_pipeline_stale_base_falls_back_to_bet_rows(monkeypatch):
    base_df = pd.DataFrame({
        "league": ["NBA"],
        "home_team": ["Old Home"],
        "away_team": ["Old Away"],
        "game_date": ["2020-01-01"],
        "market_type": ["spread_home"],
        "odds_american": [-110],
    })

    bet_rows_df = pd.DataFrame({
        "league": ["NBA"],
        "home_team": ["Fresh Home"],
        "away_team": ["Fresh Away"],
        "game_date": ["2026-04-01T00:00:00Z"],
        "market_type": ["spread_home"],
        "spread_line": [-2.5],
        "total_line": [pd.NA],
        "theover_probability": [0.55],
        "odds_american": [-110],
        "market_probability": [0.52],
        "expected_value": [0.03], # Needs to be > 0.02
        "edge": [0.03],
        "best_pick": ["Fresh Home -2.5"],
    })

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

    analyzed, best_picks_df, diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"],
        max_rows=10,
        use_ml=False,
        spreads_df=None,
        totals_df=None,
    )

    assert not analyzed.empty
    assert diagnostics["stale_base_schedule"] is True
