import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import core.streamlit_pipeline as sp


def test_run_analysis_pipeline_keeps_rows_when_live_odds_unmatched(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "game_date": ["2026-04-01T00:00:00Z"],
        }
    )

    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "market_type": ["spread_home"],
            "spread_line": [-3.5],
            "theover_probability": [0.56],
            "odds_american": [pd.NA],
        }
    )

    # Live odds exist but do not match uploaded teams
    live_odds_df = pd.DataFrame(
        {
            "home_team": ["Denver Nuggets"],
            "away_team": ["Phoenix Suns"],
            "novig_home_price": [-105],
            "novig_away_price": [-115],
            "novig_home_point": [-2.5],
            "novig_away_point": [2.5],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda _sports: live_odds_df)

    analysis_df, _best_picks_df, diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=10, use_ml=False, spreads_df=None, totals_df=None
    )

    assert len(analysis_df) == 1
    assert float(analysis_df.iloc[0]["odds_american"]) == -110.0
    assert analysis_df.iloc[0]["odds_source"] == "fallback_novig"
    assert diagnostics["total_rows"] == 1


def test_run_analysis_pipeline_keeps_reversed_live_novig_match(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "game_date": ["2026-04-01T00:00:00Z"],
        }
    )

    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "market_type": ["spread_home"],
            "spread_line": [-3.5],
            "theover_probability": [0.56],
            "odds_american": [pd.NA],
        }
    )

    # Reversed orientation vs bet_rows (home/away swapped), but same matchup.
    live_odds_df = pd.DataFrame(
        {
            "home_team": ["Miami Heat"],
            "away_team": ["Boston Celtics"],
            "novig_home_price": [-101],
            "novig_away_price": [-109],
            "novig_home_point": [3.5],
            "novig_away_point": [-3.5],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda _sports: live_odds_df)

    analysis_df, _best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=10, use_ml=False, spreads_df=None, totals_df=None
    )

    assert len(analysis_df) == 1
    assert float(analysis_df.iloc[0]["odds_american"]) == -109.0
    assert analysis_df.iloc[0]["odds_source"] == "novig_live"


def test_run_analysis_pipeline_does_not_drop_live_odds_when_base_unrelated(monkeypatch):
    # Base schedule has unrelated game; uploaded rows should still be eligible for live Novig merge.
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Denver Nuggets"],
            "away_team": ["Phoenix Suns"],
            "game_date": ["2026-04-01T00:00:00Z"],
        }
    )

    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "market_type": ["spread_home"],
            "spread_line": [-3.5],
            "theover_probability": [0.56],
            "odds_american": [pd.NA],
        }
    )

    live_odds_df = pd.DataFrame(
        {
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "novig_home_price": [-108],
            "novig_away_price": [-102],
            "novig_home_point": [-3.5],
            "novig_away_point": [3.5],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda _sports: live_odds_df)

    analysis_df, _best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=10, use_ml=False, spreads_df=None, totals_df=None
    )

    assert len(analysis_df) == 1
    assert float(analysis_df.iloc[0]["odds_american"]) == -108.0
    assert analysis_df.iloc[0]["odds_source"] == "novig_live"
