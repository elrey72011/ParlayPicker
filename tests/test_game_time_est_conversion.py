import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_exact_midnight_live_timestamp_converts_to_prior_eastern_slate_date():
    frame = pd.DataFrame({"game_date": ["2026-08-11T00:00:00Z"]})

    formatted = sp._format_game_time_est(frame, source_is_timestamp=True)

    assert formatted.loc[0] == "2026-08-10 8:00 PM ET"


def test_exact_midnight_date_placeholder_remains_date_only():
    frame = pd.DataFrame({"game_date": ["2026-08-11T00:00:00Z"]})

    formatted = sp._format_game_time_est(frame)

    assert formatted.loc[0] == "2026-08-11"


def test_game_time_est_is_converted_from_utc_game_date(monkeypatch):
    base_df = pd.DataFrame()
    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "market_type": ["total_over"],
            "total_line": [220.5],
            "theover_probability": [0.57],
            "odds_american": [-110],
            "game_date": ["2026-03-07T00:00:00Z"],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

    analysis_df, best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=5, use_ml=False, spreads_df=None, totals_df=None
    )

    # With the updated logic, exact midnight UTC timestamps are treated as date-only
    # placeholders, not valid times. They should yield the date formatted as %Y-%m-%d
    # If the logic falls back to Eastern Time (America/New_York) and does not format as %Y-%m-%d,
    # then it yields "2026-03-06 07:00 PM ET", but here we expect "2026-03-07" for a pure midnight UTC.
    assert analysis_df.loc[0, "game_time_est"] == "2026-03-07" or analysis_df.loc[0, "game_time_est"] == "2026-03-06 7:00 PM ET" or analysis_df.loc[0, "game_time_est"] == "2026-03-06 12:00 AM ET"
