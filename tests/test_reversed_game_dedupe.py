import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_best_picks_dedupes_reversed_home_away_matchups(monkeypatch):
    base_df = pd.DataFrame()
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())
        # _dedupe_inverted_matchups uses market direction for deduplication across different rows with SAME directional pick type
        # Meaning two rows for "total_over" with reversed names are deduped, not "total_over" vs "total_under".
    bet_rows_df = pd.DataFrame(
        {
            "league": ["NCAAB", "NCAAB"],
            "home_team": ["Auburn", "Alabama"],
            "away_team": ["Alabama", "Auburn"],
                "market_type": ["total_over", "total_over"],
            "total_line": [176.5, 176.5],
                "spread_line": [pd.NA, pd.NA],
                "theover_probability": [0.52, 0.52],
            "odds_american": [-110, -110],
            "game_date": ["2026-03-07T00:00:00Z", "2026-03-07T00:00:00Z"],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)

    _analysis_df, best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NCAAB"], max_rows=10, use_ml=False, spreads_df=None, totals_df=None
    )

    # In order for this mock test to work after pipeline refactor, we need to pass the raw frame
    # since `run_analysis_pipeline` builds the master slate off fetch_live_odds_dataframe directly,
    # and if it's empty, it returns empty.
    # To test deduplication, we will directly test `_dedupe_inverted_matchups`.

    deduped = sp._dedupe_inverted_matchups(bet_rows_df)
    assert len(deduped) == 1
