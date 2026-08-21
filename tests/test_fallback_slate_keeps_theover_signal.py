"""Regression test: the live-odds-empty fallback must not blank TheOver/ML signal.

When fetch_live_odds_dataframe returns empty, run_analysis_pipeline falls back to
using the uploaded theover_rows directly as the master slate. Because that slate
already carries theover_probability/ml_probability/win_prob_source, the subsequent
enrichment merge used to collide on those shared columns, suffix them to
theover_probability_x/_y, and leave the canonical theover_probability entirely NA —
silently degrading the blend to ~0.50 for the whole slate exactly when the odds API
was already down. The merge now drops the pre-existing copies first (identical
values re-attach under the canonical names).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.streamlit_pipeline as sp


def test_live_odds_empty_fallback_preserves_theover_probability(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["New York Yankees"],
            "away_team": ["Detroit"],
            "game_date": ["2026-07-01T00:00:00Z"],
            "odds_american": [-110],
        }
    )
    totals_df = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["New York Yankees"],
            "away_team": ["Detroit"],
            "line": [7.5],
            "pick": ["Under 7.5"],
            "winprobability": [0.62],
            "odds_american": [-110],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda _sports: pd.DataFrame())

    analysis_df, _best, _diags = sp.run_analysis_pipeline(
        sports=["MLB"], max_rows=20, use_ml=False, spreads_df=None, totals_df=totals_df
    )

    assert not analysis_df.empty
    # No suffix collision artifacts from the enrichment self-merge.
    suffixed = [c for c in analysis_df.columns if c.endswith("_x") or c.endswith("_y")]
    assert not suffixed, f"enrichment merge left suffixed columns: {suffixed}"
    # The uploaded probability must survive into the canonical column for both
    # orientations (P(Over)=0.62 upload semantics -> over 0.62 / under 0.38).
    totals = analysis_df[analysis_df["market_type"].astype(str).str.contains("total")]
    assert totals["theover_probability"].notna().all()
    assert set(totals["theover_probability"].round(2)) == {0.62, 0.38}
    # TheOver remains its own signal. It must not also be copied into the ML
    # slot when no target-specific totals model is available.
    assert totals["ml_probability"].isna().all()
    assert totals["model_probability"].isna().all()
