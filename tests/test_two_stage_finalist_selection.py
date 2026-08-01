import pandas as pd
import numpy as np
from core.streamlit_pipeline import build_best_picks_df, BEST_PICK_COLUMNS

def test_two_stage_finalist_selection_logic():
    # Construct a dataframe with side vs total competition
    analysis_df = pd.DataFrame({
        "league": ["NBA", "NBA", "NBA", "NBA", "NBA"],
        "home_team": ["Lakers", "Lakers", "Bulls", "Bulls", "Bulls"],
        "away_team": ["Warriors", "Warriors", "Celtics", "Celtics", "Celtics"],
        "game_date": pd.to_datetime(["2026-08-01"] * 5, utc=True),
        "market_type": ["spread_home", "total_over", "spread_away", "total_under", "h2h_home"],
        "spread_line": [-5.5, np.nan, 4.5, np.nan, np.nan],
        "total_line": [np.nan, 225.5, np.nan, 210.5, np.nan],
        "expected_value": [0.10, 0.02, 0.01, 0.15, 0.02],
        "edge": [0.05, 0.01, 0.01, 0.08, 0.02],
        "calibrated_probability": [0.55, 0.52, 0.51, 0.58, 0.52],
        "market_probability": [0.50, 0.50, 0.50, 0.50, 0.50],
        "ml_probability": [0.55, 0.52, 0.51, 0.58, 0.52],
        "odds_american": [-110, -110, -110, -110, -110],
        "odds_source": ["odds_api"] * 5,
        "is_live_data": [True] * 5,
        "candidate_source": ["upload"] * 5,
        "orientation_source": ["model"] * 5,
        "upload_match_reason": ["matched"] * 5,
        "line_source": ["live"] * 5,
        "live_spread_line": [-5.5, np.nan, 4.5, np.nan, np.nan],
        "live_total_line": [np.nan, 225.5, np.nan, 210.5, np.nan],
    })

    diagnostics = {}
    best = build_best_picks_df(analysis_df, diagnostics_out=diagnostics)

    # We expect 2 games total, meaning exactly 2 winners
    assert len(best) == 2, "One pick per game not preserved"

    # Let's check the exported strings to confirm the right finalist won
    picks = best["best_pick"].tolist()
    assert "Lakers -5.5" in picks, f"Side should have won Game 1, got {picks}"
    assert "Under 210.5" in picks, f"Total should have won Game 2, got {picks}"

    # Ensure required fields remain
    assert "market_type" in best.columns
    assert "candidate_source" in best.columns
    assert "orientation_source" in best.columns
    assert "upload_match_reason" in best.columns
    assert "odds_source" in best.columns
    assert "Pick_Status" in best.columns

    # Verify diagnostics dictionary was updated properly
    assert "selection_diagnostics" in diagnostics
    diags = diagnostics["selection_diagnostics"]
    # Production Best Available excludes moneylines, leaving the two spread rows.
    assert diags["raw_family_counts"]["side"] == 2
    assert diags["raw_family_counts"]["total"] == 2

    assert diags["finalist_family_counts"]["side"] == 2
    assert diags["finalist_family_counts"]["total"] == 2

    assert diags["final_family_counts"]["side"] == 1
    assert diags["final_family_counts"]["total"] == 1

    preview_df = diags["preview_df"]
    assert not preview_df.empty
    assert len(preview_df) == 2
