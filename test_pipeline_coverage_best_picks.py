import pandas as pd
import numpy as np
from core.streamlit_pipeline import build_best_picks_df

def test_market_family_selection():
    # Construct a pool with strong sides and totals for the same game to ensure
    # the two-stage comparison works and retains exactly one pick.
    df = pd.DataFrame({
        'league': ['NBA', 'NBA', 'NBA', 'NBA', 'NHL', 'NHL'],
        'home_team': ['Lakers', 'Lakers', 'Lakers', 'Lakers', 'Rangers', 'Rangers'],
        'away_team': ['Warriors', 'Warriors', 'Warriors', 'Warriors', 'Bruins', 'Bruins'],
        'game_date': ['2026-03-30'] * 6,
        'game_key': ['NBA|Lakers|Warriors'] * 4 + ['NHL|Rangers|Bruins'] * 2,
        'market_type': ['spread_home', 'spread_away', 'total_over', 'total_under', 'h2h_home', 'total_over'],
        'expected_value': [0.15, 0.02, 0.16, -0.05, 0.10, 0.05], # Strong side & total for NBA
        'edge': [0.05, 0.01, 0.06, -0.02, 0.04, 0.02],
        'calibrated_probability': [0.55, 0.48, 0.56, 0.45, 0.54, 0.52],
        'ml_probability': [0.55, 0.48, 0.56, 0.45, 0.54, 0.52],
        'theover_probability': [0.55, 0.48, 0.56, 0.45, 0.54, 0.52],
        'odds_american': [-110] * 6,
        'market_probability': [0.5] * 6,
        'spread_line': [-5.0, 5.0, np.nan, np.nan, np.nan, np.nan],
        'total_line': [np.nan, np.nan, 220.0, 220.0, np.nan, 5.5],
        'candidate_source': ['live_unfiltered'] * 6,
        'orientation_source': ['exact_match'] * 6,
        'upload_match_reason': ['exact'] * 6,
        'odds_source': ['odds_api'] * 6
    })

    out = build_best_picks_df(df)

    # Assert 1: one-pick-per-game preserved
    assert len(out) == 2, f"Expected 2 games in output, got {len(out)}"
    assert out['market_type'].tolist() == ['total_over', 'h2h_home'], f"Expected totals to win first game and side for second: {out['market_type'].tolist()}"

    # Verify schema outputs
    required_cols = ['market_type', 'candidate_source', 'orientation_source', 'upload_match_reason', 'Pick_Status', 'odds_source']
    for col in required_cols:
        assert col in out.columns, f"Missing required column: {col}"

    print("Test passed: Best picks retains exactly one row per game and compares fairly across families.")

    diag = out.attrs.get("selection_diagnostics")
    if diag:
        print("Diagnostics captured successfully:")
        print(f"Raw Counts: {diag['raw_counts']}")
        print(f"Finalist Counts: {diag['finalist_counts']}")
        print(f"Final Counts: {diag['final_counts']}")

if __name__ == "__main__":
    test_market_family_selection()
