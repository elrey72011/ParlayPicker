import pandas as pd
import numpy as np
from core.streamlit_pipeline import build_best_picks_df

def run_audit():
    # Simulate Scenario 1: Both Side and Total candidates exist
    print("=== SCENARIO 1: Both Side and Total candidates exist ===")
    analysis_df_1 = pd.DataFrame({
        "league": ["NBA"] * 5,
        "home_team": ["Lakers", "Lakers", "Bulls", "Bulls", "Bulls"],
        "away_team": ["Warriors", "Warriors", "Celtics", "Celtics", "Celtics"],
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"], utc=True),
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
    })

    diagnostics_1 = {}
    best_1 = build_best_picks_df(analysis_df_1, diagnostics_out=diagnostics_1)

    diags_1 = diagnostics_1["selection_diagnostics"]
    print("\n--- Diagnostics (Scenario 1) ---")
    print(f"Raw Family Counts: {diags_1['raw_family_counts']}")
    print(f"Raw Market Type Counts: {diags_1['raw_market_type_counts']}")
    print(f"Finalist Family Counts: {diags_1['finalist_family_counts']}")
    print(f"Finalist Market Type Counts: {diags_1['finalist_market_type_counts']}")
    print(f"Final Family Counts: {diags_1['final_family_counts']}")
    print(f"Final Market Type Counts: {diags_1['final_market_type_counts']}")
    print(f"Actionable Family Counts: {diags_1['actionable_family_counts']}")
    print(f"Actionable Market Type Counts: {diags_1['actionable_market_type_counts']}")

    print("\n--- Preview DF (Scenario 1) ---")
    print(diags_1["preview_df"].to_string(index=False))

    # Simulate Scenario 2: Only Total candidates exist for matched games, with some live unfiltered sides that fail guardrails
    print("\n\n=== SCENARIO 2: Upload only matched Totals ===")
    analysis_df_2 = pd.DataFrame({
        "league": ["NBA"] * 5,
        "home_team": ["Lakers", "Lakers", "Bulls", "Bulls", "Bulls"],
        "away_team": ["Warriors", "Warriors", "Celtics", "Celtics", "Celtics"],
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"], utc=True),
        "market_type": ["spread_home", "total_over", "spread_away", "total_under", "h2h_home"],
        "spread_line": [-5.5, np.nan, 4.5, np.nan, np.nan],
        "total_line": [np.nan, 225.5, np.nan, 210.5, np.nan],
        "expected_value": [0.45, 0.05, 0.50, 0.08, -0.05], # Sides have EV > 40% (guardrail) or < 0
        "edge": [0.05, 0.03, 0.08, 0.05, 0.01],
        "calibrated_probability": [0.55, 0.53, 0.58, 0.55, 0.51],
        "market_probability": [0.50, 0.50, 0.50, 0.50, 0.50],
        "ml_probability": [0.55, 0.53, 0.58, 0.55, 0.51],
        "odds_american": [-110, -110, -110, -110, -110],
        "odds_source": ["odds_api"] * 5,
        "is_live_data": [True] * 5,
        "candidate_source": ["live_unfiltered", "upload_matched", "live_unfiltered", "upload_matched", "live_unfiltered"],
        "orientation_source": ["model"] * 5,
        "upload_match_reason": ["matched"] * 5,
    })

    diagnostics_2 = {}
    best_2 = build_best_picks_df(analysis_df_2, diagnostics_out=diagnostics_2)

    diags_2 = diagnostics_2["selection_diagnostics"]
    print("\n--- Diagnostics (Scenario 2) ---")
    print(f"Raw Family Counts: {diags_2['raw_family_counts']}")
    print(f"Raw Market Type Counts: {diags_2['raw_market_type_counts']}")
    print(f"Finalist Family Counts: {diags_2['finalist_family_counts']}")
    print(f"Finalist Market Type Counts: {diags_2['finalist_market_type_counts']}")
    print(f"Final Family Counts: {diags_2['final_family_counts']}")
    print(f"Final Market Type Counts: {diags_2['final_market_type_counts']}")
    print(f"Actionable Family Counts: {diags_2['actionable_family_counts']}")
    print(f"Actionable Market Type Counts: {diags_2['actionable_market_type_counts']}")

    print("\n--- Preview DF (Scenario 2) ---")
    print(diags_2["preview_df"].to_string(index=False))

    print("\n--- Final Best DF Pick Status (Scenario 2) ---")
    print(best_2[["home_team", "market_type", "Pick_Status", "Status_Reason"]].to_string(index=False))

if __name__ == "__main__":
    run_audit()
