import logging
from core.streamlit_pipeline import run_analysis_pipeline
import pandas as pd

logging.basicConfig(level=logging.INFO)

print("Starting pipeline...")
try:
    results = run_analysis_pipeline()
    best_picks_df = results["best_picks"]

    # Calculate Kalshi metrics to emulate the Streamlit UI output
    kalshi_matches = len(best_picks_df[best_picks_df['kalshi_match_status'] == 'matched'])
    total_games = len(best_picks_df)

    # Calculate EV distribution
    ev_dist = {
        "High Edge (>5%)": len(best_picks_df[best_picks_df['expected_value'] > 0.05]),
        "Medium Edge (2-5%)": len(best_picks_df[(best_picks_df['expected_value'] > 0.02) & (best_picks_df['expected_value'] <= 0.05)]),
        "Low Edge (0-2%)": len(best_picks_df[(best_picks_df['expected_value'] > 0) & (best_picks_df['expected_value'] <= 0.02)]),
        "Negative Edge": len(best_picks_df[best_picks_df['expected_value'] < 0])
    }

    print("\n--- Kalshi Match Count ---")
    print(f"Total Matches: {kalshi_matches} / {total_games}")

    print("\n--- EV Distribution ---")
    for category, count in ev_dist.items():
        print(f"{category}: {count}")

except Exception as e:
    print(f"Error running pipeline: {e}")
