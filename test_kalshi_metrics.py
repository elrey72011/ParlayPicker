import json
import pandas as pd
from core.streamlit_pipeline import run_analysis_pipeline
import os

os.environ['ODDS_API_KEY'] = 'test'

# Run the pipeline locally to get the metrics
results = run_analysis_pipeline()
df = results['best_picks']

total_matches = df['kalshi_match_status'].eq('matched').sum()
total_games = len(df)

ev_distribution = {
    "High Edge (>5%)": len(df[df['expected_value'] > 0.05]),
    "Medium Edge (2-5%)": len(df[(df['expected_value'] > 0.02) & (df['expected_value'] <= 0.05)]),
    "Low Edge (0-2%)": len(df[(df['expected_value'] > 0.0) & (df['expected_value'] <= 0.02)]),
    "Negative Edge": len(df[df['expected_value'] <= 0.0]),
}

print("=== METRICS ===")
print(f"Kalshi Match Count: {total_matches} / {total_games}")
print("EV Distribution:")
print(json.dumps(ev_distribution, indent=2))
