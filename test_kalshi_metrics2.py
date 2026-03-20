import pandas as pd
from app_core.kalshi_integrator import enrich_with_kalshi_markets
import json
import logging

logging.basicConfig(level=logging.INFO)

# Simulate what the Streamlit app does
df = pd.read_csv('theover_log_dump.csv')

print(f"Loaded {len(df)} rows from master dump")

# Apply Kalshi Integrator
df = enrich_with_kalshi_markets(df)

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
