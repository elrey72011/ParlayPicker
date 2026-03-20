import pandas as pd
from app_core.kalshi_integrator import enrich_with_kalshi_markets
import json
import logging

logging.basicConfig(level=logging.INFO)

# Load pipeline snapshot
df = pd.read_csv('master_df_raw.csv')

print(f"Loaded {len(df)} rows from master dump")

# Needs to simulate the minimum columns for the integrator
df['home_team'] = df['Home']
df['away_team'] = df['Away']
df['league'] = df['League']
df['game_date'] = df['Commence (UTC)']

# Spread pick conversion
df['market_type'] = 'spread'
df['strike_price'] = df['spread_pick_line']
df['best_pick'] = df['spread_pick_team']
df['implied_home_prob'] = df['implied_home_prob'] if 'implied_home_prob' in df.columns else 0.5
df['odds_american'] = df['spread_pick_odds']

# Apply Kalshi Integrator
df = enrich_with_kalshi_markets(df)

total_matches = df['kalshi_match_status'].eq('matched').sum()
total_games = len(df)


print("=== METRICS ===")
print(f"Kalshi Match Count: {total_matches} / {total_games}")
