import pandas as pd
from app_core.kalshi_integrator import enrich_with_kalshi_markets

# Create dummy row
df = pd.DataFrame([{
    "league": "NBA",
    "home_team": "Atlanta Hawks",
    "away_team": "Boston Celtics",
    "game_date": "2024-01-01",
    "market_type": "spread_home",
    "best_pick": "Atlanta Hawks +5.5",
    "spread_line": "+5.5",
    "category": "spread"
}])

# Enrich
out = enrich_with_kalshi_markets(df)

# Check missing status and probability
print(out[["kalshi_match_status", "kalshi_probability"]].to_dict('records'))
