import pandas as pd
from app_core.kalshi_integrator import enrich_with_kalshi_markets

df = pd.DataFrame([{
    "league": "NBA",
    "market_type": "spread",
    "game_date": "2024-03-26",
    "away_team": "Boston Celtics",
    "home_team": "Miami Heat",
    "spread_line": -5.5
}])

try:
    res = enrich_with_kalshi_markets(df)
    print("Success")
    print(res[['kalshi_match_status', 'kalshi_match_reason']])
except Exception as e:
    import traceback
    traceback.print_exc()
