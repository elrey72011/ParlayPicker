import sys
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
sys.path.append('.')
logging.basicConfig(level=logging.DEBUG)

from app_core.kalshi_integrator import enrich_with_kalshi_markets

df = pd.DataFrame([
    {
        "league": "NBA",
        "market_type": "spread_home",
        "game_date": "2026-03-01T15:00:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "Sacramento Kings",
        "spread_line": "-5.5",
        "best_pick": "Los Angeles Lakers -5.5",
        "pick_team": "Los Angeles Lakers"
    }
])

out_df = enrich_with_kalshi_markets(df)
for row in out_df.itertuples():
    print(row.kalshi_match_status)
    print(row.kalshi_match_reason)
