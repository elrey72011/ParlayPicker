import sys
import pandas as pd
import logging
sys.path.append('.')
import app_core.kalshi_integrator as ki

game_date = pd.to_datetime("2026-03-10T00:00:00Z", utc=True)
event = {
    "event_ticker": "KXNBASPREAD-26MAR10LALBOS",
    "title": "Boston Celtics at Los Angeles Lakers: Spread",
    "sub_title": "LAL at BOS (Mar 10)",
    "close_time": "2026-03-11T05:00:00Z",
}

print(ki._is_within_24h(event, game_date))
