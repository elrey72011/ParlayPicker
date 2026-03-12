import sys
sys.path.append('.')
import app_core.kalshi_integrator as ki

event = {
    "event_ticker": "KXNBASPREAD-26MAR10LALBOS",
    "title": "Boston at Los Angeles L: Spread",
    "sub_title": "LAL at BOS (Mar 10)",
    "close_time": "2026-03-11T05:00:00Z",
}
import pandas as pd
game_date = pd.to_datetime("2026-03-10T00:00:00Z", utc=True)
print(ki._is_within_24h(event, game_date))
