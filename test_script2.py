import sys
sys.path.append('.')
from app_core.kalshi_integrator import _fetch_series_events

events = _fetch_series_events("KXNBASPREAD")
print(f"Fetched {len(events)} events.")
if events:
    print(events[0])
