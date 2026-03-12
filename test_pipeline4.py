import sys
import pandas as pd
import logging
sys.path.append('.')
from app_core.kalshi_integrator import _fetch_series_events

events = _fetch_series_events("KXNBASPREAD")
print([e["title"] for e in events])
