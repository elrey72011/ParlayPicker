import sys
sys.path.append('.')
from app_core.kalshi_integrator import _make_kalshi_request

url = "https://api.elections.kalshi.com/trade-api/v2/events"
params = {"series_ticker": "KXNBASPREAD", "limit": 100}

resp = _make_kalshi_request(url, params=params)
print(resp.json())
