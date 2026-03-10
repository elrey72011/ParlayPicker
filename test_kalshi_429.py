import os
import sys

from app_core.kalshi_integrator import api_get_markets

try:
    print("Testing api_get_markets...")
    resp = api_get_markets(series_ticker="KXNBASPREAD", status="open", limit=1)
    print("Success:", "cursor" in resp)
except Exception as e:
    print("Error:", type(e), e)
