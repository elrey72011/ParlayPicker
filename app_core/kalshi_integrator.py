"""
Kalshi Integrator with RSA Signing.
Location: app_core/kalshi_integrator.py
"""
from __future__ import annotations
import logging
import os
import time
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional
import pytz
import requests
import streamlit as st
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

class KalshiIntegrator:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY", "") or os.environ.get("KALSHI_API_KEY", "")
        self.api_secret = api_secret or st.secrets.get("KALSHI_API_SECRET", "") or os.environ.get("KALSHI_API_SECRET", "")
        
        # Clean private key format
        if self.api_secret:
            self.api_secret = self.api_secret.replace("\\n", "\n").strip()
            if "-----BEGIN RSA PRIVATE KEY-----" not in self.api_secret:
                self.api_secret = f"-----BEGIN RSA PRIVATE KEY-----\n{self.api_secret}\n-----END RSA PRIVATE KEY-----"

        # FORCE Correct Trading API URL
        self.api_url = "https://trading-api.kalshi.com/trade-api/v2"
        self._markets_cache = []
        self._markets_cache_ts = 0
        self.cache_ttl_seconds = 300
        self.last_error = None

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        if not self.api_secret: return ""
        msg_string = f"{timestamp}{method}{path}"
        try:
            private_key = serialization.load_pem_private_key(self.api_secret.encode('utf-8'), password=None)
            signature = private_key.sign(
                msg_string.encode('utf-8'),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return ""

    def _make_authenticated_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[dict]:
        url = f"{self.api_url}{endpoint}"
        path_for_signing = f"/trade-api/v2{endpoint}"
        timestamp = str(int(time.time() * 1000))
        signature = self._sign_request(method, path_for_signing, timestamp)
        
        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }
        
        try:
            resp = requests.request(method, url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                self.last_error = "401 Unauthorized - Check Keys/Signature"
                logger.error(f"Kalshi 401: {resp.text}")
            else:
                self.last_error = f"Status {resp.status_code}: {resp.text}"
        except Exception as e:
            self.last_error = str(e)
        return None

    def get_markets(self, status: str = "open") -> List[Dict[str, Any]]:
        now = time.time()
        if self._markets_cache and (now - self._markets_cache_ts) < self.cache_ttl_seconds:
            return self._markets_cache

        data = self._make_authenticated_request("GET", "/markets", params={"limit": 2000, "status": status})
        markets = data.get("markets", []) if data else []
        
        if markets:
            self._markets_cache = markets
            self._markets_cache_ts = now
            logger.info(f"✅ Loaded {len(markets)} Kalshi markets")
        
        return markets
    
    # Helpers required by app
    def get_sports_markets(self): return self.get_markets()
    def get_game_markets_for_events(self, league): return self.get_markets()
    def filter_markets_closing_today(self, markets): return markets
    def get_orderbook(self, ticker): return self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook") or {}

# Standalone helper for matching logic if imported directly
@dataclass
class KalshiMatchResult:
    matched: bool
    kalshi_available: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: Optional[str] = None
    reason: str = ""
    market_type: Optional[str] = None
    direction: Optional[str] = None
    game_date: Optional[datetime] = None
    kalshi_volume: Optional[float] = None

def price_to_prob(price):
    if price is None: return None
    try: return float(price)
    except: return None

# Bare minimum matcher to satisfy imports
def match_game_to_kalshi(league, home, away, time, integrator=None, status="open"):
    # Real logic resides in vertex_master_analyzer now, this is a stub/fallback
    return KalshiMatchResult(False, False, "", None, None, reason="use_vertex_analyzer")
