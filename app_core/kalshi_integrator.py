"""
Kalshi Integrator v6.0: Full Compatibility & Event-Centric Matching
This file goes in: app_core/kalshi_integrator.py
"""

import time
import logging
from datetime import datetime, timedelta
import pytz
import requests
import streamlit as st
from typing import Dict, List, Any, Optional, TypedDict, Tuple

# Try to import the matcher, or define a dummy one if missing
try:
    from app_core.team_name_matcher import TeamNameMatcher
except ImportError:
    class TeamNameMatcher:
        @staticmethod
        def normalize(s): return s.lower().strip()
        @staticmethod
        def similarity_score(a, b): return 0.0

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Hardcoded series ensure we NEVER miss the major sports
CORE_SERIES = ["KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXNCAAF", "KXNCAAB"]

class KalshiMatchResult(TypedDict, total=False):
    matched: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: str
    reason: str

def price_to_prob(price) -> Optional[float]:
    if price is None or price == "": return None
    try:
        p = float(price)
        if p > 1.01: p = p / 100.0
        return max(0.0, min(1.0, p))
    except: return None

def _parse_market_date(raw) -> Optional[datetime]:
    if not raw: return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw) / 1000.0, tz=pytz.UTC)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except: return None

class KalshiIntegrator:
    def __init__(self, api_key: str = None, api_secret: str = None):
        # Load keys from secrets if not provided
        if not api_key:
            try:
                api_key = st.secrets.get("KALSHI_API_KEY")
                api_secret = st.secrets.get("KALSHI_API_SECRET")
            except: pass

        self.api_key = api_key
        self.api_secret = api_secret

        # --- CORRECT URLS ---
        # Authenticated Trading
        self.prod_url = "https://trading-api.kalshi.com/trade-api/v2"
        # Public Market Data (No Auth Required)
        self.public_url = "https://api.elections.kalshi.com/trade-api/v2"
        
        # Decide which URL to use for authenticated calls (if any)
        self.api_url = self.prod_url if (self.api_key and self.api_secret) else self.public_url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        # RSA Setup (Only needed for trading, not data fetching)
        self._private_key = None
        self._auth_ready = False
        if self.api_key and self.api_secret:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                key_data = self.api_secret.strip()
                if not key_data.startswith('-----BEGIN'):
                    key_data = f"-----BEGIN RSA PRIVATE KEY-----\n{key_data}\n-----END RSA PRIVATE KEY-----"
                self._private_key = serialization.load_pem_private_key(
                    key_data.encode(), password=None, backend=default_backend()
                )
                self._auth_ready = True
            except Exception as e:
                logger.warning(f"Kalshi Auth Error: {e}")

        self._markets_cache = {}
        self._cache_time = {}
        self.last_error = None

    def _sign_request(self, method, path, timestamp):
        """Sign request for authenticated actions (buying/selling)."""
        if not self._private_key: return ""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            msg = f"{timestamp}{method}{path}"
            sig = self._private_key.sign(
                msg.encode('utf-8'),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256()
            )
            return base64.b64encode(sig).decode('utf-8')
        except: return ""

    def _make_authenticated_request(self, method, endpoint, params=None):
        """Authenticated request wrapper for trading actions."""
        import time as tm
        url = f"{self.prod_url}{endpoint}"
        timestamp = str(int(tm.time() * 1000))
        headers = self.headers.copy()
        
        if self._auth_ready:
            path = endpoint.split('?')[0]
            sig = self._sign_request(method.upper(), path, timestamp)
            headers.update({
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-SIGNATURE": sig,
                "KALSHI-ACCESS-TIMESTAMP": timestamp
            })
            
        try:
            if method == "GET": 
                resp = requests.get(url, headers=headers, params=params, timeout=10)
            else: 
                resp = requests.post(url, headers=headers, json=params, timeout=10)
            
            if resp.status_code == 200: 
                self.last_error = None
                return resp.json()
            else:
                self.last_error = f"{resp.status_code}: {resp.text}"
                logger.error(f"Kalshi Auth API Error: {self.last_error}")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Kalshi Connection Error: {e}")
        return None

    def _make_public_request(self, endpoint, params=None):
        """Always uses the public URL for data fetching to avoid auth/rate-limit errors."""
        url = f"{self.public_url}{endpoint}"
        try:
            resp = requests.get(url, headers={"Accept": "application/json"}, params=params, timeout=10)
            if resp.status_code == 200:
                self.last_error = None
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(1)
                return self._make_public_request(endpoint, params)
            else:
                self.last_error = f"{resp.status_code}: {resp.text}"
                logger.error(f"Kalshi Public API Error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Kalshi Connection Error: {e}")
            return None

    def get_todays_events(self, sport_ticker=None):
        """
        Fetch ONLY events closing in the next 48 hours.
        This is the most efficient way to find 'Today's Games'.
        """
        now = int(time.time())
        future = now + (48 * 60 * 60)
        
        target_series = [sport_ticker] if sport_ticker else CORE_SERIES
        all_events = []

        # Cache key based on sport_ticker or 'all'
        cache_key = f"events_{sport_ticker or 'all'}_{now // 300}" # 5 min cache bucket
        if cache_key in self._markets_cache:
            return self._markets_cache[cache_key]

        for series in target_series:
            params = {
                "series_ticker": series,
                "status": "open",
                "min_close_ts": now,
                "max_close_ts": future,
                "with_nested_markets": "true",
                "limit": 100
            }
            
            data = self._make_public_request("/events", params)
            if data and "events" in data:
                events = data["events"]
                if events:
                    # Flatten markets inside events for easier consumption
                    for e in events:
                        markets = e.get('markets', [])
                        # Augment markets with event info
                        for m in markets:
                            m['event_title'] = e.get('title')
                            m['event_ticker'] = e.get('event_ticker')
                            m['sport'] = series
                        all_events.extend(markets)
        
        self._markets_cache[cache_key] = all_events
        return all_events

    # --- COMPATIBILITY METHODS FOR STREAMLIT APP ---
    
    def get_sports_markets(self):
        """Alias for get_todays_events (used by health check)."""
        return self.get_todays_events()

    def get_markets(self, **kwargs):
        """Standard market fetcher."""
        return self.get_todays_events()

    def get_game_markets_for_events(self, league="NBA"):
        """Wrapper to get markets filtered by league."""
        sport_map = {
            'NBA': 'KXNBA', 'NFL': 'KXNFL', 
            'NHL': 'KXNHL', 'MLB': 'KXMLB',
            'NCAAF': 'KXNCAAF', 'NCAAB': 'KXNCAAB'
        }
        ticker = sport_map.get(league, 'KXNBA')
        return self.get_todays_events(ticker)

    def filter_markets_closing_today(self, markets):
        """Filters a list of markets to those closing today."""
        if not markets: return []
        today = datetime.now(pytz.UTC).date()
        filtered = []
        for m in markets:
            dt = _parse_market_date(m.get("close_time"))
            if dt and dt.date() == today:
                filtered.append(m)
        return filtered

    def get_orderbook(self, ticker: str):
        """Fetch orderbook for a specific market ticker."""
        data = self._make_public_request(f"/markets/{ticker}/orderbook", {"depth": 5})
        if data and 'orderbook' in data:
            return data['orderbook']
        return None

    # --- MATCHING LOGIC ---

    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        """High-level matcher: Finds the best market for a specific game."""
        sport_map = {
            'nba': 'KXNBA', 'basketball': 'KXNBA',
            'nfl': 'KXNFL', 'football': 'KXNFL',
            'nhl': 'KXNHL', 'hockey': 'KXNHL',
            'mlb': 'KXMLB', 'baseball': 'KXMLB',
            'ncaaf': 'KXNCAAF', 'college football': 'KXNCAAF',
            'ncaab': 'KXNCAAB', 'college basketball': 'KXNCAAB'
        }
        series_ticker = sport_map.get(sport.lower(), "KXNBA")
        
        # 1. Fetch relevant events (markets)
        markets = self.get_todays_events(series_ticker)
        
        # 2. Normalize input teams
        norm_home = TeamNameMatcher.normalize(home_team)
        norm_away = TeamNameMatcher.normalize(away_team)
        
        best_market = None
        best_score = 0.0
        
        # 3. Match Logic
        for m in markets:
            # Check event title (usually "Home vs Away")
            event_title = m.get("event_title", "").lower()
            event_norm = TeamNameMatcher.normalize(event_title)
            
            # Check market title/subtitle
            market_text = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
            market_norm = TeamNameMatcher.normalize(market_text)
            
            full_text_norm = event_norm + " " + market_norm

            # Match score
            home_score = TeamNameMatcher.similarity_score(norm_home, full_text_norm)
            away_score = TeamNameMatcher.similarity_score(norm_away, full_text_norm)
            
            if home_score > 0.4 and away_score > 0.4:
                avg_score = (home_score + away_score) / 2
                
                # Boost "Winner" markets
                if "winner" in market_text or "win" in market_text:
                    avg_score += 0.1
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_market = m

        res = {
            "kalshi_available": False, 
            "kalshi_prob": None, 
            "kalshi_match_debug": "no_match",
            "kalshi_label": None
        }

        if best_market and best_score > 0.55:
            # Extract probability
            yes_price = price_to_prob(best_market.get("yes_bid", 0))
            market_subtitle = best_market.get("subtitle", "").lower()
            
            # Determine if "Yes" means Home or Away wins
            is_home_winner = norm_home in TeamNameMatcher.normalize(market_subtitle)
            
            final_prob = yes_price
            if not is_home_winner and yes_price:
                final_prob = 1.0 - yes_price

            res.update({
                "kalshi_available": True,
                "kalshi_prob": final_prob,
                "kalshi_label": best_market.get("event_title"),
                "kalshi_match_debug": "match_found",
                "market_ticker": best_market.get("ticker"),
                "kalshi_volume": best_market.get("volume")
            })
            
        return res

# Global Helper
def match_game_to_kalshi(league, home, away, time, integrator=None, status="open"):
    k = integrator or KalshiIntegrator()
    r = k.get_game_market(home, away, league, time)
    return KalshiMatchResult(
        matched=r["kalshi_available"],
        label=r.get("kalshi_label"),
        probability=r.get("kalshi_prob"),
        reason=r.get("kalshi_match_debug"),
        raw_event_id=r.get("market_ticker"),
        league=league
    )
