"""
Kalshi Integrator v6.0: Event-Centric Matching
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

    def _make_public_request(self, endpoint, params=None):
        """Always uses the public URL for data fetching to avoid auth errors."""
        url = f"{self.public_url}{endpoint}"
        try:
            resp = requests.get(url, headers={"Accept": "application/json"}, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(1)
                return self._make_public_request(endpoint, params)
            else:
                logger.error(f"Kalshi Public API Error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Kalshi Connection Error: {e}")
            return None

    def get_todays_events(self, sport_ticker=None):
        """
        Fetch ONLY events closing in the next 48 hours.
        This is the most efficient way to find 'Today's Games'.
        """
        # Calculate time window (Now to +48 hours)
        now = int(time.time())
        future = now + (48 * 60 * 60)
        
        target_series = [sport_ticker] if sport_ticker else CORE_SERIES
        all_events = []

        print(f"📥 Fetching Kalshi events closing between now and +48h...")

        for series in target_series:
            # We use the /events endpoint because it groups markets (Winner, Spread, Total)
            # under a single game title like "Lakers vs Warriors"
            params = {
                "series_ticker": series,
                "status": "open",
                "min_close_ts": now,
                "max_close_ts": future,
                "with_nested_markets": "true", # Important! Gets the actual odds
                "limit": 100
            }
            
            data = self._make_public_request("/events", params)
            if data and "events" in data:
                events = data["events"]
                if events:
                    print(f"   found {len(events)} events for {series}")
                    all_events.extend(events)
        
        return all_events

    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        """
        High-level matcher: Finds the best market for a specific game.
        """
        # Map generic sport names to Kalshi tickers
        sport_map = {
            'nba': 'KXNBA', 'basketball': 'KXNBA',
            'nfl': 'KXNFL', 'football': 'KXNFL',
            'nhl': 'KXNHL', 'hockey': 'KXNHL',
            'mlb': 'KXMLB', 'baseball': 'KXMLB',
            'ncaaf': 'KXNCAAF', 'college football': 'KXNCAAF',
            'ncaab': 'KXNCAAB', 'college basketball': 'KXNCAAB'
        }
        series_ticker = sport_map.get(sport.lower(), "KXNBA")
        
        # 1. Fetch relevant events
        events = self.get_todays_events(series_ticker)
        
        # 2. Normalize input teams
        norm_home = TeamNameMatcher.normalize(home_team)
        norm_away = TeamNameMatcher.normalize(away_team)
        
        best_market = None
        best_score = 0.0
        
        # 3. Match Logic
        for event in events:
            # Event title is usually "Home vs Away" or "Away @ Home"
            event_title = event.get("title", "").lower()
            event_norm = TeamNameMatcher.normalize(event_title)
            
            # Check if BOTH teams are present in the event title
            # This is much safer than checking individual market titles
            home_score = TeamNameMatcher.similarity_score(norm_home, event_norm)
            away_score = TeamNameMatcher.similarity_score(norm_away, event_norm)
            
            # Strict check: Both teams must match reasonably well
            if home_score > 0.4 and away_score > 0.4:
                avg_score = (home_score + away_score) / 2
                
                if avg_score > best_score:
                    # Found a matching event! Now find the "Game Winner" market inside it
                    markets = event.get("markets", [])
                    # Look for the main market (usually the first one, or one with "Winner" in subtitle)
                    for m in markets:
                        # We prefer the generic "Winner" market over spreads/totals for the main prob
                        if "winner" in m.get("subtitle", "").lower() or len(markets) == 1:
                            best_market = m
                            best_score = avg_score
                            # Add event title for context
                            best_market['event_title'] = event.get('title')
                            break
        
        res = {
            "kalshi_available": False, 
            "kalshi_prob": None, 
            "kalshi_match_debug": "no_match",
            "kalshi_label": None
        }

        if best_market and best_score > 0.55:
            # Extract probability from the "Yes" price
            # In Kalshi "Game Winner" markets, the market title often corresponds to the Home team? 
            # Or usually it's "Team A vs Team B", and the market is "Team A wins?"
            # We need to be careful about which side is "Yes".
            
            # Usually the market ticker has the team code, e.g. KXNBA-23NOV-LAL
            # Or the subtitle says "Lakers win".
            
            market_subtitle = best_market.get("subtitle", "").lower()
            yes_price = price_to_prob(best_market.get("yes_bid", 0))
            
            # Determine if "Yes" means Home or Away wins
            is_home_winner = norm_home in TeamNameMatcher.normalize(market_subtitle)
            
            final_prob = yes_price
            if not is_home_winner:
                # If "Yes" is for the Away team, Home prob is 1 - Yes
                # (Assuming binary market)
                final_prob = 1.0 - yes_price if yes_price else None

            res.update({
                "kalshi_available": True,
                "kalshi_prob": final_prob,
                "kalshi_label": best_market.get("event_title"),
                "kalshi_match_debug": "match_found",
                "market_ticker": best_market.get("ticker"),
                "kalshi_volume": best_market.get("volume")
            })
            
        return res

# Helper for the rest of the app
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

# Alias for compatibility with existing code
    def get_sports_markets(self):
        return self.get_todays_events()
    
    def get_markets(self, **kwargs):
        return self.get_todays_events()
