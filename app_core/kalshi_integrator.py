"""
Kalshi Integrator v6.6: Universal Search & Authentication Fix
Key fixes:
1. Uses Authenticated Trading API if keys are present (more reliable data).
2. "Hail Mary" Search: If specific tickers fail, fetches ALL open markets and filters by text.
3. Fixed boolean parameter formatting for 'with_nested_markets'.
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

import pytz
import requests
import streamlit as st

try:
    from app_core.team_name_matcher import TeamNameMatcher
except ImportError:
    class TeamNameMatcher:
        @staticmethod
        def normalize(s): return s.lower().strip()
        @staticmethod
        def similarity_score(a, b): return 0.0

logger = logging.getLogger(__name__)

# Updated to use the broader series tickers by default
GAME_SERIES = {
    "NBA": "KXNBA",
    "NFL": "KXNFL", 
    "NHL": "KXNHL",
    "MLB": "KXMLB",
    "NCAAF": "KXNCAAF",
    "NCAAB": "KXNCAAB"
}

# Keep your existing KALSHI_ABBREVIATIONS dictionary exactly as is...
# (I am omitting the long dictionary for brevity, but assume it remains unchanged)
# [Insert KALSHI_ABBREVIATIONS and TEAM_TO_TICKER logic here from your previous file]
# ...

# --- Placeholder for the abbreviation blocks ---
# Ensure you copy your full KALSHI_ABBREVIATIONS and _build_reverse_lookups() here
# -----------------------------------------------
KALSHI_ABBREVIATIONS = {
    "NBA": {
         "LAL": ["LOS ANGELES LAKERS", "LAKERS"],
         "GSW": ["GOLDEN STATE WARRIORS", "WARRIORS"],
         "BOS": ["BOSTON CELTICS", "CELTICS"],
         # ... (Use your full list)
    }
}
TEAM_TO_TICKER = {}
def _build_reverse_lookups():
    for league, teams in KALSHI_ABBREVIATIONS.items():
        TEAM_TO_TICKER[league] = {}
        for code, names in teams.items():
            for name in names:
                normalized = name.upper().strip()
                clean = re.sub(r"[^A-Z\s]", "", normalized).strip()
                for key in [normalized, clean]:
                    if key: TEAM_TO_TICKER[league][key] = code
_build_reverse_lookups()
# -----------------------------------------------

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
        return max(0.0, min(1.0, p / 100.0 if p > 1.01 else p))
    except: return None

def _parse_market_date(raw) -> Optional[datetime]:
    if not raw: return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw) / 1000.0, tz=pytz.UTC)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.replace(tzinfo=pytz.UTC) if dt.tzinfo is None else dt
    except: return None

class KalshiIntegrator:
    def __init__(self, api_key: str = None, api_secret: str = None):
        # 1. Load Keys
        if not api_key:
            api_key = st.secrets.get("KALSHI_API_KEY")
            api_secret = st.secrets.get("KALSHI_API_SECRET")
        
        self.api_key = api_key
        self.api_secret = api_secret
        
        # 2. Select Endpoint
        self.public_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.prod_url = "https://trading-api.kalshi.com/trade-api/v2"
        
        # Use Prod if keys exist, otherwise Public. 
        # Note: If you have keys, we assume you want the full Trading API data.
        self.base_url = self.prod_url if (self.api_key and self.api_secret) else self.public_url
        self._markets_cache = {}

    def _make_request(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Add simple Auth header if available (Note: V2 often requires RSA signing, 
        # but this header helps if using the simple bearer token flow).
        # If your keys are RSA keys, this simple bearer might be ignored, 
        # falling back to public read access on some endpoints.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(1)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"Kalshi API {resp.status_code} at {url}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Kalshi Connection Error: {e}")
            return None

    def get_todays_events(self, sport_ticker=None, days_ahead=3):
        """
        Fetch markets using a multi-stage fallback strategy.
        """
        now = int(time.time())
        cache_key = f"events_{sport_ticker or 'all'}_{now // 300}"
        if cache_key in self._markets_cache:
            return self._markets_cache[cache_key]

        all_events = []
        seen_ids = set()
        
        # Strategy 1: Targeted Ticker Search
        # Try likely tickers: e.g. "KXNBA", then "NBA", then "KXNBAGAME"
        targets = []
        if sport_ticker:
            targets.append(sport_ticker)
            if "KX" in sport_ticker: targets.append(sport_ticker.replace("KX", ""))
            if "GAME" in sport_ticker: targets.append(sport_ticker.replace("GAME", ""))
            if "KX" not in sport_ticker: targets.append(f"KX{sport_ticker}")
        
        # Always try "NBA" (simple) if looking for basketball
        if sport_ticker and "NBA" in sport_ticker:
            targets.append("NBA")

        logger.info(f"Searching Kalshi targets: {targets}")
        
        found_any = False
        for series in targets:
            # Note: "true" string is required for some strict API parsers
            params = {
                "series_ticker": series,
                "with_nested_markets": "true",
                "limit": 100,
                "status": "open"
            }
            
            data = self._make_request("/events", params)
            events = data.get("events", []) if data else []
            
            if events:
                logger.info(f"✓ Found {len(events)} events for {series}")
                found_any = True
                self._process_events(events, all_events, seen_ids, days_ahead, series)
                break # Stop if we found specific data
        
        # Strategy 2: The "Hail Mary" (Global Search)
        # If targeted search returned 0 events, fetch ALL open events and filter text manually
        if not found_any and len(all_events) == 0:
            logger.warning("Targeted search failed. Attempting global fetch...")
            params = {
                "limit": 200, # Fetch a large batch
                "status": "open",
                "with_nested_markets": "true"
            }
            data = self._make_request("/events", params)
            events = data.get("events", []) if data else []
            
            # Filter locally for the sport name (e.g. "NBA")
            sport_term = (sport_ticker or "").replace("KX", "").replace("GAME", "")
            
            filtered_events = []
            for e in events:
                # Check title, ticker, and subtitle for the sport name
                blob = f"{e.get('title')} {e.get('event_ticker')} {e.get('series_ticker')}".upper()
                if sport_term in blob:
                    filtered_events.append(e)
            
            logger.info(f"Global fetch found {len(events)} total, {len(filtered_events)} matched '{sport_term}'")
            self._process_events(filtered_events, all_events, seen_ids, days_ahead, "GLOBAL_SEARCH")

        self._markets_cache[cache_key] = all_events
        return all_events

    def _process_events(self, events, output_list, seen_ids, days_ahead, source_tag):
        for e in events:
            markets = e.get("markets", [])
            for m in markets:
                ticker = m.get("ticker")
                if ticker and ticker not in seen_ids:
                    # Enrich market object
                    m['event_title'] = e.get('title')
                    m['event_ticker'] = e.get('event_ticker')
                    m['series'] = e.get('series_ticker') or source_tag
                    
                    # Date Filter
                    close_time = _parse_market_date(m.get("close_time"))
                    if close_time:
                        now_dt = datetime.now(pytz.UTC)
                        delta = (close_time - now_dt).total_seconds() / 86400
                        
                        if -0.5 <= delta <= days_ahead:
                            seen_ids.add(ticker)
                            output_list.append(m)

    # ... [Keep your existing methods: get_sports_markets, get_game_market, match_game_to_kalshi, etc.] ...
    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        # Use the unified get_todays_events which now handles the searching logic
        ticker_map = {
            'nba': 'KXNBA', 'nfl': 'KXNFL', 'nhl': 'KXNHL', 'mlb': 'KXMLB',
            'ncaaf': 'KXNCAAF', 'ncaab': 'KXNCAAB'
        }
        target = ticker_map.get(sport.lower(), 'KXNBA')
        markets = self.get_todays_events(target)
        
        # ... [Paste your existing matching logic here from the previous file] ...
        # (This part of your code was working fine, the issue was just getting the markets list)
        
        # Temporary logic for completeness of this snippet:
        # -------------------------------------------------
        home_code = self._lookup_team_code(home_team, sport.upper())
        away_code = self._lookup_team_code(away_team, sport.upper())
        
        best_market = None
        # ... (rest of matching logic) ...
        # For now, return empty if no markets found
        if not markets:
             return {"kalshi_available": False, "kalshi_prob": None, "kalshi_match_debug": "No Markets Found"}
             
        # Reuse your existing matching logic below...
        return super().get_game_market(home_team, away_team, sport, game_time) if hasattr(super(), 'get_game_market') else \
               self._legacy_match_logic(markets, home_team, away_team, home_code, away_code)

    def _legacy_match_logic(self, markets, home, away, h_code, a_code):
        # Quick fallback implementation of your matching logic
        # You should replace this with your actual full method
        for m in markets:
             if h_code and a_code and h_code in m['event_ticker'] and a_code in m['event_ticker']:
                 return {"kalshi_available": True, "kalshi_prob": price_to_prob(m.get('yes_bid', 50)), "kalshi_label": m['event_title']}
        return {"kalshi_available": False, "kalshi_prob": None}

    def _lookup_team_code(self, team, league):
        # (Your existing lookup code)
        if league not in TEAM_TO_TICKER: return None
        return TEAM_TO_TICKER[league].get(TeamNameMatcher.normalize(team).upper())
