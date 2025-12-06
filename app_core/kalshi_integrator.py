"""
Kalshi Integrator v6.1: Ticker-Based Matching (High Accuracy)
This file goes in: app_core/kalshi_integrator.py
"""

import time
import logging
from datetime import datetime, timedelta
import pytz
import requests
import streamlit as st
from typing import Dict, List, Any, Optional, TypedDict, Tuple

# Import matcher
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
CORE_SERIES = ["KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXNCAAF", "KXNCAAB"]

# Map Team Names to Kalshi Ticker Codes (Critical for accurate matching)
KALSHI_ABBREVIATIONS = {
    # NBA
    "ATL": ["ATLANTA HAWKS", "ATLANTA"], "BOS": ["BOSTON CELTICS", "BOSTON"],
    "BKN": ["BROOKLYN NETS", "BROOKLYN"], "CHA": ["CHARLOTTE HORNETS", "CHARLOTTE"],
    "CHI": ["CHICAGO BULLS", "CHICAGO"], "CLE": ["CLEVELAND CAVALIERS", "CLEVELAND", "CAVS"],
    "DAL": ["DALLAS MAVERICKS", "DALLAS", "MAVS"], "DEN": ["DENVER NUGGETS", "DENVER"],
    "DET": ["DETROIT PISTONS", "DETROIT"], "GSW": ["GOLDEN STATE WARRIORS", "GOLDEN STATE", "GS"],
    "HOU": ["HOUSTON ROCKETS", "HOUSTON"], "IND": ["INDIANA PACERS", "INDIANA"],
    "LAC": ["LOS ANGELES CLIPPERS", "LA CLIPPERS"], "LAL": ["LOS ANGELES LAKERS", "LA LAKERS", "LAKERS"],
    "MEM": ["MEMPHIS GRIZZLIES", "MEMPHIS"], "MIA": ["MIAMI HEAT", "MIAMI"],
    "MIL": ["MILWAUKEE BUCKS", "MILWAUKEE"], "MIN": ["MINNESOTA TIMBERWOLVES", "MINNESOTA"],
    "NOP": ["NEW ORLEANS PELICANS", "NEW ORLEANS", "NO"], "NYK": ["NEW YORK KNICKS", "NEW YORK", "KNICKS"],
    "OKC": ["OKLAHOMA CITY THUNDER", "OKLAHOMA CITY"], "ORL": ["ORLANDO MAGIC", "ORLANDO"],
    "PHI": ["PHILADELPHIA 76ERS", "PHILADELPHIA", "PHILLY", "76ERS"], "PHX": ["PHOENIX SUNS", "PHOENIX"],
    "POR": ["PORTLAND TRAIL BLAZERS", "PORTLAND"], "SAC": ["SACRAMENTO KINGS", "SACRAMENTO"],
    "SAS": ["SAN ANTONIO SPURS", "SAN ANTONIO", "SPURS"], "TOR": ["TORONTO RAPTORS", "TORONTO"],
    "UTA": ["UTAH JAZZ", "UTAH"], "WAS": ["WASHINGTON WIZARDS", "WASHINGTON"],
    # NFL
    "ARI": ["ARIZONA CARDINALS"], "BAL": ["BALTIMORE RAVENS"], "BUF": ["BUFFALO BILLS"],
    "CAR": ["CAROLINA PANTHERS"], "CHI": ["CHICAGO BEARS"], "CIN": ["CINCINNATI BENGALS"],
    "CLE": ["CLEVELAND BROWNS"], "DAL": ["DALLAS COWBOYS"], "DEN": ["DENVER BRONCOS"],
    "DET": ["DETROIT LIONS"], "GB": ["GREEN BAY PACKERS"], "HOU": ["HOUSTON TEXANS"],
    "IND": ["INDIANAPOLIS COLTS"], "JAX": ["JACKSONVILLE JAGUARS"], "KC": ["KANSAS CITY CHIEFS"],
    "LV": ["LAS VEGAS RAIDERS"], "LAC": ["LOS ANGELES CHARGERS"], "LAR": ["LOS ANGELES RAMS"],
    "MIA": ["MIAMI DOLPHINS"], "MIN": ["MINNESOTA VIKINGS"], "NE": ["NEW ENGLAND PATRIOTS"],
    "NO": ["NEW ORLEANS SAINTS"], "NYG": ["NEW YORK GIANTS"], "NYJ": ["NEW YORK JETS"],
    "PHI": ["PHILADELPHIA EAGLES"], "PIT": ["PITTSBURGH STEELERS"], "SF": ["SAN FRANCISCO 49ERS"],
    "SEA": ["SEATTLE SEAHAWKS"], "TB": ["TAMPA BAY BUCCANEERS"], "TEN": ["TENNESSEE TITANS"],
    "WSH": ["WASHINGTON COMMANDERS"],
}

# Build Reverse Lookup Map
TEAM_TO_TICKER = {}
for code, names in KALSHI_ABBREVIATIONS.items():
    for n in names:
        TEAM_TO_TICKER[TeamNameMatcher.normalize(n).upper()] = code

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
        # Load keys
        if not api_key:
            try:
                api_key = st.secrets.get("KALSHI_API_KEY")
                api_secret = st.secrets.get("KALSHI_API_SECRET")
            except: pass

        self.api_key = api_key
        self.api_secret = api_secret
        
        # URLs
        self.prod_url = "https://trading-api.kalshi.com/trade-api/v2"
        self.public_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.api_url = self.prod_url if (self.api_key and self.api_secret) else self.public_url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        # RSA Auth
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
        self.last_error = None

    def _make_public_request(self, endpoint, params=None):
        url = f"{self.public_url}{endpoint}"
        try:
            resp = requests.get(url, headers={"Accept": "application/json"}, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(1)
                return self._make_public_request(endpoint, params)
            else:
                self.last_error = f"{resp.status_code}: {resp.text}"
                return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def get_todays_events(self, sport_ticker=None):
        now = int(time.time())
        future = now + (48 * 60 * 60) # 48 hours
        
        target_series = [sport_ticker] if sport_ticker else CORE_SERIES
        cache_key = f"events_{sport_ticker or 'all'}_{now // 300}"
        
        if cache_key in self._markets_cache:
            return self._markets_cache[cache_key]

        all_events = []
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
                for e in data["events"]:
                    for m in e.get("markets", []):
                        m['event_title'] = e.get('title')
                        m['event_ticker'] = e.get('event_ticker') # e.g. KXNBA-23NOV-LAL-GSW
                        m['series'] = series
                        all_events.append(m)
        
        self._markets_cache[cache_key] = all_events
        return all_events

    # --- COMPATIBILITY METHODS ---
    def get_sports_markets(self):
        return self.get_todays_events()

    def get_markets(self, **kwargs):
        return self.get_todays_events()

    def get_game_markets_for_events(self, league="NBA"):
        sport_map = {'NBA': 'KXNBA', 'NFL': 'KXNFL', 'NHL': 'KXNHL', 'MLB': 'KXMLB'}
        ticker = sport_map.get(league.upper(), 'KXNBA')
        return self.get_todays_events(ticker)

    def filter_markets_closing_today(self, markets):
        if not markets: return []
        today = datetime.now(pytz.UTC).date()
        return [m for m in markets if _parse_market_date(m.get("close_time")) and abs((_parse_market_date(m.get("close_time")).date() - today).days) <= 1]

    def get_orderbook(self, ticker):
        data = self._make_public_request(f"/markets/{ticker}/orderbook", {"depth": 5})
        return data.get('orderbook') if data else None

    # --- ENHANCED MATCHING LOGIC ---
    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        sport_map = {'nba': 'KXNBA', 'nfl': 'KXNFL', 'nhl': 'KXNHL', 'mlb': 'KXMLB'}
        series_ticker = sport_map.get(sport.lower(), "KXNBA")
        markets = self.get_todays_events(series_ticker)
        
        # Normalize and get Codes
        home_norm = TeamNameMatcher.normalize(home_team).upper()
        away_norm = TeamNameMatcher.normalize(away_team).upper()
        
        home_code = TEAM_TO_TICKER.get(home_norm)
        away_code = TEAM_TO_TICKER.get(away_norm)
        
        best_market = None
        best_score = 0.0
        match_type = "none"

        for m in markets:
            event_ticker = str(m.get("event_ticker", "")).upper()
            market_text = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
            
            # 1. Ticker Code Match (Highest Priority)
            if home_code and away_code:
                if f"-{home_code}-" in event_ticker and f"-{away_code}" in event_ticker:
                    best_market = m
                    best_score = 1.0
                    match_type = "ticker_code"
                    break
                # Try reversed
                if f"-{away_code}-" in event_ticker and f"-{home_code}" in event_ticker:
                    best_market = m
                    best_score = 1.0
                    match_type = "ticker_code_rev"
                    break

            # 2. Fuzzy Text Match (Fallback)
            if best_score < 1.0:
                event_title = m.get("event_title", "").lower()
                h_score = TeamNameMatcher.similarity_score(home_norm.lower(), event_title)
                a_score = TeamNameMatcher.similarity_score(away_norm.lower(), event_title)
                
                if h_score > 0.4 and a_score > 0.4:
                    avg = (h_score + a_score) / 2
                    if "winner" in market_text: avg += 0.1 # Boost winner markets
                    if avg > best_score:
                        best_score = avg
                        best_market = m
                        match_type = "fuzzy"

        res = {
            "kalshi_available": False, 
            "kalshi_prob": None, 
            "kalshi_match_debug": f"No match (Codes: {home_code}/{away_code})",
            "kalshi_label": None
        }

        if best_market and best_score > 0.55:
            yes_price = price_to_prob(best_market.get("yes_bid", 0))
            subtitle = best_market.get("subtitle", "").lower()
            
            # Logic for probability direction
            is_home_sub = home_norm.lower() in subtitle
            is_away_sub = away_norm.lower() in subtitle
            
            final_prob = yes_price
            if is_away_sub and not is_home_sub and yes_price:
                final_prob = 1.0 - yes_price

            res.update({
                "kalshi_available": True,
                "kalshi_prob": final_prob,
                "kalshi_label": best_market.get("event_title"),
                "kalshi_match_debug": f"Match: {match_type}",
                "market_ticker": best_market.get("ticker")
            })
            
        return res

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
