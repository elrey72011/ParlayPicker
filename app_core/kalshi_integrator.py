"""
Kalshi Integrator v6.1: Abbreviation & Ticker Matching
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
CORE_SERIES = ["KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXNCAAF", "KXNCAAB"]

# Map Full Team Names to Kalshi Ticker Codes
KALSHI_ABBREVIATIONS = {
    # NBA
    "ATL": ["ATLANTA HAWKS", "ATLANTA"],
    "BOS": ["BOSTON CELTICS", "BOSTON"],
    "BKN": ["BROOKLYN NETS", "BROOKLYN"],
    "CHA": ["CHARLOTTE HORNETS", "CHARLOTTE"],
    "CHI": ["CHICAGO BULLS", "CHICAGO"],
    "CLE": ["CLEVELAND CAVALIERS", "CLEVELAND"],
    "DAL": ["DALLAS MAVERICKS", "DALLAS"],
    "DEN": ["DENVER NUGGETS", "DENVER"],
    "DET": ["DETROIT PISTONS", "DETROIT"],
    "GSW": ["GOLDEN STATE WARRIORS", "GOLDEN STATE", "GS"],
    "HOU": ["HOUSTON ROCKETS", "HOUSTON"],
    "IND": ["INDIANA PACERS", "INDIANA"],
    "LAC": ["LOS ANGELES CLIPPERS", "LA CLIPPERS"],
    "LAL": ["LOS ANGELES LAKERS", "LA LAKERS"],
    "MEM": ["MEMPHIS GRIZZLIES", "MEMPHIS"],
    "MIA": ["MIAMI HEAT", "MIAMI"],
    "MIL": ["MILWAUKEE BUCKS", "MILWAUKEE"],
    "MIN": ["MINNESOTA TIMBERWOLVES", "MINNESOTA"],
    "NOP": ["NEW ORLEANS PELICANS", "NEW ORLEANS", "NO"],
    "NYK": ["NEW YORK KNICKS", "NEW YORK"],
    "OKC": ["OKLAHOMA CITY THUNDER", "OKLAHOMA CITY"],
    "ORL": ["ORLANDO MAGIC", "ORLANDO"],
    "PHI": ["PHILADELPHIA 76ERS", "PHILADELPHIA", "PHILLY"],
    "PHX": ["PHOENIX SUNS", "PHOENIX"],
    "POR": ["PORTLAND TRAIL BLAZERS", "PORTLAND"],
    "SAC": ["SACRAMENTO KINGS", "SACRAMENTO"],
    "SAS": ["SAN ANTONIO SPURS", "SAN ANTONIO"],
    "TOR": ["TORONTO RAPTORS", "TORONTO"],
    "UTA": ["UTAH JAZZ", "UTAH"],
    "WAS": ["WASHINGTON WIZARDS", "WASHINGTON"],
    # NFL
    "ARI": ["ARIZONA CARDINALS", "ARIZONA"],
    "BAL": ["BALTIMORE RAVENS", "BALTIMORE"],
    "BUF": ["BUFFALO BILLS", "BUFFALO"],
    "CAR": ["CAROLINA PANTHERS", "CAROLINA"],
    "CIN": ["CINCINNATI BENGALS", "CINCINNATI"],
    "GB": ["GREEN BAY PACKERS", "GREEN BAY"],
    "JAX": ["JACKSONVILLE JAGUARS", "JACKSONVILLE"],
    "KC": ["KANSAS CITY CHIEFS", "KANSAS CITY"],
    "LV": ["LAS VEGAS RAIDERS", "LAS VEGAS"],
    "LAR": ["LOS ANGELES RAMS", "LA RAMS"],
    "NE": ["NEW ENGLAND PATRIOTS", "NEW ENGLAND"],
    "NO": ["NEW ORLEANS SAINTS", "NEW ORLEANS"],
    "NYG": ["NEW YORK GIANTS"],
    "NYJ": ["NEW YORK JETS"],
    "PIT": ["PITTSBURGH STEELERS", "PITTSBURGH"],
    "SF": ["SAN FRANCISCO 49ERS", "SAN FRANCISCO"],
    "SEA": ["SEATTLE SEAHAWKS", "SEATTLE"],
    "TB": ["TAMPA BAY BUCCANEERS", "TAMPA BAY"],
    "TEN": ["TENNESSEE TITANS", "TENNESSEE"],
    "WSH": ["WASHINGTON COMMANDERS", "WASHINGTON"],
}

# Reverse map for lookup: "Golden State Warriors" -> "GSW"
TEAM_TO_TICKER = {}
for code, names in KALSHI_ABBREVIATIONS.items():
    for n in names:
        TEAM_TO_TICKER[n.upper()] = code
        # Also map normalized version
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

class KalshiIntegrator:
    def __init__(self, api_key: str = None, api_secret: str = None):
        if not api_key:
            try:
                api_key = st.secrets.get("KALSHI_API_KEY")
                api_secret = st.secrets.get("KALSHI_API_SECRET")
            except: pass

        self.api_key = api_key
        self.api_secret = api_secret
        self.prod_url = "https://trading-api.kalshi.com/trade-api/v2"
        self.public_url = "https://api.elections.kalshi.com/trade-api/v2"
        
        self.api_url = self.prod_url if (self.api_key and self.api_secret) else self.public_url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
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

    def _sign_request(self, method, path, timestamp):
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

    def _make_public_request(self, endpoint, params=None):
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
                return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def get_todays_events(self, sport_ticker=None):
        now = int(time.time())
        future = now + (48 * 60 * 60)
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
                # Add series context to markets
                for e in data["events"]:
                    for m in e.get("markets", []):
                        m['event_title'] = e.get('title')
                        m['event_ticker'] = e.get('event_ticker') # Critical for matching
                        m['series'] = series
                        all_events.append(m)
        
        self._markets_cache[cache_key] = all_events
        return all_events

    def get_sports_markets(self):
        return self.get_todays_events()

    def get_game_markets_for_events(self, league="NBA"):
        sport_map = {'NBA': 'KXNBA', 'NFL': 'KXNFL', 'NHL': 'KXNHL', 'MLB': 'KXMLB'}
        ticker = sport_map.get(league, 'KXNBA')
        return self.get_todays_events(ticker)

    def filter_markets_closing_today(self, markets):
        if not markets: return []
        today = datetime.now(pytz.UTC).date()
        # Accept markets closing today OR tomorrow (timezone safety)
        return [m for m in markets if _parse_market_date(m.get("close_time")) and abs((_parse_market_date(m.get("close_time")).date() - today).days) <= 1]

    def get_orderbook(self, ticker):
        data = self._make_public_request(f"/markets/{ticker}/orderbook", {"depth": 5})
        return data.get('orderbook') if data else None

    # --- ENHANCED MATCHING LOGIC ---
    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        sport_map = {'nba': 'KXNBA', 'nfl': 'KXNFL', 'nhl': 'KXNHL', 'mlb': 'KXMLB'}
        series_ticker = sport_map.get(sport.lower(), "KXNBA")
        markets = self.get_todays_events(series_ticker)
        
        # 1. Look up abbreviations (e.g. "Golden State" -> "GSW")
        home_norm = TeamNameMatcher.normalize(home_team).upper()
        away_norm = TeamNameMatcher.normalize(away_team).upper()
        
        home_code = TEAM_TO_TICKER.get(home_norm)
        away_code = TEAM_TO_TICKER.get(away_norm)
        
        best_market = None
        best_score = 0.0
        match_type = "none"

        for m in markets:
            event_ticker = str(m.get("event_ticker", "")).upper() # e.g. KXNBA-23NOV-GSW-LAL
            market_text = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
            
            # --- STRATEGY A: Ticker Code Match (Most Accurate) ---
            if home_code and away_code:
                if f"-{home_code}-" in event_ticker and f"-{away_code}" in event_ticker:
                    best_market = m
                    best_score = 1.0
                    match_type = "ticker_code"
                    break # Exact match found
                
                # Try reversed ticker just in case
                if f"-{away_code}-" in event_ticker and f"-{home_code}" in event_ticker:
                    best_market = m
                    best_score = 1.0
                    match_type = "ticker_code_rev"
                    break

            # --- STRATEGY B: Fuzzy Text Match (Fallback) ---
            if best_score < 1.0:
                event_title = m.get("event_title", "").lower()
                # Use normalized names for similarity
                h_score = TeamNameMatcher.similarity_score(home_norm.lower(), event_title + market_text)
                a_score = TeamNameMatcher.similarity_score(away_norm.lower(), event_title + market_text)
                
                # Check for "Winner" or "Win" to prioritize game lines over props
                is_winner_market = "win" in market_text
                
                if h_score > 0.45 and a_score > 0.45:
                    avg = (h_score + a_score) / 2
                    if is_winner_market: avg += 0.15 # Boost Winner markets
                    
                    if avg > best_score:
                        best_score = avg
                        best_market = m
                        match_type = "fuzzy"

        # Prepare Result
        res = {
            "kalshi_available": False, 
            "kalshi_prob": None, 
            "kalshi_match_debug": f"No match (Codes: {home_code}/{away_code})",
            "kalshi_label": None
        }

        if best_market and best_score > 0.55:
            # Determine Probability
            yes_price = price_to_prob(best_market.get("yes_bid", 0))
            subtitle = best_market.get("subtitle", "").lower()
            
            # Logic: If subtitle contains Home Team Name, prob = Yes Price
            # If subtitle contains Away Team Name, prob = 1 - Yes Price
            is_home_sub = home_norm.lower() in subtitle
            is_away_sub = away_norm.lower() in subtitle
            
            final_prob = yes_price
            
            # If ticker match, we can be more precise
            if match_type.startswith("ticker"):
                # If market title is just the home team code? 
                # Usually Kalshi subtitle is "Golden State to win"
                pass
            
            # Fallback logic for binary markets
            if is_away_sub and not is_home_sub and yes_price:
                final_prob = 1.0 - yes_price

            res.update({
                "kalshi_available": True,
                "kalshi_prob": final_prob,
                "kalshi_label": best_market.get("event_title", best_market.get("title")),
                "kalshi_match_debug": f"Match: {match_type} ({best_score:.2f})",
                "market_ticker": best_market.get("ticker"),
                "kalshi_volume": best_market.get("volume")
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
