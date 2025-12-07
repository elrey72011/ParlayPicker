"""
Kalshi Integrator v6.3: Robust Ticker Matching Fix
This file goes in: app_core/kalshi_integrator.py
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import pytz
import requests
import streamlit as st

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
    "WSH": ["WASHINGTON COMMANDERS", "WASHINGTON"],
    # NHL
    "ANA": ["ANAHEIM DUCKS"], "ARI": ["ARIZONA COYOTES", "UTAH HOCKEY CLUB"], "BOS": ["BOSTON BRUINS"],
    "BUF": ["BUFFALO SABRES"], "CGY": ["CALGARY FLAMES"], "CAR": ["CAROLINA HURRICANES"],
    "CHI": ["CHICAGO BLACKHAWKS"], "COL": ["COLORADO AVALANCHE"], "CBJ": ["COLUMBUS BLUE JACKETS"],
    "DAL": ["DALLAS STARS"], "DET": ["DETROIT RED WINGS"], "EDM": ["EDMONTON OILERS"],
    "FLA": ["FLORIDA PANTHERS"], "LAK": ["LOS ANGELES KINGS"], "MIN": ["MINNESOTA WILD"],
    "MTL": ["MONTREAL CANADIENS"], "NSH": ["NASHVILLE PREDATORS"], "NJD": ["NEW JERSEY DEVILS"],
    "NYI": ["NEW YORK ISLANDERS"], "NYR": ["NEW YORK RANGERS"], "OTT": ["OTTAWA SENATORS"],
    "PHI": ["PHILADELPHIA FLYERS"], "PIT": ["PITTSBURGH PENGUINS"], "SJS": ["SAN JOSE SHARKS"],
    "SEA": ["SEATTLE KRAKEN"], "STL": ["ST LOUIS BLUES"], "TBL": ["TAMPA BAY LIGHTNING"],
    "TOR": ["TORONTO MAPLE LEAFS"], "VAN": ["VANCOUVER CANUCKS"], "VGK": ["VEGAS GOLDEN KNIGHTS"],
    "WSH": ["WASHINGTON CAPITALS"], "WPG": ["WINNIPEG JETS"]
}

# Build Reverse Lookup Map (Team Name -> Ticker)
TEAM_TO_TICKER: Dict[str, str] = {}

def _add_team_alias(name: str, code: str) -> None:
    if not name: return
    normalized = TeamNameMatcher.normalize(name).upper()
    mascot_preserving = re.sub(r"[^A-Z\s]", "", name.upper()).strip()
    for key in filter(None, {normalized, mascot_preserving}):
        TEAM_TO_TICKER.setdefault(key, code)

for code, names in KALSHI_ABBREVIATIONS.items():
    for n in names:
        _add_team_alias(n, code)

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
        self._markets_cache = {}
        self.last_error = None

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
        start_window = now - (24 * 60 * 60)
        future_window = now + (7 * 24 * 60 * 60)
        target_series = [sport_ticker] if sport_ticker else CORE_SERIES
        cache_key = f"events_{sport_ticker or 'all'}_{now // 300}" 
        
        if cache_key in self._markets_cache:
            return self._markets_cache[cache_key]

        all_events = []
        seen_market_tickers = set()
        for series in target_series:
            params = {
                "series_ticker": series,
                "min_close_ts": start_window,
                "max_close_ts": future_window,
                "with_nested_markets": "true",
                "limit": 100
            }
            data = self._make_public_request("/events", params)
            if not data or "events" not in data: continue

            for e in data.get("events", []):
                for m in e.get("markets", []):
                    ticker = m.get("ticker")
                    if ticker and ticker not in seen_market_tickers:
                        seen_market_tickers.add(ticker)
                        m['event_title'] = e.get('title')
                        m['event_ticker'] = e.get('event_ticker')
                        m['series'] = series
                        all_events.append(m)
        
        self._markets_cache[cache_key] = all_events
        return all_events

    def get_sports_markets(self): return self.get_todays_events()
    def get_game_markets_for_events(self, league="NBA"):
        sport_map = {'NBA': 'KXNBA', 'NFL': 'KXNFL', 'NHL': 'KXNHL', 'MLB': 'KXMLB'}
        return self.get_todays_events(sport_map.get(league.upper(), 'KXNBA'))
    def filter_markets_closing_today(self, markets):
        if not markets: return []
        today = datetime.now(pytz.UTC).date()
        return [m for m in markets if _parse_market_date(m.get("close_time")) and abs((_parse_market_date(m.get("close_time")).date() - today).days) <= 1]
    def get_orderbook(self, ticker):
        data = self._make_public_request(f"/markets/{ticker}/orderbook", {"depth": 5})
        return data.get('orderbook') if data else None

    # --- ENHANCED MATCHING LOGIC ---
    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        sport_map = {
            'nba': 'KXNBA', 'basketball_nba': 'KXNBA',
            'nfl': 'KXNFL', 'americanfootball_nfl': 'KXNFL',
            'nhl': 'KXNHL', 'icehockey_nhl': 'KXNHL',
            'mlb': 'KXMLB', 'baseball_mlb': 'KXMLB',
            'ncaaf': 'KXNCAAF', 'americanfootball_ncaaf': 'KXNCAAF',
            'ncaab': 'KXNCAAB', 'basketball_ncaab': 'KXNCAAB'
        }
        series_ticker = sport_map.get(sport.lower(), "KXNBA")
        markets = self.get_todays_events(series_ticker)
        
        home_norm = TeamNameMatcher.normalize(home_team).upper()
        away_norm = TeamNameMatcher.normalize(away_team).upper()

        def _lookup_team_code(name: str, normalized: str) -> Optional[str]:
            mascot_preserving = re.sub(r"[^A-Z\s]", "", (name or "").upper()).strip()
            for key in (mascot_preserving, normalized):
                if key and key in TEAM_TO_TICKER: return TEAM_TO_TICKER[key]
            return None

        home_code = _lookup_team_code(home_team, home_norm)
        away_code = _lookup_team_code(away_team, away_norm)
        
        best_market = None
        best_score = 0.0
        match_type = "none"

        # Helper to check ticker containment
        def _ticker_contains(event_ticker, code):
            return f"-{code}-" in event_ticker or event_ticker.endswith(f"-{code}")

        home_aliases = {home_team.lower().strip(), home_norm.lower().strip()} - {""}
        away_aliases = {away_team.lower().strip(), away_norm.lower().strip()} - {""}

        for m in markets:
            event_ticker = str(m.get("event_ticker", "")).upper()
            market_text = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
            
            # A. TICKER MATCH (Improved)
            if home_code and away_code:
                if _ticker_contains(event_ticker, home_code) and _ticker_contains(event_ticker, away_code):
                    best_market = m
                    best_score = 1.0
                    match_type = "ticker_code"
                    break

            # B. TITLE CONTAINMENT
            title_text = (m.get("event_title", "") + " " + market_text).lower()
            contains_home = any(alias in title_text for alias in home_aliases)
            contains_away = any(alias in title_text for alias in away_aliases)

            if contains_home and contains_away:
                best_market = m
                best_score = 0.9
                match_type = "title_contains"
                continue

            # C. FUZZY MATCH
            if best_score < 1.0:
                event_title = m.get("event_title", "").lower()
                full_text = event_title + " " + market_text
                h_score = TeamNameMatcher.similarity_score(home_norm.lower(), full_text)
                a_score = TeamNameMatcher.similarity_score(away_norm.lower(), full_text)
                
                if h_score > 0.45 and a_score > 0.45:
                    avg = (h_score + a_score) / 2
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

        if best_market and best_score > 0.5:
            yes_price = price_to_prob(best_market.get("yes_bid"))
            if yes_price is None:
                yes_price = price_to_prob(best_market.get("last_price") or best_market.get("yes_ask"))
            if yes_price is None:
                yes_price = 0.5

            subtitle = best_market.get("subtitle", "").lower()
            # Invert if the market is for the Away team
            final_prob = yes_price
            if away_norm.lower() in subtitle and home_norm.lower() not in subtitle:
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
