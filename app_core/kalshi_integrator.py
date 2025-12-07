"""
Kalshi Integrator v6.4: Fixed League-Specific Matching
Key fixes:
1. League-specific team codes to prevent collisions
2. Improved ticker matching logic
3. Better normalization consistency
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict

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

CORE_SERIES = ["KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXNCAAF", "KXNCAAB"]

# League-specific team mappings to prevent code collisions
KALSHI_ABBREVIATIONS = {
    # NBA Teams
    "NBA": {
        "ATL": ["ATLANTA HAWKS", "ATLANTA", "HAWKS"],
        "BOS": ["BOSTON CELTICS", "BOSTON", "CELTICS"],
        "BKN": ["BROOKLYN NETS", "BROOKLYN", "NETS"],
        "CHA": ["CHARLOTTE HORNETS", "CHARLOTTE", "HORNETS"],
        "CHI": ["CHICAGO BULLS", "BULLS"],
        "CLE": ["CLEVELAND CAVALIERS", "CLEVELAND", "CAVS", "CAVALIERS"],
        "DAL": ["DALLAS MAVERICKS", "DALLAS", "MAVS", "MAVERICKS"],
        "DEN": ["DENVER NUGGETS", "DENVER", "NUGGETS"],
        "DET": ["DETROIT PISTONS", "DETROIT", "PISTONS"],
        "GSW": ["GOLDEN STATE WARRIORS", "GOLDEN STATE", "GS", "WARRIORS"],
        "HOU": ["HOUSTON ROCKETS", "HOUSTON", "ROCKETS"],
        "IND": ["INDIANA PACERS", "INDIANA", "PACERS"],
        "LAC": ["LOS ANGELES CLIPPERS", "LA CLIPPERS", "CLIPPERS"],
        "LAL": ["LOS ANGELES LAKERS", "LA LAKERS", "LAKERS"],
        "MEM": ["MEMPHIS GRIZZLIES", "MEMPHIS", "GRIZZLIES"],
        "MIA": ["MIAMI HEAT", "MIAMI", "HEAT"],
        "MIL": ["MILWAUKEE BUCKS", "MILWAUKEE", "BUCKS"],
        "MIN": ["MINNESOTA TIMBERWOLVES", "MINNESOTA", "TIMBERWOLVES", "WOLVES"],
        "NOP": ["NEW ORLEANS PELICANS", "NEW ORLEANS", "NO", "PELICANS"],
        "NYK": ["NEW YORK KNICKS", "NEW YORK", "KNICKS"],
        "OKC": ["OKLAHOMA CITY THUNDER", "OKLAHOMA CITY", "THUNDER"],
        "ORL": ["ORLANDO MAGIC", "ORLANDO", "MAGIC"],
        "PHI": ["PHILADELPHIA 76ERS", "PHILADELPHIA", "PHILLY", "76ERS", "SIXERS"],
        "PHX": ["PHOENIX SUNS", "PHOENIX", "SUNS"],
        "POR": ["PORTLAND TRAIL BLAZERS", "PORTLAND", "BLAZERS"],
        "SAC": ["SACRAMENTO KINGS", "SACRAMENTO", "KINGS"],
        "SAS": ["SAN ANTONIO SPURS", "SAN ANTONIO", "SPURS"],
        "TOR": ["TORONTO RAPTORS", "TORONTO", "RAPTORS"],
        "UTA": ["UTAH JAZZ", "UTAH", "JAZZ"],
        "WAS": ["WASHINGTON WIZARDS", "WASHINGTON", "WIZARDS"]
    },
    # NFL Teams
    "NFL": {
        "ARI": ["ARIZONA CARDINALS", "CARDINALS"],
        "ATL": ["ATLANTA FALCONS", "FALCONS"],
        "BAL": ["BALTIMORE RAVENS", "RAVENS"],
        "BUF": ["BUFFALO BILLS", "BILLS"],
        "CAR": ["CAROLINA PANTHERS", "PANTHERS"],
        "CHI": ["CHICAGO BEARS", "BEARS"],
        "CIN": ["CINCINNATI BENGALS", "BENGALS"],
        "CLE": ["CLEVELAND BROWNS", "BROWNS"],
        "DAL": ["DALLAS COWBOYS", "COWBOYS"],
        "DEN": ["DENVER BRONCOS", "BRONCOS"],
        "DET": ["DETROIT LIONS", "LIONS"],
        "GB": ["GREEN BAY PACKERS", "PACKERS"],
        "HOU": ["HOUSTON TEXANS", "TEXANS"],
        "IND": ["INDIANAPOLIS COLTS", "COLTS"],
        "JAX": ["JACKSONVILLE JAGUARS", "JAGUARS"],
        "KC": ["KANSAS CITY CHIEFS", "CHIEFS"],
        "LV": ["LAS VEGAS RAIDERS", "RAIDERS"],
        "LAC": ["LOS ANGELES CHARGERS", "CHARGERS"],
        "LAR": ["LOS ANGELES RAMS", "RAMS"],
        "MIA": ["MIAMI DOLPHINS", "DOLPHINS"],
        "MIN": ["MINNESOTA VIKINGS", "VIKINGS"],
        "NE": ["NEW ENGLAND PATRIOTS", "PATRIOTS"],
        "NO": ["NEW ORLEANS SAINTS", "SAINTS"],
        "NYG": ["NEW YORK GIANTS", "GIANTS"],
        "NYJ": ["NEW YORK JETS", "JETS"],
        "PHI": ["PHILADELPHIA EAGLES", "EAGLES"],
        "PIT": ["PITTSBURGH STEELERS", "STEELERS"],
        "SF": ["SAN FRANCISCO 49ERS", "49ERS"],
        "SEA": ["SEATTLE SEAHAWKS", "SEAHAWKS"],
        "TB": ["TAMPA BAY BUCCANEERS", "BUCCANEERS"],
        "TEN": ["TENNESSEE TITANS", "TITANS"],
        "WSH": ["WASHINGTON COMMANDERS", "COMMANDERS", "WASHINGTON"]
    },
    # NHL Teams
    "NHL": {
        "ANA": ["ANAHEIM DUCKS", "DUCKS"],
        "ARI": ["ARIZONA COYOTES", "UTAH HOCKEY CLUB", "COYOTES"],
        "BOS": ["BOSTON BRUINS", "BRUINS"],
        "BUF": ["BUFFALO SABRES", "SABRES"],
        "CGY": ["CALGARY FLAMES", "FLAMES"],
        "CAR": ["CAROLINA HURRICANES", "HURRICANES"],
        "CHI": ["CHICAGO BLACKHAWKS", "BLACKHAWKS"],
        "COL": ["COLORADO AVALANCHE", "AVALANCHE"],
        "CBJ": ["COLUMBUS BLUE JACKETS", "BLUE JACKETS"],
        "DAL": ["DALLAS STARS", "STARS"],
        "DET": ["DETROIT RED WINGS", "RED WINGS"],
        "EDM": ["EDMONTON OILERS", "OILERS"],
        "FLA": ["FLORIDA PANTHERS", "PANTHERS"],
        "LAK": ["LOS ANGELES KINGS", "KINGS"],
        "MIN": ["MINNESOTA WILD", "WILD"],
        "MTL": ["MONTREAL CANADIENS", "CANADIENS"],
        "NSH": ["NASHVILLE PREDATORS", "PREDATORS"],
        "NJD": ["NEW JERSEY DEVILS", "DEVILS"],
        "NYI": ["NEW YORK ISLANDERS", "ISLANDERS"],
        "NYR": ["NEW YORK RANGERS", "RANGERS"],
        "OTT": ["OTTAWA SENATORS", "SENATORS"],
        "PHI": ["PHILADELPHIA FLYERS", "FLYERS"],
        "PIT": ["PITTSBURGH PENGUINS", "PENGUINS"],
        "SJS": ["SAN JOSE SHARKS", "SHARKS"],
        "SEA": ["SEATTLE KRAKEN", "KRAKEN"],
        "STL": ["ST LOUIS BLUES", "BLUES"],
        "TBL": ["TAMPA BAY LIGHTNING", "LIGHTNING"],
        "TOR": ["TORONTO MAPLE LEAFS", "MAPLE LEAFS"],
        "VAN": ["VANCOUVER CANUCKS", "CANUCKS"],
        "VGK": ["VEGAS GOLDEN KNIGHTS", "GOLDEN KNIGHTS"],
        "WSH": ["WASHINGTON CAPITALS", "CAPITALS"],
        "WPG": ["WINNIPEG JETS", "JETS"]
    }
}

# Build league-specific reverse lookup maps
TEAM_TO_TICKER: Dict[str, Dict[str, str]] = {}

def _build_reverse_lookups():
    """Build reverse lookup dictionaries for each league"""
    for league, teams in KALSHI_ABBREVIATIONS.items():
        TEAM_TO_TICKER[league] = {}
        for code, names in teams.items():
            for name in names:
                if not name:
                    continue
                normalized = name.upper().strip()
                # Remove punctuation but keep spaces for mascot matching
                clean = re.sub(r"[^A-Z\s]", "", normalized).strip()
                
                # Add both versions as keys
                for key in [normalized, clean]:
                    if key and key not in TEAM_TO_TICKER[league]:
                        TEAM_TO_TICKER[league][key] = code

_build_reverse_lookups()

class KalshiMatchResult(TypedDict, total=False):
    matched: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: str
    reason: str

def price_to_prob(price) -> Optional[float]:
    if price is None or price == "":
        return None
    try:
        p = float(price)
        if p > 1.01:
            p = p / 100.0
        return max(0.0, min(1.0, p))
    except:
        return None

def _parse_market_date(raw) -> Optional[datetime]:
    if not raw:
        return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw) / 1000.0, tz=pytz.UTC)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except:
        return None

class KalshiIntegrator:
    def __init__(self, api_key: str = None, api_secret: str = None):
        if not api_key:
            try:
                api_key = st.secrets.get("KALSHI_API_KEY")
                api_secret = st.secrets.get("KALSHI_API_SECRET")
            except:
                pass

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
            if not data or "events" not in data:
                continue

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

    def get_sports_markets(self):
        return self.get_todays_events()
    
    def get_game_markets_for_events(self, league="NBA"):
        sport_map = {'NBA': 'KXNBA', 'NFL': 'KXNFL', 'NHL': 'KXNHL', 'MLB': 'KXMLB'}
        return self.get_todays_events(sport_map.get(league.upper(), 'KXNBA'))
    
    def filter_markets_closing_today(self, markets):
        if not markets:
            return []
        today = datetime.now(pytz.UTC).date()
        return [m for m in markets if _parse_market_date(m.get("close_time")) and 
                abs((_parse_market_date(m.get("close_time")).date() - today).days) <= 1]
    
    def get_orderbook(self, ticker):
        data = self._make_public_request(f"/markets/{ticker}/orderbook", {"depth": 5})
        return data.get('orderbook') if data else None

    def _lookup_team_code(self, team_name: str, league: str) -> Optional[str]:
        """
        Look up the Kalshi ticker code for a team name in a specific league.
        
        Args:
            team_name: The team name to look up
            league: The league (NBA, NFL, NHL, etc.)
        
        Returns:
            The ticker code or None if not found
        """
        if not team_name or league not in TEAM_TO_TICKER:
            return None
        
        # Normalize the team name
        normalized = team_name.upper().strip()
        clean = re.sub(r"[^A-Z\s]", "", normalized).strip()
        
        lookup_dict = TEAM_TO_TICKER[league]
        
        # Try exact matches first
        for key in [normalized, clean]:
            if key in lookup_dict:
                return lookup_dict[key]
        
        # Try partial matches (for cases like "LA Lakers" vs "Los Angeles Lakers")
        for key, code in lookup_dict.items():
            if key in normalized or normalized in key:
                return code
        
        return None

    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        """
        Match a game to a Kalshi market using improved league-specific matching.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            sport: Sport/league identifier
            game_time: Optional game time
        
        Returns:
            Dict with kalshi_available, kalshi_prob, and debug info
        """
        sport_map = {
            'nba': ('KXNBA', 'NBA'),
            'basketball_nba': ('KXNBA', 'NBA'),
            'nfl': ('KXNFL', 'NFL'),
            'americanfootball_nfl': ('KXNFL', 'NFL'),
            'nhl': ('KXNHL', 'NHL'),
            'icehockey_nhl': ('KXNHL', 'NHL'),
            'mlb': ('KXMLB', 'MLB'),
            'baseball_mlb': ('KXMLB', 'MLB'),
            'ncaaf': ('KXNCAAF', 'NCAAF'),
            'americanfootball_ncaaf': ('KXNCAAF', 'NCAAF'),
            'ncaab': ('KXNCAAB', 'NCAAB'),
            'basketball_ncaab': ('KXNCAAB', 'NCAAB')
        }
        
        series_ticker, league = sport_map.get(sport.lower(), ("KXNBA", "NBA"))
        markets = self.get_todays_events(series_ticker)
        
        # Look up team codes using league-specific mapping
        home_code = self._lookup_team_code(home_team, league)
        away_code = self._lookup_team_code(away_team, league)
        
        logger.info(f"Matching {home_team} vs {away_team} in {league}")
        logger.info(f"Found codes: home={home_code}, away={away_code}")
        
        # Normalize team names for text matching
        home_norm = TeamNameMatcher.normalize(home_team).lower()
        away_norm = TeamNameMatcher.normalize(away_team).lower()
        
        best_market = None
        best_score = 0.0
        match_type = "none"
        
        # Create comprehensive alias sets
        home_aliases = {home_team.lower().strip(), home_norm}
        away_aliases = {away_team.lower().strip(), away_norm}
        
        if home_code and league in KALSHI_ABBREVIATIONS and home_code in KALSHI_ABBREVIATIONS[league]:
            home_aliases.update(n.lower() for n in KALSHI_ABBREVIATIONS[league][home_code])
        if away_code and league in KALSHI_ABBREVIATIONS and away_code in KALSHI_ABBREVIATIONS[league]:
            away_aliases.update(n.lower() for n in KALSHI_ABBREVIATIONS[league][away_code])
        
        for m in markets:
            event_ticker = str(m.get("event_ticker", "")).upper()
            title = m.get("title", "").lower()
            subtitle = m.get("subtitle", "").lower()
            event_title = m.get("event_title", "").lower()
            
            # A. TICKER CODE MATCH (Highest Priority)
            if home_code and away_code:
                # Check if both team codes appear in the event ticker
                # Common formats: KXNBA-2024-12-06-LAL-GSW or similar
                if home_code in event_ticker and away_code in event_ticker:
                    best_market = m
                    best_score = 1.0
                    match_type = "ticker_code"
                    logger.info(f"✓ Ticker match: {event_ticker}")
                    break
            
            # B. TITLE + SUBTITLE CONTAINMENT (High Priority)
            full_text = f"{event_title} {title} {subtitle}"
            contains_home = any(alias in full_text for alias in home_aliases if alias)
            contains_away = any(alias in full_text for alias in away_aliases if alias)
            
            if contains_home and contains_away:
                if best_score < 0.95:
                    best_market = m
                    best_score = 0.95
                    match_type = "text_contains"
                    logger.info(f"✓ Text match: {event_title}")
                continue
            
            # C. FUZZY MATCH (Fallback)
            if best_score < 0.9:
                h_score = TeamNameMatcher.similarity_score(home_norm, full_text)
                a_score = TeamNameMatcher.similarity_score(away_norm, full_text)
                
                if h_score > 0.5 and a_score > 0.5:
                    avg = (h_score + a_score) / 2
                    if avg > best_score:
                        best_score = avg
                        best_market = m
                        match_type = f"fuzzy({h_score:.2f},{a_score:.2f})"
        
        # Build result
        res = {
            "kalshi_available": False,
            "kalshi_prob": None,
            "kalshi_match_debug": f"No match (League: {league}, Codes: {home_code}/{away_code})",
            "kalshi_label": None
        }
        
        if best_market and best_score > 0.5:
            # Extract probability
            yes_price = price_to_prob(best_market.get("yes_bid"))
            if yes_price is None:
                yes_price = price_to_prob(best_market.get("last_price") or best_market.get("yes_ask"))
            if yes_price is None:
                yes_price = 0.5
            
            # Determine if we need to invert based on subtitle
            subtitle = best_market.get("subtitle", "").lower()
            final_prob = yes_price
            
            # If the market subtitle mentions the away team but not home, invert
            subtitle_has_away = any(alias in subtitle for alias in away_aliases if alias)
            subtitle_has_home = any(alias in subtitle for alias in home_aliases if alias)
            
            if subtitle_has_away and not subtitle_has_home:
                final_prob = 1.0 - yes_price
            
            res.update({
                "kalshi_available": True,
                "kalshi_prob": final_prob,
                "kalshi_label": best_market.get("event_title", best_market.get("title")),
                "kalshi_match_debug": f"✓ {match_type} (score={best_score:.2f})",
                "market_ticker": best_market.get("ticker"),
                "kalshi_volume": best_market.get("volume")
            })
            logger.info(f"Match successful: {res['kalshi_label']} - {final_prob:.3f}")
        else:
            logger.warning(f"No match found for {home_team} vs {away_team}")
        
        return res

def match_game_to_kalshi(league, home, away, time, integrator=None, status="open"):
    """Legacy wrapper function for compatibility"""
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
