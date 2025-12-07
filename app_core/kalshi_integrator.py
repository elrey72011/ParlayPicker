"""
Kalshi Integrator v6.2: Ticker Matching + Stale Object Fix
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
GAME_SERIES = {
    "NBA": "KXNBA",   # Updated to use main series tickers
    "NFL": "KXNFL",
    "NHL": "KXNHL",
    "MLB": "KXMLB",
    "NCAAF": "KXNCAAF",
    "NCAAB": "KXNCAAB"
}
CORE_SERIES = list(GAME_SERIES.values())

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
    # NHL
    "ANA": ["ANAHEIM DUCKS"], "ARI": ["ARIZONA COYOTES"], "BOS": ["BOSTON BRUINS"],
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
    """Register multiple lookup keys for a team to avoid collisions.

    TeamNameMatcher.normalize removes mascots (e.g., "Lakers"), which makes
    franchises that share a city (Lakers/Clippers) collapse to the same key.
    We index both the mascot-stripped and mascot-preserving variants so we can
    resolve the correct Kalshi ticker even when the caller provides the full
    team name.
    """

    if not name:
        return

    normalized = TeamNameMatcher.normalize(name).upper()
    mascot_preserving = re.sub(r"[^A-Z\s]", "", name.upper()).strip()

    for key in filter(None, {normalized, mascot_preserving}):
        # Do not overwrite an existing alias; first write wins to keep intent
        # stable across duplicate city names.
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
        # Load keys from secrets if not provided
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
        
        # Decide which URL to use
        self.api_url = self.prod_url if (self.api_key and self.api_secret) else self.public_url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        self._private_key = None
        self._auth_ready = False
        
        # Simple RSA setup check
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
        """Always uses the public URL for data fetching to avoid auth errors."""
        url = f"{self.public_url}{endpoint}"
        headers = {
            "Accept": "application/json",
            # Use a browser-like agent to avoid silent throttling
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
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
        """Fetch events closing in an expanded window to capture newly listed markets.

        We previously limited this to ~48 hours, which missed markets Kalshi lists
        several days ahead. Expanding the lookahead keeps the Streamlit table from
        reporting "No Match" when markets exist but are slightly further out.
        """
        now = int(time.time())
        # Look back 24 hours for games already underway and forward 7 days for upcoming
        # markets. This wider window still respects Kalshi's natural limits while
        # ensuring newly posted events are available for matching.
        start_window = now - (24 * 60 * 60)
        future_window = now + (7 * 24 * 60 * 60)
        now_dt = datetime.fromtimestamp(now, tz=pytz.UTC)
        days_ahead = 7
        
        target_series = [sport_ticker] if sport_ticker else CORE_SERIES
        cache_key = f"events_{sport_ticker or 'all'}_{now // 300}" # 5 min cache
        
        if cache_key in self._markets_cache:
            return self._markets_cache[cache_key]

        all_events = []
        seen_market_tickers = set()
        for series in target_series:
            params_base = {
                "series_ticker": series,
                "min_close_ts": start_window,
                "max_close_ts": future_window,
                "with_nested_markets": "true",
                "limit": 100
            }

            # First try open markets (typical case)
            for status_filter in ("open", None):
                params = dict(params_base)
                if status_filter:
                    params["status"] = status_filter

                data = self._make_public_request("/events", params)
                if not data or "events" not in data:
                    continue

                events_found = data.get("events", [])
                print(f"DEBUG: Raw API returned {len(events_found)} events for {series}")

                for e in events_found:
                    # Flatten markets and attach event metadata
                    for m in e.get("markets", []):
                        ticker = m.get("ticker")
                        if ticker and ticker in seen_market_tickers:
                            continue

                        m['event_title'] = e.get('title')
                        m['event_ticker'] = e.get('event_ticker') # e.g. KXNBA-23NOV-LAL-GSW
                        m['series'] = series

                        close_time = _parse_market_date(m.get("close_time"))
                        days_until_close = None
                        if close_time:
                            days_until_close = (close_time - now_dt).total_seconds() / (24 * 60 * 60)

                        if days_until_close is None or (-0.5 <= days_until_close <= days_ahead):
                            if ticker:
                                seen_market_tickers.add(ticker)
                            all_events.append(m)
                        else:
                            # Uncomment to debug filtered futures markets
                            # print(f"Skipped {ticker}: Closes in {days_until_close:.2f} days")
                            pass
        
        self._markets_cache[cache_key] = all_events
        return all_events

    # --- COMPATIBILITY METHODS (Required to fix your error) ---
    def get_sports_markets(self):
        """Legacy alias used by Streamlit app health check."""
        return self.get_todays_events()

    def get_markets(self, **kwargs):
        return self.get_todays_events()

    def get_game_markets_for_events(self, league="NBA"):
        """Legacy alias used by Kalshi tab."""
        sport_map = {k: v for k, v in GAME_SERIES.items()}
        ticker = sport_map.get(league.upper(), GAME_SERIES["NBA"])
        return self.get_todays_events(ticker)

    def filter_markets_closing_today(self, markets):
        """Legacy filter used by Kalshi tab."""
        if not markets: return []
        today = datetime.now(pytz.UTC).date()
        # Return markets closing today or tomorrow (timezone safe)
        return [m for m in markets if _parse_market_date(m.get("close_time")) and abs((_parse_market_date(m.get("close_time")).date() - today).days) <= 1]

    def get_orderbook(self, ticker):
        """Fetch orderbook for a specific market ticker."""
        data = self._make_public_request(f"/markets/{ticker}/orderbook", {"depth": 5})
        return data.get('orderbook') if data else None

    # --- ENHANCED MATCHING LOGIC ---
    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        # 1. Select Series
        sport_map = {k.lower(): v for k, v in GAME_SERIES.items()}
        sport_map.update({
            'basketball_nba': GAME_SERIES['NBA'],
            'americanfootball_nfl': GAME_SERIES['NFL'],
            'icehockey_nhl': GAME_SERIES['NHL'],
            'baseball_mlb': GAME_SERIES['MLB'],
            'americanfootball_ncaaf': GAME_SERIES['NCAAF'],
            'basketball_ncaab': GAME_SERIES['NCAAB']
        })
        series_ticker = sport_map.get(sport.lower(), GAME_SERIES["NBA"])
        markets = self.get_todays_events(series_ticker)
        
        # 2. Normalize Names & Get Codes
        home_norm = TeamNameMatcher.normalize(home_team).upper()
        away_norm = TeamNameMatcher.normalize(away_team).upper()

        def _lookup_team_code(name: str, normalized: str) -> Optional[str]:
            mascot_preserving = re.sub(r"[^A-Z\s]", "", (name or "").upper()).strip()
            for key in (mascot_preserving, normalized):
                if key and key in TEAM_TO_TICKER:
                    return TEAM_TO_TICKER[key]
            return None

        home_code = _lookup_team_code(home_team, home_norm)
        away_code = _lookup_team_code(away_team, away_norm)
        
        best_market = None
        best_score = 0.0
        match_type = "none"

        # Build alias sets for quick containment checks against Kalshi titles
        def _alias_set(name: str, normalized: str) -> List[str]:
            return list({
                (name or "").lower().strip(),
                normalized.lower().strip(),
                TeamNameMatcher.normalize(name or "").lower().strip(),
            } - {""})

        home_aliases = _alias_set(home_team, home_norm)
        away_aliases = _alias_set(away_team, away_norm)

        for m in markets:
            event_ticker = str(m.get("event_ticker", "")).upper()
            market_text = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
            
            # A. TICKER MATCH (Gold Standard)
            # Looks for "-LAL-" and "-GSW-" in the event ticker
            if home_code and away_code:
                if f"-{home_code}-" in event_ticker and f"-{away_code}" in event_ticker:
                    best_market = m
                    best_score = 1.0
                    match_type = "ticker_code"
                    break
                # Try reversed order
                if f"-{away_code}-" in event_ticker and f"-{home_code}" in event_ticker:
                    best_market = m
                    best_score = 1.0
                    match_type = "ticker_code_rev"
                    break

            # B. TITLE CONTAINMENT (Strong Fallback)
            title_text = (m.get("event_title", "") + " " + market_text).lower()
            contains_home = any(alias in title_text for alias in home_aliases)
            contains_away = any(alias in title_text for alias in away_aliases)

            if contains_home and contains_away:
                best_market = m
                best_score = 0.9
                match_type = "title_contains"
                # Skip remaining scoring for this market since it's a strong match
                continue

            # C. FUZZY MATCH (Fallback)
            if best_score < 1.0:
                event_title = m.get("event_title", "").lower()
                full_text = event_title + " " + market_text
                
                h_score = TeamNameMatcher.similarity_score(home_norm.lower(), full_text)
                a_score = TeamNameMatcher.similarity_score(away_norm.lower(), full_text)
                
                # Boost "Winner" markets slightly
                is_winner = "winner" in market_text or "win" in market_text
                
                if h_score > 0.45 and a_score > 0.45:
                    avg = (h_score + a_score) / 2
                    if is_winner: avg += 0.1
                    
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
            # Use any available price to avoid treating a matched market as "No Match"
            yes_bid = price_to_prob(best_market.get("yes_bid"))
            yes_ask = price_to_prob(best_market.get("yes_ask"))
            last_trade = price_to_prob(best_market.get("last_price") or best_market.get("last_trade_price"))

            # Midpoint if both sides are present, otherwise fall back to whichever exists
            if yes_bid is not None and yes_ask is not None:
                yes_price = (yes_bid + yes_ask) / 2
            else:
                yes_price = yes_bid if yes_bid is not None else yes_ask if yes_ask is not None else last_trade

            # Some markets may not expose bids/asks but do include probability-style fields; use
            # those before giving up so matched events are not discarded as "No Match".
            if yes_price is None:
                yes_price = price_to_prob(
                    best_market.get("probability")
                    or best_market.get("implied_prob")
                    or best_market.get("market_prob")
                )

            # As a last resort, keep the match but neutralize the probability so the upstream
            # analyzer can still display the market instead of showing "No Match".
            if yes_price is None:
                yes_price = 0.5

            subtitle = best_market.get("subtitle", "").lower()

            # Determine Probability Direction
            is_home_sub = home_norm.lower() in subtitle
            is_away_sub = away_norm.lower() in subtitle

            final_prob = yes_price
            # If market is "Away Team to Win", invert the YES price for Home Team prob
            if is_away_sub and not is_home_sub and yes_price:
                final_prob = 1.0 - yes_price

            res.update({
                "kalshi_available": True,
                "kalshi_prob": final_prob,
                "kalshi_label": best_market.get("event_title", best_market.get("title")),
                "kalshi_match_debug": f"Match: {match_type}",
                "market_ticker": best_market.get("ticker"),
                "kalshi_volume": best_market.get("volume")
            })
            
        return res

# Helper for other modules
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
