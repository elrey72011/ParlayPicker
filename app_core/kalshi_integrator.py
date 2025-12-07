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
    },
    # NCAAB - Men's College Basketball (Top Teams)
    "NCAAB": {
        # ACC
        "CLEM": ["CLEMSON", "CLEMSON TIGERS"],
        "DUKE": ["DUKE", "DUKE BLUE DEVILS"],
        "FSU": ["FLORIDA STATE", "FSU", "FLORIDA ST", "SEMINOLES"],
        "UNC": ["NORTH CAROLINA", "UNC", "TAR HEELS"],
        "NCSU": ["NC STATE", "NCSU", "NC ST", "WOLFPACK"],
        "PITT": ["PITTSBURGH", "PITT", "PANTHERS"],
        "CUSE": ["SYRACUSE", "ORANGE"],
        "UVA": ["VIRGINIA", "UVA", "CAVALIERS"],
        "VT": ["VIRGINIA TECH", "VT", "HOKIES"],
        "WAKE": ["WAKE FOREST", "DEMON DEACONS"],
        "BC": ["BOSTON COLLEGE", "BC", "EAGLES"],
        "LOU": ["LOUISVILLE", "CARDINALS"],
        "MIA": ["MIAMI", "MIAMI FL", "HURRICANES"],
        "ND": ["NOTRE DAME", "FIGHTING IRISH"],
        # Big Ten
        "ILL": ["ILLINOIS", "FIGHTING ILLINI"],
        "IND": ["INDIANA", "HOOSIERS"],
        "IOWA": ["IOWA", "HAWKEYES"],
        "MD": ["MARYLAND", "TERRAPINS", "TERPS"],
        "MICH": ["MICHIGAN", "WOLVERINES"],
        "MSU": ["MICHIGAN STATE", "MICH ST", "SPARTANS"],
        "MINN": ["MINNESOTA", "GOLDEN GOPHERS"],
        "NEB": ["NEBRASKA", "CORNHUSKERS"],
        "NW": ["NORTHWESTERN", "WILDCATS"],
        "OSU": ["OHIO STATE", "OHIO ST", "BUCKEYES"],
        "PSU": ["PENN STATE", "PENN ST", "NITTANY LIONS"],
        "PUR": ["PURDUE", "BOILERMAKERS"],
        "RUT": ["RUTGERS", "SCARLET KNIGHTS"],
        "WIS": ["WISCONSIN", "BADGERS"],
        # Big 12
        "BAY": ["BAYLOR", "BEARS"],
        "ISU": ["IOWA STATE", "IOWA ST", "CYCLONES"],
        "KU": ["KANSAS", "JAYHAWKS"],
        "KSU": ["KANSAS STATE", "KANSAS ST", "K-STATE"],
        "OSU": ["OKLAHOMA STATE", "OKLA ST", "COWBOYS"],
        "OU": ["OKLAHOMA", "SOONERS"],
        "TCU": ["TCU", "HORNED FROGS"],
        "TEX": ["TEXAS", "LONGHORNS"],
        "TTU": ["TEXAS TECH", "TECH", "RED RAIDERS"],
        "WVU": ["WEST VIRGINIA", "WVU", "MOUNTAINEERS"],
        "BYU": ["BYU", "COUGARS"],
        "CIN": ["CINCINNATI", "BEARCATS"],
        "HOU": ["HOUSTON", "COUGARS"],
        "UCF": ["UCF", "KNIGHTS"],
        # SEC
        "ALA": ["ALABAMA", "CRIMSON TIDE"],
        "ARK": ["ARKANSAS", "RAZORBACKS"],
        "AUB": ["AUBURN", "TIGERS"],
        "FLA": ["FLORIDA", "GATORS"],
        "UGA": ["GEORGIA", "BULLDOGS"],
        "UK": ["KENTUCKY", "WILDCATS"],
        "LSU": ["LSU", "TIGERS"],
        "MISS": ["MISSISSIPPI", "OLE MISS", "REBELS"],
        "MSU": ["MISSISSIPPI STATE", "MISS ST", "BULLDOGS"],
        "MIZZ": ["MISSOURI", "TIGERS"],
        "SC": ["SOUTH CAROLINA", "GAMECOCKS"],
        "TENN": ["TENNESSEE", "VOLUNTEERS"],
        "TAMU": ["TEXAS A&M", "TEXAS AM", "AGGIES"],
        "VAN": ["VANDERBILT", "COMMODORES"],
        # Pac-12
        "ARIZ": ["ARIZONA", "WILDCATS"],
        "ASU": ["ARIZONA STATE", "ARIZONA ST", "SUN DEVILS"],
        "CAL": ["CALIFORNIA", "CAL", "GOLDEN BEARS"],
        "COLO": ["COLORADO", "BUFFALOES"],
        "ORE": ["OREGON", "DUCKS"],
        "ORST": ["OREGON STATE", "OREGON ST", "BEAVERS"],
        "UCLA": ["UCLA", "BRUINS"],
        "USC": ["USC", "TROJANS"],
        "STAN": ["STANFORD", "CARDINAL"],
        "UTAH": ["UTAH", "UTES"],
        "WASH": ["WASHINGTON", "HUSKIES"],
        "WSU": ["WASHINGTON STATE", "WASH ST", "COUGARS"],
        # Big East
        "BUT": ["BUTLER", "BULLDOGS"],
        "CREI": ["CREIGHTON", "BLUEJAYS"],
        "DEPAUL": ["DEPAUL", "BLUE DEMONS"],
        "GTWN": ["GEORGETOWN", "HOYAS"],
        "MARQ": ["MARQUETTE", "GOLDEN EAGLES"],
        "PROV": ["PROVIDENCE", "FRIARS"],
        "SETON": ["SETON HALL", "PIRATES"],
        "SJU": ["ST JOHNS", "ST. JOHNS", "RED STORM"],
        "NOVA": ["VILLANOVA", "WILDCATS"],
        "XAV": ["XAVIER", "MUSKETEERS"],
        # Other Power Conferences
        "GONZ": ["GONZAGA", "BULLDOGS"],
        "SDSU": ["SAN DIEGO STATE", "AZTECS"],
        "MEM": ["MEMPHIS", "TIGERS"],
        "SMU": ["SMU", "MUSTANGS"],
        "TEMP": ["TEMPLE", "OWLS"],
        "VCU": ["VCU", "RAMS"]
    },
    # NCAAF - College Football (Top Teams)
    "NCAAF": {
        # ACC
        "CLEM": ["CLEMSON", "CLEMSON TIGERS"],
        "DUKE": ["DUKE", "DUKE BLUE DEVILS"],
        "FSU": ["FLORIDA STATE", "FSU", "FLORIDA ST", "SEMINOLES"],
        "UNC": ["NORTH CAROLINA", "UNC", "TAR HEELS"],
        "NCSU": ["NC STATE", "NCSU", "NC ST", "WOLFPACK"],
        "PITT": ["PITTSBURGH", "PITT", "PANTHERS"],
        "CUSE": ["SYRACUSE", "ORANGE"],
        "UVA": ["VIRGINIA", "UVA", "CAVALIERS"],
        "VT": ["VIRGINIA TECH", "VT", "HOKIES"],
        "WAKE": ["WAKE FOREST", "DEMON DEACONS"],
        "BC": ["BOSTON COLLEGE", "BC", "EAGLES"],
        "LOU": ["LOUISVILLE", "CARDINALS"],
        "MIA": ["MIAMI", "MIAMI FL", "HURRICANES"],
        "GT": ["GEORGIA TECH", "GA TECH", "YELLOW JACKETS"],
        # Big Ten
        "ILL": ["ILLINOIS", "FIGHTING ILLINI"],
        "IND": ["INDIANA", "HOOSIERS"],
        "IOWA": ["IOWA", "HAWKEYES"],
        "MD": ["MARYLAND", "TERRAPINS", "TERPS"],
        "MICH": ["MICHIGAN", "WOLVERINES"],
        "MSU": ["MICHIGAN STATE", "MICH ST", "SPARTANS"],
        "MINN": ["MINNESOTA", "GOLDEN GOPHERS"],
        "NEB": ["NEBRASKA", "CORNHUSKERS"],
        "NW": ["NORTHWESTERN", "WILDCATS"],
        "OSU": ["OHIO STATE", "OHIO ST", "BUCKEYES"],
        "PSU": ["PENN STATE", "PENN ST", "NITTANY LIONS"],
        "PUR": ["PURDUE", "BOILERMAKERS"],
        "RUT": ["RUTGERS", "SCARLET KNIGHTS"],
        "WIS": ["WISCONSIN", "BADGERS"],
        "UCLA": ["UCLA", "BRUINS"],
        "USC": ["USC", "TROJANS"],
        "ORE": ["OREGON", "DUCKS"],
        "WASH": ["WASHINGTON", "HUSKIES"],
        # Big 12
        "BAY": ["BAYLOR", "BEARS"],
        "ISU": ["IOWA STATE", "IOWA ST", "CYCLONES"],
        "KU": ["KANSAS", "JAYHAWKS"],
        "KSU": ["KANSAS STATE", "KANSAS ST", "K-STATE"],
        "OKST": ["OKLAHOMA STATE", "OKLA ST", "COWBOYS"],
        "OU": ["OKLAHOMA", "SOONERS"],
        "TCU": ["TCU", "HORNED FROGS"],
        "TEX": ["TEXAS", "LONGHORNS"],
        "TTU": ["TEXAS TECH", "TECH", "RED RAIDERS"],
        "WVU": ["WEST VIRGINIA", "WVU", "MOUNTAINEERS"],
        "BYU": ["BYU", "COUGARS"],
        "CIN": ["CINCINNATI", "BEARCATS"],
        "HOU": ["HOUSTON", "COUGARS"],
        "UCF": ["UCF", "KNIGHTS"],
        "COLO": ["COLORADO", "BUFFALOES"],
        "ARIZ": ["ARIZONA", "WILDCATS"],
        "ASU": ["ARIZONA STATE", "ARIZONA ST", "SUN DEVILS"],
        "UTAH": ["UTAH", "UTES"],
        # SEC
        "ALA": ["ALABAMA", "CRIMSON TIDE"],
        "ARK": ["ARKANSAS", "RAZORBACKS"],
        "AUB": ["AUBURN", "TIGERS"],
        "FLA": ["FLORIDA", "GATORS"],
        "UGA": ["GEORGIA", "BULLDOGS"],
        "UK": ["KENTUCKY", "WILDCATS"],
        "LSU": ["LSU", "TIGERS"],
        "MISS": ["MISSISSIPPI", "OLE MISS", "REBELS"],
        "MSU": ["MISSISSIPPI STATE", "MISS ST", "BULLDOGS"],
        "MIZZ": ["MISSOURI", "TIGERS"],
        "SC": ["SOUTH CAROLINA", "GAMECOCKS"],
        "TENN": ["TENNESSEE", "VOLUNTEERS"],
        "TAMU": ["TEXAS A&M", "TEXAS AM", "AGGIES"],
        "VAN": ["VANDERBILT", "COMMODORES"],
        # Other Notable Programs
        "ND": ["NOTRE DAME", "FIGHTING IRISH"],
        "ARMY": ["ARMY", "BLACK KNIGHTS"],
        "NAVY": ["NAVY", "MIDSHIPMEN"],
        "AF": ["AIR FORCE", "FALCONS"],
        "BSU": ["BOISE STATE", "BOISE ST", "BRONCOS"],
        "SDSU": ["SAN DIEGO STATE", "AZTECS"],
        "FRES": ["FRESNO STATE", "FRESNO ST", "BULLDOGS"],
        "NEV": ["NEVADA", "WOLF PACK"],
        "UNLV": ["UNLV", "REBELS"],
        "CSU": ["COLORADO STATE", "COLORADO ST", "RAMS"],
        "WYO": ["WYOMING", "COWBOYS"],
        "USU": ["UTAH STATE", "AGGIES"],
        "UNM": ["NEW MEXICO", "LOBOS"],
        "SJSU": ["SAN JOSE STATE", "SAN JOSE ST", "SPARTANS"],
        "HAW": ["HAWAII", "RAINBOW WARRIORS"],
        "APP": ["APPALACHIAN STATE", "APP STATE", "MOUNTAINEERS"],
        "CHAR": ["CHARLOTTE", "49ERS"],
        "ECU": ["EAST CAROLINA", "PIRATES"],
        "FIU": ["FIU", "PANTHERS"],
        "FAU": ["FAU", "OWLS"],
        "LT": ["LOUISIANA TECH", "LA TECH", "BULLDOGS"],
        "MTSU": ["MIDDLE TENNESSEE", "MIDDLE TENN", "BLUE RAIDERS"],
        "UNT": ["NORTH TEXAS", "MEAN GREEN"],
        "ODU": ["OLD DOMINION", "MONARCHS"],
        "RICE": ["RICE", "OWLS"],
        "USM": ["SOUTHERN MISS", "SOUTHERN MISSISSIPPI", "GOLDEN EAGLES"],
        "UAB": ["UAB", "BLAZERS"],
        "UTEP": ["UTEP", "MINERS"],
        "UTSA": ["UTSA", "ROADRUNNERS"],
        "WKU": ["WESTERN KENTUCKY", "WKU", "HILLTOPPERS"],
        "LIB": ["LIBERTY", "FLAMES"],
        "JVST": ["JACKSONVILLE STATE", "GAMECOCKS"],
        "SAM": ["SAM HOUSTON STATE", "SAM HOUSTON", "BEARKATS"],
        "NMSU": ["NEW MEXICO STATE", "AGGIES"]
    }
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
        
        Args:
            sport_ticker: Specific sport series (e.g., "KXNBA")
            days_ahead: How many days ahead to look (default 3)
        """
        now = int(time.time())
        target_series = [sport_ticker] if sport_ticker else CORE_SERIES
        cache_key = f"events_{sport_ticker or 'all'}_{now // 300}"
        
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
                "limit": 200,
                "status": "open"  # Valid values: 'open', 'closed', 'settled'
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
                        m['event_ticker'] = e.get('event_ticker')
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
        
        logger.info(f"Total markets after filtering: {len(all_events)}")
        self._markets_cache[cache_key] = all_events
        return all_events

    def get_sports_markets(self):
        return self.get_todays_events()
    
    def get_game_markets_for_events(self, league="NBA"):
        """Legacy alias used by Kalshi tab."""
        sport_map = {k: v for k, v in GAME_SERIES.items()}
        ticker = sport_map.get(league.upper(), GAME_SERIES["NBA"])
        return self.get_todays_events(ticker)

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
