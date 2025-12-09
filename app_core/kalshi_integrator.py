"""
Kalshi Integrator with Proper RSA-PSS Authentication
This file goes in: app_core/kalshi_integrator.py
"""

import copy
import time
import logging
import re
import string
import os
from datetime import datetime
import pytz
import requests
import streamlit as st
from typing import Dict, List, Any, Optional, TypedDict

try:
    from app_core.team_name_matcher import TeamNameMatcher
except ImportError:
    TeamNameMatcher = None

logger = logging.getLogger(__name__)

# --- 1. EXPANDED TEAM ABBREVIATIONS (NBA + NFL + NHL + MLB) ---
# --- MASTER TEAM LIST (NBA, NFL, NHL, MLB) ---
KALSHI_TEAM_ABBREVIATIONS = {
    # NBA
    "ATLANTA HAWKS": ["ATL"], "BOSTON CELTICS": ["BOS"], "BROOKLYN NETS": ["BKN", "BRK"],
    "CHARLOTTE HORNETS": ["CHA", "CLT"], "CHICAGO BULLS": ["CHI"], "CLEVELAND CAVALIERS": ["CLE", "CAVS"],
    "DALLAS MAVERICKS": ["DAL"], "DENVER NUGGETS": ["DEN"], "DETROIT PISTONS": ["DET"],
    "GOLDEN STATE WARRIORS": ["GSW", "GS"], "HOUSTON ROCKETS": ["HOU"], "INDIANA PACERS": ["IND"],
    "LOS ANGELES CLIPPERS": ["LAC", "LA CLIPPERS"], "LOS ANGELES LAKERS": ["LAL", "LA LAKERS"],
    "MEMPHIS GRIZZLIES": ["MEM"], "MIAMI HEAT": ["MIA"], "MILWAUKEE BUCKS": ["MIL"],
    "MINNESOTA TIMBERWOLVES": ["MIN"], "NEW ORLEANS PELICANS": ["NOP", "NO PELICANS"],
    "NEW YORK KNICKS": ["NYK", "NY KNICKS"], "OKLAHOMA CITY THUNDER": ["OKC"], "ORLANDO MAGIC": ["ORL"],
    "PHILADELPHIA 76ERS": ["PHI", "PHL", "SIXERS"], "PHOENIX SUNS": ["PHX"], "PORTLAND TRAIL BLAZERS": ["POR"],
    "SACRAMENTO KINGS": ["SAC"], "SAN ANTONIO SPURS": ["SAS", "SA SPURS"], "TORONTO RAPTORS": ["TOR"],
    "UTAH JAZZ": ["UTA"], "WASHINGTON WIZARDS": ["WAS", "WSH"],
    # NFL
    "ARIZONA CARDINALS": ["ARI", "ARZ"], "ATLANTA FALCONS": ["ATL"], "BALTIMORE RAVENS": ["BAL"],
    "BUFFALO BILLS": ["BUF"], "CAROLINA PANTHERS": ["CAR"], "CHICAGO BEARS": ["CHI"],
    "CINCINNATI BENGALS": ["CIN"], "CLEVELAND BROWNS": ["CLE"], "DALLAS COWBOYS": ["DAL"],
    "DENVER BRONCOS": ["DEN"], "DETROIT LIONS": ["DET"], "GREEN BAY PACKERS": ["GB", "GBP"],
    "HOUSTON TEXANS": ["HOU"], "INDIANAPOLIS COLTS": ["IND"], "JACKSONVILLE JAGUARS": ["JAX", "JAC"],
    "KANSAS CITY CHIEFS": ["KC", "KCC"], "LAS VEGAS RAIDERS": ["LV", "LVR"], "LOS ANGELES CHARGERS": ["LAC"],
    "LOS ANGELES RAMS": ["LAR"], "MIAMI DOLPHINS": ["MIA"], "MINNESOTA VIKINGS": ["MIN"],
    "NEW ENGLAND PATRIOTS": ["NE", "NEP"], "NEW ORLEANS SAINTS": ["NO", "NOS"], "NEW YORK GIANTS": ["NYG"],
    "NEW YORK JETS": ["NYJ"], "PHILADELPHIA EAGLES": ["PHI"], "PITTSBURGH STEELERS": ["PIT"],
    "SAN FRANCISCO 49ERS": ["SF", "SFO"], "SEATTLE SEAHAWKS": ["SEA"], "TAMPA BAY BUCCANEERS": ["TB", "TBB"],
    "TENNESSEE TITANS": ["TEN"], "WASHINGTON COMMANDERS": ["WAS", "WSH"],
    # NHL (Added)
    "ANAHEIM DUCKS": ["ANA"], "BOSTON BRUINS": ["BOS"], "BUFFALO SABRES": ["BUF"], "CALGARY FLAMES": ["CGY"],
    "CAROLINA HURRICANES": ["CAR"], "CHICAGO BLACKHAWKS": ["CHI"], "COLORADO AVALANCHE": ["COL"], "COLUMBUS BLUE JACKETS": ["CBJ"],
    "DALLAS STARS": ["DAL"], "DETROIT RED WINGS": ["DET"], "EDMONTON OILERS": ["EDM"], "FLORIDA PANTHERS": ["FLA"],
    "LOS ANGELES KINGS": ["LAK"], "MINNESOTA WILD": ["MIN"], "MONTREAL CANADIENS": ["MTL"], "NASHVILLE PREDATORS": ["NSH"],
    "NEW JERSEY DEVILS": ["NJD"], "NEW YORK ISLANDERS": ["NYI"], "NEW YORK RANGERS": ["NYR"], "OTTAWA SENATORS": ["OTT"],
    "PHILADELPHIA FLYERS": ["PHI"], "PITTSBURGH PENGUINS": ["PIT"], "SAN JOSE SHARKS": ["SJS"], "SEATTLE KRAKEN": ["SEA"],
    "ST LOUIS BLUES": ["STL"], "TAMPA BAY LIGHTNING": ["TBL"], "TORONTO MAPLE LEAFS": ["TOR"], "UTAH HOCKEY CLUB": ["UTA"],
    "VANCOUVER CANUCKS": ["VAN"], "VEGAS GOLDEN KNIGHTS": ["VGK"], "WASHINGTON CAPITALS": ["WSH"], "WINNIPEG JETS": ["WPG"],
    # MLB (Added)
    "ARIZONA DIAMONDBACKS": ["ARI"], "ATLANTA BRAVES": ["ATL"], "BALTIMORE ORIOLES": ["BAL"], "BOSTON RED SOX": ["BOS"],
    "CHICAGO CUBS": ["CHC"], "CHICAGO WHITE SOX": ["CWS"], "CINCINNATI REDS": ["CIN"], "CLEVELAND GUARDIANS": ["CLE"],
    "COLORADO ROCKIES": ["COL"], "DETROIT TIGERS": ["DET"], "HOUSTON ASTROS": ["HOU"], "KANSAS CITY ROYALS": ["KC"],
    "LOS ANGELES ANGELS": ["LAA"], "LOS ANGELES DODGERS": ["LAD"], "MIAMI MARLINS": ["MIA"], "MILWAUKEE BREWERS": ["MIL"],
    "MINNESOTA TWINS": ["MIN"], "NEW YORK METS": ["NYM"], "NEW YORK YANKEES": ["NYY"], "OAKLAND ATHLETICS": ["OAK"],
    "PHILADELPHIA PHILLIES": ["PHI"], "PITTSBURGH PIRATES": ["PIT"], "SAN DIEGO PADRES": ["SD"], "SAN FRANCISCO GIANTS": ["SF"],
    "SEATTLE MARINERS": ["SEA"], "ST LOUIS CARDINALS": ["STL"], "TAMPA BAY RAYS": ["TB"], "TEXAS RANGERS": ["TEX"],
    "TORONTO BLUE JAYS": ["TOR"], "WASHINGTON NATIONALS": ["WSH"]
}

FUTURE_EXCLUDE_KEYWORDS = {
    "champions league", "ucl", "win the league", "to win league",
    "to win the league", "to win championship", "championship", "playoffs",
    "division winner", "relegation", "bottom of table", "top of table",
    "season wins", "regular season wins", "season", "wins", "champion", "exactly",
}

SUPPORTED_LEAGUES = {"nba", "nfl", "mlb", "ncaaf", "ncaab", "nhl"}

LEAGUE_SERIES_MAP = {
    "nba": "KXNBA", "nfl": "KXNFL", "mlb": "KXMLB",
    "nhl": "KXNHL", "ncaaf": "KXNCAAF", "ncaab": "KXNCAAB",
}

# Thresholds
TEAM_FUZZY_THRESHOLD = 1.1  
DEBUG_KALSHI_MATCHING = False
TEAM_NAME_SIMILARITY = 0.80

class KalshiMatchResult(TypedDict, total=False):
    matched: bool
    kalshi_available: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: str
    reason: str
    market_type: Optional[str]
    direction: Optional[str]
    game_date: Optional[datetime]
    kalshi_volume: Optional[int]

# --- HELPER FUNCTIONS ---

def price_to_prob(price) -> Optional[float]:
    if price is None or price == "": return None
    try:
        p = float(price)
    except (TypeError, ValueError): return None
    if p > 1.01: p = p / 100.0
    if 0 <= p <= 1: return p
    return None

def _parse_market_date(raw) -> Optional[datetime]:
    if raw is None or raw == "": return None
    try:
        if isinstance(raw, (int, float)):
            value = float(raw)
            if value > 10_000_000_000: value = value / 1000.0
            return datetime.fromtimestamp(value, tz=pytz.UTC)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.UTC)
    except Exception: return None

def _extract_date_from_ticker(ticker: str) -> Optional[datetime]:
    if not ticker: return None
    match = re.search(r"(20\d{2}[01]\d[0-3]\d)", ticker)
    if match:
        try: return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=pytz.UTC)
        except Exception: pass
    match = re.search(r"(\d{2}[01]\d[0-3]\d)", ticker)
    if match:
        try: return datetime.strptime(match.group(1), "%y%m%d").replace(tzinfo=pytz.UTC)
        except Exception: pass
    return None

def _extract_market_type(title: str, ticker: str) -> Optional[str]:
    text = ((title or "") + " " + (ticker or "")).upper()
    if "MONEYLINE" in text or "ML" in text: return "ML"
    if "SPREAD" in text or "SPD" in text: return "Spread"
    if "TOTAL" in text or "OVER" in text or "UNDER" in text: return "Total"
    return None

def _build_team_codes(team_name: str) -> List[str]:
    if not team_name: return []
    codes = []
    upper_name = team_name.upper().strip()
    if upper_name in KALSHI_TEAM_ABBREVIATIONS:
        codes.extend(KALSHI_TEAM_ABBREVIATIONS[upper_name])
    for key, abbreviations in KALSHI_TEAM_ABBREVIATIONS.items():
        if key in upper_name: codes.extend(abbreviations)
    words = [w for w in re.split(r"\s+", team_name) if w]
    if words:
        codes.append(words[0][:3].upper())
        codes.append("".join(w[0] for w in words[:3]).upper())
    return list(set(codes))

def _extract_teams_from_ticker(ticker: str) -> List[str]:
    if not ticker: return []
    clean_ticker = ticker.upper()
    prefixes = ["KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXNCAAF", "KXNCAAB", "KX"]
    for p in prefixes:
        if clean_ticker.startswith(p):
            clean_ticker = clean_ticker[len(p):]
            break
    tokens = re.findall(r"[A-Z]{2,3}", clean_ticker)
    ignore = {"ML", "OU", "OVE", "UND", "SPR", "TOT", "GAM", "VS", "AT", "NBA", "NFL", "MLB", "NHL"}
    return [t for t in tokens if t not in ignore]

def normalize_name(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return s.strip()

# --- MAIN MATCHING FUNCTION ---

def match_game_to_kalshi(
    league: str,
    home_team: str,
    away_team: str,
    game_time: Optional[datetime],
    integrator: "KalshiIntegrator" = None,
    status: Optional[str] = "open",
) -> KalshiMatchResult:
    league_norm = normalize_name(league)
    kalshi = integrator or KalshiIntegrator()
    if kalshi is None:
        return KalshiMatchResult(matched=False, reason="api_error:no_integrator")

    game_dt: Optional[datetime] = None
    if isinstance(game_time, datetime):
        game_dt = game_time
    elif game_time:
        try:
            game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
        except Exception: game_dt = None

    try:
        markets = kalshi.get_markets(status=status)
    except Exception as exc:
        return KalshiMatchResult(matched=False, reason=f"api_error:{str(exc)[:50]}")

    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    home_codes = _build_team_codes(home_team)
    away_codes = _build_team_codes(away_team)

    best_market = None
    best_score = 0.0

    for mkt in markets:
        # Pre-filter by date (Tolerance: 2 days)
        mkt_date = _parse_market_date(mkt.get("event_date") or mkt.get("close_time"))
        if game_dt and mkt_date:
            if abs((mkt_date.date() - game_dt.date()).days) > 2:
                continue

        # Extract info
        title = mkt.get("title", "")
        ticker = mkt.get("ticker", "")
        if any(k in title.lower() for k in FUTURE_EXCLUDE_KEYWORDS): continue

        ticker_teams = _extract_teams_from_ticker(ticker)
        if len(ticker_teams) < 2: continue

        # Scoring
        score_home = 0.0
        score_away = 0.0
        
        # Check codes
        if any(c in ticker_teams for c in home_codes): score_home += 2.0
        if any(c in ticker_teams for c in away_codes): score_away += 2.0
        
        # Check title names
        norm_title = normalize_name(title)
        if home_norm in norm_title: score_home += 1.5
        if away_norm in norm_title: score_away += 1.5

        total_score = score_home + score_away
        if total_score > best_score:
            best_score = total_score
            best_market = mkt

    if not best_market or best_score < TEAM_FUZZY_THRESHOLD:
        return KalshiMatchResult(matched=False, reason="no_market_match")

    # Extract probability
    prob = None
    for key in ["last_price_dollars", "yes_bid_dollars", "yes_ask_dollars", "yes_bid", "last_price"]:
        prob = price_to_prob(best_market.get(key))
        if prob is not None: break

    return KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=best_market.get("title"),
        probability=prob,
        raw_event_id=best_market.get("ticker"),
        league=league_norm,
        reason="ok",
        market_type=_extract_market_type(best_market.get("title", ""), best_market.get("ticker", "")),
        game_date=_parse_market_date(best_market.get("close_time")),
        kalshi_volume=best_market.get("volume"),
    )

# --- INTEGRATOR CLASS ---

class KalshiIntegrator:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY") or os.environ.get("KALSHI_API_KEY")
        self.api_secret = api_secret or st.secrets.get("KALSHI_API_SECRET") or os.environ.get("KALSHI_API_SECRET")
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.api_url = self.base_url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        self._private_key = None
        self._auth_ready = False
        self._markets_cache = {}
        self._cache_time = {}
        self.last_error = None

        if self.api_key and self.api_secret:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                key_data = self.api_secret.strip()
                if not key_data.startswith("-----BEGIN"):
                    key_data = "-----BEGIN RSA PRIVATE KEY-----\n" + key_data + "\n-----END RSA PRIVATE KEY-----"
                self._private_key = serialization.load_pem_private_key(key_data.encode(), password=None, backend=default_backend())
                self._auth_ready = True
            except Exception as e:
                logger.warning(f"Kalshi auth setup failed: {e}")

    def _sign_request(self, method, path, timestamp):
        if not self._private_key: return ""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            msg = f"{timestamp}{method}{path}".encode('utf-8')
            sig = self._private_key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
            return base64.b64encode(sig).decode('utf-8')
        except: return ""

    def _make_authenticated_request(self, method, endpoint, params=None):
        import time as tm
        ts = str(int(tm.time() * 1000))
        headers = self.headers.copy()
        if self._auth_ready:
            sig = self._sign_request(method.upper(), f"/trade-api/v2{endpoint.split('?')[0]}", ts)
            headers.update({"KALSHI-ACCESS-KEY": self.api_key, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts})
        
        try:
            url = f"{self.api_url}{endpoint}"
            resp = requests.request(method, url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200: return resp.json()
            self.last_error = f"HTTP {resp.status_code}"
        except Exception as e: self.last_error = str(e)
        return None

    def get_markets(self, category: str = "sports", status: Optional[str] = "open") -> List[Dict]:
        """Nuclear Fetch: Download markets (including recent past) and filter locally."""
        cache_key = status or "all"
        now = time.time()
        
        if cache_key in self._markets_cache and now - self._cache_time.get(cache_key, 0) < self._cache_duration:
            return copy.deepcopy(self._markets_cache[cache_key])

        all_markets = []
        try:
            # 1. PRIORITY TICKERS FOR ALL SPORTS
            # These are fetched first to ensure we get them even if pagination stops early
            series_tickers = [
                # NBA
                "KXNBA", "KXNBASPREAD", "KXNBATOTAL", "KXNBAMONEYLINE",
                # NFL
                "KXNFL", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLMONEYLINE",
                # NHL
                "KXNHL", "KXNHLSPREAD", "KXNHLTOTAL", "KXNHLMONEYLINE",
                # MLB
                "KXMLB", "KXMLBSPREAD", "KXMLBTOTAL", "KXMLBMONEYLINE",
                # College
                "KXNCAAF", "KXNCAAB", "KXNCAAFSPREAD", "KXNCAABSPREAD",
                # Other
                "KXUFC"
            ]
            
            for ticker in series_tickers:
                # Look back 48 hours to catch live/recently finished games
                params = {
                    "series_ticker": ticker, 
                    "limit": 1000,
                    "min_close_ts": int(now - (86400 * 2)) 
                }
                if status: params["status"] = status
                
                # Small sleep to be nice to the API
                time.sleep(0.02)
                data = self._make_authenticated_request("GET", "/markets", params=params)
                if data: all_markets.extend(data.get("markets", []))

            # 2. Fallback Nuclear Scan (if specific tickers missed something)
            # Only run if we found very few markets
            if len(all_markets) < 50:
                cursor = None
                page_count = 0
                while page_count < 20:
                    params = {"limit": 200, "min_close_ts": int(now - (86400 * 2))}
                    if cursor: params["cursor"] = cursor
                    
                    data = self._make_authenticated_request("GET", "/markets", params=params)
                    if not data: break
                    
                    markets = data.get("markets", [])
                    if not markets: break
                    
                    all_markets.extend(markets)
                    cursor = data.get("cursor")
                    page_count += 1
                    if not cursor: break

            # 3. Local Filter
            sports_keywords = [
                "NFL", "NBA", "MLB", "NHL", "UFC", "SOCCER", "TENNIS", "FOOTBALL", "BASKETBALL",
                "HOCKEY", "BASEBALL", "COLLEGE", "NCAA",
                "SPREAD", "TOTAL", "MONEYLINE" 
            ]
            
            filtered = []
            seen_ids = set()
            
            for m in all_markets:
                mid = m.get("ticker")
                if mid in seen_ids: continue
                seen_ids.add(mid)

                title = (m.get("title") or "").upper()
                ticker = (m.get("ticker") or "").upper()
                
                if any(kw in title or kw in ticker for kw in sports_keywords):
                    filtered.append(m)

            if filtered:
                self._markets_cache[cache_key] = filtered
                self._cache_time[cache_key] = now
                logger.info(f"✅ Cached {len(filtered)} Kalshi markets (All Sports)")
                return filtered
                
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
        
        return []
    # Compatibility aliases
    get_sports_markets = get_markets
    get_game_markets_for_events = lambda self, league: self.get_markets()
    filter_markets_closing_today = lambda self, markets: markets # Passthrough
    get_orderbook = lambda self, ticker: (self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook") or {}).get("orderbook", {})

    def get_sports_markets(self) -> List[Dict]:
        return self.get_markets()

    def get_orderbook(self, ticker: str) -> Dict:
        if not ticker: return {}
        data = self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook")
        return data.get("orderbook", {}) if data else {}

    def get_game_markets_for_events(self, league: str) -> List[Dict]:
        return self.get_markets(status=None)

    def filter_markets_closing_today(self, markets: List[Dict]) -> List[Dict]:
        from datetime import datetime, time as dtime, timezone
        if not markets: return []
        try:
            tz = pytz.timezone("America/New_York")
            now_local = datetime.now(tz)
            start = tz.localize(datetime.combine(now_local.date(), dtime.min)).astimezone(timezone.utc)
            end = tz.localize(datetime.combine(now_local.date(), dtime.max)).astimezone(timezone.utc)
            filtered = []
            for m in markets:
                dt = _parse_market_date(m.get("close_time"))
                if dt and start <= dt <= end:
                    filtered.append(m)
            return filtered if filtered else markets
        except Exception:
            return markets
