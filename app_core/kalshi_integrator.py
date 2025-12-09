"""
Kalshi Integrator with Proper RSA-PSS Authentication
This file goes in: app_core/kalshi_integrator.py
"""

import copy
import time
import logging
import re
import string
from difflib import SequenceMatcher
from datetime import datetime
import pytz
import requests
import streamlit as st
from typing import Dict, List, Any, Optional, TypedDict

# --- 1. SHARED IMPORTS ---
try:
    from app_core.team_name_matcher import TeamNameMatcher
except ImportError:
    TeamNameMatcher = None

logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION & CONSTANTS ---

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
}

FUTURE_EXCLUDE_KEYWORDS = {
    "champions league", "ucl", "win the league", "to win league",
    "to win the league", "to win championship", "championship", "playoffs",
    "division winner", "relegation", "bottom of table", "top of table",
    "season wins", "regular season wins", "season", "wins", "champion", "exactly",
}

SUPPORTED_LEAGUES = {"nba", "nfl", "mlb", "ncaaf", "ncaab", "nhl"}

LEAGUE_SERIES_MAP = {
    "nba": "KXNBA",
    "nfl": "KXNFL",
    "mlb": "KXMLB",
    "nhl": "KXNHL",
    "ncaaf": "KXNCAAF",
    "ncaab": "KXNCAAB",
}

# Low threshold allows fuzzy matching (e.g. "St. Louis" vs "Saint Louis")
TEAM_FUZZY_THRESHOLD = 1.1  
MAX_LINE_DIFF = 3.0
DEBUG_KALSHI_MATCHING = False
TEAM_NAME_SIMILARITY = 0.80

# --- 3. DATA STRUCTURES ---

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

# --- 4. HELPER FUNCTIONS ---

def _debug_log(message: str, *args: Any) -> None:
    if DEBUG_KALSHI_MATCHING:
        logger.info(message, *args)

def price_to_prob(price) -> Optional[float]:
    if price is None or price == "":
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p > 1.01:
        p = p / 100.0
    if 0 <= p <= 1:
        return p
    return None

def _normalize_market_text(*parts: str) -> str:
    joined = " ".join([p or "" for p in parts])
    cleaned = joined.lower()
    cleaned = re.sub(r"[\.,\-_/]", " ", cleaned)
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def _parse_market_date(raw) -> Optional[datetime]:
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, (int, float)):
            value = float(raw)
            if value > 10_000_000_000:
                value = value / 1000.0
            return datetime.fromtimestamp(value, tz=pytz.UTC)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.UTC)
    except Exception:
        return None

def _extract_date_from_ticker(ticker: str) -> Optional[datetime]:
    if not ticker:
        return None
    match = re.search(r"(20\d{2}[01]\d[0-3]\d)", ticker)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=pytz.UTC)
        except Exception: pass
    match = re.search(r"(\d{4}-\d{2}-\d{2})", ticker)
    if match:
        try:
            return datetime.fromisoformat(match.group(1)).replace(tzinfo=pytz.UTC)
        except Exception: pass
    match = re.search(r"(\d{2}[01]\d[0-3]\d)", ticker)
    if match:
        try:
            return datetime.strptime(match.group(1), "%y%m%d").replace(tzinfo=pytz.UTC)
        except Exception: pass
    return None

def _extract_market_type(title: str, ticker: str) -> Optional[str]:
    text = ((title or "") + " " + (ticker or "")).upper()
    if "MONEYLINE" in text or re.search(r"\bML\b", text):
        return "ML"
    if "SPREAD" in text or re.search(r"\bSPR?\b", text):
        return "Spread"
    if "TOTAL" in text or "OVER" in text or "UNDER" in text or re.search(r"\bOU\b", text):
        return "Total"
    return None

def _build_team_codes(team_name: str) -> List[str]:
    """Generate plausible team codes from a full team name using explicit maps."""
    if not team_name:
        return []
    
    codes = []
    upper_name = team_name.upper().strip()
    
    if upper_name in KALSHI_TEAM_ABBREVIATIONS:
        codes.extend(KALSHI_TEAM_ABBREVIATIONS[upper_name])
    
    for key, abbreviations in KALSHI_TEAM_ABBREVIATIONS.items():
        if key in upper_name:
            codes.extend(abbreviations)

    words = [w for w in re.split(r"\s+", team_name) if w]
    if words:
        codes.append(words[0][:3].upper())
        codes.append("".join(w[0] for w in words[:3]).upper())
    if len(words) > 1:
        codes.append(words[-1][:3].upper())
        
    return list(set(codes))

def _extract_teams_from_ticker(ticker: str) -> List[str]:
    """Extract team codes, stripping league prefixes to fix parsing errors."""
    if not ticker:
        return []

    clean_ticker = ticker.upper()
    prefixes = ["KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXNCAAF", "KXNCAAB", "KX"]
    for p in prefixes:
        if clean_ticker.startswith(p):
            clean_ticker = clean_ticker[len(p):]
            break

    tokens = re.findall(r"[A-Z]{2,3}", clean_ticker)
    ignore = {
        "ML", "OU", "OVE", "UND", "SPR", "TOT", "GAM", "VS", "AT", 
        "NBA", "NFL", "MLB", "NHL", "NCA", "AF", "AB"
    }
    return [t for t in tokens if t not in ignore]

def normalize_name(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- 5. MAIN MATCHING FUNCTION (Must be global) ---

def match_game_to_kalshi(
    league: str,
    home_team: str,
    away_team: str,
    game_time: Optional[datetime],
    integrator: "KalshiIntegrator" = None,
    status: Optional[str] = "open",
) -> KalshiMatchResult:
    """Attempt to match a game to a Kalshi market with explicit reasons."""

    def _parse_market_metadata(market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = market.get("title") or ""
        subtitle = market.get("subtitle") or ""
        ticker = market.get("ticker") or ""

        text = f"{title} {subtitle} {ticker}".lower()
        if any(key in text for key in FUTURE_EXCLUDE_KEYWORDS):
            return None

        ticker_teams = _extract_teams_from_ticker(ticker)
        if len(ticker_teams) < 2:
            return None

        market_type = _extract_market_type(title, ticker)
        if market_type is None:
            market_type = "Unknown"

        date_from_ticker = _extract_date_from_ticker(ticker)
        parsed_date = _parse_market_date(market.get("event_date") or market.get("close_time"))
        market_date = parsed_date or date_from_ticker
        if market_date is None:
            return None

        probability = None
        for key in ["last_price_dollars", "last_price", "yes_bid_dollars", "yes_ask_dollars", "last_trade_price", "yes_bid", "yes_ask"]:
            probability = price_to_prob(market.get(key))
            if probability is not None:
                break

        teams_norm = []
        if TeamNameMatcher:
            teams_norm = [TeamNameMatcher.normalize(t) for t in ticker_teams]

        return {
            "ticker": ticker,
            "title": title,
            "teams": ticker_teams,
            "teams_norm": teams_norm,
            "market_type": market_type,
            "market_date": market_date,
            "probability": probability,
        }

    league_norm = normalize_name(league)
    if league_norm and league_norm not in SUPPORTED_LEAGUES:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, league=league_norm, reason="league_not_supported")

    kalshi = integrator or KalshiIntegrator()
    if kalshi is None:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, league=league_norm, reason="api_error:no_integrator")

    game_dt: Optional[datetime] = None
    if isinstance(game_time, datetime):
        game_dt = game_time
    elif game_time:
        try:
            game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
        except Exception:
            game_dt = None

    try:
        markets = kalshi.get_markets(status=status)
    except Exception as exc:
        short_err = str(exc)[:77] + "..." if len(str(exc)) > 80 else str(exc)
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, league=league_norm, reason=f"api_error:{short_err}")

    parsed_markets: List[Dict[str, Any]] = []
    for mkt in markets or []:
        parsed = _parse_market_metadata(mkt)
        if parsed:
            parsed_markets.append({"__meta": parsed, **mkt})

    if not parsed_markets:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, league=league_norm, reason="no_valid_game_markets")

    home_norm = TeamNameMatcher.normalize(home_team) if TeamNameMatcher else normalize_name(home_team)
    away_norm = TeamNameMatcher.normalize(away_team) if TeamNameMatcher else normalize_name(away_team)
    home_codes = _build_team_codes(home_team)
    away_codes = _build_team_codes(away_team)

    best_market: Optional[Dict[str, Any]] = None
    best_score: float = 0.0

    for market in parsed_markets:
        meta = market["__meta"]

        if league_norm and league_norm in LEAGUE_SERIES_MAP:
            expected = LEAGUE_SERIES_MAP[league_norm]
            series = str(market.get("series_ticker") or "").upper()
            if expected and series and not series.startswith(expected):
                continue

        if game_dt and meta.get("market_date"):
            # Tolerance: 2 days
            day_diff = abs((meta["market_date"].date() - game_dt.date()).days)
            if day_diff > 2:
                continue

        teams = meta.get("teams", [])
        if len(teams) < 2:
            continue

        def _team_score(team_code: str, target_norm: str, target_codes: List[str]) -> float:
            if team_code in target_codes: return 2.0
            if TeamNameMatcher and TeamNameMatcher.normalize(team_code) == target_norm: return 1.5
            if TeamNameMatcher and TeamNameMatcher.similarity_score(TeamNameMatcher.normalize(team_code), target_norm) >= TEAM_NAME_SIMILARITY: return 1.0
            return 0.0

        score_home_first = _team_score(teams[0], home_norm, home_codes) + _team_score(teams[1], away_norm, away_codes)
        score_away_first = _team_score(teams[0], away_norm, away_codes) + _team_score(teams[1], home_norm, home_codes)

        score = max(score_home_first, score_away_first)

        if score_home_first < 1.0 and score_away_first < 1.0:
            continue

        if meta.get("probability") is not None:
            score += 0.25

        if score > best_score:
            best_score = score
            best_market = market

    if not best_market or best_score < TEAM_FUZZY_THRESHOLD:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, league=league_norm, reason="no_market_match")

    meta = best_market["__meta"]
    probability = meta.get("probability")
    if probability is None and best_market.get("ticker"):
        try:
            order = kalshi.get_orderbook(best_market.get("ticker")) or {}
            yes_levels = None
            for key in ("yes", "orderbook_yes", "levels"):
                level = (order.get(key) or order.get("orderbook", {}).get(key) or {})
                if level:
                    yes_levels = level
                    break
            if isinstance(yes_levels, list) and yes_levels:
                price_val = yes_levels[0].get("price") or yes_levels[0].get("bid")
                probability = price_to_prob(price_val)
        except Exception:
            probability = None

    direction = "YES" if probability and probability >= 0.5 else "NO"

    return KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=meta.get("title") or meta.get("ticker"),
        probability=probability,
        raw_event_id=str(best_market.get("ticker") or best_market.get("id")),
        league=league_norm,
        reason="ok",
        market_type=meta.get("market_type"),
        direction=direction,
        game_date=meta.get("market_date"),
        kalshi_volume=best_market.get("volume"),
    )

# --- 6. INTEGRATOR CLASS ---

class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis"""

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY")
        self.api_secret = api_secret or st.secrets.get("KALSHI_API_SECRET")
        self.base_url = "https://api.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
        self.api_url = self.base_url if self.api_key else self.demo_url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        self._private_key = None
        self._auth_ready = False
        self.last_error: Optional[str] = None
        self._markets_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_duration = 300 

        if self.api_key and self.api_secret:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                key_data = self.api_secret.strip()
                if not key_data.startswith("-----BEGIN"):
                    key_data = "-----BEGIN RSA PRIVATE KEY-----\n" + key_data + "\n-----END RSA PRIVATE KEY-----"
                self._private_key = serialization.load_pem_private_key(
                    key_data.encode(), password=None, backend=default_backend(),
                )
                self._auth_ready = True
            except ImportError:
                logger.warning("cryptography library not installed - Kalshi auth disabled")
            except Exception as e:
                logger.warning(f"Could not load Kalshi RSA key: {e}")

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        if not self._private_key: return ""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            message = f"{timestamp}{method}{path}"
            signature = self._private_key.sign(
                message.encode("utf-8"),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")
        except Exception: return ""

    def _make_authenticated_request(self, method: str, endpoint: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None) -> Optional[dict]:
        import time as time_module
        url = f"{self.api_url}{endpoint}"
        timestamp = str(int(time_module.time() * 1000))
        headers = self.headers.copy()

        if self._auth_ready and self._private_key:
            path_suffix = endpoint.split("?", 1)[0]
            path_to_sign = f"/trade-api/v2{path_suffix}"
            signature = self._sign_request(method.upper(), path_to_sign, timestamp)
            headers["KALSHI-ACCESS-KEY"] = self.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=15)
            else:
                response = requests.post(url, headers=headers, json=json_data or params, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.last_error = f"API error: {response.status_code}"
                logger.warning(f"Kalshi error {response.status_code}: {response.text[:200]}")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Kalshi request failed: {e}")
        return None

    def get_sports_series(self) -> List[Dict]:
        try:
            data = self._make_authenticated_request("GET", "/series", params={"limit": 200})
            if not data: return []
            
            all_series = data.get("series", [])
            sports_keywords = ["NFL", "NBA", "MLB", "NHL", "UFC", "SOCCER", "TENNIS", "GOLF", "FOOTBALL", "BASKETBALL", "BASEBALL", "HOCKEY"]
            
            return [
                s for s in all_series 
                if any(kw in s.get("ticker", "").upper() or kw in s.get("title", "").upper() for kw in sports_keywords)
            ]
        except Exception: return []

    def get_markets(self, category: str = "sports", status: Optional[str] = "open") -> List[Dict]:
        """Fetch available Kalshi markets, prioritizing major sports series to ensure coverage."""
        cache_key = status or "all"
        now = time.time()
        if cache_key in self._markets_cache and now - self._cache_time.get(cache_key, 0) < self._cache_duration:
            return copy.deepcopy(self._markets_cache[cache_key])

        all_markets = []
        try:
            series_list = self.get_sports_series() or []
            priority_tickers = []
            target_prefixes = ["KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXNCAAF", "KXNCAAB"]
            
            for s in series_list:
                ticker = s.get("ticker", "")
                if any(ticker.startswith(p) for p in target_prefixes):
                    priority_tickers.append(s)
            
            priority_tickers.sort(key=lambda x: x.get("start_date", ""), reverse=True)
            series_to_fetch = priority_tickers[:30]
            if not series_to_fetch:
                series_to_fetch = series_list[:30]

            logger.info(f"Fetching markets for {len(series_to_fetch)} series...")

            for series in series_to_fetch:
                ticker = series.get("ticker")
                if not ticker: continue
                
                params = {"series_ticker": ticker, "limit": 1000}
                if status: 
                    params["status"] = status

                time.sleep(0.05) 
                
                data = self._make_authenticated_request("GET", "/markets", params=params)
                if data: 
                    new_markets = data.get("markets", [])
                    all_markets.extend(new_markets)

            if all_markets:
                self._markets_cache[cache_key] = all_markets
                self._cache_time[cache_key] = now
                logger.info(f"✅ Cached {len(all_markets)} Kalshi markets")
                
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            
        return all_markets

    def get_sports_markets(self) -> List[Dict]:
        return self.get_markets()

    def get_orderbook(self, ticker: str) -> Dict:
        if not ticker: return {}
        data = self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook")
        return data.get("orderbook", {}) if data else {}

    def get_game_markets_for_events(self, league: str) -> List[Dict]:
        """Wrapper to get markets for a specific league."""
        prefix = LEAGUE_SERIES_MAP.get(league.lower(), "")
        all_markets = self.get_markets(status=None)
        if not prefix: return all_markets
        return [m for m in all_markets if m.get("series_ticker", "").startswith(prefix)]

    def filter_markets_closing_today(self, markets: List[Dict]) -> List[Dict]:
        """Filter markets closing today (Eastern Time)."""
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
