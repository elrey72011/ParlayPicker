"""
Kalshi Integrator with RSA Signing & Pagination.
Location: app_core/kalshi_integrator.py
"""
from __future__ import annotations

import logging
import os
import time
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz
import requests
import streamlit as st

# Cryptography for RSA Signing (Required for Kalshi v2)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class KalshiMatchResult:
    matched: bool
    kalshi_available: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: Optional[str] = None
    reason: str = ""
    market_type: Optional[str] = None
    direction: Optional[str] = None
    game_date: Optional[datetime] = None
    kalshi_volume: Optional[float] = None

# ---------------------------------------------------------------------------
# Constants & Mappings
# ---------------------------------------------------------------------------

SUPPORTED_LEAGUES = {"NBA", "NFL", "MLB", "NHL", "NCAAF", "NCAAB"}

LEAGUE_SERIES_MAP: Dict[str, str] = {
    "NBA": "KXNBA",
    "NFL": "KXNFL",
    "MLB": "KXMLB",
    "NHL": "KXNHL",
    "NCAAF": "KXNCAAF",
    "NCAAB": "KXNCAAB",
}

# Extensive abbreviation list
KALSHI_TEAM_ABBREVIATIONS: Dict[str, List[str]] = {
    "ATLANTA HAWKS": ["ATL"], "BOSTON CELTICS": ["BOS"], "BROOKLYN NETS": ["BKN", "BRK"],
    "CHARLOTTE HORNETS": ["CHA", "CLT"], "CHICAGO BULLS": ["CHI"], "CLEVELAND CAVALIERS": ["CLE"],
    "DALLAS MAVERICKS": ["DAL"], "DENVER NUGGETS": ["DEN"], "DETROIT PISTONS": ["DET"],
    "GOLDEN STATE WARRIORS": ["GSW"], "HOUSTON ROCKETS": ["HOU"], "INDIANA PACERS": ["IND"],
    "LOS ANGELES CLIPPERS": ["LAC"], "LOS ANGELES LAKERS": ["LAL"], "MEMPHIS GRIZZLIES": ["MEM"],
    "MIAMI HEAT": ["MIA"], "MILWAUKEE BUCKS": ["MIL"], "MINNESOTA TIMBERWOLVES": ["MIN"],
    "NEW ORLEANS PELICANS": ["NOP"], "NEW YORK KNICKS": ["NYK"], "OKLAHOMA CITY THUNDER": ["OKC"],
    "ORLANDO MAGIC": ["ORL"], "PHILADELPHIA 76ERS": ["PHI"], "PHOENIX SUNS": ["PHX"],
    "PORTLAND TRAIL BLAZERS": ["POR"], "SACRAMENTO KINGS": ["SAC"], "SAN ANTONIO SPURS": ["SAS"],
    "TORONTO RAPTORS": ["TOR"], "UTAH JAZZ": ["UTA"], "WASHINGTON WIZARDS": ["WAS", "WSH"],
    "ARIZONA CARDINALS": ["ARI"], "ATLANTA FALCONS": ["ATL"], "BALTIMORE RAVENS": ["BAL"],
    "BUFFALO BILLS": ["BUF"], "CAROLINA PANTHERS": ["CAR"], "CHICAGO BEARS": ["CHI"],
    "CINCINNATI BENGALS": ["CIN"], "CLEVELAND BROWNS": ["CLE"], "DALLAS COWBOYS": ["DAL"],
    "DENVER BRONCOS": ["DEN"], "DETROIT LIONS": ["DET"], "GREEN BAY PACKERS": ["GB"],
    "HOUSTON TEXANS": ["HOU"], "INDIANAPOLIS COLTS": ["IND"], "JACKSONVILLE JAGUARS": ["JAX"],
    "KANSAS CITY CHIEFS": ["KC"], "LAS VEGAS RAIDERS": ["LV"], "LOS ANGELES CHARGERS": ["LAC"],
    "LOS ANGELES RAMS": ["LAR"], "MIAMI DOLPHINS": ["MIA"], "MINNESOTA VIKINGS": ["MIN"],
    "NEW ENGLAND PATRIOTS": ["NE"], "NEW ORLEANS SAINTS": ["NO"], "NEW YORK GIANTS": ["NYG"],
    "NEW YORK JETS": ["NYJ"], "PHILADELPHIA EAGLES": ["PHI"], "PITTSBURGH STEELERS": ["PIT"],
    "SAN FRANCISCO 49ERS": ["SF"], "SEATTLE SEAHAWKS": ["SEA"], "TAMPA BAY BUCCANEERS": ["TB"],
    "TENNESSEE TITANS": ["TEN"], "WASHINGTON COMMANDERS": ["WAS"],
    "BOSTON BRUINS": ["BOS"], "TORONTO MAPLE LEAFS": ["TOR"], "MONTREAL CANADIENS": ["MTL"],
    "NEW YORK RANGERS": ["NYR"], "CHICAGO BLACKHAWKS": ["CHI"], "DETROIT RED WINGS": ["DET"],
    "PITTSBURGH PENGUINS": ["PIT"], "TAMPA BAY LIGHTNING": ["TBL"], "VEGAS GOLDEN KNIGHTS": ["VGK"],
    "SEATTLE KRAKEN": ["SEA"]
}

TEAM_FUZZY_THRESHOLD = 1.0
DATE_TOLERANCE_DAYS = 5
DATE_SOFT_PENALTY = 0.10

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    if not name:
        return ""
    return (
        str(name)
        .strip()
        .upper()
        .replace("&", "AND")
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
    )

def price_to_prob(price: Any) -> Optional[float]:
    if price is None:
        return None
    try:
        val = float(price)
    except Exception:
        return None
    if val < 0:
        return None
    if 0.0 <= val <= 1.0:
        return max(0.0, min(1.0, val))
    if 0.0 <= val <= 100.0:
        return max(0.0, min(1.0, val / 100.0))
    return None

def _extract_market_type(title: str, ticker: str) -> str:
    t = (title or "").upper()
    if "SPREAD" in t or "POINTS" in t: return "spread"
    if "TOTAL" in t or "OVER/UNDER" in t or "O/U" in t: return "total"
    if "MONEYLINE" in t or "ML" in t: return "moneyline"
    return "generic"

def _extract_teams_from_ticker(ticker: str) -> List[str]:
    if not ticker: return []
    parts = ticker.split("-")
    if len(parts) < 3: return []
    middle = parts[1:3]
    return [p.strip().upper() for p in middle if p.strip()]

def _parse_market_metadata(mkt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = (mkt.get("title") or "").strip()
    if not title: return None
    ticker = (mkt.get("ticker") or mkt.get("event_ticker") or "")
    
    market_dt: Optional[datetime] = None
    close_raw = (mkt.get("close_time") or mkt.get("expiration_time") or mkt.get("expected_expiration_time"))
    if close_raw:
        try:
            market_dt = datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))
        except Exception:
            market_dt = None

    upper_title = title.upper()
    teams: List[str] = []
    
    separators = [" VS ", " VS. ", " V ", " @ ", " AT ", " - ", " / ", " | "]
    for sep in separators:
        if sep in upper_title:
            parts = upper_title.split(sep)
            if len(parts) >= 2:
                teams = [parts[0].strip(), parts[1].strip()]
            break

    if len(teams) < 2 and ticker:
        ticker_teams = _extract_teams_from_ticker(ticker)
        if len(ticker_teams) >= 2:
            teams = ticker_teams[:2]

    market_type = _extract_market_type(title, ticker)
    prob_source = (mkt.get("yes_price") or mkt.get("last_price") or mkt.get("yes_ask") or mkt.get("implied_prob"))
    prob = price_to_prob(prob_source)

    return {"title": title, "market_date": market_dt, "teams": teams, "probability": prob, "market_type": market_type}

def _build_team_codes(team_name: str) -> List[str]:
    norm = normalize_name(team_name)
    codes: List[str] = []
    for full_name, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
        if normalize_name(full_name) == norm:
            codes.extend(abbrs)
    tokens = norm.split()
    for t in tokens:
        if len(t) >= 2 and t not in codes:
            codes.append(t)
    return list(dict.fromkeys(codes))

def _team_score(team_code: str, target_norm: str, target_codes: List[str]) -> float:
    if not team_code: return 0.0
    clean_code = team_code.strip().upper()
    if clean_code in target_codes: return 2.0
    
    norm_code = normalize_name(team_code)
    if norm_code == target_norm: return 2.0
    if norm_code in target_norm or target_norm in norm_code: return 1.5
    
    words_code = set(norm_code.split())
    words_target = set(target_norm.split())
    if words_code & words_target: return 1.0
    
    return 0.0

def match_game_to_kalshi(league: str, home_team: str, away_team: str, game_time: Optional[datetime], integrator: "KalshiIntegrator" = None, status: Optional[str] = "open") -> KalshiMatchResult:
    league_key = (league or "").upper()
    kalshi = integrator or KalshiIntegrator()
    
    if not kalshi or not kalshi.api_key:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, reason="no_integrator")

    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    home_codes = _build_team_codes(home_team)
    away_codes = _build_team_codes(away_team)

    markets = kalshi.get_markets(status=status)
    if not markets:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, reason="no_markets_found")

    game_dt_utc: Optional[datetime] = None
    if isinstance(game_time, datetime):
        if game_time.tzinfo is None:
            game_dt_utc = pytz.utc.localize(game_time)
        else:
            game_dt_utc = game_time.astimezone(pytz.UTC)

    series_prefix = LEAGUE_SERIES_MAP.get(league_key)
    best_market = None
    best_score = 0.0

    for m in markets:
        meta = _parse_market_metadata(m)
        if not meta: continue
        
        ticker = (m.get("ticker") or "").upper()
        if series_prefix and not ticker.startswith(series_prefix):
            continue

        teams = meta.get("teams") or []
        if len(teams) < 2: continue

        score_home_first = _team_score(teams[0], home_norm, home_codes) + _team_score(teams[1], away_norm, away_codes)
        score_away_first = _team_score(teams[0], away_norm, away_codes) + _team_score(teams[1], home_norm, home_codes)
        score = max(score_home_first, score_away_first)

        m_date = meta.get("market_date")
        if game_dt_utc and m_date:
            try:
                if m_date.tzinfo is None:
                    m_date = pytz.utc.localize(m_date)
                diff = abs((m_date.date() - game_dt_utc.date()).days)
                if diff > DATE_TOLERANCE_DAYS: continue
                score -= (diff * DATE_SOFT_PENALTY)
            except Exception:
                pass

        if score > best_score:
            best_score = score
            best_market = m
            best_market["__meta"] = meta

    if not best_market or best_score < TEAM_FUZZY_THRESHOLD:
        return KalshiMatchResult(matched=False, kalshi_available=True, label="", probability=None, raw_event_id=None, reason=f"low_score_{best_score:.1f}")

    meta = best_market["__meta"]
    return KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=meta["title"],
        probability=meta["probability"],
        raw_event_id=best_market.get("ticker"),
        league=league_key,
        reason="matched",
        market_type=meta["market_type"],
        game_date=meta["market_date"]
    )

# ---------------------------------------------------------------------------
# KalshiIntegrator Class
# ---------------------------------------------------------------------------

class KalshiIntegrator:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        # Resolve key + secret from args, env, or Streamlit secrets (only the official names)
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY") or os.getenv("KALSHI_API_KEY")
        raw_secret = api_secret or st.secrets.get("KALSHI_API_SECRET") or os.getenv("KALSHI_API_SECRET")

        # --- Clean private key format ---
        self.api_secret_pem = self._normalize_secret(raw_secret)

        self.api_url = "https://api.elections.kalshi.com/trade-api/v2"

        # Caching + error state
        self._markets_cache: List[Dict[str, Any]] = []
        self._markets_cache_ts: float = 0.0
        self.cache_ttl_seconds: int = 300
        self.last_error: Optional[str] = None

    @staticmethod
    def _normalize_secret(secret_val: Optional[str]) -> Optional[str]:
        if not secret_val:
            return None
        cleaned = str(secret_val).replace("\\n", "\n").strip()
        if not cleaned:
            return None
        if "-----BEGIN RSA PRIVATE KEY-----" in cleaned or "-----BEGIN PRIVATE KEY-----" in cleaned:
            return cleaned
        # If no PEM markers, wrap as PKCS8
        return "-----BEGIN PRIVATE KEY-----\n" + cleaned + "\n-----END PRIVATE KEY-----"

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        if not self.api_secret_pem:
            return ""
        msg_string = f"{timestamp}{method}{path}"
        try:
            private_key = serialization.load_pem_private_key(
                self.api_secret_pem.encode("utf-8"),
                password=None,
            )
            signature = private_key.sign(
                msg_string.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return ""

    def health_check(self) -> Dict[str, Any]:
        configured = bool(self.api_key and self.api_secret_pem)
        if not configured:
            return {
                "configured": False,
                "ok": False,
                "market_count": 0,
                "sample_market": None,
                "error": "Kalshi is required but not configured.",
            }

        endpoint = "/markets"
        params = {"limit": 50}
        url = f"{self.api_url}{endpoint}"
        path_for_signing = f"/trade-api/v2{endpoint}"
        timestamp = str(int(time.time() * 1000))
        signature = self._sign_request("GET", path_for_signing, timestamp)
        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            status_code = resp.status_code
            snippet = resp.text[:300] if resp is not None else None
            if status_code != 200:
                return {
                    "configured": True,
                    "ok": False,
                    "market_count": 0,
                    "sample_market": None,
                    "status_code": status_code,
                    "url": url,
                    "response_text": snippet,
                    "error": f"Kalshi auth failed: status {status_code}",
                }
            data = resp.json()
            markets = data.get("markets", []) or []
            return {
                "configured": True,
                "ok": len(markets) > 0,
                "market_count": len(markets),
                "sample_market": markets[0] if markets else None,
                "markets": markets,
                "error": None if markets else "Kalshi returned zero markets.",
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "configured": True,
                "ok": False,
                "market_count": 0,
                "sample_market": None,
                "error": str(exc),
            }

    def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Optional[dict]:
        url = f"{self.api_url}{endpoint}"
        path_for_signing = f"/trade-api/v2{endpoint}"
        timestamp = str(int(time.time() * 1000))
        signature = self._sign_request(method, path_for_signing, timestamp)

        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

        try:
            resp = requests.request(method, url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                self.last_error = f"Status {resp.status_code}: {resp.text}"
                logger.error(f"Kalshi API Error: {self.last_error}")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Kalshi Connection Exception: {e}")
        return None

    def get_markets(self, status: Optional[str] = "open") -> List[Dict[str, Any]]:
        """Fetch all markets with pagination support and a sane cap."""
        now = time.time()
        if self._markets_cache and (now - self._markets_cache_ts) < self.cache_ttl_seconds:
            logger.info(f"Using cached markets ({len(self._markets_cache)} items)")
            return self._markets_cache

        all_markets: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        max_items = 5000

        while True:
            params = {"limit": 1000}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor

            data = self._make_authenticated_request("GET", "/markets", params=params)
            if not data:
                break

            markets = data.get("markets", []) or []
            all_markets.extend(markets)

            if len(all_markets) >= max_items:
                all_markets = all_markets[:max_items]
                logger.warning("Reached pagination cap when loading Kalshi markets")
                break

            cursor = (
                data.get("cursor")
                or data.get("next_cursor")
                or data.get("next")
                or data.get("next_token")
            )
            if not cursor:
                break
            time.sleep(0.1)

        self._markets_cache = all_markets
        self._markets_cache_ts = now
        logger.info(f"✅ Successfully loaded {len(all_markets)} Kalshi markets (paginated)")
        return all_markets

    def _filter_markets_for_league(self, markets: List[Dict[str, Any]], league: Optional[str]) -> List[Dict[str, Any]]:
        league_key = (league or "").upper()
        prefix = LEAGUE_SERIES_MAP.get(league_key)
        if not prefix:
            return markets

        ticker_upper = [str(m.get("ticker") or m.get("event_ticker") or "").upper() for m in markets]
        prefix_filtered = [m for m, t in zip(markets, ticker_upper) if t.startswith(prefix)]
        has_kxn_prefix = any(t.startswith("KXN") for t in [str(m.get("ticker") or m.get("event_ticker") or "").upper() for m in prefix_filtered])
        if prefix_filtered and has_kxn_prefix:
            return prefix_filtered

        return markets

    def _best_market_time(self, market: Dict[str, Any]) -> Optional[datetime]:
        for key in [
            "expected_expiration_time",
            "latest_expiration_time",
            "close_time",
            "expiration_time",
            "open_time",
        ]:
            val = market.get(key)
            if not val:
                continue
            try:
                raw = str(val)
                if raw.endswith("Z"):
                    raw = raw.replace("Z", "+00:00")
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt)
                else:
                    dt = dt.astimezone(pytz.UTC)
                return dt
            except Exception:
                continue
        return None

    def get_sports_markets(
        self, league: Optional[str] = None, commence_times: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        try:
            markets = self.get_markets(status=None)
            filtered = self._filter_markets_for_league(markets, league)

            commence_dt: List[datetime] = []
            if commence_times:
                for c in commence_times:
                    try:
                        raw = str(c)
                        if raw.endswith("Z"):
                            raw = raw.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(raw)
                        if dt.tzinfo is None:
                            dt = pytz.utc.localize(dt)
                        else:
                            dt = dt.astimezone(pytz.UTC)
                        commence_dt.append(dt)
                    except Exception:
                        continue

            if commence_dt:
                window = timedelta(hours=72)
                time_filtered: List[Dict[str, Any]] = []
                for m in filtered:
                    mt = self._best_market_time(m)
                    if not mt:
                        time_filtered.append(m)
                        continue
                    if any(abs(mt - cdt) <= window for cdt in commence_dt):
                        time_filtered.append(m)
                if time_filtered:
                    filtered = time_filtered

            if (league or "").upper() == "NBA":
                ticker_upper = [str(m.get("ticker") or m.get("event_ticker") or "").upper() for m in filtered]
                if not any(
                    t.startswith("KXNBAGAME")
                    or t.startswith("KXNBATOTAL")
                    or t.startswith("KXNBASPREAD")
                    for t in ticker_upper
                ):
                    filtered = markets

            return filtered
        except Exception:
            logger.error("Kalshi get_sports_markets failed", exc_info=True)
            return []

    def get_markets_for_league(self, league: str) -> List[Dict[str, Any]]:
        """Return markets for a given league without excluding game winners."""
        return self.get_sports_markets(league=league)

    # Helpers required by app
    def get_game_markets_for_events(self, league):
        return self.get_markets_for_league(league)

    def filter_markets_closing_today(self, markets):
        return markets

    @staticmethod
    def price_to_prob(price: Any) -> Optional[float]:
        return price_to_prob(price)
    
    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        return self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook") or {}

# ---------------------------------------------------------------------------
# CROSSWALK UTILITY (Included at bottom)
# ---------------------------------------------------------------------------
def get_event_crosswalk(league: str, home_team: str, away_team: str) -> Dict[str, Any]:
    """
    Returns a dictionary linking identifiers across data sources (Kalshi, OddsAPI, etc.)
    """
    league = league.upper()
    
    # Mapping for TheOddsAPI
    odds_api_keys = {
        'NFL': 'americanfootball_nfl',
        'NBA': 'basketball_nba',
        'NHL': 'icehockey_nhl',
        'MLB': 'baseball_mlb',
        'NCAAF': 'americanfootball_ncaaf',
        'NCAAB': 'basketball_ncaab'
    }
    
    # Mapping for Kalshi Series Tickers
    kalshi_series = {
        'NFL': 'KXNFL', 'NBA': 'KXNBA', 'NHL': 'KXNHL', 'MLB': 'KXMLB'
    }
    
    return {
        "Matchup": f"{away_team} @ {home_team}",
        "Sources": {
            "TheOddsAPI": {
                "sport_key": odds_api_keys.get(league),
                "home": home_team,
                "away": away_team
            },
            "Kalshi": {
                # Kalshi tickers often look like KXNBA-23DEC25-LAL-BOS
                "series_ticker": kalshi_series.get(league), 
                "fuzzy_match_query": f"{away_team} {home_team}" 
            },
            "NewsAPI": {
                # Strict query to avoid noise
                "query": f'"{away_team}" AND "{home_team}" AND {league}'
            },
            "APISports": {
                "endpoint": f"https://v1.{league.lower()}.api-sports.io/games",
            }
        }
    }
