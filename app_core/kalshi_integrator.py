"""
Kalshi Integrator with RSA Signing & Pagination.
Location: app_core/kalshi_integrator.py
"""
from __future__ import annotations

import logging
import os
import time
import base64
import random
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytz
import requests
import streamlit as st

# Cryptography for RSA Signing (Required for Kalshi v2)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)


class KalshiRateLimitError(Exception):
    """Raised when Kalshi keeps returning 429 after retries."""


class KalshiAPIError(Exception):
    """Raised for non-auth Kalshi API errors."""

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
    debug: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Constants & Mappings
# ---------------------------------------------------------------------------

SUPPORTED_LEAGUES = {"NBA", "NFL", "MLB", "NHL", "NCAAF", "NCAAB"}
SAFE_STATUS_ALLOWLIST = {"active", "finalized", "settled", "closed"}

LEAGUE_SERIES_MAP: Dict[str, Any] = {
    "NBA": [
        "KXNBAGAME",
        "KXNBA",
    ],
    "NFL": "KXNFL",
    "MLB": "KXMLB",
    "NHL": "KXNHL",
    "NCAAF": "KXNCAAF",
    "NCAAB": "KXNCAAB",
}

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
NBA_TZ = pytz.timezone("America/New_York")
ALLOW_SAME_DAY_TEXT_FALLBACK = False

NBA_TEAM_CODE_MAP: Dict[str, str] = {
    "NEW YORK KNICKS": "NYK",
    "SAN ANTONIO SPURS": "SAS",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "MEMPHIS GRIZZLIES": "MEM",
    "LOS ANGELES CLIPPERS": "LAC",
    "LA CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "LA LAKERS": "LAL",
}

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

def normalize_status(status: Optional[str]) -> Optional[str]:
    """Kalshi API status filter normalization."""
    if not status:
        return None
    s = str(status).strip().lower()
    if not s or s == "open":
        return None
    return s if s in SAFE_STATUS_ALLOWLIST else None

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

def _nba_date_token(dt: datetime, tz_name: str = "America/New_York") -> str:
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.UTC
    dt_local = dt.astimezone(tz)
    return dt_local.strftime("%y%b%d").upper()

def match_game_to_kalshi(league: str, home_team: str, away_team: str, game_time: Optional[datetime], integrator: "KalshiIntegrator" = None, status: Optional[str] = None) -> KalshiMatchResult:
    league_key = (league or "").upper()
    kalshi = integrator or KalshiIntegrator()
    
    if not kalshi or not kalshi.api_key:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, reason="no_integrator")

    def _nba_code(team: str) -> Optional[str]:
        norm = normalize_name(team)
        return NBA_TEAM_CODE_MAP.get(norm)

    if league_key == "NBA":
        if not isinstance(game_time, datetime):
            return KalshiMatchResult(matched=False, kalshi_available=True, label="", probability=None, raw_event_id=None, league=league_key, reason="missing_game_time")
        
        game_dt = game_time
        if game_dt.tzinfo is None: game_dt = NBA_TZ.localize(game_dt)
        game_dt_utc = game_dt.astimezone(pytz.UTC)
        date_token = _nba_date_token(game_dt_utc)
        away_code, home_code = _nba_code(away_team), _nba_code(home_team)

        if not away_code or not home_code:
            return KalshiMatchResult(matched=False, kalshi_available=True, label="", probability=None, raw_event_id=None, league=league_key, reason="missing_team_code")

        matchup_code, alt_matchup_code = f"{away_code}{home_code}", f"{home_code}{away_code}"
        bucket_info = kalshi.get_markets_for_date_token(league_key, date_token, status=status)
        all_markets = bucket_info.get("all_markets") or []
        
        strict_prefix = f"KXNBAGAME-{date_token}"
        matchup_candidates = [m for m in all_markets if str(m.get("event_ticker") or "").upper().startswith(strict_prefix) and (matchup_code in str(m.get("event_ticker") or "").upper() or alt_matchup_code in str(m.get("event_ticker") or "").upper())]

        if not matchup_candidates:
            return KalshiMatchResult(matched=False, kalshi_available=True, label="", probability=None, raw_event_id=None, league=league_key, reason="no_strict_match")

        matchup_candidates.sort(key=lambda x: abs((kalshi._best_market_time(x) - game_dt_utc).total_seconds()) if kalshi._best_market_time(x) else 999999)
        exact_match = matchup_candidates[0]
        
        prob = price_to_prob(exact_match.get("last_price"))
        return KalshiMatchResult(matched=True, kalshi_available=True, label=str(exact_match.get("title") or ""), probability=prob, raw_event_id=exact_match.get("event_ticker") or exact_match.get("ticker"), league=league_key, reason="matched_exact_event_ticker", market_type="winner", game_date=game_dt_utc)

    # Non-NBA Fuzzy Logic
    home_norm, away_norm = normalize_name(home_team), normalize_name(away_team)
    home_codes, away_codes = _build_team_codes(home_team), _build_team_codes(away_team)
    markets = kalshi.get_markets(status=status)
    
    best_market, best_score = None, 0.0
    for m in markets:
        meta = _parse_market_metadata(m)
        if not meta or (LEAGUE_SERIES_MAP.get(league_key) and not (str(m.get("ticker") or "").upper().startswith(LEAGUE_SERIES_MAP.get(league_key)))): continue
        teams = meta.get("teams") or []
        if len(teams) < 2: continue
        score = max(_team_score(teams[0], home_norm, home_codes) + _team_score(teams[1], away_norm, away_codes), _team_score(teams[0], away_norm, away_codes) + _team_score(teams[1], home_norm, home_codes))
        if score > best_score: best_score, best_market = score, {**m, "__meta": meta}

    if not best_market or best_score < TEAM_FUZZY_THRESHOLD:
        return KalshiMatchResult(matched=False, kalshi_available=True, label="", probability=None, raw_event_id=None, reason=f"low_score_{best_score:.1f}")

    meta = best_market["__meta"]
    return KalshiMatchResult(matched=True, kalshi_available=True, label=meta["title"], probability=meta["probability"], raw_event_id=best_market.get("ticker"), league=league_key, reason="matched", market_type=meta["market_type"], game_date=meta["market_date"])

# ---------------------------------------------------------------------------
# KalshiIntegrator Class
# ---------------------------------------------------------------------------

class KalshiIntegrator:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, *, required: bool = False) -> None:
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY") or os.getenv("KALSHI_API_KEY")
        raw_secret = api_secret or st.secrets.get("KALSHI_API_SECRET") or os.getenv("KALSHI_API_SECRET")
        self.api_secret_pem = self._normalize_secret(raw_secret)
        self.api_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.required = required
        self._markets_cache = []
        self._markets_cache_ts = 0.0
        self.cache_ttl_seconds = 120
        self._markets_cache_by_key = {}
        self._markets_cache_ttl_seconds = 600
        self.session = requests.Session()
        self.last_error_info, self.last_status_code, self.last_response_text, self.last_request_params = {}, None, None, None

    @staticmethod
    def _normalize_secret(secret_val: Optional[str]) -> Optional[str]:
        if not secret_val: return None
        cleaned = str(secret_val).replace("\\n", "\n").strip()
        if "-----BEGIN" in cleaned: return cleaned
        return "-----BEGIN PRIVATE KEY-----\n" + cleaned + "\n-----END PRIVATE KEY-----"

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        if not self.api_secret_pem: return ""
        msg_string = f"{timestamp}{method}{path}"
        try:
            private_key = serialization.load_pem_private_key(self.api_secret_pem.encode("utf-8"), password=None)
            signature = private_key.sign(msg_string.encode("utf-8"), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
            return base64.b64encode(signature).decode("utf-8")
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return ""

    def _request(self, method: str, path: str, params: Optional[Dict] = None, json_body: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.api_key or not self.api_secret_pem: raise RuntimeError("Kalshi missing keys.")
        url = f"{self.api_url}{path}"
        path_for_signing = f"/trade-api/v2{path}"
        backoff = 1.0
        self.last_request_params = params or json_body

        for i in range(5):
            timestamp = str(int(time.time() * 1000))
            signature = self._sign_request(method, path_for_signing, timestamp)
            headers = {"Content-Type": "application/json", "KALSHI-ACCESS-KEY": self.api_key, "KALSHI-ACCESS-SIGNATURE": signature, "KALSHI-ACCESS-TIMESTAMP": timestamp}
            try:
                resp = self.session.request(method, url, headers=headers, params=params, json=json_body, timeout=10)
                self.last_status_code, self.last_response_text = resp.status_code, resp.text[:1000]
                if resp.status_code == 429:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if i == 4: raise KalshiAPIError(f"Kalshi request failed: {e}")
                time.sleep(backoff)
                backoff *= 2
        return {}

    def get_markets_paginated(self, status: Optional[str] = None, limit: int = 200, max_pages: int = 5, cursor: Optional[str] = None, extra_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        all_markets, next_cursor, pages = [], cursor, 0
        while pages < max_pages:
            params = {"limit": limit, "cursor": next_cursor}
            norm_s = normalize_status(status)
            if norm_s: params["status"] = norm_s
            if extra_params: params.update(extra_params)
            data = self._request("GET", "/markets", params=params)
            chunk = data.get("markets", [])
            all_markets.extend(chunk)
            next_cursor = data.get("cursor")
            if not next_cursor or not chunk: break
            pages += 1
        return all_markets

    def get_league_markets(self, league: str, status: Optional[str] = None, min_prefix_hits: int = 200, max_pages: int = 5) -> List[Dict[str, Any]]:
        league_key = league.upper()
        prefix = LEAGUE_SERIES_MAP.get(league_key)
        targets = prefix if isinstance(prefix, list) else [prefix] if prefix else []
        collected = {}
        for series in targets:
            chunk = self.get_markets_paginated(status=status, extra_params={"series_ticker": series}, max_pages=max_pages)
            for m in chunk: collected[str(m.get("ticker") or m.get("event_ticker") or "")] = m
        return list(collected.values())

    def get_markets(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        now = time.time()
        if self._markets_cache and (now - self._markets_cache_ts) < self.cache_ttl_seconds: return self._markets_cache
        all_markets = self.get_markets_paginated(status=status)
        self._markets_cache, self._markets_cache_ts = all_markets, now
        return all_markets

    def get_markets_for_date_token(self, league: str, date_token: str, status: Optional[str] = None) -> Dict[str, Any]:
        all_markets = self.get_league_markets(league, status=status)
        bucket = [m for m in all_markets if date_token in str(m.get("event_ticker") or "").upper()]
        return {"bucket": bucket, "all_markets": all_markets}

    def _best_market_time(self, m: Dict[str, Any]) -> Optional[datetime]:
        for k in ["expected_expiration_time", "close_time", "expiration_time"]:
            val = m.get(k)
            if not val: continue
            try:
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                return dt if dt.tzinfo else pytz.utc.localize(dt)
            except: continue
        return None

    def split_market_kinds(self, markets: List[Dict[str, Any]], league: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        single, multi, other = [], [], []
        for m in (markets or []):
            t = str(m.get("ticker") or "").upper()
            if t.startswith("KXNBAGAME"): single.append(m)
            else: other.append(m)
        return {"single_game_candidates": single, "multivariate_bundles": multi, "other": other}

    def assert_available(self) -> None:
        if not self.api_key or not self.api_secret_pem: raise RuntimeError("Kalshi keys missing.")
    
    def get_sports_markets(self, league: str) -> List[Dict[str, Any]]:
        return self.get_league_markets(league)
