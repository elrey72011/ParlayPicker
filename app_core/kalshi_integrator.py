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

LEAGUE_SERIES_MAP: Dict[str, Any] = {
    "NBA": [
        "KXNBAGAME",  # Pull single-game slates; KXNBA alone only returns futures
        "KXNBA",  # Keep legacy futures as a secondary source for health/coverage
    ],
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
NBA_TZ = pytz.timezone("America/New_York")
# Do not fallback to cross-date or fuzzy matches; only same-day, exact event tickers
ALLOW_SAME_DAY_TEXT_FALLBACK = False

# Strict NBA code mapping for winner event tickers
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

def match_game_to_kalshi(league: str, home_team: str, away_team: str, game_time: Optional[datetime], integrator: "KalshiIntegrator" = None, status: Optional[str] = None) -> KalshiMatchResult:
    league_key = (league or "").upper()
    kalshi = integrator or KalshiIntegrator()
    
    if not kalshi or not kalshi.api_key:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, reason="no_integrator")

    def _nba_code(team: str) -> Optional[str]:
        norm = normalize_name(team)
        return NBA_TEAM_CODE_MAP.get(norm)

    def _nba_date_token(dt: datetime) -> str:
        """Kalshi winner tickers use the UTC date token (YYMONDD)."""
        dt_utc = dt.astimezone(pytz.UTC)
        return dt_utc.strftime("%y%b%d").upper()

    def _nba_bucket_and_match() -> KalshiMatchResult:
        if not isinstance(game_time, datetime):
            return KalshiMatchResult(
                matched=False,
                kalshi_available=True,
                label="",
                probability=None,
                raw_event_id=None,
                league=league_key,
                reason="missing_game_time",
                debug={"note": "no datetime for NBA match"},
            )
        game_dt = game_time
        if game_dt.tzinfo is None:
            game_dt = NBA_TZ.localize(game_dt)
        game_dt_utc = game_dt.astimezone(pytz.UTC)
        date_token = _nba_date_token(game_dt_utc)
        away_code = _nba_code(away_team)
        home_code = _nba_code(home_team)
        if not away_code or not home_code:
            return KalshiMatchResult(
                matched=False,
                kalshi_available=True,
                label="",
                probability=None,
                raw_event_id=None,
                league=league_key,
                reason="missing_team_code",
                debug={
                    "date_token": date_token,
                    "away_code": away_code,
                    "home_code": home_code,
                },
            )

        matchup_code = f"{away_code}{home_code}"
        alt_matchup_code = f"{home_code}{away_code}"
        bucket_info = kalshi.get_markets_for_date_token(
            league_key, date_token, status=status
        )
        # Use all markets to allow fallback when the date bucket is missing but still require the matchup code
        all_markets = bucket_info.get("all_markets") or []
        meta = bucket_info.get("meta", {})

        # De-dupe by event_ticker and enforce strict UTC date + matchup code
        deduped: Dict[str, Dict[str, Any]] = {}
        for m in all_markets:
            et = str(m.get("event_ticker") or m.get("ticker") or "")
            if et and et not in deduped:
                deduped[et] = m
        markets = list(deduped.values())

        strict_prefix = f"KXNBAGAME-{date_token}"
        matchup_candidates: List[Dict[str, Any]] = []
        for m in markets:
            et_upper = str(m.get("event_ticker") or "").upper()
            if not et_upper.startswith(strict_prefix):
                continue
            if matchup_code not in et_upper and alt_matchup_code not in et_upper:
                continue
            matchup_candidates.append(m)

        debug_info = {
            "date_token": date_token,
            "away_code": away_code,
            "home_code": home_code,
            "matchup_code": matchup_code,
            "kalshi_date_token_used": date_token,
            "candidate_event_tickers": [
                f"KXNBAGAME-{date_token}{away_code}{home_code}",
                f"KXNBAGAME-{date_token}{home_code}{away_code}",
            ],
            "bucket_meta": meta,
            "matchup_candidate_count": len(matchup_candidates),
        }

        if not matchup_candidates:
            return KalshiMatchResult(
                matched=False,
                kalshi_available=True,
                label="",
                probability=None,
                raw_event_id=None,
                league=league_key,
                reason="no_strict_kalshi_game_match_for_utc_date",
                debug=debug_info,
            )

        window_lower = game_dt_utc - timedelta(hours=12)
        window_upper = game_dt_utc + timedelta(hours=36)
        timed_candidates: List[Tuple[float, Dict[str, Any]]] = []
        for m in matchup_candidates:
            mt = kalshi._best_market_time(m)
            if not mt:
                continue
            if not (window_lower <= mt <= window_upper):
                continue
            diff_hours = abs((mt - game_dt_utc).total_seconds()) / 3600.0
            timed_candidates.append((diff_hours, m))

        debug_info["candidates_in_window"] = len(timed_candidates)
        if not timed_candidates:
            return KalshiMatchResult(
                matched=False,
                kalshi_available=True,
                label="",
                probability=None,
                raw_event_id=None,
                league=league_key,
                reason="no_candidate_in_time_window",
                debug=debug_info,
            )

        timed_candidates.sort(key=lambda x: x[0])
        exact_match = timed_candidates[0][1]

        yes_bid = exact_match.get("yes_bid")
        yes_ask = exact_match.get("yes_ask")
        probability = None
        try:
            vals = [v for v in [yes_bid, yes_ask] if v is not None]
            if len(vals) == 2:
                probability = max(0.0, min(1.0, ((float(vals[0]) + float(vals[1])) / 2) / 100))
            elif len(vals) == 1:
                probability = max(0.0, min(1.0, float(vals[0]) / 100))
            else:
                probability = price_to_prob(exact_match.get("last_price"))
        except Exception:
            probability = price_to_prob(exact_match.get("last_price"))

        debug_info["matched_event_ticker"] = exact_match.get("event_ticker")
        debug_info["matched_side_ticker"] = exact_match.get("ticker")
        debug_info["matched_close_time"] = str(exact_match.get("close_time"))
        return KalshiMatchResult(
            matched=True,
            kalshi_available=True,
            label=str(exact_match.get("title") or ""),
            probability=probability,
            raw_event_id=exact_match.get("event_ticker") or exact_match.get("ticker"),
            league=league_key,
            reason="matched_exact_event_ticker",
            market_type="winner",
            game_date=game_dt.astimezone(pytz.UTC),
            debug=debug_info,
        )

    if league_key == "NBA":
        return _nba_bucket_and_match()

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
        if not meta:
            continue

        ticker = (m.get("ticker") or "").upper()
        if series_prefix and not ticker.startswith(series_prefix):
            continue

        teams = meta.get("teams") or []
        if len(teams) < 2:
            continue

        score_home_first = _team_score(teams[0], home_norm, home_codes) + _team_score(teams[1], away_norm, away_codes)
        score_away_first = _team_score(teams[0], away_norm, away_codes) + _team_score(teams[1], home_norm, home_codes)
        score = max(score_home_first, score_away_first)

        m_date = meta.get("market_date")
        if game_dt_utc and m_date:
            try:
                if m_date.tzinfo is None:
                    m_date = pytz.utc.localize(m_date)
                diff = abs((m_date.date() - game_dt_utc.date()).days)
                if diff > DATE_TOLERANCE_DAYS:
                    continue
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
        game_date=meta["market_date"],
    )

# ---------------------------------------------------------------------------
# KalshiIntegrator Class
# ---------------------------------------------------------------------------

class KalshiIntegrator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        required: bool = False,
    ) -> None:
        # Resolve key + secret from args, env, or Streamlit secrets (only the official names)
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY") or os.getenv("KALSHI_API_KEY")
        raw_secret = api_secret or st.secrets.get("KALSHI_API_SECRET") or os.getenv("KALSHI_API_SECRET")

        # --- Clean private key format ---
        self.api_secret_pem = self._normalize_secret(raw_secret)

        self.api_url = "https://api.elections.kalshi.com/trade-api/v2"

        # Required flag
        self.required = required

        # Caching + error state
        self._markets_cache: List[Dict[str, Any]] = []
        self._markets_cache_ts: float = 0.0
        self.cache_ttl_seconds: int = 120
        self._markets_cache_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._markets_cache_ttl_seconds: int = 600
        self.last_error: Optional[str] = None
        self._league_cache: Dict[str, Dict[str, Any]] = {}
        self._league_cache_ttl: int = 300
        self.last_fetch_meta: Dict[str, Any] = {}
        self.session = requests.Session()
        self.last_error_info: Dict[str, Any] = {}
        self.last_status_code: Optional[int] = None
        self.last_response_text: Optional[str] = None
        self.last_request_params: Optional[Dict[str, Any]] = None

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

    def assert_available(self) -> None:
        if not self.api_key or not self.api_secret_pem:
            raise RuntimeError("Kalshi is required but missing keys in secrets.")
        health = self.health_check()
        if not health.get("ok"):
            raise RuntimeError(
                health.get("error")
                or "Kalshi is required but unavailable (auth/keys/API)."
            )

    def health_check(self) -> Dict[str, Any]:
        configured = bool(self.api_key and self.api_secret_pem)
        if not configured:
            return {
                "configured": False,
                "ok": False,
                "market_count": 0,
                "sample_market": None,
                "has_game_markets": False,
                "has_futures_markets": False,
                "error": "Kalshi is required but not configured.",
            }

        try:
            data = self._request("GET", "/markets", params={"limit": 50})
            try:
                markets = (data.get("markets") or []) if isinstance(data, dict) else []
            except Exception:
                markets = []
            if not markets and self.last_response_text:
                try:
                    parsed = json.loads(self.last_response_text)
                    if isinstance(parsed, dict):
                        markets = parsed.get("markets") or markets
                except Exception:
                    markets = markets or []

            markets = markets or []

            def _ticker(m: Dict[str, Any]) -> str:
                return str(m.get("event_ticker") or m.get("ticker") or "").upper()

            has_game = any(_ticker(m).startswith("KXNBAGAME") for m in markets)
            has_futures = any(
                _ticker(m).startswith("KXNBA") and not _ticker(m).startswith("KXNBAGAME")
                for m in markets
            )
            ok = True
            warning: Optional[str] = None
            if not has_game:
                warning = "Kalshi reachable, but no NBA KXNBAGAME markets returned (futures-only or slate not listed)."
            return {
                "configured": True,
                "ok": ok,
                "market_count": len(markets),
                "sample_market": markets[0] if markets else None,
                "has_game_markets": has_game,
                "has_futures_markets": has_futures,
                "warning": warning,
                "markets": markets,
                "error": None,
                "status_code": self.last_status_code,
                "response_text": (self.last_response_text or "")[:500],
            }
        except Exception as exc:  # pragma: no cover - defensive
            info = self.last_error_info or {}
            return {
                "configured": True,
                "ok": False,
                "market_count": 0,
                "sample_market": None,
                "has_game_markets": False,
                "has_futures_markets": False,
                "error": str(exc),
                "status_code": info.get("status_code"),
                "response_text": info.get("response_text"),
            }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Authenticated request with retry/backoff handling for rate limits and server errors.
        """

        if not self.api_key or not self.api_secret_pem:
            raise RuntimeError("Kalshi is required but missing keys in secrets.")

        url = f"{self.api_url}{path}"
        path_for_signing = f"/trade-api/v2{path}"

        def backoff_delay(base: float) -> float:
            return base + random.uniform(0, max(0.1, base * 0.25))

        retry_429 = 0
        retry_other = 0
        backoff = 1.0

        self.last_status_code = None
        self.last_response_text = None
        self.last_request_params = params or json or None

        while True:
            timestamp = str(int(time.time() * 1000))
            signature = self._sign_request(method, path_for_signing, timestamp)
            headers = {
                "Content-Type": "application/json",
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-SIGNATURE": signature,
                "KALSHI-ACCESS-TIMESTAMP": timestamp,
            }
            try:
                resp = self.session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    timeout=10,
                )
            except requests.Timeout as exc:
                retry_other += 1
                if retry_other > 3:
                    raise RuntimeError(f"Kalshi request timeout: {exc}")
                time.sleep(backoff_delay(backoff))
                backoff = min(backoff * 2, 10.0)
                continue
            except Exception as exc:
                retry_other += 1
                if retry_other > 3:
                    raise RuntimeError(f"Kalshi request error: {exc}")
                time.sleep(backoff_delay(backoff))
                backoff = min(backoff * 2, 10.0)
                continue

            status = resp.status_code
            self.last_status_code = status
            self.last_response_text = resp.text[:1000]
            if status == 429:
                retry_429 += 1
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after)
                except Exception:
                    delay = backoff
                if retry_429 > 6:
                    self.last_error_info = {
                        "status_code": status,
                        "response_text": resp.text[:300],
                    }
                    raise KalshiRateLimitError("Kalshi rate limited (429)")
                time.sleep(max(0.5, backoff_delay(min(delay, 32.0))))
                backoff = min(backoff * 2, 32.0)
                continue
            if 500 <= status < 600:
                retry_other += 1
                if retry_other > 3:
                    self.last_error_info = {
                        "status_code": status,
                        "response_text": resp.text[:300],
                    }
                    raise KalshiAPIError(f"Kalshi server error {status}")
                time.sleep(backoff_delay(min(backoff, 5.0)))
                backoff = min(backoff * 2, 20.0)
                continue
            if status >= 400:
                self.last_error_info = {
                    "status_code": status,
                    "response_text": resp.text[:300],
                }
                if status in (401, 403):
                    raise KalshiAPIError("Kalshi auth failed (401/403). Check key/secret.")
                raise KalshiAPIError(f"Kalshi API error {status}: {resp.text[:300]}")

            try:
                data = resp.json()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Kalshi response parse error: {exc}")
            self.last_error_info = {"status_code": status, "response_text": None}
            return data

    @staticmethod
    def normalize_status(status: Optional[str]) -> Optional[str]:
        """Sanitize caller-provided status to only supported Kalshi values."""
        if not isinstance(status, str):
            return None
        status_clean = status.strip().lower()
        if not status_clean:
            return None
        if status_clean == "open":  # Kalshi rejects "open"; treat as omit
            return None
        if status_clean == "final":
            status_clean = "finalized"
        allowed = {"active", "closed", "finalized", "settled"}
        return status_clean if status_clean in allowed else None

    @staticmethod
    def _status_param(status: Optional[str]) -> Dict[str, Any]:
        """Return a valid status parameter for /markets calls."""
        normalized = KalshiIntegrator.normalize_status(status)
        return {"status": normalized} if normalized else {}

    def _build_market_params(
        self,
        *,
        status: Optional[str],
        limit: Optional[int],
        cursor: Optional[str],
        extra_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if limit is not None and limit != "":
            params["limit"] = limit
        params.update(self._status_param(status))
        if cursor:
            params["cursor"] = cursor
        if extra_params:
            for key, val in extra_params.items():
                if val is None or val == "":
                    continue
                params[key] = val
        return params

    def get_markets_paginated(
        self,
        status: Optional[str] = None,
        limit: int = 200,
        max_pages: int = 5,
        cursor: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        all_markets: List[Dict[str, Any]] = []
        next_cursor = cursor
        pages = 0
        while pages < max_pages:
            params = self._build_market_params(
                status=status, limit=limit, cursor=next_cursor, extra_params=extra_params
            )
            self.last_request_params = params
            try:
                data = self._request("GET", "/markets", params=params)
            except KalshiAPIError:
                status = (self.last_error_info or {}).get("status_code")
                if status == 429:
                    logger.warning("Kalshi rate limited; returning cached markets where available.")
                    cached: List[Dict[str, Any]] = []
                    if self._markets_cache:
                        cached = list(self._markets_cache)
                    elif self._league_cache:
                        cached = next(iter(self._league_cache.values()), {}).get("markets", [])
                    return cached or all_markets
                raise
            markets = data.get("markets", []) or []
            all_markets.extend(markets)
            pages += 1
            next_cursor = (
                data.get("cursor")
                or data.get("next_cursor")
                or data.get("next")
                or data.get("next_token")
            )
            prefix_debug = {"page": pages, "received": len(markets), "cursor": bool(next_cursor)}
            logger.debug(f"Kalshi page debug: {prefix_debug}")
            if not next_cursor:
                break
            time.sleep(0.1)
        return all_markets

    def _get_nba_markets_targeted(
        self,
        *,
        status: Optional[str] = None,
        min_hits: int = 100,
        max_pages: int = 5,
    ) -> Dict[str, Any]:
        collected: Dict[str, Dict[str, Any]] = {}
        total_pages = 0
        total_hits = 0
        normalized_status = self.normalize_status(status)
        # Prioritize single-game slates before futures so we do not short-circuit on KXNBA finals
        # KXNBAGAME carries the daily slate winners; KXNBA alone mostly serves season-long futures.
        series_candidates = ["KXNBAGAME", "KXNBATOTAL", "KXNBASPREAD", "KXNBA"]
        for series in series_candidates:
            series_pages = 0
            series_hits = 0
            next_cursor: Optional[str] = None
            # Give game winners their own hit target; futures/totals should not prevent slate discovery.
            series_min_hits = 50 if series == "KXNBAGAME" else min_hits
            while series_pages < max_pages and series_hits < series_min_hits:
                params = {"limit": 200, "series_ticker": series}
                # NBA slate discovery should not be filtered out by status; omit invalid/"open" entirely.
                params.update(self._status_param(normalized_status))
                if next_cursor:
                    params["cursor"] = next_cursor
                try:
                    data = self._request("GET", "/markets", params=params)
                except KalshiAPIError:
                    status = (self.last_error_info or {}).get("status_code")
                    if status == 429:
                        logger.warning("Kalshi NBA targeted fetch rate limited; using cached data.")
                        cached = self._markets_cache or []
                        merged = list(collected.values()) or cached
                        return {"markets": merged, "pages": total_pages, "prefix_hits": total_hits}
                    raise
                chunk = data.get("markets", []) or []
                for m in chunk:
                    key = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                    if key and key not in collected:
                        collected[key] = m
                tickers = [str(m.get("ticker") or m.get("event_ticker") or "").upper() for m in chunk]
                series_hits += len([t for t in tickers if t.startswith(series)])
                series_pages += 1
                next_cursor = (
                    data.get("cursor")
                    or data.get("next_cursor")
                    or data.get("next")
                    or data.get("next_token")
                )
                if series == "KXNBAGAME" and series_hits >= series_min_hits:
                    break
                if not next_cursor:
                    break
                time.sleep(0.1)
            # Do not let futures (KXNBA) decide we are "done" when still seeking game markets
            total_pages += series_pages
            if series != "KXNBA":
                total_hits += series_hits

        deduped_markets_all = list(collected.values())
        deduped_markets = [
            m
            for m in deduped_markets_all
            if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith("KXNBAGAME-")
        ]
        if not deduped_markets:
            deduped_markets = deduped_markets_all
        return {
            "markets": deduped_markets,
            "pages": total_pages,
            "prefix_hits": total_hits,
        }

    def get_markets(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all markets with pagination support and a sane cap."""
        now = time.time()
        if self._markets_cache and (now - self._markets_cache_ts) < self.cache_ttl_seconds:
            logger.info(f"Using cached markets ({len(self._markets_cache)} items)")
            return self._markets_cache

        all_markets = self.get_markets_paginated(status=status)
        self._markets_cache = all_markets
        self._markets_cache_ts = now
        logger.info(f"✅ Successfully loaded {len(all_markets)} Kalshi markets (paginated)")
        return all_markets

    def get_league_markets(
        self,
        league: str,
        *,
        status: Optional[str] = None,
        min_prefix_hits: int = 200,
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        league_key = (league or "").upper()
        prefix = LEAGUE_SERIES_MAP.get(league_key)
        normalized_status = self.normalize_status(status)
        cache_key = f"{league_key}:{normalized_status or 'any'}"
        now = time.time()
        cached = self._league_cache.get(cache_key)
        if cached and (now - cached.get("ts", 0)) < self._league_cache_ttl:
            self.last_fetch_meta = cached.get("meta", {})
            return cached.get("markets", [])

        futures_noise: List[Dict[str, Any]] = []
        series_targets = (
            prefix
            if isinstance(prefix, list)
            else [prefix] if prefix else []
        )
        collected: Dict[str, Dict[str, Any]] = {}
        pages = 0
        prefix_hits = 0
        game_hits = 0
        futures_hits = 0

        if series_targets:
            for series in series_targets:
                series_params = {"series_ticker": series} if series else None
                series_status = normalized_status
                chunk = self.get_markets_paginated(
                    status=series_status,
                    limit=200,
                    max_pages=max_pages,
                    extra_params=series_params,
                )
                pages = max(pages, min(max_pages, len(chunk) // 200 + 1))
                for m in chunk or []:
                    key = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                    if key not in collected:
                        collected[key] = m
                    if league_key == "NBA":
                        if key.startswith("KXNBAGAME-"):
                            game_hits += 1
                        elif key.startswith("KXNBA-"):
                            futures_hits += 1
            prefix_hits = len(
                [
                    k
                    for k in collected
                    if any(k.startswith(str(s).upper()) for s in series_targets if s)
                ]
            )
        else:
            all_markets = self.get_markets_paginated(
                status=normalized_status, limit=200, max_pages=max_pages
            )
            for m in all_markets:
                key = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                collected[key] = m
                if league_key == "NBA":
                    if key.startswith("KXNBAGAME-"):
                        game_hits += 1
                    elif key.startswith("KXNBA-"):
                        futures_hits += 1
            pages = min(max_pages, len(all_markets) // 200 + 1)
            prefix_hits = (
                len([k for k in collected if prefix and k.startswith(prefix)])
                if prefix
                else 0
            )

        all_markets = list(collected.values())
        # If NBA still lacks game markets, sweep broadly so futures do not short-circuit slate discovery.
        if league_key == "NBA" and not any(
            str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith("KXNBAGAME-")
            for m in all_markets
        ):
            broad_chunk = self.get_markets_paginated(status=normalized_status, limit=200, max_pages=max_pages)
            for m in broad_chunk or []:
                key = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                if key.startswith("KXNBAGAME-") and key not in collected:
                    collected[key] = m
                    game_hits += 1
            all_markets = list(collected.values())
        if league_key == "NBA":
            for m in all_markets:
                ticker = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                if ticker.startswith("KXNBA-") and not ticker.startswith("KXNBAGAME-"):
                    futures_noise.append(m)
                if ticker.startswith("KXNBAGAME-"):
                    game_hits += 0
                elif ticker.startswith("KXNBA-"):
                    futures_hits += 0

        self.last_fetch_meta = {
            "league": league_key,
            "status": normalized_status,
            "status_param": bool(self._status_param(normalized_status)),
            "pages": pages,
            "total_markets": len(all_markets),
            "prefix_hits": prefix_hits,
            "prefix": prefix,
            "futures_noise": len(futures_noise) if league_key == "NBA" else None,
            "nba_game_hits": game_hits if league_key == "NBA" else None,
            "nba_futures_hits": futures_hits if league_key == "NBA" else None,
            "filtered_to_game_markets": None,
        }
        if not all_markets and not self.last_error_info:
            self.last_fetch_meta["note"] = "reachable_but_empty"
        self._league_cache[cache_key] = {
            "ts": now,
            "markets": all_markets,
            "meta": self.last_fetch_meta,
        }
        return all_markets

    def _filter_markets_for_league(self, markets: List[Dict[str, Any]], league: Optional[str]) -> List[Dict[str, Any]]:
        league_key = (league or "").upper()
        prefix = LEAGUE_SERIES_MAP.get(league_key)
        if not prefix:
            return markets

        ticker_upper = [str(m.get("ticker") or m.get("event_ticker") or "").upper() for m in markets]
        if isinstance(prefix, list):
            prefix_filtered = [
                m
                for m, t in zip(markets, ticker_upper)
                if any(t.startswith(pfx) for pfx in prefix if pfx)
            ]
        else:
            prefix_filtered = [m for m, t in zip(markets, ticker_upper) if t.startswith(prefix)]
        if prefix_filtered:
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

    @staticmethod
    def is_multivariate_bundle(market: Dict[str, Any]) -> bool:
        ticker = str(market.get("event_ticker") or market.get("ticker") or "").upper()
        if ticker.startswith("KXMV") or "MVE" in ticker:
            return True
        if market.get("mve_collection_ticker") or market.get("mve_selected_legs"):
            return True
        custom = str(market.get("custom_strike") or "")
        if custom and "Associated Events" in custom:
            return True
        return False

    def split_market_kinds(self, markets: List[Dict[str, Any]], league: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        league_key = (league or "").upper()
        prefix = LEAGUE_SERIES_MAP.get(league_key, "")
        single_game: List[Dict[str, Any]] = []
        multivariate: List[Dict[str, Any]] = []
        other: List[Dict[str, Any]] = []
        for m in markets or []:
            if self.is_multivariate_bundle(m):
                multivariate.append(m)
                continue
            t = str(m.get("event_ticker") or m.get("ticker") or "").upper()
            if league_key == "NBA":
                if t.startswith("KXNBAGAME-"):
                    single_game.append(m)
                else:
                    other.append(m)
                continue
            if isinstance(prefix, list):
                if any(t.startswith(pfx) for pfx in prefix):
                    single_game.append(m)
                else:
                    other.append(m)
                continue
            if prefix and t.startswith(prefix):
                single_game.append(m)
            elif not prefix:
                single_game.append(m)
            else:
                other.append(m)
        return {
            "single_game_candidates": single_game,
            "multivariate_bundles": multivariate,
            "other": other,
        }

    def get_sports_markets(
        self, league: Optional[str] = None, commence_times: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        try:
            league_key = (league or "").upper()
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
            date_key = None
            if commence_dt:
                earliest = min(commence_dt)
                date_key = earliest.astimezone(pytz.UTC).strftime("%Y%m%d")
            cache_key = None
            if league_key and date_key:
                cache_key = (league_key, date_key)
                cached = self._markets_cache_by_key.get(cache_key)
                if cached and (time.time() - cached.get("ts", 0)) < self._markets_cache_ttl_seconds:
                    return cached.get("markets", [])

            markets = self.get_league_markets(league_key, status=None)
            filtered = markets

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

            if cache_key:
                self._markets_cache_by_key[cache_key] = {
                    "ts": time.time(),
                    "markets": filtered,
                }

            return filtered
        except Exception:
            logger.error("Kalshi get_sports_markets failed", exc_info=True)
            return []

    def get_markets_for_league(self, league: str) -> List[Dict[str, Any]]:
        """Return markets for a given league without excluding game winners."""
        return self.get_sports_markets(league=league)

    def _filter_markets_by_date_token(
        self, markets: List[Dict[str, Any]], date_token: str
    ) -> List[Dict[str, Any]]:
        token_upper = (date_token or "").upper()
        if not token_upper:
            return []
        bucket: List[Dict[str, Any]] = []
        for m in markets or []:
            et_upper = str(m.get("event_ticker") or m.get("ticker") or "").upper()
            if et_upper.startswith(f"KXNBAGAME-{token_upper}"):
                bucket.append(m)
        return bucket

    def get_markets_for_date_token(
        self,
        league: str,
        date_token: str,
        *,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        league_key = (league or "").upper()
        cache_key = (league_key, date_token, status or "any")
        now = time.time()
        cached = self._markets_cache_by_key.get(cache_key)
        if cached and (now - cached.get("ts", 0)) < self._markets_cache_ttl_seconds:
            return cached.get("payload", {})

        base_markets = self.get_league_markets(league_key, status=status)
        bucket = self._filter_markets_by_date_token(base_markets, date_token)

        fetch_meta = {
            "league": league_key,
            "date_token": date_token,
            "initial_total": len(base_markets),
            "initial_date_token_count": len(bucket),
        }

        # If bucket empty, broaden with limited pagination and targeted prefix pull
        all_markets = list(base_markets)
        if not bucket:
            targeted_prefix = f"KXNBAGAME-{date_token}"
            try:
                targeted = self.get_markets_paginated(
                    status=status,
                    limit=200,
                    max_pages=5,
                    extra_params={"event_ticker_prefix": targeted_prefix},
                )
                all_markets.extend(targeted)
            except Exception:
                targeted = []
            extra = self.get_markets_paginated(
                status=status, limit=200, max_pages=5
            )
            all_markets.extend(extra)
            # De-dupe merged markets by event_ticker or ticker
            dedup: Dict[str, Dict[str, Any]] = {}
            for m in all_markets:
                key = str(m.get("event_ticker") or m.get("ticker") or "")
                if key and key not in dedup:
                    dedup[key] = m
            all_markets = list(dedup.values())
            bucket = self._filter_markets_by_date_token(all_markets, date_token)
            fetch_meta.update(
                {
                    "broadened_total": len(all_markets),
                    "broadened_date_token_count": len(bucket),
                    "targeted_prefix": targeted_prefix,
                    "targeted_added": len(targeted),
                }
            )

        # Final de-dupe for bucket by event_ticker
        final_bucket: Dict[str, Dict[str, Any]] = {}
        for m in bucket:
            key = str(m.get("event_ticker") or m.get("ticker") or "")
            if key and key not in final_bucket:
                final_bucket[key] = m

        # Date-token summary counts from all_markets (for debug)
        token_counts: Dict[str, int] = {}
        token_samples: Dict[str, List[str]] = {}
        for m in all_markets:
            et = str(m.get("event_ticker") or m.get("ticker") or "").upper()
            if "KXNBAGAME-" in et:
                try:
                    after = et.split("KXNBAGAME-")[1]
                    token = after[:7]
                    token_counts[token] = token_counts.get(token, 0) + 1
                    if len(token_samples.get(token, [])) < 10:
                        token_samples.setdefault(token, []).append(et)
                except Exception:
                    continue
        fetch_meta["token_counts"] = token_counts
        fetch_meta["token_samples"] = token_samples

        payload = {
            "bucket": list(final_bucket.values()),
            "all_markets": all_markets,
            "meta": fetch_meta,
        }
        self._markets_cache_by_key[cache_key] = {"ts": now, "payload": payload}
        return payload

    # Helpers required by app
    def get_game_markets_for_events(self, league):
        return self.get_markets_for_league(league)

    def filter_markets_closing_today(self, markets):
        return markets

    @staticmethod
    def price_to_prob(price: Any) -> Optional[float]:
        return price_to_prob(price)
    
    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}/orderbook") or {}

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
