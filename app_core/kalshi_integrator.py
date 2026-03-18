from __future__ import annotations

import json
import logging
import re
import time
import math
from datetime import timedelta
from dataclasses import dataclass
import os
from typing import Any, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


NCAAB_SUFFIX_TOKENS = {"bulldogs", "cowboys", "wildcats", "shockers"}

TEAM_NOISE_WORDS = {
    "university", "college", "state", "st", "inc", "team", "club",
    # Common mascots/noise words frequently omitted by prediction markets.
    "bulldogs", "tigers", "wildcats", "shockers", "eagles", "hawks", "bears", "lions", "panthers",
    "wolves", "wolfpack", "pack", "knights", "raiders", "pirates", "spartans", "trojans",
    "gators", "huskies", "cougars", "mustangs", "longhorns", "cowboys", "buckeyes",
    "crimson", "blue", "orange", "golden", "red", "white", "sox", "yankees", "dodgers",
    "cardinals", "rangers", "bruins", "leafs", "senators", "penguins", "capitals", "jets",
    "devils", "islanders", "kraken", "flames", "oilers", "canucks", "lightning", "stars",
    "avalanche", "coyotes", "ducks", "sharks", "predators", "sabres", "blackhawks",
}


def _safe_text(value: Any) -> str:
    """Stringify values without triggering pandas NA boolean ambiguity."""
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _normalized_merge_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def clean_team_name(name: Any) -> str:
    """Normalize team naming noise for robust fuzzy matching."""
    text = _safe_text(name).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in TEAM_NOISE_WORDS]
    return " ".join(tokens)

from core.team_mapper import normalize_team_name
API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

KALSHI_LINE_TOLERANCE_SPREAD = 3.0
KALSHI_LINE_TOLERANCE_TOTAL = 3.0

MAX_LINE_TOLERANCE = {
    "NBA": 2.5,
    "NCAAB": 2.5,
    "NHL": 0.5,
    "MLB": 0.5,
    "NFL": 2.5,
}

def _infer_market_family_from_text(title: str, subtitle: str = "", market_type_hint: str = "") -> str | None:
    combined_text = f"{_safe_text(title)} {_safe_text(subtitle)} {_safe_text(market_type_hint)}".lower()

    spread_terms = ("wins by", "spread")
    total_terms = ("points scored", "total points", "runs scored", "goals scored", "total", "over", "under")

    if any(term in combined_text for term in spread_terms):
        return "spread"
    if any(term in combined_text for term in total_terms):
        return "total"
    return None


def market_type_matches(market_type: str, title: str, subtitle: str = "") -> bool:
    market_type = _safe_text(market_type).lower()
    inferred = _infer_market_family_from_text(title, subtitle, market_type)

    if "total" in market_type:
        return inferred in {"total", None}
    if "spread" in market_type:
        return inferred in {"spread", None}

    # If explicit market_type is missing or ambiguous, trust keyword inference.
    return inferred is not None or not market_type

API_URL = API_BASE

LEAGUE_SERIES_MAP = {
    "NCAAB": {"spread": "KXNCAAMBSPREAD", "total": "KXNCAAMBTOTAL", "moneyline": "KXNCAABGAME"},
    "NBA": {"spread": "KXNBASPREAD", "total": "KXNBATOTAL", "moneyline": "KXNBAGAME"},
    "NHL": {"spread": "KXNHLSPREAD", "total": "KXNHLTOTAL", "moneyline": "KXNHLGAME"},
    "NFL": {"spread": "KXNFLSPREAD", "total": "KXNFLTOTAL", "moneyline": "KXNFLGAME"},
    "MLB": {"spread": "KXMLBSPREAD", "total": "KXMLBTOTAL", "moneyline": "KXMLBGAME"},
}

KALSHI_TEAM_CODES = {
    # NBA
    "Atlanta Hawks": "ATL", "Atlanta": "ATL",
    "Boston Celtics": "BOS", "Boston": "BOS",
    "Brooklyn Nets": "BKN", "Brooklyn": "BKN",
    "Charlotte Hornets": "CHA", "Charlotte": "CHA",
    "Chicago Bulls": "CHI", "Chicago": "CHI",
    "Cleveland Cavaliers": "CLE", "Cleveland": "CLE",
    "Dallas Mavericks": "DAL", "Dallas": "DAL",
    "Denver Nuggets": "DEN", "Denver": "DEN",
    "Detroit Pistons": "DET", "Detroit": "DET",
    "Golden State Warriors": "GSW", "Golden State": "GSW",
    "Houston Rockets": "HOU", "Houston": "HOU",
    "Indiana Pacers": "IND", "Indiana": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "L.A. Clippers": "LAC",
    "LA Lakers": "LAL", "Los Angeles Lakers": "LAL", "L.A. Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Memphis": "MEM",
    "Miami Heat": "MIA", "Miami": "MIA",
    "Milwaukee Bucks": "MIL", "Milwaukee": "MIL",
    "Minnesota Timberwolves": "MIN", "Minnesota": "MIN",
    "New Orleans Pelicans": "NOP", "New Orleans": "NOP",
    "New York Knicks": "NYK", "New York": "NYK",
    "Oklahoma City Thunder": "OKC", "Oklahoma City": "OKC",
    "Orlando Magic": "ORL", "Orlando": "ORL",
    "Philadelphia 76ers": "PHI", "Philadelphia": "PHI",
    "Phoenix Suns": "PHX", "Phoenix": "PHX",
    "Portland Trail Blazers": "POR", "Portland": "POR",
    "Sacramento Kings": "SAC", "Sacramento": "SAC",
    "San Antonio Spurs": "SAS", "San Antonio": "SAS",
    "Toronto Raptors": "TOR", "Toronto": "TOR",
    "Utah Jazz": "UTA", "Utah": "UTA",
    "Washington Wizards": "WAS", "Washington": "WAS",
    # NHL
    "Anaheim Ducks": "ANA", "Anaheim": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF", "Buffalo": "BUF",
    "Calgary Flames": "CGY", "Calgary": "CGY",
    "Carolina Hurricanes": "CAR", "Carolina": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ", "Columbus": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM", "Edmonton": "EDM",
    "Florida Panthers": "FLA", "Florida": "FLA",
    "Los Angeles Kings": "LAK", "Los Angeles": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL", "Montreal": "MTL",
    "Nashville Predators": "NSH", "Nashville": "NSH",
    "New Jersey Devils": "NJD", "New Jersey": "NJD",
    "New York Islanders": "NYI", "NY Islanders": "NYI",
    "New York Rangers": "NYR", "NY Rangers": "NYR",
    "Ottawa Senators": "OTT", "Ottawa": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "Pittsburgh": "PIT",
    "San Jose Sharks": "SJS", "San Jose": "SJS",
    "Seattle Kraken": "SEA", "Seattle": "SEA",
    "St. Louis Blues": "STL", "St. Louis": "STL",
    "Tampa Bay Lightning": "TBL", "Tampa Bay": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN", "Vancouver": "VAN",
    "Vegas Golden Knights": "VGK", "Vegas": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG", "Winnipeg": "WPG",
}

_KALSHI_TEAM_CODES_NORMALIZED = {
    re.sub(r"\s+", " ", k.lower().strip().replace(".", "")): v
    for k, v in KALSHI_TEAM_CODES.items()
}



KALSHI_NCAAB_TEAM_CODES = {
    "Merrimack": "MRMK",
}

TEAM_CODE_ALIASES = {
    "manhattan": "MAN",
    "wagner": "WAG",
    "princeton": "PRIN",
    "vermont": "UVM",
    "washington state": "WSU",
    "washington st": "WSU",
    "seton hall": "HALL",
    "st johns": "SJU",
    "state johns": "SJU",
    "saint johns": "SJU",
    "idaho state": "IDST",
    "sam houston": "SHSU",
    "sam houston state": "SHSU",
    "florida gulf coast": "FGCU",
    "kennesaw state": "KENN",
    "fairfield": "FAIR",
    "columbia": "CLMB",
    "pepperdine": "PEPP",
    "unc wilmington": "UNCW",
    "se louisiana": "SELA",
    "louisiana tech": "LT",
    "indiana state": "INST",
    "temple": "TEM",
    "rhode island": "URI",
    "saint marys": "SMC",
    "saint mary's": "SMC",
    "st marys": "SMC",
    "state marys": "SMC",
    "st mary's": "SMC",
    "wichita state": "WICH",
    "memphis": "MEM",
    "southern illinois": "SIU",
}

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


@dataclass
class KalshiMatchResult:
    market_ticker: str | None = None
    event_ticker: str | None = None
    market_title: str | None = None
    market_subtitle: str | None = None
    probability: float | None = None
    status: str = "no_match"
    reason: str = "no_market_for_tickers"


class KalshiAPIError(RuntimeError):
    pass


def _normalize_team_token(name: str) -> str:
    s = _safe_text(name).lower().strip()
    s = s.replace("\u2019", "'").replace("&", " and ")
    s = s.replace("-", " ").replace(".", " ").replace("'", "")
    s = re.sub(r"\bst\b", "state", s)
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"\bsoutheastern\b", "se", s)
    s = re.sub(r"\bnorthwestern\b", "nw", s)
    s = re.sub(r"\bnortheastern\b", "ne", s)
    s = re.sub(r"\bsouthwestern\b", "sw", s)
    s = re.sub(r"\bunc\b", "unc", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_team_for_kalshi(team_name: str) -> str:
    return _normalize_team_token(team_name)


def build_kalshi_date_code(game_date: Any) -> str:
    dt = pd.to_datetime(game_date, errors="coerce", utc=True)
    if pd.isna(dt):
        return ""
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return dt.strftime("%y%b%d").upper()

    dt_local = dt.tz_convert("America/New_York")
    return dt_local.strftime("%y%b%d").upper()


def _market_family(row: pd.Series) -> str | None:
    market_type_val = row.get("market_type")
    market_type = str(market_type_val).strip().lower() if pd.notna(market_type_val) else ""
    if market_type.startswith("spread"):
        return "spread"
    if market_type.startswith("total"):
        return "total"
    best_pick_val = row.get("best_pick")
    best_pick = str(best_pick_val).strip().lower() if pd.notna(best_pick_val) else ""
    if best_pick.startswith("over") or best_pick.startswith("under"):
        return "total"
    if best_pick:
        return "spread"
    return None


def _lookup_kalshi_team_code(team: str) -> str | None:
    raw = _safe_text(team).strip()
    if raw in KALSHI_TEAM_CODES:
        return KALSHI_TEAM_CODES[raw]
    normalized = re.sub(r"\s+", " ", raw.lower().replace(".", "").strip())
    return _KALSHI_TEAM_CODES_NORMALIZED.get(normalized)


def _guess_code(team: str, is_ncaab: bool = False, date_code: str = "") -> str | None:
    mapped = _lookup_kalshi_team_code(team)
    if mapped:
        return mapped
    team_text = _safe_text(team).strip()
    if team_text in KALSHI_NCAAB_TEAM_CODES:
        return KALSHI_NCAAB_TEAM_CODES[team_text]
    token = _normalize_team_token(team)
    if token in TEAM_CODE_ALIASES:
        return TEAM_CODE_ALIASES[token]
    words = [w for w in token.split() if w not in {"the", "of", "and", "university", "college", "state"}]
    if not words:
        return None
    if len(words) == 1:
        return words[0][:4].upper()
    return "".join(w[0] for w in words)[:4].upper()


def _fetch_event_markets_legacy(league: str, game_date: Any, home_team: str, away_team: str) -> dict[str, Any] | None:
    import os
    kalshi_api_key = os.environ.get("KALSHI_API_KEY", "")
    if not kalshi_api_key:
        return None
    date_obj = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(date_obj):
        return None
    date_str = date_obj.strftime("%y%b%d").upper()

    series_map = {
        "NBA": ["KXNBATOTAL", "KXNBASPREAD"],
        "NCAAB": ["KXNCAAMBTOTAL", "KXNCAAMBSPREAD"],
        "NHL": ["KXNHLTOTAL", "KXNHLSPREAD"],
    }
    series_list = series_map.get(str(league or "").upper(), [])

    home_code = _lookup_kalshi_team_code(home_team) or ""
    away_code = _lookup_kalshi_team_code(away_team) or ""

    headers = {"Authorization": f"Bearer {kalshi_api_key}"}
    for series in series_list:
        try:
            url = f"{API_BASE}/markets?series_ticker={series}&limit=100"
            resp = _make_kalshi_request(url, headers=headers, timeout=5)
            markets = resp.json().get("markets", [])
            for market in markets:
                ticker = str(market.get("ticker") or "")
                if date_str not in ticker:
                    continue
                # Bidirectional permutation check
                has_home = home_code and home_code in ticker
                has_away = away_code and away_code in ticker
                if has_home and has_away:
                    return market
                title = str(market.get("title") or "").lower()
                if str(home_team or "").lower() in title or str(away_team or "").lower() in title:
                    return market
        except Exception:
            continue
    return None




def _det_team_code(league: str, team: str) -> str | None:
    """Deterministic team-code resolver used by tests and ingestion code."""
    _ = league
    return _guess_code(team)


def league_series_ticker(league: str, market_type: str) -> str:
    league_upper = str(league or "").upper()
    market_lower = str(market_type or "").lower()

    # Map raw market types to standard families
    if "spread" in market_lower:
        family = "spread"
    elif "total" in market_lower or "over" in market_lower or "under" in market_lower:
        family = "total"
    elif "moneyline" in market_lower or "game" in market_lower:
        family = "moneyline"
    else:
        # Default fallback
        family = "spread"

    return str(LEAGUE_SERIES_MAP.get(league_upper, {}).get(family, ""))


def team_code_map(league: str, team: str) -> str:
    _ = league
    token = _normalize_team_token(team)
    if token in TEAM_CODE_ALIASES:
        return TEAM_CODE_ALIASES[token]
    return str(_guess_code(team) or "")


def team_code_for_league(league: str, team: str) -> str:
    code = _det_team_code(league, team)
    return str(code or "")


def _team_tokens_for_match(name: str) -> set[str]:
    normalized = _normalize_team_token(name)
    cleaned = clean_team_name(name)
    stop = {"the", "of", "and", "university", "college", "state", "team", "st"}
    normalized_tokens = {w for w in normalized.split() if len(w) > 2 and w not in stop}
    cleaned_tokens = {w for w in cleaned.split() if len(w) > 1 and w not in stop}
    return normalized_tokens.union(cleaned_tokens)




def _clean_ncaab_team_for_match(name: Any) -> str:
    text = _safe_text(name).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in NCAAB_SUFFIX_TOKENS]
    return " ".join(tokens)

def _event_match_score(event: dict[str, Any], home_team: str, away_team: str, league: str, date_code: str = "") -> int:
    title = str(event.get("title") or "")
    subtitle = str(event.get("sub_title") or "")
    ticker = str(event.get("event_ticker") or "")

    combined_norm = " ".join([
        _normalize_team_token(title),
        _normalize_team_token(subtitle),
        _normalize_team_token(ticker),
    ]).strip()
    combined_upper = f"{title} {subtitle} {ticker}".upper()
    combined_key = _normalized_merge_key(f"{title} {subtitle} {ticker}")

    home_norm = _normalize_team_token(home_team)
    away_norm = _normalize_team_token(away_team)
    home_clean = clean_team_name(home_team)
    away_clean = clean_team_name(away_team)
    home_key = _normalized_merge_key(home_team)
    away_key = _normalized_merge_key(away_team)
    home_tokens = _team_tokens_for_match(home_team)
    away_tokens = _team_tokens_for_match(away_team)

    home_code = team_code_for_league(league, home_team).upper()
    away_code = team_code_for_league(league, away_team).upper()

    score = 0

    # Strong signal: explicit Kalshi codes in ticker/title payload
    if home_code and home_code in combined_upper:
        score += 60
    if away_code and away_code in combined_upper:
        score += 60

    # Strong signal: canonical normalized names
    if home_norm and home_norm in combined_norm:
        score += 45
    if away_norm and away_norm in combined_norm:
        score += 45

    # Extra resilience for mascot/noise-token differences (e.g. Wichita St Shockers vs Wichita State)
    if home_clean and home_clean in clean_team_name(f"{title} {subtitle} {ticker}"):
        score += 35
    if away_clean and away_clean in clean_team_name(f"{title} {subtitle} {ticker}"):
        score += 35

    # Additional signal: fully sanitized alphanumeric keys reduce punctuation/spacing mismatch misses
    if home_key and home_key in combined_key:
        score += 12
    if away_key and away_key in combined_key:
        score += 12

    # Medium signal: token overlap allows abbreviation/morphology tolerance
    combined_tokens = set(combined_norm.split())
    if home_tokens:
        score += min(25, 10 * len(home_tokens.intersection(combined_tokens)))
    if away_tokens:
        score += min(25, 10 * len(away_tokens.intersection(combined_tokens)))

    # Penalize one-sided matches to avoid false positives
    has_home = (home_code and home_code in combined_upper) or (home_norm and home_norm in combined_norm) or bool(home_tokens.intersection(combined_tokens))
    has_away = (away_code and away_code in combined_upper) or (away_norm and away_norm in combined_norm) or bool(away_tokens.intersection(combined_tokens))
    if has_home ^ has_away:
        score -= 30

    # Date affinity improves precision while still allowing missing date fields.
    if date_code:
        dc = str(date_code).upper()
        if dc and dc in combined_upper:
            score += 25
        else:
            score -= 10

    return score


def _make_kalshi_request(url: str, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, timeout: int = 30) -> requests.Response:
    """Helper to make rate-limited Kalshi API requests with exponential backoff for 429 errors."""
    time.sleep(0.2)
    max_retries = 3
    backoff = 2.0

    for attempt in range(max_retries + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 429:
            if attempt < max_retries:
                logger.warning(f"Kalshi API 429 Too Many Requests. Retrying in {backoff} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(backoff)
                backoff *= 2.0
                continue
            else:
                logger.error(f"Kalshi API 429 Too Many Requests. Max retries ({max_retries}) exceeded.")
                resp.raise_for_status()
        else:
            resp.raise_for_status()
            return resp

    # Fallback return, should not reach here if raise_for_status happens above
    return requests.get(url, headers=headers, params=params, timeout=timeout)


def _get_markets(params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        response = _make_kalshi_request(f"{API_BASE}/markets", params=params, timeout=8)
        payload = response.json()
    except Exception as exc:
        raise KalshiAPIError(str(exc)) from exc
    if not isinstance(payload, dict):
        return []
    return payload.get("markets", [])


def api_get_markets(**params: Any) -> dict[str, Any]:
    """Compatibility wrapper for Kalshi market lookups."""
    try:
        response = _make_kalshi_request(f"{API_BASE}/markets", params=params, timeout=8)
        payload = response.json()
    except Exception as exc:
        raise KalshiAPIError(str(exc)) from exc
    return payload if isinstance(payload, dict) else {}


def _extract_markets(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        # We need to correctly handle the structure, checking "markets" first is standard for Kalshi V2
        # but the tests might return "data".
        markets = response.get("markets")
        if isinstance(markets, list):
            return markets
        markets = response.get("data")
        if isinstance(markets, list):
            return markets
    return []


def _extract_kalshi_line(mkt: dict[str, Any], is_total: bool) -> float | None:
    """
    Extracts the strike price from a Kalshi market.
    Primary source: subtitle and title.
    Fallback: ticker string (if total and ends with integer, append .5).
    """
    m_title = str(mkt.get("title") or "").lower()
    m_subtitle = str(mkt.get("subtitle") or "").lower()
    combined_text = f"{m_title} {m_subtitle}"

    # 1. Primary: Strict Regex Match based on betting terminology
    match = re.search(r'(?:Over|Under|by over|by at least)\s*(\d+(?:\.\d+)?)', combined_text, re.IGNORECASE)
    if match:
        val = abs(float(match.group(1)))
        if not is_total and "wins by" in combined_text:
            return val + 0.5
        return val

    # 2. Secondary Fallback: Original extraction purely from subtitle first, then combined text
    if m_subtitle:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", m_subtitle)
        for num_str in numbers:
            try:
                val = abs(float(num_str))
                if not is_total and "wins by" in m_subtitle:
                    return val + 0.5
                return val
            except ValueError:
                continue

    # Fallback to combined text
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", combined_text)
    for num_str in numbers:
        try:
            val = abs(float(num_str))
            # Spread translations (e.g. "wins by over 3.5")
            if not is_total and "wins by" in combined_text:
                return val + 0.5
            return val
        except ValueError:
            continue

    # 3. Tertiary: Fallback to ticker string
    ticker = str(mkt.get("ticker") or "").strip()
    if ticker:
        parts = ticker.split("-")
        if parts:
            last_part = parts[-1]
            if last_part.isdigit():
                try:
                    line = float(last_part)
                    if is_total:
                        # For totals, append .5 if it's an integer
                        if line == int(line):
                            line += 0.5
                    return line
                except ValueError:
                    pass
    return None

def _safe_float(val: Any) -> float:
    try:
        f_val = pd.to_numeric(val, errors="coerce")
        return 0.0 if pd.isna(f_val) else float(f_val)
    except (TypeError, ValueError):
        return 0.0

def _select_probability(market: dict[str, Any]) -> float | None:
    # Explicitly use float casting for fixed-point _dollars migration
    bid = _safe_float(market.get("yes_bid_dollars"))
    ask = _safe_float(market.get("yes_ask_dollars"))

    # Phase 2: Invalidate Kalshi market if bid-ask spread is > 20 cents, OR if order book is empty on one side
    # Since prices might be in dollars (e.g. 0.55) or cents (e.g. 55.0)
    # the threshold needs to match the magnitude.

    # If one side of the order book is completely empty (0 bids or 0 asks), it's illiquid
    if bid == 0.0 or ask == 0.0:
        logger.warning(f"Kalshi market illiquid: empty order book side (bid {bid}, ask {ask}).")
        return None

    bid_ask_spread = abs(bid - ask)
    if (bid + ask) > 2.0:
        # Values are in cents
        if bid_ask_spread > 20.0:
            logger.warning(f"Kalshi market illiquid: bid {bid}, ask {ask}. Spread > 20 cents.")
            return None
    else:
        # Values are in dollars
        if bid_ask_spread > 0.20:
            logger.warning(f"Kalshi market illiquid: bid {bid}, ask {ask}. Spread > 20 cents.")
            return None

    if (bid + ask) > 2.0:
        return float((bid + ask) / 200.0)
    else:
        return float((bid + ask) / 2.0)

    last = _safe_float(market.get("last_price_dollars"))

    if last > 0:
        if last > 1.0:
            return float(last / 100.0)
        return float(last)

    for key in ("yes_bid_dollars", "yes_ask_dollars"):
        val = _safe_float(market.get(key))

        if val > 0:
            if val > 1.0:
                return float(val / 100.0)
            return float(val)
    return 0.0




def _fetch_series_events(series_ticker: str) -> list[dict[str, Any]]:
    """Fetch all open events for a specific series with pagination."""
    events = []
    params = {"series_ticker": series_ticker, "status": "open", "limit": 100}

    try:
        # Loop up to 20 times for massive slates like NCAAB
        for _ in range(20):
            # Using _make_kalshi_request directly for the /events endpoint
            resp = _make_kalshi_request(f"{API_BASE}/events", params=params, timeout=8)
            payload = resp.json()

            page_events = payload.get("events", [])
            if not page_events:
                break

            events.extend(page_events)

            cursor = payload.get("cursor")
            if not cursor:
                break
            params["cursor"] = cursor

    except KalshiAPIError as exc:
        logger.warning(f"Kalshi series events fetch failed for {series_ticker}: {exc}")
    except Exception as exc:
        logger.warning(f"Kalshi series events fetch failed for {series_ticker}: {exc}")

    return events


def _fetch_series_cache(series_set: set[str], date_codes: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Fetch all open markets for each unique series in one call per series with pagination."""
    cache: dict[str, dict[str, Any]] = {}

    first_market_logged = False

    for series in series_set:
        try:
            params = {"series_ticker": series, "status": "open", "limit": 100}
            for _ in range(10):
                resp = api_get_markets(**params)
                markets = _extract_markets(resp)
                if not markets:
                    break

                if not first_market_logged and markets:
                    logger.info(f"KALSHI PAYLOAD VERIFICATION (First Market): {json.dumps(markets[0], indent=2)}")
                    first_market_logged = True

                for m in markets:
                    ticker = str(m.get("ticker") or "")
                    if not ticker:
                        continue
                    if date_codes and not any(dc in ticker for dc in date_codes):
                        continue
                    cache[ticker] = m

                cursor = resp.get("cursor") if isinstance(resp, dict) else None
                if not cursor:
                    break
                params["cursor"] = cursor
        except KalshiAPIError as exc:
            logger.warning("Kalshi series fetch failed for %s: %s", series, exc)
        except Exception as exc:
            logger.warning("Kalshi series fetch failed for %s: %s", series, exc)
    return cache





def _is_within_48h(item: dict[str, Any], game_date_obj: pd.Timestamp) -> bool:
    if pd.isna(game_date_obj):
        return True

    close_time_str = item.get("close_time") or item.get("expiration_time") or item.get("last_updated_ts")
    if not close_time_str:
        return True

    try:
        kalshi_dt = pd.to_datetime(close_time_str, utc=True)
        # Ensure game_date_obj is strictly timezone aware (UTC)
        if getattr(game_date_obj, 'tz', None) is None:
            game_date_obj = game_date_obj.tz_localize('UTC')
        else:
            game_date_obj = game_date_obj.tz_convert('UTC')

        return abs(kalshi_dt - game_date_obj) <= pd.Timedelta(hours=48)
    except Exception as e:
        logger.warning(f"Timezone matching error: {e}")
        return True


def enrich_with_kalshi_markets(best_picks_df: pd.DataFrame) -> pd.DataFrame:
    if best_picks_df is None or best_picks_df.empty:
        return best_picks_df.copy() if isinstance(best_picks_df, pd.DataFrame) else pd.DataFrame()

    out = best_picks_df.copy()
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce", utc=True)

    for col in [
        "kalshi_probability",
        "kalshi_market_title",
        "kalshi_event_ticker",
        "kalshi_market_ticker",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    out["kalshi_match_status"] = "miss"
    out["kalshi_match_reason"] = "no_market_for_tickers"

    for idx, row in out.iterrows():
        league_val = row.get("league")
        league = str(league_val).upper().strip() if pd.notna(league_val) else ""
        family_guess = _market_family(row)
        family = "spread" if family_guess == "spread" else "total"
        series = league_series_ticker(league, family)

        game_date = pd.to_datetime(row.get("game_date"), errors="coerce", utc=True)
        game_date_date = game_date.date() if pd.notna(game_date) else None
        if pd.notna(game_date):
            # If the time is exactly midnight UTC, it's a fallback date. Do NOT shift timezone.
            if game_date.hour == 0 and game_date.minute == 0 and game_date.second == 0:
                date_code = game_date.strftime("%y%b%d").upper()
            else:
                dt_local = game_date.tz_convert("America/New_York")
                date_code = dt_local.strftime("%y%b%d").upper()
        else:
            date_code = ""

        if not date_code:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "missing_date"
            continue
        if not series:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "missing_series"
            continue

        # Cache for series events
        if not hasattr(enrich_with_kalshi_markets, "series_cache"):
            enrich_with_kalshi_markets.series_cache = {}

        if series not in enrich_with_kalshi_markets.series_cache:
            enrich_with_kalshi_markets.series_cache[series] = _fetch_series_events(series)

        series_events = enrich_with_kalshi_markets.series_cache[series]

        home_team_val = row.get("home_team")
        away_team_val = row.get("away_team")
        home_team_name = str(home_team_val).strip() if pd.notna(home_team_val) else ""
        away_team_name = str(away_team_val).strip() if pd.notna(away_team_val) else ""
        if league == "NCAAB":
            home_team_name = _clean_ncaab_team_for_match(home_team_name)
            away_team_name = _clean_ncaab_team_for_match(away_team_name)

        best_event_match = None
        best_event_score = -1

        candidate_dates = set()
        if game_date_date is not None:
            candidate_dates = {
                game_date_date,
                game_date_date - timedelta(days=1),
                game_date_date + timedelta(days=1),
            }

        for event in series_events:
            if not _is_within_48h(event, game_date):
                continue

            if candidate_dates:
                close_time = event.get("close_time") or event.get("expiration_time")
                event_dt = pd.to_datetime(close_time, errors="coerce", utc=True)
                if pd.notna(event_dt) and event_dt.date() not in candidate_dates:
                    continue

            score = _event_match_score(event, home_team_name, away_team_name, league, date_code=date_code)
            if score > best_event_score:
                best_event_score = score
                best_event_match = event

        # Conservative acceptance threshold; tuned to reduce false misses for abbreviated Kalshi events.
        if best_event_score < 35:
            # last-chance fallback: accept strongest candidate if it clearly references both teams via code/name tokens
            if best_event_score < 20:
                best_event_match = None

        # Last-mile fallback: if there's exactly one near-time candidate event in the target series,
        # use it so line-level matching can decide match quality.
        if not best_event_match:
            near_time_events = []
            for e in series_events:
                if not _is_within_48h(e, game_date):
                    continue
                if candidate_dates:
                    close_time = e.get("close_time") or e.get("expiration_time")
                    event_dt = pd.to_datetime(close_time, errors="coerce", utc=True)
                    if pd.notna(event_dt) and event_dt.date() not in candidate_dates:
                        continue
                near_time_events.append(e)
            if len(near_time_events) == 1:
                best_event_match = near_time_events[0]
                out.at[idx, "kalshi_match_quality"] = "event_single_candidate_fallback"

        if not best_event_match:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "no_fuzzy_event_match"

            # Queue unmatched row for offline LLM resolution (Tier 5)
            queue_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "unmatched_queue.json")
            try:
                unmatched_list = []
                if os.path.exists(queue_file):
                    with open(queue_file, "r") as f:
                        unmatched_list = json.load(f)

                candidate_events = [
                    {
                        "title": str(e.get("title")),
                        "subtitle": str(e.get("sub_title"))
                    } for e in series_events if _is_within_48h(e, game_date)
                ]

                unmatched_list.append({
                    "home_team": home_team_name,
                    "away_team": away_team_name,
                    "game_date": str(game_date),
                    "league": league,
                    "candidates": candidate_events[:15] # Keep top 15 candidates for context
                })

                with open(queue_file, "w") as f:
                    json.dump(unmatched_list, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to append to unmatched_queue.json: {e}")

            continue

        event_ticker = best_event_match.get("event_ticker")
        if not event_ticker:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "matched_event_missing_ticker"
            continue

        # Step 5: Fetch exact event ticker with nested markets
        nested_markets = []
        try:
            url = f"{API_BASE}/events/{event_ticker}"
            # Extend timeout for large/nested event lookups
            resp = _make_kalshi_request(url, params={"with_nested_markets": "true"}, timeout=30)
            payload = resp.json()
            if payload and "event" in payload:
                nested_markets = payload["event"].get("markets", [])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Event ticker not found on Kalshi: {event_ticker}")
                out.at[idx, "kalshi_match_status"] = "miss"
                out.at[idx, "kalshi_match_reason"] = "event_not_found"
            else:
                logger.error(f"Error fetching event {event_ticker}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error fetching event {event_ticker}: {e}")
            continue

        if not nested_markets:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "no_markets_in_event"
            continue

        best_pick = _safe_text(row.get("best_pick")).strip()
        is_totals_query = "Over " in best_pick or "Under " in best_pick

        best_market = None
        best_delta = float("inf")
        match_reason = "no_market_for_tickers"
        match_status = "miss"

        if is_totals_query:
            match = re.search(r"[-+]?\s*(\d+(?:\.\d+)?)", best_pick)
            if not match:
                out.at[idx, "kalshi_match_status"] = "miss"
                out.at[idx, "kalshi_match_reason"] = "totals_line_unextractable"
                continue

            extracted_totals_line = abs(float(match.group(1)))

            # Since we fetched by precise event ticker, we don't need semantic team string matching.
            # Just verify it's a total.
            row_market_type = _safe_text(row.get("market_type"))
            totals_markets = [
                m for m in nested_markets
                if market_type_matches(row_market_type or "total", m.get("title"), m.get("subtitle"))
                or _infer_market_family_from_text(m.get("title"), m.get("subtitle"), row_market_type) == "total"
            ]

            kalshi_lines = []
            for mkt in totals_markets:
                k_line = _extract_kalshi_line(mkt, is_total=True)
                if k_line is not None:
                    bid = _safe_float(mkt.get("yes_bid_dollars"))
                    ask = _safe_float(mkt.get("yes_ask_dollars"))

                    if bid == 0.0 or ask == 0.0:
                        kalshi_prob = None
                    elif (bid + ask) > 2.0:
                        if abs(bid - ask) > 20.0:
                            kalshi_prob = None
                        else:
                            kalshi_prob = (bid + ask) / 200.0
                    elif bid > 0 and ask > 0:
                        if abs(bid - ask) > 0.20:
                            kalshi_prob = None
                        else:
                            kalshi_prob = (bid + ask) / 2.0
                    else:
                        kalshi_prob = _select_probability(mkt)

                    if kalshi_prob is not None:
                        kalshi_lines.append((k_line, kalshi_prob, mkt))

            if kalshi_lines:
                target_line_abs = abs(float(extracted_totals_line))
                nearest = min(kalshi_lines, key=lambda x: abs(float(x[0]) - target_line_abs))
                delta = abs(float(nearest[0]) - target_line_abs)

                tolerance = MAX_LINE_TOLERANCE.get(league, 3.0)
                if delta <= tolerance:
                    if delta == 0:
                        best_market = nearest[2]
                        match_status = "matched"
                        match_reason = "total_match_exact"
                        out.at[idx, "kalshi_line_diff"] = 0.0
                    else:
                        best_market = nearest[2]
                        match_status = "matched"
                        match_reason = "nearest_line_proxy"
                        out.at[idx, "kalshi_line_diff"] = delta
                else:
                    out.at[idx, "kalshi_match_status"] = "miss"
                    out.at[idx, "kalshi_match_reason"] = "alt_line_mismatch"
                    out.at[idx, "kalshi_match_quality"] = "line_mismatched"
                    continue

        else:
            # SPREAD LOGIC
            # Use spread-like markets inside the event and fallback by inferred text family.
            row_market_type = _safe_text(row.get('market_type'))
            markets = [
                m for m in nested_markets
                if market_type_matches(row_market_type, m.get('title'), m.get('subtitle'))
                or _infer_market_family_from_text(m.get('title'), m.get('subtitle'), row_market_type) == "spread"
            ]

            raw_spread_line = _safe_text(row.get("spread_line"))
            match = re.search(r"[-+]?\s*(\d+(?:\.\d+)?)", raw_spread_line)

            if not match:
                # Moneyline fallback
                is_ml_pick = _safe_text(row.get("market_type")).lower() == "moneyline"
                if is_ml_pick:
                    for mkt in markets:
                        m_title = str(mkt.get("title") or "").lower()
                        m_subtitle = str(mkt.get("subtitle") or "").lower()
                        combined_text = f"{m_title} {m_subtitle}"

                        if "moneyline" in combined_text or "to win" in combined_text:
                            pick_team_val = row.get("pick_team")
                            if pd.notna(pick_team_val):
                                pick_team = str(pick_team_val)
                            else:
                                home_team_val = row.get("home_team")
                                pick_team = str(home_team_val) if pd.notna(home_team_val) else ""
                            pick_team_norm = _normalize_team_token(pick_team)
                            if pick_team_norm and pick_team_norm in _normalize_team_token(m_title):
                                best_market = mkt
                                match_status = "matched"
                                match_reason = "moneyline_match"
                                break
            else:
                target_line = abs(float(match.group(1)))

                kalshi_lines = []
                for mkt in markets:
                    m_title = str(mkt.get("title") or "").lower()
                    m_subtitle = str(mkt.get("subtitle") or "").lower()
                    combined_text = f"{m_title} {m_subtitle}"

                    m_type = str(row.get("market_type")).lower()
                    book_line = pd.to_numeric(row.get("spread_line"), errors="coerce")
                    is_favorite_bet = book_line < 0

                    home_t = home_team_name
                    away_t = away_team_name

                    home_shared = {w for w in set(_normalize_team_token(home_t).split()).intersection(set(_normalize_team_token(combined_text).split())) if len(w) > 2}
                    away_shared = {w for w in set(_normalize_team_token(away_t).split()).intersection(set(_normalize_team_token(combined_text).split())) if len(w) > 2}

                    ticker_suffix = str(mkt.get("ticker", "")).split("-")[-1]

                    # Fuzzy match code relies on string intersection primarily now,
                    # but we can try to grab the code from the ticker itself if present
                    # However, we don't have home_code/away_code generated from the exact string generator anymore
                    # so we will rely more on the shared string tokens.
                    kalshi_subject_is_home = bool(home_shared)
                    kalshi_subject_is_away = bool(away_shared)

                    expected_subject_is_home = ("home" in m_type) if is_favorite_bet else not ("home" in m_type)

                    is_correct_match = (expected_subject_is_home and kalshi_subject_is_home) or \
                                       (not expected_subject_is_home and kalshi_subject_is_away)

                    if not kalshi_subject_is_home and not kalshi_subject_is_away:
                        is_correct_match = True

                    if is_correct_match:
                        k_line = _extract_kalshi_line(mkt, is_total=False)
                        if k_line is not None:
                            bid = _safe_float(mkt.get("yes_bid_dollars"))
                            ask = _safe_float(mkt.get("yes_ask_dollars"))
                            if bid == 0.0 or ask == 0.0:
                                kalshi_prob = None
                            elif (bid + ask) > 2.0:
                                if abs(bid - ask) > 20.0:
                                    kalshi_prob = None
                                else:
                                    kalshi_prob = (bid + ask) / 200.0
                            elif bid > 0 and ask > 0:
                                if abs(bid - ask) > 0.20:
                                    kalshi_prob = None
                                else:
                                    kalshi_prob = (bid + ask) / 2.0
                            else:
                                kalshi_prob = _select_probability(mkt)

                            if kalshi_prob is not None:
                                kalshi_lines.append((k_line, kalshi_prob, mkt))

                if kalshi_lines:
                    target_line_abs = abs(float(target_line))
                    nearest = min(kalshi_lines, key=lambda x: abs(float(x[0]) - target_line_abs))
                    delta = abs(float(nearest[0]) - target_line_abs)

                    tolerance = MAX_LINE_TOLERANCE.get(league, 3.0)
                    if delta <= tolerance:
                        if delta == 0:
                            best_market = nearest[2]
                            match_status = "matched"
                            match_reason = "spread_match_exact"
                            out.at[idx, "kalshi_line_diff"] = 0.0
                        else:
                            best_market = nearest[2]
                            match_status = "matched"
                            match_reason = "nearest_line_proxy"
                            out.at[idx, "kalshi_line_diff"] = delta
                    else:
                        out.at[idx, "kalshi_match_status"] = "miss"
                        out.at[idx, "kalshi_match_reason"] = "alt_line_mismatch"
                        out.at[idx, "kalshi_match_quality"] = "line_mismatched"
                        continue

        if best_market is None:
            # We found no markets or candidates at all
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "alt_line_mismatch"
            out.at[idx, "kalshi_match_quality"] = "line_mismatched"
        else:
            # If we interpolated the probability, use it directly
            if "_interpolated_probability" in best_market:
                kalshi_prob = best_market["_interpolated_probability"]
            else:
                bid = _safe_float(best_market.get("yes_bid_dollars"))
                ask = _safe_float(best_market.get("yes_ask_dollars"))

                if bid == 0.0 or ask == 0.0:
                    logger.warning(f"Kalshi market illiquid: empty order book side for {best_market.get('ticker')}")
                    kalshi_prob = None
                elif bid > 0 and ask > 0:
                    # Values are in dollars
                    if abs(bid - ask) > 0.20:
                        logger.warning(f"Kalshi market illiquid: spread > 0.20 for {best_market.get('ticker')}")
                        kalshi_prob = None
                    else:
                        kalshi_prob = (bid + ask) / 2.0
                else:
                    kalshi_prob = _select_probability(best_market)

            if kalshi_prob is None:
                out.at[idx, "kalshi_match_status"] = "miss"
                out.at[idx, "kalshi_match_reason"] = "illiquid_market"
                # Keep kalshi_probability pd.NA to fall back 100% to ML probability
                continue

            kalshi_prob = _safe_float(kalshi_prob)
            if kalshi_prob == 0.0:
                out.at[idx, "kalshi_match_status"] = "miss"
                out.at[idx, "kalshi_match_reason"] = "zero_probability"
                out.at[idx, "kalshi_probability"] = 0.0
                continue

            # NEW: Invert probability if we are betting the underdog or the under
            m_type = str(row.get("market_type")).lower()
            if "total_under" in m_type and kalshi_prob > 0:
                kalshi_prob = 1.0 - kalshi_prob
            elif "spread" in m_type:
                book_line = pd.to_numeric(row.get("spread_line"), errors="coerce")
                if pd.notna(book_line) and book_line > 0 and kalshi_prob > 0:
                    kalshi_prob = 1.0 - kalshi_prob

            # Sanity check: probabilities must be between 0 and 1
            if pd.isna(kalshi_prob) or kalshi_prob <= 0.0 or kalshi_prob > 1.0:
                logger.warning(f"⚠️ Invalid Kalshi probability {kalshi_prob} for {row.get('game_id', 'unknown')}, skipping row")
                out.at[idx, "kalshi_match_status"] = "error"
                out.at[idx, "kalshi_match_reason"] = f"invalid_probability_{kalshi_prob}"
                out.at[idx, "kalshi_probability"] = 0.0
                continue

            out.at[idx, "kalshi_probability"] = float(kalshi_prob)
            out.at[idx, "kalshi_market_title"] = best_market.get("title")
            out.at[idx, "kalshi_event_ticker"] = best_market.get("event_ticker")
            out.at[idx, "kalshi_market_ticker"] = best_market.get("ticker")
            out.at[idx, "kalshi_match_status"] = match_status
            out.at[idx, "kalshi_match_reason"] = match_reason
            out.at[idx, "kalshi_match_quality"] = "line_matched"

            # Dynamic EV Recalibration for Kalshi Alternate Lines
            # Scale probability down by roughly 2.5% per point of delta shift
            if "calibrated_probability" in out.columns and pd.notna(out.at[idx, "kalshi_line_diff"]):
                line_diff = float(out.at[idx, "kalshi_line_diff"])
                if line_diff > 0:
                    orig_prob = float(out.at[idx, "calibrated_probability"])

                    # Heuristic probability adjustment (can be tuned per sport)
                    # A shift of 1 point typically drops probability by ~0.025
                    adj_prob = orig_prob - (line_diff * 0.025)
                    adj_prob = max(0.01, min(0.99, adj_prob))

                    out.at[idx, "calibrated_probability"] = adj_prob

            # Recalculate Expected Value applying exact Kalshi Fees
            # EV = (P_win * (1 - P_contract - Fee)) - ((1 - P_win) * (P_contract + Fee))
            if "calibrated_probability" in out.columns:
                p_win = float(out.at[idx, "calibrated_probability"])
                p_contract = kalshi_prob

                # Assume standard order size of 1 contract for basic EV evaluation
                C = 1.0
                # Exact Kalshi Taker fee formula: math.ceil(0.07 * C * P * (1-P) * 100) / 100
                # Exact Kalshi Maker fee formula: math.ceil(0.0175 * C * P * (1-P) * 100) / 100
                # Fees are calculated in cents and then converted back to dollars
                # We use taker fee to be more conservative since maker orders may not fill
                raw_taker_fee_cents = 0.07 * C * p_contract * (1.0 - p_contract) * 100.0
                fee_dollars = math.ceil(raw_taker_fee_cents) / 100.0

                ev = (p_win * (1.0 - p_contract - fee_dollars)) - ((1.0 - p_win) * (p_contract + fee_dollars))
                out.at[idx, "expected_value"] = ev

                # Recalculate simple edge without fees for display
                out.at[idx, "edge"] = p_win - p_contract

    # NA-safe columns to prevent ambiguous boolean evaluation downstream.
    out["kalshi_probability"] = pd.to_numeric(out.get("kalshi_probability"), errors="coerce").fillna(0.0)
    out["is_matched"] = out.get("kalshi_match_status", "").astype(str).eq("matched").fillna(False).astype(bool)
    out["kalshi_probability_is_valid"] = out["kalshi_probability"].notna().fillna(False)

    return out




def canonical_team_name(team: str) -> str:
    return _safe_text(team).strip()


def match_nba_spread(row: dict[str, Any], markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    spread_pick_team_val = row.get("spread_pick_team")
    pick_team = canonical_team_name(str(spread_pick_team_val)) if pd.notna(spread_pick_team_val) else ""
    pick_line = pd.to_numeric(row.get("spread_pick_line"), errors="coerce")

    home_raw = row.get("Home")
    if pd.notna(home_raw):
        home_team = canonical_team_name(str(home_raw))
    else:
        home_fallback = row.get("home_team")
        home_team = canonical_team_name(str(home_fallback)) if pd.notna(home_fallback) else ""

    away_raw = row.get("Away")
    if pd.notna(away_raw):
        away_team = canonical_team_name(str(away_raw))
    else:
        away_fallback = row.get("away_team")
        away_team = canonical_team_name(str(away_fallback)) if pd.notna(away_fallback) else ""
    if pd.isna(pick_line) or not pick_team:
        return None

    is_favorite_pick = float(pick_line) < 0
    target_team = pick_team if is_favorite_pick else (away_team if pick_team == home_team else home_team)
    target_line = abs(float(pick_line)) - 0.5

    best_market = None
    best_delta = float("inf")
    for market in markets or []:
        yes_side = str(market.get("yes_side") or "")
        m = re.search(r"^(.+?)\s+wins\s+by\s+over\s+([0-9]+(?:\.[0-9]+)?)", yes_side, re.IGNORECASE)
        if not m:
            continue
        market_team = canonical_team_name(m.group(1))
        market_line = float(m.group(2))
        # Kalshi 'yes_side' uses just the city/nickname often, so we'll match by simple substring inclusion.
        if market_team.lower() not in target_team.lower() and target_team.lower() not in market_team.lower():
            continue
        delta = abs(market_line - target_line)
        if delta < best_delta:
            best_delta = delta
            best_market = market

    if best_market is None:
        return None

    prob = pd.to_numeric(best_market.get("probability"), errors="coerce")
    if pd.isna(prob):
        prob = pd.to_numeric(best_market.get("yes_bid_dollars"), errors="coerce")
    if pd.isna(prob):
        return None
    pick_prob = float(prob) if is_favorite_pick else float(1.0 - float(prob))
    return {"market": best_market, "kalshi_prob_for_pick": pick_prob}

class KalshiIntegrator:
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return enrich_with_kalshi_markets(df)
