from __future__ import annotations

import json
import logging
import re
import time
import math
from dataclasses import dataclass
import os
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from core.team_mapper import aggressive_sanitize_team_name

try:
    import rapidfuzz
    from rapidfuzz import fuzz
except ImportError:
    try:
        from thefuzz import fuzz
    except ImportError:
        import difflib
        class FuzzFallback:
            @staticmethod
            def token_set_ratio(s1, s2):
                if not s1 or not s2: return 0
                matcher = difflib.SequenceMatcher(None, s1, s2)
                return matcher.ratio() * 100
        fuzz = FuzzFallback()

logger = logging.getLogger(__name__)
API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

KALSHI_LINE_TOLERANCE_SPREAD = 1.5
KALSHI_LINE_TOLERANCE_TOTAL = 1.5

MAX_LINE_TOLERANCE = {
    "NBA": 1.5,
    "NCAAB": 1.5,
    "NHL": 1.5,
    "NFL": 1.5,
    "MLB": 1.5
}

def market_type_matches(market_type: str, title: str, subtitle: str = "") -> bool:
    market_type = str(market_type or '').lower()
    combined_text = f"{str(title or '')} {str(subtitle or '')}".lower()

    if 'total' in market_type:
        return any(word in combined_text for word in ['total', 'points', 'goals', 'over', 'under'])
    elif 'spread' in market_type:
        return 'wins by' in combined_text or 'covers' in combined_text
    return True

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
    s = str(name or "").lower().strip()
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
    market_type = str(row.get("market_type") or "").strip().lower()
    if market_type.startswith("spread"):
        return "spread"
    if market_type.startswith("total"):
        return "total"
    best_pick = str(row.get("best_pick") or "").strip().lower()
    if best_pick.startswith("over") or best_pick.startswith("under"):
        return "total"
    if best_pick:
        return "spread"
    return None


def _lookup_kalshi_team_code(team: str) -> str | None:
    raw = str(team or "").strip()
    if raw in KALSHI_TEAM_CODES:
        return KALSHI_TEAM_CODES[raw]
    normalized = re.sub(r"\s+", " ", raw.lower().replace(".", "").strip())
    return _KALSHI_TEAM_CODES_NORMALIZED.get(normalized)


def _guess_code(team: str, is_ncaab: bool = False, date_code: str = "") -> str | None:
    mapped = _lookup_kalshi_team_code(team)
    if mapped:
        return mapped
    if str(team or "").strip() in KALSHI_NCAAB_TEAM_CODES:
        return KALSHI_NCAAB_TEAM_CODES[str(team).strip()]
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

def _select_probability(market: dict[str, Any]) -> float:
    # Explicitly use float casting for fixed-point _dollars migration
    bid = _safe_float(market.get("yes_bid_dollars"))
    ask = _safe_float(market.get("yes_ask_dollars"))

    if bid > 0 or ask > 0:
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
        league = str(row.get("league") or "").upper()
        family_guess = _market_family(row)
        family = "spread" if family_guess == "spread" else "total"
        series = league_series_ticker(league, family)

        game_date = pd.to_datetime(row.get("game_date"), errors="coerce", utc=True)
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

        home_team_name = str(row.get("home_team") or "")
        away_team_name = str(row.get("away_team") or "")

        # Tier 2 & 4: Aggressive Sanitization and Elevated Probabilistic Match
        home_team_name_norm = aggressive_sanitize_team_name(home_team_name)
        away_team_name_norm = aggressive_sanitize_team_name(away_team_name)

        concatenated_teams = f"{away_team_name_norm} {home_team_name_norm}"

        best_event_match = None
        best_event_score = 0.0

        for event in series_events:
            # Date Verification (Tier 3 Temporal Bounding)
            if not _is_within_48h(event, game_date):
                continue

            e_title = str(event.get("title") or "")
            e_subtitle = str(event.get("sub_title") or "")
            combined_event_text = f"{e_title} {e_subtitle}"

            combined_sanitized = aggressive_sanitize_team_name(combined_event_text)

            # Use token_set_ratio to bypass positional reliance (Tier 4)
            score = fuzz.token_set_ratio(concatenated_teams, combined_sanitized)

            # Lowered threshold to 50 to bypass "Conference Tournament" prefix issues
            if score >= 50 and score > best_event_score:
                best_event_score = score
                best_event_match = event

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

        best_pick = str(row.get("best_pick") or "").strip()
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
            totals_markets = [
                m for m in nested_markets
                if market_type_matches("total", m.get("title"), m.get("subtitle"))
            ]

            kalshi_lines = []
            for mkt in totals_markets:
                k_line = _extract_kalshi_line(mkt, is_total=True)
                if k_line is not None:
                    bid = _safe_float(mkt.get("yes_bid_dollars"))
                    ask = _safe_float(mkt.get("yes_ask_dollars"))

                    if (bid + ask) > 2.0:
                        kalshi_prob = (bid + ask) / 200.0
                    elif bid > 0 and ask > 0:
                        kalshi_prob = (bid + ask) / 2.0
                    else:
                        kalshi_prob = _select_probability(mkt)

                    if kalshi_prob is not None:
                        kalshi_lines.append((k_line, kalshi_prob, mkt))

            if kalshi_lines:
                target_line_abs = abs(float(extracted_totals_line))
                nearest = min(kalshi_lines, key=lambda x: abs(float(x[0]) - target_line_abs))
                delta = abs(float(nearest[0]) - target_line_abs)

                tolerance = MAX_LINE_TOLERANCE.get(league, 1.5)
                # if abs(kalshi_line - odds_line) <= 1.5:
                if delta <= 1.5:
                    if delta == 0:
                        best_market = nearest[2]
                        match_status = "matched"
                        match_reason = "total_match_exact"
                        out.at[idx, "kalshi_line_diff"] = 0.0
                    else:
                        best_market = nearest[2]
                        match_status = "matched"
                        match_reason = "total_match_nearest"
                        out.at[idx, "kalshi_line_diff"] = delta
                else:
                    out.at[idx, "kalshi_match_status"] = "miss"
                    out.at[idx, "kalshi_match_reason"] = "alt_line_mismatch"
                    out.at[idx, "kalshi_match_quality"] = "line_mismatched"
                    continue

        else:
            # SPREAD LOGIC
            # Use all spread markets inside the event
            markets = [m for m in nested_markets if market_type_matches(row.get('market_type'), m.get('title'), m.get('subtitle'))]

            raw_spread_line = str(row.get("spread_line") or "")
            match = re.search(r"[-+]?\s*(\d+(?:\.\d+)?)", raw_spread_line)

            if not match:
                # Moneyline fallback
                is_ml_pick = str(row.get("market_type") or "").lower() == "moneyline"
                if is_ml_pick:
                    for mkt in markets:
                        m_title = str(mkt.get("title") or "").lower()
                        m_subtitle = str(mkt.get("subtitle") or "").lower()
                        combined_text = f"{m_title} {m_subtitle}"

                        if "moneyline" in combined_text or "to win" in combined_text:
                            pick_team_norm = _normalize_team_token(str(row.get("pick_team") or row.get("home_team")))
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
                            if (bid + ask) > 2.0:
                                kalshi_prob = (bid + ask) / 200.0
                            elif bid > 0 and ask > 0:
                                kalshi_prob = (bid + ask) / 2.0
                            else:
                                kalshi_prob = _select_probability(mkt)

                            if kalshi_prob is not None:
                                kalshi_lines.append((k_line, kalshi_prob, mkt))

                if kalshi_lines:
                    target_line_abs = abs(float(target_line))
                    nearest = min(kalshi_lines, key=lambda x: abs(float(x[0]) - target_line_abs))
                    delta = abs(float(nearest[0]) - target_line_abs)

                    tolerance = MAX_LINE_TOLERANCE.get(league, 1.5)
                    # if abs(kalshi_line - odds_line) <= 1.5:
                    if delta <= 1.5:
                        if delta == 0:
                            best_market = nearest[2]
                            match_status = "matched"
                            match_reason = "spread_match_exact"
                            out.at[idx, "kalshi_line_diff"] = 0.0
                        else:
                            best_market = nearest[2]
                            match_status = "matched"
                            match_reason = "spread_match_nearest"
                            out.at[idx, "kalshi_line_diff"] = delta

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

                if bid > 0 and ask > 0:
                    # Values are in dollars
                    kalshi_prob = (bid + ask) / 2.0
                else:
                    kalshi_prob = _select_probability(best_market)

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

    return out




def canonical_team_name(team: str) -> str:
    return str(team or "").strip()


def match_nba_spread(row: dict[str, Any], markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    pick_team = canonical_team_name(str(row.get("spread_pick_team") or ""))
    pick_line = pd.to_numeric(row.get("spread_pick_line"), errors="coerce")
    home_team = canonical_team_name(str(row.get("Home") or row.get("home_team") or ""))
    away_team = canonical_team_name(str(row.get("Away") or row.get("away_team") or ""))
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
