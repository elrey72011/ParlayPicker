from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pandas as pd
import requests

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

KALSHI_LINE_TOLERANCE_SPREAD = 0.5
KALSHI_LINE_TOLERANCE_TOTAL = 0.5

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
    "NCAAB": {"spread": "KXNCAAMBSPREAD", "total": "KXNCAAMBTOTAL"},
    "NBA": {"spread": "KXNBASPREAD", "total": "KXNBATOTAL"},
    "NHL": {"spread": "KXNHLSPREAD", "total": "KXNHLTOTAL"},
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


def _guess_code(team: str) -> str | None:
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
                if home_code and away_code and home_code in ticker and away_code in ticker:
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


def league_series_ticker(league: str, family: str) -> str:
    return str(LEAGUE_SERIES_MAP.get(str(league or "").upper(), {}).get(str(family or "").lower(), ""))


def team_code_map(league: str, team: str) -> str:
    _ = league
    token = _normalize_team_token(team)
    if token in TEAM_CODE_ALIASES:
        return TEAM_CODE_ALIASES[token]
    return str(_guess_code(team) or "")


def team_code_for_league(league: str, team: str) -> str:
    code = _det_team_code(league, team)
    return str(code or "")


def _make_kalshi_request(url: str, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, timeout: int = 8) -> requests.Response:
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


def _select_probability(market: dict[str, Any]) -> float | None:
    bid = pd.to_numeric(market.get("yes_bid_dollars"), errors="coerce")
    ask = pd.to_numeric(market.get("yes_ask_dollars"), errors="coerce")
    if pd.notna(bid) and pd.notna(ask):
        if (bid + ask) > 2.0:
            return float((bid + ask) / 200.0)
        else:
            return float((bid + ask) / 2.0)

    last = pd.to_numeric(market.get("last_price_dollars"), errors="coerce")
    if pd.notna(last):
        if last > 1.0:
            return float(last / 100.0)
        return float(last)

    for key in ("yes_bid_dollars", "yes_ask_dollars"):
        val = pd.to_numeric(market.get(key), errors="coerce")
        if pd.notna(val):
            if val > 1.0:
                return float(val / 100.0)
            return float(val)
    return None


    if not date_code:
        return [], series, away_code, home_code, date_code, family
    if not away_code or not home_code:
        return [], series, away_code, home_code, date_code, family
    if not series:
        return [], series, away_code, home_code, date_code, family

    prefix = f"{series}-{date_code}"
    candidates = [f"{prefix}{away_code}{home_code}", f"{prefix}{home_code}{away_code}"]
    return list(dict.fromkeys(candidates)), series, away_code, home_code, date_code, family


def _fetch_series_cache(series_set: set[str], date_codes: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Fetch all open markets for each unique series in one call per series with pagination."""
    cache: dict[str, dict[str, Any]] = {}
    for series in series_set:
        try:
            params = {"series_ticker": series, "status": "open", "limit": 100}
            for _ in range(10):
                resp = api_get_markets(**params)
                markets = _extract_markets(resp)
                if not markets:
                    break

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


def _markets_by_team_text(series_markets: list[dict[str, Any]], home_team: str, away_team: str, date_code: str) -> list[dict[str, Any]]:
    """Fallback matcher that uses partial substring and initials matching."""
    candidates = []
    home_norm = _normalize_team_token(home_team)
    away_norm = _normalize_team_token(away_team)

    h_3 = home_norm.replace(" ", "")[:3].upper()
    a_3 = away_norm.replace(" ", "")[:3].upper()
    h_initials = "".join([w[0] for w in home_norm.split() if w]).upper()
    a_initials = "".join([w[0] for w in away_norm.split() if w]).upper()

    for m in series_markets or []:
        ticker = str(m.get("ticker") or "").upper()
        ev_ticker = str(m.get("event_ticker") or "").upper()
        hay = (str(m.get("title", "")) + " " + str(m.get("subtitle", "")) + " " + ev_ticker).lower()

        # Date constraint intentionally removed to prevent timezone shift misses

        home_word_match = home_norm.split()[0] in hay if home_norm else False
        away_word_match = away_norm.split()[0] in hay if away_norm else False

        suffix = ev_ticker.split("-")[-1] if "-" in ev_ticker else ev_ticker
        home_abbr_match = h_3 in suffix or (h_initials and h_initials in suffix)
        away_abbr_match = a_3 in suffix or (a_initials and a_initials in suffix)

        if (home_word_match or home_abbr_match) and (away_word_match or away_abbr_match):
            candidates.append(m)

    return candidates


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

    # Cache the heavy series payloads so we only download them once per league
    series_cache = {}

    for idx, row in out.iterrows():
        league = str(row.get("league") or "").upper()
        family_guess = _market_family(row)
        family = "spread" if family_guess == "spread" else "total"
        series = league_series_ticker(league, family)

        if not series:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "missing_series"
            continue

        markets: list[dict[str, Any]] = []

        # 1. Fetch Series Cache to avoid API Rate Limits
        if series not in series_cache:
            series_markets = []
            try:
                params = {"series_ticker": series, "status": "open", "limit": 100}
                for _ in range(20): # Paginate up to 2000 markets per series
                    resp = api_get_markets(**params)
                    page = _extract_markets(resp)
                    if not page:
                        break
                    series_markets.extend(page)
                    cursor = resp.get("cursor") if isinstance(resp, dict) else None
                    if not cursor:
                        break
                    params["cursor"] = cursor
            except Exception as e:
                logger.warning(f"Cache fetch failed for {series}: {e}")

            series_cache[series] = series_markets

        series_markets = series_cache.get(series, [])

        # 2. Extract specific game markets from the cache
        markets = [
            m for m in series_markets
            if m.get("event_ticker") in candidates
        ]

        if not markets:
            markets = _markets_by_team_text(
                series_markets,
                str(row.get("home_team") or ""),
                str(row.get("away_team") or ""),
                date_code,
            )

        best_market = None
        best_delta = float("inf")
        match_reason = "no_market_for_tickers"
        match_status = "miss"

        if not markets:
            out.at[idx, "kalshi_match_status"] = match_status
            out.at[idx, "kalshi_match_reason"] = match_reason
            continue

        # In case we found no matching markets via dynamic discovery
        if not markets:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "no_market_for_tickers"
            continue

        markets = [m for m in markets if market_type_matches(row.get('market_type'), m.get('title'), m.get('subtitle'))]
        logger.info(f"FILTERED {len(markets)} markets for {row.get('game_id')}, type={row.get('market_type')}")

        if family == "spread":
            target_line_raw = pd.to_numeric(row.get("spread_line"), errors="coerce")
        else:
            target_line_raw = pd.to_numeric(row.get("total_line"), errors="coerce")

        if pd.isna(target_line_raw):
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
            target_line = abs(target_line_raw)

            for mkt in markets:
                m_title = str(mkt.get("title") or "").lower()
                m_subtitle = str(mkt.get("subtitle") or "").lower()
                combined_text = f"{m_title} {m_subtitle}"

                # Regex to find any standard decimal or integer line (e.g., "-12.5", "+6.5", "222.5", "-12", "+6", "4")
                numbers = re.findall(r'[-+]?\s*(\d+(?:\.\d+)?)', combined_text)

                for num_str in numbers:
                    try:
                        extracted_val = abs(float(num_str))

                        if family == "spread":
                            # Kalshi spread math translation: "wins by over 3.5" means -4.0 spread line.
                            k_line = extracted_val + 0.5
                            delta = abs(k_line - target_line)
                            if delta <= KALSHI_LINE_TOLERANCE_SPREAD:
                                m_type = str(row.get("market_type")).lower()
                                book_line = pd.to_numeric(row.get("spread_line"), errors="coerce")
                                is_favorite_bet = book_line < 0

                                home_t = str(row.get("home_team") or "")
                                away_t = str(row.get("away_team") or "")
                                combined_text = f"{str(mkt.get('title', ''))} {str(mkt.get('subtitle', ''))}".lower()

                                home_shared = {w for w in set(_normalize_team_token(home_t).split()).intersection(set(_normalize_team_token(combined_text).split())) if len(w) > 2}
                                away_shared = {w for w in set(_normalize_team_token(away_t).split()).intersection(set(_normalize_team_token(combined_text).split())) if len(w) > 2}

                                ticker_suffix = str(mkt.get("ticker", "")).split("-")[-1]
                                home_abbr = str(_guess_code(home_t) or "").upper()
                                away_abbr = str(_guess_code(away_t) or "").upper()

                                kalshi_subject_is_home = bool(home_shared) or (home_abbr and home_abbr in ticker_suffix)
                                kalshi_subject_is_away = bool(away_shared) or (away_abbr and away_abbr in ticker_suffix)

                                expected_subject_is_home = ("home" in m_type) if is_favorite_bet else not ("home" in m_type)

                                is_correct_match = (expected_subject_is_home and kalshi_subject_is_home) or \
                                                   (not expected_subject_is_home and kalshi_subject_is_away)

                                if not kalshi_subject_is_home and not kalshi_subject_is_away:
                                    is_correct_match = True # Assume match if Kalshi subject is completely ambiguous

                                if is_correct_match:
                                    if delta < best_delta:
                                        best_delta = delta
                                        best_market = mkt
                                        match_status = "matched"
                                        match_reason = "spread_match"

                        elif family == "total":
                            k_line = extracted_val
                            delta = abs(k_line - target_line)
                            if delta <= KALSHI_LINE_TOLERANCE_TOTAL:
                                if delta < best_delta:
                                    best_delta = delta
                                    best_market = mkt
                                    match_status = "matched"
                                    match_reason = "total_match"

                    except ValueError:
                        continue

                if best_market is not None:
                    break

        if best_market is None:
            # We found no markets or candidates at all
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "alt_line_mismatch"
            out.at[idx, "kalshi_match_quality"] = "line_mismatched"
        else:
            bid = float(pd.to_numeric(best_market.get("yes_bid_dollars"), errors="coerce") or 0.0)
            ask = float(pd.to_numeric(best_market.get("yes_ask_dollars"), errors="coerce") or 0.0)

            # REPLACE WITH ADAPTIVE LOGIC:
            if (bid + ask) > 2.0:
                # Values are in cents
                kalshi_prob = (bid + ask) / 200.0
            elif bid > 0 and ask > 0:
                # Values are in dollars
                kalshi_prob = (bid + ask) / 2.0
            else:
                kalshi_prob = _select_probability(best_market)

            if kalshi_prob is None:
                out.at[idx, "kalshi_match_status"] = "miss"
                out.at[idx, "kalshi_match_reason"] = "null_probability_extracted"
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
            if pd.isna(kalshi_prob) or kalshi_prob < 0.0 or kalshi_prob > 1.0:
                logger.warning(f"⚠️ Invalid Kalshi probability {kalshi_prob} for {row.get('game_id', 'unknown')}, skipping row")
                out.at[idx, "kalshi_match_status"] = "error"
                out.at[idx, "kalshi_match_reason"] = f"invalid_probability_{kalshi_prob}"
                continue

            out.at[idx, "kalshi_probability"] = kalshi_prob
            out.at[idx, "kalshi_market_title"] = best_market.get("title")
            out.at[idx, "kalshi_event_ticker"] = best_market.get("event_ticker")
            out.at[idx, "kalshi_market_ticker"] = best_market.get("ticker")
            out.at[idx, "kalshi_match_status"] = match_status
            out.at[idx, "kalshi_match_reason"] = match_reason
            out.at[idx, "kalshi_match_quality"] = "line_matched"

            # Optional: Add kalshi_line_diff if you have a way to track the final line matched.
            # Here we default to setting it properly later or it implies 0 / within tolerance.

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
