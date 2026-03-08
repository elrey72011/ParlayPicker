from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)
API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Total line parser (simple)
TOTAL_LINE_PATTERN = re.compile(
    r'(?:over|under|total points? over|total points? under|covers\s*\+?)\s*([0-9]+\.?[0-9]*)',
    re.IGNORECASE
)

def parse_kalshi_total_line(title: str) -> Tuple[Optional[str], Optional[float]]:
    """Extract ('over'/'under', line) from Kalshi total title."""
    m = TOTAL_LINE_PATTERN.search(title)
    if m:
        line = float(m.group(1))
        return "over" if "over" in title.lower() else "under", line
    return None, None

# Spread line parser
SPREAD_LINE_PATTERN = re.compile(
    r'(?:wins by over|by|by over|by more than|covers\s*\+?)\s*([0-9]+\.?[0-9]*)(?:\s*points?)?',
    re.IGNORECASE
)

def parse_kalshi_spread_line(title: str) -> Tuple[Optional[str], Optional[float]]:
    """Extract ('favorite', spread_line) from Kalshi spread title."""
    m = SPREAD_LINE_PATTERN.search(title)
    if m:
        return "favorite", float(m.group(1))
    return None, None

KALSHI_LINE_TOLERANCE = 2.5

def kalshi_line_matches_book(
    kalshi_title: str,
    book_side: str,      # 'over'/'under'
    book_line: float,
    market_type: str     # 'TOTAL' or 'SPREAD'
) -> bool:
    """Strict line match within tolerance."""
    if market_type.upper() == "TOTAL":
        side, k_line = parse_kalshi_total_line(kalshi_title)
        if k_line is None:
            logger.debug(f"Kalshi line parse failed for '{kalshi_title}', allowing match")
            return True

        diff = abs(k_line - book_line)
        if side and side.lower() == book_side.lower() and diff <= KALSHI_LINE_TOLERANCE:
            return True
        else:
            logger.info(f"REJECTED: title='{kalshi_title}' book_line={book_line} parsed={k_line} diff={diff}")
            return False

    elif market_type.upper() == "SPREAD":
        _, k_line = parse_kalshi_spread_line(kalshi_title)
        if k_line is None:
            logger.debug(f"Kalshi line parse failed for '{kalshi_title}', allowing match")
            return True

        book_spread = abs(book_line)
        diff = abs(k_line - book_spread)
        if diff <= KALSHI_LINE_TOLERANCE:
            return True
        else:
            logger.info(f"REJECTED: title='{kalshi_title}' book_line={book_line} parsed={k_line} diff={diff}")
            return False

    return False
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
    tried_tickers: list[str] | None = None


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


def _kalshi_search_fallback(league: str, home_team: str, away_team: str, game_date: str) -> dict[str, Any] | None:
    """Query Kalshi series for a date/team match when deterministic ticker guesses fail."""
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
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200:
                continue
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

def _get_markets(params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE}/markets", params=params, timeout=8)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise KalshiAPIError(str(exc)) from exc
    if not isinstance(payload, dict):
        return []
    return payload.get("markets", [])


def api_get_markets(**params: Any) -> dict[str, Any]:
    """Compatibility wrapper for Kalshi market lookups."""
    try:
        response = requests.get(f"{API_BASE}/markets", params=params, timeout=8)
        response.raise_for_status()
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
        return float((bid + ask) / 2.0)

    last = pd.to_numeric(market.get("last_price_dollars"), errors="coerce")
    if pd.notna(last):
        return float(last)

    for key in ("yes_bid_dollars", "yes_ask_dollars"):
        val = pd.to_numeric(market.get(key), errors="coerce")
        if pd.notna(val):
            return float(val)
    return None


def _deterministic_tickers(row: pd.Series) -> tuple[list[str], str | None, str | None, str | None, str, str | None]:
    league = str(row.get("league") or "").upper()
    family = _market_family(row)
    series = LEAGUE_SERIES_MAP.get(league, {}).get(family or "")
    away_code = _guess_code(str(row.get("away_team") or ""))
    home_code = _guess_code(str(row.get("home_team") or ""))
    date_code = build_kalshi_date_code(row.get("game_date"))

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
    """Fetch all open markets for each unique series in one call per series.
    Returns a dict of {ticker: market} for fast lookup.
    """
    cache: dict[str, dict[str, Any]] = {}
    for series in series_set:
        try:
            markets = _get_markets({"series_ticker": series, "status": "open", "limit": 200})
            for m in markets:
                ticker = str(m.get("ticker") or "")
                if not ticker:
                    continue
                if date_codes and not any(dc in ticker for dc in date_codes):
                    continue
                cache[ticker] = m
        except KalshiAPIError as exc:
            logger.warning("Kalshi series fetch failed for %s: %s", series, exc)
        except Exception as exc:
            logger.warning("Kalshi series fetch failed for %s: %s", series, exc)
    return cache


def _markets_by_team_text(series_markets: list[dict[str, Any]], home_team: str, away_team: str, date_code: str) -> list[dict[str, Any]]:
    """Fallback matcher when event_ticker team codes differ from local aliases."""
    home_token = _normalize_team_token(home_team)
    away_token = _normalize_team_token(away_team)
    candidates: list[dict[str, Any]] = []
    for m in series_markets or []:
        ticker = str(m.get("ticker") or "").upper()
        event_ticker = str(m.get("event_ticker") or "").upper()
        title = _normalize_team_token(str(m.get("title") or ""))
        subtitle = _normalize_team_token(str(m.get("subtitle") or ""))
        hay = " ".join([title, subtitle, event_ticker.lower()])
        if date_code and date_code not in ticker and date_code not in event_ticker:
            continue
        if home_token and away_token and home_token in hay and away_token in hay:
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
        "kalshi_tried_tickers",
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
        away_code = team_code_map(league, str(row.get("away_team") or ""))
        home_code = team_code_map(league, str(row.get("home_team") or ""))

        if not date_code:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "missing_date"
            continue
        if not away_code or not home_code:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "missing_team_code"
            continue
        if not series:
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "missing_series"
            continue

        base = f"{series}-{date_code}"
        candidates = [f"{base}{away_code}{home_code}", f"{base}{home_code}{away_code}"]

        markets: list[dict[str, Any]] = []
        try:
            direct_resp = api_get_markets(tickers=",".join(candidates))
            markets = _extract_markets(direct_resp)
        except Exception:
            markets = []

        if not markets:
            series_markets = []
            try:
                params = {"series_ticker": series, "status": "open", "limit": 100}
                for _ in range(5):
                    resp = api_get_markets(**params)
                    # Support both test mocks returning "data" or actual API returning "markets"
                    markets_page = _extract_markets(resp)
                    if not markets_page:
                        break
                    series_markets.extend(markets_page)
                    cursor = resp.get("cursor") if isinstance(resp, dict) else None
                    if not cursor:
                        break
                    params["cursor"] = cursor
            except Exception:
                pass # allow series_markets to retain collected items

            markets = [
                m for m in series_markets
                if away_code in str(m.get("event_ticker") or "") and home_code in str(m.get("event_ticker") or "")
            ]

            if not markets:
                # Provide a relaxed date match since the event ticker prefix format could be different.
                markets = _markets_by_team_text(
                    series_markets,
                    str(row.get("home_team") or ""),
                    str(row.get("away_team") or ""),
                    "", # Allow match by title even if date code differs. It's an open market.
                )

        best_market = None
        best_delta = float("inf")
        match_reason = "no_market_for_tickers"
        match_status = "miss"

        # We need to verify that the Kalshi line matches our book line
        target_spread = pd.to_numeric(row.get("spread_line"), errors="coerce")
        target_total = pd.to_numeric(row.get("total_line"), errors="coerce")
        pick = str(row.get("best_pick") or "").lower()

        if not markets:
            out.at[idx, "kalshi_tried_tickers"] = json.dumps(candidates)
            out.at[idx, "kalshi_match_status"] = match_status
            out.at[idx, "kalshi_match_reason"] = match_reason
            continue

        for mkt in markets:
            m_title = str(mkt.get("title") or "").lower()
            m_subtitle = str(mkt.get("subtitle") or "").lower()
            combined_text = f"{m_title} {m_subtitle}"

            # NEW: Strict line matching filter
            if pd.notna(row.get("total_line")) and family == "total":
                if not kalshi_line_matches_book(
                    kalshi_title=combined_text,  # Kalshi title + subtitle contains the line
                    book_side=row.get("total_pick_side", ""),  # 'over'/'under'
                    book_line=float(row["total_line"]),
                    market_type="TOTAL"
                ):
                    logger.info(f"Kalshi TOTAL rejected: line mismatch {combined_text}")
                    continue  # Skip this candidate

            if pd.notna(row.get("spread_line")) and family == "spread":
                if not kalshi_line_matches_book(
                    kalshi_title=combined_text,
                    book_side="",  # Spreads don't need side, just line tolerance
                    book_line=float(row["spread_line"]),
                    market_type="SPREAD"
                ):
                    logger.info(f"Kalshi SPREAD rejected: line mismatch {combined_text}")
                    continue

            # Extract number from Kalshi title/subtitle
            # Often formatted as "wins by over X.5", "total points over X.5", or "covers +X.5"
            line_match = re.search(r'(?:over|under|total points? over|total points? under|wins by over|by|by over|by more than|covers\s*\+?)\s*([0-9]+(?:\.[0-9]+)?)', combined_text)

            if not line_match:
                # Moneyline check: if the market family is spread (Kalshi groups moneyline under spread series or as its own)
                # and target is a moneyline (we represent it as NaN spread or explicit market type), verify.
                # Actually, ML in our system typically has market_type="moneyline".
                # But our current code maps everything to family="spread" if it's not "total".
                # So if there's no line but it says "to win" and we are looking for a moneyline bet
                is_ml_pick = str(row.get("market_type") or "").lower() == "moneyline"
                if is_ml_pick and ("moneyline" in combined_text or "to win" in combined_text or "wins" in combined_text):
                    pick_team_norm = _normalize_team_token(str(row.get("pick_team") or row.get("home_team") if pick.startswith(str(row.get("home_team") or "").lower()) else row.get("away_team")))
                    if pick_team_norm and pick_team_norm in _normalize_team_token(m_title):
                        best_market = mkt
                        match_status = "matched"
                        match_reason = "exact_match"
                continue

            k_line = float(line_match.group(1))

            # Match spread lines
            if family == "spread" and pd.notna(target_spread) and str(row.get("market_type") or "").lower() != "moneyline":
                # Spread targets are often stored as relative to the home/away team (+4.5, -3.5)
                # Kalshi stores favorites as "Team A wins by over 3.5" (-4.0)
                # Kalshi stores underdogs as "Team B covers +3.5" (+3.5)

                # Determine pick direction: favorite (<0) or underdog (>0)
                pick_is_favorite = target_spread < 0

                # Determine market direction
                market_is_favorite = "by over" in combined_text or "wins by" in combined_text
                market_is_underdog = "covers" in combined_text or "spread" in combined_text and "+" in combined_text

                # Default to matching absolute value if we can't definitively determine market direction
                # but if we can, ensure they match
                direction_match = True
                if market_is_favorite and not pick_is_favorite:
                    direction_match = False
                elif market_is_underdog and pick_is_favorite:
                    direction_match = False

                if direction_match:
                    delta = abs(k_line - abs(target_spread))
                    if delta <= KALSHI_LINE_TOLERANCE:
                        # Also need to match the specific side (Over vs Under, or specific team)
                        # We look for the pick team name in the title
                        pick_team_norm = _normalize_team_token(str(row.get("pick_team") or row.get("home_team") if pick.startswith(str(row.get("home_team") or "").lower()) else row.get("away_team")))
                        if pick_team_norm and pick_team_norm in _normalize_team_token(m_title):
                            if delta < best_delta:
                                best_delta = delta
                                best_market = mkt
                                match_status = "matched"
                                match_reason = "exact_match" if delta == 0 else "close_match"

            # Match total lines
            elif family == "total" and pd.notna(target_total):
                delta = abs(k_line - target_total)
                if delta <= KALSHI_LINE_TOLERANCE:
                    # Match specific side (over/under)
                    is_over_pick = "over" in pick
                    is_over_mkt = "over" in m_title or "over" in m_subtitle
                    if is_over_pick == is_over_mkt:
                        if delta < best_delta:
                            best_delta = delta
                            best_market = mkt
                            match_status = "matched"
                            match_reason = "exact_match" if delta == 0 else "close_match"

        if best_market is None:
            # We found markets, but none matched our line criteria
            out.at[idx, "kalshi_tried_tickers"] = json.dumps(candidates)
            out.at[idx, "kalshi_match_status"] = "miss"
            out.at[idx, "kalshi_match_reason"] = "alt_line_mismatch"
            out.at[idx, "kalshi_match_quality"] = "line_mismatched"
        else:
            bid = float(pd.to_numeric(best_market.get("yes_bid_dollars"), errors="coerce") or 0.0)
            ask = float(pd.to_numeric(best_market.get("yes_ask_dollars"), errors="coerce") or 0.0)

            if (bid + ask) > 2.0:
                # Handle legacy/mock cents data safely without breaking probability bounds
                out.at[idx, "kalshi_probability"] = (bid + ask) / 200.0
            else:
                out.at[idx, "kalshi_probability"] = (bid + ask) / 2.0
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
        if market_team != target_team:
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
