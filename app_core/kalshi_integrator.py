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

from core.team_mapper import normalize_team_name
from app_core.team_name_matcher import TeamNameMatcher

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


def _safe_is_missing(value: Any) -> bool:
    """True when value is None/NA-like, without raising on extension dtypes."""
    try:
        return value is None or pd.isna(value)
    except Exception:
        return value is None


def _safe_first_present(*values: Any) -> Any:
    """Return the first non-missing value using NA-safe checks."""
    for value in values:
        if not _safe_is_missing(value):
            return value
    return None


def _row_text(row: pd.Series, key: str, lowercase: bool = False) -> str:
    """Read row text safely with pandas NA-awareness for Pandas 2.x."""
    val = row.get(key)
    if pd.notna(val):
        text = str(val).strip()
        return text.lower() if lowercase else text
    return ""


def _normalized_merge_key(value: Any) -> str:
    if _safe_is_missing(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _normalize_identity_merge_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Normalize merge-key identity columns as stripped pandas StringDtype."""
    out = df.copy()
    for key in keys:
        if key not in out.columns:
            out[key] = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
        out[key] = out[key].astype("string").str.strip()
    return out


def clean_team_name(name: Any) -> str:
    """Normalize team naming noise for robust fuzzy matching."""
    text = _safe_text(name).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in TEAM_NOISE_WORDS]
    return " ".join(tokens)

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

KALSHI_LINE_TOLERANCE_SPREAD = 3.0
KALSHI_LINE_TOLERANCE_TOTAL = 3.0
KALSHI_SERIES_CACHE_TTL_SECONDS = 300.0

MAX_LINE_TOLERANCE = {
    "NBA": 5.5,
    "NCAAB": 10.5,
    "NHL": 5.5,
    "MLB": 8.5,
    "NFL": 2.5,
}

# Totals need a MUCH tighter line-match than the generic MAX_LINE_TOLERANCE above.
# A pick's total ("Over 7.5") must match a Kalshi contract at essentially the SAME
# line — matching it to a contract several runs/points away pulls in a probability
# for a different bet. MLB was the worst offender: the 8.5-run generic tolerance let
# an "Over 7.5" pick match almost ANY Kalshi total for the game (e.g. an "Over 5.5"
# contract at P(over)~0.70), systematically inflating Kalshi P(over) to 0.55-0.62 on
# every game and producing all-Over cards (17 Jun). These are sized to each league's
# total range and contract granularity. When no Kalshi contract is within tolerance,
# the matcher reports a miss (Kalshi is simply absent from the blend) rather than
# proxying a wrong-line probability.
MAX_TOTAL_LINE_TOLERANCE = {
    "NBA": 4.5,
    "NCAAB": 6.5,
    "NHL": 1.0,
    "MLB": 1.0,
    "NFL": 3.5,
}

def _normalize_league_for_kalshi(league: str) -> str:
    l_up = str(league or "").upper()
    if l_up in ["NCAAM", "NCAAMB", "NCAA MENS BASKETBALL"]:
        return "NCAAB"
    return l_up

_NCAAB_FALLBACK_SERIES = ["KXMARMADROUND"]
_NCAAB_SERIES_TITLE_HINTS = ("march madness", "ncaa", "kxmarmadround")

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

    # New: allow side bets to match both pure winners and spread fallbacks
    if "moneyline" in market_type or "h2h" in market_type:
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
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Oklahoma City": "OKC",
    "Orlando Magic": "ORL", "Orlando": "ORL",
    "Philadelphia 76ers": "PHI", "Philadelphia": "PHI",
    "Phoenix Suns": "PHX", "Phoenix": "PHX",
    "Portland Trail Blazers": "POR", "Portland": "POR",
    "Sacramento Kings": "SAC", "Sacramento": "SAC",
    "San Antonio Spurs": "SAS", "San Antonio": "SAS",
    "Toronto Raptors": "TOR",
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
    "New York Islanders": "NYI", "NY Islanders": "NYI", "Ny Islanders": "NYI", "Islanders": "NYI",
    "New York Rangers": "NYR", "NY Rangers": "NYR", "Ny Rangers": "NYR", "Rangers": "NYR",
    "Ottawa Senators": "OTT", "Ottawa": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "Pittsburgh": "PIT",
    "San Jose Sharks": "SJS", "San Jose": "SJS",
    "Seattle Kraken": "SEA", "Seattle": "SEA",
    "St. Louis Blues": "STL", "St. Louis": "STL", "St Louis": "STL",
    "Tampa Bay Lightning": "TBL", "Tampa Bay": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN", "Vancouver": "VAN",
    "Vegas Golden Knights": "VGK", "Vegas": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG", "Winnipeg": "WPG",
    # MLB — codes as they appear in Kalshi event tickers (e.g.
    # KXMLBTOTAL-26JUL042008STLCHC). Only 3+ char codes: the scorer ignores
    # shorter codes for substring checks (see _event_match_score), because a
    # guessed 2-letter code matching inside another game's ticker is exactly
    # how Cubs/StL kept event-matching to "Boston vs Los Angeles A" (4 Jul:
    # guessed code "SL" ⊂ ticker "...BOSLAA"). City-only keys are added only
    # where they don't collide with an existing NBA/NHL mapping.
    "New York Yankees": "NYY", "NY Yankees": "NYY",
    "New York Mets": "NYM", "NY Mets": "NYM",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Saint Louis": "STL",  # upload naming; "St. Louis"/"St Louis" already map above
    "Los Angeles Dodgers": "LAD", "L.A. Dodgers": "LAD",
    "Los Angeles Angels": "LAA", "L.A. Angels": "LAA",
    "Athletics": "ATH", "Oakland Athletics": "ATH",
    "Cincinnati": "CIN",
    "Baltimore": "BAL",
    "Texas": "TEX",
    "Toronto": "TOR",
    "Colorado": "COL",
    "Cleveland Guardians": "CLE",
    "Milwaukee Brewers": "MIL",
    "Houston Astros": "HOU",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFG",
    "Pittsburgh Pirates": "PIT",
    "Philadelphia Phillies": "PHI",
    "Detroit Tigers": "DET",
    "Minnesota Twins": "MIN",
    "Atlanta Braves": "ATL",
    "Miami Marlins": "MIA",
    "Boston Red Sox": "BOS",
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
    # First apply robust centralized normalization mapping from core pipeline
    s = normalize_team_name(name)

    s = _safe_text(s).lower().strip()
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
    market_type = _row_text(row, "market_type", lowercase=True)
    if market_type.startswith("spread"):
        return "spread"
    if market_type.startswith("total"):
        return "total"
    best_pick = _row_text(row, "best_pick", lowercase=True)
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
    league_text = str(league).upper() if pd.notna(league) else ""
    series_list = series_map.get(league_text, [])

    home_code = _lookup_kalshi_team_code(home_team) or ""
    away_code = _lookup_kalshi_team_code(away_team) or ""

    headers = {"Authorization": f"Bearer {kalshi_api_key}"}
    for series in series_list:
        try:
            url = f"{API_BASE}/markets?series_ticker={series}&limit=100"
            resp = _make_kalshi_request(url, headers=headers, timeout=5)
            markets = resp.json().get("markets", [])
            for market in markets:
                ticker_val = market.get("ticker")
                ticker = str(ticker_val) if pd.notna(ticker_val) else ""
                if date_str not in ticker:
                    continue
                # Bidirectional permutation check
                has_home = home_code and home_code in ticker
                has_away = away_code and away_code in ticker
                if has_home and has_away:
                    return market
                title_val = market.get("title")
                title = str(title_val).lower() if pd.notna(title_val) else ""
                home_text = str(home_team).lower() if pd.notna(home_team) else ""
                away_text = str(away_team).lower() if pd.notna(away_team) else ""
                if home_text in title or away_text in title:
                    return market
        except Exception:
            continue
    return None




def _det_team_code(league: str, team: str) -> str | None:
    """Deterministic team-code resolver used by tests and ingestion code."""
    _ = league
    return _guess_code(team)


def league_series_ticker(league: str, market_type: str) -> str:
    league_upper = str(league).upper() if pd.notna(league) else ""
    market_lower = str(market_type).lower() if pd.notna(market_type) else ""

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

    series_val = LEAGUE_SERIES_MAP.get(league_upper, {}).get(family, "")
    return str(series_val) if pd.notna(series_val) else ""


def team_code_map(league: str, team: str) -> str:
    _ = league
    token = _normalize_team_token(team)
    if token in TEAM_CODE_ALIASES:
        return TEAM_CODE_ALIASES[token]
    return str(_guess_code(team) or "")


def team_code_for_league(league: str, team: str) -> str:
    code = _det_team_code(league, team)
    return str(code) if pd.notna(code) else ""


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
    from core.team_mapper import normalize_team_name
    # Normalize bookmaker strings
    norm_home = normalize_team_name(home_team)
    norm_away = normalize_team_name(away_team)

    # Normalize Kalshi strings
    title = normalize_team_name(str(event.get("title") or ""))
    subtitle = normalize_team_name(str(event.get("sub_title") or ""))
    ticker = normalize_team_name(str(event.get("event_ticker") or ""))

    combined_norm = " ".join([
        _normalize_team_token(title),
        _normalize_team_token(subtitle),
        _normalize_team_token(ticker),
    ]).strip()
    combined_upper = f"{title} {subtitle} {ticker}".upper()
    combined_key = _normalized_merge_key(f"{title} {subtitle} {ticker}")

    home_norm = _normalize_team_token(norm_home)
    away_norm = _normalize_team_token(norm_away)
    home_clean = clean_team_name(norm_home)
    away_clean = clean_team_name(norm_away)
    home_key = _normalized_merge_key(norm_home)
    away_key = _normalized_merge_key(norm_away)
    home_tokens = _team_tokens_for_match(norm_home)
    away_tokens = _team_tokens_for_match(norm_away)

    home_code = team_code_for_league(league, norm_home).upper()
    away_code = team_code_for_league(league, norm_away).upper()

    # STRICT ALIAS OVERRIDE FOR NCAAB (Moved to top to bypass penalties)
    if league.upper() in ['NCAAB', 'NCAAM']:
        alias_map = {
            "queens nc": "queens university",
            "connecticut": "uconn",
            "wright st": "wright state",
            "liu": "long island university",
            "central florida": "ucf"
        }
        for k_alias, standard in alias_map.items():
            # 1. Is this specific row trying to match one of our aliased teams?
            if standard in home_norm or standard in away_norm:
                # 2. Is the alias (or its standard name) present in the Kalshi string?
                if k_alias in combined_norm or standard in combined_norm:
                    return 100 # Force perfect score immediately


    score = 0

    # Strong signal: explicit Kalshi codes in ticker/title payload. Codes shorter
    # than 3 chars are ignored: _guess_code invents initials for unmapped teams,
    # and a 2-letter guess substring-matching inside another game's ticker is a
    # false strong signal (4 Jul: "Saint Louis" guessed "SL" ⊂ "...BOSLAA", so
    # Cubs/StL event-matched to Boston vs LA Angels every run — the wrong-game
    # title guard then killed the price, leaving a permanent Kalshi miss).
    if home_code and len(home_code) >= 3 and home_code in combined_upper:
        score += 60
    if away_code and len(away_code) >= 3 and away_code in combined_upper:
        score += 60

    # Strong signal: canonical normalized names
    if home_norm and home_norm in combined_norm:
        score += 45
    if away_norm and away_norm in combined_norm:
        score += 45

    # NHL resilience: "Ny" vs "New York", truncated city names
    if league.upper() == 'NHL':
        if "ny " in home_norm and "new york " in combined_norm:
            score += 35
        if "ny " in away_norm and "new york " in combined_norm:
            score += 35
        # Leniency for short names like "Toronto" vs "Toronto Maple Leafs"
        if len(home_clean) >= 5 and home_clean in clean_team_name(f"{title} {subtitle} {ticker}"):
            score += 20
        if len(away_clean) >= 5 and away_clean in clean_team_name(f"{title} {subtitle} {ticker}"):
            score += 20

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

    # Penalize / reject one-sided matches to avoid false positives. A team counts as
    # STRONGLY present only via its CODE or FULL normalized name — NOT a bare token
    # overlap or the cleaned name, because both collide on the shared CITY across teams
    # ("Los Angeles" Dodgers vs Angels; clean_team_name strips the mascot to the city).
    home_strong = bool(
        (home_code and len(home_code) >= 3 and home_code in combined_upper)
        or (home_norm and home_norm in combined_norm)
    )
    away_strong = bool(
        (away_code and len(away_code) >= 3 and away_code in combined_upper)
        or (away_norm and away_norm in combined_norm)
    )
    has_home = home_strong or bool(home_tokens.intersection(combined_tokens))
    has_away = away_strong or bool(away_tokens.intersection(combined_tokens))

    if not has_home and not has_away:
        # No evidence of EITHER team. If the title positively names a DIFFERENT
        # matchup ("Boston vs Los Angeles A: Total Runs" for a Cubs/StL row),
        # reject outright — the +25 date affinity alone reaches the acceptance
        # threshold (25), so any same-day event could otherwise match a game
        # whose true event was missing from the fetch. A teamless generic title
        # ("NBA Total") stays neutral: it can't confirm or deny the game, and
        # the single-candidate fallback is allowed to keep it.
        if not kalshi_title_references_matchup(f"{title} {subtitle}", norm_home, norm_away):
            return -100

    if has_home ^ has_away:
        # Exactly one team present. If that lone team matched only by a weak token
        # overlap (e.g. the shared city "los angeles" while the real opponent is
        # absent — 17 Jun: Dodgers/Tampa Bay matched "Los Angeles A vs Arizona"),
        # it's the WRONG game: reject it well below the floor so we take a Kalshi miss
        # instead of another game's price. A strong (code/name) one-sided match is
        # only penalized, not rejected — the opponent may just be coded differently.
        present_is_strong = home_strong if has_home else away_strong
        if not present_is_strong:
            return -100
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
    """Helper to make rate-limited Kalshi API requests with exponential backoff for 429 and Connection errors."""
    time.sleep(0.2)
    max_retries = 3
    backoff = 2.0

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                if attempt < max_retries:
                    logger.warning(f"Kalshi API 429 Too Many Requests. Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                else:
                    logger.error(f"Kalshi API 429 Too Many Requests. Max retries ({max_retries}) exceeded.")
                    resp.raise_for_status()
            else:
                resp.raise_for_status()
                return resp
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(f"Kalshi API Network Error ({e}). Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2.0
                continue
            else:
                logger.error(f"Kalshi API Request Failed after {max_retries} retries: {e}")
                raise

    # Fallback return, should not reach here if raise happens above
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
    Fallback: ticker string.
    Universal totals math applied at the very end (+0.5 for integer totals).
    """
    m_title = str(mkt.get("title") or "").lower()
    m_subtitle = str(mkt.get("subtitle") or "").lower()
    combined_text = f"{m_title} {m_subtitle}"

    val = None

    # 0. Authoritative source for TOTALS: the structured Kalshi strike. A KXMLBTOTAL
    # market resolves YES if the total is OVER floor_strike, so floor_strike IS the
    # over line directly (e.g. floor_strike 6.5 -> "over 6.5", YES = P(over 6.5)).
    # The text-parse + integer "+0.5" below was returning floor_strike + 1.0 (parsing
    # the "7" in a "7 runs"/ticker label as 7 -> 7.5 for a 6.5-strike contract), so
    # P(over 6.5) was mislabeled as the over-7.5 probability and every MLB over was
    # inflated ~+0.09 (17 Jun all-Over cards). Confirmed from exported floor_strike.
    if is_total:
        for _strike in (mkt.get("floor_strike"), mkt.get("cap_strike")):
            if _strike is not None:
                try:
                    _s = float(_strike)
                except (TypeError, ValueError):
                    continue
                if _s > 0:
                    return _s

    # 1. Primary: Strict Regex Match based on betting terminology
    match = re.search(r'(?:Over|Under|by over|by at least|more than)\s*(\d+(?:\.\d+)?)', combined_text, re.IGNORECASE)
    if match:
        val = abs(float(match.group(1)))
        if not is_total and "wins by" in combined_text.lower():
            if "more than" not in combined_text.lower() and "over" not in combined_text.lower() and "at least" not in combined_text.lower():
                val += 0.5

    # 2. Secondary Fallback: Original extraction purely from subtitle first, then combined text
    if val is None and m_subtitle:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", m_subtitle)
        for num_str in numbers:
            try:
                val = abs(float(num_str))
                if not is_total and "wins by" in combined_text.lower():
                    if "more than" not in combined_text.lower() and "over" not in combined_text.lower() and "at least" not in combined_text.lower():
                        val += 0.5
                break
            except ValueError:
                continue

    # Fallback to combined text
    if val is None:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", combined_text)
        for num_str in numbers:
            try:
                val = abs(float(num_str))
                if not is_total and "wins by" in combined_text.lower():
                    if "more than" not in combined_text.lower() and "over" not in combined_text.lower() and "at least" not in combined_text.lower():
                        val += 0.5
                break
            except ValueError:
                continue

    # 3. Tertiary: Fallback to ticker string
    if val is None:
        ticker = str(mkt.get("ticker") or "").strip()
        if ticker:
            parts = ticker.split("-")
            if parts:
                last_part = parts[-1]
                t_match = re.search(r'(\d+(?:\.\d+)?)', last_part)
                if t_match:
                    try:
                        val = float(t_match.group(1))
                    except ValueError:
                        pass

    if val is not None and is_total and val == int(val):
        return val + 0.5

    return val

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
        if bid_ask_spread > 40.0:
            logger.warning(f"Kalshi market illiquid: bid {bid}, ask {ask}. Spread > 40 cents.")
            return None
    else:
        # Values are in dollars
        if bid_ask_spread > 0.40:
            logger.warning(f"Kalshi market illiquid: bid {bid}, ask {ask}. Spread > 40 cents.")
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


def _fetch_ncaab_fallback_series_events() -> list[dict[str, Any]]:
    """Fetch NCAAB fallback events for tournament-style series (March Madness/NCAA)."""
    events: list[dict[str, Any]] = []
    for series in _NCAAB_FALLBACK_SERIES:
        events.extend(_fetch_series_events(series))
    deduped: dict[str, dict[str, Any]] = {}
    for event in events:
        ticker = str(event.get("event_ticker") or "").strip()
        if ticker:
            deduped[ticker] = event
    return list(deduped.values())


def _filter_ncaab_series_candidates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only events that look like March Madness/NCAA tournament series."""
    filtered: list[dict[str, Any]] = []
    for event in events:
        text = " ".join(
            [
                _safe_text(event.get("title")),
                _safe_text(event.get("sub_title")),
                _safe_text(event.get("subtitle")),
                _safe_text(event.get("event_ticker")),
                _safe_text(event.get("series_ticker")),
            ]
        ).lower()
        if any(hint in text for hint in _NCAAB_SERIES_TITLE_HINTS):
            filtered.append(event)
    return filtered


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





def _is_within_72h(item: dict[str, Any], game_date_obj: pd.Timestamp) -> bool:
    if pd.isna(game_date_obj):
        return True

    close_time_str = _safe_first_present(
        item.get("close_time"),
        item.get("expiration_time"),
        item.get("last_updated_ts"),
    )
    if not close_time_str:
        return True

    try:
        kalshi_dt = pd.to_datetime(close_time_str, utc=True)
        # Ensure game_date_obj is strictly timezone aware (UTC)
        if getattr(game_date_obj, 'tz', None) is None:
            game_date_obj = game_date_obj.tz_localize('UTC')
        else:
            game_date_obj = game_date_obj.tz_convert('UTC')

        return abs(kalshi_dt - game_date_obj) <= pd.Timedelta(hours=72)
    except Exception as e:
        logger.warning(f"Timezone matching error: {e}")
        return True


_NCAAB_LEAGUE_HINTS = {
    "state", "st", "university", "college", "tech", "a&m", "saint", "unc", "uc",
}


def _looks_like_college_team_name(team: Any) -> bool:
    text = _safe_text(team).strip().lower()
    if not text:
        return False
    tokens = set(re.findall(r"[a-z0-9&']+", text))
    if tokens.intersection(_NCAAB_LEAGUE_HINTS):
        return True
    # Common CBB style abbreviations (e.g. OK St, NC State, etc.)
    return bool(re.search(r"\b(st|state|u)\b", text))


def _infer_row_league(league: Any, home_team: Any, away_team: Any) -> str:
    league_text = _safe_text(league).strip().upper()
    if league_text:
        return league_text
    if _looks_like_college_team_name(home_team) or _looks_like_college_team_name(away_team):
        return "NCAAB"
    return ""


def kalshi_total_spread_price_extreme(prob, market_type) -> bool:
    """True when a TOTAL / run-line Kalshi price is implausibly extreme.

    Books set totals and run lines near pick'em, so a de-vigged Kalshi price at near-
    certainty (outside [KALSHI_TOTAL_SPREAD_MIN_PROB, MAX]) is a settled/illiquid/mis-
    scraped market, not a confident read (20 Jun: "over 12.5 runs" at 0.995). Moneyline
    is exempt — heavy favorites/dogs legitimately price near the edges.
    """
    if prob is None or pd.isna(prob):
        return False
    mt = str(market_type).lower()
    if "moneyline" in mt or "h2h" in mt:
        return False
    if not ("total" in mt or "spread" in mt):
        return False
    from app_core.weights_config import KALSHI_TOTAL_SPREAD_MIN_PROB, KALSHI_TOTAL_SPREAD_MAX_PROB
    return not (KALSHI_TOTAL_SPREAD_MIN_PROB <= float(prob) <= KALSHI_TOTAL_SPREAD_MAX_PROB)


def find_team_reference(norm_title: str, team_name: str) -> int:
    """Index of a team reference in a normalized market title, or -1.

    Kalshi truncates long team names in market titles ("New York Y wins by over
    1.5 runs?", "Chicago C", "Los Angeles D"), so an exact-containment
    ``title.find(" new york yankees ")`` misses the subject entirely — that made
    every Yankees run-line unpriceable (4 Jul: kalshi_match_reason
    spread_side_unpriceable on a liquid market). Besides the exact name, this
    accepts any leading prefix of the team name (>= 4 chars, longest first) that
    appears as a space-delimited fragment, mirroring the wrong-game guard's
    truncation rule.
    """
    title = f" {str(norm_title or '').strip().lower()} "
    team = str(team_name or "").strip().lower()
    if not team or title.strip() == "":
        return -1
    idx = title.find(f" {team} ")
    if idx >= 0:
        return idx
    for length in range(len(team) - 1, 3, -1):
        frag = team[:length].strip()
        if len(frag) < 4:
            break
        idx = title.find(f" {frag} ")
        if idx >= 0:
            return idx
    return -1


def kalshi_title_references_matchup(norm_title: str, norm_home: str, norm_away: str) -> bool:
    """True when a Kalshi market title plausibly belongs to the row's game.

    Wrong-game guard (3 Jul): the event matcher paired a Cubs/Cardinals Under 10.5
    with a market titled "Boston vs Los Angeles A Total Runs?" — a 10.5-strike
    contract from a DIFFERENT game's ladder. Its bogus 73% P(Under) flipped
    consensus to Agrees, inflated the blend to a fake +29% EV, and the empty-card
    recovery staked it $40 at S-Tier. Spreads catch this implicitly (the subject-
    orientation step misses and returns unpriceable); totals had no title check.

    Inputs are already normalized (lowercased, space-padded title; normalized team
    names). Kalshi truncates long names in titles ("Los Angeles D", "New York Y",
    "Chicago WS"), so besides direct containment we accept a "vs"-fragment that is
    a prefix of the team name (>= 4 chars, so a bare "a" can't match everything).
    ANY-match: one referenced team is enough — this guard exists to catch titles
    naming two entirely different teams, not to litigate abbreviations. Returns
    True (trust) when the title is empty/unparseable: the guard only fires on
    positive evidence of a wrong game.
    """
    # Case-insensitive throughout: the production normalizer (core.team_mapper.
    # normalize_team_name) TITLE-CASES its output ("Boston Vs Los Angeles A Total
    # Runs"), which silently defeated the first version of this guard — its
    # lowercase " vs " split never matched "Vs", the title read as non-matchup
    # format, and the wrong-game row was trusted (3 Jul 15:03Z run, on the new
    # build stamp). Never assume the normalizer's casing.
    title = f" {str(norm_title or '').strip().lower()} "
    if title.strip() == "":
        return True
    teams = [str(norm_home or "").strip().lower(), str(norm_away or "").strip().lower()]
    teams = [t for t in teams if t]
    if not teams:
        return True
    for team in teams:
        if f" {team} " in title:
            return True
    # Fragment pass: split the title body on the "vs" separator and strip the
    # market-name tail; a fragment that prefixes a team name is a match.
    body = title.replace("?", " ")
    for tail in (" total runs ", " total points ", " total goals ", " total "):
        cut = body.find(tail)
        if cut >= 0:
            body = body[:cut] + " "
            break
    parts = re.split(r"\s+vs\.?\s+|\s+@\s+", body, flags=re.IGNORECASE)
    if len(parts) < 2:
        # Not a "TeamA vs TeamB" matchup-format title (e.g. "Over 7.5 runs
        # scored") — it cannot confirm OR deny the game. Trust it and leave
        # identity to the other guards; only titles that positively name a
        # different matchup are rejected.
        return True
    for frag in parts:
        frag = frag.strip()
        if len(frag) < 4:
            continue
        for team in teams:
            if team.startswith(frag):
                return True
    return False


def orient_spread_kalshi_prob(raw_prob, subject_is_home, pick_is_home, pick_line):
    """P(pick covers) from a run-line contract "S wins by over L" (raw_prob = P(S by >L)).

    Such a contract prices exactly two outcomes: S -L (YES = raw_prob) and Opp(S) +L
    (NO = 1 - raw_prob). The 1-run margin band sits between the two -L sides, so the
    other two picks — an away/opponent favorite (Opp(S) -L) or the subject as underdog
    (S +L) — are NOT 1 - raw_prob and cannot be derived from this contract. Return None
    for those so the caller records a miss instead of a wrong number (19 Jun: Pittsburgh
    -1.5 was priced 1 - P(Colorado by 2+) = 0.705 = P(PIT +1.5), a 70% phantom edge).

    ``subject_is_home`` / ``pick_is_home`` identify the title's subject team and the
    pick's team; ``pick_line`` is the pick's signed spread (negative = favorite).
    """
    if pd.isna(pick_line):
        return None
    pick_is_subject = bool(subject_is_home) == bool(pick_is_home)
    pick_is_favorite = float(pick_line) < 0.0
    if pick_is_subject and pick_is_favorite:
        return float(raw_prob)            # pick == S -L (the YES side)
    if (not pick_is_subject) and (not pick_is_favorite):
        return 1.0 - float(raw_prob)      # pick == Opp(S) +L (the NO side)
    return None                           # unpriceable from this contract


def enrich_with_kalshi_markets(best_picks_df: pd.DataFrame) -> pd.DataFrame:
    if best_picks_df is None or best_picks_df.empty:
        return best_picks_df.copy() if isinstance(best_picks_df, pd.DataFrame) else pd.DataFrame()

    out = best_picks_df.copy()
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce", utc=True)
    out = _normalize_identity_merge_keys(out, ["league", "home_team", "away_team"])

    event_data_cache = {}

    for col in [
        "kalshi_probability",
        "kalshi_market_title",
        "kalshi_event_ticker",
        "kalshi_market_ticker",
        "kalshi_line_diff",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    out["kalshi_match_status"] = "miss"
    out["kalshi_match_reason"] = "no_market_for_tickers"

    for idx, row in out.iterrows():
        home_team_val = row.get("home_team")
        away_team_val = row.get("away_team")

        league_val = row.get("league")
        league = _infer_row_league(league_val, home_team_val, away_team_val)
        league = _normalize_league_for_kalshi(league)
        clean_league = str(league).upper().strip()
        market_type = _row_text(row, "market_type", lowercase=True)
        strike_price_text = _row_text(row, "strike_price")
        category = _row_text(row, "category", lowercase=True)

        family_guess = _market_family(row)
        if not family_guess and "spread" in category:
            family_guess = "spread"
        elif not family_guess and ("total" in category or "over" in category or "under" in category):
            family_guess = "total"
        elif not family_guess and strike_price_text:
            family_guess = "spread" if "spread" in market_type else "total"

        if any(x in market_type for x in ["moneyline", "h2h"]):
             family = "moneyline"
        else:
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

        # Kalshi's baseball page groups totals/spreads under parent KXMLBGAME
        # events. Keep the dedicated family series as a fallback, then merge and
        # deduplicate both event sources before applying strict game/line checks.
        series_candidates = [series]
        if clean_league == "MLB" and family in {"total", "spread"}:
            parent_series = LEAGUE_SERIES_MAP["MLB"]["moneyline"]
            if parent_series not in series_candidates:
                series_candidates.append(parent_series)

        # Cache by series and slate date, with a short TTL. Streamlit sessions
        # persist across reruns (and sometimes across dates).
        if not hasattr(enrich_with_kalshi_markets, "series_cache"):
            enrich_with_kalshi_markets.series_cache = {}
        if not hasattr(enrich_with_kalshi_markets, "series_cache_meta"):
            enrich_with_kalshi_markets.series_cache_meta = {}

        now = time.monotonic()
        combined_series_events: list[dict[str, Any]] = []
        for event_series in series_candidates:
            cache_meta = enrich_with_kalshi_markets.series_cache_meta.get(event_series, {})
            cache_stale = (
                event_series not in enrich_with_kalshi_markets.series_cache
                or cache_meta.get("date_code") != date_code
                or now - float(cache_meta.get("fetched_at", 0.0)) >= KALSHI_SERIES_CACHE_TTL_SECONDS
            )
            if cache_stale:
                fetched = _fetch_series_events(event_series)
                from core.team_mapper import normalize_team_name
                for e in fetched:
                    if str(league).upper() in ['NCAAB', 'NCAAM']:
                        alias_map = {
                            "queens nc": "queens university",
                            "connecticut": "uconn",
                            "wright st": "wright state",
                            "liu": "long island university",
                            "ucf": "central florida"
                        }
                        for key in ['title', 'sub_title', 'subtitle']:
                            val = e.get(key)
                            if val:
                                score_string = f" {normalize_team_name(val)} "
                                for k_alias, standard in alias_map.items():
                                    score_string = score_string.replace(f" {k_alias} ", f" {standard} ")
                                e[key] = score_string.strip()
                enrich_with_kalshi_markets.series_cache[event_series] = fetched
                enrich_with_kalshi_markets.series_cache_meta[event_series] = {
                    "date_code": date_code,
                    "fetched_at": now,
                    "event_count": len(fetched),
                }
            combined_series_events.extend(
                enrich_with_kalshi_markets.series_cache.get(event_series, [])
            )

        deduped_series_events: dict[str, dict[str, Any]] = {}
        for event in combined_series_events:
            ticker = str(event.get("event_ticker") or "").strip()
            if ticker:
                deduped_series_events[ticker] = event
        series_events = list(deduped_series_events.values())

        if league == "NCAAB":
            fallback_key = "__NCAAB_FALLBACK_EVENTS__"
            if fallback_key not in enrich_with_kalshi_markets.series_cache:
                enrich_with_kalshi_markets.series_cache[fallback_key] = _fetch_ncaab_fallback_series_events()
            fallback_events = _filter_ncaab_series_candidates(
                enrich_with_kalshi_markets.series_cache.get(fallback_key, [])
            )
            if fallback_events:
                combined_events = series_events + fallback_events
                deduped: dict[str, dict[str, Any]] = {}
                for event in combined_events:
                    ticker_val = event.get("event_ticker")
                    ticker = str(ticker_val).strip() if pd.notna(ticker_val) else ""
                    if ticker:
                        deduped[ticker] = event
                series_events = list(deduped.values())

        home_team_name = str(home_team_val).strip() if pd.notna(home_team_val) else ""
        away_team_name = str(away_team_val).strip() if pd.notna(away_team_val) else ""
        if league == "NCAAB":
            home_team_name = _clean_ncaab_team_for_match(home_team_name)
            away_team_name = _clean_ncaab_team_for_match(away_team_name)

        best_event_match = None
        best_event_score = -1

        for event in series_events:
            if not _is_within_72h(event, game_date):
                continue



            score = _event_match_score(event, home_team_name, away_team_name, league, date_code=date_code)
            if score > best_event_score:
                best_event_score = score
                best_event_match = event

        # Conservative acceptance threshold; tuned to reduce false misses for abbreviated Kalshi events.
        if best_event_score < 25:
            # last-chance fallback: accept strongest candidate if it clearly references both teams via code/name tokens
            if best_event_score < 10:
                best_event_match = None

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
                    } for e in series_events if _is_within_72h(e, game_date)
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
        if event_ticker in event_data_cache:
            nested_markets = event_data_cache[event_ticker]
        else:
            try:
                url = f"{API_BASE}/events/{event_ticker}"
                # Extend timeout for large/nested event lookups
                resp = _make_kalshi_request(url, params={"with_nested_markets": "true"}, timeout=30)
                payload = resp.json()
                if payload and "event" in payload:
                    nested_markets = payload["event"].get("markets", [])
                    # Pre-normalize nested market titles for efficiency
                    for mkt in nested_markets:
                        m_title = str(mkt.get("title") or "").lower()
                        mkt["market_title_std"] = TeamNameMatcher.normalize(m_title)
                    event_data_cache[event_ticker] = nested_markets
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
            row_market_type = market_type
            totals_markets = [
                m for m in nested_markets
                if market_type_matches(row_market_type or "total", m.get("title"), m.get("subtitle"))
                or _infer_market_family_from_text(m.get("title"), m.get("subtitle"), row_market_type) == "total"
            ]

            kalshi_lines = []
            for mkt in totals_markets:
                ticker_str = str(mkt.get("ticker", ""))
                ticker_parts = ticker_str.split("-")

                if ticker_parts:
                    ticker_prefix = (ticker_parts[0].replace('KX', '')
                                     .replace('MBSPREAD', 'B')
                                     .replace('MBTOTAL', 'B')
                                     .replace('BGAME', 'B'))

                    if ticker_prefix == "NCAAMB":
                        ticker_prefix = "NCAAB"

                    if not ticker_prefix.startswith(clean_league):
                        continue

                k_line = _extract_kalshi_line(mkt, is_total=True)
                if k_line is not None:
                    bid = _safe_float(mkt.get("yes_bid_dollars"))
                    ask = _safe_float(mkt.get("yes_ask_dollars"))

                    if bid == 0.0 or ask == 0.0:
                        kalshi_prob = None
                    elif (bid + ask) > 2.0:
                        if abs(bid - ask) > 40.0:
                            kalshi_prob = None
                        else:
                            kalshi_prob = (bid + ask) / 200.0
                    elif bid > 0 and ask > 0:
                        if abs(bid - ask) > 0.40:
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
                # Totals use the tight per-league totals tolerance, not the loose
                # generic one, so a pick only matches a Kalshi contract at (near) the
                # SAME line. See MAX_TOTAL_LINE_TOLERANCE.
                league_tolerance = MAX_TOTAL_LINE_TOLERANCE.get(league, 1.0)

                if delta <= league_tolerance:
                    best_market = nearest[2]
                    if delta == 0:
                        match_status = "matched"
                        match_reason = "total_match_exact"
                        out.at[idx, "kalshi_line_diff"] = 0.0
                    else:
                        match_status = "matched"
                        match_reason = "nearest_line_proxy"
                        out.at[idx, "kalshi_line_diff"] = delta
                else:
                    best_market = None
                    match_status = "miss"
                    match_reason = "alt_line_mismatch"

        else:
            # SPREAD LOGIC
            # Use spread-like markets inside the event and fallback by inferred text family.
            row_market_type = market_type
            markets = [
                m for m in nested_markets
                if market_type_matches(row_market_type, m.get('title'), m.get('subtitle'))
                or _infer_market_family_from_text(m.get('title'), m.get('subtitle'), row_market_type) == "spread"
            ]

            raw_spread_line = _safe_text(row.get("spread_line"))
            match = re.search(r"[-+]?\s*(\d+(?:\.\d+)?)", raw_spread_line)

            target_line = None
            if match:
                target_line = float(match.group(1))

            # Moneyline fallback
            is_ml_pick = any(x in _safe_text(row.get("market_type")).lower() for x in ["moneyline", "h2h"]) or target_line == 0.0 or target_line is None
            if is_ml_pick:
                for mkt in markets:
                    ticker_str = str(mkt.get("ticker", ""))
                    ticker_parts = ticker_str.split("-")

                    if ticker_parts:
                        ticker_prefix = (ticker_parts[0].replace('KX', '')
                                         .replace('MBSPREAD', 'B')
                                         .replace('MBTOTAL', 'B')
                                         .replace('BGAME', 'B'))

                        if ticker_prefix == "NCAAMB":
                            ticker_prefix = "NCAAB"

                        if not ticker_prefix.startswith(clean_league):
                            continue

                    m_title = str(mkt.get("title") or "").lower()
                    m_subtitle = str(mkt.get("subtitle") or "").lower()
                    combined_text = f"{m_title} {m_subtitle}"

                    e_ticker = str(best_event_match.get('event_ticker') or "")
                    e_title = str(best_event_match.get('title') or "")
                    # Include event ticker context for guaranteed code identification (e.g. NYR, TOR)
                    combined_upper = f"{combined_text} {e_title} {e_ticker}".upper()

                    if family == "moneyline" or any(x in combined_text for x in ["moneyline", "to win", " win ", "winner", " winner ", "vs", "win by", "wins by"]):
                        # Identify codes for both teams
                        h_code = team_code_for_league(league, row.get("home_team")).upper()
                        a_code = team_code_for_league(league, row.get("away_team")).upper()

                        # Upgrade identification to token-intersection
                        h_tokens = _team_tokens_for_match(row.get("home_team"))
                        a_tokens = _team_tokens_for_match(row.get("away_team"))
                        market_tokens = set(_normalize_team_token(combined_text).split())

                        # Accept if EITHER team is mentioned. Inversion is handled downstream.
                        is_h_match = (h_code != "" and h_code in combined_upper) or \
                                     bool(h_tokens.intersection(market_tokens))
                        is_a_match = (a_code != "" and a_code in combined_upper) or \
                                     bool(a_tokens.intersection(market_tokens))

                        if not is_h_match and not is_a_match:
                            # Trust the event-level match if this is a general 'win' or 'winner' market
                            is_h_match = True

                        if is_h_match or is_a_match:
                            best_market = mkt
                            match_status = "matched"
                            match_reason = "moneyline_match_fuzzy"
                            break
            else:
                target_line = abs(target_line)

                kalshi_lines = []
                for mkt in markets:
                    ticker_str = str(mkt.get("ticker", ""))
                    ticker_parts = ticker_str.split("-")

                    if ticker_parts:
                        ticker_prefix = (ticker_parts[0].replace('KX', '')
                                         .replace('MBSPREAD', 'B')
                                         .replace('MBTOTAL', 'B')
                                         .replace('BGAME', 'B'))

                        if ticker_prefix == "NCAAMB":
                            ticker_prefix = "NCAAB"

                        if not ticker_prefix.startswith(clean_league):
                            continue

                    m_title = str(mkt.get("title") or "").lower()
                    m_subtitle = str(mkt.get("subtitle") or "").lower()
                    combined_text = f"{m_title} {m_subtitle}"

                    m_type = market_type
                    book_line = pd.to_numeric(row.get("spread_line"), errors="coerce")
                    is_favorite_bet = book_line < 0

                    home_t = home_team_name
                    away_t = away_team_name

                    home_shared = {w for w in set(_normalize_team_token(home_t).split()).intersection(set(_normalize_team_token(combined_text).split())) if len(w) > 2}
                    away_shared = {w for w in set(_normalize_team_token(away_t).split()).intersection(set(_normalize_team_token(combined_text).split())) if len(w) > 2}

                    ticker_suffix = str(mkt.get("ticker", "")).split("-")[-1]

                    # Fuzzy match code relies on string intersection primarily now,
                    # but we can try to grab the code from the ticker itself if present
                    h_code = team_code_for_league(league, home_t).upper()
                    a_code = team_code_for_league(league, away_t).upper()

                    # Include event context so we can identify 'Home team' or 'Away team' mentions
                    e_title = str(best_event_match.get('title') or "")
                    e_subtitle = str(best_event_match.get('sub_title') or "")
                    combined_upper = f"{combined_text} {e_title} {e_subtitle}".upper()

                    # Identify subject using both name tokens AND tickers
                    kalshi_subject_is_home = bool(home_shared) or (h_code != "" and h_code in combined_upper)
                    kalshi_subject_is_away = bool(away_shared) or (a_code != "" and a_code in combined_upper)

                    # 1. Identify which team the pick is actually on
                    pick_team_name = _safe_text(row.get("pick_team")).strip()
                    # home_team_name was defined earlier as home_team_val, we'll re-extract to be safe
                    home_team_name_for_match = _safe_text(row.get("home_team")).strip()

                    # 2. Determine if the pick is on the Home or Away team
                    # We use this to know which Kalshi subject we are looking for
                    if pick_team_name == home_team_name_for_match:
                        expected_subject_is_home = True
                    else:
                        expected_subject_is_home = False

                    # Loosen the filter: Accept the market if it's about EITHER team in the game.
                    # The orientation logic (around line 1150) will handle the probability inversion.
                    is_correct_match = kalshi_subject_is_home or kalshi_subject_is_away

                    # Fallback: if we can't identify either subject, trust the event match
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
                                if abs(bid - ask) > 40.0:
                                    kalshi_prob = None
                                else:
                                    kalshi_prob = (bid + ask) / 200.0
                            elif bid > 0 and ask > 0:
                                if abs(bid - ask) > 0.40:
                                    kalshi_prob = None
                                else:
                                    kalshi_prob = (bid + ask) / 2.0
                            else:
                                kalshi_prob = _select_probability(mkt)

                            if kalshi_prob is not None:
                                kalshi_lines.append((k_line, kalshi_prob, mkt))

                if kalshi_lines:
                    target_line_abs = abs(float(target_line))

                    # Side-aware tie-break (7 Jul): a run-line event carries TWO
                    # markets at the same strike — "Toronto wins by over 1.5" and
                    # "San Francisco wins by over 1.5". A pick on SF +1.5 is only
                    # priceable off the TORONTO-subject contract (its NO side);
                    # picking the same-line market by list order chose the SF one
                    # and orientation correctly refused it -> permanent
                    # spread_side_unpriceable miss. Among equal-|Δline| candidates,
                    # prefer a market whose subject makes the pick priceable.
                    def _spread_unpriceable_rank(mkt) -> int:
                        try:
                            from core.team_mapper import normalize_team_name as _norm
                            t = f" {_norm(str(mkt.get('title') or ''))} "
                            h_idx = find_team_reference(t, str(_norm(str(row.get('home_team') or ''))))
                            a_idx = find_team_reference(t, str(_norm(str(row.get('away_team') or ''))))
                            if h_idx < 0 and a_idx < 0:
                                return 1
                            subj_is_home = h_idx >= 0 and (a_idx < 0 or h_idx <= a_idx)
                            ok = orient_spread_kalshi_prob(
                                0.5, subj_is_home, "home" in str(market_type).lower(),
                                pd.to_numeric(row.get("spread_line"), errors="coerce"),
                            )
                            return 0 if ok is not None else 1
                        except Exception:
                            return 1

                    nearest = min(
                        kalshi_lines,
                        key=lambda x: (abs(float(x[0]) - target_line_abs), _spread_unpriceable_rank(x[2])),
                    )
                    delta = abs(float(nearest[0]) - target_line_abs)
                    league_tolerance = MAX_LINE_TOLERANCE.get(league, 3.5)

                    if delta <= league_tolerance:
                        best_market = nearest[2]
                        if delta == 0:
                            match_status = "matched"
                            match_reason = "spread_match_exact"
                            out.at[idx, "kalshi_line_diff"] = 0.0
                        else:
                            match_status = "matched"
                            match_reason = "nearest_line_proxy"
                            out.at[idx, "kalshi_line_diff"] = delta
                    else:
                        best_market = None
                        match_status = "miss"
                        match_reason = "alt_line_mismatch"

        if best_market is None:
            # We found no markets or candidates at all
            out.at[idx, "kalshi_match_status"] = "miss"
            if match_reason == "no_market_for_tickers":
                 out.at[idx, "kalshi_match_reason"] = "no_suitable_market_in_event"
            else:
                 out.at[idx, "kalshi_match_reason"] = match_reason
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
                    if abs(bid - ask) > 0.40:
                        logger.warning(f"Kalshi market illiquid: spread > 0.40 for {best_market.get('ticker')}")
                        kalshi_prob = None
                    else:
                        kalshi_prob = (bid + ask) / 2.0
                else:
                    kalshi_prob = _select_probability(best_market)

            if kalshi_prob is None:
                out.at[idx, "kalshi_match_status"] = "miss"
                out.at[idx, "kalshi_match_reason"] = "illiquid_market"
                continue

            # Extreme-price guard: a totals/run-line contract at near-certainty is settled
            # or illiquid, not a real read. Tight-spread settled markets (bid 0.99/ask 1.00)
            # slip past the >0.40 spread guard above, so check the price itself.
            if kalshi_total_spread_price_extreme(kalshi_prob, row.get("market_type")):
                out.at[idx, "kalshi_match_status"] = "miss"
                out.at[idx, "kalshi_match_reason"] = "extreme_implausible_price"
                out.at[idx, "kalshi_probability"] = 0.0
                continue

            from core.team_mapper import normalize_team_name as _global_normalize

            # 1. Extract raw Kalshi probability safely
            raw_prob = _safe_float(kalshi_prob)

            if raw_prob == 0.0:
                out.at[idx, "kalshi_match_status"] = "miss"
                out.at[idx, "kalshi_match_reason"] = "zero_probability"
                out.at[idx, "kalshi_probability"] = 0.0
                continue

            # 2. Extract specific market title and normalize
            raw_title = str(best_market.get('title', ''))
            norm_title = f" {_global_normalize(raw_title)} "

            # 3. Apply aliases to the market title for accurate indexing
            if league.upper() in ['NCAAB', 'NCAAM']:
                alias_map = {
                    "queens nc": "queens university",
                    "connecticut": "uconn",
                    "wright st": "wright state",
                    "liu": "long island university",
                    "central florida": "ucf"
                }
                for k_alias, standard in alias_map.items():
                    norm_title = norm_title.replace(f" {k_alias} ", f" {standard} ")

            pad_home = f" {_global_normalize(str(row.get('home_team', '')))} "
            pad_away = f" {_global_normalize(str(row.get('away_team', '')))} "

            # 4. First-Subject Orientation Logic
            market_type_str = str(row.get('market_type', '')).lower()
            if "total" in raw_title.lower() or "total" in norm_title or "total" in market_type_str:
                # Wrong-game title guard: spreads implicitly reject a title that names
                # neither team (subject orientation misses -> unpriceable), but totals
                # took the price on faith. A same-strike contract from another game's
                # ladder (3 Jul: Cubs/StL Under 10.5 priced off "Boston vs Los Angeles A
                # Total Runs?" at 73%) must be a miss, not a 73% consensus flip.
                if not kalshi_title_references_matchup(norm_title, pad_home.strip(), pad_away.strip()):
                    out.at[idx, "kalshi_match_status"] = "miss"
                    out.at[idx, "kalshi_match_reason"] = "wrong_game_title"
                    out.at[idx, "kalshi_probability"] = 0.0
                    continue
                # Totals contracts always represent the OVER
                final_prob = raw_prob
            else:
                # Spread/Moneyline: The first team mentioned is the subject.
                # Prefix-tolerant: Kalshi truncates names in titles ("New York Y
                # wins by over 1.5 runs?"), which exact containment can't see.
                home_idx = find_team_reference(norm_title, pad_home.strip()) if pad_home.strip() else 999
                away_idx = find_team_reference(norm_title, pad_away.strip()) if pad_away.strip() else 999

                home_idx = 999 if home_idx == -1 else home_idx
                away_idx = 999 if away_idx == -1 else away_idx

                if home_idx == 999 and " home " in norm_title:
                    home_idx = norm_title.find(" home ")
                if away_idx == 999 and " away " in norm_title:
                    away_idx = norm_title.find(" away ")

                if away_idx < home_idx:
                    # Away team is the subject. Invert to anchor to Home team.
                    final_prob = 1.0 - raw_prob
                else:
                    final_prob = raw_prob

            # Sanity check
            if pd.isna(final_prob) or final_prob <= 0.0 or final_prob > 1.0:
                logger.warning(f"⚠️ Invalid Kalshi probability {final_prob}, skipping row")
                out.at[idx, "kalshi_match_status"] = "error"
                out.at[idx, "kalshi_match_reason"] = f"invalid_probability_{final_prob}"
                out.at[idx, "kalshi_probability"] = 0.0
                continue

            # 5. Orient to the pick side.
            if "spread" in market_type_str:
                # Run lines: a "S wins by over L" contract prices only S -L and Opp(S) +L;
                # the other two sides are unpriceable (see orient_spread_kalshi_prob). The
                # subject S is the first team named in the title.
                _oriented = (
                    None
                    if home_idx == 999 and away_idx == 999
                    else orient_spread_kalshi_prob(
                        raw_prob,
                        home_idx <= away_idx,
                        "home" in market_type_str,
                        pd.to_numeric(row.get("spread_line"), errors="coerce"),
                    )
                )
                if _oriented is None:
                    out.at[idx, "kalshi_match_status"] = "miss"
                    out.at[idx, "kalshi_match_reason"] = "spread_side_unpriceable"
                    out.at[idx, "kalshi_probability"] = 0.0
                    continue
                final_pick_prob = float(_oriented)
            else:
                # Totals (over/under) and moneyline are exact complements at the same line,
                # so the away/under side is genuinely 1 - the over/home probability.
                is_away_or_under = any(x in market_type_str for x in ["away", "under"])
                final_pick_prob = 1.0 - float(final_prob) if is_away_or_under else float(final_prob)

            # PROXY DECAY (New Logic)
            delta = _safe_float(out.at[idx, "kalshi_line_diff"])
            if delta > 0:
                # Decay a proxy (non-exact line) match toward 0.5, scaled by the SAME
                # tolerance the family was matched under, so a totals proxy decays on
                # the tight totals scale rather than the loose generic one.
                is_total_row = "total" in market_type_str or "over" in market_type_str or "under" in market_type_str
                tolerance = (
                    MAX_TOTAL_LINE_TOLERANCE.get(league, 1.0)
                    if is_total_row
                    else MAX_LINE_TOLERANCE.get(league, 3.5)
                )
                decay = min(0.4, delta / (tolerance * 1.5))
                final_pick_prob = (final_pick_prob * (1 - decay)) + (0.5 * decay)

            out.at[idx, "kalshi_probability"] = final_pick_prob
            out.at[idx, "kalshi_market_title"] = best_market.get("title")
            out.at[idx, "kalshi_event_ticker"] = best_market.get("event_ticker")
            out.at[idx, "kalshi_market_ticker"] = best_market.get("ticker")
            out.at[idx, "kalshi_match_status"] = match_status
            out.at[idx, "kalshi_match_reason"] = match_reason
            out.at[idx, "kalshi_match_quality"] = "line_matched"
            # Instrumentation (17 Jun): surface HOW each P(over) was derived so a
            # systematic over-bias can be diagnosed from the export instead of guessed.
            #   kalshi_raw_over_prob : Kalshi's P(over) BEFORE pick-side orientation and
            #                          proxy decay (i.e. the raw matched-contract price).
            #   kalshi_matched_line  : the Kalshi contract line actually used.
            #   kalshi_line_diff     : |Kalshi line - pick line| (already set on match).
            # If matched_line == pick line and raw_over_prob is still ~0.59 across the
            # slate, the bias is in the Kalshi feed/de-vig, not our line-matching.
            out.at[idx, "kalshi_raw_over_prob"] = round(float(final_prob), 4)
            try:
                out.at[idx, "kalshi_matched_line"] = _extract_kalshi_line(
                    best_market, is_total=bool(is_totals_query)
                )
            except Exception:
                out.at[idx, "kalshi_matched_line"] = pd.NA
            # Raw Kalshi contract fields, to confirm strike SEMANTICS (and rule out a
            # half/full-run offset from the text-parse + integer "+0.5" in
            # _extract_kalshi_line vs the structured floor/cap strike). If title says
            # "more than N" / floor_strike=N while our matched_line is N+0.5 and the
            # YES price is ~0.59, the over is being attributed to a line one run too high.
            out.at[idx, "kalshi_floor_strike"] = _safe_float(best_market.get("floor_strike"))
            out.at[idx, "kalshi_cap_strike"] = _safe_float(best_market.get("cap_strike"))
            out.at[idx, "kalshi_yes_bid"] = _safe_float(best_market.get("yes_bid_dollars"))
            out.at[idx, "kalshi_yes_ask"] = _safe_float(best_market.get("yes_ask_dollars"))

    # NA-safe columns to prevent ambiguous boolean evaluation downstream.
    out["kalshi_probability"] = pd.to_numeric(out.get("kalshi_probability"), errors="coerce")
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
