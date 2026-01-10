"""
Kalshi Integrator with RSA Signing & Multi-Sport Fetching.
Location: app_core/kalshi_integrator.py
"""
from __future__ import annotations

import logging
import os
import time
import base64
import random
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytz
import requests
import streamlit as st

try:
    import rapidfuzz
    from rapidfuzz import fuzz
except ImportError:
    rapidfuzz = None
    fuzz = None

# Cryptography for RSA Signing (Required for Kalshi v2)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

__all__ = [
    "KalshiIntegrator",
    "LEAGUE_SERIES_MAP",
    "KalshiMatchResult",
    "KalshiAPIError",
    "KalshiRateLimitError",
    "league_game_prefix",
    "league_series_ticker",
    "team_code_for_league",
    "parse_event_ticker_codes",
    "resolve_team_code",
    "NCAAB_CODE_ALIASES",
]

# Timezone for NBA date buckets (games are bucketed by their US/Eastern date usually, or strict UTC date tokens)
# Kalshi NBA Date Tokens (YYMONDD) often align with UTC, but game times are local.
# We need consistent handling.
NBA_TZ = pytz.timezone("US/Eastern")

class KalshiAPIError(Exception):
    """Base error for Kalshi API issues."""
    pass


class KalshiRateLimitError(KalshiAPIError):
    """Raised when 429 is encountered."""
    pass


@dataclass
class KalshiMatchResult:
    matched: bool = False
    league: str = ""
    event_ticker: Optional[str] = None
    market_ticker: Optional[str] = None
    title: Optional[str] = None
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    mid_prob: Optional[float] = None
    reason: Optional[str] = None
    market_type: Optional[str] = None
    game_date: Optional[datetime] = None
    kalshi_available: bool = True
    label: str = ""
    probability: Optional[float] = None
    raw_event_id: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Constants & Mappings
# ---------------------------------------------------------------------------
SUPPORTED_LEAGUES = {"NBA", "NFL", "MLB", "NHL", "NCAAF", "NCAAB"}
SAFE_KALSHI_STATUSES = {"active", "finalized", "settled", "closed"}


def normalize_status(status: Optional[str]) -> Optional[str]:
    if not status:
        return None
    s = str(status).strip().lower()
    if not s or s == "open":
        return None
    if s in SAFE_KALSHI_STATUSES:
        return s
    return None

LEAGUE_SERIES_MAP: Dict[str, Any] = {
    "NBA": ["KXNBAGAME", "KXNBATOTAL", "KXNBASPREAD", "KXNBA"],
    "NFL": ["KXNFLGAME", "KXNFLTOTAL", "KXNFLSPREAD", "KXNFL"],
    "MLB": ["KXMLBGAME", "KXMLB"],
    "NHL": ["KXNHLGAME", "KXNHL"],
    "NCAAF": ["KXNCAAFGAME", "KXNCAAF"],
    "NCAAB": ["KXNCAAMBGAME", "KXNCAABGAME", "KXNCAAB"],
}


def parse_event_ticker_codes(event_ticker: str) -> Dict[str, str]:
    """
    Extracts away/home codes from Kalshi’s event_ticker.
    Examples:
      KXNBAGAME-26JAN09NYKPHX -> away=NYK, home=PHX
      KXNCAAMBGAME-26JAN10NCSTFSU -> away/home are the trailing 6–8 chars after date token.
    """
    if not event_ticker:
        return {}

    parts = event_ticker.split('-')
    if len(parts) < 2:
        return {}

    # parts[0] is like KXNBAGAME
    # parts[1] is like 26JAN09NYKPHX

    suffix = parts[-1]

    # Regex to find date token at start of suffix
    # Date token: 2 digits, 3 letters, 2 digits.
    match = re.match(r"^(\d{2}[A-Z]{3}\d{2})([A-Z0-9]+)$", suffix)
    if not match:
        return {}

    date_token = match.group(1)
    team_block = match.group(2)

    length = len(team_block)
    away = ""
    home = ""

    if length == 6:
        # 3+3
        away = team_block[:3]
        home = team_block[3:]
    elif length == 8:
        # 4+4
        away = team_block[:4]
        home = team_block[4:]
    else:
        # Fallback: 3/3 from end as requested
        if length >= 3:
            home = team_block[-3:]
            away = team_block[:-3]

    return {"away": away, "home": home, "date_token": date_token}


def league_series_ticker(league: str) -> Optional[str]:
    league_key = (league or "").upper()
    prefix = LEAGUE_SERIES_MAP.get(league_key)
    if isinstance(prefix, list):
        for candidate in prefix:
            if not candidate:
                continue
            cand_upper = candidate.upper()
            if cand_upper == f"KX{league_key}":
                return candidate
        for candidate in prefix:
            if candidate and "GAME" not in candidate.upper():
                return candidate
        return prefix[-1] if prefix else None
    return prefix


def league_game_prefix(league: str) -> str:
    league_key = (league or "").upper()
    series = league_series_ticker(league_key) or f"KX{league_key}"
    return f"{series}GAME"


def clean_team_name(name: str) -> str:
    """Robust cleaning preserving spaces for map lookup."""
    return re.sub(r"[^A-Z0-9 ]", " ", str(name or "").upper()).strip()


def normalize_name(name: str) -> str:
    """Legacy normalize - strips everything non-alpha. Kept for back-compat but generally avoided now."""
    return re.sub(r"[^A-Z]", "", (name or "").upper())


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

# NBA: full name -> 3-letter code
NBA_TEAM_CODE_MAP = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}

# NFL: full name -> 2–3 letter code
NFL_TEAM_CODE_MAP = {
    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
}

# NHL: full name -> 3-letter-ish code
NHL_TEAM_CODE_MAP = {
    "ANAHEIM DUCKS": "ANA",
    "ARIZONA COYOTES": "ARI",
    "BOSTON BRUINS": "BOS",
    "BUFFALO SABRES": "BUF",
    "CALGARY FLAMES": "CGY",
    "CAROLINA HURRICANES": "CAR",
    "CHICAGO BLACKHAWKS": "CHI",
    "COLORADO AVALANCHE": "COL",
    "COLUMBUS BLUE JACKETS": "CBJ",
    "DALLAS STARS": "DAL",
    "DETROIT RED WINGS": "DET",
    "EDMONTON OILERS": "EDM",
    "FLORIDA PANTHERS": "FLA",
    "LOS ANGELES KINGS": "LAK",
    "MINNESOTA WILD": "MIN",
    "MONTREAL CANADIENS": "MTL",
    "NASHVILLE PREDATORS": "NSH",
    "NEW JERSEY DEVILS": "NJD",
    "NEW YORK ISLANDERS": "NYI",
    "NEW YORK RANGERS": "NYR",
    "OTTAWA SENATORS": "OTT",
    "PHILADELPHIA FLYERS": "PHI",
    "PITTSBURGH PENGUINS": "PIT",
    "SAN JOSE SHARKS": "SJS",
    "SEATTLE KRAKEN": "SEA",
    "ST LOUIS BLUES": "STL",
    "TAMPA BAY LIGHTNING": "TBL",
    "TORONTO MAPLE LEAFS": "TOR",
    "VANCOUVER CANUCKS": "VAN",
    "VEGAS GOLDEN KNIGHTS": "VGK",
    "WINNIPEG JETS": "WPG",
}

# MLB: full name -> 3-letter-ish code
MLB_TEAM_CODE_MAP = {
    "ARIZONA DIAMONDBACKS": "ARI",
    "ATLANTA BRAVES": "ATL",
    "BALTIMORE ORIOLES": "BAL",
    "BOSTON RED SOX": "BOS",
    "CHICAGO CUBS": "CHC",
    "CHICAGO WHITE SOX": "CWS",
    "CINCINNATI REDS": "CIN",
    "CLEVELAND GUARDIANS": "CLE",
    "COLORADO ROCKIES": "COL",
    "DETROIT TIGERS": "DET",
    "HOUSTON ASTROS": "HOU",
    "KANSAS CITY ROYALS": "KC",
    "LOS ANGELES ANGELS": "LAA",
    "LOS ANGELES DODGERS": "LAD",
    "MIAMI MARLINS": "MIA",
    "MILWAUKEE BREWERS": "MIL",
    "MINNESOTA TWINS": "MIN",
    "NEW YORK METS": "NYM",
    "NEW YORK YANKEES": "NYY",
    "OAKLAND ATHLETICS": "OAK",
    "PHILADELPHIA PHILLIES": "PHI",
    "PITTSBURGH PIRATES": "PIT",
    "SAN DIEGO PADRES": "SD",
    "SAN FRANCISCO GIANTS": "SF",
    "SEATTLE MARINERS": "SEA",
    "ST LOUIS CARDINALS": "STL",
    "TAMPA BAY RAYS": "TB",
    "TEXAS RANGERS": "TEX",
    "TORONTO BLUE JAYS": "TOR",
    "WASHINGTON NATIONALS": "WSH",
}

# Skeletons for college – fill with your existing mappings and new schools from logs
NCAAF_TEAM_CODE_MAP: Dict[str, str] = {
    # "ALABAMA CRIMSON TIDE": "ALA",
    # "GEORGIA BULLDOGS": "UGA",
}

NCAAB_TEAM_CODE_MAP: Dict[str, str] = {
    "DUKE": "DUK", "NORTH CAROLINA": "UNC", "KANSAS": "KAN", "KENTUCKY": "KEN",
    "GONZAGA": "GON", "BAYLOR": "BAY", "ARIZONA": "ARI", "UCLA": "UCL",
    "HOUSTON": "HOU", "PURDUE": "PUR", "UCONN": "CON", "CONNECTICUT": "CON",
    "VILLANOVA": "VIL", "MICHIGAN STATE": "MSU", "TENNESSEE": "TEN", "ALABAMA": "ALA",
    "AUBURN": "AUB", "TEXAS": "TEX", "VIRGINIA": "VIR", "ILLINOIS": "ILL",
    "ARKANSAS": "ARK", "INDIANA": "IND", "MICHIGAN": "MIC", "OHIO STATE": "OSU",
    "FLORIDA": "FLO", "TEXAS TECH": "TTU", "WISCONSIN": "WIS", "MARYLAND": "MAR",
    "IOWA": "IOW", "XAVIER": "XAV", "CREIGHTON": "CRE", "MARQUETTE": "MAR",
    "PROVIDENCE": "PRO", "SETON HALL": "SET", "ST. JOHN'S": "STJ", "ST JOHNS": "STJ",
    "GEORGETOWN": "GEO", "BUTLER": "BUT", "DEPAUL": "DEP", "MEMPHIS": "MEM",
    "CINCINNATI": "CIN", "SMU": "SMU", "WICHITA STATE": "WIC", "TEMPLE": "TEM",
    "TULANE": "TUL", "USF": "USF", "UCF": "UCF", "ECU": "ECU", "TULSA": "TUL",
    "DAYTON": "DAY", "VCU": "VCU", "SAINT LOUIS": "SLU", "ST. BONAVENTURE": "SBU",
    "RICHMOND": "RIC", "DAVIDSON": "DAV", "LOYOLA CHICAGO": "LOY", "SAN DIEGO STATE": "SDS",
    "SAN DIEGO ST": "SDS", "NEVADA": "NEV", "UTAH STATE": "USU", "BOISE STATE": "BOI",
    "BOISE ST": "BOI", "UNLV": "UNLV", "NEW MEXICO": "UNM", "COLORADO STATE": "CSU",
    "SAINT MARY'S": "SMC", "ST MARYS": "SMC", "SAN FRANCISCO": "USF", "BYU": "BYU",
    "SANTA CLARA": "SCU", "PEPPERDINE": "PEP", "LMU": "LMU", "PACIFIC": "PAC",
    "PORTLAND": "POR", "SAN DIEGO": "USD", "TCU": "TCU", "IOWA STATE": "ISU",
    "KANSAS STATE": "KSU", "OKLAHOMA": "OKL", "OKLAHOMA STATE": "OSU", "WEST VIRGINIA": "WVU",
    "LOUISVILLE": "LOU", "SYRACUSE": "SYR", "NOTRE DAME": "UND", "MIAMI": "MIA",
    "FLORIDA STATE": "FSU", "CLEMSON": "CLE", "NC STATE": "NCS", "WAKE FOREST": "WAK",
    "PITTSBURGH": "PIT", "BOSTON COLLEGE": "BC", "GEORGIA TECH": "GAT", "VIRGINIA TECH": "VAT",
    "LSU": "LSU", "TEXAS A&M": "TAM", "MISSISSIPPI STATE": "MSU", "OLE MISS": "MIS",
    "MISSOURI": "MIZ", "SOUTH CAROLINA": "SCA", "GEORGIA": "GEO", "VANDERBILT": "VAN",
    "OREGON": "ORE", "OREGON STATE": "OSU", "USC": "USC", "WASHINGTON": "WAS",
    "WASHINGTON STATE": "WSU", "COLORADO": "COL", "UTAH": "UTA", "ARIZONA STATE": "ASU",
    "CALIFORNIA": "CAL", "STANFORD": "STA", "RUTGERS": "RUT", "PENN STATE": "PSU",
    "MINNESOTA": "MIN", "NORTHWESTERN": "NOR", "NEBRASKA": "NEB",
    # New additions from logs/user
    "LOUISVILLE CARDINALS": "LOU",
    "BOSTON COLLEGE EAGLES": "BC",
    "FLORIDA STATE SEMINOLES": "FSU",
    "NC STATE WOLFPACK": "NCS",
    "ST BONAVENTURE BONNIES": "SBU",
    "MIAMI HURRICANES": "MIA",
    "GEORGIA TECH YELLOW JACKETS": "GAT",
}

# Alias Maps: Kalshi Variant -> Canonical Internal Code
NCAAB_CODE_ALIASES: Dict[str, str] = {
    "NCST": "NCS",
    "MICH": "MIC",
    "MISS": "MIS",
    "TENN": "TEN",
    "PITT": "PIT",
    "CONN": "CON",
    "MINN": "MIN",
    "WISC": "WIS",
    "ARIZ": "ARI",
    "CINC": "CIN",
    "GONZ": "GON",
    "VILL": "VIL",
    "PROV": "PRO",
    "MARQ": "MAR",
    "CREI": "CRE",
    "XAVI": "XAV",
    "BUTL": "BUT",
    "SETO": "SET",
    "GEOR": "GEO",
    "DEPA": "DEP",
}

NCAAF_CODE_ALIASES: Dict[str, str] = {
    "NCST": "NCS",
    "MICH": "MIC",
    "MISS": "MIS",
    "TENN": "TEN",
    "PITT": "PIT",
    "CONN": "CON",
    "MINN": "MIN",
    "WISC": "WIS",
    "ARIZ": "ARI",
    "CINC": "CIN",
}

def resolve_team_code(code: str, league: str) -> str:
    """
    Resolve a team code (from event ticker or map) to its canonical form
    using alias maps if applicable.
    """
    if not code:
        return ""

    c = code.upper().strip()
    l = (league or "").upper()

    if l == "NCAAB":
        return NCAAB_CODE_ALIASES.get(c, c)
    elif l == "NCAAF":
        return NCAAF_CODE_ALIASES.get(c, c)

    return c


def team_name_to_code(league: str, team_name: str) -> Optional[str]:
    """Translate a full team name into its Kalshi ticker code when available."""
    if not team_name:
        return None

    league_u = (league or "").upper()
    team_clean = clean_team_name(team_name)

    map_to_use = None
    if league_u == "NBA":
        map_to_use = NBA_TEAM_CODE_MAP
    elif league_u == "NFL":
        map_to_use = NFL_TEAM_CODE_MAP
    elif league_u == "NHL":
        map_to_use = NHL_TEAM_CODE_MAP
    elif league_u == "MLB":
        map_to_use = MLB_TEAM_CODE_MAP
    elif league_u == "NCAAF":
        map_to_use = NCAAF_TEAM_CODE_MAP
    elif league_u == "NCAAB":
        map_to_use = NCAAB_TEAM_CODE_MAP

    if map_to_use:
        # 1. Direct lookup
        if team_clean in map_to_use:
            return map_to_use[team_clean]

        # 2. Fuzzy / Subset lookup
        # Iterate keys sorted by length descending to match longest specific keys first
        # (e.g. match "IOWA STATE" before "IOWA")
        sorted_keys = sorted(map_to_use.keys(), key=len, reverse=True)
        for key in sorted_keys:
            # Check if map key is substring of input OR input is substring of map key
            if key in team_clean or team_clean in key:
                return map_to_use[key]

        # 3. Special handling for College (State -> St, Saint -> St)
        if league_u in ["NCAAB", "NCAAF"]:
            alt = team_clean.replace("STATE", "ST").replace("SAINT", "ST")
            if alt in map_to_use:
                return map_to_use[alt]

    return None


def team_code_for_league(league: str, team_name: str) -> str:
    """Return a non-empty ticker-friendly code for a team within a league."""
    if not team_name:
        return "UNK"

    league_u = (league or "").upper()

    # 1. Try explicit mapping
    mapped = team_name_to_code(league_u, team_name)
    if mapped:
        return mapped

    cleaned = clean_team_name(team_name)

    # 2. Check general abbreviations list
    if cleaned in KALSHI_TEAM_ABBREVIATIONS:
        codes = KALSHI_TEAM_ABBREVIATIONS.get(cleaned) or []
        if codes:
            return str(codes[0]).upper()

    # 3. Heuristic generation
    # Keep only letters for fallback
    letters = re.sub(r"[^A-Z]", "", cleaned)
    if letters:
        return letters[:3]

    return "UNK"

def price_to_prob(price: Any) -> Optional[float]:
    if price is None: return None
    try:
        val = float(price)
        if 0.0 <= val <= 100.0: return max(0.0, min(1.0, val / 100.0))
    except: pass
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
    """Generate potential ticker codes from a team name, preserving spaces."""
    # Ensure spaces are preserved by clean_team_name (it replaces non-alphanum with space)
    cleaned = clean_team_name(team_name)
    codes: List[str] = []

    # 1. Full matches in abbrev list
    if cleaned in KALSHI_TEAM_ABBREVIATIONS:
        codes.extend(KALSHI_TEAM_ABBREVIATIONS[cleaned])

    # 2. Token based codes
    tokens = [t for t in cleaned.split() if t]

    # Add full tokens (e.g. "GOLDEN", "STATE", "WARRIORS")
    for t in tokens:
        if len(t) >= 2 and t not in codes:
            codes.append(t)

    # Add first 3 chars of tokens
    for t in tokens:
        if len(t) >= 3:
            sub = t[:3]
            if sub not in codes:
                codes.append(sub)

    # Add first 2 chars
    for t in tokens:
        if len(t) >= 2:
            sub = t[:2]
            if sub not in codes:
                codes.append(sub)

    # 3. Initials (e.g. GSW from Golden State Warriors)
    if tokens:
        initials = "".join(t[0] for t in tokens)
        if len(initials) >= 2 and initials not in codes:
            codes.append(initials)
        if len(initials) >= 3:
            if initials[:2] not in codes:
                codes.append(initials[:2])

    return list(dict.fromkeys(codes))  # Dedup

def _team_score(team_code: str, target_clean: str, target_codes: List[str]) -> float:
    if not team_code: return 0.0
    clean_code = clean_team_name(team_code)

    # 1. Exact or Code Match (Highest)
    if clean_code in target_codes: return 100.0
    if clean_code == target_clean: return 100.0

    # 2. Token overlap / Containment
    words_code = set(clean_code.split())
    words_target = set(target_clean.split())

    # "LAKERS" in "LOS ANGELES LAKERS"
    if clean_code in target_clean: return 90.0
    if target_clean in clean_code: return 90.0

    if words_code & words_target:
        return 80.0

    # 3. Fuzzy Match (Fallback)
    if rapidfuzz:
        # Simple ratio
        ratio = fuzz.ratio(clean_code, target_clean)
        # Partial ratio (good for "Lakers" vs "LA Lakers")
        partial = fuzz.partial_ratio(clean_code, target_clean)
        return max(ratio, partial)
    
    return 0.0

def _match_via_events(
    integrator: KalshiIntegrator,
    league: str,
    home_codes: List[str],
    away_codes: List[str],
    game_dt_utc: datetime,
    status: Optional[str]
) -> Optional[KalshiMatchResult]:
    """
    Attempt to match a game to an event by scanning the /events endpoint first.
    This is more efficient and accurate for leagues with structured tickers (NBA/NFL/NCAA).
    """
    # 1. Determine series ticker
    series_ticker = None
    if league == "NBA": series_ticker = "KXNBAGAME"
    elif league == "NFL": series_ticker = "KXNFLGAME"
    elif league == "NCAAB": series_ticker = "KXNCAAMBGAME"
    elif league == "NCAAF": series_ticker = "KXNCAAFGAME"
    elif league == "MLB": series_ticker = "KXMLBGAME"
    elif league == "NHL": series_ticker = "KXNHLGAME"

    if not series_ticker:
        return None

    # 2. Fetch events (using cache inside get_events)
    # Use a safe lookback window (e.g., event close time >= game_time - buffer)
    # But get_events min_close_ts filters events that close AFTER this time.
    # We want events that close around game time.
    # If we use status="active", we get current ones.
    # If we want to catch games that might have just started or are about to, "active" is good.
    # For robust matching, we just fetch active + cache.
    try:
        events_resp = integrator.get_events(series_ticker, status=status)
    except Exception:
        return None

    events = events_resp.get("events", [])
    if not events:
        return None

    best_event = None
    best_score = 0.0

    # Time window for matching (hours)
    TIME_WINDOW_HOURS = 36 # Generous window

    for evt in events:
        ticker = evt.get("ticker")
        parsed = parse_event_ticker_codes(ticker)
        if not parsed:
            continue

        evt_away_code = resolve_team_code(parsed.get("away"), league)
        evt_home_code = resolve_team_code(parsed.get("home"), league)

        # Resolve our candidates too
        resolved_home = {resolve_team_code(c, league) for c in home_codes}
        resolved_away = {resolve_team_code(c, league) for c in away_codes}

        # Check codes against our candidates
        # Orientation 1: Event Away == Game Away, Event Home == Game Home
        score_1 = 0
        if evt_away_code in resolved_away: score_1 += 50
        if evt_home_code in resolved_home: score_1 += 50

        # Orientation 2: Swap (unlikely but possible)
        score_2 = 0
        if evt_away_code in resolved_home: score_2 += 50
        if evt_home_code in resolved_away: score_2 += 50

        match_score = max(score_1, score_2)

        if match_score < 50:
            continue

        # Time check
        close_ts = evt.get("close_time") # ISO string
        if close_ts:
            try:
                dt = datetime.fromisoformat(str(close_ts).replace("Z", "+00:00"))
                if dt.tzinfo is None: dt = pytz.utc.localize(dt)

                diff_hours = abs((dt - game_dt_utc).total_seconds()) / 3600.0
                if diff_hours > TIME_WINDOW_HOURS:
                    match_score -= 20 # Penalty for time mismatch
            except:
                pass

        if match_score > best_score:
            best_score = match_score
            best_event = evt

    if best_event and best_score >= 90: # High confidence match
        # Now fetch markets for this event
        evt_ticker = best_event.get("ticker")

        # We need the markets for this event to get the probability.
        # We can use get_markets with event_ticker param if supported, or filter from broad list.
        # But wait, get_markets allows filtering by event_ticker?
        # Typically yes, or we can use the "markets" field if nested (but we set with_nested_markets=False).
        # Let's fetch markets for this specific event ticker.

        # Efficient way: call get_markets with event_ticker param?
        # Kalshi API usually supports event_ticker filter on /markets.
        try:
            markets_resp = integrator._request("GET", "/markets", params={"event_ticker": evt_ticker})
            markets = markets_resp.get("markets", [])
        except Exception:
            markets = []

        if not markets:
            return None

        # Find the main game market (Winner)
        # Usually checking "Winner" title or market type
        target_market = None
        for m in markets:
            # We prefer the main line.
            # Usually generic game winner.
            # Avoid spread/total if we just want the main prob, but function returns a generic result.
            # match_game_to_kalshi usually returns the "Winner" market or best fit.
            t = (m.get("title") or "").lower()
            if "winner" in t:
                target_market = m
                break

        if not target_market and markets:
            target_market = markets[0] # Fallback

        if target_market:
             # Calculate prob
            yes_bid = target_market.get("yes_bid")
            yes_ask = target_market.get("yes_ask")
            prob = None
            if yes_bid and yes_ask:
                 prob = ((yes_bid + yes_ask) / 2) / 100.0
            elif target_market.get("last_price"):
                 prob = target_market.get("last_price") / 100.0

            return KalshiMatchResult(
                matched=True,
                kalshi_available=True,
                label=target_market.get("title"),
                probability=prob if prob is not None else 0.5, # Default if missing
                raw_event_id=evt_ticker,
                league=league,
                reason="matched_via_events_api",
                market_type="winner",
                game_date=game_dt_utc,
                debug={"score": best_score, "event": evt_ticker}
            )

    return None

def match_game_to_kalshi(league: str, home_team: str, away_team: str, game_time: Optional[datetime], integrator: "KalshiIntegrator" = None, status: Optional[str] = None) -> KalshiMatchResult:
    league_key = (league or "").upper()
    kalshi = integrator or KalshiIntegrator()
    
    if not kalshi or not kalshi.api_key:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, reason="no_integrator")

    # Use robust candidate generation
    home_clean = clean_team_name(home_team)
    away_clean = clean_team_name(away_team)

    # Generate extended candidates including league-specific mappings
    home_codes = _build_team_codes(home_team)
    away_codes = _build_team_codes(away_team)

    # Inject mapped codes if available
    mapped_home = team_code_for_league(league_key, home_team)
    if mapped_home and mapped_home != "UNK" and mapped_home not in home_codes:
        home_codes.insert(0, mapped_home)

    mapped_away = team_code_for_league(league_key, away_team)
    if mapped_away and mapped_away != "UNK" and mapped_away not in away_codes:
        away_codes.insert(0, mapped_away)

    # NEW: Try Event-Based Matching First
    if game_time and league_key in ["NBA", "NFL", "NCAAB", "NCAAF", "MLB", "NHL"]:
        # Normalize game_time to UTC
        if game_time.tzinfo is None:
            # Assume UTC if naive? Or try to match without TZ.
            # Best practice: ensure it has timezone.
            gt_utc = pytz.utc.localize(game_time)
        else:
            gt_utc = game_time.astimezone(pytz.UTC)

        event_match = _match_via_events(
            kalshi,
            league_key,
            home_codes,
            away_codes,
            gt_utc,
            status=status
        )
        if event_match:
            return event_match

    # GENERIC MATCHING (Non-NBA or Fallback)
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

    # Constants for fuzzy logic
    DATE_TOLERANCE_DAYS = 1
    # If using rapidfuzz (0-100 scale), threshold needs to be high.
    # We sum two scores (Home + Away), so max is 200.
    # Accept if sum > 130 (avg 65 per team) to be safe but permissive.
    TEAM_FUZZY_THRESHOLD = 130.0

    # Coverage Debug
    markets_considered = 0

    # Scan
    for m in markets:
        meta = _parse_market_metadata(m)
        if not meta:
            continue

        ticker = (m.get("ticker") or "").upper()
        if series_prefix and not ticker.startswith(series_prefix):
            continue

        markets_considered += 1

        teams = meta.get("teams") or []
        if len(teams) < 2:
            continue

        # Score matching
        # Check Home vs Team A/B
        score_home_A = _team_score(teams[0], home_clean, home_codes)
        score_away_B = _team_score(teams[1], away_clean, away_codes)

        # Check Home vs Team B/A (swap)
        score_home_B = _team_score(teams[1], home_clean, home_codes)
        score_away_A = _team_score(teams[0], away_clean, away_codes)

        score_direct = score_home_A + score_away_B
        score_swap = score_home_B + score_away_A

        score = max(score_direct, score_swap)

        m_date = meta.get("market_date")
        if game_dt_utc and m_date:
            try:
                if m_date.tzinfo is None:
                    m_date = pytz.utc.localize(m_date)
                diff = abs((m_date.date() - game_dt_utc.date()).days)

                # Hard Date Cutoff
                if diff > DATE_TOLERANCE_DAYS:
                    continue

                # No penalty for date diff within tolerance in new logic
            except Exception:
                pass

        if score > best_score:
            best_score = score
            best_market = m
            best_market["__meta"] = meta

    if not best_market or best_score < TEAM_FUZZY_THRESHOLD:
        # Debug Logging for Failure
        debug_fail = {
            "markets_considered": markets_considered,
            "best_score": best_score,
            "home_candidates": home_codes,
            "away_candidates": away_codes,
            "best_candidate_ticker": best_market.get("ticker") if best_market else None
        }
        if league_key in ["NBA", "NFL", "NCAAB"]: # Reduce spam
             logger.info(f"Kalshi Match Failed [{league_key}]: {home_clean} vs {away_clean}. Best Score: {best_score}")

        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label="",
            probability=None,
            raw_event_id=None,
            reason=f"low_score_{best_score:.1f}",
            debug=debug_fail
        )

    meta = best_market["__meta"]
    return KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=meta["title"],
        probability=meta["probability"],
        raw_event_id=best_market.get("ticker"),
        league=league_key,
        reason="matched_fuzzy",
        market_type=meta["market_type"],
        game_date=meta["market_date"],
    )

# ---------------------------------------------------------------------------
# KalshiIntegrator Class
# ---------------------------------------------------------------------------

class KalshiIntegrator:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, *, required: bool = False):
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY") or os.getenv("KALSHI_API_KEY")
        raw_secret = api_secret or st.secrets.get("KALSHI_API_SECRET") or os.getenv("KALSHI_API_SECRET")
        self.api_secret_pem = self._normalize_secret(raw_secret)
        self.api_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.session = requests.Session()
        self.required = required
        self.last_error_info = {}
        self.last_status_code = None
        self.last_response_text = None
        self._markets_cache = []
        self._league_cache = {}

        # Caching + error state
        self._markets_cache: List[Dict[str, Any]] = []
        self._markets_cache_ts: float = 0.0
        self.cache_ttl_seconds: int = 120
        self._markets_cache_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._markets_cache_ttl_seconds: int = 600
        self._events_cache: Dict[str, Dict[str, Any]] = {}  # Cache for /events by series_ticker
        self._events_cache_ttl: int = 300
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
        if not secret_val: return None
        cleaned = str(secret_val).replace("\\n", "\n").strip()
        if "-----BEGIN" in cleaned: return cleaned
        return f"-----BEGIN PRIVATE KEY-----\n{cleaned}\n-----END PRIVATE KEY-----"

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

    def health_check(self, league: Optional[str] = None) -> Dict[str, Any]:
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

        league_key = (league or "NBA").upper()
        game_prefix = league_game_prefix(league_key)
        series_prefix = league_series_ticker(league_key) or f"KX{league_key}"

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

            has_game = any(_ticker(m).startswith(f"{game_prefix}-") for m in markets)
            has_futures = any(
                _ticker(m).startswith(series_prefix)
                and not _ticker(m).startswith(f"{game_prefix}-")
                for m in markets
            )
            ok = True
            warning: Optional[str] = None
            if markets and not has_game and has_futures:
                warning = (
                    f"Kalshi reachable, but no {league_key} {game_prefix} markets returned (futures-only or slate not listed)."
                )
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
                "request_params": self.last_request_params,
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

    def self_test_no_open_status(self) -> Dict[str, Any]:
        """Quick live call to ensure we never send status="open" in params."""
        try:
            payload = self._request("GET", "/markets", params={"limit": 10})
            params_sent = self.last_request_params or {}
            return {
                "ok": True,
                "status_code": self.last_status_code,
                "request_params": params_sent,
                "contains_status_open": params_sent.get("status") == "open",
                "market_count": len(payload.get("markets", [])) if isinstance(payload, dict) else 0,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "request_params": self.last_request_params,
                "status_code": self.last_status_code,
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
        return globals()["normalize_status"](status)

    def get_events(
        self,
        series_ticker: str,
        status: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        limit: int = 200,
        cursor: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch events for a series with optional caching.
        """
        cache_key = f"{series_ticker}:{status}:{min_close_ts}"
        now = time.time()

        if use_cache and not cursor:
            cached = self._events_cache.get(cache_key)
            if cached and (now - cached.get("ts", 0)) < self._events_cache_ttl:
                return cached.get("payload", {})

        params = {
            "limit": limit,
            "cursor": cursor,
            "with_nested_markets": False,
            "series_ticker": series_ticker,
        }
        if status:
            params["status"] = normalize_status(status)
        if min_close_ts:
            params["min_close_ts"] = int(min_close_ts)

        params = {k: v for k, v in params.items() if v is not None}

        try:
            resp = self._request("GET", "/events", params=params)
        except Exception:
            # If rate limited or error, return cached if available
            cached = self._events_cache.get(cache_key)
            if use_cache and cached:
                 logger.warning(f"Kalshi get_events failed for {series_ticker}, using cache.")
                 return cached.get("payload", {})
            raise

        if use_cache and not cursor and resp:
            self._events_cache[cache_key] = {"ts": now, "payload": resp}

        return resp

    def scan_and_verify_team_codes(self, league: str) -> Dict[str, Any]:
        """
        Scans events for the given league and verifies if extracted codes exist in our maps.
        """
        series_ticker = LEAGUE_SERIES_MAP.get(league)
        if isinstance(series_ticker, list):
            series_ticker = series_ticker[0]

        if league == "NCAAB":
            series_ticker = "KXNCAAMBGAME"

        if not series_ticker:
            return {"error": f"No series ticker for {league}"}

        # Fetch active events
        try:
            events_resp = self.get_events(series_ticker, status="active", limit=100)
        except Exception as e:
            return {"error": f"Failed to fetch events: {e}"}

        events = events_resp.get("events", [])

        unknown_codes = set()
        known_codes = set()

        map_to_use = None
        if league == "NBA": map_to_use = NBA_TEAM_CODE_MAP
        elif league == "NFL": map_to_use = NFL_TEAM_CODE_MAP
        elif league == "NHL": map_to_use = NHL_TEAM_CODE_MAP
        elif league == "MLB": map_to_use = MLB_TEAM_CODE_MAP
        elif league == "NCAAF": map_to_use = NCAAF_TEAM_CODE_MAP
        elif league == "NCAAB": map_to_use = NCAAB_TEAM_CODE_MAP

        valid_codes = set(map_to_use.values()) if map_to_use else set()
        sample_unknowns = {}

        for evt in events:
            ticker = evt.get("ticker")
            parsed = parse_event_ticker_codes(ticker)
            if not parsed:
                continue

            for side in ["home", "away"]:
                raw_code = parsed.get(side)
                if not raw_code:
                    continue

                # Use resolved code for verification
                code = resolve_team_code(raw_code, league)

                if code in valid_codes:
                    known_codes.add(code)
                else:
                    unknown_codes.add(raw_code) # Store raw code as unknown to prompt alias
                    if raw_code not in sample_unknowns:
                        sample_unknowns[raw_code] = ticker

        return {
            "series_ticker": series_ticker,
            "events_scanned": len(events),
            "unknown_codes": sorted(list(unknown_codes)),
            "known_codes_count": len(known_codes),
            "sample_unknowns": sample_unknowns
        }

    @staticmethod
    def _status_param(status: Optional[str]) -> Dict[str, Any]:
        """Return a valid status parameter for /markets calls."""
        normalized = normalize_status(status)
        return {"status": normalized} if normalized is not None else {}

    def _build_market_params(
        self,
        *,
        status: Optional[str],
        limit: Optional[int],
        cursor: Optional[str],
        extra_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        norm = normalize_status(status)
        params: Dict[str, Any] = {}
        if limit is not None and limit != "":
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        if extra_params:
            for key, val in extra_params.items():
                if val is None or val == "":
                    continue
                params[key] = val
        if norm:
            params["status"] = norm
        return {k: v for k, v in params.items() if v is not None and v != ""}

    def get_markets_paginated(
        self,
        status: Optional[str] = None,
        limit: int = 200,
        max_pages: int = 5,
        cursor: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        norm = normalize_status(status)
        all_markets: List[Dict[str, Any]] = []
        next_cursor = cursor
        pages = 0
        while pages < max_pages:
            params = {
                "limit": limit,
                "cursor": next_cursor,
            }
            if extra_params:
                params.update(extra_params)
            if norm:
                params["status"] = norm
            params = {k: v for k, v in params.items() if v is not None and v != ""}
            self.last_request_params = params
            try:
                data = self._request("GET", "/markets", params=params)
            except KalshiAPIError:
                status_code = (self.last_error_info or {}).get("status_code")
                if status_code == 429:
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
        normalized_status = normalize_status(status)
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

        norm = normalize_status(status)
        all_markets = self.get_markets_paginated(status=norm)
        self._markets_cache = all_markets
        self._markets_cache_ts = now
        logger.info(f"✅ Successfully loaded {len(all_markets)} Kalshi markets (paginated)")
        return all_markets

    def get_nba_game_markets(
        self, *, status: Optional[str] = None, limit: int = 200, max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch NBA markets broadly, returning KXNBAGAME entries when available."""

        normalized_status = normalize_status(status)
        markets: List[Dict[str, Any]] = []
        try:
            markets = self.get_markets_paginated(
                status=normalized_status, limit=limit, max_pages=max_pages
            )
        except Exception:
            logger.exception("NBA market pagination failed")
            return []

        game_markets = [
            m
            for m in markets or []
            if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith("KXNBAGAME-")
        ]

        return game_markets if game_markets else markets

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
        normalized_status = normalize_status(status)
        cache_key = f"{league_key}:{normalized_status or 'any'}"
        now = time.time()
        cached = self._league_cache.get(cache_key)
        if cached and (now - cached.get("ts", 0)) < self._league_cache_ttl:
            cached_markets = cached.get("markets") or []
            # Re-fetch if the cached result was empty so we can pick up newly available markets.
            if cached_markets:
                self.last_fetch_meta = cached.get("meta", {})
                return cached_markets

        futures_noise: List[Dict[str, Any]] = []
        collected: Dict[str, Dict[str, Any]] = {}
        pages = 0
        prefix_hits = 0
        game_hits = 0
        futures_hits = 0

        series_targets: List[str] = []
        game_prefix = league_game_prefix(league_key)
        series_base = league_series_ticker(league_key)

        for candidate in [game_prefix, series_base]:
            if candidate and candidate not in series_targets:
                series_targets.append(candidate)
        if isinstance(prefix, list):
            for candidate in prefix:
                if candidate and candidate not in series_targets:
                    series_targets.append(candidate)
        elif prefix and prefix not in series_targets:
            series_targets.append(prefix)

        for series in series_targets or [None]:
            params = {"series_ticker": series} if series else None
            chunk = self.get_markets_paginated(
                status=None,
                limit=200,
                max_pages=max_pages,
                extra_params=params,
            )
            pages = max(pages, min(max_pages, len(chunk) // 200 + 1))
            for m in chunk or []:
                key = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                if key and key not in collected:
                    collected[key] = m

        all_markets = list(collected.values())
        used_game_prefix = game_prefix
        game_markets = [
            m
            for m in all_markets
            if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith(f"{game_prefix}-")
        ]

        # Fallback: detect alternative game prefixes (e.g., NCAA basketball variants)
        if not game_markets and league_key == "NCAAB":
            alt_game_candidates: List[Dict[str, Any]] = []
            alt_prefix: Optional[str] = None
            date_token_pattern = re.compile(r"\d{2}[A-Z]{3}\d{2}")
            for m in all_markets:
                t_upper = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                if "GAME" in t_upper and ("NCAAB" in t_upper or "NCAA" in t_upper):
                    alt_game_candidates.append(m)
                    if not alt_prefix:
                        alt_prefix = t_upper.split("-")[0]
                elif ("NCAAB" in t_upper or "NCAA" in t_upper) and date_token_pattern.search(t_upper):
                    alt_game_candidates.append(m)
                    if not alt_prefix:
                        alt_prefix = t_upper.split("-")[0]
            if alt_game_candidates:
                game_markets = alt_game_candidates
                used_game_prefix = alt_prefix or game_prefix

        if not all_markets:
            broad = self.get_markets_paginated(
                status=normalized_status, limit=200, max_pages=max_pages
            )
            filtered_broad = self._filter_markets_for_league(broad, league_key)
            if filtered_broad:
                all_markets = filtered_broad
                game_markets = [
                    m
                    for m in all_markets
                    if str(m.get("event_ticker") or m.get("ticker") or "")
                    .upper()
                    .startswith(f"{game_prefix}-")
                ]
                if not game_markets and league_key == "NCAAB":
                    alt_candidates: List[Dict[str, Any]] = []
                    alt_prefix = None
                    date_token_pattern = re.compile(r"\d{2}[A-Z]{3}\d{2}")
                    for m in all_markets:
                        t_upper = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                        if "GAME" in t_upper and ("NCAAB" in t_upper or "NCAA" in t_upper):
                            alt_candidates.append(m)
                            if not alt_prefix:
                                alt_prefix = t_upper.split("-")[0]
                        elif ("NCAAB" in t_upper or "NCAA" in t_upper) and date_token_pattern.search(t_upper):
                            alt_candidates.append(m)
                            if not alt_prefix:
                                alt_prefix = t_upper.split("-")[0]
                    if alt_candidates:
                        game_markets = alt_candidates
                        used_game_prefix = alt_prefix or game_prefix
        futures_noise = [
            m
            for m in all_markets
            if series_base
            and str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith(f"{series_base}-")
            and not str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith(f"{game_prefix}-")
        ]

        ticker_keys = [
            str(m.get("event_ticker") or m.get("ticker") or "").upper()
            for m in all_markets
        ]
        prefix_hits = len(
            [
                k
                for k in ticker_keys
                if any(k.startswith(str(s).upper()) for s in series_targets if s)
            ]
        )
        game_hits = len(game_markets)
        futures_hits = len(futures_noise)

        if game_markets:
            all_markets = game_markets

        self.last_fetch_meta = {
            "league": league_key,
            "status": normalized_status,
            "status_param": bool(self._status_param(normalized_status)),
            "pages": pages,
            "total_markets": len(all_markets),
            "prefix_hits": prefix_hits,
            "prefix": prefix,
            "futures_noise": len(futures_noise) if futures_noise else None,
            "game_hits": game_hits,
            "filtered_to_game_markets": bool(game_hits),
            "series_targets": series_targets,
            "game_prefix_used": used_game_prefix,
        }
        if not game_markets and all_markets:
            self.last_fetch_meta["warning"] = "game_markets_missing_or_futures_only"
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
        if league_key == "NCAAB":
            prefix_filtered = [
                m
                for m, t in zip(markets, ticker_upper)
                if ("NCAAB" in t)
                or ("NCAA" in t and "NCAAF" not in t and "FOOT" not in t)
            ]
        elif league_key == "NCAAF":
            prefix_filtered = [
                m
                for m, t in zip(markets, ticker_upper)
                if ("NCAAF" in t)
                or ("NCAA" in t and "BASK" not in t and "NCAAB" not in t)
            ]
        elif isinstance(prefix, list):
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
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                return dt if dt.tzinfo else pytz.utc.localize(dt)
            except: pass
        return None

    def is_multivariate_bundle(self, market: Dict[str, Any]) -> bool:
        """Identify multi-leg/bundle style markets that are not single-game lines."""

        if not isinstance(market, dict):
            return False

        market_type = str(market.get("type") or "").lower()
        if market_type in {"bundle", "multivariate", "portfolio"}:
            return True

        if market.get("legs") or market.get("leg_markets"):
            return True

        ticker = str(market.get("event_ticker") or market.get("ticker") or "").upper()
        if ticker.startswith("KXMV"):
            return True

        return False

    def split_market_kinds(
        self, markets: List[Dict[str, Any]], league: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        league_key = (league or "").upper()
        prefix = LEAGUE_SERIES_MAP.get(league_key, "")
        game_prefix = league_game_prefix(league_key)
        date_token_pattern = re.compile(r"\d{2}[A-Z]{3}\d{2}")

        single_game: List[Dict[str, Any]] = []
        multivariate: List[Dict[str, Any]] = []
        other: List[Dict[str, Any]] = []

        for m in markets or []:
            if self.is_multivariate_bundle(m):
                multivariate.append(m)
                continue

            t = str(m.get("event_ticker") or m.get("ticker") or "").upper()

            if game_prefix and t.startswith(f"{game_prefix}-"):
                single_game.append(m)
                continue

            if league_key == "NCAAB" and "GAME" in t and ("NCAAB" in t or "NCAA" in t):
                single_game.append(m)
                continue

            if league_key == "NCAAB" and ("NCAAB" in t or "NCAA" in t) and date_token_pattern.search(t):
                single_game.append(m)
                continue

            if isinstance(prefix, list):
                if any(pfx and t.startswith(pfx) for pfx in prefix):
                    single_game.append(m)
                else:
                    other.append(m)
            else:
                if prefix and t.startswith(prefix):
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
        self, markets: List[Dict[str, Any]], date_token: str, league: Optional[str]
    ) -> List[Dict[str, Any]]:
        token_upper = (date_token or "").upper()
        if not token_upper:
            return []
        game_prefix = league_game_prefix(league or "")
        bucket: List[Dict[str, Any]] = []
        for m in markets or []:
            et_upper = str(m.get("event_ticker") or m.get("ticker") or "").upper()
            if et_upper.startswith(f"{game_prefix}-{token_upper}"):
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
        bucket = self._filter_markets_by_date_token(base_markets, date_token, league_key)

        fetch_meta = {
            "league": league_key,
            "date_token": date_token,
            "initial_total": len(base_markets),
            "initial_date_token_count": len(bucket),
        }

        # If bucket empty, broaden with limited pagination and targeted prefix pull
        all_markets = list(base_markets)
        targeted: List[Dict[str, Any]] = []
        targeted_prefix = f"{league_game_prefix(league_key)}-{date_token}"
        extra: List[Dict[str, Any]] = []
        if not bucket:
            try:
                targeted = self.get_markets_paginated(
                    status=status,
                    limit=200,
                    max_pages=5,
                    extra_params={"event_ticker_prefix": targeted_prefix},
                )
            except Exception:
                targeted = []
            try:
                extra = self.get_markets_paginated(
                    status=status, limit=200, max_pages=5
                )
            except Exception:
                extra = []
            all_markets.extend(targeted)
            all_markets.extend(extra)

        dedup: Dict[str, Dict[str, Any]] = {}
        for m in all_markets:
            key = str(m.get("event_ticker") or m.get("ticker") or "")
            if key and key not in dedup:
                dedup[key] = m
        all_markets = list(dedup.values())
        bucket = self._filter_markets_by_date_token(all_markets, date_token, league_key)
        fetch_meta.update(
            {
                "broadened_total": len(all_markets),
                "broadened_date_token_count": len(bucket),
                "targeted_prefix": targeted_prefix if not bucket else None,
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
        prefix_token = f"{league_game_prefix(league_key)}-"
        for m in all_markets:
            et = str(m.get("event_ticker") or m.get("ticker") or "").upper()
            if prefix_token in et:
                try:
                    after = et.split(prefix_token)[1]
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

    def assert_available(self) -> None:
        if not self.api_key or not self.api_secret_pem:
            raise RuntimeError("Kalshi keys missing from secrets.")


def self_test() -> Dict[str, Any]:
    """Ensure status="open" is never sent even when requested explicitly."""

    integ = KalshiIntegrator()
    result: Dict[str, Any] = {
        "configured": bool(integ.api_key and integ.api_secret_pem),
        "request_params": None,
        "status_code": None,
        "contains_status": None,
        "error": None,
    }

    try:
        markets = integ.get_markets_paginated(status="open", limit=5, max_pages=1)
        params = integ.last_request_params or {}
        result.update(
            {
                "request_params": params,
                "status_code": integ.last_status_code,
                "contains_status": "status" in params,
                "market_count": len(markets or []),
            }
        )
        if integ.last_status_code == 400:
            result["error"] = "Bad request when testing open status filter"
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = str(exc)
        result["request_params"] = integ.last_request_params
        result["status_code"] = integ.last_status_code
    return result

def _sanity_check():
    """Manual self-test for deployment verification."""
    print("--- Sanity Check: Kalshi Integrator ---")

    # 1. Check Normalization
    cases = [
        ("Los Angeles Lakers", "LAL", "NBA"),
        ("New York Knicks", "NYK", "NBA"),
        ("Golden State Warriors", "GSW", "NBA"),
    ]
    for team, expected_code, league in cases:
        code = team_code_for_league(league, team)
        status = "PASS" if code == expected_code else f"FAIL (Got {code})"
        print(f"Code Lookup: {team} -> {code} [{status}]")

    # 2. Check Tokenization
    test_team = "Golden State Warriors"
    tokens = _build_team_codes(test_team)
    has_initials = "GSW" in tokens
    has_parts = "GOLDEN" in tokens and "STATE" in tokens
    print(f"Tokenization '{test_team}': {tokens}")
    print(f"Token Check: Initials={has_initials}, Parts={has_parts}")

    print("--- Sanity Check Complete ---")

if __name__ == "__main__":
    if os.environ.get("KALSHI_SELF_TEST"):
        print(json.dumps(self_test(), indent=2))
    else:
        _sanity_check()
