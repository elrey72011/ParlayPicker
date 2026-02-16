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
from functools import lru_cache
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

from app_core.sportsdata import SportsDataNCAABClient
from app_core.team_name_matcher import TeamNameMatcher

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
    "KALSHI_NCAAB_TEAM_CODES",
    "normalize_team_for_kalshi",
]

# Timezone for NBA date buckets (games are bucketed by their US/Eastern date usually, or strict UTC date tokens)
# Kalshi NBA Date Tokens (YYMONDD) often align with UTC, but game times are local.
# We need consistent handling.
NBA_TZ = pytz.timezone("US/Eastern")

# Global counter for debug logging limit
_DEBUG_GAME_LOG_COUNT = 0

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
    "MLB": ["KXMLBGAME", "KXMLBTOTAL", "KXMLBSPREAD", "KXMLB"],
    "NHL": ["KXNHLGAME", "KXNHLTOTAL", "KXNHLSPREAD", "KXNHL"],
    "NCAAF": ["KXNCAAFGAME", "KXNCAAFTOTAL", "KXNCAAFSPREAD", "KXNCAAF"],
    "NCAAB": ["KXNCAAMBGAME", "KXNCAAMB", "KXNCAAMBTOTAL", "KXNCAAMBSPREAD"],
}


@lru_cache(maxsize=4096)
def parse_event_ticker_codes(event_ticker: str) -> Dict[str, str]:
    """
    Extracts away/home codes from Kalshi's event_ticker using team code map matching.
    Examples:
      KXNBAGAME-26JAN09NYKPHX -> away=NYK, home=PHX
      KXNCAAMBGAME-26JAN15MERVMI -> away=MER, home=VMI (variable length)
    """
    if not event_ticker:
        return {}

    parts = event_ticker.split('-')
    if len(parts) < 2:
        return {}

    prefix = parts[0].upper()
    suffix = parts[-1]

    match = re.match(r"^(\d{2}[A-Z]{3}\d{2})([A-Z0-9]+)$", suffix)
    if not match:
        logger.warning(f"Failed to parse event ticker suffix: {suffix} (full: {event_ticker})")
        return {}

    date_token = match.group(1)
    team_block = match.group(2)

    league = None
    if "NBA" in prefix: league = "NBA"
    elif "NFL" in prefix: league = "NFL"
    elif "NHL" in prefix: league = "NHL"
    elif "MLB" in prefix: league = "MLB"
    elif "NCAAF" in prefix: league = "NCAAF"
    elif "NCAAB" in prefix or "NCAA" in prefix: league = "NCAAB"

    away = ""
    home = ""

    if league in ["NCAAB", "NCAAF"]:
        code_map = NCAAB_TEAM_CODE_MAP if league == "NCAAB" else NCAAF_TEAM_CODE_MAP
        all_codes = set(code_map.values())

        # Add values from alias map to known codes to catch resolved aliases (e.g. "DUKE" -> "DUK")
        if league == "NCAAB":
            all_codes.update(NCAAB_CODE_ALIASES.values())
            # FIX: Also add comprehensive team codes (Task: Ensure newly added codes like LCHI are recognized)
            all_codes.update(KALSHI_NCAAB_TEAM_CODES.values())

        best_split = None
        best_score = 0

        # LOGGING: Show what we are parsing (Issue #3)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Parsing NCAAB ticker: {event_ticker} (block={team_block})")

        # Try all possible split points
        # For 7-char strings (common in NCAAB), this explicitly iterates through:
        # i=3 (3+4 split) AND i=4 (4+3 split)
        # This allows matching both BSUUNLV (3+4) and UNLVBSU (4+3) as long as codes are in the map
        min_len = max(2, len(team_block) - 5)
        for i in range(min_len, len(team_block) - 1):
            potential_away = team_block[:i]
            potential_home = team_block[i:]

            # Try direct resolution first
            away_resolved = resolve_team_code(potential_away, league)
            home_resolved = resolve_team_code(potential_home, league)

            # Check validity (Original Direction)
            away_match = away_resolved in all_codes or potential_away in all_codes
            home_match = home_resolved in all_codes or potential_home in all_codes

            # Score this split attempt
            score = 0
            if away_match: score += 1
            if home_match: score += 1

            if away_match and home_match:
                away = away_resolved if away_resolved in all_codes else potential_away
                home = home_resolved if home_resolved in all_codes else potential_home
                logger.debug(f"NCAAB ticker parse: {event_ticker} -> away={away}, home={home} (perfect match at split {i})")
                best_split = (away, home)
                best_score = 2
                break

            elif score > best_score:
                best_score = score
                # Store the resolved versions if they matched, else raw
                a_cand = away_resolved if away_match else potential_away
                h_cand = home_resolved if home_match else potential_home
                best_split = (a_cand, h_cand)

        if not away and not home:
            # 1. Try Partial Match via Secondary API (Cross-Reference)
            # This is slow, so only do it if we failed to match both sides
            xref_result = cross_reference_unmapped_ticker(league, date_token, team_block)
            if xref_result:
                away = xref_result["away"]
                home = xref_result["home"]
                logger.info(f"API Cross-Ref Resolved: {team_block} -> {away} @ {home}")

            # 2. Fallback to Best Partial Split from map logic
            elif best_split:
                away, home = best_split
                logger.debug(f"NCAAB ticker fallback: {event_ticker} -> best split {away}/{home} (score={best_score})")

            # 3. Final Fallback: Heuristic Blind Bisection
            else:
                length = len(team_block)
                # Intelligent length-based heuristics for NCAAB
                if length == 6:
                    # Most common: 3+3 (e.g. MERVMI)
                    away = team_block[:3]
                    home = team_block[3:]
                elif length == 7:
                    # Ambiguous: could be 3+4 or 4+3
                    # Check if suffix looks like common suffix (State, Tech, etc.)
                    if team_block.endswith("ST") or team_block.endswith("TE"):
                        # If ends with STATE (5 chars), implies 2+5? No, total 7.
                        # If STATE is suffix, length 7 means XXSTATE (2+5).
                        # If ST is suffix, length 7 means XXXXXST (5+2) or XXXXST (4+2)?
                        # Wait, logic was:
                        # away = team_block[:5] if team_block.endswith("STATE") else team_block[:3]
                        # If ends with STATE: away = first 2 chars. home = STATE (5).
                        # If ends with ST: away = first 5 chars? No, wait.

                        # Let's simplify. Standard college code is 3 letters.
                        # If 7 chars, likely 3+4 or 4+3.
                        # If we have suffix ST (e.g. OSUST), it's likely OSU (3) + ST (2)? No length is 7.
                        # OSUOKST (7) -> OSU (3) + OKST (4).

                        # Default to 3+4 which seems most common for college (3-letter code + 4-letter code)
                        away = team_block[:3]
                        home = team_block[3:]
                    else:
                        away = team_block[:3]
                        home = team_block[3:]
                elif length == 8:
                    # 4+4 or 3+5/5+3
                    away = team_block[:4]
                    home = team_block[4:]
                elif length >= 4:
                    mid = length // 2
                    away = team_block[:mid]
                    home = team_block[mid:]
    else:
        # For pro leagues (NBA, NFL, NHL, MLB), use fixed-length logic
        length = len(team_block)

        if length == 6:
            # 3+3 (most common for pro leagues)
            away = team_block[:3]
            home = team_block[3:]
        elif length == 7:
            # 7 characters: Try 4+3 or 3+4
            # Default to 4+3 for NFL (most common)
            away = team_block[:4]
            home = team_block[4:]
        elif length == 8:
            # 4+4
            away = team_block[:4]
            home = team_block[4:]
        elif length % 2 == 0 and length >= 4:
            # Variable length handling (split in half if even)
            mid = length // 2
            away = team_block[:mid]
            home = team_block[mid:]
        else:
            # Fallback for odd lengths: try to split smartly
            # Prefer taking last 3 as home if length >= 6
            if length >= 6:
                home = team_block[-3:]
                away = team_block[:-3]
            elif length >= 4:
                # For shorter odd lengths, try 2+3 or 3+2
                home = team_block[-3:]
                away = team_block[:-3]
            else:
                # Very short, just split
                home = team_block[-min(3, length):]
                away = team_block[:-min(3, length)]

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
    # Convert to uppercase, replace non-alphanumeric with space, collapse multiple spaces
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", str(name or "").upper())
    # Collapse multiple spaces into one and strip
    return re.sub(r"\s+", " ", cleaned).strip()


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
    "UTAH": "UTA",
    "UTAH MAMMOTH": "UTA",
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
    "FLORIDA STATE": "FSU",
}

NCAAB_TEAM_CODE_MAP: Dict[str, str] = {
    "DUKE": "DUK", "NORTH CAROLINA": "UNC", "KANSAS": "KAN", "KENTUCKY": "KEN",
    "KANSAS JAYHAWKS": "KAN",
    "GONZAGA": "GON", "BAYLOR": "BAY", "ARIZONA": "ARI", "UCLA": "UCL",
    "ARIZONA WILDCATS": "ARI",
    "HOUSTON": "HOU", "PURDUE": "PUR", "UCONN": "CON", "CONNECTICUT": "CON",
    "VILLANOVA": "VIL", "MICHIGAN STATE": "MSU", "TENNESSEE": "TEN", "ALABAMA": "ALA",
    "AUBURN": "AUB", "TEXAS": "TEX", "VIRGINIA": "UVA", "ILLINOIS": "ILL",
    "ARKANSAS": "ARK", "INDIANA": "IND", "MICHIGAN": "MIC", "OHIO STATE": "OSU",
    "FLORIDA": "FLO", "TEXAS TECH": "TTU", "WISCONSIN": "WIS", "MARYLAND": "MAR",
    "IOWA": "IOW", "XAVIER": "XAV", "CREIGHTON": "CRE", "MARQUETTE": "MAR",
    "PROVIDENCE": "PRO", "SETON HALL": "SET", "ST. JOHN'S": "STJ", "ST JOHNS": "STJ",
    "ST. JOHN'S RED STORM": "STJ", "SAINT JOHN'S": "STJ",
    "GEORGETOWN": "GEO", "BUTLER": "BUT", "DEPAUL": "DEP", "MEMPHIS": "MEM",
    "CINCINNATI": "CIN", "SMU": "SMU", "WICHITA STATE": "WIC", "TEMPLE": "TEM",
    "TULANE": "TUL", "USF": "USF", "UCF": "UCF", "ECU": "ECU", "TULSA": "TUL",
    "DAYTON": "DAY", "VCU": "VCU", "SAINT LOUIS": "SLU", "ST. BONAVENTURE": "SBU",
    "RICHMOND": "RIC", "DAVIDSON": "DAV", "LOYOLA CHICAGO": "LOY", "SAN DIEGO STATE": "SDS",
    "SAN DIEGO ST": "SDS", "NEVADA": "NEV", "UTAH STATE": "USU", "BOISE STATE UNIVERSITY": "BSU", "BOISE STATE": "BSU",
    "BOISE ST": "BSU", "UNLV": "UNLV", "NEW MEXICO": "UNM", "COLORADO STATE": "CSU",
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
    "WASHINGTON STATE": "WSU", "WASHINGTON ST": "WSU", "WASHINGTON ST COUGARS": "WSU",
    "WASHINGTON STATE COUGARS": "WSU",
    "COLORADO": "COL", "UTAH": "UTAH", "ARIZONA STATE": "ASU",
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
    "MERCER BEARS": "MER", # Added per user request
    "MERCER": "MER",
    "VMI": "VMI",  # Virginia Military Institute - 3-letter code
    "VIRGINIA MILITARY INSTITUTE": "VMI",
    "VIRGINIA MILITARY": "VMI",
    # Fix for Issue #2: Missing NCAAB Aliases
    "FORT WAYNE": "PFW",
    "PURDUE FORT WAYNE": "PFW",
    "IPFW": "PFW",
    "MILWAUKEE": "MILW",
    "GREEN BAY": "GB",
    "IUPUI": "IUIN",
    "UIC": "UIC",
    "NORTHERN KENTUCKY": "NKU",
    "WRIGHT STATE": "WRI",
    "YOUNGSTOWN STATE": "YSU",
    "ROBERT MORRIS": "RMU",
    "DETROIT MERCY": "DET",
    "OAKLAND": "OAK",
    "CLEVELAND STATE": "CSU",
    # v108 Updates
    "CHARLOTTE": "CHAR",
    "LOYOLA MD": "LMD",
    "LOYOLA MARYLAND": "LMD",
    "OMAHA": "NEOM",
    "FLORIDA ATLANTIC": "FAU",
    "TULANE": "TULN",
    "UAB": "UAB",
    "MARIST": "MRST",
    "ILLINOIS STATE": "ILST",
    "CAMPBELL": "CAMP",
    "OREGON STATE": "ORST",
    "SEATTLE": "SEA",
    "CHARLESTON": "COFC",
    # --- SWAC (Southwestern Athletic Conference) ---
    "ALABAMA A&M": "AAMU", "ALABAMA A M": "AAMU", "ALABAMA A&M BULLDOGS": "AAMU",
    "GRAMBLING": "GRAM", "GRAMBLING ST": "GRAM", "GRAMBLING STATE": "GRAM",
    "GRAMBLING TIGERS": "GRAM", "GRAMBLING ST TIGERS": "GRAM",
    "ALABAMA ST": "ALST", "ALABAMA STATE": "ALST", "ALABAMA ST HORNETS": "ALST",
    "SOUTHERN JAGUARS": "SOU", "SOUTHERN UNIVERSITY": "SOU",
    "TEXAS SOUTHERN": "TXSO", "TEXAS SOUTHERN TIGERS": "TXSO",
    "BETHUNE-COOKMAN": "BCU", "BETHUNE COOKMAN": "BCU", "BETHUNE COOKMAN WILDCATS": "BCU",
    "PRAIRIE VIEW": "PVAM", "PRAIRIE VIEW A&M": "PVAM", "PRAIRIE VIEW A M": "PVAM",
    "PRAIRIE VIEW A&M PANTHERS": "PVAM",
    "FLORIDA A&M": "FAMU", "FLORIDA A M": "FAMU", "FLORIDA A&M RATTLERS": "FAMU",
    "JACKSON ST": "JKST", "JACKSON STATE": "JKST", "JACKSON ST TIGERS": "JKST",
    "ALCORN ST": "ALC", "ALCORN STATE": "ALC", "ALCORN ST BRAVES": "ALC",
    "MISS VALLEY ST": "MVSU", "MISSISSIPPI VALLEY STATE": "MVSU",
    "MISSISSIPPI VALLEY ST": "MVSU",
    "ARKANSAS-PINE BLUFF": "UAPB", "ARKANSAS PINE BLUFF": "UAPB",
    # --- Southland Conference ---
    "NORTHWESTERN ST": "NWST", "NORTHWESTERN STATE": "NWST",
    "NORTHWESTERN ST DEMONS": "NWST",
    "LAMAR": "LAM", "LAMAR CARDINALS": "LAM",
    "NICHOLLS": "NICH", "NICHOLLS ST": "NICH", "NICHOLLS STATE": "NICH",
    "NICHOLLS ST COLONELS": "NICH",
    "STEPHEN F. AUSTIN": "SFA", "STEPHEN F AUSTIN": "SFA",
    "STEPHEN F AUSTIN LUMBERJACKS": "SFA", "SFA LUMBERJACKS": "SFA",
    "CENTRAL ARKANSAS": "CARK", "CENTRAL ARKANSAS BEARS": "CARK",
    "NORTH ALABAMA": "UNA", "NORTH ALABAMA LIONS": "UNA",
    "EAST TEXAS A&M": "ETAM", "EAST TEXAS A M": "ETAM",
    "EAST TEXAS A&M LIONS": "ETAM",
    "NEW ORLEANS PRIVATEERS": "UNO",
    "UT RIO GRANDE VALLEY": "UTRGV", "UTRGV": "UTRGV", "UTRGV VAQUEROS": "UTRGV",
    # --- Missouri Valley Conference ---
    "SOUTHERN ILLINOIS": "SIU", "SOUTHERN ILLINOIS SALUKIS": "SIU",
    "INDIANA ST": "INST", "INDIANA STATE": "INST", "INDIANA ST SYCAMORES": "INST",
    "ILLINOIS ST": "ILST", "ILLINOIS STATE": "ILST", "ILLINOIS ST REDBIRDS": "ILST",
    "EVANSVILLE": "EVAN", "EVANSVILLE ACES": "EVAN", "EVANSVILLE PURPLE ACES": "EVAN",
    "BELMONT": "BEL", "BELMONT BRUINS": "BEL",
    "BRADLEY": "BRAD", "BRADLEY BRAVES": "BRAD",
    "VALPARAISO": "VAL", "VALPARAISO BEACONS": "VAL",
    "DRAKE": "DRKE", "DRAKE BULLDOGS": "DRKE",
    "MISSOURI ST": "MOST", "MISSOURI STATE": "MOST",
    "NORTHERN IOWA": "UNI", "NORTHERN IOWA PANTHERS": "UNI",
    "MURRAY ST": "MURS", "MURRAY STATE": "MURS", "MURRAY ST RACERS": "MURS",
    # --- Other missing teams ---
    "MONMOUTH": "MON", "MONMOUTH HAWKS": "MON",
    "TOWSON": "TOW", "TOWSON TIGERS": "TOW",
    "SAN FRANSISCO": "USF", "SAN FRANCISCO": "USF", # Fix for typo
    "TEXAS A&M-CC": "AMCC", "TEXAS A&M CORPUS CHRISTI": "AMCC",
    "TEXAS A M CC": "AMCC", "TEXAS A M CORPUS CHRISTI": "AMCC",
    "ST FRANCIS PA": "SFP", "ST. FRANCIS PA": "SFP", "SAINT FRANCIS PA": "SFP",
    "ST FRANCIS": "SFP",
    "CHICAGO ST": "CHST", "CHICAGO STATE": "CHST", "CHICAGO ST COUGARS": "CHST",
    "SOUTHEAST MISSOURI": "SEMO", "SOUTHEAST MISSOURI ST": "SEMO",
    "SOUTHEAST MISSOURI STATE": "SEMO",
    "UT MARTIN": "UTM", "TENNESSEE MARTIN": "UTM",
    "SIU EDWARDSVILLE": "SIUE", "SIUE": "SIUE",
    "EASTERN ILLINOIS": "EIU", "EASTERN ILLINOIS PANTHERS": "EIU",
    "MOREHEAD ST": "MORE", "MOREHEAD STATE": "MORE",
    "TENNESSEE ST": "TNST", "TENNESSEE STATE": "TNST",
    "TENNESSEE TECH": "TNTH",
    "LINDENWOOD": "LIND",
    "HAMPTON": "HAMP",
    "NORTH CAROLINA A T": "NCAT",
    "LITTLE ROCK": "UALR", "ARKANSAS LITTLE ROCK": "UALR",
    "SOUTHERN MISS": "USM", "SOUTHERN MISSISSIPPI": "USM",
    # --- Feb 9 missing teams (discovered via Kalshi API) ---
    "BUCKNELL": "BUCK", "BUCKNELL BISON": "BUCK",
    "CHARLESTON": "COFC", "COLLEGE OF CHARLESTON": "COFC", "CHARLESTON COUGARS": "COFC",
    "DELAWARE ST": "DSU", "DELAWARE STATE": "DSU", "DELAWARE ST HORNETS": "DSU",
    "HOUSTON CHRISTIAN": "HCU", "HOUSTON CHRISTIAN HUSKIES": "HCU",
    "HOWARD": "HOW", "HOWARD BISON": "HOW",
    "INCARNATE WORD": "IW", "INCARNATE WORD CARDINALS": "IW",
    "MCNEESE": "MCNS", "MCNEESE ST": "MCNS", "MCNEESE STATE": "MCNS",
    "MCNEESE COWBOYS": "MCNS",
    "NAVY": "NAVY", "NAVY MIDSHIPMEN": "NAVY",
    "NORTH CAROLINA CENTRAL": "NCCU", "NC CENTRAL": "NCCU", "NCCU EAGLES": "NCCU",
    "SOUTHEASTERN LOUISIANA": "SELA", "SE LOUISIANA": "SELA",
    "SOUTHEASTERN LOUISIANA LIONS": "SELA",
    "UNC WILMINGTON": "UNCW", "UNC WILMINGTON SEAHAWKS": "UNCW",
    "YALE": "YALE", "YALE BULLDOGS": "YALE",
    # --- Feb 10 audit additions ---
    "SAN JOSE ST": "SJSU", "SAN JOSE STATE": "SJSU", "SAN JOSE STATE SPARTANS": "SJSU",
    "SJSU": "SJSU", "SAN JOSE": "SJSU",
    "COLORADO ST": "CSU", "COLORADO STATE RAMS": "CSU", "COLORADO ST RAMS": "CSU",
    "DUKE BLUE DEVILS": "DUK",
    "NOTRE DAME FIGHTING IRISH": "UND",
    "VIRGINIA CAVALIERS": "UVA",
    "OKLAHOMA ST": "OSU", "OKLAHOMA ST COWBOYS": "OSU", "OKLAHOMA STATE COWBOYS": "OSU",
    "HOUSTON COUGARS": "HOU",
    "ARIZONA ST": "ASU", "ARIZONA STATE SUN DEVILS": "ASU", "ARIZONA ST SUN DEVILS": "ASU",
    # --- v96 team code fixes (Kalshi code corrections) ---
    "UTAH UTES": "UTAH",
    "FLORIDA ST SEMINOLES": "FSU",
    "AIR FORCE": "AFA", "AIR FORCE FALCONS": "AFA",
    "FRESNO ST": "FRES", "FRESNO STATE": "FRES", "FRESNO ST BULLDOGS": "FRES",
    "FRESNO STATE BULLDOGS": "FRES", "FRESNO": "FRES",
    "WESTERN ILLINOIS": "WIU", "WESTERN ILLINOIS LEATHERNECKS": "WIU",
    "SAINT JOSEPH'S": "JOES", "ST JOSEPH'S": "JOES", "ST. JOSEPH'S": "JOES",
    "SAINT JOSEPH'S HAWKS": "JOES", "ST JOSEPH'S HAWKS": "JOES",
    "GEORGE WASHINGTON": "GW", "GW": "GW", "GW REVOLUTIONARIES": "GW",
    "GEORGE WASHINGTON REVOLUTIONARIES": "GW",
    "RHODE ISLAND": "URI", "RHODE ISLAND RAMS": "URI",
    "GEORGE MASON": "GMU", "GEORGE MASON PATRIOTS": "GMU",
    "MILWAUKEE PANTHERS": "MILW",
    "IUPUI JAGUARS": "IUIN", "IU INDIANAPOLIS": "IUIN",
    "IU INDIANAPOLIS JAGUARS": "IUIN",
    "SAN JOSÉ ST": "SJSU", "SAN JOSÉ ST SPARTANS": "SJSU",
    # Task 1: Lexical Tokenization Void Fixes
    "FLORIDA INT'L": "FIU", "FLORIDA INTERNATIONAL": "FIU", "FIU PANTHERS": "FIU",
    "ST. THOMAS (MN)": "UST", "ST THOMAS MN": "UST", "ST. THOMAS": "UST", "ST THOMAS": "UST",
    "ST. FRANCIS (PA)": "SFP", "ST FRANCIS PA": "SFP", "SAINT FRANCIS (PA)": "SFP",
    "CHARLESTON SO": "CSO", "CHARLESTON SOUTHERN": "CSO", "CHARLESTON SOUTHERN BUCCANEERS": "CSO",
    "GARDNER-WEBB": "GW", "GARDNER WEBB": "GW", "GARDNER WEBB BULLDOGS": "GW",
    "HIGH POINT": "HPU", "HIGH POINT PANTHERS": "HPU",
    "PRESBYTERIAN": "PRE", "PRESBYTERIAN BLUE HOSE": "PRE",
    "RADFORD": "RAD", "RADFORD HIGHLANDERS": "RAD",
    "UNC ASHEVILLE": "UNCA", "UNC-ASHEVILLE": "UNCA", "UNC ASHEVILLE BULLDOGS": "UNCA",
    "USC UPSTATE": "USCU", "SOUTH CAROLINA UPSTATE": "USCU", "USC UPSTATE SPARTANS": "USCU",
    "WINTHROP": "WIN", "WINTHROP EAGLES": "WIN",
    # --- WAC (Western Athletic Conference) ---
    "UTAH VALLEY": "UVU", "UTAH VALLEY WOLVERINES": "UVU",
    "UTAH TECH": "UTT", "UTAH TECH TRAILBLAZERS": "UTT",
    "TARLETON ST": "TAR", "TARLETON STATE": "TAR", "TARLETON": "TAR", "TARLETON ST TEXANS": "TAR",
    "GRAND CANYON": "GCU", "GRAND CANYON ANTELOPES": "GCU",
    "CAL BAPTIST": "CBU", "CALIFORNIA BAPTIST": "CBU", "CAL BAPTIST LANCERS": "CBU",
    "SEATTLE U": "SEA", "SEATTLE UNIVERSITY": "SEA", "SEATTLE REDHAWKS": "SEA",
    "ABILENE CHRISTIAN": "ACU", "ABILENE CHRISTIAN WILDCATS": "ACU",
    "SOUTHERN UTAH": "SUU", "SOUTHERN UTAH THUNDERBIRDS": "SUU",
    "UT ARLINGTON": "UTA", "TEXAS ARLINGTON": "UTA", "UT ARLINGTON MAVERICKS": "UTA",
    # Fix 4: Add Missing Team Code Aliases
    "HOLY CROSS": "HC", "HOLY CROSS CRUSADERS": "HC",
    "LOYOLA MD": "LMD", "LOYOLA MARYLAND": "LMD", "LOYOLA (MD)": "LMD",
    "UTSA": "UTSA", "UTSA ROADRUNNERS": "UTSA",
    "CHARLOTTE": "CHAR", "CHARLOTTE 49ERS": "CHAR", "CHAR": "CHAR",
    "OMAHA": "NEOM", "OMAHA MAVERICKS": "NEOM", "NEBRASKA OMAHA": "NEOM",
    "DENVER": "DEN", "DENVER PIONEERS": "DEN",
    "FLORIDA ATLANTIC": "FAU", "FLORIDA ATLANTIC OWLS": "FAU", "FAU": "FAU",
    "TULANE": "TULN", "TULANE GREEN WAVE": "TULN", "TULN": "TULN",
    "UAB": "UAB", "UAB BLAZERS": "UAB",
    "MARIST": "MRST", "MARIST RED FOXES": "MRST", "MRST": "MRST",
    "ILLINOIS STATE": "ILST", "ILLINOIS ST": "ILST", "ILST": "ILST",
    "CAMPBELL": "CAMP", "CAMPBELL FIGHTING CAMELS": "CAMP", "CAMP": "CAMP",
    "OREGON STATE": "ORST", "OREGON ST": "ORST", "ORST": "ORST", "OSU": "ORST",
    "SEATTLE": "SEA", "SEATTLE U": "SEA", "SEATTLE REDHAWKS": "SEA",
    "CHARLESTON": "COFC", "COLLEGE OF CHARLESTON": "COFC", "COFC": "COFC",
    # Issue #2: Missing NCAAB Aliases
    "LCHI": "LCHI", "LOYOLA CHICAGO": "LCHI",
    "IUIN": "IUIN", "IU INDIANAPOLIS": "IUIN",
    "MILW": "MILW", "MILWAUKEE": "MILW",
    "PFW": "PFW", "PURDUE FORT WAYNE": "PFW",
    "CHAR": "CHAR", "CHARLOTTE": "CHAR",
    "NEOM": "NEOM", "NEBRASKA OMAHA": "NEOM",
    "FAU": "FAU", "FLORIDA ATLANTIC": "FAU",
    "TULN": "TULN", "TULANE": "TULN",
    "UAB": "UAB",
    "MRST": "MRST", "MARIST": "MRST",
    "ILST": "ILST", "ILLINOIS STATE": "ILST",
    "CAMP": "CAMP", "CAMPBELL": "CAMP",
    "ORST": "ORST", "OREGON STATE": "ORST",
    "SEA": "SEA", "SEATTLE": "SEA",
    "WAGNER": "WAG", "WAGNER SEAHAWKS": "WAG",
    "LIU": "LIU", "LIU SHARKS": "LIU", "LONG ISLAND": "LIU", "LONG ISLAND UNIVERSITY": "LIU",
    "SOUTH ALABAMA": "SOAL", "SOUTH ALABAMA JAGUARS": "SOAL",
    "MARSHALL": "MARS", "MARSHALL THUNDERING HERD": "MARS",
}

# ADD THIS COMPREHENSIVE NCAAB TEAM NAME → KALSHI CODE MAPPING
KALSHI_NCAAB_TEAM_CODES = {
    "Abilene Christian": "AC",
    "Air Force": "AFA",
    "Akron": "AKR",
    "Arizona": "ARIZ",
    "Arizona State": "ASU",
    "Arkansas": "ARK",
    "Auburn": "AUB",
    "Bellarmine": "BELL",
    "Boise State": "BSU",
    "Brown": "BRWN",
    "Brown Bears": "BRWN",
    "Bucknell": "BUCK",
    "Butler": "BUT",
    "Cal State Fullerton": "CSF",
    "California": "CAL",
    "California Golden Bears": "CAL",
    "Canisius": "CAN",
    "Canisius Golden Griffins": "CAN",
    "Central Michigan": "CMU",
    "Cincinnati": "CIN",
    "Cincinnati Bearcats": "CIN",
    "Clemson": "CLEM",
    "Columbia": "CLMB",
    "Columbia Lions": "CLMB",
    "Cornell": "CORN",
    "Cornell Big Red": "CORN",
    "Dartmouth": "DART",
    "Dartmouth Big Green": "DART",
    "DePaul": "DEP",
    "Duke": "DUKE",
    "East Tennessee St.": "ETSU",
    "Florida": "FLA",
    "Fresno State": "FRES",
    "GW": "GW",
    "GW Revolutionaries": "GW",
    "George Mason": "GMU",
    "George Mason Patriots": "GMU",
    "Georgia": "UGA",
    "Gonzaga": "GONZ",
    "Hampton": "HAMP",
    "Hampton Pirates": "HAMP",
    "Harvard": "HARV",
    "Harvard Crimson": "HARV",
    "Houston": "HOU",
    "Idaho": "IDHO",
    "Illinois": "ILL",
    "Incarnate Word": "IW",
    "Indiana State": "INST",
    "Iona": "IONA",
    "Iona Gaels": "IONA",
    "Jacksonville State": "JVST",
    "Kansas City": "UMKC",
    "Kansas City Roos": "UMKC",
    "Kansas St": "KSU",
    "Kansas State Wildcats": "KSU",
    "Kentucky": "UK",
    "Liberty": "LIB",
    "Long Beach State": "LBSU",
    "Louisville": "LOU",
    "Loyola (Chi)": "LCHI",
    "Loyola Chi Ramblers": "LCHI",
    "Loyola Chicago": "LCHI",
    "Loyola (Chicago)": "LCHI",
    "Manhattan": "MAN",
    "Manhattan Jaspers": "MAN",
    "Massachusetts": "MASS",
    "Massachusetts Minutemen": "MASS",
    "Monmouth": "MON",
    "Monmouth Hawks": "MON",
    "Mercer": "MER",
    "Miami (OH)": "MOH",
    "Miami OH RedHawks": "MOH",
    "Michigan": "MICH",
    "Michigan State": "MSU",
    "Milwaukee": "MILW",
    "Middle Tennessee": "MTU",
    "Mississippi (Ole Miss)": "MISS",
    "Morehead State": "MORE",
    "Mt. St. Mary's": "MSM",
    "Mt St Marys": "MSM",
    "Nevada": "NEV",
    "Niagara": "NIAG",
    "Niagara Purple Eagles": "NIAG",
    "North Carolina A&T": "NCAT",
    "North Carolina A T": "NCAT",
    "North Carolina AT Aggies": "NCAT",
    "North Texas": "UNT",
    "Northern Iowa": "UNI",
    "Ohio": "OHIO",
    "Ohio Bobcats": "OHIO",
    "Oklahoma": "OKLA",
    "Ole Miss": "MISS",
    "Oral Roberts": "ORU",
    "Oral Roberts Golden Eagles": "ORU",
    "Pennsylvania": "PENN",
    "Pennsylvania Quakers": "PENN",
    "Portland": "PORT",
    "Portland Pilots": "PORT",
    "Purdue Fort Wayne": "PFW",
    "Fort Wayne": "PFW",
    "Princeton": "PRIN",
    "Princeton Tigers": "PRIN",
    "Providence": "PROV",
    "Quinnipiac": "QUIN",
    "Quinnipiac Bobcats": "QUIN",
    "Rice": "RICE",
    "Richmond": "RICH",
    "Rider": "RID",
    "Rider Broncs": "RID",
    "Sacred Heart": "SHU",
    "Sacred Heart Pioneers": "SHU",
    "Saint Louis": "SLU",
    "Saint Louis Billikens": "SLU",
    "Saint Peter's": "SPC",
    "Saint Peter's Peacocks": "SPC",
    "Saint Peters": "SPC",
    "San Diego": "USD",
    "Seton Hall": "SET",
    "Seton Hall Pirates": "SET",
    "Siena": "SIE",
    "Siena Saints": "SIE",
    "South Carolina St.": "SCUS",
    "South Florida": "USF",
    "St. Bonaventure": "SBON",
    "St. John's": "SJU",
    "Syracuse": "SYR",
    "Syracuse Orange": "SYR",
    "Temple": "TEM",
    "Tennessee Tech": "TNTC",
    "Texas": "TEX",
    "Texas Longhorns": "TEX",
    "Texas Tech": "TTU",
    "Towson": "TOW",
    "Towson Tigers": "TOW",
    "Tulsa": "TLSA",
    "UNCG": "UNCG",
    "UNLV": "UNLV",
    "UT Arlington": "UTA",
    "UTEP": "UTEP",
    "UTRGV": "UTRGV",
    "Utah Tech": "UTU",
    "Utah Valley": "UVU",
    "VCU": "VCU",
    "Villanova": "VILL",
    "Virginia": "UVA",
    "Wake Forest": "WAKE",
    "Washington": "WASH",
    "Weber State": "WEB",
    "Wisconsin": "WIS",
    "Wofford": "WOF",
    "Wyoming": "WYO",
    "Xavier": "XAV",
    "Yale": "YALE",
    "Yale Bulldogs": "YALE",
    # New Mappings for v108 (Feb 2026 Fixes)
    "Campbell": "CAMP",
    "Charleston": "COFC",
    "College of Charleston": "COFC",
    "Charlotte": "CHAR",
    "Florida Atlantic": "FAU",
    "Holy Cross": "HC",
    "Illinois State": "ILST",
    "IU Indianapolis": "IUIN",
    "Loyola (MD)": "LMD",
    "Loyola Maryland": "LMD",
    "Marist": "MRST",
    "Omaha": "NEOM",
    "Nebraska Omaha": "NEOM",
    "Oregon": "ORE",
    "Oregon State": "ORST",
    "Purdue Fort Wayne": "PFW",
    "Seattle": "SEA",
    "Seattle U": "SEA",
    "Seattle University": "SEA",
    "Tulane": "TULN",
    "UAB": "UAB",
    "UTSA": "UTSA",
    # Feb 15, 2026 Game Overrides
    "IUPUI Jaguars": "IUIN",
    "Fort Wayne Mastodons": "PFW",
    "Illinois St Redbirds": "ILST",
    "UIC Flames": "UIC",
    "Wagner": "WAG",
    "Wagner Seahawks": "WAG",
    "LIU": "LIU",
    "LIU Sharks": "LIU",
    "Long Island": "LIU",
    "South Alabama": "SOAL",
    "South Alabama Jaguars": "SOAL",
    "Marshall": "MARS",
    "Marshall Thundering Herd": "MARS",
}

def normalize_team_for_kalshi(team_name: str) -> str:
    """Convert full team name to Kalshi 4-letter code with enhanced normalization"""
    # Clean the name first
    team_clean = team_name.strip()

    # 1. Direct lookup
    if team_clean in KALSHI_NCAAB_TEAM_CODES:
        return KALSHI_NCAAB_TEAM_CODES[team_clean]

    # 2. Try removing common suffixes/noise words (University, State, etc.)
    # User Request: Strip "University", "State", and plural mascots.

    # Try removing "University"
    cleaned_uni = team_clean.replace("University", "").replace("Univ", "").strip()
    if cleaned_uni in KALSHI_NCAAB_TEAM_CODES:
        return KALSHI_NCAAB_TEAM_CODES[cleaned_uni]

    parts = team_clean.split()

    # Try removing last word (likely mascot)
    if len(parts) > 1:
        without_last = " ".join(parts[:-1])
        if without_last in KALSHI_NCAAB_TEAM_CODES:
            return KALSHI_NCAAB_TEAM_CODES[without_last]

        # Try removing "State" if it was part of the name but not in map (risky, but requested)
        without_state = without_last.replace("State", "").strip()
        if without_state in KALSHI_NCAAB_TEAM_CODES:
             return KALSHI_NCAAB_TEAM_CODES[without_state]

    # Try stripping "State" from the full name
    without_state_full = team_clean.replace("State", "").strip()
    if without_state_full in KALSHI_NCAAB_TEAM_CODES:
        return KALSHI_NCAAB_TEAM_CODES[without_state_full]

    # Try base name (first word)
    if parts:
        base_name = parts[0]
        if base_name in KALSHI_NCAAB_TEAM_CODES:
            return KALSHI_NCAAB_TEAM_CODES[base_name]
        return base_name[:4].upper()

    return "UNK"
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
    "BUTLER": "BUT", # Add full name just in case
    "BUT": "BUT",    # Explicit keep
    "SETO": "SET",
    "SETON": "SET",  # 5-char token
    "HALL": "SET",   # Common split artifact
    "SHU": "SET",    # Seton Hall University
    "GEOR": "GEO",
    "DEPA": "DEP",
    # Kalshi ticker variants discovered from Feb 9 events
    "ALCN": "ALC",     # Alcorn St. (Kalshi uses ALCN, we use ALC)
    "ARPB": "UAPB",    # Arkansas-Pine Bluff (Kalshi uses ARPB, we use UAPB)
    "COOK": "BCU",     # Bethune-Cookman (Kalshi uses COOK, we use BCU)
    "KU": "KAN",       # Kansas (Kalshi uses KU, we use KAN)
    "MURR": "MURS",    # Murray St. (Kalshi uses MURR, we use MURS)
    "PV": "PVAM",      # Prairie View A&M (Kalshi uses PV, we use PVAM)
    "SJU": "STJ",      # St. John's (Kalshi uses SJU, we use STJ)
    "VALP": "VAL",     # Valparaiso (Kalshi uses VALP, we use VAL)
    # Proactive aliases for common Kalshi variants (Feb 10 audit)
    "DUKE": "DUK",     # Kalshi likely uses DUKE, we use DUK
    "ND": "UND",       # Notre Dame (Kalshi likely uses ND, we use UND)
    "NDAME": "UND",    # Notre Dame alternate
    "OKST": "OSU",     # Oklahoma St (Kalshi likely uses OKST, we use OSU)
    "CLEM": "CLE",     # Clemson
    "WASH": "WAS",     # Washington
    "IOWA": "IOW",     # Iowa
    "OHIO": "OHIO",    # Ohio Bobcats (Explicit keep)
    "UTAH": "UTAH",    # Utah Utes (Explicit keep)
    # --- v96 reverse lookup aliases (old system codes → correct Kalshi codes) ---
    "VIR": "UVA",      # Virginia: was VIR, Kalshi uses UVA
    # "UTA": "UTAH",   # REMOVED: UTA is UT Arlington. Utah Utes is UTAH.
    "SAN": "SJSU",     # San Jose St: heuristic generated SAN, Kalshi uses SJSU
    "AIR": "AFA",      # Air Force: heuristic generated AIR, Kalshi uses AFA
    "FRE": "FRES",     # Fresno St: heuristic generated FRE, Kalshi uses FRES
    "SAI": "JOES",     # Saint Joseph's: heuristic generated SAI, Kalshi uses JOES
    "GWR": "GW",       # GW Revolutionaries: heuristic generated GWR, Kalshi uses GW
    "RHO": "URI",      # Rhode Island: heuristic generated RHO, Kalshi uses URI
    "MIL": "MILW",     # Milwaukee: was MIL, Kalshi uses MILW
    "IUP": "IUIN",     # IUPUI: was IUP, Kalshi uses IUIN
    "COLM": "CLMB",    # Columbia: Kalshi variant fallback
    "MANH": "MAN",     # Manhattan: internal code update
    "RIDR": "RID",     # Rider: internal code update
    "SPU": "SPC",      # St. Peter's: internal code update
    "MSM": "MSM",      # Mt St Mary's
    "MTST": "MSM",     # Mt St Mary's variant
    "BOISE": "BSU",    # Boise State: Kalshi uses BOISE, we use BSU
    "HAM": "HAMP",     # Hampton: Kalshi uses HAMP
    # Issue #2: Missing NCAAB Aliases
    "LCHI": "LCHI", "LOYOLA CHICAGO": "LCHI",
    "IUIN": "IUIN", "IU INDIANAPOLIS": "IUIN",
    "MILW": "MILW", "MILWAUKEE": "MILW",
    "PFW": "PFW", "PURDUE FORT WAYNE": "PFW",
    "CHAR": "CHAR", "CHARLOTTE": "CHAR",
    "NEOM": "NEOM", "NEBRASKA OMAHA": "NEOM",
    "FAU": "FAU", "FLORIDA ATLANTIC": "FAU",
    "TULN": "TULN", "TULANE": "TULN",
    "UAB": "UAB",
    "MRST": "MRST", "MARIST": "MRST",
    "ILST": "ILST", "ILLINOIS STATE": "ILST",
    "CAMP": "CAMP", "CAMPBELL": "CAMP",
    "ORST": "ORST", "OREGON STATE": "ORST",
    "SEA": "SEA", "SEATTLE": "SEA", "SEAU": "SEA", "SEAT": "SEA",
    "COFC": "COFC",
    "UO": "ORE", "ORG": "ORE", "OREG": "ORE",
    "STLU": "SLU",
    "WASH": "WAS",
    # Fixes for Murray St/Belmont, San Diego/San Francisco, Monmouth/Towson
    "MONM": "MON", "TOWS": "TOW",
    "BELM": "BEL", "BELMT": "BEL",
    "SF": "USF", "SFC": "USF", "SFR": "USF",
    "SD": "USD", "SDG": "USD",
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

@lru_cache(maxsize=1024)
def resolve_team_code(code: str, league: str) -> str:
    """
    Resolve a team code (from event ticker or map) to its canonical form
    using alias maps if applicable.
    """
    if not code:
        return ""

    c = code.upper().strip()
    l = (league or "").upper()

    # Direct Lookup
    if l == "NCAAB":
        if c in NCAAB_CODE_ALIASES:
            return NCAAB_CODE_ALIASES[c]
        # FIX: Do NOT fuzzy match if the code is already a known canonical code
        # This prevents valid codes (e.g. MASS) from being fuzzy-matched to aliases (e.g. MISS)
        if c in KALSHI_NCAAB_TEAM_CODES.values():
            return c
    elif l == "NCAAF":
        if c in NCAAF_CODE_ALIASES:
            return NCAAF_CODE_ALIASES[c]

    # Fuzzy Lookup (Task 1) - If direct lookup fails
    # Only for NCAAB where variance is high
    if l == "NCAAB" and rapidfuzz:
        # Increase threshold for short codes to prevent false positives (e.g. VMI -> VIR)
        threshold = 90 if len(c) <= 3 else 75

        # Check against Alias Keys
        alias_keys = list(NCAAB_CODE_ALIASES.keys())
        match = rapidfuzz.process.extractOne(
            c, alias_keys, scorer=fuzz.ratio, score_cutoff=threshold
        )
        if match:
            # match is (key, score, index)
            best_key = match[0]
            logger.debug(f"Fuzzy Resolved Code: {c} -> {best_key} -> {NCAAB_CODE_ALIASES[best_key]} (score={match[1]})")
            return NCAAB_CODE_ALIASES[best_key]

    return c

def cross_reference_unmapped_ticker(league: str, date_token: str, team_block: str) -> Optional[Dict[str, str]]:
    """
    Uses SportsDataIO to find games on the date and match the team block.
    """
    if league != "NCAAB":
        return None

    try:
        # Parse date token (YYMONDD -> Date)
        # e.g. 26JAN15 -> 2026-01-15
        dt = datetime.strptime(date_token, "%y%b%d").date()

        # Initialize Client
        client = SportsDataNCAABClient()
        if not client.is_configured():
            return None

        # Fetch games
        games = client.get_games_by_date(dt)
        if not games:
            return None

        # Match logic: Try to find a game where Home/Away abbreviations combine to team_block
        # or share significant overlap
        for g in games:
            home = str(g.get("HomeTeam") or "").upper()
            away = str(g.get("AwayTeam") or "").upper()

            # Simple check: Does team_block look like Away+Home?
            # Remove non-alpha
            combined = re.sub(r"[^A-Z]", "", away + home)

            # If team_block is a substring of combined, or vice versa
            # Or if team_block matches abbreviations
            if team_block in combined or combined in team_block:
                return {"away": away, "home": home}

            # Check 3-letter codes if available
            home_id = str(g.get("HomeTeamID") or "")
            away_id = str(g.get("AwayTeamID") or "")

            # Heuristic: First 3 chars of name
            h_code = home[:3]
            a_code = away[:3]

            if (a_code + h_code) == team_block:
                return {"away": away, "home": home}

    except Exception as e:
        logger.warning(f"Cross-reference failed: {e}")

    return None


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
        # Check user-provided comprehensive mapping FIRST (Task: Team Name Normalization)
        # Use logic from normalize_team_for_kalshi but integrated safely

        # 1. Try direct/base lookup via KALSHI_NCAAB_TEAM_CODES
        # Note: We use the raw team_name for this lookup as keys are mixed case
        team_clean_raw = team_name.strip()
        if team_clean_raw in KALSHI_NCAAB_TEAM_CODES:
            return KALSHI_NCAAB_TEAM_CODES[team_clean_raw]

        # 2. Try removing last word (likely mascot)
        # This prevents "South Florida Bulls" -> "South" (wrong) but allows "South Florida" (correct)
        parts = team_clean_raw.split()
        if len(parts) > 1:
            without_last = " ".join(parts[:-1])
            if without_last in KALSHI_NCAAB_TEAM_CODES:
                return KALSHI_NCAAB_TEAM_CODES[without_last]

        # 3. Try base name split (fallback)
        base_name = parts[0]
        if base_name in KALSHI_NCAAB_TEAM_CODES:
             return KALSHI_NCAAB_TEAM_CODES[base_name]

        # Fallback to existing map
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
    """Convert a Kalshi price to a 0-1 probability.

    Handles both legacy cent integers (0-99) and current dollar values (0.0-1.0).
    As of Jan 2026, the Kalshi API only returns dollar-range values, but we
    keep the cent fallback for any cached/historical data.
    """
    if price is None: return None
    try:
        val = float(price)
        if 0.0 <= val <= 1.0:
            return val  # Already in dollar/probability range
        if 1.0 < val <= 100.0:
            return val / 100.0  # Legacy cent value
    except Exception:
        pass
    return None


def safe_float(x: Any) -> Optional[float]:
    """Convert to float; return None on blanks/NaN/non-numeric."""
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"", "none", "nan", "n/a"}:
        return None
    try:
        val = float(x)
        if val != val:  # NaN
            return None
        return val
    except Exception:
        return None


def _kalshi_price_norm(mkt: Dict[str, Any], dollars_key: str, cents_key: str) -> Optional[float]:
    """Read a Kalshi price field, preferring *_dollars (0-1 string) over deprecated cent int."""
    d = safe_float(mkt.get(dollars_key))
    if d is not None and d >= 0:
        return d
    c = safe_float(mkt.get(cents_key))
    if c is not None and c >= 0:
        return c / 100.0
    return None


def _extract_market_type(title: str, ticker: str, subtitle: str = "", market: Dict[str, Any] = None) -> str:
    t = (title or "").upper()
    tick = (ticker or "").upper()
    sub = (subtitle or "").upper()

    # 0. Check extra metadata if available (floor/cap/strike often indicate range markets)
    has_strikes = False
    if market:
        has_strikes = bool(market.get("floor_strike") or market.get("cap_strike") or market.get("strike"))

    # 1. Spread detection
    # "Winning Margin" is often used for spreads in some contexts, but usually "Point Spread"
    if "SPREAD" in tick or "KXNBASPREAD" in tick or "KXNFLSPREAD" in tick: return "spread"
    if "SPREAD" in t or "POINT SPREAD" in t or "POINTS" in t: return "spread"
    if "SPREAD" in sub or "POINT SPREAD" in sub or "WINNING MARGIN" in sub: return "spread"

    # Fix Issue #1: Aggressive suffix check for spread
    # Check if ticker ends with -TeamCode-Number (e.g. -PHX-6.5)
    # or contains negative/positive number
    # Regex for spread-like suffix: -[A-Z]{2,4}-[\d\.]+
    # EXCEPTION: Ensure "OVER" and "UNDER" are not mistaken for team codes
    if re.search(r'-[A-Z]{2,4}-[\d\.]+$', tick) and "OVER" not in tick and "UNDER" not in tick:
         # If subtitle has "winner", it's a winner market. If it has numbers, likely spread.
         if "WINNER" not in sub and "TOTAL" not in sub:
             return "spread"

    # Check if subtitle implies spread (e.g. "Chicago -3.5")
    # This is a heuristic for when "Spread" isn't explicitly in the text
    if "-" in sub and any(c.isdigit() for c in sub) and "TOTAL" not in sub and "OVER" not in sub:
         # Weak signal, but if has_strikes is true, likely a spread
         if has_strikes: return "spread"

    # 2. Total detection
    if "TOTAL" in tick or "KXNBATOTAL" in tick or "KXNFLTOTAL" in tick: return "total"
    if "TOTAL" in t or "OVER/UNDER" in t or "O/U" in t or "TOTAL POINTS" in t: return "total"
    if "TOTAL" in sub or "TOTAL POINTS" in sub or "OVER/UNDER" in sub: return "total"

    # Fix Issue #1: Aggressive suffix check for total (e.g. -OVER220, -UNDER220)
    if "OVER" in tick or "UNDER" in tick:
        return "total"

    # Task 2 Fix: Check for explicit league-specific total/spread keywords in ticker
    # e.g. KXNCAAMBTOTAL, KXNCAAMBSPREAD
    if "KXNCAAMBTOTAL" in tick or "KXNCAAFTOTAL" in tick or "KXMLBTOTAL" in tick or "KXNHLTOTAL" in tick:
        return "total"
    if "KXNCAAMBSPREAD" in tick or "KXNCAAFSPREAD" in tick or "KXMLBSPREAD" in tick or "KXNHLSPREAD" in tick:
        return "spread"

    # 3. Moneyline/Winner detection
    if "MONEYLINE" in t or "ML" in t or "WINNER" in t: return "moneyline"
    if "WINNER" in sub: return "moneyline"

    # Check ticker for Game Winner pattern (usually just ends with team code)
    # e.g. KXNBAGAME-26JAN19MIAGSW-MIA
    # If we haven't matched spread/total yet, and it looks like a winner ticker...
    if "GAME" in tick and not ("SPREAD" in tick or "TOTAL" in tick):
        return "moneyline"

    # 4. Fallback based on strikes if generic ticker
    if has_strikes:
        # If we have strikes but no text signal, it's likely a spread or total.
        # Winner markets usually don't have floor/cap/strike (binary yes/no).
        # We need to distinguish Spread vs Total.
        # Totals usually have higher numbers (e.g. > 30 for NBA/NFL) or "Over/Under" text.
        # Spreads usually have smaller numbers or negative numbers.
        pass

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

    market_type = _extract_market_type(title, ticker, subtitle=mkt.get("subtitle", ""), market=mkt)
    # Use midpoint of yes_bid/yes_ask if both available, else fall back to last_price.
    # Prefer _dollars fields (current API); fall back to deprecated cent fields.
    # Uses the same normalization as _kalshi_prices() in streamlit_app.py.
    _yb = _kalshi_price_norm(mkt, "yes_bid_dollars", "yes_bid")
    _ya = _kalshi_price_norm(mkt, "yes_ask_dollars", "yes_ask")
    _nb = _kalshi_price_norm(mkt, "no_bid_dollars", "no_bid")
    _lp = _kalshi_price_norm(mkt, "last_price_dollars", "last_price")

    prob = None
    if _yb is not None and _ya is not None and _yb > 0 and _ya > 0:
        prob = (_yb + _ya) / 2.0
    elif _yb is not None and _nb is not None:
        implied_ya = 1.0 - _nb
        prob = (_yb + implied_ya) / 2.0
    elif _yb is not None:
        prob = _lp if (_lp is not None and _lp > 0) else _yb
    elif _nb is not None:
        prob = 1.0 - _nb

    # Final fallback to last_price
    if prob is None and _lp is not None and _lp > 0:
        prob = _lp

    if prob is not None:
        logger.info(f"Kalshi prob calc: market={ticker}, yes_side={title}, yes_bid={_yb}, yes_ask={_ya}, mid_prob={prob:.3f}")

    # Add yes_side inference for logging
    yes_side = title  # Default yes side is the title
    if prob is not None:
         logger.info(f"Kalshi metadata: yes_side='{yes_side}', prob={prob:.3f}")

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
    status: Optional[str],
    requested_market_type: Optional[str] = None,
    home_team_name: str = None,
    away_team_name: str = None
) -> Optional[KalshiMatchResult]:
    """
    Attempt to match a game to an event by scanning the /events endpoint first.
    This is more efficient and accurate for leagues with structured tickers (NBA/NFL/NCAA).
    """
    # DEBUG: Log input parameters
    logger.info(f"🔍 KALSHI MATCH ATTEMPT [{league}]:")
    logger.info(f"   Game Time (UTC): {game_dt_utc}")
    logger.info(f"   Home Codes: {home_codes}")
    logger.info(f"   Away Codes: {away_codes}")
    logger.info(f"   Home Name: {home_team_name}")
    logger.info(f"   Away Name: {away_team_name}")
    logger.info(f"   Status Filter: {status}")

    # 1. Determine series ticker
    series_ticker = None
    if league == "NBA": series_ticker = "KXNBAGAME"
    elif league == "NFL": series_ticker = "KXNFLGAME"
    elif league == "NCAAB": series_ticker = "KXNCAAMBGAME"
    elif league == "NCAAF": series_ticker = "KXNCAAFGAME"
    elif league == "MLB": series_ticker = "KXMLBGAME"
    elif league == "NHL": series_ticker = "KXNHLGAME"

    if not series_ticker:
        logger.info(f"   ❌ No series ticker for league {league}")
        return None

    logger.info(f"   Series Ticker: {series_ticker}")

    # 2. Fetch events (using cache inside get_events)
    # IMPORTANT: Try without status filter first to get all events
    # This significantly improves match rate as it doesn't filter out events in different statuses
    try:
        # First try without status filter to get ALL events
        events_resp = integrator.get_events(series_ticker, status=None)
        events = events_resp.get("events", [])
        logger.info(f"   Total Events Fetched (no status filter): {len(events)}")

        # If no events found and status was specified, try with status filter
        if not events and status:
            logger.info(f"   Retrying with status={status}...")
            events_resp = integrator.get_events(series_ticker, status=status)
            events = events_resp.get("events", [])
            logger.info(f"   Total Events Fetched (with status={status}): {len(events)}")
    except Exception as e:
        logger.warning(f"   ❌ Failed to fetch events: {e}")
        return None

    if not events:
        logger.info(f"   ❌ No events found for {series_ticker}")
        return None

    # Log sample events for debugging
    logger.info(f"   Sample Event Tickers (first 5):")
    for i, evt in enumerate(events[:5]):
        ticker = evt.get("ticker", "N/A")
        close_time = evt.get("close_time", "N/A")
        logger.info(f"      [{i+1}] {ticker} (closes: {close_time})")

    # Capture available blocks for debug
    available_blocks = []
    if league == "NCAAB":
        for evt in events[:50]: # Sample first 50
            t = evt.get("ticker", "")
            # Pattern: KXNCAAMBGAME-DATE[BLOCK]
            if "-" in t:
                parts = t.split("-")
                if len(parts) > 1:
                    suffix = parts[1] # 26FEB11USCOSU
                    # Strip date (YYMONDD)
                    match = re.match(r"^(\d{2}[A-Z]{3}\d{2})([A-Z0-9]+)$", suffix)
                    if match:
                        available_blocks.append(match.group(2))
        logger.info(f"   Sample Available Blocks (NCAAB): {available_blocks[:15]}")

    best_event = None
    best_score = 0.0
    best_details = None
    all_candidates = []  # Track all potential candidates for debug (Fix #4)

    # Time window for matching (hours)
    # Increased from 36 to 72 to be more generous with date matching
    # League-specific time windows:
    # - NCAAB: 72h (games bucket by date, not time)
    # - NBA/NFL/NHL/MLB: 24h (games scheduled to the minute)
    TIME_WINDOW_HOURS = 72 if league == 'NCAAB' else 24

    # Resolve our candidates once before loop (optimization + fix for UnboundLocalError)
    resolved_home = {resolve_team_code(c, league) for c in home_codes}
    resolved_away = {resolve_team_code(c, league) for c in away_codes}

    for evt in events:
        ticker = evt.get("ticker")
        logger.info(f"Event ticker parsing: input='{ticker}' → parsed={parse_event_ticker_codes(ticker)}")
        parsed = parse_event_ticker_codes(ticker)
        if not parsed:
            continue

        evt_away_code = resolve_team_code(parsed.get("away"), league)
        evt_home_code = resolve_team_code(parsed.get("home"), league)

        # Check codes against our candidates
        score_1 = 0

        # Calculate scores with fuzzy support (Task 1)
        # 50 = Exact match
        # 40 = Strong fuzzy match (ratio >= 80)
        # 30 = Medium fuzzy match (ratio >= 65, NCAAB/NCAAF only)

        def _get_code_score(code: str, candidates: set, league: str) -> int:
            if code in candidates:
                return 50

            # Fuzzy fallback
            if rapidfuzz and code and len(code) >= 2:
                best_r = 0
                for cand in candidates:
                    if not cand: continue
                    r = fuzz.ratio(code, cand)
                    if r > best_r:
                        best_r = r

                if best_r >= 80:
                    return 40
                if best_r >= 65 and league in ['NCAAB', 'NCAAF']:
                    return 30
            return 0

        s_away_1 = _get_code_score(evt_away_code, resolved_away, league)
        s_home_1 = _get_code_score(evt_home_code, resolved_home, league)
        score_1 = s_away_1 + s_home_1

        s_away_2 = _get_code_score(evt_away_code, resolved_home, league)
        s_home_2 = _get_code_score(evt_home_code, resolved_away, league)
        score_2 = s_away_2 + s_home_2

        # Flags for logging
        away_match_1 = s_away_1 > 0
        home_match_1 = s_home_1 > 0
        away_match_2 = s_away_2 > 0
        home_match_2 = s_home_2 > 0

        match_score = max(score_1, score_2)

        if match_score > 0:
            all_candidates.append({
                "ticker": ticker,
                "away": evt_away_code,
                "home": evt_home_code,
                "score": match_score
            })

        # Fallback: Try Name Matching on Event Title if code match failed
        if match_score < 50 and home_team_name and away_team_name:
            evt_title = evt.get("title", "")
            if evt_title:
                # Normalize everything using TeamNameMatcher
                # Note: TeamNameMatcher.normalize handles upper casing and special chars
                title_norm = TeamNameMatcher.normalize(evt_title)
                h_norm = TeamNameMatcher.normalize(home_team_name)
                a_norm = TeamNameMatcher.normalize(away_team_name)

                # Check for containment of BOTH teams
                # This is a very strong signal (e.g. "Lakers vs Celtics" contains "LAKERS" and "CELTICS")
                if h_norm and a_norm and h_norm in title_norm and a_norm in title_norm:
                    # High confidence match based on full names
                    match_score = 90 # Treat as high confidence (overrides low code score)
                    logger.info(f"   ✅ Name Fallback Match: '{home_team_name}' & '{away_team_name}' found in '{evt_title}'")

                    # Add to candidates for debug
                    all_candidates.append({
                        "ticker": ticker,
                        "away": "NAME_MATCH",
                        "home": "NAME_MATCH",
                        "score": match_score,
                        "note": "fallback_name_match"
                    })

        if match_score < 50:
            continue

        # Time check and scoring adjustment
        time_diff_hours = None
        time_score = 0

        close_ts = evt.get("close_time") # ISO string
        if close_ts:
            try:
                dt = datetime.fromisoformat(str(close_ts).replace("Z", "+00:00"))
                if dt.tzinfo is None: dt = pytz.utc.localize(dt)

                time_diff_hours = abs((dt - game_dt_utc).total_seconds()) / 3600.0

                # Time Scoring Logic (Bonus for tight match, penalty for wide miss)
                is_pro = league in ["NBA", "NFL", "NHL", "MLB"]
                # Tighter window for pros (exact schedule), looser for college (daily buckets)
                # v106: Relaxed wide_window for Pro from 24h to 36h to prevent penalties on timezone drifts
                tight_window = 12 if not is_pro else 6
                wide_window = 36  # Unified 36h window for all leagues

                if time_diff_hours <= tight_window:
                    time_score = 25  # Bonus for date confirmation
                elif time_diff_hours > wide_window:
                    time_score = -25 # Penalty for wrong day

            except:
                pass

        final_score = match_score + time_score

        # Enhanced logging for EVERY potential match attempt (score >= 50) (Fix #5)
        logger.info(f"   🎲 Evaluating: {ticker}")
        logger.info(f"      Raw Codes: away={parsed.get('away')}, home={parsed.get('home')}")
        logger.info(f"      Resolved Codes: away={evt_away_code}, home={evt_home_code}")
        logger.info(f"      Expected Away Codes: {list(resolved_away)[:3]}")
        logger.info(f"      Expected Home Codes: {list(resolved_home)[:3]}")
        logger.info(f"      Score Calculation:")
        logger.info(f"         - Away Match: {away_match_1} (score={s_away_1})")
        logger.info(f"         - Home Match: {home_match_1} (score={s_home_1})")
        logger.info(f"         - Direct Score: {score_1}")
        logger.info(f"         - Swap Score: {score_2}")
        logger.info(f"         - Team Score: {match_score}")
        if time_diff_hours is not None:
            logger.info(f"      Time Check: {time_diff_hours:.1f}h diff (Score Adj: {time_score:+})")
        logger.info(f"      Final Score: {final_score} (Threshold: 70)")

        if final_score > best_score:
            best_score = final_score
            best_event = evt
            best_details = {
                "ticker": ticker,
                "parsed_away": parsed.get("away"),
                "parsed_home": parsed.get("home"),
                "resolved_away": evt_away_code,
                "resolved_home": evt_home_code,
                "score_1": score_1,
                "score_2": score_2,
                "time_diff_hours": time_diff_hours,
                "time_score": time_score
            }

    # Log final result

    # DIAGNOSTIC: Log ALL candidates regardless of score (Fix missing matches)
    if not best_event:
        logger.warning(f"   ❌ NO CANDIDATES FOUND for {league}")
        logger.warning(f"      Total events scanned: {len(events)}")
        logger.warning(f"      Expected home codes: {list(resolved_home)[:10]}")
        logger.warning(f"      Expected away codes: {list(resolved_away)[:10]}")

        # Sample first 10 events to show what's available
        logger.warning(f"      Sample available events:")
        for i, evt in enumerate(events[:10]):
            ticker = evt.get("ticker", "")
            parsed = parse_event_ticker_codes(ticker)
            logger.warning(f"         [{i+1}] {ticker} → home={parsed.get('home')}, away={parsed.get('away')}")
        return None

    # Dynamic Threshold (Task 1)
    # Pro Leagues: 75 (Relaxed from 80 to allow 100-25 time penalty cases)
    # College: 65 (Relaxed from 70)
    # Fix 2: Further Relax Team Code Matching Threshold
    if league in ['NBA', 'NFL', 'NHL', 'MLB']:
        MATCH_THRESHOLD = 70  # Was 75
    else:
        MATCH_THRESHOLD = 50  # Was 65 (More permissive for NCAAB)

    if best_event:
        logger.info(f"   Best Match Found: {best_details['ticker']}")
        logger.info(f"      Score: {best_score} (threshold: {MATCH_THRESHOLD})")
        logger.info(f"      Details: {best_details}")
        if best_score < MATCH_THRESHOLD:
            logger.warning(f"   ❌ MATCH FAILED for {league}: score={best_score}/{MATCH_THRESHOLD}")
            logger.warning(f"      Expected home: {list(resolved_home)[:5]}")
            logger.warning(f"      Expected away: {list(resolved_away)[:5]}")
            logger.warning(f"      Best candidate: {best_event.get('ticker')} (score={best_score})")

            # ALIAS SUGGESTION: Show which codes were close
            if best_details:
                kalshi_home = best_details.get('resolved_home')
                kalshi_away = best_details.get('resolved_away')

                # Check if ONE side matched (50 points = one team correct)
                if best_score == 50:
                    if kalshi_home not in resolved_home and kalshi_away in resolved_away:
                        logger.warning(f"      💡 ALIAS NEEDED: Add '{kalshi_home}' → one of {list(resolved_home)[:3]} to NCAAB_CODE_ALIASES")
                    elif kalshi_away not in resolved_away and kalshi_home in resolved_home:
                        logger.warning(f"      💡 ALIAS NEEDED: Add '{kalshi_away}' → one of {list(resolved_away)[:3]} to NCAAB_CODE_ALIASES")

                logger.warning(f"      Kalshi codes: home={kalshi_home}, away={kalshi_away}")
                logger.warning(f"      All non-zero candidates: {all_candidates[:10]}")

            return None  # Match failed
        logger.info(f"   ✅ MATCH SUCCESSFUL")

    if best_event and best_score >= MATCH_THRESHOLD: # High confidence match
        # CRITICAL: Verify this is the correct league before processing markets
        # This prevents NCAAB-specific logic from corrupting NBA/NFL matches
        event_ticker = best_event.get("ticker", "")
        if league == "NCAAB" and "NCAAMB" not in event_ticker.upper():
            logger.warning(f"   ❌ League mismatch: Expected NCAAB but event ticker is {event_ticker}")
            return None
        elif league == "NBA" and "NBA" not in event_ticker.upper():
            logger.warning(f"   ❌ League mismatch: Expected NBA but event ticker is {event_ticker}")
            return None
        elif league == "NFL" and "NFL" not in event_ticker.upper():
            logger.warning(f"   ❌ League mismatch: Expected NFL but event ticker is {event_ticker}")
            return None

        # Extract nested markets from the event object directly
        markets = best_event.get("markets", [])
        evt_ticker = best_event.get("ticker")

        # Initialize fallback
        force_match_result = None

        # If markets missing (e.g. from cache without nested), fetch them explicitly
        # ONLY for leagues where we know this is necessary (NCAAB primarily)
        if not markets and league in ["NCAAB"]:
            logger.info(f"   NCAAB: Attempting aggressive market fetch for {evt_ticker}")

            # Attempt 1: Fetch directly by event_ticker
            try:
                markets_resp = integrator._request("GET", "/markets", params={"event_ticker": evt_ticker})
                markets = markets_resp.get("markets", [])
                logger.info(f"   Fetched {len(markets)} markets by event_ticker")
            except Exception as e:
                logger.warning(f"   Market fetch failed: {e}")
                markets = []

            # Attempt 2: If still empty, try series_ticker search (last resort)
            if not markets:
                logger.info(f"   NCAAB: Attempting series-wide market search for {evt_ticker}")
                try:
                    series = "KXNCAAMB"
                    all_series_mkts = integrator.get_markets_paginated(
                        status=None,
                        limit=200,
                        max_pages=20,
                        extra_params={"series_ticker": series}
                    )
                    # Filter for this specific event
                    markets = [m for m in all_series_mkts if m.get("event_ticker") == evt_ticker]
                    logger.info(f"   Found {len(markets)} markets via series search")
                except Exception as e:
                    logger.warning(f"   Series search failed: {e}")
        elif not markets:
            # For non-NCAAB leagues, try ONE direct fetch only (no expensive series search)
            try:
                markets_resp = integrator._request("GET", "/markets", params={"event_ticker": evt_ticker})
                markets = markets_resp.get("markets", [])
                if markets:
                    logger.info(f"   Fetched {len(markets)} markets by event_ticker for {league}")
            except Exception:
                markets = []

        # SAFETY CHECK: For non-NCAAB leagues, empty markets after fetch means no valid event
        # For NCAAB, we allow proceeding to force match logic below
        if not markets:
            logger.info(f"   ⚠️ No markets found in main event {best_event.get('ticker')} (league={league}). Proceeding to check for Spread/Total series...")

        # Allow flow to proceed even if markets are empty (Spread/Total search or NCAAB force match will handle it)

        # ENHANCED: Classify all markets as winner/spread/total
        winner_market = None
        spread_markets = []
        total_markets = []
        target_market = None
        match_reason_detail = None

        # Logic for existing complex classification logic for other leagues (NBA, etc.) AND NCAAB (now unified)

        # ========== FIX: Search for spread/total events separately ==========
        # Spread and total markets are in DIFFERENT event series (KXNBATOTAL, KXNBASPREAD)
        # Extract the date-team identifier from the matched GAME event ticker
        # e.g., "KXNBAGAME-26JAN27BKNPHX" -> "26JAN27BKNPHX"
        game_evt_ticker = best_event.get("ticker", "")
        # Only perform search if not NCAAB or if NCAAB needs it (NCAAB usually has nested markets but let's allow it)
        # Note: Previous code split logic here. We will apply search logic generally but keep NCAAB specific series inside loop.
        # FIX: Allow spread/total search for all leagues including NCAAB
        if True:
            game_ticker_parts = game_evt_ticker.split("-")
            if len(game_ticker_parts) >= 2:
                date_team_id = game_ticker_parts[1]  # e.g., "26JAN27BKNPHX"

                # Determine the spread/total series tickers based on league
                spread_series_list = []
                total_series_list = []
                if league == "NBA":
                    spread_series_list = ["KXNBASPREAD"]
                    total_series_list = ["KXNBATOTAL"]
                elif league == "NFL":
                    spread_series_list = ["KXNFLSPREAD"]
                    total_series_list = ["KXNFLTOTAL"]
                elif league == "NHL":
                    spread_series_list = ["KXNHLSPREAD"]
                    total_series_list = ["KXNHLTOTAL"]
                elif league == "MLB":
                    spread_series_list = ["KXMLBSPREAD"]
                    total_series_list = ["KXMLBTOTAL"]
                elif league == "NCAAB":
                    # Try multiple variants for NCAAB to cover inconsistencies
                    spread_series_list = ["KXNCAAMBSPREAD", "KXNCAABSPREAD"]
                    total_series_list = ["KXNCAAMBTOTAL", "KXNCAABTOTAL"]
                elif league == "NCAAF":
                    spread_series_list = ["KXNCAAFSPREAD"]
                    total_series_list = ["KXNCAAFTOTAL"]

                logger.info(f"🔍 KALSHI SPREAD/TOTAL SEARCH: Looking for events matching '{date_team_id}'")
                logger.info(f"   Spread series candidates: {spread_series_list}, Total series candidates: {total_series_list}")

                # Use the first candidate as primary for fallbacks
                primary_spread_series = spread_series_list[0] if spread_series_list else None
                primary_total_series = total_series_list[0] if total_series_list else None

                # Search for spread/total events using the date-team identifier
                # Method 1: Try to fetch markets directly using series_ticker and matching date-team
                try:
                    spread_markets_found = False
                    for spread_series in spread_series_list:
                        if not spread_series: continue

                        spread_event_ticker = f"{spread_series}-{date_team_id}"
                        logger.info(f"   Searching for spread event: {spread_event_ticker}")
                        try:
                            spread_mkts_resp = integrator._request("GET", "/markets", params={"event_ticker": spread_event_ticker})
                            spread_mkts = spread_mkts_resp.get("markets", [])
                            if spread_mkts:
                                spread_markets.extend(spread_mkts)
                                logger.info(f"   ✅ Found {len(spread_mkts)} spread markets from event {spread_event_ticker}")
                                spread_markets_found = True
                                break # Found valid series, stop looking
                        except Exception:
                            continue # Try next variant

                    if not spread_markets_found and primary_spread_series:
                        # Fix 1: Fuzzy search fallback using primary series
                        logger.info(f"   No direct spread event match, trying fuzzy match on {primary_spread_series}...")
                        try:
                            all_spread_events = integrator.get_events(primary_spread_series, status=None)
                            date_token = date_team_id[:7] if len(date_team_id) >= 7 else date_team_id
                            for evt in all_spread_events.get("events", []):
                                evt_ticker = evt.get("ticker", "")
                                # Match by date token AND team codes (partial)
                                if date_token in evt_ticker:
                                    parsed = parse_event_ticker_codes(evt_ticker)
                                    evt_codes = {parsed.get("home"), parsed.get("away")}
                                    our_codes = set(home_codes + away_codes)
                                    if evt_codes & our_codes:  # Any overlap
                                        logger.info(f"   Found fuzzy spread event match: {evt_ticker}")
                                        spread_markets.extend(evt.get("markets", []))
                        except Exception as e:
                            logger.warning(f"   Fuzzy spread search failed: {e}")

                        # Fallback 1: Search in series EVENTS for matching date-team ID
                        logger.info(f"   Searching in series events for strict match...")
                        try:
                            series_resp = integrator.get_events(primary_spread_series, status=None)
                            series_events = series_resp.get("events", [])
                            for evt in series_events:
                                evt_tick = evt.get("ticker", "")
                                if date_team_id in evt_tick:
                                    logger.info(f"   Found matching spread event: {evt_tick}")
                                    evt_markets = evt.get("markets", [])
                                    if not evt_markets:
                                        evt_mkts_resp = integrator._request("GET", "/markets", params={"event_ticker": evt_tick})
                                        evt_markets = evt_mkts_resp.get("markets", [])
                                    spread_markets.extend(evt_markets)
                                    logger.info(f"   ✅ Added {len(evt_markets)} spread markets from {evt_tick}")
                        except Exception: pass

                        # Fallback 2: If still no spread markets, fetch MARKETS directly by series_ticker
                        if not spread_markets:
                            logger.info(f"   No spread events found, fetching markets directly by series_ticker={primary_spread_series}...")
                            try:
                                series_markets = integrator.get_markets_paginated(
                                    status=None,
                                    limit=200,
                                    max_pages=50,  # Fix 5: Ensure we fetch ALL spread/total markets
                                    extra_params={"series_ticker": primary_spread_series}
                                )
                                logger.info(f"   📊 Spread market pagination: Fetched {len(series_markets)} markets from series {primary_spread_series} (max_pages=50)")
                                # Add sample ticker logging (Fix #5)
                                if series_markets:
                                    sample_tickers = [m.get('ticker', '')[:50] for m in series_markets[:5]]
                                    logger.info(f"      Sample spread tickers: {sample_tickers}")

                                # Filter markets by date_team_id in ticker or event_ticker
                                for mkt in series_markets:
                                    mkt_ticker = str(mkt.get("ticker") or "").upper()
                                    mkt_event_ticker = str(mkt.get("event_ticker") or "").upper()
                                    if date_team_id.upper() in mkt_ticker or date_team_id.upper() in mkt_event_ticker:
                                        spread_markets.append(mkt)
                                if spread_markets:
                                    logger.info(f"   ✅ Found {len(spread_markets)} spread markets matching {date_team_id} from series")
                            except Exception as e:
                                logger.warning(f"   Pagination fetch failed: {e}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to fetch spread markets: {e}")

                try:
                    total_markets_found = False
                    for total_series in total_series_list:
                        if not total_series: continue

                        total_event_ticker = f"{total_series}-{date_team_id}"
                        logger.info(f"   Searching for total event: {total_event_ticker}")
                        try:
                            total_mkts_resp = integrator._request("GET", "/markets", params={"event_ticker": total_event_ticker})
                            total_mkts = total_mkts_resp.get("markets", [])
                            if total_mkts:
                                total_markets.extend(total_mkts)
                                logger.info(f"   ✅ Found {len(total_mkts)} total markets from event {total_event_ticker}")
                                total_markets_found = True
                                break # Found valid series
                        except Exception:
                            continue

                    if not total_markets_found and primary_total_series:
                        # Fix 1: Fuzzy search fallback using primary series
                        logger.info(f"   No direct total event match, trying fuzzy match on {primary_total_series}...")
                        try:
                            all_total_events = integrator.get_events(primary_total_series, status=None)
                            date_token = date_team_id[:7] if len(date_team_id) >= 7 else date_team_id
                            for evt in all_total_events.get("events", []):
                                evt_ticker = evt.get("ticker", "")
                                # Match by date token AND team codes (partial)
                                if date_token in evt_ticker:
                                    parsed = parse_event_ticker_codes(evt_ticker)
                                    evt_codes = {parsed.get("home"), parsed.get("away")}
                                    our_codes = set(home_codes + away_codes)
                                    if evt_codes & our_codes:  # Any overlap
                                        logger.info(f"   Found fuzzy total event match: {evt_ticker}")
                                        total_markets.extend(evt.get("markets", []))
                        except Exception as e:
                            logger.warning(f"   Fuzzy total search failed: {e}")

                        # Fallback 1: Search in series EVENTS for matching date-team ID
                        logger.info(f"   Searching in series events for strict match...")
                        try:
                            series_resp = integrator.get_events(primary_total_series, status=None)
                            series_events = series_resp.get("events", [])
                            for evt in series_events:
                                evt_tick = evt.get("ticker", "")
                                if date_team_id in evt_tick:
                                    logger.info(f"   Found matching total event: {evt_tick}")
                                    evt_markets = evt.get("markets", [])
                                    if not evt_markets:
                                        evt_mkts_resp = integrator._request("GET", "/markets", params={"event_ticker": evt_tick})
                                        evt_markets = evt_mkts_resp.get("markets", [])
                                    total_markets.extend(evt_markets)
                                    logger.info(f"   ✅ Added {len(evt_markets)} total markets from {evt_tick}")
                        except Exception: pass

                        # Fallback 2: If still no total markets, fetch MARKETS directly by series_ticker
                        if not total_markets:
                            logger.info(f"   No total events found, fetching markets directly by series_ticker={primary_total_series}...")
                            try:
                                series_markets = integrator.get_markets_paginated(
                                    status=None,
                                    limit=200,
                                    max_pages=50,  # Fix 5: Ensure we fetch ALL spread/total markets
                                    extra_params={"series_ticker": primary_total_series}
                                )
                                logger.info(f"   📊 Total market pagination: Fetched {len(series_markets)} markets from series {primary_total_series} (max_pages=50)")
                                # Filter markets by date_team_id in ticker or event_ticker
                                for mkt in series_markets:
                                    mkt_ticker = str(mkt.get("ticker") or "").upper()
                                    mkt_event_ticker = str(mkt.get("event_ticker") or "").upper()
                                    if date_team_id.upper() in mkt_ticker or date_team_id.upper() in mkt_event_ticker:
                                        total_markets.append(mkt)
                                if total_markets:
                                    logger.info(f"   ✅ Found {len(total_markets)} total markets matching {date_team_id} from series")
                            except Exception as e:
                                logger.warning(f"   Pagination fetch failed: {e}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to fetch total markets: {e}")

                logger.info(f"   📊 After spread/total search: {len(spread_markets)} spread, {len(total_markets)} total markets")
        # ========== END FIX ==========

        # Fix Issue #1: Log FULL market list for debug analysis
        # Only log full list for the first few events to avoid spam.
        global _DEBUG_GAME_LOG_COUNT
        should_log_debug = _DEBUG_GAME_LOG_COUNT < 3

        logger.debug(f"🔍 KALSHI DEBUG [{league}]: Found {len(markets)} markets for event {best_event.get('ticker')}")
        if should_log_debug:
             logger.info(f"DEBUG: Full market list for {best_event.get('ticker')}:")
             _DEBUG_GAME_LOG_COUNT += 1

        for idx, m in enumerate(markets):
            ticker = m.get("ticker", "")
            title = (m.get("title") or "").lower()
            subtitle = (m.get("subtitle") or "").lower()

            # Classify market type with enhanced logic
            market_type = _extract_market_type(title, ticker, subtitle, market=m)

            # Extract line information for spread/total
            floor_str = m.get("floor_strike") or m.get("floor")
            cap_str = m.get("cap_strike") or m.get("cap")
            strike_str = m.get("strike")

            # VERBOSE LOGGING (Requested by user)
            # Log the EXACT raw ticker and key fields for debugging
            # Show ALL markets in logs for debugging if within limit
            if should_log_debug:
                logger.info(f"   RAW MARKET [{idx+1}]: ticker='{ticker}' | type='{market_type}' | title='{title}' | sub='{subtitle}' | strike='{strike_str}'")

            if floor_str or cap_str or strike_str:
                logger.debug(f"      Line info: floor={floor_str}, cap={cap_str}, strike={strike_str}")

            # Expanded logic using market_type from ticker check
            # Prioritize explicit classification from _extract_market_type
            if market_type == "moneyline":
                winner_market = m
            elif market_type == "spread":
                spread_markets.append(m)
            elif market_type == "total":
                total_markets.append(m)
            else:
                # Fallback keywords if "generic"
                if "winner" in title or "winner" in subtitle:
                    winner_market = m
                elif "spread" in title or "spread" in subtitle or "points" in title:
                    spread_markets.append(m)
                elif "total" in title or "total" in subtitle or "over" in title or "under" in title:
                    total_markets.append(m)

        # Log summary of classifications
        logger.info(f"🎯 KALSHI MATCH [{league}]: Event {best_event.get('ticker')} - "
                   f"Winner: {'✓' if winner_market else '✗'}, "
                   f"Spread: {len(spread_markets)}, "
                   f"Total: {len(total_markets)}")

        if spread_markets:
            logger.info(f"   📊 Spread markets found: {[m.get('ticker')[:40] for m in spread_markets[:3]]}")
        if total_markets:
            logger.info(f"   📊 Total markets found: {[m.get('ticker')[:40] for m in total_markets[:3]]}")

        # TARGET SELECTION LOGIC
        # 1. Prefer requested type if available
        # 2. Fallback to any available type (Spread/Total) if requested type missing
        # 3. Fallback to Winner/Default

        if requested_market_type:
            req_upper = requested_market_type.upper()
            logger.info(f"   🎯 Requested Market Type: {req_upper}")

            if "SPREAD" in req_upper and spread_markets:
                target_market = spread_markets[0]
                match_reason_detail = "matched_spread"
            elif "TOTAL" in req_upper and total_markets:
                target_market = total_markets[0]
                match_reason_detail = "matched_total"
            # Fallback: Requested type missing, check if other type exists
            elif spread_markets:
                target_market = spread_markets[0]
                match_reason_detail = "matched_spread_fallback"
                logger.info(f"   ⚠️ Requested {req_upper} but found SPREAD. Using fallback.")
            elif total_markets:
                target_market = total_markets[0]
                match_reason_detail = "matched_total_fallback"
                logger.info(f"   ⚠️ Requested {req_upper} but found TOTAL. Using fallback.")

        # If no target selected yet (or no request), use default logic
        if not target_market:
            # NCAAB: Prefer Spread/Total over Winner if not requested
            if league == 'NCAAB':
                if spread_markets:
                    target_market = spread_markets[0]
                    match_reason_detail = "matched_spread_default"
                elif total_markets:
                    target_market = total_markets[0]
                    match_reason_detail = "matched_total_default"
                elif winner_market:
                    target_market = winner_market
                    match_reason_detail = "matched_winner"

                # Force Match Logic (moved here): If still no target, try to find ANY valid market
                # Relaxed from 80 to MATCH_THRESHOLD (50) for NCAAB per user request
                if not target_market and best_score >= MATCH_THRESHOLD:
                    logger.info(f"🎯 NCAAB FORCE MATCH ATTEMPT: {evt_ticker} score={best_score}")
                    # Iterate ALL markets to find one with valid probability
                    for cand in markets:
                        # Calculate probability
                        yes_bid = _kalshi_price_norm(cand, "yes_bid_dollars", "yes_bid")
                        yes_ask = _kalshi_price_norm(cand, "yes_ask_dollars", "yes_ask")
                        no_bid = _kalshi_price_norm(cand, "no_bid_dollars", "no_bid")
                        last_price = _kalshi_price_norm(cand, "last_price_dollars", "last_price")

                        prob = None
                        if yes_bid is not None and yes_ask is not None:
                            prob = (yes_bid + yes_ask) / 2.0
                        elif yes_bid is not None and no_bid is not None:
                            prob = (yes_bid + (1.0 - no_bid)) / 2.0
                        elif last_price is not None and last_price > 0:
                            prob = last_price

                        # Validate title length and probability
                        cand_title = cand.get('title', '')
                        if len(cand_title) > 5 and prob is not None and 0.01 < prob < 0.99:
                            logger.info(f"   ✅ NCAAB FORCE MATCH SUCCESS: {cand.get('ticker')} prob={prob:.3f}")
                            return KalshiMatchResult(
                                matched=True,
                                kalshi_available=True,
                                label=cand_title,
                                probability=prob,
                                raw_event_id=evt_ticker,
                                market_ticker=cand.get("ticker"),
                                league=league,
                                reason='ncaab_force_match',
                                market_type='force',
                                game_date=game_dt_utc
                            )
                    logger.warning(f"   ⚠️ NCAAB force match failed: No valid markets found in {len(markets)} candidates")

                if not target_market and markets:
                    target_market = markets[0]
                    match_reason_detail = "matched_first_available"
            else:
                # Other leagues: Prefer Winner
                if winner_market:
                    target_market = winner_market
                    match_reason_detail = "matched_winner"
                elif spread_markets:
                    target_market = spread_markets[0]
                    match_reason_detail = "matched_spread_fallback"
                elif total_markets:
                    target_market = total_markets[0]
                    match_reason_detail = "matched_total_fallback"
                elif markets:
                    target_market = markets[0]
                    match_reason_detail = "matched_first_available"


        if target_market:
            # Calculate prob using _dollars fields (current API) with cent fallback.
            # Matches the same cascade as _kalshi_prices()/winner_prob() in streamlit_app.py.
            yes_bid = _kalshi_price_norm(target_market, "yes_bid_dollars", "yes_bid")
            yes_ask = _kalshi_price_norm(target_market, "yes_ask_dollars", "yes_ask")
            no_bid = _kalshi_price_norm(target_market, "no_bid_dollars", "no_bid")
            last_price = _kalshi_price_norm(target_market, "last_price_dollars", "last_price")
            prob = None
            if yes_bid is not None and yes_ask is not None:
                prob = (yes_bid + yes_ask) / 2.0
            elif yes_bid is not None and no_bid is not None:
                implied_yes_ask = 1.0 - no_bid
                prob = (yes_bid + implied_yes_ask) / 2.0
            elif yes_bid is not None:
                # Prefer last_price (actual trade) over yes_bid (lowest buy offer)
                prob = last_price if (last_price is not None and last_price > 0) else yes_bid
            elif no_bid is not None:
                prob = 1.0 - no_bid
            # Final fallback to last_price
            if prob is None and last_price is not None and last_price > 0:
                prob = last_price

            # Enhanced Match Logging (Task 1)
            target_ticker = target_market.get("ticker")
            target_title = target_market.get("title")
            logger.info(f"🎯 Kalshi Match Selected: {target_ticker}")
            logger.info(f"   Title (Yes Side): {target_title}")
            logger.info(f"   Prob (Raw): {prob if prob is not None else 'None'}")
            if prob is not None and abs(prob - 0.5) < 0.01:
                logger.warning(f"   ⚠️ Neutral probability (0.50) for {target_ticker} - likely default or no data")

            # Enhanced debug info
            debug_info = {
                "score": best_score,
                "event": best_event.get("ticker"),
                "total_markets": len(markets),
                "winner_found": bool(winner_market),
                "spread_count": len(spread_markets),
                "total_count": len(total_markets),
                "spread_tickers": [m.get("ticker") for m in spread_markets[:2]],
                "total_tickers": [m.get("ticker") for m in total_markets[:2]],
                # Store full market objects for spread/total to be processed later
                "spread_markets": spread_markets,
                "total_markets": total_markets,
                "requested_market_type": requested_market_type,
                "matched_market_type": match_reason_detail,
                "is_fallback": "fallback" in (match_reason_detail or "")
            }

            m_type = "winner"
            if "spread" in (match_reason_detail or ""): m_type = "spread"
            elif "total" in (match_reason_detail or ""): m_type = "total"

            return KalshiMatchResult(
                matched=True,
                kalshi_available=True,
                label=target_market.get("title"),
                probability=prob if prob is not None else 0.5,
                raw_event_id=best_event.get("ticker"),
                market_ticker=target_market.get("ticker"),
                league=league,
                reason=match_reason_detail or "matched_via_events_api",
                market_type=m_type,
                game_date=game_dt_utc,
                debug=debug_info
            )

    return None

def _normalize_series_prefix(prefix: Any) -> Tuple[str, ...]:
    """Convert league series prefix to tuple for startswith() matching."""
    if isinstance(prefix, (list, tuple)):
        return tuple(str(p) for p in prefix if p)
    elif prefix:
        return (str(prefix),)
    return ()

def match_game_to_kalshi(league: str, home_team: str, away_team: str, game_time: Optional[datetime], integrator: "KalshiIntegrator" = None, status: Optional[str] = None, requested_market_type: Optional[str] = None) -> KalshiMatchResult:
    league_key = (league or "").upper()
    kalshi = integrator or KalshiIntegrator()

    # --- FEB 15, 2026 OVERRIDES ---
    if game_time:
        try:
            g_date_str = game_time.strftime("%Y-%m-%d")
            # User provided key: "Away@Home@Date" (e.g. "IUPUI Jaguars@Fort Wayne Mastodons@2026-02-15")
            override_key = f"{away_team}@{home_team}@{g_date_str}"

            GAME_OVERRIDES = {
                "IUPUI Jaguars@Fort Wayne Mastodons@2026-02-15": "IUINPFW",
                "Illinois St Redbirds@UIC Flames@2026-02-15": "ILSTUIC",
            }

            if override_key in GAME_OVERRIDES:
                suffix = GAME_OVERRIDES[override_key]
                logger.info(f"⚡ Applying KALSHI OVERRIDE for {override_key} -> {suffix}")

                game_ticker = f"KXNCAAMBGAME-26FEB15{suffix}"
                spread_ticker = f"KXNCAAMBSPREAD-26FEB15{suffix}"
                total_ticker = f"KXNCAAMBTOTAL-26FEB15{suffix}"

                # Fetch real data if possible
                game_event = {}
                spread_markets = []
                total_markets = []

                if kalshi and kalshi.api_key:
                    try:
                        # Fetch Game Event
                        resp = kalshi._request("GET", "/events", params={"event_ticker": game_ticker, "with_nested_markets": True})
                        events = resp.get("events", [])
                        if events:
                            game_event = events[0]

                        # Fetch Spread Markets
                        s_resp = kalshi._request("GET", "/markets", params={"event_ticker": spread_ticker})
                        spread_markets = s_resp.get("markets", [])

                        # Fetch Total Markets
                        t_resp = kalshi._request("GET", "/markets", params={"event_ticker": total_ticker})
                        total_markets = t_resp.get("markets", [])

                        logger.info(f"   Override fetched: {len(spread_markets)} spread, {len(total_markets)} total markets")
                    except Exception as e:
                        logger.warning(f"   Override fetch failed: {e}")

                # Construct Result
                debug_info = {
                    "score": 100,
                    "event": game_ticker,
                    "spread_markets": spread_markets,
                    "total_markets": total_markets,
                    "override_used": True
                }

                # Use game event title if available
                label = game_event.get("title", f"{away_team} @ {home_team}")

                return KalshiMatchResult(
                    matched=True,
                    kalshi_available=True,
                    label=label,
                    probability=0.5, # Default, downstream logic will pick from markets
                    raw_event_id=game_ticker,
                    market_ticker=game_event.get("markets", [{}])[0].get("ticker") if game_event.get("markets") else None,
                    league="NCAAB",
                    reason="manual_override",
                    market_type="winner",
                    game_date=game_time,
                    debug=debug_info
                )
        except Exception as e:
            logger.warning(f"Override check failed: {e}")

    if not kalshi or not kalshi.api_key:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, reason="no_integrator")

    # Use robust candidate generation
    home_clean = clean_team_name(home_team)
    away_clean = clean_team_name(away_team)

    # DEBUG: Log team name cleaning
    logger.info(f"🎯 Kalshi Match Request [{league_key}]: {away_team} @ {home_team}")
    logger.info(f"   Cleaned Names: {away_clean} @ {home_clean}")

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

    # DEBUG: Log generated codes
    logger.info(f"   Mapped Codes: away={mapped_away}, home={mapped_home}")
    logger.info(f"   Full Code Candidates: away={away_codes}, home={home_codes}")

    # Log searching blocks for NCAAB debug
    if league_key == "NCAAB":
        searching_blocks = []
        for ac in away_codes:
            for hc in home_codes:
                searching_blocks.append(f"{ac}{hc}")
        logger.info(f"   Searching Blocks (NCAAB): {searching_blocks[:10]}...")

    # NEW: Try Event-Based Matching First
    if game_time and league_key in ["NBA", "NFL", "NCAAB", "NCAAF", "MLB", "NHL"]:
        # Normalize game_time to UTC
        if game_time.tzinfo is None:
            gt_utc = pytz.utc.localize(game_time)
        else:
            gt_utc = game_time.astimezone(pytz.UTC)

        event_match = _match_via_events(
            kalshi,
            league_key,
            home_codes,
            away_codes,
            gt_utc,
            status=status,
            requested_market_type=requested_market_type,
            home_team_name=home_team,
            away_team_name=away_team
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
    series_prefix_tuple = _normalize_series_prefix(series_prefix)
    best_market = None
    best_score = 0.0

    # Constants for fuzzy logic
    DATE_TOLERANCE_DAYS = 2
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
        if series_prefix_tuple and not ticker.startswith(series_prefix_tuple):
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
        # Clear stale events cache on initialization
        self._events_cache: Dict[str, Dict[str, Any]] = {}  # Cache for /events by series_ticker
        self._events_cache_ttl: int = 300
        logger.info("✅ Kalshi integrator initialized, events cache cleared")
        self.last_error: Optional[str] = None
        self._league_cache: Dict[str, Dict[str, Any]] = {}
        self._league_cache_ttl: int = 300
        self.last_fetch_meta: Dict[str, Any] = {}
        self.session = requests.Session()

        # Add connection pooling for performance
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"]  # Don't retry PUT (not idempotent)
        )
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

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
                logger.info(f"Kalshi API Call: key={'SET' if self.api_key else 'MISSING'}, url={url}")
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
        Enables with_nested_markets=True to get market data in one call.
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
            "with_nested_markets": True, # ENABLED NESTED MARKETS
            "series_ticker": series_ticker,
        }
        if status:
            params["status"] = normalize_status(status)
        if min_close_ts:
            params["min_close_ts"] = int(min_close_ts)

        params = {k: v for k, v in params.items() if v is not None}

        try:
            resp = self._request("GET", "/events", params=params)

            # DIAGNOSTIC: Log raw response structure
            events = resp.get("events", [])
            if events:
                sample_event = events[0]
                logger.info(f"🔍 KALSHI /events RAW RESPONSE SAMPLE:")
                logger.info(f"   Keys in first event: {list(sample_event.keys())}")
                logger.info(f"   Ticker value: {sample_event.get('ticker')}")
                logger.info(f"   Ticker type: {type(sample_event.get('ticker'))}")

                # Check for nested ticker
                if 'event' in sample_event:
                    logger.info(f"   Nested 'event' found: {list(sample_event['event'].keys())}")
                    logger.info(f"   Nested ticker: {sample_event['event'].get('ticker')}")

            # VALIDATION: Filter out events with null/invalid tickers
            valid_events = []
            invalid_count = 0
            for evt in events:
                ticker = evt.get("ticker")

                # Check if ticker is in nested 'event' object (API v2 structure)
                if not ticker and isinstance(evt.get("event"), dict):
                    ticker = evt["event"].get("ticker")
                    if ticker:
                        # Flatten: Copy ticker to top level
                        evt["ticker"] = ticker

                # Validate ticker
                if ticker and ticker != "None" and isinstance(ticker, str) and len(ticker) > 5:
                    valid_events.append(evt)
                else:
                    invalid_count += 1
                    if invalid_count <= 3:  # Log first 3 invalid for diagnosis
                        logger.warning(f"⚠️ Invalid event ticker: {evt.get('id', 'unknown')} → ticker={ticker}")

            # Replace with validated list
            if valid_events:
                resp["events"] = valid_events
                logger.info(f"✅ Validated {len(valid_events)}/{len(events)} events (filtered {invalid_count} invalid)")
            elif events:
                logger.error(f"❌ ALL {len(events)} events had invalid tickers! Clearing cache.")
                # Clear corrupted cache
                if cache_key in self._events_cache:
                    del self._events_cache[cache_key]
                # Return empty to avoid cascading failures
                return {"events": [], "cursor": None}

        except Exception:
            # If rate limited or error, return cached if available
            cached = self._events_cache.get(cache_key)
            if use_cache and cached:
                 logger.warning(f"Kalshi get_events failed for {series_ticker}, using cache.")
                 return cached.get("payload", {})
            raise

        if use_cache and not cursor and resp and resp.get("events"):
            self._events_cache[cache_key] = {"ts": now, "payload": resp}

        return resp

    def clear_events_cache(self) -> Dict[str, Any]:
        """Manually clear events cache (useful for debugging API changes)."""
        count = len(self._events_cache)
        self._events_cache.clear()
        logger.info(f"🗑️ Cleared {count} cached event entries")
        return {"cleared": count, "status": "ok"}

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
        max_pages: int = 50,
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

    def get_league_markets(
        self,
        league: str,
        *,
        status: Optional[str] = None,
        min_prefix_hits: int = 200,
        max_pages: int = 50,
    ) -> List[Dict[str, Any]]:
        league_key = (league or "").upper()
        prefix = LEAGUE_SERIES_MAP.get(league_key)
        normalized_status = normalize_status(status)
        cache_key = f"{league_key}:{normalized_status or 'any'}:{max_pages}"
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

            # v104 FIX: Log per-series fetch count for pagination diagnostics
            logger.info(f"KALSHI POOL: series={series} fetched={len(chunk)} markets (max_pages={max_pages})")

            # --- DEBUG LOGGING (Task 2) ---
            if chunk:
                logger.info(f"Kalshi debug: fetched {len(chunk)} markets for series={series}")
                sample_tickers = [str(m.get("ticker")) for m in chunk[:10]]
                logger.info(f"Kalshi debug sample tickers: {sample_tickers}")

                # Count by prefix
                prefix_counts = {}
                for m in chunk:
                    t = str(m.get("ticker") or "").upper()
                    p = t.split('-')[0] if '-' in t else t
                    prefix_counts[p] = prefix_counts.get(p, 0) + 1
                logger.info(f"Kalshi debug prefix counts: {prefix_counts}")
            # ------------------------------

            pages = max(pages, min(max_pages, len(chunk) // 200 + 1))
            for m in chunk or []:
                # FIX: Dedup by individual market ticker, NOT event_ticker
                # event_ticker is shared by all markets in an event (Over/Under both have same event_ticker)
                key = str(m.get("ticker") or "").upper()
                if key and key not in collected:
                    collected[key] = m

        all_markets = list(collected.values())
        logger.info(f"Kalshi get_league_markets: {len(all_markets)} unique markets after ticker-level dedup")

        # --- DIAGNOSTIC: dump all unique team blocks from NCAAB tickers ---
        # Only run if DEBUG logging is enabled (performance optimization)
        if logger.isEnabledFor(logging.DEBUG) and league_key in ("NCAAB", "NCAAF"):
            _diag_blocks: set = set()
            _date_re = re.compile(r"\d{2}[A-Z]{3}\d{2}")
            for _m in all_markets:
                _t = str(_m.get("event_ticker") or _m.get("ticker") or "").upper()
                _parts = _t.split("-")
                if len(_parts) >= 2:
                    _suffix = "-".join(_parts[1:])
                    _dm = _date_re.match(_suffix)
                    if _dm:
                        _block = _suffix[_dm.end():]
                        if _block:
                            _diag_blocks.add(_block)
            logger.info(
                f"🔍 KALSHI ALL {league_key} TEAM BLOCKS ({len(_diag_blocks)}): "
                f"{sorted(_diag_blocks)}"
            )
        # --- END DIAGNOSTIC ---

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

        # FIX: Do NOT replace all_markets with game_markets here!
        # This was discarding TOTAL and SPREAD markets for NBA/NHL.
        # We need ALL market types (GAME, TOTAL, SPREAD) in the pool.
        # The filtering for specific game types happens in streamlit_app.py's game_pool.

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
            "includes_all_market_types": True,  # FIX: Now includes GAME, TOTAL, SPREAD
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

        # FIX: Log market type breakdown to verify TOTAL/SPREAD are included
        game_count = len([m for m in all_markets if "GAME" in str(m.get("ticker", "")).upper()])
        total_count = len([m for m in all_markets if "TOTAL" in str(m.get("ticker", "")).upper()])
        spread_count = len([m for m in all_markets if "SPREAD" in str(m.get("ticker", "")).upper()])
        logger.info(f"✅ get_league_markets returning {len(all_markets)} markets for {league_key}: GAME={game_count}, TOTAL={total_count}, SPREAD={spread_count}")

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

    def _date_to_kalshi_token(self, dt: datetime) -> str:
        """
        Convert datetime to Kalshi date token format (YYMONDD).
        e.g. 2025-01-26 -> 25JAN26
        """
        # Ensure UTC
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        else:
            dt = dt.astimezone(pytz.UTC)

        # Format: %y%b%d, but Month needs to be UPPERCASE
        token = dt.strftime("%y%b%d").upper()
        return token

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

            # --- FIXED: Use per-game date tokens for batch fetching ---
            # Instead of one "min" date key, we iterate all unique date tokens
            # and fetch/combine markets. If no times provided, fall back to league fetch.

            # 1. Identify unique date tokens involved in the batch
            unique_tokens = set()
            if commence_dt:
                for dt in commence_dt:
                    token = self._date_to_kalshi_token(dt)
                    unique_tokens.add(token)

            # 2. Fetch markets
            all_filtered_markets = []

            if unique_tokens:
                # Optimized Fetch: For each date token, get specific markets
                # This fixes the "batch date key" bug where it only fetched the first date
                for token in unique_tokens:
                    # Uses internal caching inside get_markets_for_date_token
                    # This method handles "YYMONDD" tokens correctly
                    result = self.get_markets_for_date_token(league_key, token)
                    markets = result.get("bucket", [])
                    # Also include broadened results if bucket empty?
                    # get_markets_for_date_token does fallback internally.
                    all_filtered_markets.extend(markets)

                # De-duplicate results across tokens (just in case)
                deduped = {str(m.get("ticker")): m for m in all_filtered_markets}
                all_filtered_markets = list(deduped.values())

            else:
                # Fallback: Fetch entire league if no specific times provided
                # (Existing logic, but simplified)
                markets = self.get_league_markets(league_key, status=None)
                all_filtered_markets = markets

            # 3. Final Filtering by Time Window (only if times were provided)
            # This ensures even if we fetched extra markets for a token, we narrow down
            # to the specific games requested (±72h window)
            filtered = all_filtered_markets
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

            # Cache the result?
            # If we used unique_tokens logic, caching is handled per token inside get_markets_for_date_token.
            # We don't necessarily need to cache the *batch* result unless it's frequent.
            # The original code cached by (league, min_date). We can keep that for backward compat
            # or just rely on the granular token cache.
            # Given the prompt's focus on "per-game datekey", relying on token cache is better.

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
                    max_pages=50,
                    extra_params={"event_ticker_prefix": targeted_prefix},
                )
            except Exception:
                targeted = []
            try:
                extra = self.get_markets_paginated(
                    status=status, limit=200, max_pages=50
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

def generate_missing_games_report(
    games_attempted: List[Dict[str, Any]],
    matched_tickers: List[str],
    kalshi_events: List[Dict[str, Any]]
) -> str:
    """
    Generate a report of games that failed to match with suggested aliases.

    Args:
        games_attempted: List of game dicts with 'home', 'away', 'league' keys
        matched_tickers: List of event_tickers that matched successfully
        kalshi_events: All Kalshi events fetched for comparison

    Returns:
        Markdown report string
    """
    report = ["# Kalshi Matching Diagnostic Report\n"]
    report.append(f"**Games Attempted:** {len(games_attempted)}")
    report.append(f"**Successful Matches:** {len(matched_tickers)}")
    report.append(f"**Failed Matches:** {len(games_attempted) - len(matched_tickers)}\n")

    report.append("## Failed Games (Needing Aliases)\n")

    for game in games_attempted:
        home = game.get('home', '')
        away = game.get('away', '')
        league = game.get('league', '')

        # Check if this game matched
        # (This requires you to pass in match status, which you'll add)
        matched = game.get('kalshi_matched', False)

        if not matched:
            report.append(f"### {away} @ {home} ({league})")

            # Show generated codes
            home_codes = _build_team_codes(home)
            away_codes = _build_team_codes(away)

            report.append(f"- **Generated Home Codes:** {home_codes[:5]}")
            report.append(f"- **Generated Away Codes:** {away_codes[:5]}")

            # Find potential Kalshi events for this date
            game_date = game.get('commence_time')
            if game_date:
                potential_events = [
                    evt for evt in kalshi_events
                    if league.upper() in evt.get('ticker', '').upper()
                ]

                if potential_events:
                    report.append(f"- **Potential Kalshi Events ({len(potential_events)}):**")
                    for evt in potential_events[:5]:
                        ticker = evt.get('ticker', '')
                        parsed = parse_event_ticker_codes(ticker)
                        report.append(f"  - `{ticker}` → home={parsed.get('home')}, away={parsed.get('away')}")

            report.append("")  # Blank line

    return "\n".join(report)
