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
    "generate_comprehensive_team_variants",
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


def generate_comprehensive_team_variants(team_name: str, league: str = None) -> List[str]:
    """
    Generate comprehensive team name variants for Kalshi matching.
    
    This function produces a wide range of variations to maximize matching success:
    - Full cleaned name
    - Individual tokens (words)
    - Initials
    - Common abbreviations
    - Partial matches (2-4 char prefixes)
    - League-specific mapped codes
    
    Args:
        team_name: Original team name (e.g., "Golden State Warriors")
        league: Optional league identifier for specialized mapping
        
    Returns:
        List of unique variant strings, prioritized by likelihood
    """
    if not team_name:
        return []
    
    variants = []
    
    # Step 1: Clean the base name
    cleaned = clean_team_name(team_name)
    if cleaned:
        variants.append(cleaned)
    
    # Step 2: Try league-specific mapping first (highest priority)
    if league:
        mapped = team_name_to_code(league, team_name)
        if mapped and mapped != "UNK":
            variants.insert(0, mapped)
    
    # Step 3: Extract tokens (words)
    tokens = [t for t in cleaned.split() if t and len(t) >= 2]
    
    # Add full tokens
    variants.extend(tokens)
    
    # Step 4: Generate initials (e.g., GSW from Golden State Warriors)
    if len(tokens) >= 2:
        initials = "".join(t[0] for t in tokens)
        if len(initials) >= 2:
            variants.append(initials)
            # Also add 2-char and 3-char versions
            if len(initials) >= 3:
                variants.append(initials[:3])
            if len(initials) >= 2:
                variants.append(initials[:2])
    
    # Step 5: Add prefixes of each token (2-4 chars)
    for token in tokens:
        if len(token) >= 4:
            variants.append(token[:4])
        if len(token) >= 3:
            variants.append(token[:3])
        if len(token) >= 2:
            variants.append(token[:2])
    
    # Step 6: Check common abbreviations dictionary
    if cleaned in KALSHI_TEAM_ABBREVIATIONS:
        variants.extend(KALSHI_TEAM_ABBREVIATIONS[cleaned])
    
    # Step 7: For college teams, try without mascot
    if league and league.upper() in ["NCAAB", "NCAAF"]:
        stripped = strip_mascot(team_name)
        if stripped != team_name:
            stripped_clean = clean_team_name(stripped)
            if stripped_clean:
                variants.append(stripped_clean)
                # Also add tokens from stripped version
                stripped_tokens = [t for t in stripped_clean.split() if t and len(t) >= 2]
                variants.extend(stripped_tokens)
    
    # Step 8: Try KALSHI_NCAAB_TEAM_CODES mapping
    if league and league.upper() == "NCAAB":
        # Try direct lookup
        if team_name in KALSHI_NCAAB_TEAM_CODES:
            variants.insert(0, KALSHI_NCAAB_TEAM_CODES[team_name])
        
        # Try without last word (mascot)
        parts = team_name.split()
        if len(parts) > 1:
            without_last = " ".join(parts[:-1])
            if without_last in KALSHI_NCAAB_TEAM_CODES:
                variants.insert(0, KALSHI_NCAAB_TEAM_CODES[without_last])
    
    # Step 9: Deduplicate while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        v_upper = v.upper()
        if v_upper and v_upper not in seen:
            seen.add(v_upper)
            unique_variants.append(v_upper)
    
    logger.debug(f"Generated {len(unique_variants)} variants for '{team_name}': {unique_variants[:10]}...")
    
    return unique_variants


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


@lru_cache(maxsize=4096)
def clean_team_name(name: str) -> str:
    """Robust cleaning preserving spaces for map lookup."""
    # Convert to uppercase, replace non-alphanumeric with space, collapse multiple spaces
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", str(name or "").upper())
    # Collapse multiple spaces into one and strip
    return re.sub(r"\s+", " ", cleaned).strip()

def strip_mascot(team_name: str) -> str:
    """Remove common college mascots from team names for code lookup."""
    # Common mascots to remove
    mascots = ['Wildcats', 'Bulldogs', 'Bears', 'Eagles', 'Tigers', 'Cardinals',
               'Warriors', 'Knights', 'Spartans', 'Huskies', 'Panthers', 'Cougars',
               'Red Storm', 'Blue Devils', 'Tar Heels', 'Musketeers', 'Wolfpack']

    for mascot in mascots:
        if team_name.endswith(mascot):
            return team_name[:-len(mascot)].strip()

    # Fallback: Remove last word if team name has 2+ words
    parts = team_name.split()
    if len(parts) >= 2:
        return ' '.join(parts[:-1])

    return team_name


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

# Skeleton for college – fill with your existing mappings
NCAAF_TEAM_CODE_MAP: Dict[str, str] = {
    "FLORIDA STATE": "FSU",
}

# [Truncated for brevity - include full NCAAB_TEAM_CODE_MAP, KALSHI_NCAAB_TEAM_CODES, aliases from original file]
# Due to length constraints, copy the complete maps from lines ~500-1500 of your original file
NCAAB_TEAM_CODE_MAP: Dict[str, str] = {}  # Add full map from original
KALSHI_NCAAB_TEAM_CODES = {}  # Add full map from original
NCAAB_CODE_ALIASES: Dict[str, str] = {}  # Add full map from original
NCAAF_CODE_ALIASES: Dict[str, str] = {}  # Add full map from original

# [Continue with remaining functions - resolve_team_code, team_name_to_code, etc.]
# Due to response length limits, I'll indicate where to copy the rest

def normalize_team_for_kalshi(team_name: str) -> str:
    """Convert full team name to Kalshi 4-letter code with enhanced normalization"""
    # [Copy implementation from original file]
    pass

def resolve_team_code(code: str, league: str) -> str:
    """Resolve a team code to its canonical form using alias maps"""
    # [Copy implementation from original file]
    pass

def team_name_to_code(league: str, team_name: str) -> Optional[str]:
    """Translate a full team name into its Kalshi ticker code"""
    # [Copy implementation from original file]
    pass

def team_code_for_league(league: str, team_name: str) -> str:
    """Return a non-empty ticker-friendly code for a team within a league"""
    # [Copy implementation from original file]
    pass

def cross_reference_unmapped_ticker(league: str, date_token: str, team_block: str) -> Optional[Dict[str, str]]:
    """Uses SportsDataIO to find games on the date and match the team block"""
    # [Copy implementation from original file]
    pass

# [Copy remaining utility functions and KalshiIntegrator class from original file]
# Lines ~1500-5000+

# NOTE: Due to the 65536 character limit, I cannot include the entire file.
# Please manually:
# 1. Add the new generate_comprehensive_team_variants() function (shown above)
# 2. Update _build_team_codes() to use the new function
# 3. Keep all existing team maps, aliases, and functions intact
