"""
TheOver.ai Public Betting Ingestion Module

Parses raw text from TheOver.ai public betting picks and normalizes it for integration.
Supports parsing from Excel uploads (Totals/Sides) and text paste fallbacks.
Now uses Canonical Keys for reliable matching via Advanced Fuzzy Matching.
"""
import pandas as pd
import re
import logging
from typing import Tuple, Optional, Dict, List, Any, Union
from datetime import datetime
from zoneinfo import ZoneInfo
from app_core.feature_processing import robust_normalize_team
from app_core.kalshi_integrator import team_code_for_league
from app_core.team_name_matcher import TeamNameMatcher

# Assuming rapidfuzz is available as per requirements
try:
    import rapidfuzz
    from rapidfuzz import process, fuzz
except ImportError:
    rapidfuzz = None
    process = None
    fuzz = None
    logging.getLogger("app_core.theover_ingest").warning("rapidfuzz not installed. Fuzzy matching will be disabled.")

logger = logging.getLogger("app_core.theover_ingest")

# League-specific team alias mappings to prevent cross-league contamination
TEAM_ALIAS_MAP_BY_LEAGUE = {
    "NHL": {
        "Seattle": "Seattle Kraken", "New Jersey": "New Jersey Devils",
        "Buffalo": "Buffalo Sabres", "Philadelphia": "Philadelphia Flyers",
        "Ottawa": "Ottawa Senators", "Winnipeg": "Winnipeg Jets",
        # Issue 3: Explicit mappings for TheOdds API mismatches
        "Toronto": "Toronto Maple Leafs", "Florida": "Florida Panthers",
        "Vancouver": "Vancouver Canucks", "Montreal": "Montreal Canadiens",
        "NY Rangers": "New York Rangers", "NY Islanders": "New York Islanders",
        "SJ": "San Jose Sharks", "LA": "Los Angeles Kings",
        "St Louis": "St Louis Blues", "Tampa Bay": "Tampa Bay Lightning",
        "Vegas": "Vegas Golden Knights", "Washington": "Washington Capitals",
        "Arizona": "Arizona Coyotes", "Anaheim": "Anaheim Ducks",
        "Boston": "Boston Bruins", "Calgary": "Calgary Flames",
        "Carolina": "Carolina Hurricanes", "Chicago": "Chicago Blackhawks",
        "Colorado": "Colorado Avalanche", "Columbus": "Columbus Blue Jackets",
        "Dallas": "Dallas Stars", "Detroit": "Detroit Red Wings",
        "Edmonton": "Edmonton Oilers", "Minnesota": "Minnesota Wild",
        "Nashville": "Nashville Predators", "Pittsburgh": "Pittsburgh Penguins",
    },
    "NBA": {
        "New York": "New York Knicks", "Sacramento": "Sacramento Kings",
        "LA": "Los Angeles Lakers", "PHI": "Philadelphia 76ers",
        "GSW": "Golden State Warriors", "LAL": "Los Angeles Lakers",
        "LAC": "Los Angeles Clippers", "BKN": "Brooklyn Nets",
        "OKC": "Oklahoma City Thunder", "NOP": "New Orleans Pelicans",
        "SAS": "San Antonio Spurs", "UTA": "Utah Jazz",
        "WAS": "Washington Wizards", "CHA": "Charlotte Hornets",
        "PHX": "Phoenix Suns", "MIL": "Milwaukee Bucks",
        "MEM": "Memphis Grizzlies", "IND": "Indiana Pacers",
        "DET": "Detroit Pistons", "ATL": "Atlanta Hawks",
        "BOS": "Boston Celtics", "CHI": "Chicago Bulls",
        "CLE": "Cleveland Cavaliers", "MIA": "Miami Heat",
        "ORL": "Orlando Magic", "TOR": "Toronto Raptors",
        # Issue 3: Explicit City -> Full Name mappings for TheOdds API mismatches
        "Denver": "Denver Nuggets",
        "Chicago": "Chicago Bulls",
        "Houston": "Houston Rockets",
        # Issue #1: Fix Contamination
        "Atlanta": "Atlanta Hawks",
        "Dallas": "Dallas Mavericks",
        "Detroit": "Detroit Pistons",
        "Washington": "Washington Wizards",
        "Miami": "Miami Heat",
        "Minnesota": "Minnesota Timberwolves",
        "Phoenix": "Phoenix Suns",
        "Charlotte": "Charlotte Hornets",
        "Cleveland": "Cleveland Cavaliers",
        "Indiana": "Indiana Pacers",
        "Milwaukee": "Milwaukee Bucks",
        "Brooklyn": "Brooklyn Nets",
        "Orlando": "Orlando Magic",
        "Philadelphia": "Philadelphia 76ers",
        "Portland": "Portland Trail Blazers",
        "San Antonio": "San Antonio Spurs",
        "Utah": "Utah Jazz",
    },
    "NFL": {
        "Arizona": "Arizona Cardinals", "Atlanta": "Atlanta Falcons",
        "Baltimore": "Baltimore Ravens", "Buffalo": "Buffalo Bills",
        "Carolina": "Carolina Panthers", "Chicago": "Chicago Bears",
        "Cincinnati": "Cincinnati Bengals", "Cleveland": "Cleveland Browns",
        "Dallas": "Dallas Cowboys", "Denver": "Denver Broncos",
        "Detroit": "Detroit Lions", "Green Bay": "Green Bay Packers",
        "Houston": "Houston Texans", "Indianapolis": "Indianapolis Colts",
        "Jacksonville": "Jacksonville Jaguars", "Kansas City": "Kansas City Chiefs",
        "Las Vegas": "Las Vegas Raiders", "LA": "Los Angeles Rams",
        "Miami": "Miami Dolphins", "Minnesota": "Minnesota Vikings",
        "New England": "New England Patriots", "New Orleans": "New Orleans Saints",
        "NY Giants": "New York Giants", "NY Jets": "New York Jets",
        "Philadelphia": "Philadelphia Eagles", "Pittsburgh": "Pittsburgh Steelers",
        "San Francisco": "San Francisco 49ers", "Seattle": "Seattle Seahawks",
        "Tampa Bay": "Tampa Bay Buccaneers", "Tennessee": "Tennessee Titans",
        "Washington": "Washington Commanders",
        # Issue 3: Explicit City -> Full Name mappings for TheOdds API mismatches
        "Denver": "Denver Broncos",
        "Chicago": "Chicago Bears",
        "Houston": "Houston Texans",
        "New England": "New England Patriots",
    },
    "NCAAB": {
        "South Carolina": "South Carolina Gamecocks", "Arkansas": "Arkansas Razorbacks",
        "Pittsburgh": "Pittsburgh Panthers", "Georgia Tech": "Georgia Tech Yellow Jackets",
        "Missouri": "Missouri Tigers", "Ole Miss": "Ole Miss Rebels",
        "Mississippi St": "Mississippi State Bulldogs", "Vanderbilt": "Vanderbilt Commodores",
        "LSU": "LSU Tigers", "Kansas": "Kansas Jayhawks", "Kansas St": "Kansas State Wildcats",
        "Illinois": "Illinois Fighting Illini", "Ohio St": "Ohio State Buckeyes",
        "Boston Univ": "Boston Univ. Terriers", "Western Carolina": "Western Carolina Catamounts",
        "Chattanooga": "Chattanooga Mocs", "NC State": "NC State Wolfpack",
        "Iowa": "Iowa Hawkeyes", "Iowa St": "Iowa State Cyclones",
        "Michigan": "Michigan Wolverines", "Michigan St": "Michigan State Spartans",
        "Penn St": "Penn State Nittany Lions", "Texas A&M": "Texas A&M Aggies",
        "Virginia": "Virginia Cavaliers", "Virginia Tech": "Virginia Tech Hokies",
        "West Virginia": "West Virginia Mountaineers", "Florida St": "Florida State Seminoles",
        "Miami (FL)": "Miami Hurricanes", "Miami": "Miami Hurricanes",
        "Kentucky": "Kentucky Wildcats", "Tennessee": "Tennessee Volunteers",
        "Auburn": "Auburn Tigers", "Alabama": "Alabama Crimson Tide",
        "Arizona St": "Arizona State Sun Devils", "Oregon": "Oregon Ducks",
        "Oregon St": "Oregon State Beavers", "UCLA": "UCLA Bruins",
        "USC": "USC Trojans", "Washington St": "Washington State Cougars",
        "Colorado St": "Colorado State Rams", "San Diego St": "San Diego State Aztecs",
        "Boise St": "Boise State Broncos", "Nevada": "Nevada Wolf Pack",
        "UNLV": "UNLV Rebels", "Utah St": "Utah State Aggies",
        "Duke": "Duke Blue Devils", "North Carolina": "North Carolina Tar Heels",
        "Gonzaga": "Gonzaga Bulldogs", "Villanova": "Villanova Wildcats",
        # Issue #2 & #3 Fixes
        "UMBC": "UMBC Retrievers",
        "UUMC": "UMBC Retrievers",
        "UAB": "UAB Blazers",
        "UAB BLAZERS": "UAB Blazers",
        "Uab": "UAB Blazers",
        "BINGHAMTON": "Binghamton Bearcats",
        "Saint Joseph's": "Saint Joseph's Hawks",
        "LIU": "LIU Sharks",
        # Fix #5: Team Name Mapping Issues
        "New Orleans": "New Orleans Privateers",
        "NEW ORLEANS": "New Orleans Privateers",
        "FIU": "Florida Int'l Golden Panthers",
        "Sacramento State": "Sacramento St Hornets",
    },
    "NCAAF": {
        # Add NCAAF specific mappings if needed
        "Alabama": "Alabama Crimson Tide", "Georgia": "Georgia Bulldogs",
        "Ohio St": "Ohio State Buckeyes", "Michigan": "Michigan Wolverines",
        "USC": "USC Trojans", "Texas": "Texas Longhorns",
        "Oklahoma": "Oklahoma Sooners", "Florida": "Florida Gators",
        "UAB": "UAB Blazers",
        "Uab": "UAB Blazers",
        "FIU": "Florida Int'l Golden Panthers",
        "Sacramento State": "Sacramento St Hornets",
    }
}

# Backward compatibility: flatten all into single map for non-league-aware lookups
TEAM_ALIAS_MAP = {}
for league_aliases in TEAM_ALIAS_MAP_BY_LEAGUE.values():
    TEAM_ALIAS_MAP.update(league_aliases)

def generate_canonical_key(league: str, date_str: str, home_code: str, away_code: str) -> str:
    """
    Generates a canonical key for matching against the master schedule.
    Format: {league}|{home_code}|{away_code}|{local_date}
    """
    return f"{league}|{home_code}|{away_code}|{date_str}"

def load_theover_file(uploaded_file):
    """
    Robust file loader that attempts Excel first, then CSV.
    Never attempts UTF-8 decode on XLSX to prevent binary errors.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    # Reset pointer if possible
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    # 1. Try Excel (openpyxl)
    try:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        # 2. Try CSV
        try:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            # Ensure messy files are handled with skip_blank_lines and on_bad_lines
            # Enforce utf-8-sig to handle BOM if present (Task 1: Constraint met)
            return pd.read_csv(uploaded_file, on_bad_lines='skip', skip_blank_lines=True, encoding='utf-8-sig')
        except Exception:
            return pd.DataFrame()

def _parse_block_layout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses "block" style spreadsheets where games are listed in blocks.
    Identifies games by looking for the @ symbol in text lines rather than just row headers.
    Implements a scan window logic (8-10 lines) to find disconnected lines/picks.
    """
    records = []

    # Flatten to list of strings
    rows = df.astype(str).values.tolist()
    # Flatten structure: list of text lines (joining columns)
    text_lines = []
    for r in rows:
        # Join non-empty cells
        line = " ".join([x for x in r if x and x.lower() != 'nan' and x.lower() != 'none']).strip()
        if line:
            text_lines.append(line)

    i = 0
    while i < len(text_lines):
        line = text_lines[i]

        # 1. Find Game Line (contains "@")
        if "@" in line:
            parts = line.split("@")
            if len(parts) >= 2:
                away_raw = parts[0].strip()
                # Clean home team (remove parens etc)
                home_raw = parts[1].split('(')[0].split('vs')[0].strip()

                # We found a game. Now scan next 10 lines for Pick info.
                found_pick = False

                # Look ahead window
                for j in range(1, 11):
                    if i + j >= len(text_lines):
                        break

                    next_line = text_lines[i + j]

                    # Regex for Over/Under
                    # Matches "Over 45.5" or just "45.5" then "Over" logic would require state,
                    # but typically block format has "Over 55.5" or "55.5 Over" or similar.
                    ou_match = re.search(r'(Over|Under)\s*(\d+\.?\d*)', next_line, re.IGNORECASE)
                    if ou_match:
                        records.append({
                            "HOMETEAM": home_raw,
                            "AWAYTEAM": away_raw,
                            "PICK": f"{ou_match.group(1)} {ou_match.group(2)}",
                            "MARKET": "TOTAL",
                            "LEAGUE": "UNKNOWN"
                        })
                        found_pick = True
                        break # Found the total for this game

        i += 1

    return pd.DataFrame(records)

def parse_theover_csv(uploaded_file) -> pd.DataFrame:
    """
    Unified parser that handles:
    1. Standard Table (TotalsRaw.csv, Table1.csv)
    2. Vertical/Semi-structured layouts (Sheet1.csv)
    3. Header-agnostic Column Mapping
    """
    df = load_theover_file(uploaded_file)
    if df.empty:
        return df

    # Normalize Headers: Uppercase, Strip
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Priorities for mapping to ensure correct assignment
    # (Target, List of Keywords)
    # Task 1: Explicitly include requested aliases
    mappings = [
        ("LEAGUE", ["LEAGUE"]),
        ("HOMETEAM", ["HOMETEAM", "HOME", "TEAM1", "TEAM 1", "HOMETEAM"]),
        ("AWAYTEAM", ["AWAYTEAM", "AWAY", "TEAM2", "TEAM 2", "AWAYTEAM"]),
        ("WINPROBABILITY", ["WINPROBABILITY", "PROB", "HITRATE", "SCORE"]),
        ("PICK", ["PICK"]),
        ("COMMENCE", ["COMMENCE", "DATE", "TIME", "START"])
    ]

    # Robust Coalescing Logic:
    # Identify all columns that match keywords for a target, then coalesce them.
    # This handles mixed-schema files or files with alternative headers.

    for target, keywords in mappings:
        matching_cols = []
        for col in df.columns:
            # Check for keyword match
            if any(k in col for k in keywords):
                matching_cols.append(col)

        if matching_cols:
            # Combine first found column with others as fallback
            combined = df[matching_cols[0]]
            for other_col in matching_cols[1:]:
                combined = combined.combine_first(df[other_col])

            # Assign to standardized target column
            df[target] = combined

    # Check for Standard Format (post-mapping)
    required_std = {"HOMETEAM", "AWAYTEAM", "PICK"}

    # Validation Logic: Only return if at least 2 key columns are present.
    # If standard columns are missing, we MUST fallback to block parser.
    if len(required_std.intersection(set(df.columns))) >= 2:
        # Ensure 'LEAGUE' column exists
        if "LEAGUE" not in df.columns:
            df["LEAGUE"] = "UNKNOWN"
        return df

    # Fallback: Block Parser for Sheet1 style (Triggered if <2 standard cols found)
    parsed_block = _parse_block_layout(df)
    if not parsed_block.empty:
        return parsed_block

    return pd.DataFrame()

def _normalize_league_str(raw_league: str) -> str:
    """Normalize league strings to standard keys (NBA, NFL, etc)."""
    raw_league = str(raw_league).strip().upper()
    if "NBA" in raw_league: return "NBA"
    if "NFL" in raw_league: return "NFL"
    if "NHL" in raw_league: return "NHL"
    if "MLB" in raw_league: return "MLB"
    if any(x in raw_league for x in ["NCAAB", "CBB", "COLLEGE BASKETBALL"]): return "NCAAB"
    if any(x in raw_league for x in ["NCAAF", "CFB", "COLLEGE FOOTBALL"]): return "NCAAF"
    return raw_league

def _resolve_team_alias(name: str, league: str) -> str:
    """
    Resolve team alias with league-specific context.
    Uses league-specific mappings to prevent cross-sport contamination.

    CRITICAL: Only uses league-specific map when league is known.
    NO FALLBACK to global map to prevent cross-league contamination.
    """
    if not isinstance(name, str):
        return str(name)

    # Fix: Strip quotes and whitespace
    name = name.strip().strip('"').strip("'").strip()

    # Normalize league string
    league_norm = _normalize_league_str(league)

    # 1. League-Specific Lookup (PRIORITY)
    # LEAGUE-LOCK: Only use league-specific map when league is known
    if league_norm != "UNKNOWN" and league_norm in TEAM_ALIAS_MAP_BY_LEAGUE:
        league_map = TEAM_ALIAS_MAP_BY_LEAGUE[league_norm]

        # Exact match (case-sensitive)
        if name in league_map:
            return league_map[name]

        # Case-insensitive match
        for k, v in league_map.items():
            if k.lower() == name.lower():
                return v

        # LEAGUE-LOCK: If league is known but no match found, return original
        # DO NOT fall back to global map to prevent cross-league contamination
        return name

    # 2. Fallback to Global Map ONLY if league is UNKNOWN
    if league_norm == "UNKNOWN":
        if name in TEAM_ALIAS_MAP:
            return TEAM_ALIAS_MAP[name]

        # 3. Case-insensitive global lookup
        for k, v in TEAM_ALIAS_MAP.items():
            if k.lower() == name.lower():
                return v

    # 4. Return original if no match
    return name

def _validate_team_for_league(team_name: str, league: str) -> bool:
    """
    Strict validation to catch cross-sport contamination.
    Returns True if valid (or unknown), False if clearly invalid.
    Implements cross-league checks for NBA, NFL, NHL, NCAAB, NCAAF.
    """
    league = _normalize_league_str(league)
    team_upper = team_name.upper()

    # Cross-League Contamination Prevention
    # Check if team_name matches a team from a DIFFERENT league
    if league in TEAM_ALIAS_MAP_BY_LEAGUE:
        # Get all teams from OTHER leagues
        for other_league, other_teams in TEAM_ALIAS_MAP_BY_LEAGUE.items():
            if other_league == league:
                continue  # Skip same league

            # Check if team_name matches any FULL TEAM NAME from other league
            # CRITICAL FIX: Only check full names, NOT aliases (aliases are often shared city names)
            # This prevents false positives like "Boston" (NBA) matching "Boston" (NHL alias)
            for alias, full_name in other_teams.items():
                if team_upper == full_name.upper():
                    # Exact match to a full team name from a different league
                    logger.warning(f"Cross-League Contamination: '{team_name}' matches {other_league} team '{full_name}' but league is {league}")
                    return False

    # Legacy token-based validation for additional safety
    if league == "NBA":
        invalid_tokens = ["FALCONS", "COWBOYS", "BRUINS", "LIONS", "HURRICANES", "COMMANDERS", "EAGLES", "GIANTS", "SAINTS", "PRIVATEERS"]
        for token in invalid_tokens:
            if token in team_upper:
                logger.warning(f"Invalid token '{token}' found in NBA team name '{team_name}'")
                return False
    elif league == "NFL":
        invalid_tokens = ["PELICANS", "WARRIORS", "LAKERS", "CELTICS", "KNICKS", "PRIVATEERS"]
        for token in invalid_tokens:
            if token in team_upper:
                logger.warning(f"Invalid token '{token}' found in NFL team name '{team_name}'")
                return False
    elif league == "NCAAB":
        invalid_tokens = ["PELICANS", "SAINTS", "FALCONS", "COWBOYS"]
        for token in invalid_tokens:
            if token in team_upper:
                logger.warning(f"Invalid token '{token}' found in NCAAB team name '{team_name}'")
                return False

    return True

def _transform_theover_df(df: pd.DataFrame, pick_type_default: str, games: List[Dict[str, Any]], stats_collector: List[Dict]) -> pd.DataFrame:
    """
    Transforms the parsed TheOver dataframe into standardized records.
    Uses 'games' (Master Schedule) for Advanced Fuzzy Matching.
    """
    if df.empty:
        return pd.DataFrame()

    records = []
    # Use the current slate date in US/Eastern timezone to match OddsAPI date handling
    slate_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    logger.info(f"Transforming TheOver DataFrame ({pick_type_default}) with {len(df)} rows.")

    # Pre-process master schedule for matching
    master_teams_norm_map = {}
    teams_by_league: Dict[str, List[str]] = {}

    # Store tuples for robust matching fallback
    game_tuples_by_league: Dict[str, List[Tuple[str, str, Dict]]] = {}

    if games:
        for g in games:
            h = g.get("home_team", "")
            a = g.get("away_team", "")
            # Determine league for this game
            g_league = _normalize_league_str(g.get("league", "UNKNOWN"))

            if g_league not in game_tuples_by_league:
                game_tuples_by_league[g_league] = []

            if h and a:
                h_norm = TeamNameMatcher.normalize(h).upper().strip()
                a_norm = TeamNameMatcher.normalize(a).upper().strip()
                game_tuples_by_league[g_league].append((h_norm, a_norm, g))

            if h:
                # Normalization: Convert all incoming team names to UPPERCASE and strip extra spaces
                # This ensures the canonical name is used for matching.
                h_norm = TeamNameMatcher.normalize(h).upper().strip()
                if h_norm not in master_teams_norm_map:
                    master_teams_norm_map[h_norm] = []
                master_teams_norm_map[h_norm].append(g)

                # Add to league bucket
                if g_league not in teams_by_league:
                    teams_by_league[g_league] = []
                if h_norm not in teams_by_league[g_league]:
                    teams_by_league[g_league].append(h_norm)

            if a:
                a_norm = TeamNameMatcher.normalize(a).upper().strip()
                if a_norm not in master_teams_norm_map:
                    master_teams_norm_map[a_norm] = []
                master_teams_norm_map[a_norm].append(g)

                # Add to league bucket
                if g_league not in teams_by_league:
                    teams_by_league[g_league] = []
                if a_norm not in teams_by_league[g_league]:
                    teams_by_league[g_league].append(a_norm)

    # Normalize column names one last time to be safe
    df.columns = [str(c).upper().strip() for c in df.columns]

    for _, row in df.iterrows():
        # League normalization
        raw_league = str(row.get("LEAGUE", "UNKNOWN")).strip().upper()
        league = _normalize_league_str(raw_league)

        # Extract Team Names
        csv_home = str(row.get("HOMETEAM", "")).strip()
        csv_away = str(row.get("AWAYTEAM", "")).strip()

        # Apply Context-Aware Alias Resolution
        csv_home = _resolve_team_alias(csv_home, league)
        csv_away = _resolve_team_alias(csv_away, league)

        # Issue 1: Validate Team Names against League
        # CRITICAL: Prevent cross-league contamination (e.g. NHL teams in NBA)
        if not _validate_team_for_league(csv_home, league):
            logger.warning(f"Contamination detected: {csv_home} in {league}. Skipping.")
            continue
        if not _validate_team_for_league(csv_away, league):
            logger.warning(f"Contamination detected: {csv_away} in {league}. Skipping.")
            continue

        # Extract Date (if available) for Time Window Filtering
        input_date_str = str(row.get("COMMENCE", "")).strip()
        input_dt = None
        if input_date_str and input_date_str.lower() != "nan":
            try:
                # Try parsing basic formats
                from dateutil import parser
                input_dt = parser.parse(input_date_str)
            except Exception:
                pass

        # If teams are missing, skip
        if not csv_home or not csv_away or csv_home.lower() == "nan" or csv_away.lower() == "nan":
            continue

        # --- ADVANCED FUZZY MATCHING ---
        matched_game_obj = None
        match_confidence = 0.0
        match_status = "FAIL"
        closest_matches = []

        # Helper to find CANDIDATES (plural) for a team name
        def find_candidate_matches(name_raw, candidates) -> List[Tuple[str, float]]:
            """
            Returns list of (candidate, score) tuples for top matches.
            Prioritizes exact match and substrings.
            """
            if not name_raw or not candidates:
                return []

            # Normalization
            norm = TeamNameMatcher.normalize(name_raw).upper().strip()
            matches = []

            # 1. Exact Match (Score 100)
            if norm in candidates:
                matches.append((norm, 100.0))

            # 2. Substring Priority (Task 2)
            # If input is a leading substring of candidate (e.g. "VANCOUVER" in "VANCOUVER CANUCKS")
            # we treat it as very high confidence (100.0).
            # We filter for reasonable length (>3) to avoid spurious matches.
            if len(norm) > 3:
                for cand in candidates:
                    # STRICTER: Only accept Leading Substring (Task 2 Requirement)
                    # "Vancouver" -> "Vancouver Canucks" (Good)
                    # "Duke" -> "James Madison Dukes" (Bad - avoided by startswith)
                    if cand.startswith(norm):
                        matches.append((cand, 100.0))
                    elif norm in cand:
                         # Still give a boost to non-leading substrings, but slightly less than 100 to prefer leading
                         matches.append((cand, 95.0))

            # 3. Fuzzy Match (Extract Top N)
            if process:
                # Get top 5 candidates to allow for pairwise resolution
                # Use token_set_ratio to handle word reordering/partials
                results = process.extract(norm, candidates, scorer=fuzz.token_set_ratio, limit=5)
                for m, s, _ in results:
                    matches.append((m, float(s)))

            # Sort by score desc
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches

        # League-Gated Matching Candidates
        # Task 1: Strict League Filtering
        # If league is UNKNOWN, do not attempt to match against the entire world.
        if league == "UNKNOWN":
            candidates = []
            logger.debug(f"Skipping match for {csv_home} vs {csv_away}: League is UNKNOWN")
        else:
            # Verification Pattern: Ensure the logic follows: league_candidates = [g for g in games if g['league'] == row['league']]
            # Explicitly filter games for the current league to prevent cross-league pollution (e.g. NFL Houston vs NCAAB Houston)
            league_candidates = [g for g in games if _normalize_league_str(g.get("league", "UNKNOWN")) == league]

            # Re-derive candidates strictly from this filtered list
            candidates_set = set()
            for g in league_candidates:
                if g.get("home_team"):
                    candidates_set.add(TeamNameMatcher.normalize(g.get("home_team")).upper().strip())
                if g.get("away_team"):
                    candidates_set.add(TeamNameMatcher.normalize(g.get("away_team")).upper().strip())
            candidates = list(candidates_set)

            # Log league-specific filtering for debugging
            logger.debug(f"League-Gated Matching: Found {len(candidates)} unique teams from {len(league_candidates)} {league} games for matching {csv_home} vs {csv_away}")

        if candidates:
            # Pairwise Resolution Logic (Task 2)
            # Instead of picking best Home and best Away independently, we pick the best PAIR that forms a valid game.

            # 1. Get Candidates
            h_candidates = find_candidate_matches(csv_home, candidates)
            a_candidates = find_candidate_matches(csv_away, candidates)

            # 2. Iterate through pairs to find a valid game
            # We look for a game where HomeCandidate vs AwayCandidate exists.
            # We also check the reverse order (Task 2: Swapped Pairs), though our intersection logic is inherently order-agnostic.

            best_pair_game = None
            best_pair_score = 0.0
            best_pair_names = (None, None)

            # Limit to top 5 to keep complexity low (5x5 = 25 checks max)
            # Task 3: Location-Agnostic Verification (Solved by common_games logic, but expanded check)
            for h_cand, h_s in h_candidates[:5]:
                for a_cand, a_s in a_candidates[:5]:

                    # Relaxed Threshold for Pair Validation:
                    # If we find a VALID GAME, we can tolerate slightly lower individual scores
                    # because the existence of the pairing confirms the identity.
                    # Lowering 75.0 -> 50.0 allows "AR-Pine Bluff" (58.8) to match if pair is valid.
                    if h_s < 50.0 or a_s < 50.0:
                        continue

                    # Check for valid game intersection (Home/Away Agnostic)
                    h_games = master_teams_norm_map.get(h_cand, [])
                    a_games = master_teams_norm_map.get(a_cand, [])

                    # common_games contains games where BOTH candidates are playing (regardless of side)
                    common_games = [g for g in h_games if g in a_games]

                    # Strict League Filter
                    if league != "UNKNOWN":
                        common_games = [g for g in common_games if _normalize_league_str(g.get("league", "UNKNOWN")) == league]

                    if common_games:
                        # Task 3: 12-Hour Time Buffer Implementation
                        # Filter (Relaxed 12h Buffer) to ignore UTC vs local offsets
                        valid_game = None
                        for g_cand in common_games:
                            # If input has no date, accept the first game (legacy behavior)
                            if not input_dt:
                                valid_game = g_cand
                                break

                            # If input has date, check buffer
                            g_date_str = g_cand.get("commence_time") or g_cand.get("commence_time_iso_utc")
                            if g_date_str:
                                try:
                                    from dateutil import parser
                                    from datetime import timedelta
                                    g_dt = parser.parse(g_date_str)
                                    # Handle timezone naive/aware comparison
                                    if input_dt.tzinfo and not g_dt.tzinfo:
                                        from datetime import timezone
                                        g_dt = g_dt.replace(tzinfo=timezone.utc)
                                    elif not input_dt.tzinfo and g_dt.tzinfo:
                                        input_dt = input_dt.replace(tzinfo=timezone.utc)

                                    # 12 Hour Buffer Logic
                                    diff_hours = abs((g_dt - input_dt).total_seconds()) / 3600.0
                                    if diff_hours <= 12:
                                        valid_game = g_cand
                                        break
                                except Exception:
                                    # Date parse error on game side, accept as fallback?
                                    valid_game = g_cand
                                    break
                            else:
                                valid_game = g_cand
                                break

                        if valid_game:
                            # Boost score if a valid game exists (Confirmation Bonus)
                            # If we found a real game match, this is almost certainly correct.
                            # We use the raw fuzzy score but guarantee acceptance if > best.
                            avg_score = (h_s + a_s) / 2.0

                            if avg_score > best_pair_score:
                                best_pair_score = avg_score
                                best_pair_game = valid_game
                                best_pair_names = (h_cand, a_cand)

            # 3. Final Decision
            if best_pair_game:
                matched_game_obj = best_pair_game
                match_confidence = best_pair_score
                match_status = "MATCH"

                # Update debug logging vars
                csv_home_disp = f"{csv_home} -> {best_pair_names[0]} ({h_candidates[0][1]:.1f})"
                # Note: h_candidates[0] might not be the one we picked if we picked a lower score to make a pair
                # But for debug, we usually want to see what we picked.
                # Let's construct a helpful string.
                closest_matches = [
                    f"H: {best_pair_names[0]} (Score {dict(h_candidates).get(best_pair_names[0], 0):.1f})",
                    f"A: {best_pair_names[1]} (Score {dict(a_candidates).get(best_pair_names[1], 0):.1f})"
                ]

            else:
                # Fallback: Robust check using TeamNameMatcher.match_game which iterates all tuples
                # This explicitly handles cases where our intersection logic failed due to data issues,
                # or confirms MISMATCH_PAIR if it still fails.
                # This explicitly satisfies the requirement to check for swapped pairs in the schedule.
                league_tuples = game_tuples_by_league.get(league, [])
                if league_tuples:
                    # Pass just the (home, away) tuples to the matcher
                    simple_tuples = [(t[0], t[1]) for t in league_tuples]
                    matched_tuple = TeamNameMatcher.match_game(csv_home, csv_away, simple_tuples, threshold=0.70)

                    if matched_tuple:
                        # Find the game object for this tuple
                        for h_t, a_t, g_obj in league_tuples:
                            if (h_t, a_t) == matched_tuple:
                                matched_game_obj = g_obj
                                match_confidence = 0.70 # Fallback confidence
                                match_status = "MATCH"
                                closest_matches = [f"Robust Fallback: {matched_tuple}"]
                                break

                if not matched_game_obj:
                    # Failed to find a valid pair game.
                    # Report status based on whether individual matches were good or bad.
                    top_h = h_candidates[0] if h_candidates else (None, 0.0)
                    top_a = a_candidates[0] if a_candidates else (None, 0.0)

                    if top_h[1] >= 75.0 and top_a[1] >= 75.0:
                        # If high confidence but NO game found, it's a mismatch pair (could be swapped leagues, or just bad schedule data)
                        match_status = "MISMATCH_PAIR"
                    else:
                        match_status = "FAIL" # Low confidence matches

                    # detailed logging for failed match (Task 2c)
                    closest_matches = [
                        f"Top H: {top_h[0]} ({top_h[1]:.1f})",
                        f"Top A: {top_a[0]} ({top_a[1]:.1f})"
                    ]

        # CRITICAL: Enforce strict league matching to prevent cross-sport contamination
        if matched_game_obj and league != "UNKNOWN":
            matched_league = _normalize_league_str(matched_game_obj.get("league", "UNKNOWN"))
            if matched_league != league:
                logger.warning(f"Cross-League Contamination Blocked: Input {league} matched {matched_league} game ({matched_game_obj.get('home_team')} vs {matched_game_obj.get('away_team')}). Voiding match.")
                matched_game_obj = None
                match_confidence = 0.0
                match_status = "FAIL"
                closest_matches.append(f"BLOCKED: Cross-League {league}->{matched_league}")

        # Logging
        stats_collector.append({
            "csv_home": csv_home,
            "csv_away": csv_away,
            "matched_home": matched_game_obj.get("home_team") if matched_game_obj else None,
            "matched_away": matched_game_obj.get("away_team") if matched_game_obj else None,
            "source_date": slate_date,
            "target_date": matched_game_obj.get("commence_date_local") if matched_game_obj else None,
            "confidence": f"{match_confidence:.2f}",
            "status": match_status,
            "closest_matches": "; ".join(closest_matches)
        })

        # --- KEY GENERATION ---
        # Critical: Use matched canonical names if available
        if matched_game_obj:
            league = matched_game_obj.get("league", league)
            # Use commence_date_local from matched game if available
            date_val = matched_game_obj.get("commence_date_local") or slate_date
            home_code = team_code_for_league(league, matched_game_obj.get("home_team"))
            away_code = team_code_for_league(league, matched_game_obj.get("away_team"))

            # Issue 5: Detect Home/Away Swap
            # Check if csv_home matches matched_game_obj["away_team"] better than home_team
            # We assume matched_game_obj["home_team"] maps to home_code
            # If csv_home is actually the Away team, we should swap home_team_raw/away_team_raw
            # in the output record so they align with codes.

            # Normalize for comparison
            matched_home_norm = TeamNameMatcher.normalize(matched_game_obj.get("home_team")).upper()
            matched_away_norm = TeamNameMatcher.normalize(matched_game_obj.get("away_team")).upper()
            csv_home_norm = TeamNameMatcher.normalize(csv_home).upper()

            # Simple check: if csv_home matches matched_away more than matched_home
            if fuzz:
                score_home = fuzz.token_set_ratio(csv_home_norm, matched_home_norm)
                score_away = fuzz.token_set_ratio(csv_home_norm, matched_away_norm)

                if score_away > score_home + 10: # Significant difference
                    # Swap logic
                    temp = csv_home
                    csv_home = csv_away
                    csv_away = temp
                    logger.info(f"Detected Home/Away Swap for {csv_away} @ {csv_home}. Swapped raw names to align with schedule.")

        else:
            date_val = slate_date
            home_code = team_code_for_league(league, csv_home)
            away_code = team_code_for_league(league, csv_away)

        canon_key = generate_canonical_key(league=league, date_str=date_val, home_code=home_code, away_code=away_code)

        # Pick & Line
        raw_pick = str(row.get("PICK", "")).strip()

        line_val = None
        try:
            l_raw = row.get("LINE")
            if pd.notnull(l_raw):
                line_val = float(l_raw)
        except (ValueError, TypeError):
            pass


        # Hit Rate (Issue 4: NaN Handling)
        hit_rate = None
        try:
            wp = row.get("WINPROBABILITY")
            if wp is not None and pd.notnull(wp) and str(wp).strip() != "":
                s_wp = str(wp).replace("%", "").strip()
                if s_wp:
                    hit_rate = float(s_wp)
                    if hit_rate > 1.0: hit_rate /= 100.0
        except (ValueError, TypeError):
            hit_rate = None

        # Market Type (Moved BEFORE probability calculation to fix UnboundLocalError)
        market_raw = str(row.get("MARKET", pick_type_default)).upper()
        final_pick_type = pick_type_default
        final_pick_val = raw_pick

        if "TOTAL" in market_raw:
             final_pick_type = "TOTAL"
        elif "SPREAD" in market_raw or "SIDE" in market_raw:
             final_pick_type = "SIDE"

        # Line Extraction
        if line_val is None:
            match = re.search(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', raw_pick, re.IGNORECASE)
            if match:
                line_val = float(match.group(2))
                if final_pick_type == "TOTAL":
                    final_pick_val = match.group(1).upper()
            else:
                # Improved regex to handle parenthesized lines like "(-102)" or "-5.5"
                # Match a number at the end, optionally surrounded by parens, tolerating whitespace
                match_spread = re.search(r'\(?\s*(-?\d+\.?\d*)\s*\)?$', raw_pick)
                if match_spread:
                     line_val = float(match_spread.group(1))

        # Fix Issue #2: Calculate probability for Spread picks if missing
        if hit_rate is None and final_pick_type == "SIDE" and line_val is not None:
             try:
                 # Implement formula: raw_prob = 0.50 - (line_val / 100.0)
                 # Example: -2.5 -> 0.5 - (-0.025) = 0.525 (52.5%)
                 # Example: -9.5 -> 0.5 - (-0.095) = 0.595 (59.5%)
                 # Example: +9.5 -> 0.5 - (0.095) = 0.405 (40.5%)

                 line_float = float(line_val)
                 raw_prob = 0.50 - (line_float / 100.0)

                 # Apply edge boost (+0.07)
                 adjusted_prob = raw_prob + 0.07

                 # Clamp
                 hit_rate = max(0.10, min(0.95, adjusted_prob))

                 # Log if debugging enabled
                 logger.debug(f"Calculated Spread Prob for {line_val}: {hit_rate:.3f} (Raw: {raw_prob:.3f})")
             except Exception:
                 pass

        records.append({
            "theover_key": canon_key,
            "league": league,
            "date_local": date_val,
            "away_code": away_code,
            "home_code": home_code,
            "theover_pick": final_pick_val,
            "theover_market_type": final_pick_type,
            "theover_line": line_val,
            "theover_model": "TheOver",
            "theover_hit_rate": hit_rate,
            "raw_text": str(row.to_dict()),
            "away_team_raw": csv_away,
            "home_team_raw": csv_home
        })

    return pd.DataFrame(records)

def parse_theover_public_betting_text(raw_text: str, pick_type_hint: str = "UNKNOWN") -> pd.DataFrame:
    """
    Parse raw text paste from TheOver.ai.
    """
    rows = []
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

    current_league = "UNKNOWN"
    current_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    current_away = ""
    current_home = ""

    i = 0
    while i < len(lines):
        line = lines[i]

        if any(x in line.upper() for x in ["NBA", "NFL", "NHL", "NCAAB", "NCAAF", "CBB", "CFB"]):
            upper = line.upper()
            if "NBA" in upper: current_league = "NBA"
            elif "NFL" in upper: current_league = "NFL"
            elif "NHL" in upper: current_league = "NHL"
            elif "NCAAB" in upper or "CBB" in upper: current_league = "NCAAB"
            elif "NCAAF" in upper or "CFB" in upper: current_league = "NCAAF"
            i += 1
            continue

        if " @ " in line or " vs " in line:
            splitter = " @ " if " @ " in line else " vs "
            parts = line.split(splitter)
            if len(parts) >= 2:
                current_away = parts[0].strip()
                current_home = parts[1].strip()
                i += 1
                continue

        total_match = re.search(r'^(Over|Under)\s+([0-9]+\.?[0-9]*)', line, re.IGNORECASE)
        pick_type = "UNKNOWN"
        pick_val = None
        pick_line = None

        if total_match:
            pick_type = "TOTAL"
            pick_val = total_match.group(1).upper()
            pick_line = float(total_match.group(2))
        else:
            if not line.startswith("Backed by") and not line.startswith("Analyze"):
                pick_type = "SIDE"
                # Try to extract line from side pick
                # Format might be "Team (-110)" or "Team -5.5"
                line_match = re.search(r'\(?\s*(-?\d+\.?\d*)\s*\)?$', line)
                if line_match:
                     try:
                        pick_line = float(line_match.group(1))
                        # Remove the line part from the team name
                        pick_val = line[:line_match.start()].strip()
                     except (ValueError, TypeError):
                        pick_val = line.split("(")[0].strip()
                else:
                     pick_val = line.split("(")[0].strip()

        if pick_type != "UNKNOWN":
            model_name = "TheOver"
            hit_rate = 0.0

            if i + 1 < len(lines):
                next_line = lines[i+1]
                if next_line.startswith("Backed by"):
                    m = re.search(r'Backed by (.*?) \((\d+)%\)', next_line)
                    if m:
                        model_name = m.group(1).strip()
                        hit_rate = float(m.group(2)) / 100.0
                    else:
                        model_name = next_line.replace("Backed by ", "").strip()
                    i += 1

            if pick_type_hint != "UNKNOWN":
                market_type = pick_type_hint
            else:
                market_type = pick_type

            away_code = team_code_for_league(current_league, current_away)
            home_code = team_code_for_league(current_league, current_home)
            canon_key = generate_canonical_key(league=current_league, date_str=current_date, home_code=home_code, away_code=away_code)

            rows.append({
                "theover_key": canon_key,
                "league": current_league,
                "date_local": current_date,
                "away_code": away_code,
                "home_code": home_code,
                "theover_pick": pick_val,
                "theover_market_type": market_type,
                "theover_line": pick_line,
                "theover_model": model_name,
                "theover_hit_rate": hit_rate,
                "raw_text": line,
                "away_team_raw": current_away,
                "home_team_raw": current_home
            })

        i += 1

    return pd.DataFrame(rows)

def process_theover_inputs(
    totals_file=None,
    sides_file=None,
    totals_paste=None,
    sides_paste=None,
    games=None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main ingestion entry point.
    Merges all inputs into a single dataframe indexed by 'theover_key'.
    Returns (DataFrame, stats_dict).
    """
    dfs = []
    stats = {
        "raw_total_rows": 0,
        "raw_sides_rows": 0,
        "raw_totals_rows": 0,
        "files_processed": [],
        "unmatched_examples": [],
        "matched_rows": 0,
        "unmatched_rows": 0
    }

    if games is None:
        logger.warning("No Master Schedule (games) provided to process_theover_inputs. Fuzzy matching will be skipped.")

    matching_logs = []

    if totals_file:
        try:
            df = parse_theover_csv(totals_file)
            stats["raw_totals_rows"] = len(df)
            if not df.empty:
                processed = _transform_theover_df(df, "TOTAL", games, matching_logs)
                if not processed.empty:
                    dfs.append(processed)
            stats["files_processed"].append("totals_file")
        except Exception as e:
            logger.error(f"Error processing Totals file: {e}", exc_info=True)

    if sides_file:
        try:
            df = parse_theover_csv(sides_file)
            stats["raw_sides_rows"] = len(df)
            if not df.empty:
                processed = _transform_theover_df(df, "SIDE", games, matching_logs)
                if not processed.empty:
                    dfs.append(processed)
            stats["files_processed"].append("sides_file")
        except Exception as e:
            logger.error(f"Error processing Sides file: {e}", exc_info=True)

    if totals_paste and totals_paste.strip():
        dfs.append(parse_theover_public_betting_text(totals_paste, pick_type_hint="TOTAL"))

    if sides_paste and sides_paste.strip():
        dfs.append(parse_theover_public_betting_text(sides_paste, pick_type_hint="SIDE"))

    stats["total_rows"] = stats["raw_totals_rows"] + stats["raw_sides_rows"]

    failed_logs = [l for l in matching_logs if l["status"] == "FAIL" or l["status"] == "MISMATCH_PAIR"]
    stats["unmatched_examples"] = failed_logs[:10]
    stats["unmatched_rows"] = len(failed_logs)
    stats["matched_rows"] = len([l for l in matching_logs if l["status"] == "MATCH"])
    stats["full_debug_log"] = matching_logs

    if not dfs:
        # Task 3: Ensure raw_df is empty if no dfs
        stats["raw_df"] = pd.DataFrame()
        return pd.DataFrame(), stats

    # Fix: Pandas Concat Warning (filter empty DFs first)
    # Ensure no None entries and no empty dataframes
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if valid_dfs:
        combined = pd.concat(valid_dfs, ignore_index=True)
    else:
        combined = pd.DataFrame()

    # Task 3: Capture the raw combined dataframe before deduplication for debugging
    stats["raw_df"] = combined.copy()

    if combined.empty:
        return combined, stats

    combined = combined.sort_values(by="theover_hit_rate", ascending=False)
    deduped = combined.drop_duplicates(subset=["theover_key", "theover_market_type"], keep="first")

    return deduped, stats
