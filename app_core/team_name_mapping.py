"""
Enhanced team name mapping for NCAAB and other sports.
This module provides comprehensive team name normalization and mapping.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

logger = logging.getLogger(__name__)

# Comprehensive NCAAB team abbreviation mapping
NCAAB_TEAM_ABBREV_MAP = {
    # Atlantic 10
    "Rhode Island": ["URI", "Rhode", "RI", "Rhode Island Rams"],
    "Saint Louis": ["SLU", "St Louis", "Saint Louis Billikens", "St. Louis"],
    "VCU": ["VCU", "Virginia Commonwealth"],
    "Dayton": ["DAY", "Dayton Flyers"],
    "Richmond": ["RICH", "Richmond Spiders"],

    # Big East
    "Villanova": ["NOVA", "Villanova Wildcats"],
    "UConn": ["CONN", "Connecticut", "Connecticut Huskies"],
    "Marquette": ["MARQ", "Marquette Golden Eagles"],
    "Creighton": ["CRE", "Creighton Bluejays"],
    "Xavier": ["XAV", "Xavier Musketeers"],
    "Providence": ["PROV", "Providence Friars"],
    "Seton Hall": ["SHU", "Seton Hall Pirates"],
    "St. John's": ["SJU", "St John's", "Saint John's", "St. John's Red Storm"],
    "Georgetown": ["GTWN", "Georgetown Hoyas"],
    "Butler": ["BUT", "Butler Bulldogs"],
    "DePaul": ["DEP", "DePaul Blue Demons"],

    # Big 12
    "Kansas": ["KU", "Kansas Jayhawks"],
    "Baylor": ["BAY", "Baylor Bears"],
    "Texas Tech": ["TTU", "Texas Tech Red Raiders"],
    "West Virginia": ["WVU", "West Virginia Mountaineers"],
    "Kansas State": ["KSU", "K-State", "Kansas State Wildcats"],
    "Oklahoma State": ["OKST", "Oklahoma State Cowboys"],
    "TCU": ["TCU", "Texas Christian"],
    "Iowa State": ["ISU", "Iowa State Cyclones"],
    "Texas": ["TEX", "Texas Longhorns"],
    "Oklahoma": ["OU", "Oklahoma Sooners"],

    # Big Ten
    "Michigan State": ["MSU", "Michigan State Spartans"],
    "Illinois": ["ILL", "Illinois Fighting Illini"],
    "Purdue": ["PUR", "Purdue Boilermakers"],
    "Indiana": ["IND", "Indiana Hoosiers"],
    "Ohio State": ["OSU", "Ohio State Buckeyes"],
    "Michigan": ["MICH", "Michigan Wolverines"],
    "Wisconsin": ["WISC", "Wisconsin Badgers"],
    "Maryland": ["MD", "Maryland Terrapins"],
    "Penn State": ["PSU", "Penn State Nittany Lions"],
    "Iowa": ["IOWA", "Iowa Hawkeyes"],
    "Minnesota": ["MINN", "Minnesota Golden Gophers"],
    "Northwestern": ["NW", "Northwestern Wildcats"],
    "Nebraska": ["NEB", "Nebraska Cornhuskers"],
    "Rutgers": ["RU", "Rutgers Scarlet Knights"],

    # SEC
    "Kentucky": ["UK", "Kentucky Wildcats"],
    "Tennessee": ["TENN", "Tennessee Volunteers"],
    "Auburn": ["AUB", "Auburn Tigers"],
    "Alabama": ["ALA", "BAMA", "Alabama Crimson Tide"],
    "Arkansas": ["ARK", "Arkansas Razorbacks"],
    "Florida": ["FLA", "Florida Gators"],
    "Mississippi State": ["MSST", "Miss State", "Mississippi State Bulldogs"],
    "Ole Miss": ["MISS", "Mississippi", "Ole Miss Rebels"],
    "LSU": ["LSU", "Louisiana State"],
    "Missouri": ["MIZZ", "Mizzou", "Missouri Tigers"],
    "Texas A&M": ["TAMU", "Texas A&M Aggies"],
    "South Carolina": ["SC", "South Carolina Gamecocks"],
    "Georgia": ["UGA", "Georgia Bulldogs"],
    "Vanderbilt": ["VAN", "Vanderbilt Commodores"],

    # ACC
    "Duke": ["DUKE", "Duke Blue Devils"],
    "North Carolina": ["UNC", "North Carolina Tar Heels"],
    "Virginia": ["UVA", "Virginia Cavaliers"],
    "Miami": ["MIA", "Miami Hurricanes", "Miami FL"],
    "Syracuse": ["SYR", "Syracuse Orange"],
    "Clemson": ["CLEM", "Clemson Tigers"],
    "Florida State": ["FSU", "Florida State Seminoles"],
    "NC State": ["NCST", "N.C. State", "NC State Wolfpack"],
    "Pittsburgh": ["PITT", "Pittsburgh Panthers"],
    "Virginia Tech": ["VT", "Virginia Tech Hokies"],
    "Wake Forest": ["WAKE", "Wake Forest Demon Deacons"],
    "Georgia Tech": ["GT", "Georgia Tech Yellow Jackets"],
    "Louisville": ["LOU", "Louisville Cardinals"],
    "Boston College": ["BC", "Boston College Eagles"],
    "Notre Dame": ["ND", "Notre Dame Fighting Irish"],

    # Pac-12 (now defunct but still relevant)
    "Arizona": ["ARIZ", "Arizona Wildcats"],
    "UCLA": ["UCLA", "UC Los Angeles"],
    "USC": ["USC", "Southern California"],
    "Oregon": ["ORE", "Oregon Ducks"],
    "Washington": ["WASH", "Washington Huskies"],

    # Mountain West
    "San Diego State": ["SDSU", "San Diego State Aztecs"],
    "Nevada": ["NEV", "Nevada Wolf Pack"],
    "UNLV": ["UNLV", "Nevada Las Vegas"],
    "New Mexico": ["UNM", "New Mexico Lobos"],
    "Boise State": ["BSU", "Boise State Broncos"],
    "Colorado State": ["CSU", "Colorado State Rams"],

    # American Athletic Conference
    "Houston": ["HOU", "Houston Cougars"],
    "Memphis": ["MEM", "Memphis Tigers"],
    "Cincinnati": ["CIN", "Cincinnati Bearcats"],
    "SMU": ["SMU", "Southern Methodist"],
    "UCF": ["UCF", "Central Florida"],
    "Temple": ["TEM", "Temple Owls"],
    "Wichita State": ["WICH", "Wichita State Shockers"],
    "East Carolina": ["ECU", "East Carolina Pirates"],

    # West Coast Conference
    "Gonzaga": ["GONZ", "Gonzaga Bulldogs"],
    "Saint Mary's": ["SMC", "St Mary's", "Saint Mary's Gaels"],
    "BYU": ["BYU", "Brigham Young"],
    "San Francisco": ["USF", "San Francisco Dons"],
    "Santa Clara": ["SCU", "Santa Clara Broncos"],

    # MAC
    "Bowling Green": ["BGSU", "Bowling Green Falcons"],
    "Kent State": ["KENT", "Kent State Golden Flashes"],
    "Akron": ["AKR", "Akron Zips"],
    "Buffalo": ["BUF", "Buffalo Bulls"],
    "Ohio": ["OHIO", "Ohio Bobcats"],
    "Miami (OH)": ["MIA", "MOH", "Miami RedHawks"],
    "Toledo": ["TOL", "Toledo Rockets"],
    "Ball State": ["BSU", "Ball State Cardinals"],
    "Central Michigan": ["CMU", "Central Michigan Chippewas"],
    "Eastern Michigan": ["EMU", "Eastern Michigan Eagles"],
    "Northern Illinois": ["NIU", "Northern Illinois Huskies"],
    "Western Michigan": ["WMU", "Western Michigan Broncos"],

    # Add more as needed...
}

# Reverse mapping: abbreviation -> full name
ABBREV_TO_FULL_MAP = {}
for full_name, abbrevs in NCAAB_TEAM_ABBREV_MAP.items():
    for abbrev in abbrevs:
        ABBREV_TO_FULL_MAP[abbrev.upper()] = full_name

def normalize_team_name(team_name: str) -> str:
    """
    Normalize a team name by removing common suffixes and standardizing format.

    Args:
        team_name: Raw team name from API

    Returns:
        Normalized team name
    """
    if not team_name:
        return ""

    # Convert to string and strip
    team_name = str(team_name).strip()

    # Remove common suffixes
    suffixes = [
        "Rams", "Wildcats", "Eagles", "Bears", "Tigers", "Bulldogs",
        "Cardinals", "Trojans", "Bruins", "Sun Devils", "Ducks",
        "Huskies", "Cougars", "Spartans", "Wolverines", "Buckeyes",
        "Fighting Irish", "Tar Heels", "Blue Devils", "Demon Deacons",
        "Yellow Jackets", "Hokies", "Cavaliers", "Mountaineers",
        "Cyclones", "Sooners", "Cowboys", "Jayhawks", "Longhorns",
        "Aggies", "Razorbacks", "Volunteers", "Gamecocks", "Gators",
        "Crimson Tide", "Roll Tide", "Tide", "War Eagle",
        "Men", "Women", "Basketball", "College", "University",
        "State", "Tech", "A&M", "International",
        "Falcons", "Golden Flashes", "Zips", "Bulls", "Bobcats", "RedHawks", "Rockets", "Chippewas",
        "Friars", "Red Storm", "Billikens", "Spiders", "Flyers", "Patriots", "Dukes", "Explorers", "Revolutionaries"
    ]

    normalized = team_name
    for suffix in suffixes:
        # Remove suffix if it appears at the end
        if normalized.endswith(f" {suffix}"):
            normalized = normalized[:-len(suffix)-1].strip()

    # Remove "University of" or "The" prefix
    prefixes = ["University of ", "The ", "College of "]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()

    return normalized.strip()

def get_team_variants(team_name: str) -> List[str]:
    """
    Get all possible variants of a team name for matching.

    Args:
        team_name: Team name to get variants for

    Returns:
        List of team name variants
    """
    variants = [team_name]

    # Add normalized version
    normalized = normalize_team_name(team_name)
    if normalized and normalized != team_name:
        variants.append(normalized)

    # Check if it's in our abbreviation map
    normalized_upper = normalized.upper()
    for full_name, abbrevs in NCAAB_TEAM_ABBREV_MAP.items():
        # Check if the team_name matches any known variant
        if (team_name in abbrevs or
            normalized in abbrevs or
            team_name.upper() in [a.upper() for a in abbrevs]):
            variants.extend([full_name] + abbrevs)
            break

    # Check reverse mapping (abbreviation -> full name)
    if normalized_upper in ABBREV_TO_FULL_MAP:
        full_name = ABBREV_TO_FULL_MAP[normalized_upper]
        variants.append(full_name)
        if full_name in NCAAB_TEAM_ABBREV_MAP:
            variants.extend(NCAAB_TEAM_ABBREV_MAP[full_name])

    # Add common variations
    # Handle "State" variations
    if " State" in team_name:
        base = team_name.replace(" State", "")
        variants.append(base)
        variants.append(f"{base} St")
        variants.append(f"{base} St.")

    # Handle "Saint" vs "St." variations
    if "Saint " in team_name:
        variants.append(team_name.replace("Saint ", "St. "))
        variants.append(team_name.replace("Saint ", "St "))
    elif "St. " in team_name:
        variants.append(team_name.replace("St. ", "Saint "))
        variants.append(team_name.replace("St. ", "St "))

    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for variant in variants:
        variant_upper = variant.upper().strip()
        if variant_upper not in seen:
            seen.add(variant_upper)
            unique_variants.append(variant.strip())

    return unique_variants

def fuzzy_match_teams(team1: str, team2: str, threshold: int = 85) -> Tuple[bool, int]:
    """
    Perform fuzzy matching between two team names.

    Args:
        team1: First team name
        team2: Second team name
        threshold: Minimum similarity score (0-100) to consider a match

    Returns:
        Tuple of (is_match, similarity_score)
    """
    if fuzz is None:
        # Fallback if rapidfuzz not installed
        # Simple exact match (case insensitive)
        is_match = team1.upper().strip() == team2.upper().strip()
        return is_match, 100 if is_match else 0

    # Get all variants for both teams
    variants1 = get_team_variants(team1)
    variants2 = get_team_variants(team2)

    # Try all combinations and find the best match
    best_score = 0
    for v1 in variants1:
        for v2 in variants2:
            # Try multiple fuzzy matching algorithms
            ratio_score = fuzz.ratio(v1.upper(), v2.upper())
            partial_score = fuzz.partial_ratio(v1.upper(), v2.upper())
            token_sort_score = fuzz.token_sort_ratio(v1.upper(), v2.upper())

            # Use the best score from all algorithms
            score = max(ratio_score, partial_score, token_sort_score)
            best_score = max(best_score, score)

            # Early exit if we found a perfect match
            if best_score == 100:
                return True, 100

    return best_score >= threshold, best_score

def extract_team_abbreviations_from_ticker(ticker: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract team abbreviations from a Kalshi ticker.

    Examples:
        "KXNCAAMBGAME-26FEB19URISLU-URI" -> ("URI", "SLU")
        "KXNCAAMBSPREAD-26FEB18BYUARIZ-BYU1" -> ("BYU", "ARIZ")

    Args:
        ticker: Kalshi market ticker

    Returns:
        Tuple of (away_team_abbrev, home_team_abbrev) or (None, None) if parsing fails
    """
    try:
        # Pattern: SERIES-DATECODETEAM1TEAM2-OUTCOME
        # Example: KXNCAAMBGAME-26FEB19URISLU-URI

        # Split by hyphens
        parts = ticker.split('-')
        if len(parts) < 3:
            return None, None

        # The middle part contains date + teams
        date_teams = parts[1]

        # Extract date portion (7 characters: 26FEB19)
        if len(date_teams) < 7:
            return None, None

        teams_portion = date_teams[7:]  # Everything after date

        # Teams are concatenated uppercase letters
        # Need to find the split point
        # Heuristic: typically 3-5 characters per team

        # Try to find team abbreviations by checking our mapping
        team_len = len(teams_portion)

        for split in range(2, team_len - 1):
            team1 = teams_portion[:split]
            team2 = teams_portion[split:]

            # Check if both parts look like valid team abbreviations
            if (team1.upper() in ABBREV_TO_FULL_MAP or
                team2.upper() in ABBREV_TO_FULL_MAP):
                return team1, team2

        # Fallback: try even split
        mid = team_len // 2
        return teams_portion[:mid], teams_portion[mid:]

    except Exception as e:
        logger.warning(f"Failed to parse ticker {ticker}: {e}")
        return None, None
