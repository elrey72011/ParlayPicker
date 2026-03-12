"""
Team name normalization for consistent matching across data sources.
"""
import re

import json
import logging
import os

try:
    from rapidfuzz import fuzz
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Track missing keys for terminal warnings
_MISSING_KEYS_WARNED = set()

# Use __file__ instead of __dirname__ since __dirname__ is not a Python built-in
DYNAMIC_ALIASES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dynamic_aliases.json")

def load_dynamic_aliases() -> dict[str, str]:
    if not os.path.exists(DYNAMIC_ALIASES_FILE):
        return {}
    try:
        with open(DYNAMIC_ALIASES_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading dynamic aliases: {e}")
        return {}

# Keep legacy exact mappings for backwards compatibility
TEAM_MAP = {
    # Existing explicit mappings
    "FIU Panthers": "FIU",
    "UC Santa Barbara": "UCSB",
    "Texas-El Paso": "UTEP",

    # NBA Full to Short
    "Atlanta Hawks": "Atlanta",
    "Boston Celtics": "Boston",
    "Brooklyn Nets": "Brooklyn",
    "Charlotte Hornets": "Charlotte",
    "Chicago Bulls": "Chicago",
    "Cleveland Cavaliers": "Cleveland",
    "Dallas Mavericks": "Dallas",
    "Denver Nuggets": "Denver",
    "Detroit Pistons": "Detroit",
    "Golden State Warriors": "Golden State",
    "Houston Rockets": "Houston",
    "Indiana Pacers": "Indiana",
    "LA Clippers": "LA Clippers",
    "L.A. Clippers": "LA Clippers",
    "Los Angeles Clippers": "LA Clippers",
    "LA Lakers": "L.A. Lakers",
    "Los Angeles Lakers": "L.A. Lakers",
    "Memphis Grizzlies": "Memphis",
    "Miami Heat": "Miami",
    "Milwaukee Bucks": "Milwaukee",
    "Minnesota Timberwolves": "Minnesota",
    "New Orleans Pelicans": "New Orleans",
    "New York Knicks": "New York",
    "Oklahoma City Thunder": "Oklahoma City",
    "Orlando Magic": "Orlando",
    "Philadelphia 76ers": "Philadelphia",

    # Missing explicit mappings from logs
    "Los Angeles Salle": "La Salle",
    "Cal Poly Slo": "Cal Poly",
    "Unlv": "UNLV",
    "Uc San Diego": "UC San Diego",
    "Uc Davis": "UC Davis",
    "L.A. Lakers": "Los Angeles Lakers",
    "NY Rangers": "New York Rangers",
    "NY Jets": "New York Jets",
    "NY Islanders": "New York Islanders",
    "Penn St.": "Penn State",
    "Missouri St.": "Missouri State",
    "Florida St Seminoles": "Florida State",
    "Manhattan Jaspers": "Manhattan",
    "Jacksonville St.": "Jacksonville State",
    "Prairie View A&M": "Prairie View A&M",
    "Virginia Tech": "Virginia Tech",
    "Virginia Tech Hokies": "Virginia Tech",
    "Phoenix Suns": "Phoenix",
    "Portland Trail Blazers": "Portland",
    "Sacramento Kings": "Sacramento",
    "San Antonio Spurs": "San Antonio",
    "Toronto Raptors": "Toronto",
    "Utah Jazz": "Utah",
    "Washington Wizards": "Washington",

    # NHL Full to Short
    "Anaheim Ducks": "Anaheim",
    "Arizona Coyotes": "Arizona",
    "Boston Bruins": "Boston",
    "Buffalo Sabres": "Buffalo",
    "Calgary Flames": "Calgary",
    "Carolina Hurricanes": "Carolina",
    "Chicago Blackhawks": "Chicago",
    "Colorado Avalanche": "Colorado",
    "Columbus Blue Jackets": "Columbus",
    "Dallas Stars": "Dallas",
    "Detroit Red Wings": "Detroit",
    "Edmonton Oilers": "Edmonton",
    "Florida Panthers": "Florida",
    "Los Angeles Kings": "Los Angeles",
    "Minnesota Wild": "Minnesota",
    "Montreal Canadiens": "Montreal",
    "Montréal Canadiens": "Montreal",
    "Nashville Predators": "Nashville",
    "New Jersey Devils": "New Jersey",
    "New York Islanders": "NY Islanders",
    "New York Rangers": "NY Rangers",
    "Ottawa Senators": "Ottawa",
    "Philadelphia Flyers": "Philadelphia",
    "Pittsburgh Penguins": "Pittsburgh",
    "San Jose Sharks": "San Jose",
    "Seattle Kraken": "Seattle",
    "St. Louis Blues": "St. Louis",
    "Tampa Bay Lightning": "Tampa Bay",
    "Toronto Maple Leafs": "Toronto",
    "Utah Hockey Club": "Utah",
    "Utah Mammoth": "Utah", # Temporary/mock name sometimes used
    "Vancouver Canucks": "Vancouver",
    "Vegas Golden Knights": "Vegas",
    "Washington Capitals": "Washington",
    "Winnipeg Jets": "Winnipeg",

    # NFL Full to Short (Common ones)
    "Arizona Cardinals": "Arizona",
    "Atlanta Falcons": "Atlanta",
    "Baltimore Ravens": "Baltimore",
    "Buffalo Bills": "Buffalo",
    "Carolina Panthers": "Carolina",
    "Chicago Bears": "Chicago",
    "Cincinnati Bengals": "Cincinnati",
    "Cleveland Browns": "Cleveland",
    "Dallas Cowboys": "Dallas",
    "Denver Broncos": "Denver",
    "Detroit Lions": "Detroit",
    "Green Bay Packers": "Green Bay",
    "Houston Texans": "Houston",
    "Indianapolis Colts": "Indianapolis",
    "Jacksonville Jaguars": "Jacksonville",
    "Kansas City Chiefs": "Kansas City",
    "Las Vegas Raiders": "Las Vegas",
    "Los Angeles Chargers": "L.A. Chargers",
    "Los Angeles Rams": "L.A. Rams",
    "Miami Dolphins": "Miami",
    "Minnesota Vikings": "Minnesota",
    "New England Patriots": "New England",
    "New Orleans Saints": "New Orleans",
    "New York Giants": "NY Giants",
    "New York Jets": "NY Jets",
    "Philadelphia Eagles": "Philadelphia",
    "Pittsburgh Steelers": "Pittsburgh",
    "San Francisco 49ers": "San Francisco",
    "Seattle Seahawks": "Seattle",
    "Tampa Bay Buccaneers": "Tampa Bay",
    "Tennessee Titans": "Tennessee",
    "Washington Commanders": "Washington",

    # NCAAB Full to Short
    "Duke Blue Devils": "Duke",
    "North Carolina Tar Heels": "North Carolina",
    "Kentucky Wildcats": "Kentucky",
    "Kansas Jayhawks": "Kansas",
    "UConn Huskies": "UConn",
    "Connecticut Huskies": "UConn",
    "Purdue Boilermakers": "Purdue",
    "Houston Cougars": "Houston",
    "Tennessee Volunteers": "Tennessee",
    "Arizona Wildcats": "Arizona",
    "Marquette Golden Eagles": "Marquette",
    "Iowa State Cyclones": "Iowa State",
    "Creighton Bluejays": "Creighton",
    "Illinois Fighting Illini": "Illinois",
    "Baylor Bears": "Baylor",
    "Auburn Tigers": "Auburn",
    "Gonzaga Bulldogs": "Gonzaga",
    "Alabama Crimson Tide": "Alabama",
    "San Diego State Aztecs": "San Diego State",
    "Utah State Aggies": "Utah State",
    "Florida Gators": "Florida",
    "BYU Cougars": "BYU",
    "Brigham Young Cougars": "BYU",
    "Saint Mary's Gaels": "Saint Mary's",
    "Texas Longhorns": "Texas",
    "Washington State Cougars": "Washington State",
    "South Carolina Gamecocks": "South Carolina",
    "Dayton Flyers": "Dayton",
    "Nevada Wolf Pack": "Nevada",
    "Texas Tech Red Raiders": "Texas Tech",
    "Clemson Tigers": "Clemson",
    "New Mexico Lobos": "New Mexico",
    "Mississippi State Bulldogs": "Mississippi State",
    "Michigan State Spartans": "Michigan State",
    "Texas A&M Aggies": "Texas A&M",
    "Nebraska Cornhuskers": "Nebraska",
    "Florida Atlantic Owls": "Florida Atlantic",
    "FAU Owls": "Florida Atlantic",
    "Grand Canyon Antelopes": "Grand Canyon",
    "Drake Bulldogs": "Drake",
    "Colorado Buffaloes": "Colorado",
    "TCU Horned Frogs": "TCU",
    "Northwestern Wildcats": "Northwestern",
    "Boise State Broncos": "Boise State",
    "James Madison Dukes": "James Madison",
    "Oregon Ducks": "Oregon",
    "NC State Wolfpack": "NC State",
    "Colorado State Rams": "Colorado State",
    "St. John's Red Storm": "St. John's",
    "Virginia Cavaliers": "Virginia",
    "Syracuse Orange": "Syracuse",
    "Villanova Wildcats": "Villanova",
    "Ohio State Buckeyes": "Ohio State",
    "Michigan Wolverines": "Michigan",
    "Indiana Hoosiers": "Indiana",
    "UCLA Bruins": "UCLA",
    "USC Trojans": "USC",
    "Arkansas Razorbacks": "Arkansas",
    "Memphis Tigers": "Memphis",
    "Wisconsin Badgers": "Wisconsin",
    "Miami Hurricanes": "Miami",
    "Rutgers Scarlet Knights": "Rutgers",
    "Providence Friars": "Providence",
    "Maryland Terrapins": "Maryland",
    "Iowa Hawkeyes": "Iowa",
    "Xavier Musketeers": "Xavier",
    "Cincinnati Bearcats": "Cincinnati",
    "Wake Forest Demon Deacons": "Wake Forest",
    "Pittsburgh Panthers": "Pittsburgh",
    "Florida State Seminoles": "Florida State",
    "Notre Dame Fighting Irish": "Notre Dame",
    "Stanford Cardinal": "Stanford",
    "Georgetown Hoyas": "Georgetown",
    "Boston College Eagles": "Boston College",
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Penn State Nittany Lions": "Penn State",
    "Missouri Tigers": "Missouri",
    "Oklahoma Sooners": "Oklahoma",
    "Oklahoma State Cowboys": "Oklahoma State",
    "Kansas State Wildcats": "Kansas State",
    "West Virginia Mountaineers": "West Virginia",
    "Seton Hall Pirates": "Seton Hall",
    "DePaul Blue Demons": "DePaul",
    "St. John's Red Storm": "St. John's",
    "Ole Miss Rebels": "Ole Miss",
    "Vanderbilt Commodores": "Vanderbilt",
    "Georgia Bulldogs": "Georgia",
    "LSU Tigers": "LSU",
    "Arizona State Sun Devils": "Arizona State",
    "Washington Huskies": "Washington",
    "Oregon State Beavers": "Oregon State",
    "California Golden Bears": "California",
    "Colorado Buffaloes": "Colorado",
    "Utah Utes": "Utah",
    "Miamiflorida": "Miami FL",
    "Miami Oh": "Miami OH",
    "Texasarlington": "UT Arlington",
    "Texas Christian": "TCU",
    "Central Florida": "UCF",
    "Ny Rangers": "New York Rangers",
    "Massachusetts": "UMass",
    "Connecticut": "UConn",
}

# Merge dynamic aliases into the primary mapping dictionary
TEAM_MAP.update(load_dynamic_aliases())

def aggressive_sanitize_team_name(name: str) -> str:
    """
    Tier 2: Aggressive Lexical Sanitization and Tokenization
    Strips all non-alphanumeric characters, structural punctuation,
    and redundant descriptors (e.g., 'University', 'College', 'Team', 'FC', 'State', 'St').
    Returns a pristine, tokenized string for robust entity resolution.
    """
    if name is None or not isinstance(name, str):
        return ""

    # Lowercase
    s = name.lower()

    # Remove structural punctuation
    s = re.sub(r'[^\w\s]', ' ', s)

    # Strip redundant descriptors
    descriptors = [r'\buniversity\b', r'\bcollege\b', r'\bteam\b', r'\bfc\b', r'\bstate\b', r'\bst\b']
    for desc in descriptors:
        s = re.sub(desc, '', s)

    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_team_name(name: str) -> str:
    """
    Normalize team names for consistent matching.

    Handles:
    - Case normalization (lowercase)
    - Common abbreviations (L.A. → los angeles, St. → state)
    - Punctuation removal
    - Extra whitespace

    Examples:
        "L.A. Lakers" → "los angeles lakers"
        "Northwestern St." → "northwestern state"
        "St. Bonaventure" → "saint bonaventure"
    """
    if name is None or not isinstance(name, str):
        return str(name) if name is not None else ""

    # First apply legacy exact mappings
    name = TEAM_MAP.get(name.strip(), name.strip())

    # Convert to lowercase
    normalized = name.lower()

    # Track if team name is missing from exact map and is likely a long-form API name
    if name not in TEAM_MAP and name != "Over" and name != "Under":
        # Attempt Probabilistic Matching Fallback with rapidfuzz using token_sort_ratio
        if 'fuzz' in globals():
            best_match = None
            best_score = 0

            # Use unique target names from TEAM_MAP values as the schedule names pool
            # Add existing keys as well, to map against valid long-form names
            schedule_names = list(set(TEAM_MAP.values()) | set(TEAM_MAP.keys()))

            for schedule_name in schedule_names:
                score = fuzz.token_sort_ratio(name, schedule_name)
                if score >= 85 and score > best_score:
                    best_score = score
                    best_match = schedule_name

            if best_match:
                # Map to the short form if it was a key, otherwise it's already a short form
                return TEAM_MAP.get(best_match, best_match)

        # If rapidfuzz doesn't find a match, just proceed with regex normalization
        pass

    # Expand common abbreviations BEFORE removing punctuation

    # Use word boundaries to avoid partial matches
    replacements = [
        (r'\bl\.a\.\s', 'los angeles '),
        (r'\bla\s', 'los angeles '),
        (r'^st\.\s', 'saint '),   # "St." at beginning (Saint) - must be before \bst\.\s
        (r'\bst\.\s', 'state '),  # "St." at end or before space
        (r'\bst\.$', 'state'),    # "St." at end of string
        (r'\bn\.c\.\s', 'north carolina '),
        (r'\bunc\s', 'north carolina '),
        (r'\bunc$', 'north carolina'),
        (r'\bs\.c\.\s', 'south carolina '),
        (r'\bu\.c\.\s', 'uc '),
        (r'\bpenn st\b', 'penn state'),
    ]

    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)

    # Remove all punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Collapse multiple spaces to single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized.title()
