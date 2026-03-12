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
    "fiu panthers": "FIU",
    "uc santa barbara": "UCSB",
    "texas-el paso": "UTEP",

    # NBA Full to Short
    "atlanta hawks": "Atlanta",
    "boston celtics": "Boston",
    "brooklyn nets": "Brooklyn",
    "charlotte hornets": "Charlotte",
    "chicago bulls": "Chicago",
    "cleveland cavaliers": "Cleveland",
    "dallas mavericks": "Dallas",
    "denver nuggets": "Denver",
    "detroit pistons": "Detroit",
    "golden state warriors": "Golden State",
    "houston rockets": "Houston",
    "indiana pacers": "Indiana",
    "la clippers": "LA Clippers",
    "l.a. clippers": "LA Clippers",
    "los angeles clippers": "LA Clippers",
    "la lakers": "L.A. Lakers",
    "los angeles lakers": "L.A. Lakers",
    "memphis grizzlies": "Memphis",
    "miami heat": "Miami",
    "milwaukee bucks": "Milwaukee",
    "minnesota timberwolves": "Minnesota",
    "new orleans pelicans": "New Orleans",
    "new york knicks": "New York",
    "oklahoma city thunder": "Oklahoma City",
    "orlando magic": "Orlando",
    "philadelphia 76ers": "Philadelphia",

    # Missing explicit mappings from logs
    "los angeles salle": "La Salle",
    "cal poly slo": "Cal Poly",
    "unlv": "UNLV",
    "uc san diego": "UC San Diego",
    "uc davis": "UC Davis",
    "l.a. lakers": "Los Angeles Lakers",
    "ny rangers": "New York Rangers",
    "ny jets": "New York Jets",
    "ny islanders": "New York Islanders",
    "penn st.": "Penn State",
    "missouri st.": "Missouri State",
    "florida st seminoles": "Florida State",
    "manhattan jaspers": "Manhattan",
    "jacksonville st.": "Jacksonville State",
    "prairie view a&m": "Prairie View A&M",
    "virginia tech": "Virginia Tech",
    "virginia tech hokies": "Virginia Tech",
    "phoenix suns": "Phoenix",
    "portland trail blazers": "Portland",
    "sacramento kings": "Sacramento",
    "san antonio spurs": "San Antonio",
    "toronto raptors": "Toronto",
    "utah jazz": "Utah",
    "washington wizards": "Washington",

    # NHL Full to Short
    "anaheim ducks": "Anaheim",
    "arizona coyotes": "Arizona",
    "boston bruins": "Boston",
    "buffalo sabres": "Buffalo",
    "calgary flames": "Calgary",
    "carolina hurricanes": "Carolina",
    "chicago blackhawks": "Chicago",
    "colorado avalanche": "Colorado",
    "columbus blue jackets": "Columbus",
    "dallas stars": "Dallas",
    "detroit red wings": "Detroit",
    "edmonton oilers": "Edmonton",
    "florida panthers": "Florida",
    "los angeles kings": "Los Angeles",
    "minnesota wild": "Minnesota",
    "montreal canadiens": "Montreal",
    "montréal canadiens": "Montreal",
    "nashville predators": "Nashville",
    "new jersey devils": "New Jersey",
    "new york islanders": "NY Islanders",
    "new york rangers": "NY Rangers",
    "ottawa senators": "Ottawa",
    "philadelphia flyers": "Philadelphia",
    "pittsburgh penguins": "Pittsburgh",
    "san jose sharks": "San Jose",
    "seattle kraken": "Seattle",
    "st. louis blues": "St. Louis",
    "tampa bay lightning": "Tampa Bay",
    "toronto maple leafs": "Toronto",
    "utah hockey club": "Utah",
    "utah mammoth": "Utah", # Temporary/mock name sometimes used
    "vancouver canucks": "Vancouver",
    "vegas golden knights": "Vegas",
    "washington capitals": "Washington",
    "winnipeg jets": "Winnipeg",
    "winnipeg": "Winnipeg Jets",
    "ny rangers": "New York Rangers",

    # NFL Full to Short (Common ones)
    "arizona cardinals": "Arizona",
    "atlanta falcons": "Atlanta",
    "baltimore ravens": "Baltimore",
    "buffalo bills": "Buffalo",
    "carolina panthers": "Carolina",
    "chicago bears": "Chicago",
    "cincinnati bengals": "Cincinnati",
    "cleveland browns": "Cleveland",
    "dallas cowboys": "Dallas",
    "denver broncos": "Denver",
    "detroit lions": "Detroit",
    "green bay packers": "Green Bay",
    "houston texans": "Houston",
    "indianapolis colts": "Indianapolis",
    "jacksonville jaguars": "Jacksonville",
    "kansas city chiefs": "Kansas City",
    "las vegas raiders": "Las Vegas",
    "los angeles chargers": "L.A. Chargers",
    "los angeles rams": "L.A. Rams",
    "miami dolphins": "Miami",
    "minnesota vikings": "Minnesota",
    "new england patriots": "New England",
    "new orleans saints": "New Orleans",
    "new york giants": "NY Giants",
    "new york jets": "NY Jets",
    "philadelphia eagles": "Philadelphia",
    "pittsburgh steelers": "Pittsburgh",
    "san francisco 49ers": "San Francisco",
    "seattle seahawks": "Seattle",
    "tampa bay buccaneers": "Tampa Bay",
    "tennessee titans": "Tennessee",
    "washington commanders": "Washington",

    # NCAAB Full to Short
    "duke blue devils": "Duke",
    "north carolina tar heels": "North Carolina",
    "kentucky wildcats": "Kentucky",
    "kansas jayhawks": "Kansas",
    "uconn huskies": "UConn",
    "connecticut huskies": "UConn",
    "purdue boilermakers": "Purdue",
    "houston cougars": "Houston",
    "tennessee volunteers": "Tennessee",
    "arizona wildcats": "Arizona",
    "marquette golden eagles": "Marquette",
    "iowa state cyclones": "Iowa State",
    "creighton bluejays": "Creighton",
    "illinois fighting illini": "Illinois",
    "baylor bears": "Baylor",
    "auburn tigers": "Auburn",
    "gonzaga bulldogs": "Gonzaga",
    "alabama crimson tide": "Alabama",
    "san diego state aztecs": "San Diego State",
    "utah state aggies": "Utah State",
    "florida gators": "Florida",
    "byu cougars": "BYU",
    "brigham young cougars": "BYU",
    "saint mary's gaels": "Saint Mary's",
    "texas longhorns": "Texas",
    "washington state cougars": "Washington State",
    "south carolina gamecocks": "South Carolina",
    "dayton flyers": "Dayton",
    "nevada wolf pack": "Nevada",
    "texas tech red raiders": "Texas Tech",
    "clemson tigers": "Clemson",
    "new mexico lobos": "New Mexico",
    "mississippi state bulldogs": "Mississippi State",
    "michigan state spartans": "Michigan State",
    "texas a&m aggies": "Texas A&M",
    "nebraska cornhuskers": "Nebraska",
    "florida atlantic owls": "Florida Atlantic",
    "fau owls": "Florida Atlantic",
    "grand canyon antelopes": "Grand Canyon",
    "drake bulldogs": "Drake",
    "colorado buffaloes": "Colorado",
    "tcu horned frogs": "TCU",
    "northwestern wildcats": "Northwestern",
    "boise state broncos": "Boise State",
    "james madison dukes": "James Madison",
    "oregon ducks": "Oregon",
    "nc state wolfpack": "NC State",
    "colorado state rams": "Colorado State",
    "st. john's red storm": "St. John's",
    "virginia cavaliers": "Virginia",
    "syracuse orange": "Syracuse",
    "villanova wildcats": "Villanova",
    "ohio state buckeyes": "Ohio State",
    "michigan wolverines": "Michigan",
    "indiana hoosiers": "Indiana",
    "ucla bruins": "UCLA",
    "usc trojans": "USC",
    "arkansas razorbacks": "Arkansas",
    "memphis tigers": "Memphis",
    "wisconsin badgers": "Wisconsin",
    "miami hurricanes": "Miami",
    "rutgers scarlet knights": "Rutgers",
    "providence friars": "Providence",
    "maryland terrapins": "Maryland",
    "iowa hawkeyes": "Iowa",
    "xavier musketeers": "Xavier",
    "cincinnati bearcats": "Cincinnati",
    "wake forest demon deacons": "Wake Forest",
    "pittsburgh panthers": "Pittsburgh",
    "florida state seminoles": "Florida State",
    "notre dame fighting irish": "Notre Dame",
    "stanford cardinal": "Stanford",
    "georgetown hoyas": "Georgetown",
    "boston college eagles": "Boston College",
    "georgia tech yellow jackets": "Georgia Tech",
    "penn state nittany lions": "Penn State",
    "missouri tigers": "Missouri",
    "oklahoma sooners": "Oklahoma",
    "oklahoma state cowboys": "Oklahoma State",
    "kansas state wildcats": "Kansas State",
    "west virginia mountaineers": "West Virginia",
    "seton hall pirates": "Seton Hall",
    "depaul blue demons": "DePaul",
    "st. john's red storm": "St. John's",
    "ole miss rebels": "Ole Miss",
    "vanderbilt commodores": "Vanderbilt",
    "georgia bulldogs": "Georgia",
    "lsu tigers": "LSU",
    "arizona state sun devils": "Arizona State",
    "washington huskies": "Washington",
    "oregon state beavers": "Oregon State",
    "california golden bears": "California",
    "colorado buffaloes": "Colorado",
    "utah utes": "Utah",
    "miamiflorida": "Miami FL",
    "miami oh": "miami (oh)",
    "texasarlington": "UT Arlington",
    "texas christian": "TCU",
    "central florida": "UCF",
    "ny rangers": "new york rangers",
    "massachusetts": "umass",
    "connecticut": "UConn",
    "umass": "massachusetts",
    "missouri": "mizzou",
    "mizzou": "missouri",
    "saint bonaventure": "st. bonaventure",
    "st bonaventure": "st. bonaventure",
    "miami oh": "miami (oh)",
    "bowling green": "bgsu",
    "bgsu": "bowling green",
    "alabama am": "alabama a&m",
    "st. bonaventure": "st. bonaventure",
    "saint bonaventure": "st. bonaventure",
    "massachusetts": "umass",
    "missouri": "mizzou",
    "ny rangers": "new york rangers",
    "winnipeg": "winnipeg jets",
    "kentucky": "kentucky",
    "texas tech": "texas tech",
    "iowa state": "iowa state",
    "winnipeg": "winnipeg jets",
    "saint bonaventure": "st. bonaventure",
    "massachusetts": "umass",
    "miami oh": "miami (oh)",
    "mizzou": "missouri",
    "alabama am": "alabama a&m",
    "bgsu": "bowling green",
    "ucf": "central florida",
    "miami fl": "miami (fl)",
    "saint bonaventure": "st. bonaventure",
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

    # Baseline Python string methods
    s = name.lower().replace("university", "").replace("college", "").replace("team", "").replace("fc", "").replace("state", "").replace("st", "").strip()

    # Single simple regex pass to remove non-alphanumeric characters
    s = re.sub(r'[^a-z0-9]', ' ', s)

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

    # 1. Apply strip and lower immediately to the incoming parameter
    cleaned_name = name.strip().lower()

    # 2. Perform dictionary lookup on the explicitly cleaned string
    if cleaned_name in TEAM_MAP:
        name = TEAM_MAP[cleaned_name]
    else:
        name = name.strip()

    # Convert to lowercase
    normalized = name.lower()

    # Track if team name is missing from exact map and is likely a long-form API name
    # We also check the lowercase name to ensure missing logic works properly with lowercased map
    if name not in TEAM_MAP and name != "Over" and name != "Under" and cleaned_name not in TEAM_MAP:
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
