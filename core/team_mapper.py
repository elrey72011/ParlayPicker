"""
Team name normalization for consistent matching across data sources.
"""
import re

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
    "LA Clippers": "L.A. Clippers",
    "Los Angeles Clippers": "L.A. Clippers",
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
}

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
