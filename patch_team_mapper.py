import re

content = """
import logging

logger = logging.getLogger(__name__)

# Track missing keys for terminal warnings
_MISSING_KEYS_WARNED = set()
"""

with open("core/team_mapper.py", "r") as f:
    orig_content = f.read()

# Add logging import
orig_content = orig_content.replace('import re\n', 'import re\n' + content)

# Enhance TEAM_MAP with major NCAAB programs
ncaab_map = """
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
}
"""

orig_content = orig_content.replace('    "Washington Commanders": "Washington",\n}', '    "Washington Commanders": "Washington",\n' + ncaab_map + '}')


new_norm = """
    # Convert to lowercase
    normalized = name.lower()

    # Track if team name is missing from exact map and is likely a long-form API name
    if name not in TEAM_MAP and name != "Over" and name != "Under" and len(name.split()) > 1:
        if name not in _MISSING_KEYS_WARNED:
            logger.warning(f"Warning: Team '{name}' not found in dictionary mapping, falling back to substring normalization.")
            _MISSING_KEYS_WARNED.add(name)

    # Expand common abbreviations BEFORE removing punctuation
"""

orig_content = orig_content.replace("""
    # Convert to lowercase
    normalized = name.lower()

    # Expand common abbreviations BEFORE removing punctuation""", new_norm)

with open("core/team_mapper.py", "w") as f:
    f.write(orig_content)
