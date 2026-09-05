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

KALSHI_NCAAB_OVERRIDES = {
    "ucf": "ucf",
    "wright st": "wright state",
    "queens university": "queens nc",
    "queens university of charlotte": "queens nc",
    "liu sharks": "long island university",
    "furman": "furman", # ensure standard capitalization doesn't break it
    "iowa": "iowa",
}

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
NBA_EXACT_MAP = {
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
    "indiana": "Indiana",
    "los angeles clippers": "LA Clippers",
    "la clippers": "LA Clippers",
    "los angeles lakers": "Los Angeles Lakers",
    "la lakers": "Los Angeles Lakers",
    "memphis grizzlies": "Memphis",
    "memphis": "Memphis",
    "miami heat": "Miami",
    "milwaukee bucks": "Milwaukee",
    "minnesota timberwolves": "Minnesota",
    "new orleans pelicans": "New Orleans",
    "new york knicks": "New York",
    "oklahoma city thunder": "Oklahoma City",
    "orlando magic": "Orlando",
    "philadelphia 76ers": "Philadelphia",
    "phoenix suns": "Phoenix",
    "portland trail blazers": "Portland",
    "sacramento kings": "Sacramento",
    "san antonio spurs": "San Antonio",
    "toronto raptors": "Toronto",
    "utah jazz": "Utah",
    "washington wizards": "Washington",
}


# WNBA uses full franchise names in The Odds API while TheOver commonly exports
# city-only names. Normalize both representations to the same stable identity.
# Includes the Portland Fire and Toronto Tempo, which joined for the 2026 season.
WNBA_EXACT_MAP = {
    "atlanta dream": "Atlanta",
    "chicago sky": "Chicago",
    "connecticut sun": "Connecticut",
    "dallas wings": "Dallas",
    "golden state valkyries": "Golden State",
    "indiana fever": "Indiana",
    "las vegas aces": "Las Vegas",
    "los angeles sparks": "Los Angeles",
    "minnesota lynx": "Minnesota",
    "new york liberty": "New York",
    "phoenix mercury": "Phoenix",
    "portland fire": "Portland",
    "seattle storm": "Seattle",
    "toronto tempo": "Toronto",
    "washington mystics": "Washington",
}

NHL_EXACT_MAP = {
    "anaheim ducks": "Anaheim",
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
    "la kings": "Los Angeles",
    "minnesota wild": "Minnesota",
    "montreal canadiens": "Montreal",
    "nashville predators": "Nashville",
    "new jersey devils": "New Jersey",
    "new york islanders": "New York",
    "new york rangers": "New York",
    "ottawa senators": "Ottawa",
    "philadelphia flyers": "Philadelphia",
    "pittsburgh penguins": "Pittsburgh",
    "san jose sharks": "San Jose",
    "seattle kraken": "Seattle",
    "st. louis blues": "St. Louis",
    "st louis blues": "St. Louis",
    "tampa bay lightning": "Tampa Bay",
    "toronto maple leafs": "Toronto",
    "utah hockey club": "Utah",
    "vancouver canucks": "Vancouver",
    "vegas golden knights": "Vegas",
    "washington capitals": "Washington",
    "winnipeg jets": "Winnipeg",
}

NCAAB_EXTRA_MAP = {
    "prairie view a&m panthers": "Prairie View Am",
    "utah tech trailblazers": "Utah Tech",
    "fiu panthers": "Florida Intl",
    "fau owls": "Florida Atlantic",
    "north carolina wilmington": "UNC Wilmington",
    "michigan st spartans": "Michigan State",
    "saint louis": "St. Louis",
    "st. louis": "St. Louis",
    "st louis": "St. Louis",
    "north carolina state": "NC State",
    "nc state": "NC State",
}

# Consolidated Production Mascot Bridge
PRODUCTION_MASCOT_MAP = {
    # Pro Teams
    "utah": "Utah",
    "philly": "Philadelphia",
    "la lakers": "Los Angeles",
    "la clippers": "LA",
    "okc": "Oklahoma City",
    "montreal": "Montreal",
    "san jose": "San Jose",
    "ny islanders": "New York",
    "vancouver": "Vancouver",

    # Tournament Heavyweights (Requested)
    "unc": "North Carolina",
    "nc state": "NC State",
    "ncsu": "NC State",
    "purdue": "Purdue",
    "arizona": "Arizona",
    "tennessee": "Tennessee",

    # Additional High-Volume Tournament Programs
    "uconn": "UConn",
    "connecticut": "UConn",
    "u of c": "UConn",
    "duke": "Duke",
    "kansas": "Kansas",
    "kentucky": "Kentucky",
    "alabama": "Alabama",
    "houston": "Houston",
    "nebraska": "Nebraska"
}
TEAM_MAP = {}
TEAM_MAP.update(PRODUCTION_MASCOT_MAP)

TEAM_MAP.update({
    "north carolina wilmington": "UNC Wilmington",
    "michigan st spartans": "Michigan State",
    "saint louis": "St. Louis",
    "st. louis": "St. Louis",
    "st louis": "St. Louis",
    "north carolina state": "NC State",
    "nc state": "NC State",
    "yale": "Yale",
    "yale bulldogs": "Yale",
    "george mason": "George Mason",
    "george mason patriots": "George Mason",
    "gmu": "George Mason",
    "uconn": "Uconn",
    "connecticut": "Uconn",
    "ucf": "UCF",
    "wright st": "Wright State",
    "wright state": "Wright State",
    "queens university": "Queens NC",
    "queens nc": "Queens NC",
    "queens university of charlotte": "Queens NC",
    "liu sharks": "Long Island University",
    "liu": "Long Island University",
    "long island university": "Long Island University",
    "g-state": "Golden State",
    "g-state warriors": "Golden State",
    "lehigh": "Lehigh",
    "navy": "Navy",
    "gw revolutionaries": "George Washington",
    "gw": "George Washington",
    "george washington": "George Washington",
    "revolutionaries": "George Washington",
    "uic": "UIC",
    "uvu": "Utah Valley",
    "utah valley": "Utah Valley",
    "valley": "Utah Valley",
    "alabama am": "alabama a&m",
    "alabama crimson tide": "Alabama",
    "anaheim ducks": "Anaheim",
    "arizona cardinals": "Arizona",
    "arizona coyotes": "Arizona",
    "arizona state sun devils": "Arizona State",
    "arizona wildcats": "Arizona",
    "arkansas razorbacks": "Arkansas",
    "atlanta falcons": "Atlanta",
    "atlanta hawks": "Atlanta",
    "auburn tigers": "Auburn",
    "baltimore ravens": "Baltimore",
    "baylor bears": "Baylor",
    "bgsu": "bowling green",
    "boise state broncos": "Boise State",
    "boston bruins": "Boston",
    "boston celtics": "Boston",
    "boston college eagles": "Boston College",
    "bowling green": "bgsu",
    "brigham young cougars": "BYU",
    "brooklyn nets": "Brooklyn",
    "buffalo bills": "Buffalo",
    "buffalo sabres": "Buffalo",
    "byu cougars": "BYU",
    "cal poly slo": "Cal Poly",
    "calgary flames": "Calgary",
    "california baptist lancers": "Ca Baptist",
    "california golden bears": "California",
    "carolina hurricanes": "Carolina",
    "carolina panthers": "Carolina",
    "central florida": "UCF",
    "charlotte hornets": "Charlotte",
    "chicago bears": "Chicago",
    "chicago blackhawks": "Chicago",
    "chicago bulls": "Chicago",
    "cincinnati bearcats": "Cincinnati",
    "cincinnati bengals": "Cincinnati",
    "clemson tigers": "Clemson",
    "cleveland browns": "Cleveland",
    "cleveland cavaliers": "Cleveland",
    "colorado avalanche": "Colorado",
    "colorado buffaloes": "Colorado",
    "colorado state rams": "Colorado State",
    "columbus blue jackets": "Columbus",
    "connecticut": "Uconn",
    "connecticut huskies": "Uconn",
    "creighton bluejays": "Creighton",
    "dallas cowboys": "Dallas",
    "dallas mavericks": "Dallas",
    "dallas stars": "Dallas",
    "dayton flyers": "Dayton",
    "denver broncos": "Denver",
    "denver nuggets": "Denver",
    "depaul blue demons": "DePaul",
    "detroit lions": "Detroit",
    "detroit pistons": "Detroit",
    "detroit red wings": "Detroit",
    "drake bulldogs": "Drake",
    "duke blue devils": "Duke",
    "edmonton oilers": "Edmonton",
    "fau owls": "Florida Atlantic",
    "fiu panthers": "FIU",
    "florida atlantic owls": "Florida Atlantic",
    "florida gators": "Florida",
    "florida panthers": "Florida",
    "florida st seminoles": "Florida State",
    "florida state seminoles": "Florida State",
    "georgetown hoyas": "Georgetown",
    "georgia bulldogs": "Georgia",
    "georgia tech yellow jackets": "Georgia Tech",
    "golden state warriors": "Golden State",
    "gonzaga bulldogs": "Gonzaga",
    "grand canyon antelopes": "Grand Canyon",
    "green bay packers": "Green Bay",
    "houston cougars": "Houston",
    "houston rockets": "Houston",
    "houston texans": "Houston",
    "illinois fighting illini": "Illinois",
    "indiana hoosiers": "Indiana",
    "indiana pacers": "Indiana",
    "indiana": "Indiana",
    "indianapolis colts": "Indianapolis",
    "iowa hawkeyes": "Iowa",
    "iowa state": "iowa state",
    "iowa state cyclones": "Iowa State",
    "jacksonville jaguars": "Jacksonville",
    "jacksonville st.": "Jacksonville State",
    "james madison dukes": "James Madison",
    "kansas city chiefs": "Kansas City",
    "kansas jayhawks": "Kansas",
    "kansas state wildcats": "Kansas State",
    "kentucky": "kentucky",
    "kentucky wildcats": "Kentucky",
    "l.a. clippers": "LA Clippers",
    "l.a. lakers": "Los Angeles Lakers",
    "la clippers": "LA Clippers",
    "la lakers": "L.A. Lakers",
    "las vegas raiders": "Las Vegas",
    "los angeles chargers": "L.A. Chargers",
    "los angeles clippers": "LA Clippers",
    "los angeles kings": "Los Angeles",
    "los angeles lakers": "L.A. Lakers",
    "los angeles rams": "L.A. Rams",
    "los angeles salle": "La Salle",
    "lsu tigers": "LSU",
    "manhattan jaspers": "Manhattan",
    "marquette golden eagles": "Marquette",
    "maryland terrapins": "Maryland",
    "massachusetts": "umass",
    "memphis grizzlies": "Memphis",
    "memphis tigers": "Memphis",
    "memphis": "Memphis",
    "miami (fl)": "Miami Fl",
    "miami (fl) hurricanes": "Miami Fl",
    "miami (oh) redhawks": "Miami Oh",
    "miami dolphins": "Miami",
    "miami fl": "Miami Fl",
    "miami florida hurricanes": "Miami Fl",
    "miami heat": "Miami",
    "miami hurricanes": "Miami",
    "miami oh": "miami (oh)",
    "miamiflorida": "Miami Fl",
    "michigan state spartans": "Michigan State",
    "michigan wolverines": "Michigan",
    "milwaukee bucks": "Milwaukee",
    "minnesota timberwolves": "Minnesota",
    "minnesota vikings": "Minnesota",
    "minnesota wild": "Minnesota",
    "mississippi state bulldogs": "Mississippi State",
    "missouri": "mizzou",
    "missouri st.": "Missouri State",
    "missouri tigers": "Missouri",
    "mizzou": "missouri",
    "montreal canadiens": "Montreal",
    "montréal canadiens": "Montreal",
    "nashville predators": "Nashville",
    "nc state wolfpack": "NC State",
    "nebraska cornhuskers": "Nebraska",
    "nevada wolf pack": "Nevada",
    "new england patriots": "New England",
    "new jersey devils": "New Jersey",
    "new mexico lobos": "New Mexico",
    "new orleans pelicans": "New Orleans",
    "new orleans saints": "New Orleans",
    "new york giants": "NY Giants",
    "new york islanders": "New York",
    "new york jets": "NY Jets",
    "new york knicks": "New York",
    "new york rangers": "New York",
    "north carolina tar heels": "North Carolina",
    "north carolina state": "NC State",
    "nc state": "NC State",
    "northwestern wildcats": "Northwestern",
    "notre dame fighting irish": "Notre Dame",
    "ny islanders": "New York",
    "ny jets": "New York Jets",
    "ny rangers": "New York",
    "ohio state buckeyes": "Ohio State",
    "oklahoma city thunder": "Oklahoma City",
    "oklahoma sooners": "Oklahoma",
    "oklahoma state cowboys": "Oklahoma State",
    "ole miss rebels": "Ole Miss",
    "oregon ducks": "Oregon",
    "oregon state beavers": "Oregon State",
    "orlando magic": "Orlando",
    "ottawa senators": "Ottawa",
    "penn st.": "Penn State",
    "penn state nittany lions": "Penn State",
    "philadelphia 76ers": "Philadelphia",
    "philadelphia eagles": "Philadelphia",
    "philadelphia flyers": "Philadelphia",
    "phoenix suns": "Phoenix",
    "pittsburgh panthers": "Pittsburgh",
    "pittsburgh penguins": "Pittsburgh",
    "pittsburgh steelers": "Pittsburgh",
    "portland trail blazers": "Portland",
    "prairie view a&m": "Prairie View A&M",
    "providence friars": "Providence",
    "purdue boilermakers": "Purdue",
    "rutgers scarlet knights": "Rutgers",
    "sacramento kings": "Sacramento",
    "saint bonaventure": "st. bonaventure",
    "saint mary's gaels": "Saint Mary's",
    "san antonio spurs": "San Antonio",
    "san diego state aztecs": "San Diego State",
    "san francisco 49ers": "San Francisco",
    "san jose sharks": "San Jose",
    "seattle kraken": "Seattle",
    "seattle seahawks": "Seattle",
    "seton hall pirates": "Seton Hall",
    "south carolina gamecocks": "South Carolina",
    "st bonaventure": "st. bonaventure",
    "st. bonaventure": "st. bonaventure",
    "st. john's red storm": "St. John's",
    "st. louis blues": "St. Louis",
    "saint louis": "St. Louis",
    "saint louis billikens": "St. Louis",
    "st louis": "St. Louis",
    "st. louis": "St. Louis",
    "stanford cardinal": "Stanford",
    "syracuse orange": "Syracuse",
    "tampa bay buccaneers": "Tampa Bay",
    "tampa bay lightning": "Tampa Bay",
    "tcu horned frogs": "TCU",
    "tennessee titans": "Tennessee",
    "tennessee volunteers": "Tennessee",
    "texas a&m aggies": "Texas A&M",
    "texas christian": "TCU",
    "texas longhorns": "Texas",
    "texas tech": "texas tech",
    "texas tech red raiders": "Texas Tech",
    "texas-el paso": "UTEP",
    "texasarlington": "UT Arlington",
    "toronto maple leafs": "Toronto",
    "toronto raptors": "Toronto",
    "uc davis": "UC Davis",
    "uc san diego": "UC San Diego",
    "uc santa barbara": "UCSB",
    "central florida": "UCF",
    "ucla bruins": "UCLA",
    "uconn huskies": "Uconn",
    "umass": "massachusetts",
    "unlv": "UNLV",
    "usc trojans": "USC",
    "utah hockey club": "Utah",
    "utah jazz": "Utah",
    "utah mammoth": "Utah",
    "utah state aggies": "Utah State",
    "utah utes": "Utah",
    "vancouver canucks": "Vancouver",
    "vanderbilt commodores": "Vanderbilt",
    "vegas golden knights": "Vegas",
    "villanova wildcats": "Villanova",
    "virginia cavaliers": "Virginia",
    "virginia tech": "Virginia Tech",
    "virginia tech hokies": "Virginia Tech",
    "wake forest demon deacons": "Wake Forest",
    "washington capitals": "Washington",
    "washington commanders": "Washington",
    "washington huskies": "Washington",
    "washington state cougars": "Washington State",
    "washington wizards": "Washington",
    "west virginia mountaineers": "West Virginia",
    "winnipeg": "winnipeg jets",
    "winnipeg jets": "Winnipeg",
    "wisconsin badgers": "Wisconsin",
    "xavier musketeers": "Xavier",
})

# Auto-injected exact mappings for Odds API full mascot names to standard names
ODDS_API_EXACT_MAP = {
    # MLB multi-team cities — full official names from Odds API
    "chicago cubs": "Chicago Cubs",
    "chicago white sox": "Chicago White Sox",
    "new york yankees": "New York Yankees",
    "new york mets": "New York Mets",
    "los angeles dodgers": "Los Angeles Dodgers",
    "los angeles angels": "Los Angeles Angels",
    # TheOver CSV abbreviated forms (e.g. "Chi. Cubs", "LA Dodgers", "NY Mets")
    # Two variants needed: with punctuation (looked up before removal) and without.
    "chi. cubs": "Chicago Cubs",
    "chi. white sox": "Chicago White Sox",
    "chi cubs": "Chicago Cubs",
    "chi white sox": "Chicago White Sox",
    "la dodgers": "Los Angeles Dodgers",
    "la angels": "Los Angeles Angels",
    "ny yankees": "New York Yankees",
    "ny mets": "New York Mets",
    # Athletics — team moved from Oakland; Odds API uses "Athletics" with no city
    "athletics": "Athletics",
    "oakland athletics": "Athletics",
    "sacramento athletics": "Athletics",
    "montreal": "Montreal",
    "san jose": "San Jose",
    "ny islanders": "New York",
    "ny rangers": "New York",
    "ottawa": "Ottawa",
    "utah": "Utah",
    "philly": "Philadelphia",
    "la lakers": "Los Angeles",
    "la clippers": "LA",
    "abilene christian wildcats": "Abilene Christian",
    "air force falcons": "Air Force",
    "akron zips": "Akron",
    "alabama a&m bulldogs": "Alabama A&M",
    "alabama crimson tide": "Alabama",
    "alabama st hornets": "Alabama State",
    "albany great danes": "Albany",
    "alcorn st braves": "Alcorn State",
    "american university eagles": "American University",
    "appalachian st mountaineers": "Appalachian State",
    "arizona st sun devils": "Arizona State",
    "arizona wildcats": "Arizona",
    "arkansas razorbacks": "Arkansas",
    "arkansas st red wolves": "Arkansas State",
    "arkansas-little rock trojans": "Arkansas-Little Rock",
    "arkansas-pine bluff golden lions": "Arkansas-Pine Bluff",
    "army knights": "Army",
    "auburn tigers": "Auburn",
    "austin peay governors": "Austin Peay",
    "ball state cardinals": "Ball State",
    "baylor bears": "Baylor",
    "bellarmine knights": "Bellarmine",
    "belmont bruins": "Belmont",
    "bethune-cookman wildcats": "Bethune-Cookman",
    "binghamton bearcats": "Binghamton",
    "boise state broncos": "Boise State",
    "boston college eagles": "Boston College",
    "boston university terriers": "Boston University",
    "bowling green falcons": "Bowling Green",
    "bradley braves": "Bradley",
    "brown bears": "Brown",
    "bryant bulldogs": "Bryant",
    "bucknell bison": "Bucknell",
    "buffalo bulls": "Buffalo",
    "butler bulldogs": "Butler",
    "byu cougars": "BYU",
    "cal baptist lancers": "Cal Baptist",
    "cal poly mustangs": "Cal Poly",
    "cal state fullerton titans": "Cal State Fullerton",
    "cal state northridge matadors": "Cal State Northridge",
    "california golden bears": "California",
    "campbell fighting camels": "Campbell",
    "canisius golden griffins": "Canisius",
    "central arkansas bears": "Central Arkansas",
    "central connecticut st blue devils": "Central Connecticut State",
    "central michigan chippewas": "Central Michigan",
    "charleston cougars": "Charleston",
    "charleston southern buccaneers": "Charleston Southern",
    "charlotte 49ers": "Charlotte",
    "chattanooga mocs": "Chattanooga",
    "chicago st cougars": "Chicago State",
    "cincinnati bearcats": "Cincinnati",
    "citadel bulldogs": "Citadel",
    "clemson tigers": "Clemson",
    "cleveland st vikings": "Cleveland State",
    "coastal carolina chanticleers": "Coastal Carolina",
    "colgate raiders": "Colgate",
    "college of charleston cougars": "College of Charleston",
    "colorado buffaloes": "Colorado",
    "colorado st rams": "Colorado State",
    "columbia lions": "Columbia",
    "connecticut huskies": "Connecticut",
    "coppin st eagles": "Coppin State",
    "cornell big red": "Cornell",
    "creighton bluejays": "Creighton",
    "dartmouth big green": "Dartmouth",
    "davidson wildcats": "Davidson",
    "dayton flyers": "Dayton",
    "delaware blue hens": "Delaware",
    "delaware st hornets": "Delaware State",
    "denver pioneers": "Denver",
    "depaul blue demons": "DePaul",
    "detroit mercy titans": "Detroit Mercy",
    "drake bulldogs": "Drake",
    "drexel dragons": "Drexel",
    "duke blue devils": "Duke",
    "duquesne dukes": "Duquesne",
    "east carolina pirates": "East Carolina",
    "east tennessee st buccaneers": "East Tennessee State",
    "eastern illinois panthers": "Eastern Illinois",
    "eastern kentucky colonels": "Eastern Kentucky",
    "eastern michigan eagles": "Eastern Michigan",
    "eastern washington eagles": "Eastern Washington",
    "elon phoenix": "Elon",
    "evansville purple aces": "Evansville",
    "fairfield stags": "Fairfield",
    "fairleigh dickinson knights": "Fairleigh Dickinson",
    "fcu knights": "FCU",
    "fgcu eagles": "FGCU",
    "fiu panthers": "FIU",
    "florida a&m rattlers": "Florida A&M",
    "florida atlantic owls": "Florida Atlantic",
    "florida gators": "Florida",
    "florida gulf coast eagles": "Florida Gulf Coast",
    "florida state seminoles": "Florida State",
    "fordham rams": "Fordham",
    "fresno st bulldogs": "Fresno State",
    "furman paladins": "Furman",
    "gardner-webb runnin' bulldogs": "Gardner-Webb",
    "george mason patriots": "George Mason",
    "george washington colonials": "George Washington",
    "george washington revolutionaries": "George Washington",
    "georgetown hoyas": "Georgetown",
    "georgia bulldogs": "Georgia",
    "georgia southern eagles": "Georgia Southern",
    "georgia st panthers": "Georgia State",
    "georgia tech yellow jackets": "Georgia Tech",
    "gonzaga bulldogs": "Gonzaga",
    "grambling st tigers": "Grambling State",
    "grand canyon antelopes": "Grand Canyon",
    "hampton pirates": "Hampton",
    "hartford hawks": "Hartford",
    "harvard crimson": "Harvard",
    "hawaii rainbow warriors": "Hawaii",
    "high point panthers": "High Point",
    "hofstra pride": "Hofstra",
    "holy cross crusaders": "Holy Cross",
    "houston christian huskies": "Houston Christian",
    "houston cougars": "Houston",
    "howard bison": "Howard",
    "idaho st bengals": "Idaho State",
    "idaho vandals": "Idaho",
    "illinois fighting illini": "Illinois",
    "illinois st redbirds": "Illinois State",
    "illinois-chicago flames": "Illinois-Chicago",
    "incarnate word cardinals": "Incarnate Word",
    "indiana hoosiers": "Indiana",
    "indiana st sycamores": "Indiana State",
    "iona gaels": "Iona",
    "iowa hawkeyes": "Iowa",
    "iowa state cyclones": "Iowa State",
    "iupui jaguars": "IUPUI",
    "jackson st tigers": "Jackson State",
    "jacksonville dolphins": "Jacksonville",
    "jacksonville st gamecocks": "Jacksonville State",
    "james madison dukes": "James Madison",
    "kansas city roos": "Kansas City",
    "kansas jayhawks": "Kansas",
    "kansas state wildcats": "Kansas State",
    "kennesaw st owls": "Kennesaw State",
    "kent state golden flashes": "Kent State",
    "kentucky wildcats": "Kentucky",
    "la salle explorers": "La Salle",
    "lafayette leopards": "Lafayette",
    "lamar cardinals": "Lamar",
    "lehigh mountain hawks": "Lehigh",
    "liberty flames": "Liberty",
    "lipscomb bisons": "Lipscomb",
    "long beach st beach": "Long Beach State",
    "long island university sharks": "Long Island University",
    "longwood lancers": "Longwood",
    "louisiana ragin' cajuns": "Louisiana",
    "louisiana tech bulldogs": "Louisiana Tech",
    "louisiana-monroe warhawks": "Louisiana-Monroe",
    "louisville cardinals": "Louisville",
    "loyola chicago ramblers": "Loyola Chicago",
    "loyola marymount lions": "Loyola Marymount",
    "loyola maryland greyhounds": "Loyola Maryland",
    "lsu tigers": "LSU",
    "maine black bears": "Maine",
    "manhattan jaspers": "Manhattan",
    "marist red foxes": "Marist",
    "marquette golden eagles": "Marquette",
    "marshall thundering herd": "Marshall",
    "maryland terrapins": "Maryland",
    "maryland-eastern shore hawks": "Maryland-Eastern Shore",
    "massachusetts minutemen": "Massachusetts",
    "mcneese st cowboys": "McNeese State",
    "memphis tigers": "Memphis",
    "mercer bears": "Mercer",
    "merrimack warriors": "Merrimack",
    "miami (fl) hurricanes": "Miami (FL)",
    "miami (oh) redhawks": "Miami (OH)",
    "miami florida hurricanes": "Miami (FL)",
    "miami ohio redhawks": "Miami (OH)",
    "michigan state spartans": "Michigan State",
    "michigan wolverines": "Michigan",
    "middle tennessee blue raiders": "Middle Tennessee",
    "milwaukee panthers": "Milwaukee",
    "minnesota golden gophers": "Minnesota",
    "mississippi state bulldogs": "Mississippi State",
    "mississippi valley st delta devils": "Mississippi Valley State",
    "missouri state bears": "Missouri State",
    "missouri tigers": "Missouri",
    "monmouth hawks": "Monmouth",
    "montana grizzlies": "Montana",
    "montana st bobcats": "Montana State",
    "morehead st eagles": "Morehead State",
    "morgan st bears": "Morgan State",
    "mount st. mary's mountaineers": "Mount St. Mary's",
    "mt. st. mary's mountaineers": "Mt. St. Mary's",
    "murray st racers": "Murray State",
    "n.c. a&t aggies": "N.C. A&T",
    "n.c. central eagles": "N.C. Central",
    "n colorado bears": "Northern Colorado",
    "navy midshipmen": "Navy",
    "nc state wolfpack": "NC State",
    "nebraska cornhuskers": "Nebraska",
    "nevada wolf pack": "Nevada",
    "new hampshire wildcats": "New Hampshire",
    "new mexico lobos": "New Mexico",
    "new mexico st aggies": "New Mexico State",
    "new orleans privateers": "New Orleans",
    "niagara purple eagles": "Niagara",
    "nicholls st colonels": "Nicholls State",
    "njit highlanders": "NJIT",
    "norfolk st spartans": "Norfolk State",
    "north alabama lions": "North Alabama",
    "north carolina tar heels": "North Carolina",
    "north dakota st bison": "North Dakota State",
    "north dakota fighting hawks": "North Dakota",
    "north florida ospreys": "North Florida",
    "north texas mean green": "North Texas",
    "northeastern huskies": "Northeastern",
    "northern arizona lumberjacks": "Northern Arizona",
    "northern colorado bears": "Northern Colorado",
    "northern illinois huskies": "Northern Illinois",
    "northern iowa panthers": "Northern Iowa",
    "northern kentucky norse": "Northern Kentucky",
    "northwestern st demons": "Northwestern State",
    "northwestern wildcats": "Northwestern",
    "notre dame fighting irish": "Notre Dame",
    "oakland golden grizzlies": "Oakland",
    "ohio bobcats": "Ohio",
    "ohio state buckeyes": "Ohio State",
    "oklahoma sooners": "Oklahoma",
    "oklahoma state cowboys": "Oklahoma State",
    "old dominion monarchs": "Old Dominion",
    "ole miss rebels": "Ole Miss",
    "omaha mavericks": "Omaha",
    "oral roberts golden eagles": "Oral Roberts",
    "oregon ducks": "Oregon",
    "oregon state beavers": "Oregon State",
    "pacific tigers": "Pacific",
    "penn state nittany lions": "Penn State",
    "pennsylvania quakers": "Pennsylvania",
    "pepperdine waves": "Pepperdine",
    "pittsburgh panthers": "Pittsburgh",
    "portland pilots": "Portland",
    "portland st vikings": "Portland State",
    "prairie view a&m panthers": "Prairie View A&M",
    "presbyterian blue hose": "Presbyterian",
    "princeton tigers": "Princeton",
    "providence friars": "Providence",
    "purdue boilermakers": "Purdue",
    "purdue fort wayne mastodons": "Purdue Fort Wayne",
    "queens university royals": "Queens University",
    "quinnipiac bobcats": "Quinnipiac",
    "radford highlanders": "Radford",
    "rhode island rams": "Rhode Island",
    "rice owls": "Rice",
    "richmond spiders": "Richmond",
    "rider broncs": "Rider",
    "robert morris colonials": "Robert Morris",
    "rutgers scarlet knights": "Rutgers",
    "sacramento st hornets": "Sacramento State",
    "sacred heart pioneers": "Sacred Heart",
    "saint joseph's hawks": "Saint Joseph's",
    "saint louis billikens": "Saint Louis",
    "saint mary's gaels": "Saint Mary's",
    "saint peter's peacocks": "Saint Peter's",
    "sam houston st bearkats": "Sam Houston State",
    "samford bulldogs": "Samford",
    "san diego state aztecs": "San Diego State",
    "san diego toreros": "San Diego",
    "san francisco dons": "San Francisco",
    "san jose st spartans": "San Jose State",
    "santa clara broncos": "Santa Clara",
    "se missouri st redhawks": "SE Missouri State",
    "seattle u redhawks": "Seattle U",
    "seton hall pirates": "Seton Hall",
    "siena saints": "Siena",
    "siu-edwardsville cougars": "SIU-Edwardsville",
    "smu mustangs": "SMU",
    "south alabama jaguars": "South Alabama",
    "south carolina gamecocks": "South Carolina",
    "south carolina st bulldogs": "South Carolina State",
    "south carolina upstate spartans": "South Carolina Upstate",
    "south dakota coyotes": "South Dakota",
    "south dakota st jackrabbits": "South Dakota State",
    "south florida bulls": "South Florida",
    "southern illinois salukis": "Southern Illinois",
    "southern methodist mustangs": "SMU",
    "southern miss golden eagles": "Southern Miss",
    "southern university jaguars": "Southern University",
    "southern utah thunderbirds": "Southern Utah",
    "st. bonaventure bonnies": "St. Bonaventure",
    "st. francis (ny) terriers": "St. Francis (NY)",
    "st. francis (pa) red flash": "St. Francis (PA)",
    "st. john's red storm": "St. John's",
    "st. thomas - minnesota tommies": "St. Thomas",
    "stanford cardinal": "Stanford",
    "stephen f. austin lumberjacks": "Stephen F. Austin",
    "stetson hatters": "Stetson",
    "stonehill skyhawks": "Stonehill",
    "stony brook seawolves": "Stony Brook",
    "syracuse orange": "Syracuse",
    "tarleton state texans": "Tarleton State",
    "tcu horned frogs": "TCU",
    "temple owls": "Temple",
    "tennessee state tigers": "Tennessee State",
    "tennessee tech golden eagles": "Tennessee Tech",
    "tennessee volunteers": "Tennessee",
    "tennessee-martin skyhawks": "Tennessee-Martin",
    "texas a&m aggies": "Texas A&M",
    "texas a&m-cc islanders": "Texas A&M-CC",
    "texas a&m-commerce lions": "Texas A&M-Commerce",
    "texas christian horned frogs": "TCU",
    "texas longhorns": "Texas",
    "texas southern tigers": "Texas Southern",
    "texas state bobcats": "Texas State",
    "texas tech red raiders": "Texas Tech",
    "the citadel bulldogs": "The Citadel",
    "toledo rockets": "Toledo",
    "towson tigers": "Towson",
    "troy trojans": "Troy",
    "tulane green wave": "Tulane",
    "tulsa golden hurricane": "Tulsa",
    "uab blazers": "UAB",
    "uc davis aggies": "UC Davis",
    "uc irvine anteaters": "UC Irvine",
    "uc riverside highlanders": "UC Riverside",
    "uc san diego tritons": "UC San Diego",
    "uc santa barbara gauchos": "UC Santa Barbara",
    "ucf knights": "UCF",
    "ucla bruins": "UCLA",
    "uconn huskies": "UConn",
    "uic flames": "UIC",
    "ul monroe warhawks": "UL Monroe",
    "umass lowell river hawks": "UMass Lowell",
    "umass minutemen": "UMass",
    "umbc retrievers": "UMBC",
    "umkc kangaroos": "UMKC",
    "unc asheville bulldogs": "UNC Asheville",
    "unc greensboro spartans": "UNC Greensboro",
    "unc wilmington seahawks": "UNC Wilmington",
    "unlv rebels": "UNLV",
    "usc trojans": "USC",
    "usc upstate spartans": "USC Upstate",
    "ut arlington mavericks": "UT Arlington",
    "utah state aggies": "Utah State",
    "utah tech trailblazers": "Utah Tech",
    "utah utes": "Utah",
    "utah valley wolverines": "Utah Valley",
    "utep miners": "UTEP",
    "utrgv vaqueros": "UTRGV",
    "utsa roadrunners": "UTSA",
    "valparaiso beacons": "Valparaiso",
    "vanderbilt commodores": "Vanderbilt",
    "vcu rams": "VCU",
    "vermont catamounts": "Vermont",
    "villanova wildcats": "Villanova",
    "virginia cavaliers": "Virginia",
    "virginia tech hokies": "Virginia Tech",
    "vmi keydets": "VMI",
    "wagner seahawks": "Wagner",
    "wake forest demon deacons": "Wake Forest",
    "washington huskies": "Washington",
    "washington st cougars": "Washington State",
    "weber st wildcats": "Weber State",
    "west virginia mountaineers": "West Virginia",
    "western carolina catamounts": "Western Carolina",
    "western illinois leathernecks": "Western Illinois",
    "western kentucky hilltoppers": "Western Kentucky",
    "western michigan broncos": "Western Michigan",
    "wichita st shockers": "Wichita State",
    "william & mary tribe": "William & Mary",
    "winthrop eagles": "Winthrop",
    "wisconsin badgers": "Wisconsin",
    "wofford terriers": "Wofford",
    "wright st raiders": "Wright State",
    "wyoming cowboys": "Wyoming",
    "xavier musketeers": "Xavier",
    "yale bulldogs": "Yale",
    "youngstown st penguins": "Youngstown State"
}


NCAAB_MASCOT_MAP = {
    "uconn": "Connecticut Huskies",
    "duke": "Duke",
    "purdue": "Purdue",
    "kansas": "Kansas",
    "houston": "Houston",
    "gonzaga": "Gonzaga Bulldogs",
    "arizona": "Arizona",
    "illinois": "Illinois Fighting Illini",
    "michigan state": "Michigan State Spartans",
    "vcu": "VCU Rams",
    "tcu": "TCU Horned Frogs",
    "ucf": "UCF Knights",
    "st johns": "St. John's",
    "villanova": "Villanova Wildcats",
    "louisville": "Louisville Cardinals",
    "arkansas": "Arkansas Razorbacks",
    "tennessee": "Tennessee",
    "kentucky": "Kentucky",
    "creighton": "Creighton Bluejays",
    "baylor": "Baylor Bears",
    "marquette": "Marquette Golden Eagles",
    "wisconsin": "Wisconsin Badgers",
    "ucla": "UCLA Bruins",
    "ohio state": "Ohio State Buckeyes",
    "vanderbilt": "Vanderbilt Commodores",
    "virginia": "Virginia Cavaliers",
    "clemson": "Clemson Tigers",
    "saint marys": "Saint Mary's Gaels",
    "high point": "High Point Panthers",
    "furman": "Furman Paladins",
    "queens": "Queens NC Royals",
    "prairie view am": "Prairie View A&M",
    "california baptist": "California Baptist"
}

# Merge newly added exact maps
TEAM_MAP.update(NCAAB_MASCOT_MAP)

TEAM_MAP.update(NBA_EXACT_MAP)
TEAM_MAP.update(WNBA_EXACT_MAP)
TEAM_MAP.update(NHL_EXACT_MAP)
TEAM_MAP.update(NCAAB_EXTRA_MAP)
# Merge dynamic aliases into the primary mapping dictionary
TEAM_MAP.update(ODDS_API_EXACT_MAP)

TEAM_MAP.update(load_dynamic_aliases())



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
    name = name.strip().lower()

    # 2. Intercept hardcoded overrides BEFORE any string replacement or matching
    if name in KALSHI_NCAAB_OVERRIDES:
        return KALSHI_NCAAB_OVERRIDES[name]

    cleaned_name = name

    # 3. Perform dictionary lookup on the explicitly cleaned string
    if cleaned_name in TEAM_MAP:
        mapped_name = TEAM_MAP[cleaned_name]
        # Check if mapped name is one of our exact overrides we want returned exactly as is
        # (Since it was mapped from a lowercase key, we can't just return it title-cased if it has special casing)
        if mapped_name.lower() in ("connecticut", "uconn", "connecticut huskies"):
            return "UConn"
        if mapped_name.lower() in ("queens nc", "queens university", "queens university royals"):
            return "Queens University"
        if mapped_name.lower() in ("california baptist", "ca baptist", "cal baptist", "california baptist lancers"):
            return "California Baptist"
        if mapped_name.lower() in ("st. john's", "saint johns", "st johns", "st. john's red storm"):
            return "St. John's"
        if mapped_name.lower() in ("prairie view a&m", "prairie view panthers", "prairie view am", "prairie view a&m panthers"):
            return "Prairie View A&M"
        if mapped_name.lower() in ("miami fl", "miami (fl)", "miami florida", "miami (fl) hurricanes"):
            return "Miami (FL)"
        if mapped_name.lower() in ("nc state", "nc state wolfpack", "ncsu"):
            return "NC State"
        # MLB multi-team cities and relocated franchises — return immediately so the
        # suffix-stripping loop cannot collapse "Chicago Cubs" → "Chicago" etc.
        _mlb_preserve = {
            "chicago cubs", "chicago white sox",
            "new york yankees", "new york mets",
            "los angeles dodgers", "los angeles angels",
            "athletics",
        }
        if mapped_name.lower() in _mlb_preserve:
            return mapped_name

        name = mapped_name

    # Second check right here since Uconn/Queens might be incoming directly unmapped
    if name.lower() in ("connecticut", "uconn", "connecticut huskies", "uconn huskies"):
        return "UConn"
    if name.lower() in ("queens nc", "queens university", "queens university royals"):
        return "Queens University"

    # Convert to lowercase (in case the dictionary mapping has uppercase)
    normalized = name.lower()

    # Track if team name is missing from exact map and is likely a long-form API name
    # We also check the lowercase name to ensure missing logic works properly with lowercased map
    if name not in TEAM_MAP and name != "Over" and name != "Under" and cleaned_name not in TEAM_MAP:
        # SYSTEM OVERRIDE: Deprecate fuzzy logic. Remove any reliance on naive fuzzywuzzy,
        # Levenshtein distance algorithms, or unanchored substring matching for primary team identification.
        # Strict dictionary mapping only.
        pass

    # Expand common abbreviations BEFORE removing punctuation

    # Use word boundaries to avoid partial matches
    replacements = [
        (r'\bl\.?a\.?\s', 'los angeles '),
        (r'\bl\.?a\.?$', 'los angeles'),     # ADDED THIS for standalone/terminal "LA"
        (r'\bn\.?y\.?\s', 'new york '),
        (r'\bn\.?y\.?$', 'new york'),        # ADDED THIS for standalone/terminal "NY"
        (r'^st\.\s', 'st '),   # "St." at beginning (Saint) - must be before \bst\.\s
        (r'\bst\.\s', 'st '),  # "St." at end or before space
        (r'\bst\.$', 'st'),    # "St." at end of string
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

    words = normalized.split()
    if len(words) >= 2:
        from app_core.team_name_mapping import TEAM_SUFFIXES
        # Do not strip these specific generic tokens for display names
        exclude_from_display_strip = {"state", "tech", "a&m", "university", "college", "men", "women", "basketball", "international"}
        for suffix in TEAM_SUFFIXES:
            s_lower = suffix.lower()
            if s_lower in exclude_from_display_strip:
                continue
            if normalized.endswith(" " + s_lower):
                temp_name = normalized[:-len(s_lower)-1].strip()
                if len(temp_name.split()) >= 1:
                    normalized = temp_name
                    break

    # If normalized name is not strictly mapped, check if there's a punctuation-retaining override
    # using the original `name` variable or the post-processed title case.
    # We do this before titlecasing the stripped 'normalized' string to preserve things like A&M.
    if name.lower() in overrides:
         return overrides[name.lower()]

    title_cased = normalized.title()

    # Check post-processing overrides again
    if title_cased.lower() in overrides:
        return overrides[title_cased.lower()]

    # Re-apply force maps on the finalized string to ensure exact capitalization and punctuation logic
    # since `normalized` removes punctuation which breaks St. John's, A&M, etc.
    for k, v in overrides.items():
        if title_cased.lower() == k.lower() or title_cased.lower() == re.sub(r'[^\w\s]', '', k.lower()):
            return v

    # Specific override checks for problematic names with punctuation
    if title_cased.lower() in ("st johns", "saint johns", "saint johns red storm", "st johns red storm", "st. john's red storm", "st. john's"):
        return "St. John's"
    if title_cased.lower() in ("prairie view am", "prairie view am panthers", "prairie view panthers", "prairie view a&m panthers", "prairie view a&m"):
        return "Prairie View A&M"
    if title_cased.lower() in ("connecticut", "uconn", "connecticut huskies", "uconn huskies"):
        return "UConn"
    if title_cased.lower() in ("queens nc", "queens university", "queens university royals", "queens university of charlotte", "queens nc royals"):
        return "Queens University"
    if title_cased.lower() in ("miami (fl) hurricanes", "miami florida hurricanes", "miami fl", "miami (fl)"):
        return "Miami (FL)"
    if title_cased.lower() in ("nc state", "nc state wolfpack", "ncsu"):
        return "NC State"

    return title_cased


# Merge dynamic aliases into the primary mapping dictionary
TEAM_MAP.update(ODDS_API_EXACT_MAP)
TEAM_MAP.update(load_dynamic_aliases())





# POST-PROCESSING OVERRIDES TO ENSURE EXACT MATCHES
overrides = {
    # Whole-name feed aliases. Keep renamed schools and mascot variants on one
    # identity for odds recovery, candidate grouping, and upload matching.
    "houston baptist": "Houston Christian",
    "houston baptist huskies": "Houston Christian",
    "houston christian": "Houston Christian",
    "houston christian huskies": "Houston Christian",
    "citadel": "Citadel",
    "citadel bulldogs": "Citadel",
    "the citadel": "Citadel",
    "the citadel bulldogs": "Citadel",
    "nicholls": "Nicholls State",
    "nicholls colonels": "Nicholls State",
    "nicholls state": "Nicholls State",
    "nicholls state colonels": "Nicholls State",
    "nicholls st colonels": "Nicholls State",
    "se louisiana": "Southeastern Louisiana",
    "se louisiana lions": "Southeastern Louisiana",
    "southeastern louisiana": "Southeastern Louisiana",
    "southeastern louisiana lions": "Southeastern Louisiana",
    "california baptist lancers": "California Baptist",
    "ca baptist lancers": "California Baptist",
    "california baptist": "California Baptist",
    "cal baptist": "California Baptist",
    "ca baptist": "California Baptist",
    "miami (fl) hurricanes": "Miami (FL)",
    "miami (fl)": "Miami (FL)",
    "miami florida hurricanes": "Miami (FL)",
    "miami florida": "Miami (FL)",
    "miami fl": "Miami (FL)",
    "connecticut huskies": "UConn",
    "uconn huskies": "UConn",
    "connecticut": "UConn",
    "uconn": "UConn",
    "dallas mavericks": "Dallas",
    "golden state warriors": "Golden State",
    "queens university": "Queens University",
    "queens university of charlotte": "Queens University",
    "queens university royals": "Queens University",
    "queens nc": "Queens University",
    "saint johns": "St. John's",
    "st johns": "St. John's",
    "st. john's": "St. John's",
    "st. john's red storm": "St. John's",
    "saint john's": "St. John's",
    "saint louis": "Saint Louis",
    "st louis": "Saint Louis",
    "st. louis": "Saint Louis",
    "saint louis billikens": "Saint Louis",
    "st louis billikens": "Saint Louis",
    "st. louis billikens": "Saint Louis",
    "prairie view panthers": "Prairie View A&M",
    "prairie view a&m": "Prairie View A&M",
    "prairie view a&m panthers": "Prairie View A&M",
    "prairie view am": "Prairie View A&M",
    "nc state wolfpack": "NC State",
    "athletics mlb": "Oakland"
}
TEAM_MAP.update(overrides)

# Force any value in the dictionary that maps to variations of Uconn or Miami Fl
for k, v in TEAM_MAP.items():
    if v.lower() in ("uconn", "connecticut", "connecticut huskies", "uconn huskies"):
        TEAM_MAP[k] = "UConn"
    if v.lower() in ("miami fl", "miami (fl)", "miami florida", "miami (fl) hurricanes"):
        TEAM_MAP[k] = "Miami (FL)"
    if v.lower() in ("ca baptist", "cal baptist", "california baptist", "california baptist lancers"):
        TEAM_MAP[k] = "California Baptist"
    if v.lower() in ("queens nc", "queens university", "queens university royals", "queens university of charlotte"):
        TEAM_MAP[k] = "Queens University"
    if v.lower() in ("st. john's", "saint johns", "st johns", "saint john's", "st. john's red storm"):
        TEAM_MAP[k] = "St. John's"
    if v.lower() in ("prairie view a&m", "prairie view panthers", "prairie view am", "prairie view a&m panthers"):
        TEAM_MAP[k] = "Prairie View A&M"
    if v.lower() in ("nc state", "nc state wolfpack", "ncsu"):
        TEAM_MAP[k] = "NC State"

