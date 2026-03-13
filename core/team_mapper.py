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
    "massachusetts": "umass",
    "miami oh": "miami (oh)",
    "mizzou": "missouri",
    "bowling green": "bgsu",
    "texas tech": "texas tech",
    "iowa state": "iowa state",
}

# Auto-injected exact mappings for Odds API full mascot names to standard names
ODDS_API_EXACT_MAP = {
    "abilene christian wildcats": "Abilene Christian",
    "air force falcons": "Air Force",
    "akron zips": "Akron",
    "alabama a&m bulldogs": "Alabama A&M",
    "alabama crimson tide": "Alabama",
    "alabama st hornets": "Alabama St",
    "albany great danes": "Albany",
    "alcorn st braves": "Alcorn St",
    "american university eagles": "American University",
    "appalachian st mountaineers": "Appalachian St",
    "arizona st sun devils": "Arizona St",
    "arizona wildcats": "Arizona",
    "arkansas razorbacks": "Arkansas",
    "arkansas st red wolves": "Arkansas St",
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
    "central connecticut st blue devils": "Central Connecticut St",
    "central michigan chippewas": "Central Michigan",
    "charleston cougars": "Charleston",
    "charleston southern buccaneers": "Charleston Southern",
    "charlotte 49ers": "Charlotte",
    "chattanooga mocs": "Chattanooga",
    "chicago st cougars": "Chicago St",
    "cincinnati bearcats": "Cincinnati",
    "citadel bulldogs": "Citadel",
    "clemson tigers": "Clemson",
    "cleveland st vikings": "Cleveland St",
    "coastal carolina chanticleers": "Coastal Carolina",
    "colgate raiders": "Colgate",
    "college of charleston cougars": "College of Charleston",
    "colorado buffaloes": "Colorado",
    "colorado st rams": "Colorado St",
    "columbia lions": "Columbia",
    "connecticut huskies": "Connecticut",
    "coppin st eagles": "Coppin St",
    "cornell big red": "Cornell",
    "creighton bluejays": "Creighton",
    "dartmouth big green": "Dartmouth",
    "davidson wildcats": "Davidson",
    "dayton flyers": "Dayton",
    "delaware blue hens": "Delaware",
    "delaware st hornets": "Delaware St",
    "denver pioneers": "Denver",
    "depaul blue demons": "DePaul",
    "detroit mercy titans": "Detroit Mercy",
    "drake bulldogs": "Drake",
    "drexel dragons": "Drexel",
    "duke blue devils": "Duke",
    "duquesne dukes": "Duquesne",
    "east carolina pirates": "East Carolina",
    "east tennessee st buccaneers": "East Tennessee St",
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
    "fresno st bulldogs": "Fresno St",
    "furman paladins": "Furman",
    "gardner-webb runnin' bulldogs": "Gardner-Webb",
    "george mason patriots": "George Mason",
    "george washington colonials": "George Washington",
    "george washington revolutionaries": "George Washington",
    "georgetown hoyas": "Georgetown",
    "georgia bulldogs": "Georgia",
    "georgia southern eagles": "Georgia Southern",
    "georgia st panthers": "Georgia St",
    "georgia tech yellow jackets": "Georgia Tech",
    "gonzaga bulldogs": "Gonzaga",
    "grambling st tigers": "Grambling St",
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
    "idaho st bengals": "Idaho St",
    "idaho vandals": "Idaho",
    "illinois fighting illini": "Illinois",
    "illinois st redbirds": "Illinois St",
    "illinois-chicago flames": "Illinois-Chicago",
    "incarnate word cardinals": "Incarnate Word",
    "indiana hoosiers": "Indiana",
    "indiana st sycamores": "Indiana St",
    "iona gaels": "Iona",
    "iowa hawkeyes": "Iowa",
    "iowa state cyclones": "Iowa State",
    "iupui jaguars": "IUPUI",
    "jackson st tigers": "Jackson St",
    "jacksonville dolphins": "Jacksonville",
    "jacksonville st gamecocks": "Jacksonville St",
    "james madison dukes": "James Madison",
    "kansas city roos": "Kansas City",
    "kansas jayhawks": "Kansas",
    "kansas state wildcats": "Kansas State",
    "kennesaw st owls": "Kennesaw St",
    "kent state golden flashes": "Kent State",
    "kentucky wildcats": "Kentucky",
    "la salle explorers": "La Salle",
    "lafayette leopards": "Lafayette",
    "lamar cardinals": "Lamar",
    "lehigh mountain hawks": "Lehigh",
    "liberty flames": "Liberty",
    "lipscomb bisons": "Lipscomb",
    "long beach st beach": "Long Beach St",
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
    "mcneese st cowboys": "McNeese St",
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
    "mississippi valley st delta devils": "Mississippi Valley St",
    "missouri state bears": "Missouri State",
    "missouri tigers": "Missouri",
    "monmouth hawks": "Monmouth",
    "montana grizzlies": "Montana",
    "montana st bobcats": "Montana St",
    "morehead st eagles": "Morehead St",
    "morgan st bears": "Morgan St",
    "mount st. mary's mountaineers": "Mount St. Mary's",
    "mt. st. mary's mountaineers": "Mt. St. Mary's",
    "murray st racers": "Murray St",
    "n.c. a&t aggies": "N.C. A&T",
    "n.c. central eagles": "N.C. Central",
    "n colorado bears": "Northern Colorado",
    "navy midshipmen": "Navy",
    "nc state wolfpack": "NC State",
    "nebraska cornhuskers": "Nebraska",
    "nevada wolf pack": "Nevada",
    "new hampshire wildcats": "New Hampshire",
    "new mexico lobos": "New Mexico",
    "new mexico st aggies": "New Mexico St",
    "new orleans privateers": "New Orleans",
    "niagara purple eagles": "Niagara",
    "nicholls st colonels": "Nicholls St",
    "njit highlanders": "NJIT",
    "norfolk st spartans": "Norfolk St",
    "north alabama lions": "North Alabama",
    "north carolina tar heels": "North Carolina",
    "north dakota st bison": "North Dakota St",
    "north dakota fighting hawks": "North Dakota",
    "north florida ospreys": "North Florida",
    "north texas mean green": "North Texas",
    "northeastern huskies": "Northeastern",
    "northern arizona lumberjacks": "Northern Arizona",
    "northern colorado bears": "Northern Colorado",
    "northern illinois huskies": "Northern Illinois",
    "northern iowa panthers": "Northern Iowa",
    "northern kentucky norse": "Northern Kentucky",
    "northwestern st demons": "Northwestern St",
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
    "portland st vikings": "Portland St",
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
    "sacramento st hornets": "Sacramento St",
    "sacred heart pioneers": "Sacred Heart",
    "saint joseph's hawks": "Saint Joseph's",
    "saint louis billikens": "Saint Louis",
    "saint mary's gaels": "Saint Mary's",
    "saint peter's peacocks": "Saint Peter's",
    "sam houston st bearkats": "Sam Houston St",
    "samford bulldogs": "Samford",
    "san diego state aztecs": "San Diego State",
    "san diego toreros": "San Diego",
    "san francisco dons": "San Francisco",
    "san jose st spartans": "San Jose St",
    "santa clara broncos": "Santa Clara",
    "se missouri st redhawks": "SE Missouri St",
    "seattle u redhawks": "Seattle U",
    "seton hall pirates": "Seton Hall",
    "siena saints": "Siena",
    "siu-edwardsville cougars": "SIU-Edwardsville",
    "smu mustangs": "SMU",
    "south alabama jaguars": "South Alabama",
    "south carolina gamecocks": "South Carolina",
    "south carolina st bulldogs": "South Carolina St",
    "south carolina upstate spartans": "South Carolina Upstate",
    "south dakota coyotes": "South Dakota",
    "south dakota st jackrabbits": "South Dakota St",
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
    "washington st cougars": "Washington St",
    "weber st wildcats": "Weber St",
    "west virginia mountaineers": "West Virginia",
    "western carolina catamounts": "Western Carolina",
    "western illinois leathernecks": "Western Illinois",
    "western kentucky hilltoppers": "Western Kentucky",
    "western michigan broncos": "Western Michigan",
    "wichita st shockers": "Wichita St",
    "william & mary tribe": "William & Mary",
    "winthrop eagles": "Winthrop",
    "wisconsin badgers": "Wisconsin",
    "wofford terriers": "Wofford",
    "wright st raiders": "Wright St",
    "wyoming cowboys": "Wyoming",
    "xavier musketeers": "Xavier",
    "yale bulldogs": "Yale",
    "youngstown st penguins": "Youngstown St"
}

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
    cleaned_name = name

    # 2. Perform dictionary lookup on the explicitly cleaned string
    if cleaned_name in TEAM_MAP:
        name = TEAM_MAP[cleaned_name]

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
    cleaned_name = name

    # 2. Perform dictionary lookup on the explicitly cleaned string
    if cleaned_name in TEAM_MAP:
        name = TEAM_MAP[cleaned_name]

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
