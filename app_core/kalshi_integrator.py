"""
Kalshi Integrator with Proper RSA-PSS Authentication
This file goes in: app_core/kalshi_integrator.py
"""

import copy
import time
import logging
import re
import string
import os
from datetime import datetime
import pytz
import requests
import streamlit as st
from typing import Dict, List, Any, Optional, TypedDict

# --- 1. SHARED IMPORTS ---
try:
    from app_core.team_name_matcher import TeamNameMatcher
except ImportError:
    TeamNameMatcher = None

logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION & CONSTANTS ---

KALSHI_TEAM_ABBREVIATIONS = {
    # NBA
    "ATLANTA HAWKS": ["ATL"], "BOSTON CELTICS": ["BOS"], "BROOKLYN NETS": ["BKN", "BRK"],
    "CHARLOTTE HORNETS": ["CHA", "CLT"], "CHICAGO BULLS": ["CHI"], "CLEVELAND CAVALIERS": ["CLE", "CAVS"],
    "DALLAS MAVERICKS": ["DAL"], "DENVER NUGGETS": ["DEN"], "DETROIT PISTONS": ["DET"],
    "GOLDEN STATE WARRIORS": ["GSW", "GS"], "HOUSTON ROCKETS": ["HOU"], "INDIANA PACERS": ["IND"],
    "LOS ANGELES CLIPPERS": ["LAC", "LA CLIPPERS"], "LOS ANGELES LAKERS": ["LAL", "LA LAKERS"],
    "MEMPHIS GRIZZLIES": ["MEM"], "MIAMI HEAT": ["MIA"], "MILWAUKEE BUCKS": ["MIL"],
    "MINNESOTA TIMBERWOLVES": ["MIN"], "NEW ORLEANS PELICANS": ["NOP", "NO PELICANS"],
    "NEW YORK KNICKS": ["NYK", "NY KNICKS"], "OKLAHOMA CITY THUNDER": ["OKC"], "ORLANDO MAGIC": ["ORL"],
    "PHILADELPHIA 76ERS": ["PHI", "PHL", "SIXERS"], "PHOENIX SUNS": ["PHX"], "PORTLAND TRAIL BLAZERS": ["POR"],
    "SACRAMENTO KINGS": ["SAC"], "SAN ANTONIO SPURS": ["SAS", "SA SPURS"], "TORONTO RAPTORS": ["TOR"],
    "UTAH JAZZ": ["UTA"], "WASHINGTON WIZARDS": ["WAS", "WSH"],
    # NFL
    "ARIZONA CARDINALS": ["ARI", "ARZ"], "ATLANTA FALCONS": ["ATL"], "BALTIMORE RAVENS": ["BAL"],
    "BUFFALO BILLS": ["BUF"], "CAROLINA PANTHERS": ["CAR"], "CHICAGO BEARS": ["CHI"],
    "CINCINNATI BENGALS": ["CIN"], "CLEVELAND BROWNS": ["CLE"], "DALLAS COWBOYS": ["DAL"],
    "DENVER BRONCOS": ["DEN"], "DETROIT LIONS": ["DET"], "GREEN BAY PACKERS": ["GB", "GBP"],
    "HOUSTON TEXANS": ["HOU"], "INDIANAPOLIS COLTS": ["IND"], "JACKSONVILLE JAGUARS": ["JAX", "JAC"],
    "KANSAS CITY CHIEFS": ["KC", "KCC"], "LAS VEGAS RAIDERS": ["LV", "LVR"], "LOS ANGELES CHARGERS": ["LAC"],
    "LOS ANGELES RAMS": ["LAR"], "MIAMI DOLPHINS": ["MIA"], "MINNESOTA VIKINGS": ["MIN"],
    "NEW ENGLAND PATRIOTS": ["NE", "NEP"], "NEW ORLEANS SAINTS": ["NO", "NOS"], "NEW YORK GIANTS": ["NYG"],
    "NEW YORK JETS": ["NYJ"], "PHILADELPHIA EAGLES": ["PHI"], "PITTSBURGH STEELERS": ["PIT"],
    "SAN FRANCISCO 49ERS": ["SF", "SFO"], "SEATTLE SEAHAWKS": ["SEA"], "TAMPA BAY BUCCANEERS": ["TB", "TBB"],
    "TENNESSEE TITANS": ["TEN"], "WASHINGTON COMMANDERS": ["WAS", "WSH"],
    # NHL
    "ANAHEIM DUCKS": ["ANA"], "BOSTON BRUINS": ["BOS"], "BUFFALO SABRES": ["BUF"], "CALGARY FLAMES": ["CGY"],
    "CAROLINA HURRICANES": ["CAR"], "CHICAGO BLACKHAWKS": ["CHI"], "COLORADO AVALANCHE": ["COL"], "COLUMBUS BLUE JACKETS": ["CBJ"],
    "DALLAS STARS": ["DAL"], "DETROIT RED WINGS": ["DET"], "EDMONTON OILERS": ["EDM"], "FLORIDA PANTHERS": ["FLA"],
    "LOS ANGELES KINGS": ["LAK"], "MINNESOTA WILD": ["MIN"], "MONTREAL CANADIENS": ["MTL"], "NASHVILLE PREDATORS": ["NSH"],
    "NEW JERSEY DEVILS": ["NJD"], "NEW YORK ISLANDERS": ["NYI"], "NEW YORK RANGERS": ["NYR"], "OTTAWA SENATORS": ["OTT"],
    "PHILADELPHIA FLYERS": ["PHI"], "PITTSBURGH PENGUINS": ["PIT"], "SAN JOSE SHARKS": ["SJS"], "SEATTLE KRAKEN": ["SEA"],
    "ST LOUIS BLUES": ["STL"], "TAMPA BAY LIGHTNING": ["TBL"], "TORONTO MAPLE LEAFS": ["TOR"], "UTAH HOCKEY CLUB": ["UTA"],
    "VANCOUVER CANUCKS": ["VAN"], "VEGAS GOLDEN KNIGHTS": ["VGK"], "WASHINGTON CAPITALS": ["WSH"], "WINNIPEG JETS": ["WPG"],
    # MLB
    "ARIZONA DIAMONDBACKS": ["ARI"], "ATLANTA BRAVES": ["ATL"], "BALTIMORE ORIOLES": ["BAL"], "BOSTON RED SOX": ["BOS"],
    "CHICAGO CUBS": ["CHC"], "CHICAGO WHITE SOX": ["CWS"], "CINCINNATI REDS": ["CIN"], "CLEVELAND GUARDIANS": ["CLE"],
    "COLORADO ROCKIES": ["COL"], "DETROIT TIGERS": ["DET"], "HOUSTON ASTROS": ["HOU"], "KANSAS CITY ROYALS": ["KC"],
    "LOS ANGELES ANGELS": ["LAA"], "LOS ANGELES DODGERS": ["LAD"], "MIAMI MARLINS": ["MIA"], "MILWAUKEE BREWERS": ["MIL"],
    "MINNESOTA TWINS": ["MIN"], "NEW YORK METS": ["NYM"], "NEW YORK YANKEES": ["NYY"], "OAKLAND ATHLETICS": ["OAK"],
    "PHILADELPHIA PHILLIES": ["PHI"], "PITTSBURGH PIRATES": ["PIT"], "SAN DIEGO PADRES": ["SD"], "SAN FRANCISCO GIANTS": ["SF"],
    "SEATTLE MARINERS": ["SEA"], "ST LOUIS CARDINALS": ["STL"], "TAMPA BAY RAYS": ["TB"], "TEXAS RANGERS": ["TEX"],
    "TORONTO BLUE JAYS": ["TOR"], "WASHINGTON NATIONALS": ["WSH"],
    # NCAAF (FBS) — 133 TEAMS
    "AIR FORCE": ["AF"], "AKRON": ["AKR"], "ALABAMA": ["ALA", "BAMA"], "APPALACHIAN STATE": ["APP", "APP ST"], "ARIZONA": ["ARIZ"],
    "ARIZONA STATE": ["ASU"], "ARKANSAS": ["ARK"], "ARKANSAS STATE": ["ARST"], "ARMY": ["ARMY"], "AUBURN": ["AUB"], "BALL STATE": ["BALL"],
    "BAYLOR": ["BAY"], "BOISE STATE": ["BOISE", "BSU"], "BOSTON COLLEGE": ["BC"], "BOWLING GREEN": ["BGSU"], "BUFFALO": ["BUF"], "BYU": ["BYU"],
    "CALIFORNIA": ["CAL"], "CENTRAL MICHIGAN": ["CMU"], "CHARLOTTE": ["CHAR", "CLT"], "CINCINNATI": ["CIN"], "CLEMSON": ["CLEM"],
    "COASTAL CAROLINA": ["CCU"], "COLORADO": ["COLO"], "COLORADO STATE": ["CSU"], "DUKE": ["DUKE"], "EAST CAROLINA": ["ECU"],
    "EASTERN MICHIGAN": ["EMU"], "FLORIDA": ["FLA", "UF"], "FLORIDA ATLANTIC": ["FAU"], "FLORIDA INTERNATIONAL": ["FIU"], "FLORIDA STATE": ["FSU"],
    "FRESNO STATE": ["FRES"], "GEORGIA": ["UGA", "GA"], "GEORGIA SOUTHERN": ["GASO"], "GEORGIA STATE": ["GAST"], "GEORGIA TECH": ["GT"],
    "HAWAII": ["HAW"], "HOUSTON": ["HOU"], "ILLINOIS": ["ILL"], "ILLINOIS STATE": ["ILST"], "INDIANA": ["IU"], "IOWA": ["IOWA"], 
    "IOWA STATE": ["ISU"], "JAMES MADISON": ["JMU"], "KANSAS": ["KU"], "KANSAS STATE": ["KSU"], "KENT STATE": ["KENT"], "KENTUCKY": ["UK"],
    "LIBERTY": ["LIB"], "LOUISIANA": ["ULL", "LA"], "LOUISIANA MONROE": ["ULM"], "LOUISIANA TECH": ["LT"], "LOUISVILLE": ["LOU"],
    "LSU": ["LSU"], "MARSHALL": ["MRSH"], "MARYLAND": ["MD"], "MEMPHIS": ["MEM"], "MIAMI FL": ["MIA", "MIAMI"], "MIAMI OH": ["M-OH"],
    "MICHIGAN": ["MICH"], "MICHIGAN STATE": ["MSU"], "MIDDLE TENNESSEE": ["MTSU"], "MINNESOTA": ["MINN"], "MISSISSIPPI STATE": ["MSST"],
    "MISSISSIPPI": ["MISS", "OLE MISS"], "MISSOURI": ["MIZZ"], "NAVY": ["NAVY"], "NEBRASKA": ["NEB"], "NEVADA": ["NEV"], "NEW MEXICO": ["NM"],
    "NEW MEXICO STATE": ["NMSU"], "NORTH CAROLINA": ["UNC"], "NORTH CAROLINA STATE": ["NCST"], "NORTH TEXAS": ["UNT"], "NORTHERN ILLINOIS": ["NIU"],
    "NORTHWESTERN": ["NW"], "NOTRE DAME": ["ND"], "OHIO": ["OHIO"], "OHIO STATE": ["OSU"], "OKLAHOMA": ["OU"], "OKLAHOMA STATE": ["OKST"],
    "OLD DOMINION": ["ODU"], "OLE MISS": ["MISS", "OLE MISS"], "OREGON": ["ORE"], "OREGON STATE": ["ORST"], "PENN STATE": ["PSU"],
    "PITTSBURGH": ["PITT"], "PURDUE": ["PUR"], "RICE": ["RICE"], "RUTGERS": ["RUT"], "SAN DIEGO STATE": ["SDSU"], "SAN JOSE STATE": ["SJSU"],
    "SMU": ["SMU"], "SOUTH ALABAMA": ["USA"], "SOUTH CAROLINA": ["SC"], "SOUTH FLORIDA": ["USF"], "SOUTHERN MISS": ["USM"], "STANFORD": ["STAN"],
    "SYRACUSE": ["SYR"], "TCU": ["TCU"], "TEMPLE": ["TEMP"], "TENNESSEE": ["TENN"], "TEXAS": ["TEX"], "TEXAS A&M": ["TAMU"], 
    "TEXAS STATE": ["TXST"], "TEXAS TECH": ["TTU"], "TOLEDO": ["TOL"], "TROY": ["TROY"], "TULANE": ["TUL"], "TULSA": ["TLSA"], "UAB": ["UAB"], 
    "UCF": ["UCF"], "UCLA": ["UCLA"], "UCONN": ["CONN"], "UMASS": ["UMASS"], "UNLV": ["UNLV"], "USC": ["USC"], "UTAH": ["UTAH"], "UTAH STATE": ["USU"],
    "UTEP": ["UTEP"], "UTSA": ["UTSA"], "VANDERBILT": ["VAN"], "VIRGINIA": ["UVA"], "VIRGINIA TECH": ["VT"], "WAKE FOREST": ["WAKE"], "WASHINGTON": ["UW"],
    "WASHINGTON STATE": ["WSU"], "WEST VIRGINIA": ["WVU"], "WESTERN KENTUCKY": ["WKU"], "WESTERN MICHIGAN": ["WMU"], "WISCONSIN": ["WISC"],
    "WYOMING": ["WYO"],
    # NCAAB — 362 Division-I Teams
    "ABILENE CHRISTIAN": ["ACU"], "AIR FORCE": ["AF"], "AKRON": ["AKR"], "ALABAMA": ["ALA"], "ALABAMA A&M": ["AAMU"], "ALABAMA STATE": ["ALST"],
    "ALBANY": ["ALBY"], "ALCORN STATE": ["ALCN"], "AMERICAN": ["AMER"], "APPALACHIAN STATE": ["APP"], "ARIZONA": ["ARIZ"], "ARIZONA STATE": ["ASU"],
    "ARKANSAS": ["ARK"], "ARKANSAS STATE": ["ARST"], "ARKANSAS-PINE BLUFF": ["UAPB"], "ARMY": ["ARMY"], "AUBURN": ["AUB"], "AUSTIN PEAY": ["PEAY"],
    "BALL STATE": ["BALL"], "BAYLOR": ["BAY"], "BELMONT": ["BEL"], "BETHUNE-COOKMAN": ["BCU"], "BINGHAMTON": ["BING"], "BOISE STATE": ["BOISE"],
    "BOSTON COLLEGE": ["BC"], "BOSTON UNIVERSITY": ["BU"], "BOWLING GREEN": ["BGSU"], "BRADLEY": ["BRAD"], "BRIGHAM YOUNG": ["BYU"],     
    "BROWN": ["BROWN"], "BRYANT": ["BRY"], "BUCKNELL": ["BUCK"], "BUFFALO": ["BUF"], "BUTLER": ["BUT"], "CAL POLY": ["CP"], 
    "CAL STATE BAKERSFIELD": ["CSUB"], "CAL STATE FULLERTON": ["CSUF"], "CAL STATE NORTHRIDGE": ["CSUN"], "CALIFORNIA": ["CAL"], "CAMPBELL": ["CAMP"],
    "CANISIUS": ["CAN"], "CENTRAL ARKANSAS": ["UCA"], "CENTRAL CONNECTICUT": ["CCSU"], "CENTRAL MICHIGAN": ["CMU"], "CHARLESTON": ["CHAR"],
    "CHARLESTON SOUTHERN": ["CHSO"], "CHARLOTTE": ["CLT"], "CHICAGO STATE": ["CHST"], "CINCINNATI": ["CIN"], "CITADEL": ["CIT"], "CLEMSON": ["CLEM"],
    "CLEVELAND STATE": ["CLEV"], "COASTAL CAROLINA": ["CCU"], "COLGATE": ["COLG"], "COLLEGE OF CHARLESTON": ["COFC"], "COLORADO": ["COLO"],
    "COLORADO STATE": ["CSU"], "COLUMBIA": ["COL"], "CONNECTICUT": ["UCONN"], "COPPIN STATE": ["COPP"], "CORNELL": ["COR"], "CREIGHTON": ["CREI"],
    "DARTMOUTH": ["DART"], "DAVIDSON": ["DAV"], "DAYTON": ["DAY"], "DEPAUL": ["DEP"], "DELAWARE": ["DEL"], "DELAWARE STATE": ["DSU"],
    "DENVER": ["DEN"], "DETROIT MERCY": ["DMU"], "DRAKE": ["DRKE"], "DREXEL": ["DRX"], "DUKE": ["DUKE"], "DUQUESNE": ["DUQ"], "EAST CAROLINA": ["ECU"],
    "EAST TENNESSEE STATE": ["ETSU"], "EASTERN ILLINOIS": ["EIU"], "EASTERN KENTUCKY": ["EKU"], "EASTERN MICHIGAN": ["EMU"], "EASTERN WASHINGTON": ["EWU"],
    "ELON": ["ELON"], "EVANSVILLE": ["UE"], "FAIRFIELD": ["FAIR"], "FAIRLEIGH DICKINSON": ["FDU"], "FLORIDA": ["FLA"], "FLORIDA A&M": ["FAMU"],
    "FLORIDA ATLANTIC": ["FAU"], "FLORIDA GULF COAST": ["FGCU"], "FLORIDA INTERNATIONAL": ["FIU"], "FLORIDA STATE": ["FSU"], "FORDHAM": ["FORD"],
    "FRESNO STATE": ["FRES"], "FURMAN": ["FUR"], "GARDNER-WEBB": ["GWU"], "GEORGE MASON": ["GMU"], "GEORGE WASHINGTON": ["GW"], "GEORGETOWN": ["GTWN"],
    "GEORGIA": ["UGA"], "GEORGIA SOUTHERN": ["GASO"], "GEORGIA STATE": ["GAST"], "GEORGIA TECH": ["GT"], "GONZAGA": ["GONZ"], "GRAND CANYON": ["GCU"],
    "HAMPTON": ["HAMP"], "HARVARD": ["HARV"], "HAWAII": ["HAW"], "HIGH POINT": ["HPU"], "HOFSTRA": ["HOF"], "HOLY CROSS": ["HC"], "HOUSTON": ["HOU"],
    "HOUSTON CHRISTIAN": ["HCU"], "HOWARD": ["HOW"], "IDAHO": ["IDHO"], "IDAHO STATE": ["IDST"], "ILLINOIS": ["ILL"], "ILLINOIS STATE": ["ILST"],
    "ILLINOIS-CHICAGO": ["UIC"], "INCARNATE WORD": ["UIW"], "INDIANA": ["IU"], "INDIANA STATE": ["INS"], "IONA": ["IONA"], "IOWA": ["IOWA"],
    "IOWA STATE": ["ISU"], "IPFW": ["IPFW"], "JACKSONVILLE": ["JAX"], "JACKSONVILLE STATE": ["JSU"], "JAMES MADISON": ["JMU"], "KANSAS": ["KU"],
    "KANSAS STATE": ["KSU"], "KENNESAW STATE": ["KSU"], "KENT STATE": ["KENT"], "KENTUCKY": ["UK"], "LA SALLE": ["LAS"], "LAFAYETTE": ["LAF"],
    "LAMAR": ["LAM"], "LEHIGH": ["LEH"], "LIBERTY": ["LIB"], "LIPSCOMB": ["LIP"], "LOUISIANA": ["ULL"], "LOUISIANA TECH": ["LT"],
    "LOUISVILLE": ["LOU"], "LOYOLA CHICAGO": ["LUC"], "LOYOLA MARYLAND": ["LMD"], "LOYOLA MARYMOUNT": ["LMU"], "LSU": ["LSU"],
    "LONG BEACH STATE": ["LBSU"], "LONG ISLAND": ["LIU"], "LONGWOOD": ["LONG"], "MAINE": ["ME"], "MANHATTAN": ["MAN"], "MARIST": ["MAR"],
    "MARQUETTE": ["MARQ"], "MARSHALL": ["MRSH"], "MARYLAND": ["MD"], "MARYLAND-EASTERN SHORE": ["UMES"], "MASSACHUSETTS": ["UMASS"],
    "MCCNEESE": ["MCN"], "MEMPHIS": ["MEM"], "MERCER": ["MER"], "MIAMI FL": ["MIA"], "MIAMI OH": ["M-OH"], "MICHIGAN": ["MICH"],
    "MICHIGAN STATE": ["MSU"], "MIDDLE TENNESSEE": ["MTSU"], "MINNESOTA": ["MINN"], "MISSISSIPPI": ["MISS"], "MISSISSIPPI STATE": ["MSST"],
    "MISSOURI": ["MIZZ"], "MISSOURI STATE": ["MOST"], "MONMOUTH": ["MON"], "MONTANA": ["MONT"], "MONTANA STATE": ["MSU"], "MOREHEAD STATE": ["MORE"],
    "MORGAN STATE": ["MORG"], "MOUNT ST MARYS": ["MSM"], "NAVY": ["NAVY"], "NC CENTRAL": ["NCCU"], "NC STATE": ["NCST"], "NEBRASKA": ["NEB"],
    "NEVADA": ["NEV"], "NEW HAMPSHIRE": ["UNH"], "NEW MEXICO": ["NM"], "NEW MEXICO STATE": ["NMSU"], "NEW ORLEANS": ["UNO"], "NIAGARA": ["NIA"],
    "NICHOLLS": ["NICH"], "NJIT": ["NJIT"], "NORFOLK STATE": ["NSU"], "NORTH ALABAMA": ["UNA"], "NORTH CAROLINA": ["UNC"], "NORTH CAROLINA A&T": ["NCAT"],
    "NORTH CAROLINA ASHEVILLE": ["UNCA"], "NORTH CAROLINA GREENSBORO": ["UNCG"], "NORTH CAROLINA WILMINGTON": ["UNCW"], "NORTH DAKOTA": ["ND"],
    "NORTH DAKOTA STATE": ["NDSU"], "NORTH FLORIDA": ["UNF"], "NORTH TEXAS": ["UNT"], "NORTHEASTERN": ["NE"], "NORTHERN ARIZONA": ["NAU"],
    "NORTHERN COLORADO": ["UNCO"], "NORTHERN ILLINOIS": ["NIU"], "NORTHERN IOWA": ["UNI"], "NORTHWESTERN": ["NW"], "NORTHWESTERN STATE": ["NWST"],
    "NOTRE DAME": ["ND"], "OAKLAND": ["OAK"], "OHIO": ["OHIO"], "OHIO STATE": ["OSU"], "OKLAHOMA": ["OU"], "OKLAHOMA STATE": ["OKST"],
    "OLD DOMINION": ["ODU"], "OLE MISS": ["MISS"], "OMAHA": ["OMA"], "ORAL ROBERTS": ["ORU"], "OREGON": ["ORE"], "OREGON STATE": ["ORST"],
    "PACIFIC": ["PAC"], "PENN": ["PENN"], "PENN STATE": ["PSU"], "PEPPERDINE": ["PEPP"], "PITTSBURGH": ["PITT"], "PORTLAND": ["PORT"],
    "PORTLAND STATE": ["PRST"], "PRAIRIE VIEW A&M": ["PVAM"], "PRESBYTERIAN": ["PRES"], "PRINCETON": ["PRIN"], "PROVIDENCE": ["PROV"],
    "PURDUE": ["PUR"], "QUINNIPIAC": ["QUIN"], "RADFORD": ["RAD"], "RHODE ISLAND": ["URI"], "RICE": ["RICE"], "RICHMOND": ["RICH"],
    "RIDER": ["RID"], "RUTGERS": ["RUT"], "SACRAMENTO STATE": ["SAC"], "SACRED HEART": ["SHU"], "SAINT JOSEPH'S": ["SJU"],
    "SAINT LOUIS": ["SLU"], "SAINT MARY'S": ["SMC"], "SAINT PETER'S": ["SPU"], "SAM HOUSTON": ["SHSU"], "SAMFORD": ["SAM"],
    "SAN DIEGO": ["USD"], "SAN DIEGO STATE": ["SDSU"], "SAN FRANCISCO": ["USF"], "SAN JOSE STATE": ["SJSU"], "SANTA CLARA": ["SCU"],
    "SEATTLE": ["SEA"],  "SETON HALL": ["HALL"], "SIENA": ["SIE"], "SOUTH ALABAMA": ["USA"], "SOUTH CAROLINA": ["SC"],
    "SOUTH CAROLINA STATE": ["SCSU"], "SOUTH DAKOTA": ["SD"], "SOUTH DAKOTA STATE": ["SDSU"], "SOUTH FLORIDA": ["USF"],
    "SOUTHERN ILLINOIS": ["SIU"], "SOUTHERN MISS": ["USM"], "SOUTHERN UTAH": ["SUU"], "STANFORD": ["STAN"],
    "STETSON": ["STET"], "STONY BROOK": ["SB"], "SYRACUSE": ["SYR"], "TCU": ["TCU"], "TEMPLE": ["TEMP"], "TENNESSEE": ["TENN"],
    "TENNESSEE STATE": ["TSU"], "TENNESSEE TECH": ["TNTECH"], "TEXAS": ["TEX"], "TEXAS A&M": ["TAMU"], "TEXAS A&M CC": ["TAMUCC"],
    "TEXAS STATE": ["TXST"], "TEXAS TECH": ["TTU"], "TOLEDO": ["TOL"], "TOWSON": ["TOW"], "TROY": ["TROY"], "TULANE": ["TUL"],
    "TULSA": ["TLSA"],  "UAB": ["UAB"], "UC DAVIS": ["UCD"], "UC IRVINE": ["UCI"], "UC RIVERSIDE": ["UCR"], "UC SAN DIEGO": ["UCSD"],
    "UC SANTA BARBARA": ["UCSB"], "UCF": ["UCF"], "UCLA": ["UCLA"], "ULM": ["ULM"], "UMKC": ["UMKC"], "UNC ASHEVILLE": ["UNCA"],
    "UNC GREENSBORO": ["UNCG"], "UNC WILMINGTON": ["UNCW"], "UNLV": ["UNLV"], "USC": ["USC"], "UTAH": ["UTAH"], "UTAH STATE": ["USU"],
    "UTEP": ["UTEP"], "UTSA": ["UTSA"], "VALPARAISO": ["VAL"], "VANDERBILT": ["VAN"], "VCU": ["VCU"], "VERMONT": ["UVM"], 
    "VILLANOVA": ["NOVA"],  "VIRGINIA": ["UVA"], "VIRGINIA TECH": ["VT"], "VMI": ["VMI"], "WAKE FOREST": ["WAKE"], "WASHINGTON": ["UW"],
    "WASHINGTON STATE": ["WSU"], "WEBER STATE": ["WEB"], "WEST VIRGINIA": ["WVU"], "WESTERN CAROLINA": ["WCU"], "WESTERN ILLINOIS": ["WIU"],
    "WESTERN KENTUCKY": ["WKU"], "WESTERN MICHIGAN": ["WMU"], "WICHITA STATE": ["WICH"], "WILLIAM & MARY": ["WM"], "WINTHROP": ["WIN"],
    "WISCONSIN": ["WISC"], "WOFFORD": ["WOF"], "WRIGHT STATE": ["WRST"], "WYOMING": ["WYO"], "XAVIER": ["XAV"], "YALE": ["YALE"],
    "YOUNGSTOWN STATE": ["YSU"],
}

FUTURE_EXCLUDE_KEYWORDS = {
    "champions league", "ucl", "win the league", "to win league",
    "to win the league", "to win championship", "championship", "playoffs",
    "division winner", "relegation", "bottom of table", "top of table",
    "season wins", "regular season wins", "season", "wins", "champion", "exactly",
}

SUPPORTED_LEAGUES = {"nba", "nfl", "mlb", "ncaaf", "ncaab", "nhl"}

LEAGUE_SERIES_MAP = {
    "nba": "KXNBA",
    "nfl": "KXNFL",
    "mlb": "KXMLB",
    "nhl": "KXNHL",
    "ncaaf": "KXNCAAF",
    "ncaab": "KXNCAAB",
}

TEAM_FUZZY_THRESHOLD = 0.80  
MAX_LINE_DIFF = 3.0
DEBUG_KALSHI_MATCHING = False
TEAM_NAME_SIMILARITY = 0.70

# 🔽 NEW: date tolerances
DATE_TOLERANCE_DAYS = 5          # hard cutoff: ignore markets more than this many days away
DATE_SOFT_PENALTY = 0.10         # subtract this * day_diff from score (soft penalty)

# --- 3. DATA STRUCTURES ---

class KalshiMatchResult(TypedDict, total=False):
    matched: bool
    kalshi_available: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: str
    reason: str
    market_type: Optional[str]
    direction: Optional[str]
    game_date: Optional[datetime]
    kalshi_volume: Optional[int]

# --- 4. HELPER FUNCTIONS ---

def _debug_log(message: str, *args: Any) -> None:
    if DEBUG_KALSHI_MATCHING:
        logger.info(message, *args)

def price_to_prob(price) -> Optional[float]:
    if price is None or price == "":
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p > 1.01:
        p = p / 100.0
    if 0 <= p <= 1:
        return p
    return None

def _normalize_market_text(*parts: str) -> str:
    joined = " ".join([p or "" for p in parts])
    cleaned = joined.lower()
    cleaned = re.sub(r"[\.,\-_/]", " ", cleaned)
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def _parse_market_date(raw) -> Optional[datetime]:
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, (int, float)):
            value = float(raw)
            if value > 10_000_000_000:
                value = value / 1000.0
            return datetime.fromtimestamp(value, tz=pytz.UTC)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.UTC)
    except Exception:
        return None

def _extract_date_from_ticker(ticker: str) -> Optional[datetime]:
    if not ticker:
        return None
    match = re.search(r"(20\d{2}[01]\d[0-3]\d)", ticker)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=pytz.UTC)
        except Exception: pass
    match = re.search(r"(\d{4}-\d{2}-\d{2})", ticker)
    if match:
        try:
            return datetime.fromisoformat(match.group(1)).replace(tzinfo=pytz.UTC)
        except Exception: pass
    match = re.search(r"(\d{2}[01]\d[0-3]\d)", ticker)
    if match:
        try:
            return datetime.strptime(match.group(1), "%y%m%d").replace(tzinfo=pytz.UTC)
        except Exception: pass
    return None

def _extract_market_type(title: str, ticker: str) -> Optional[str]:
    text = ((title or "") + " " + (ticker or "")).upper()
    if "MONEYLINE" in text or re.search(r"\bML\b", text):
        return "ML"
    if "SPREAD" in text or re.search(r"\bSPR?\b", text):
        return "Spread"
    if "TOTAL" in text or "OVER" in text or "UNDER" in text or re.search(r"\bOU\b", text):
        return "Total"
    return None

def _build_team_codes(team_name: str) -> List[str]:
    if not team_name:
        return []
    codes = []
    upper_name = team_name.upper().strip()
    if upper_name in KALSHI_TEAM_ABBREVIATIONS:
        codes.extend(KALSHI_TEAM_ABBREVIATIONS[upper_name])
    for key, abbreviations in KALSHI_TEAM_ABBREVIATIONS.items():
        if key in upper_name:
            codes.extend(abbreviations)
    words = [w for w in re.split(r"\s+", team_name) if w]
    if words:
        codes.append(words[0][:3].upper())
        codes.append("".join(w[0] for w in words[:3]).upper())
    if len(words) > 1:
        codes.append(words[-1][:3].upper())
    return list(set(codes))

def _extract_teams_from_ticker(ticker: str) -> List[str]:
    if not ticker:
        return []
    clean_ticker = ticker.upper()
    prefixes = ["KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXNCAAF", "KXNCAAB", "KX"]
    for p in prefixes:
        if clean_ticker.startswith(p):
            clean_ticker = clean_ticker[len(p):]
            break
    tokens = re.findall(r"[A-Z]{2,3}", clean_ticker)
    ignore = {
        "ML", "OU", "OVE", "UND", "SPR", "TOT", "GAM", "VS", "AT", 
        "NBA", "NFL", "MLB", "NHL", "NCA", "AF", "AB"
    }
    return [t for t in tokens if t not in ignore]

def normalize_name(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- 5. MAIN MATCHING FUNCTION ---

from datetime import datetime
from typing import Dict, Any, Optional, List

def _parse_market_metadata(mkt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Lightweight parser to extract title, approximate date, teams, market type,
    and probability from a Kalshi market.
    """
    title = (mkt.get("title") or "").strip()
    if not title:
        return None

    # --- basic fields ---
    ticker = (
        mkt.get("ticker")
        or mkt.get("event_ticker")
        or mkt.get("series_ticker")
        or ""
    )

    # --- market date (from close_time / expected_expiration_time / ticker) ---
    market_dt: Optional[datetime] = None
    close_raw = mkt.get("close_time") or mkt.get("expected_expiration_time")
    if close_raw:
        try:
            market_dt = datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))
        except Exception:
            market_dt = None
    # you may have a helper like _parse_market_date or _extract_date_from_ticker;
    # if so, you can call it here as a fallback.

    # --- teams: first try title, then ticker codes ---
    upper_title = title.upper()
    teams: List[str] = []

    # Try formats like "LAKERS @ CELTICS", "MIAMI VS ORLANDO", etc.
    separators = [" VS ", " VS. ", " V ", " @ ", " AT ", " - ", " / ", " | "]
    for sep in separators:
        if sep in upper_title:
            parts = upper_title.split(sep)
            if len(parts) >= 2:
                teams = [parts[0].strip(), parts[1].strip()]
            break

    # If we still don't have 2 teams, fall back to ticker abbreviations
    if len(teams) < 2 and ticker:
        ticker_teams = _extract_teams_from_ticker(ticker)
        if len(ticker_teams) >= 2:
            teams = ticker_teams[:2]

    # --- market type & probability ---
    market_type = _extract_market_type(title, ticker)

    prob_source = (
        mkt.get("yes_price")
        or mkt.get("last_price")
        or mkt.get("last_yes_price")
        or mkt.get("implied_prob")
        or mkt.get("probability")
    )
    prob = price_to_prob(prob_source)

    return {
        "title": title,
        "market_date": market_dt,
        "teams": teams,
        "probability": prob,
        "market_type": market_type,
    }

def match_game_to_kalshi(
    league: str,
    home_team: str,
    away_team: str,
    game_time: Optional[datetime],
    integrator: "KalshiIntegrator" = None,
    status: Optional[str] = "open",
) -> KalshiMatchResult:
    """
    Match an OddsAPI game to the best Kalshi market using TEAM-FIRST scoring,
    with date as a soft constraint (penalty + hard cutoff).

    - Team codes / names drive most of the score.
    - Date difference subtracts from score and can hard-cut if too far.
    """
    league_norm = normalize_name(league)
    if league_norm and league_norm not in SUPPORTED_LEAGUES:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="league_not_supported",
        )

    kalshi = integrator or KalshiIntegrator()
    if kalshi is None:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="api_error:no_integrator",
        )

    # Normalize game info
    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    home_codes = _build_team_codes(home_team)
    away_codes = _build_team_codes(away_team)

    # Pull markets from Kalshi
    markets = kalshi.get_markets(status=status)
    if not markets:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="no_kalshi_markets_for_league",
        )

    # Parse metadata once
    parsed_markets: List[Dict[str, Any]] = []
    for mkt in markets:
        meta = _parse_market_metadata(mkt)
        if not meta:
            continue
        mkt2 = mkt.copy()
        mkt2["__meta"] = meta
        parsed_markets.append(mkt2)

    if not parsed_markets:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="no_parsable_markets",
        )

    series_prefix = LEAGUE_SERIES_MAP.get(league_norm)

    # Convert game time to aware datetime if possible
    game_dt: Optional[datetime] = None
    if isinstance(game_time, datetime):
        if game_time.tzinfo is None:
            game_dt = pytz.UTC.localize(game_time)
        else:
            game_dt = game_time.astimezone(pytz.UTC)

    # --- nested scorer ---
    def _team_score(team_code: str, target_norm: str, target_codes: List[str]) -> float:
        """
        Score how well a Kalshi team token matches a sportsbook team name.

        2.0  - exact code / abbreviation match (e.g. NYK, MIA)
        1.5  - exact normalized name match (e.g. "miami heat" == "miami heat")
        1.2  - one normalized name contains the other
        1.0  - ≥ 50% word overlap
        0.5  - some word overlap but weaker
        0.0  - no meaningful match
        """
        if not team_code:
            return 0.0

        clean_code = team_code.strip().upper()

        # 1) Code / abbreviation match (fast path)
        if clean_code in target_codes:
            return 2.0

        # 2) If TeamNameMatcher is available, use it
        if TeamNameMatcher:
            norm_code = TeamNameMatcher.normalize(team_code)
            if norm_code == target_norm:
                return 1.5
            sim = TeamNameMatcher.similarity_score(norm_code, target_norm)
            if sim >= TEAM_NAME_SIMILARITY:
                return 1.0

        # 3) Fallback: simple normalization + word overlap
        norm_code = normalize_name(team_code)

        # exact normalized name
        if norm_code == target_norm and norm_code:
            return 1.5

        # one name contained inside the other
        if norm_code and target_norm and (norm_code in target_norm or target_norm in norm_code):
            return 1.2

        # word overlap
        words_code = set(norm_code.split())
        words_target = set(target_norm.split())
        overlap = words_code & words_target
        if overlap:
            ratio = len(overlap) / max(len(words_target), 1)
            if ratio >= 0.5:
                return 1.0
            return 0.5

        return 0.0

    best_market: Optional[Dict[str, Any]] = None
    best_score: float = 0.0
    best_day_diff: Optional[int] = None

    # --- loop through all parsed markets ---
    for market in parsed_markets:
        meta = market["__meta"]
        title = meta.get("title", "").upper()
        ticker = (market.get("ticker") or market.get("event_ticker") or "").upper()

        # 0. Filter by series prefix (NBA vs NFL vs NCAAB etc.)
        if series_prefix and ticker and not ticker.startswith(series_prefix):
            continue

        # 1. Skip obvious futures / season props
        futures_noise = [
            "APPROVE", "FRANCHISE", "DRAFT", "MVP", "ROOKIE",
            "CHAMPION", "WINNER", "SEASON", "CONFERENCE", "DIVISION",
            "REGULAR SEASON WINS", "SEASON WINS"
        ]
        if any(bad in title for bad in futures_noise):
            continue

        # 2. Teams from meta
        teams = meta.get("teams") or []
        if len(teams) < 2:
            continue

        # Score both orientations
        score_home_first = (
            _team_score(teams[0], home_norm, home_codes)
            + _team_score(teams[1], away_norm, away_codes)
        )
        score_away_first = (
            _team_score(teams[0], away_norm, away_codes)
            + _team_score(teams[1], home_norm, home_codes)
        )

        team_score = max(score_home_first, score_away_first)
        if team_score <= 0.0:
            continue

        # Base score is team score
        score = team_score

        # 3. Soft date penalty + hard cutoff
        day_diff: Optional[int] = None
        market_dt: Optional[datetime] = meta.get("market_date")
        if game_dt and market_dt:
            day_diff = abs((market_dt.date() - game_dt.date()).days)
            if day_diff > DATE_TOLERANCE_DAYS:
                # too far away in time
                continue
            # penalize the score slightly the further away we get
            score -= DATE_SOFT_PENALTY * day_diff

        # 4. Small bump if we have a usable probability
        if meta.get("probability") is not None:
            score += 0.25

        if score > best_score:
            best_score = score
            best_market = market
            best_day_diff = day_diff

    # --- no viable candidate at all ---
    if not best_market:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="no_candidate_for_game",
        )

    # --- candidate exists but below acceptance threshold ---
    if best_score < TEAM_FUZZY_THRESHOLD:
        reason = f"low_score:{best_score:.2f}"
        if best_day_diff is not None:
            reason += f":day_diff={best_day_diff}"
        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason=reason,
        )

    # --- accept best market ---
    best_meta = best_market["__meta"]
    return KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=best_meta.get("title", ""),
        probability=best_meta.get("probability"),
        raw_event_id=best_market.get("event_ticker") or best_market.get("ticker"),
        league=league_norm,
        reason="ok",
        market_type=best_meta.get("market_type"),
        direction=None,
        game_date=best_meta.get("market_date"),
        kalshi_volume=best_market.get("volume"),
    )

        def _team_score(team_code: str, target_norm: str, target_codes: List[str]) -> float:
            """
            Score how well a Kalshi team token matches a sportsbook team name.

            2.0  - exact code / abbreviation match (e.g. NYK, MIA)
            1.5  - exact normalized name match (e.g. "miami heat" == "miami heat")
            1.2  - one normalized name is contained in the other
            1.0  - >= 50% word overlap
            0.5  - some word overlap but weaker
            0.0  - no meaningful match
            """
            if not team_code:
                return 0.0

            # Normalize the raw token a bit
            clean_code = team_code.strip().upper()

            # 1) Code / abbreviation match first (fast path)
            if clean_code in target_codes:
                return 2.0

            # 2) If TeamNameMatcher is available, use it
            if TeamNameMatcher:
                norm_code = TeamNameMatcher.normalize(team_code)
                if norm_code == target_norm:
                    return 1.5

                sim = TeamNameMatcher.similarity_score(norm_code, target_norm)
                if sim >= TEAM_NAME_SIMILARITY:
                    return 1.0

            # 3) Fallback: simple normalization + word overlap
            norm_code = normalize_name(team_code)

            # exact normalized name
            if norm_code == target_norm and norm_code:
                return 1.5

            # one name contained inside the other
            if norm_code and target_norm and (norm_code in target_norm or target_norm in norm_code):
                return 1.2

            # word overlap
            words_code = set(norm_code.split())
            words_target = set(target_norm.split())
            overlap = words_code & words_target
            if overlap:
                ratio = len(overlap) / max(len(words_target), 1)
                if ratio >= 0.5:
                    return 1.0
                return 0.5

            return 0.0


        score_home_first = _team_score(teams[0], home_norm, home_codes) + _team_score(teams[1], away_norm, away_codes)
        score_away_first = _team_score(teams[0], away_norm, away_codes) + _team_score(teams[1], home_norm, home_codes)

        score = max(score_home_first, score_away_first)

        if score_home_first < 1.0 and score_away_first < 1.0:
            continue

        if meta.get("probability") is not None:
            score += 0.25

        if score > best_score:
            best_score = score
            best_market = market

    # No viable market at all
    if not parsed_markets:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="no_kalshi_markets_for_league",
        )

    # We had markets, but none cleared the threshold
    if not best_market:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="no_candidate_above_min_score",
        )

    if best_score < TEAM_FUZZY_THRESHOLD:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason=f"low_score:{best_score:.2f}",
        )


    meta = best_market["__meta"]
    probability = meta.get("probability")
    if probability is None and best_market.get("ticker"):
        try:
            order = kalshi.get_orderbook(best_market.get("ticker")) or {}
            yes_levels = None
            for key in ("yes", "orderbook_yes", "levels"):
                level = (order.get(key) or order.get("orderbook", {}).get(key) or {})
                if level:
                    yes_levels = level
                    break
            if isinstance(yes_levels, list) and yes_levels:
                price_val = yes_levels[0].get("price") or yes_levels[0].get("bid")
                probability = price_to_prob(price_val)
        except Exception:
            probability = None

    direction = "YES" if probability and probability >= 0.5 else "NO"

# --- 6. INTEGRATOR CLASS ---

class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis"""

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = (
            api_key 
            or st.secrets.get("KALSHI_API_KEY") 
            or st.secrets.get("kalshi_api_key") 
            or st.session_state.get("kalshi_api_key")
            or os.environ.get("KALSHI_API_KEY")
        )
        self.api_secret = (
            api_secret 
            or st.secrets.get("KALSHI_API_SECRET") 
            or st.secrets.get("kalshi_secret_key") 
            or st.session_state.get("kalshi_secret_key")
            or os.environ.get("KALSHI_API_SECRET")
        )
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
        self.api_url = self.base_url if self.api_key else self.demo_url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        self._private_key = None
        self._auth_ready = False
        self.last_error: Optional[str] = None
        self._markets_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_duration = 300 

        if self.api_key and self.api_secret:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                key_data = self.api_secret.strip()
                if not key_data.startswith("-----BEGIN"):
                    key_data = "-----BEGIN RSA PRIVATE KEY-----\n" + key_data + "\n-----END RSA PRIVATE KEY-----"
                self._private_key = serialization.load_pem_private_key(
                    key_data.encode(), password=None, backend=default_backend(),
                )
                self._auth_ready = True
            except ImportError:
                logger.warning("cryptography library not installed - Kalshi auth disabled")
            except Exception as e:
                logger.warning(f"Could not load Kalshi RSA key: {e}")

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        if not self._private_key: return ""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            message = f"{timestamp}{method}{path}"
            signature = self._private_key.sign(
                message.encode("utf-8"),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")
        except Exception: return ""

    def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
    ) -> Optional[dict]:
        """Make an authenticated Kalshi API request using RSA-PSS signing if available."""
        import time as time_module

        url = f"{self.api_url}{endpoint}"
        timestamp = str(int(time_module.time() * 1000))
        headers = self.headers.copy()

        # Attach Kalshi auth headers if we have a key
        if self._auth_ready and self._private_key:
            # Only sign the path portion, not query string
            path_suffix = endpoint.split("?", 1)[0]
            path_to_sign = f"/trade-api/v2{path_suffix}"
            signature = self._sign_request(method.upper(), path_to_sign, timestamp)
            headers["KALSHI-ACCESS-KEY"] = self.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp

        try:
            response = requests.request(
                method.upper(),
                url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=10,
            )
            if response.status_code == 200:
                return response.json()
            else:
                self.last_error = f"API error: {response.status_code}"
                logger.warning(
                    f"Kalshi error {response.status_code}: {response.text[:200]}"
                )
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Kalshi request failed: {e}")

        return None


    def get_markets(self, category: str = "sports", status: Optional[str] = "open") -> List[Dict]:
        """Nuclear Fetch: Download markets (including recent past) and filter locally."""
        cache_key = status or "all"
        now = time.time()
        
        if cache_key in self._markets_cache and now - self._cache_time.get(cache_key, 0) < self._cache_duration:
            return copy.deepcopy(self._markets_cache[cache_key])

        all_markets = []
        try:
            cursor = None
            page_count = 0
            lookback_ts = int(now - (86400 * 2))
            
            while page_count < 20:
                # FIX: Increased sleep to 1.1s to guarantee no 429 Errors
                time.sleep(1.1)
                
                params = {"limit": 200}
                if cursor: params["cursor"] = cursor
                params["min_close_ts"] = lookback_ts 
                
                data = self._make_authenticated_request("GET", "/markets", params=params)
                if not data: break
                
                markets = data.get("markets", [])
                if not markets: break
                
                all_markets.extend(markets)
                cursor = data.get("cursor")
                page_count += 1
                if not cursor: break

            # ... (rest of the filtering logic remains the same) ...

            # Expanded Keyword Filter for All Sports
            sports_keywords = [
                "NFL", "NBA", "MLB", "NHL", "UFC", "SOCCER", "TENNIS", "FOOTBALL", "BASKETBALL",
                "HOCKEY", "BASEBALL", "COLLEGE", "NCAA", "MMA", "FIGHT",
                "SPREAD", "TOTAL", "MONEYLINE" 
            ]
            
            filtered = []
            for m in all_markets:
                title = (m.get("title") or "").upper()
                ticker = (m.get("ticker") or "").upper()
                if any(kw in title or kw in ticker for kw in sports_keywords):
                    filtered.append(m)

            if filtered:
                self._markets_cache[cache_key] = filtered
                self._cache_time[cache_key] = now
                logger.info(f"✅ Cached {len(filtered)} Kalshi markets (Nuclear Fetch)")
                return filtered
                
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
        
        return []

    # Alias methods for compatibility
    get_sports_markets = get_markets
    get_game_markets_for_events = lambda self, league: self.get_markets()
    get_sports_series = lambda self: [] # Deprecated but kept for safety
    filter_markets_closing_today = lambda self, markets: markets # Passthrough
    get_orderbook = lambda self, ticker: (self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook") or {}).get("orderbook", {})
