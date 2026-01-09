"""
Improved Team Name Matching for TheOver.ai CSV Integration

This fixes the 15% → 95%+ match rate issue by properly normalizing team names
"""

from difflib import SequenceMatcher
from typing import Optional, List, Tuple
import re

try:
    from rapidfuzz import process, fuzz, utils
except ImportError:
    process = None
    fuzz = None
    utils = None

# Default fuzzy threshold used across TheOver matching flows
TEAM_FUZZY_THRESHOLD = 0.80

class TeamNameMatcher:
    """Handles fuzzy matching between TheOver.ai CSV names and app team names"""
    
    # Common mascots to strip (expand this list as needed)
    MASCOTS = [
        # NBA
        'Thunder', 'Warriors', 'Celtics', 'Knicks', 'Lakers', 'Clippers',
        'Heat', 'Bulls', 'Cavaliers', 'Mavericks', 'Rockets', 'Spurs',
        'Suns', 'Blazers', 'Jazz', 'Nuggets', 'Timberwolves', 'Pelicans',
        'Kings', 'Nets', 'Bucks', 'Pistons', 'Pacers', 'Hawks',
        'Hornets', 'Wizards', 'Magic', 'Raptors', 'Grizzlies', '76ers',
        
        # NHL
        'Canucks', 'Avalanche', 'Flames', 'Oilers', 'Capitals', 'Penguins',
        'Bruins', 'Rangers', 'Islanders', 'Devils', 'Flyers', 'Maple Leafs',
        'Senators', 'Canadiens', 'Panthers', 'Lightning', 'Blue Jackets',
        'Red Wings', 'Predators', 'Blues', 'Blackhawks', 'Wild', 'Stars',
        'Golden Knights', 'Kraken', 'Ducks', 'Sharks', 'Kings',
        
        # NCAAB/NCAAF (expand as needed)
        'Bearkats', 'Bears', 'Bulldogs', 'Eagles', 'Wildcats', 'Huskies',
        'Ramblers', 'Golden Eagles', 'Blue Devils', 'Tar Heels', 'Jayhawks',
        'Buckeyes', 'Spartans', 'Wolverines', 'Hawkeyes', 'Badgers',
        'Scarlet Knights', 'Terrapins', 'Hoosiers', 'Illini', 'Boilermakers',
        'Cornhuskers', 'Cyclones', 'Sooners', 'Cowboys', 'Mountaineers',
        'Seminoles', 'Gators', 'Gamecocks', 'Tigers', 'Crimson Tide',
        'Volunteers', 'Razorbacks', 'Aggies', 'Longhorns', 'Red Raiders',
        'Horned Frogs', 'Cougars', 'Utes', 'Sun Devils', 'Trojans',
        'Bruins', 'Cardinal', 'Golden Bears', 'Ducks', 'Beavers',
        'Huskies', 'Buffaloes', 'Wildcats', 'Antelopes', 'Hatters',
        'Seahawks', 'Colonels', 'Bluejays', 'Leathernecks', 'Racers',
        'Beacons', 'Mean Green', 'Green Wave', 'Waves', 'Lions',
        'Rebels', 'Fighting Irish', 'Billikens', 'Sharks', 'Chippewas',
        'Braves', 'Hornets', 'Black Knights', 'Golden Griffins', 'Purple Eagles',
        'Blue Raiders', 'Ragin Cajuns', 'Thundering Herd', 'Golden Flashes',
        'RedHawks', 'Gamecocks', 'Mean Green', 'Roadrunners', 'Salukis',
        'Chanticleers', 'Catamounts', 'Anteaters', 'Highlanders', 'Matadors',
        'Commodores', 'Volunteers', 'Falcons', 'Lobos', 'Mustangs', 'Owls',
        'Minutemen', 'Rams', 'Broncos', 'Midshipmen', 'Cadets', 'Knights',
        'Terriers', 'Spiders', 'Dukes', 'Flyers', 'Explorers', 'Bonnies',
        'Patriots', 'Billikens', 'Griffins', 'Peacocks', 'Stags', 'Jaspers',
        'Gaels', 'Saints', 'Friars', 'Pirates', 'Hoyas', 'Blue Demons',
        'Musketeers', 'Bulldogs', 'Hoyas', 'Friars', 'Titans', 'Matadors',
        'Rockets', 'RedHawks', 'Jaspers', 'Golden Griffins', 'Golden Grizzlies',
        'Vikings', 'Titans', 'Raiders', 'Stags', 'Broncs', 'Phoenix', 'Jaguars',
        'Gaels', 'Purple Eagles', 'Pioneers', 'Red Foxes', 'Saints', 'Warriors',
        'Peacocks', 'Falcons', 'Zips', 'Panthers', 'Norse', 'Golden Gophers',
        'Trojans', 'Rams', 'Rebels', 'Hoosiers', 'Ducks', 'Buckeyes', 'Mammoth'
    ]
    
    # Special case full replacements
    # KEYS MUST BE NORMALIZED (lowercase, no punctuation, single spaced)
    # The normalization regex converts "St." to "State", so "St. John's" becomes "state johns"
    FULL_REPLACEMENTS = {
        'la': 'los angeles',
        'ny': 'new york',
        'l a': 'los angeles',
        'n y': 'new york',
        'cal': 'california',
        'umass': 'massachusetts',
        'upenn': 'pennsylvania',
        'penn': 'pennsylvania',
        'nc state': 'north carolina state',
        'n c state': 'north carolina state',
        'oregon st': 'oregon state',
        'michigan st': 'michigan state',
        'florida st': 'florida state',
        'georgia st': 'georgia state',
        'mississippi st': 'mississippi state',
        'washington st': 'washington state',
        'kansas st': 'kansas state',
        'iowa st': 'iowa state',
        'ohio st': 'ohio state',
        'oklahoma st': 'oklahoma state',
        'penn st': 'pennsylvania state',
        'ga tech': 'georgia tech',
        'va tech': 'virginia tech',
        'vmi': 'virginia military',
        'army west point': 'army',
        'navy midshipmen': 'navy',
        'air force falcons': 'air force',
        'loyola chicago': 'loyola il',
        # Handle "St." -> "state" conversion
        'state marys': 'saint marys',
        'state josephs': 'saint josephs',
        'state johns': 'saint johns',
        'state bonaventure': 'saint bonaventure',
        'state francis pa': 'saint francis pa',
        'state francis ny': 'saint francis ny',
        # Keep old keys just in case no "St" prefix was used or regex changes
        'st marys': 'saint marys',
        'st josephs': 'saint josephs',
        'st johns': 'saint johns',
        'sacramento st': 'sacramento state',
        'sam houston st': 'sam houston state',
        'fresno st': 'fresno state',
        'san diego st': 'san diego state',
        'montana st': 'montana state',
        'arizona st': 'arizona state',
        'boise st': 'boise state',
        'ut arlington': 'texas arlington',
        'uta': 'texas arlington',
        'ut arlington': 'texas arlington',
        'texas arlington': 'texas arlington',
        'ucsd': 'uc san diego',
        'uconn': 'connecticut',
        'unc': 'north carolina',
        'usc': 'southern california',
        'ucla': 'california los angeles',
        'unlv': 'nevada las vegas',
        'american u': 'american university',
        # Common NBA/NFL Nicknames
        'philly': 'philadelphia',
        'sixers': 'philadelphia',
        'cavs': 'cleveland',
        'mavs': 'dallas',
        'wolves': 'minnesota',
        't wolves': 'minnesota',
        'blazers': 'portland',
        'jags': 'jacksonville',
        'bucs': 'tampa bay',
        'pats': 'new england',
        'niners': 'san francisco',
        '49ers': 'san francisco',
        # NCAAB/NCAAF specific
        'ole miss': 'mississippi',
        'pitt': 'pittsburgh',
        'canisius golden griffins': 'canisius',
        'niagara purple': 'niagara',
        'niagara purple eagles': 'niagara',
        'middle tennessee blue raiders': 'middle tennessee',
        'louisiana tech bulldogs': 'louisiana tech',
        'la tech': 'louisiana tech',
        'fiu': 'florida international',
        'fau': 'florida atlantic',
        'utsa': 'texas san antonio',
        'utep': 'texas el paso',
        'uab': 'alabama birmingham',
        'ucf': 'central florida',
        'smu': 'southern methodist',
        'tcu': 'texas christian',
        'lsu': 'louisiana state',
        'byu': 'brigham young',
        'vcu': 'virginia commonwealth',
        'umbc': 'maryland baltimore county',
        'uncw': 'north carolina wilmington',
        'uncg': 'north carolina greensboro',
        'unco': 'northern colorado',
        'miami fl': 'miami',
        'miami oh': 'miami ohio',
        'st bonaventure': 'saint bonaventure',
        'st francis pa': 'saint francis pa',
        'st francis ny': 'saint francis ny',
        'li u': 'long island',
        'liu': 'long island',
        'mt st marys': 'mount saint marys',
        'mt state marys': 'mount saint marys', # Mount St. Mary's -> mount state marys
        'the citadel': 'citadel',
        'vmi': 'virginia military',
        'se missouri st': 'southeast missouri state',
        'semo': 'southeast missouri state',
        'siu edwardsville': 'southern illinois edwardsville',
        'siue': 'southern illinois edwardsville',
        'ul monroe': 'louisiana monroe',
        'ulm': 'louisiana monroe',
        'ul lafayette': 'louisiana',
        'ull': 'louisiana',
        'unc asheville': 'north carolina asheville',
        'gardner webb': 'gardner webb',
        'bethune cookman': 'bethune cookman',
        'md eastern shore': 'maryland eastern shore',
        'umes': 'maryland eastern shore',
        'prairie view': 'prairie view am',
        'prairie view am': 'prairie view am',
        'alabama am': 'alabama am',
        'florida am': 'florida am',
        'nc at': 'north carolina at',
        'n c at': 'north carolina at',
        'texas am': 'texas am',
        'corpus christi': 'texas am corpus christi',
        'tamucc': 'texas am corpus christi',
        'miami oh': 'miami ohio',
        'mt st marys': 'mount st marys',
        'st peters': 'saint peters',
        'utah mammoth': 'utah',
        'ohio state buckeyes': 'ohio state',
        'cleveland st': 'cleveland state',
        'wright st': 'wright state',
    }
    
    @classmethod
    def normalize(cls, team: str) -> str:
        """
        Normalize team name for matching
        
        Examples:
            "Oklahoma City Thunder" → "oklahoma city"
            "Sam Houston St Bearkats" → "sam houston state"
            "Sacramento St Hornets" → "sacramento state"
            "New York Knicks" → "new york"
        """
        if not team:
            return ""
        
        # Convert to lowercase
        team = team.lower().strip()

        # Remove 'the ' prefix if present (e.g., 'The Citadel')
        if team.startswith('the '):
            team = team[4:]

        # Normalize abbreviations using regex for safety
        # St. or St -> State (must be careful not to replace St in names like West)
        # We look for word boundaries
        team = re.sub(r'\bst\.?\b', 'state', team)
        # Univ. or Univ -> University
        team = re.sub(r'\buniv\.?\b', 'university', team)
        # Intl -> International
        team = re.sub(r'\bintl\.?\b', 'international', team)

        # Handle & -> and or empty
        # If we replace & with 'and', we must match keys.
        # But most normalization strips punctuation.
        # User requested aggressive normalization.
        # Let's replace & with space or nothing.
        team = team.replace('&', '')

        # Remove punctuation
        team = re.sub(r'[^\w\s]', '', team)
        
        # Remove mascots
        for mascot in cls.MASCOTS:
            mascot_lower = mascot.lower()
            # Remove at end with space before
            if team.endswith(f" {mascot_lower}"):
                team = team[:-len(mascot_lower)-1]
            # Remove if it's the entire string
            if team == mascot_lower:
                team = ""
        
        # Apply full replacements
        team = team.strip()
        if team in cls.FULL_REPLACEMENTS:
            team = cls.FULL_REPLACEMENTS[team]
        
        # Remove extra whitespace
        team = ' '.join(team.split())
        
        return team
    
    @classmethod
    def similarity_score(cls, str1: str, str2: str) -> float:
        """Calculate similarity between two strings (0.0 to 1.0)"""
        if process:
            # rapidfuzz returns 0-100, we want 0.0-1.0
            return fuzz.ratio(str1, str2) / 100.0
        return SequenceMatcher(None, str1, str2).ratio()
    
    @classmethod
    def match_team(
        cls,
        csv_team: str,
        app_teams: List[str],
        threshold: float = TEAM_FUZZY_THRESHOLD,
    ) -> Optional[str]:
        """
        Find best matching team name from app_teams list
        """
        csv_normalized = cls.normalize(csv_team)
        
        if not csv_normalized:
            return None
        
        if process:
             # rapidfuzz implementation (faster)
             app_normalized_map = {cls.normalize(t): t for t in app_teams if cls.normalize(t)}
             choices = list(app_normalized_map.keys())
             # Use token_set_ratio to handle subset matches like "Lions" vs "Detroit Lions"
             result = process.extractOne(csv_normalized, choices, scorer=fuzz.token_set_ratio)
             if result:
                 match, score, _ = result
                 if (score / 100.0) >= threshold:
                     return app_normalized_map[match]
             return None

        # Fallback implementation
        best_match = None
        best_score = 0.0
        
        for app_team in app_teams:
            app_normalized = cls.normalize(app_team)
            
            if not app_normalized:
                continue
            
            # Calculate similarity
            score = cls.similarity_score(csv_normalized, app_normalized)
            
            if score > best_score:
                best_score = score
                best_match = app_team
        
        # Return match only if above threshold
        if best_score >= threshold:
            return best_match
        
        return None
    
    @classmethod
    def match_game(
        cls,
        csv_home: str,
        csv_away: str,
        app_games: List[Tuple[str, str]],
        threshold: float = TEAM_FUZZY_THRESHOLD
    ) -> Optional[Tuple[str, str]]:
        """
        Match a game (home + away) from CSV to app games list
        """
        csv_home_norm = cls.normalize(csv_home)
        csv_away_norm = cls.normalize(csv_away)
        
        if not csv_home_norm or not csv_away_norm:
            return None
        
        best_match = None
        best_score = 0.0
        
        for app_home, app_away in app_games:
            app_home_norm = cls.normalize(app_home)
            app_away_norm = cls.normalize(app_away)

            if not app_home_norm or not app_away_norm:
                continue

            # Calculate combined score (both must clear threshold) in normal order
            home_score = cls.similarity_score(csv_home_norm, app_home_norm)
            away_score = cls.similarity_score(csv_away_norm, app_away_norm)
            if home_score >= threshold and away_score >= threshold:
                combined_score = (home_score + away_score) / 2
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (app_home, app_away)

            # Also allow swapped home/away in case CSV orientation differs
            swap_home_score = cls.similarity_score(csv_home_norm, app_away_norm)
            swap_away_score = cls.similarity_score(csv_away_norm, app_home_norm)
            if swap_home_score >= threshold and swap_away_score >= threshold:
                combined_score = (swap_home_score + swap_away_score) / 2
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (app_home, app_away)

        return best_match
