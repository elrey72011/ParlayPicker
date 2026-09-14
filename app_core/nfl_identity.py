"""NFL-only grading aliases. Does not change immutable lock identities."""
import re
import unicodedata

_TEAMS = {
    "arizona": "cardinals", "atlanta": "falcons", "baltimore": "ravens",
    "buffalo": "bills", "carolina": "panthers", "chicago": "bears",
    "cincinnati": "bengals", "cleveland": "browns", "dallas": "cowboys",
    "denver": "broncos", "detroit": "lions", "green bay": "packers",
    "houston": "texans", "indianapolis": "colts", "jacksonville": "jaguars",
    "kansas city": "chiefs", "las vegas": "raiders", "los angeles chargers": "",
    "los angeles rams": "", "miami": "dolphins", "minnesota": "vikings",
    "new england": "patriots", "new orleans": "saints", "new york giants": "",
    "new york jets": "", "philadelphia": "eagles", "pittsburgh": "steelers",
    "san francisco": "49ers", "seattle": "seahawks", "tampa bay": "buccaneers",
    "tennessee": "titans", "washington": "commanders",
}

def _key(value):
    value = unicodedata.normalize("NFKD", str(value))
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

_ALIASES = {}
for city, mascot in _TEAMS.items():
    canonical = (city + " " + mascot).strip()
    for alias in (city, canonical, mascot):
        if alias:
            _ALIASES[alias] = canonical
# Bare New York and Los Angeles deliberately remain unresolved.

def nfl_result_name(value):
    key = _key(value)
    return _ALIASES.get(key, key).upper()
