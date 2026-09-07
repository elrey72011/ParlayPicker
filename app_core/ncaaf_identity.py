"""Exact NCAAF provider aliases, isolated from other sports' identities."""
import re
import unicodedata
from core.team_mapper import normalize_team_name


def _key(value):
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


GROUPS = (
    ("missouri", "mizzou", "missouri tigers"),
    ("app state", "appalachian state", "appalachian state mountaineers"),
    ("army", "army black knights"),
    ("illinois", "illinois fighting illini"),
    ("vanderbilt", "vanderbilt commodores"),
    ("gardner webb", "gardner-webb", "gardner-webb runnin bulldogs", "gardner-webb running bulldogs"),
    ("bowling green", "bgsu", "bowling green falcons"),
    ("kennesaw state", "kennesaw state owls"),
    ("southern", "southern university", "southern university jaguars"),
    ("sam houston", "sam houston state", "sam houston state bearkats", "sam houston bearkats"),
    ("southern miss", "southern mississippi", "southern mississippi golden eagles", "southern miss golden eagles"),
    ("tcu", "tcu horned frogs"),
    ("grambling", "grambling state", "grambling state tigers"),
    ("san jose state", "san josé state", "san jose state spartans"),
    ("north dakota state", "north dakota state bison"),
)
ALIASES = {_key(alias): group[0] for group in GROUPS for alias in group}


def normalize_ncaaf_team(value):
    # Resolve full source names before the generic mapper can remove only part
    # of a mascot or map the two providers in opposite directions.
    key = _key(value)
    if not key:
        return ""
    if key in ALIASES:
        return ALIASES[key]
    mapped = _key(normalize_team_name(str(value)))
    return ALIASES.get(mapped, mapped)
