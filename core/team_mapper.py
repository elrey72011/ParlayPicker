"""
Team name normalization for consistent matching across data sources.
"""
import re

# Keep legacy exact mappings for backwards compatibility
TEAM_MAP = {
    "FIU Panthers": "FIU",
    "UC Santa Barbara": "UCSB",
    "Texas-El Paso": "UTEP",
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
    replacements = {
        "l.a. ": "los angeles ",
        "la ": "los angeles ",
        " st.": " state",
        " st ": " state ",
        "st. ": "saint ",
        "n.c. ": "north carolina ",
        "unc ": "north carolina ",
        "s.c. ": "south carolina ",
        "u.c. ": "uc ",
        "usc ": "south carolina ",
        "penn st": "penn state",
        "northwestern st": "northwestern state",
        "nicholls st": "nicholls state",
        "michigan st": "michigan state",
    }

    for abbrev, full in replacements.items():
        normalized = normalized.replace(abbrev, full)

    # Remove all punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Collapse multiple spaces to single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Titlecase so that we return "Miami Heat" instead of "miami heat"
    # This ensures backward compatibility with tests and display consistency.
    return normalized.title()
