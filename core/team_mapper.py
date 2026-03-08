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
