from __future__ import annotations

TEAM_MAP = {
    "L.A. Clippers": "LA Clippers",
    "L.A. Lakers": "LA Lakers",
    "FIU Panthers": "FIU",
    "St. Bonaventure": "St Bonaventure",
    "UC Santa Barbara": "UCSB",
    "Texas-El Paso": "UTEP",
    "LAL": "Los Angeles Lakers",
    "LA Lakers": "Los Angeles Lakers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "LAC": "Los Angeles Clippers",
    "LA Clippers": "Los Angeles Clippers",
    "Los Angeles Clippers": "Los Angeles Clippers",
}


def normalize_team_name(name: str) -> str:
    if name is None:
        return name
    return TEAM_MAP.get(str(name).strip(), str(name).strip())
