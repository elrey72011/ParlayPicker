TEAM_REMAP = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "GS Warriors": "Golden State Warriors",
    "NY Knicks": "New York Knicks",
}


def normalize_team(name):
    return TEAM_REMAP.get(name, name)
