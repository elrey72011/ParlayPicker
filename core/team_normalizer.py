TEAM_MAP = {
    "LA Lakers": "Los Angeles Lakers",
    "LAL": "Los Angeles Lakers",
    "BOS": "Boston Celtics",
}


def normalize_team(name):
    return TEAM_MAP.get(name, name)
