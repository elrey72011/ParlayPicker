import pandas as pd
from core.streamlit_pipeline import _apply_team_name_overrides, TEAM_NAME_OVERRIDES

df = pd.DataFrame([
    {"league": "NHL", "home_team": "Colorado", "away_team": "Florida"},
    {"league": "NHL", "home_team": "Carolina", "away_team": "Tampa Bay"},
    {"league": "NHL", "home_team": "New Jersey", "away_team": "San Jose"},
    {"league": "NHL", "home_team": "Vegas", "away_team": "Dallas"},
])

TEAM_NAME_OVERRIDES.update({
    ("NHL", "FLORIDA"): "Florida Panthers",
    ("NHL", "CAROLINA"): "Carolina Hurricanes",
    ("NHL", "TAMPA BAY"): "Tampa Bay Lightning",
    ("NHL", "NEW JERSEY"): "New Jersey Devils",
    ("NHL", "SAN JOSE"): "San Jose Sharks",
    ("NHL", "VEGAS"): "Vegas Golden Knights",
})

res = _apply_team_name_overrides(df)
print(res)
