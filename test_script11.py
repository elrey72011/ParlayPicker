import pandas as pd
from core.streamlit_pipeline import _apply_team_name_overrides, TEAM_NAME_OVERRIDES

df = pd.DataFrame([
    {"league": "NHL", "home_team": "Colorado", "away_team": "Florida"},
    {"league": "NHL", "home_team": "Carolina", "away_team": "Tampa Bay"},
    {"league": "NHL", "home_team": "New Jersey", "away_team": "San Jose"},
    {"league": "NHL", "home_team": "Vegas", "away_team": "Dallas"},
    {"league": "NHL", "home_team": "Buffalo", "away_team": "Calgary"},
    {"league": "NHL", "home_team": "Ottawa", "away_team": "Montreal"},
    {"league": "NHL", "home_team": "Nashville", "away_team": "Philadelphia"},
    {"league": "NHL", "home_team": "Chicago", "away_team": "Detroit"},
    {"league": "NHL", "home_team": "Columbus", "away_team": "Winnipeg"},
    {"league": "NHL", "home_team": "Minnesota", "away_team": "Pittsburgh"},
])

res = _apply_team_name_overrides(df)
print(res)
