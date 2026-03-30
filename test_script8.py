import pandas as pd
from app_core.feature_processing import robust_normalize_team

for team in ["Colorado", "Florida", "Carolina", "Tampa Bay", "New Jersey", "San Jose", "Vegas", "Dallas", "Washington"]:
    print(f"{team} -> {robust_normalize_team(team)}")

for team in ["Colorado", "Florida", "Carolina", "Tampa Bay", "New Jersey", "San Jose", "Vegas", "Dallas", "Washington"]:
    print(f"{team} (NHL) -> {robust_normalize_team(team, is_nhl=True)}")
