import pandas as pd
from app_core.feature_processing import fetch_from_espn_mlb, robust_normalize_team
from core.team_mapper import normalize_team_name

# Test MLB fetcher
stats = fetch_from_espn_mlb(2025)
print(f"MLB Teams Fetched: {len(stats)}")
if len(stats) > 0:
    print(f"Sample: {stats[0]['team_norm']}")

# Test Athletics / Single words
print("HEAT:", robust_normalize_team("HEAT"))
print("MAGIC:", robust_normalize_team("MAGIC"))
print("ATHLETICS:", robust_normalize_team("ATHLETICS"))

# Test NY / LA
print("ny rangers ->", normalize_team_name("ny rangers"))
print("Ny ->", normalize_team_name("Ny"))
print("LA Rams ->", normalize_team_name("LA Rams"))
print("LA ->", normalize_team_name("LA"))
