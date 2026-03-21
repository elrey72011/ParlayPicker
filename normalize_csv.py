import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("data/master_all_sports.csv")
df = pd.read_csv(csv_path)

# 1. Standardize Headers
header_map = {
    "home_avg_points": "home_ppg", "away_avg_points": "away_ppg",
    "home_def_rating": "home_oppg", "away_def_rating": "away_oppg"
}
df = df.rename(columns=header_map)

# 2. Sport-Specific Normalization Ranges
ranges = {
    "NBA": (90, 130), "NCAAB": (50, 90), "NHL": (1.5, 4.5),
    "NFL": (14, 35), "NCAAF": (14, 45)
}

for sport, (low, high) in ranges.items():
    mask = df['sport'] == sport
    for col in ['home_ppg', 'away_ppg', 'home_oppg', 'away_oppg']:
        if col in df.columns:
            df.loc[mask, col] = (df.loc[mask, col] - low) / (high - low)

# 3. Final Cleanup
df[['home_ppg', 'away_ppg', 'home_oppg', 'away_oppg']] = df[['home_ppg', 'away_ppg', 'home_oppg', 'away_oppg']].clip(0, 1)
df.to_csv(csv_path, index=False)
