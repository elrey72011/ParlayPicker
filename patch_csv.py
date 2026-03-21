import pandas as pd
from pathlib import Path

csv_path = Path("data/master_all_sports.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "home_avg_points": "home_ppg",
        "away_avg_points": "away_ppg",
        "home_def_rating": "home_oppg",
        "away_def_rating": "away_oppg"
    })
    df.to_csv(csv_path, index=False)
    print("Successfully patched data/master_all_sports.csv headers.")
else:
    print("CSV file not found!")
