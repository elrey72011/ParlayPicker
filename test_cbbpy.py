import cbbpy.mens_scraper as cbb_s
import pandas as pd

# Let's see what cbb_s has and how we can get team stats
print("Functions in cbb_s:", [f for f in dir(cbb_s) if not f.startswith('_')])

# Maybe get_games_season?
try:
    df = cbb_s.get_games_season(season=2024)
    print("get_games_season returned columns:", df.columns.tolist() if df is not None else "None")
    if df is not None and not df.empty:
        print(df.head(2))
except Exception as e:
    print("Error calling get_games_season:", e)
