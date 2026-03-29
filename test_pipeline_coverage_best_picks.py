import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

df = pd.DataFrame({
    'league': ['NBA', 'NBA'],
    'home_team': ['Lakers', 'Bulls'],
    'away_team': ['Warriors', 'Bucks'],
    'game_date': ['2026-03-30', '2026-03-30'],
    'market_type': ['spread_home', 'total_over'],
    'expected_value': [0.1, 0.2],
    'edge': [0.05, 0.05],
    'calibrated_probability': [0.55, 0.55],
    'ml_probability': [0.55, 0.55],
    'theover_probability': [0.5, 0.5],
    'odds_american': [-110, -110],
    'market_probability': [0.5, 0.5],
    'spread_line': [-5, -5],
    'total_line': [220, 220]
})

out = build_best_picks_df(df)
print(out)
