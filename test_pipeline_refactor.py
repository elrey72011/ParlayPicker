import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.streamlit_pipeline import run_analysis_pipeline
import pandas as pd

# Creating a dummy spreads_df mimicking user upload
data = {
    'league': ['NBA', 'NBA', 'NBA'],
    'home_team': ['Atlanta Hawks', 'Boston Celtics', 'Cleveland Cavaliers'], # Note Cle in reverse
    'away_team': ['Cleveland Cavaliers', 'Detroit Pistons', 'Atlanta Hawks'],
    'market_type': ['spread_home', 'total_over', 'spread_away'],
    'pick_team': ['Atlanta Hawks', '', 'Fake Away'],
    'spread_line': [-3.5, pd.NA, 5.5],
    'total_line': [pd.NA, 220.5, pd.NA],
    'odds_american': [-110.0, -110.0, -110.0],
    'theover_probability': [0.55, 0.52, 0.51]
}
df = pd.DataFrame(data)

# Let's mock `fetch_live_odds_dataframe` to return some dummy novig data
import core.streamlit_pipeline
def mock_fetch_live_odds(sports=None):
    return pd.DataFrame({
        'game_id': ['1', '2'],
        'commence_time': ['2023-11-20T12:00:00Z', '2023-11-20T13:00:00Z'],
        'home_team': ['Atlanta Hawks', 'Boston Celtics'],
        'away_team': ['Cleveland Cavaliers', 'Detroit Pistons'],
        'novig_home_point': [-4.5, pd.NA],
        'novig_home_price': [102.0, pd.NA],
        'novig_away_point': [4.5, pd.NA],
        'novig_away_price': [-102.0, pd.NA],
        'novig_over_point': [pd.NA, 222.5],
        'novig_over_price': [pd.NA, -107.0],
        'novig_under_point': [pd.NA, 222.5],
        'novig_under_price': [pd.NA, -107.0]
    })
core.streamlit_pipeline.fetch_live_odds_dataframe = mock_fetch_live_odds

analysis_df, best_picks, diag = run_analysis_pipeline(sports=["NBA"], use_ml=False, spreads_df=df)
print(f"analysis_df rows: {len(analysis_df)}")
if len(analysis_df) > 0:
    print(analysis_df[['home_team', 'away_team', 'market_type', 'spread_line', 'total_line', 'odds_american', 'odds_source']].head())
