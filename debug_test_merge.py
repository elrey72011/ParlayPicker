import pandas as pd
from core.streamlit_pipeline import _matchup_id

# Testing the _matchup_id formatting
data = {
    'home_team': ['Lakers', 'Heat'],
    'away_team': ['Warriors', 'Knicks'],
    'game_date': [pd.Timestamp('2026-03-30T00:00:00Z'), pd.Timestamp('2026-03-30T00:00:00Z')]
}
df = pd.DataFrame(data)
# df['game_date'] = _et_day_string(_game_dates(df))
# Actually just test the _matchup_id functionality inside pipeline

print(_matchup_id(df))
