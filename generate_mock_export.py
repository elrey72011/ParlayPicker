import pandas as pd
from datetime import datetime, timedelta
import os

yesterday = datetime.now() - timedelta(days=1)
filename = f"app_exports/best_picks_export - {yesterday.strftime('%Y-%m-%d')} mock.csv"

data = {
    'league': ['NBA', 'NHL', 'NCAAB', 'MLB', 'NFL'],
    'matchup_id': ['NBA_LAL_BOS', 'NHL_NYR_NJD', 'NCAAB_DUKE_UNC', 'MLB_NYY_BOS', 'NFL_KC_SF'],
    'matchup': ['Los Angeles Lakers @ Boston Celtics', 'New York Rangers @ New Jersey Devils', 'Duke Blue Devils @ North Carolina Tar Heels', 'New York Yankees @ Boston Red Sox', 'Kansas City Chiefs @ San Francisco 49ers'],
    'game_date': [yesterday.strftime('%Y-%m-%d')] * 5,
    'Pick_Status': ['Actionable', 'Actionable', 'Actionable', 'High Variance', 'Actionable'],
    'best_pick': ['Los Angeles Lakers', 'Under 5.5', 'Duke Blue Devils -3.5', 'New York Yankees', 'Over 47.5'],
    'expected_value': [0.05, 0.08, 0.03, 0.06, 0.02],
    'edge': [0.04, 0.06, 0.02, 0.05, 0.01],
    'Combined_Model_Prob': [0.55, 0.58, 0.53, 0.56, 0.52],
    'Combined_Market_Prob': [0.50, 0.50, 0.50, 0.50, 0.50],
    'decimal_odds': [1.95, 1.95, 1.90, 2.05, 1.90],
    'american_odds': [-105, -105, -110, +105, -110],
    'home_team': ['Boston Celtics', 'New Jersey Devils', 'North Carolina Tar Heels', 'Boston Red Sox', 'San Francisco 49ers'],
    'away_team': ['Los Angeles Lakers', 'New York Rangers', 'Duke Blue Devils', 'New York Yankees', 'Kansas City Chiefs'],
}

df = pd.DataFrame(data)
df.to_csv(filename, index=False)
print(f"Created {filename}")
