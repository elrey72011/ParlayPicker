import pandas as pd
import sys
from unittest.mock import MagicMock

# Mock module
mock_vertex = MagicMock()
mock_vertex.VERTEX_FEATURE_COLUMNS = ['feature_home_ppg', 'feature_home_oppg', 'feature_home_win_pct']
sys.modules['app_core.vertex_ai_endpoint'] = mock_vertex

from app_core.vertex_ai_endpoint import VERTEX_FEATURE_COLUMNS

master_df = pd.DataFrame({
    'Home': ['Lakers', 'Bulls'],
    'Away': ['Celtics', 'Heat'],
    'feature_home_ppg': [110.0, 105.0],
    'feature_home_oppg': [108.0, 106.0],
    'feature_home_win_pct': [0.6, 0.5],
    'Implied_Prob': [0.55, 0.45]
})

print(f"Master columns: {master_df.columns.tolist()}")
print(f"Vertex columns: {VERTEX_FEATURE_COLUMNS}")

# --- LOGIC EXTRACTED FROM streamlit_app.py ---
inference_df = master_df[VERTEX_FEATURE_COLUMNS].copy()

# 2. Batch Prediction Call
for idx, row in inference_df.iterrows():
    game_id = master_df.loc[idx, 'Home'] + " vs " + master_df.loc[idx, 'Away']
    home_stats = {
        'ppg': row.get('feature_home_ppg'),
        'oppg': row.get('feature_home_oppg'),
        'win_pct': row.get('feature_home_win_pct')
    }
    print(f"DEBUG MODEL INPUTS for {game_id}: {home_stats}")
