import pandas as pd
import sys
from unittest.mock import MagicMock

# Mock streamlit and app_core dependencies
sys.modules['streamlit'] = MagicMock()
sys.modules['app_core.vertex_ai_endpoint'] = MagicMock()
sys.modules['app_core.sentiment_pipeline'] = MagicMock()
sys.modules['app_core.team_name_matcher'] = MagicMock()
sys.modules['app_core.apisports'] = MagicMock()
sys.modules['app_core.feature_processing'] = MagicMock()
sys.modules['app_core.sportsdata'] = MagicMock()
sys.modules['vertex_master_analyzer'] = MagicMock()
sys.modules['app_core.kalshi_integrator'] = MagicMock()
sys.modules['app_core.llm_assistant'] = MagicMock()
sys.modules['app_core.reddit_sentiment'] = MagicMock()

# Setup mocks for imports inside functions
from app_core.vertex_ai_endpoint import VERTEX_FEATURE_COLUMNS, predict_win_probabilities

# Mock VERTEX_FEATURE_COLUMNS
VERTEX_FEATURE_COLUMNS[:] = ['feature_home_ppg', 'feature_home_oppg', 'feature_home_win_pct']

# We will manually extract the logic block from streamlit_app.py since we can't easily run the whole app
# I'll create a dataframe resembling master_df and run the modified logic

master_df = pd.DataFrame({
    'Home': ['Lakers', 'Bulls'],
    'Away': ['Celtics', 'Heat'],
    'feature_home_ppg': [110.0, 105.0],
    'feature_home_oppg': [108.0, 106.0],
    'feature_home_win_pct': [0.6, 0.5],
    'Implied_Prob': [0.55, 0.45]
})

print("Running debug print logic verification...")

# --- LOGIC EXTRACTED FROM streamlit_app.py ---
# 1. Sanitize the feature batch
inference_df = master_df[VERTEX_FEATURE_COLUMNS].copy()
# ... (skipping sanitation loop for test simplicity) ...

# 2. Batch Prediction Call
# Debug output as requested: print inputs for each game
for idx, row in inference_df.iterrows():
    game_id = master_df.loc[idx, 'Home'] + " vs " + master_df.loc[idx, 'Away']
    # Construct dictionary of relevant stats for debug
    home_stats = {
        'ppg': row.get('feature_home_ppg'),
        'oppg': row.get('feature_home_oppg'),
        'win_pct': row.get('feature_home_win_pct')
    }
    print(f"DEBUG MODEL INPUTS for {game_id}: {home_stats}")

probs = predict_win_probabilities(inference_df)
# ...
