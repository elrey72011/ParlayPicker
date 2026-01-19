"""
Configuration file for Sports Betting ML Pipeline
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
TRAINING_DATA_DIR = PROJECT_ROOT / 'training_data'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# Create directories if they don't exist
for directory in [DATA_DIR, TRAINING_DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
API_KEYS = {
    'NFL': os.getenv('SPORTSDATA_NFL_KEY', 'YOUR_NFL_API_KEY'),
    'NBA': os.getenv('SPORTSDATA_NBA_KEY', 'YOUR_NBA_API_KEY'),
    'NCAAB': os.getenv('SPORTSDATA_NCAAB_KEY', 'YOUR_NCAAB_API_KEY'),
    'NCAAF': os.getenv('SPORTSDATA_NCAAF_KEY', 'YOUR_NCAAF_API_KEY'),
    'NHL': os.getenv('SPORTSDATA_NHL_KEY', 'YOUR_NHL_API_KEY')
}

# Google Cloud Configuration
GCP_CONFIG = {
    'project_id': os.getenv('GCP_PROJECT_ID', 'your-project-id'),
    'location': os.getenv('GCP_LOCATION', 'us-central1'),
    'staging_bucket': os.getenv('GCP_STAGING_BUCKET', 'your-bucket-name')
}

# Training Configuration
TRAINING_CONFIG = {
    # Seasons to train on
    'seasons': ['2021', '2022', '2023', '2024'],
    
    # Train/test split
    'test_size': 0.2,
    
    # Model parameters
    'xgboost_params': {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    },
    
    # Cross-validation
    'cv_folds': 5
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    # Rolling windows for recent form
    'rolling_windows': [3, 5, 10],
    
    # ELO rating K-factor
    'elo_k_factor': 20,
    
    # Head-to-head window
    'h2h_window': 10,
    
    # Minimum games for rolling stats
    'min_games': 1
}

# Betting Strategy Configuration
BETTING_CONFIG = {
    # Minimum edge required
    'min_edge': 0.03,  # 3%
    
    # High confidence threshold
    'high_confidence_ev': 0.10,  # 10%
    
    # Parlay configuration
    'parlay_sizes': [2, 3, 4, 5],
    'max_parlays_per_size': 10,
    'allow_correlated_parlays': False,
    
    # Kelly Criterion
    'use_kelly': True,
    'kelly_fraction': 0.25,  # Use 1/4 Kelly
    'max_bet_size': 0.10,  # Max 10% of bankroll per bet
    
    # Bankroll management
    'starting_bankroll': 10000,
    'min_bankroll': 5000
}

# Sport-specific configurations
SPORT_CONFIGS = {
    'NFL': {
        'max_games_per_week': 16,
        'season_start_month': 9,
        'season_end_month': 2,
        'importance_factors': {
            'rest_days': 1.2,
            'injuries': 1.5,
            'weather': 1.3
        }
    },
    'NBA': {
        'max_games_per_day': 15,
        'season_start_month': 10,
        'season_end_month': 6,
        'importance_factors': {
            'back_to_back': 1.5,
            'pace': 1.3,
            'home_court': 1.1
        }
    },
    'NCAAB': {
        'max_games_per_day': 50,
        'season_start_month': 11,
        'season_end_month': 4,
        'importance_factors': {
            'conference': 1.4,
            'tournament': 1.6
        }
    },
    'NCAAF': {
        'max_games_per_week': 50,
        'season_start_month': 8,
        'season_end_month': 1,
        'importance_factors': {
            'rivalry': 1.3,
            'bowl_game': 1.5
        }
    },
    'NHL': {
        'max_games_per_day': 15,
        'season_start_month': 10,
        'season_end_month': 6,
        'importance_factors': {
            'goalie': 1.6,
            'back_to_back': 1.4
        }
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'pipeline.log'
}

# theover.ai Integration
THEOVER_CONFIG = {
    'data_dir': DATA_DIR / 'theover',
    'csv_format': 'theover_{sport}_{date}.csv',
    'required_columns': [
        'GameID', 'HomeTeam', 'AwayTeam', 'HomeMoneyLine', 'AwayMoneyLine',
        'PointSpread', 'OverUnder', 'HomeSpreadOdds', 'OverOdds', 'UnderOdds'
    ]
}

# Output Configuration
OUTPUT_CONFIG = {
    'betting_card_file': OUTPUT_DIR / 'betting_card_{date}.csv',
    'predictions_file': OUTPUT_DIR / 'predictions_{sport}_{date}.csv',
    'model_metrics_file': MODELS_DIR / 'metrics_{sport}.json',
    'feature_importance_file': MODELS_DIR / 'feature_importance_{sport}.csv'
}

# Notification Configuration (optional)
NOTIFICATION_CONFIG = {
    'enabled': False,
    'email': os.getenv('NOTIFICATION_EMAIL', ''),
    'webhook_url': os.getenv('SLACK_WEBHOOK_URL', ''),
    'notify_on': ['high_value_bets', 'model_training_complete', 'errors']
}

# --- ADD THESE NEW KEYS ---
# The Odds API (for live odds)
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY', 'b722c798f7bca605da45a09dba155152')

# Gemini API Key (Removed dummy fallback to prevent invalid key usage)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Vertex AI Configuration
VERTEX_CONFIG = {
    'project_id': GCP_CONFIG['project_id'],
    'location': GCP_CONFIG['location'],
    'endpoint_id': os.getenv('VERTEX_ENDPOINT_ID', '155434660283809792'), # Your Working ID
    'feature_cols': [
        "home_win_pct", "away_win_pct", "home_avg_points", "away_avg_points",
        "home_form_last5", "away_form_last5", "home_pd_last5", "away_pd_last5",
        "home_sos_last5", "away_sos_last5", "win_pct_diff", "avg_points_diff",
        "form_diff", "pd_last5_diff", "sos_diff", "def_rating_diff", "streak_diff",
        "spread_normalized", "public_betting_centered", "sharp_vs_public",
        "rest_advantage", "back_to_back", "primetime_game", "division_game",
        "injuries_impact", "weather_factor", "line_movement", "total_movement",
        "model_consensus", "theover_probability"
    ]
}

# Path to your master history file
MASTER_DATA_FILE = DATA_DIR / 'master_all_sports.csv'
