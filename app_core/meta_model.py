import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import xgboost as xgb
import os

logger = logging.getLogger(__name__)

# Define features used in the meta-model
META_FEATURES = [
    'implied_prob', 'implied_missing',
    'kalshi_prob', 'kalshi_missing',
    'model_prob', 'model_missing',
    'theover_prob', 'theover_missing',
    'sentiment_adj', 'sentiment_missing',
    'is_spread', 'is_total',
    'is_nfl', 'is_nba', 'is_nhl', 'is_ncaab', 'is_ncaaf'
]

def extract_meta_features(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts features for the meta-model from a game row.
    Imputes 0.5 for missing probabilities and sets missing flags.
    """
    features = {}

    # 1. Market Implied Probability
    imp = row.get('Implied_Prob')
    if imp is not None and not pd.isna(imp):
        features['implied_prob'] = float(imp)
        features['implied_missing'] = 0.0
    else:
        features['implied_prob'] = 0.5
        features['implied_missing'] = 1.0

    # 2. Kalshi Probability
    kp = row.get('kalshi_prob_for_pick')
    if kp is None or pd.isna(kp):
        kp = row.get('spread_prob_pick_kalshi')
    if kp is None or pd.isna(kp):
        kp = row.get('total_prob_pick_kalshi')

    if kp is not None and not pd.isna(kp):
        features['kalshi_prob'] = float(kp)
        features['kalshi_missing'] = 0.0
    else:
        features['kalshi_prob'] = 0.5
        features['kalshi_missing'] = 1.0

    # 3. Model Probability
    mp = row.get('model_spread_prob')
    if mp is None or pd.isna(mp):
        mp = row.get('model_total_prob')
    if mp is None or pd.isna(mp):
        mp = row.get('AI_Prob')

    if mp is not None and not pd.isna(mp):
        features['model_prob'] = float(mp)
        features['model_missing'] = 0.0
    else:
        features['model_prob'] = 0.5
        features['model_missing'] = 1.0

    # 4. TheOver Probability
    to = row.get('theover_prob')
    if to is None or pd.isna(to):
        to = row.get('theover_prob_used')

    if to is not None and not pd.isna(to):
        features['theover_prob'] = float(to)
        features['theover_missing'] = 0.0
    else:
        features['theover_prob'] = 0.5
        features['theover_missing'] = 1.0

    # 5. Sentiment Adjustment
    sent = row.get('sentiment_adj')
    if sent is None or pd.isna(sent):
        sent = row.get('spread_sentiment_adj')
    if sent is None or pd.isna(sent):
        sent = row.get('total_sentiment_adj')

    if sent is not None and not pd.isna(sent):
        # Scale sentiment from adjustment (e.g. +/- 0.03) to prob base (0.5 + adj)
        features['sentiment_adj'] = 0.5 + float(sent)
        features['sentiment_missing'] = 0.0
    else:
        features['sentiment_adj'] = 0.5
        features['sentiment_missing'] = 1.0

    # 6. Market Type One-Hot
    market = str(row.get('Market') or row.get('best_pick_type') or '').upper()
    features['is_spread'] = 1.0 if 'SPREAD' in market else 0.0
    features['is_total'] = 1.0 if 'TOTAL' in market else 0.0

    # 7. League One-Hot
    league = str(row.get('league', '')).upper()
    features['is_nfl'] = 1.0 if league == 'NFL' else 0.0
    features['is_nba'] = 1.0 if league == 'NBA' else 0.0
    features['is_nhl'] = 1.0 if league == 'NHL' else 0.0
    features['is_ncaab'] = 1.0 if league == 'NCAAB' else 0.0
    features['is_ncaaf'] = 1.0 if league == 'NCAAF' else 0.0

    return features

def predict_meta_model(row: Dict[str, Any], meta_model: xgb.Booster) -> Optional[float]:
    """
    Predicts the final blended probability using the trained meta-model.
    """
    try:
        features = extract_meta_features(row)
        df_in = pd.DataFrame([features])[META_FEATURES]
        dmatrix = xgb.DMatrix(df_in)
        prediction = meta_model.predict(dmatrix)
        # Assuming binary classification logistic regression model
        return float(prediction[0])
    except Exception as e:
        logger.error(f"Meta-model prediction failed: {e}")
        return None

def load_meta_model(model_path: str = 'models/meta_model.json') -> Optional[xgb.Booster]:
    """
    Loads the trained meta-model.
    """
    if os.path.exists(model_path):
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            return model
        except Exception as e:
            logger.error(f"Failed to load meta-model: {e}")
            return None
    return None
