import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import VERTEX_CONFIG

VERTEX_FEATURE_COLUMNS = VERTEX_CONFIG['feature_cols']

def build_features_from_master(df: pd.DataFrame) -> pd.DataFrame:
    """Maps columns from master_all_sports.csv to the strict 21 Vertex schema."""
    out = pd.DataFrame(index=df.index)

    mapping = {
        "implied_home_prob": "implied_home_prob",
        "sentiment_diff": "sentiment_diff",
        "kalshi_prob": "kalshi_prob",
        "injuries_home_count": "injuries_impact",
        "injuries_away_count": "injuries_away_count",
        "weather_flag": "weather_factor",
        "feature_home_win_pct": "home_win_pct",
        "feature_home_ppg": "home_ppg",
        "feature_home_oppg": "home_oppg",
        "feature_home_streak": "home_streak",
        "feature_away_win_pct": "away_win_pct",
        "feature_away_ppg": "away_ppg",
        "feature_away_oppg": "away_oppg",
        "feature_away_streak": "away_streak",
        "feature_diff_win_pct": "feature_diff_win_pct",
        "feature_diff_ppg": "feature_diff_ppg",
        "feature_diff_oppg": "feature_diff_oppg",
        "feature_diff_last5": "feature_diff_last5",
        "feature_diff_streak": "feature_diff_streak",
        "feature_home_rest_days": "rest_advantage",
        "feature_away_rest_days": "feature_away_rest_days"
    }

    # Calculate derived/synthetic columns if missing
    if "feature_diff_win_pct" not in df.columns:
        df["feature_diff_win_pct"] = df["home_win_pct"].fillna(0.5) - df["away_win_pct"].fillna(0.5)
    if "feature_diff_ppg" not in df.columns:
        df["feature_diff_ppg"] = df["home_ppg"].fillna(0.0) - df["away_ppg"].fillna(0.0)
    if "feature_diff_oppg" not in df.columns:
        df["feature_diff_oppg"] = df.get("home_oppg", pd.Series(0.0, index=df.index)).fillna(0.0) - df.get("away_oppg", pd.Series(0.0, index=df.index)).fillna(0.0)
    if "feature_diff_streak" not in df.columns:
         df["feature_diff_streak"] = df.get("home_streak", pd.Series(0.0, index=df.index)).fillna(0.0) - df.get("away_streak", pd.Series(0.0, index=df.index)).fillna(0.0)
    if "feature_diff_last5" not in df.columns:
         # Use win pct diff as proxy for last 5 diff if missing
         df["feature_diff_last5"] = df["feature_diff_win_pct"]

    # Default Kalshi
    if "kalshi_prob" not in df.columns:
         df["kalshi_prob"] = df.get("theover_probability", pd.Series(0.5, index=df.index))

    if "sentiment_diff" not in df.columns:
         df["sentiment_diff"] = df.get("sharp_vs_public", pd.Series(0.0, index=df.index))

    for target_col in VERTEX_FEATURE_COLUMNS:
        source_col = mapping.get(target_col)

        # Probs default to 0.5, counts/diffs default to 0.0
        default_val = 0.5 if "prob" in target_col else 0.0

        if source_col and source_col in df.columns:
            out[target_col] = pd.to_numeric(df[source_col], errors='coerce').fillna(default_val)
        else:
            out[target_col] = default_val

    # Specific fallbacks for implied prob if missing or exactly 0.5
    if "implied_home_prob" in df.columns:
         mask = out["implied_home_prob"] == 0.5
         out.loc[mask, "implied_home_prob"] = pd.to_numeric(df.loc[mask, "home_win_pct"], errors='coerce').fillna(0.5)

    return out[VERTEX_FEATURE_COLUMNS]

def evaluate_model(model, X_val, y_val, name="Model"):
    dval = xgb.DMatrix(X_val)
    probs = model.predict(dval)

    acc = accuracy_score(y_val, probs > 0.5)
    ll = log_loss(y_val, probs)
    try:
        auc = roc_auc_score(y_val, probs)
    except:
        auc = 0.5

    unique_probs = len(np.unique(probs))
    unique_ratio = unique_probs / len(probs)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.4f} | Log Loss: {ll:.4f} | AUC: {auc:.4f}")
    print(f"Raw Unique Count: {unique_probs} / {len(probs)}")
    print(f"Raw Unique Ratio: {unique_ratio:.1%}")

    # Analyze distribution
    from collections import Counter
    counts = Counter(probs)
    top_3 = sorted([(p, c) for p, c in counts.items()], key=lambda x: x[1], reverse=True)[:3]
    print(f"Top 3 repeated probabilities: {top_3}")

    return unique_ratio, model

def main():
    print("Loading data...")
    df = pd.read_csv('data/master_all_sports.csv')

    # Sort by commence_time if available to do time-based split
    if 'commence_time' in df.columns:
        df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce')
        df = df.sort_values('commence_time')

    # Target
    if 'home_won' not in df.columns:
        print("ERROR: 'home_won' target not found.")
        return

    y = df['home_won'].astype(int)
    X = build_features_from_master(df)

    # Split
    # Since the dataset is small (~400 rows), use a 62-row equivalent validation set
    # to mimic the size of the production run the user mentioned
    val_size = min(62, int(len(df) * 0.2))
    X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    models = {}

    # Config 1: Current/Baseline approximation (Shallow, few trees)
    params_baseline = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Baseline...")
    model_base = xgb.train(params_baseline, dtrain, num_boost_round=50)
    ratio, _ = evaluate_model(model_base, X_val, y_val, "Baseline (depth=3, n=50)")
    models['baseline'] = {'ratio': ratio, 'model': model_base}

    # Config 2: Medium Capacity
    params_med = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Medium Capacity...")
    model_med = xgb.train(params_med, dtrain, num_boost_round=150)
    ratio, _ = evaluate_model(model_med, X_val, y_val, "Medium (depth=5, n=150)")
    models['medium'] = {'ratio': ratio, 'model': model_med}

    # Config 3: High Capacity (More expressive, deeper, more trees)
    params_high = {
        'objective': 'binary:logistic',
        'max_depth': 7,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining High Capacity...")
    model_high = xgb.train(params_high, dtrain, num_boost_round=300)
    ratio, _ = evaluate_model(model_high, X_val, y_val, "High (depth=7, n=300)")
    models['high'] = {'ratio': ratio, 'model': model_high}

    # Save the best unique ratio model if it's over 80%
    best_config = max(models.items(), key=lambda x: x[1]['ratio'])
    best_name, best_data = best_config[0], best_config[1]

    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    print(f"Best model for uniqueness: {best_name.upper()} with ratio {best_data['ratio']:.1%}")
    if best_data['ratio'] >= 0.80:
        print("SUCCESS: Target of >= 80% raw unique ratio ACHIEVED via model retraining.")
        os.makedirs('models', exist_ok=True)
        save_path = 'models/xgboost_model_v2.json'
        best_data['model'].save_model(save_path)
        print(f"Saved new trained model to {save_path}")
        print("To use this model in production, update app_core/prediction_engine.py to point to models/xgboost_model_v2.json")
    else:
        print("FAILED: Target of >= 80% raw unique ratio NOT achieved even with higher capacity models.")
        print("This suggests the *feature set* itself on the validation split is lacking diversity (feature collapse),")
        print("or the target labels combined with the small dataset cause extreme compression.")

if __name__ == "__main__":
    main()
