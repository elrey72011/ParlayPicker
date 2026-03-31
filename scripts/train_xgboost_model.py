import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from collections import Counter
import datetime
import traceback

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

        # Probs default to 0.5, counts/stats/diffs default to 0.0, just like production `app_core/prediction_engine.py`
        default_val = 0.5 if "prob" in target_col else 0.0

        if source_col and source_col in df.columns:
            out[target_col] = pd.to_numeric(df[source_col], errors='coerce').fillna(default_val)
        else:
            out[target_col] = default_val

    # Specific fallbacks for implied prob if missing or exactly 0.5
    if "implied_home_prob" in df.columns:
         mask = out["implied_home_prob"] == 0.5
         out.loc[mask, "implied_home_prob"] = pd.to_numeric(df.loc[mask, "home_win_pct"], errors='coerce').fillna(0.5)

    # Ensure all data is float to prevent XGBoost DMatrix issues
    out = out.astype(float)
    return out[VERTEX_FEATURE_COLUMNS]

def evaluate_model(model, X_val, y_val, name="Model"):
    """Evaluates a model with first-class focus on Uniqueness Metrics."""
    dval = xgb.DMatrix(X_val)
    probs = model.predict(dval)

    # 1. Predictive Metrics
    acc = accuracy_score(y_val, probs > 0.5)
    ll = log_loss(y_val, probs)
    try:
        auc = roc_auc_score(y_val, probs)
    except:
        auc = 0.5

    # 2. Uniqueness Metrics
    unique_probs = len(np.unique(probs))
    unique_ratio = unique_probs / len(probs)

    counts = Counter(probs)
    top_3 = sorted([(p, c) for p, c in counts.items()], key=lambda x: x[1], reverse=True)[:3]

    # Spread/Distribution summary
    percentiles = np.percentile(probs, [0, 25, 50, 75, 100])
    spread_range = percentiles[4] - percentiles[0]

    print(f"\n" + "="*50)
    print(f"--- {name} Results ---")
    print(f"Validation Row Count: {len(probs)}")
    print(f"Accuracy: {acc:.4f} | Log Loss: {ll:.4f} | AUC: {auc:.4f}")
    print(f"Raw Unique Count: {unique_probs} / {len(probs)}")
    print(f"Raw Unique Ratio: {unique_ratio:.1%}")
    print(f"Probability Spread (Max - Min): {spread_range:.4f}")
    print(f"Distribution [Min, 25th, Median, 75th, Max]:")
    print(f"  [{percentiles[0]:.4f}, {percentiles[1]:.4f}, {percentiles[2]:.4f}, {percentiles[3]:.4f}, {percentiles[4]:.4f}]")
    print(f"Top 3 repeated probabilities and counts: {top_3}")
    print("="*50)

    return unique_ratio, model, {"acc": acc, "ll": ll, "auc": auc, "unique_count": unique_probs}

def main():
    print("Loading historical training data...")
    try:
        df = pd.read_csv('data/master_all_sports.csv')
    except Exception as e:
        print(f"ERROR: Could not load data/master_all_sports.csv: {e}")
        return

    # Sort by commence_time if available to do time-based split
    if 'commence_time' in df.columns:
        df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce')
        df = df.sort_values('commence_time')

    # Target
    if 'home_won' not in df.columns:
        print("ERROR: 'home_won' target not found in historical data.")
        return

    y = df['home_won'].astype(int)
    X = build_features_from_master(df)

    # Split
    # We want a validation set that roughly mimics a 62-row production slate.
    val_size = min(62, int(len(df) * 0.2))

    # Chronological Split (predicting the future based on the past)
    X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

    print(f"Schema exactly matches production 21-feature schema: {list(X.columns) == VERTEX_FEATURE_COLUMNS}")
    print(f"Train set: {X_train.shape}, Validation set (Slate equivalent): {X_val.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)

    models = {}

    # Config 1: Baseline approximation (Shallow, few trees - similar to what might be causing current coarseness)
    params_baseline = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Candidate 1: Baseline (Low Capacity)...")
    model_base = xgb.train(params_baseline, dtrain, num_boost_round=50)
    ratio, _, metrics = evaluate_model(model_base, X_val, y_val, "Candidate 1 (depth=3, n=50)")
    models['candidate_1_baseline'] = {'ratio': ratio, 'model': model_base, 'metrics': metrics}

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
    print("\nTraining Candidate 2: Medium Capacity...")
    model_med = xgb.train(params_med, dtrain, num_boost_round=150)
    ratio, _, metrics = evaluate_model(model_med, X_val, y_val, "Candidate 2 (depth=5, n=150)")
    models['candidate_2_medium'] = {'ratio': ratio, 'model': model_med, 'metrics': metrics}

    # Config 3: High Capacity (More expressive, deeper, more trees)
    params_high = {
        'objective': 'binary:logistic',
        'max_depth': 7,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Candidate 3: High Capacity...")
    model_high = xgb.train(params_high, dtrain, num_boost_round=300)
    ratio, _, metrics = evaluate_model(model_high, X_val, y_val, "Candidate 3 (depth=7, n=300, lr=0.02)")
    models['candidate_3_high'] = {'ratio': ratio, 'model': model_high, 'metrics': metrics}

    # Config 4: Ultra Capacity with Regularization (Deepest, most trees, but reg_alpha/lambda)
    params_ultra = {
        'objective': 'binary:logistic',
        'max_depth': 9,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Candidate 4: Ultra Capacity w/ L1/L2 Regularization...")
    model_ultra = xgb.train(params_ultra, dtrain, num_boost_round=500)
    ratio, _, metrics = evaluate_model(model_ultra, X_val, y_val, "Candidate 4 (depth=9, n=500, reg)")
    models['candidate_4_ultra'] = {'ratio': ratio, 'model': model_ultra, 'metrics': metrics}

    # Save the candidates separately
    os.makedirs('models', exist_ok=True)
    best_ratio = -1
    best_candidate_name = None

    print("\n" + "="*50)
    print("RETRAINING HARNESS SUMMARY")
    print("="*50)
    for name, data in models.items():
        save_path = f'models/{name}.json'
        data['model'].save_model(save_path)
        print(f"- Saved {name} to {save_path} (Unique Ratio: {data['ratio']:.1%})")

        if data['ratio'] > best_ratio:
            best_ratio = data['ratio']
            best_candidate_name = name

    print(f"\nBest model for uniqueness: {best_candidate_name.upper()} with ratio {best_ratio:.1%}")
    if best_ratio >= 0.80:
        print("\nSUCCESS: Target of >= 80% raw unique ratio ACHIEVED via model retraining.")
        print(f"Candidate '{best_candidate_name}' appears ready to resolve the model coarseness.")
        print(f"Command to promote: cp models/{best_candidate_name}.json models/xgboost_model_v2.json")
    else:
        print("\nFAILED: Target of >= 80% raw unique ratio NOT achieved even with Ultra capacity models.")
        print("DIAGNOSIS: Model family capacity is NOT the sole blocker. The issue is likely Feature Collapse/Dataset Size:")
        print("1. The training dataset is too small (~400 rows) or lacks sufficient variance to teach the model to separate 62 distinct buckets.")
        print("2. The feature schema itself (win%, ppg, etc) on the validation split is naturally clumped/identical across games without enough high-granularity separators.")

if __name__ == "__main__":
    main()
