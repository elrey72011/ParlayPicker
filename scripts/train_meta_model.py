import pandas as pd
import numpy as np
import xgboost as xgb
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from app_core.meta_model import extract_meta_features, META_FEATURES

def build_training_dataset(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the dataset for training the meta-model.
    Requires historical_df with outcomes attached (spread_result, total_result).
    """
    rows = []
    for idx, row in historical_df.iterrows():
        # Determine outcome
        market = str(row.get('Market') or row.get('best_pick_type') or '').upper()
        outcome = None
        if 'SPREAD' in market:
            res = row.get('spread_result')
            if res == 'WIN': outcome = 1
            elif res == 'LOSS': outcome = 0
        elif 'TOTAL' in market:
            res = row.get('total_result')
            if res == 'WIN': outcome = 1
            elif res == 'LOSS': outcome = 0

        # Ignore PUSH or N/A
        if outcome is not None:
            features = extract_meta_features(row.to_dict())
            features['target'] = outcome
            rows.append(features)

    return pd.DataFrame(rows)

def train_model(df: pd.DataFrame, save_path: str = 'models/meta_model.json'):
    if df.empty:
        print("Empty training dataset.")
        return

    X = df[META_FEATURES]
    y = df['target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    print("Training meta-model...")
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=False
    )

    # Metrics
    preds = model.predict(dtest)
    ll = log_loss(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    auc = roc_auc_score(y_test, preds)

    print(f"Test Log Loss: {ll:.4f}")
    print(f"Test Brier Score: {brier:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    print(f"Meta-model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to historical enriched CSV')
    parser.add_argument('--output', type=str, default='models/meta_model.json')
    args = parser.parse_args()

    if os.path.exists(args.input):
        df = pd.read_csv(args.input)
        train_df = build_training_dataset(df)
        train_model(train_df, args.output)
    else:
        print(f"File {args.input} not found.")
