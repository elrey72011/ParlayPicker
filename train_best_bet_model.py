"""
Train Vertex AI model to predict:
1. Win probability
2. Spread cover probability  
3. Over/Under probability
"""
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from google.cloud import storage
import pickle
import json
from datetime import datetime

print("🤖 Training Best Bet Finder Model\n")
print(f"NumPy: {np.__version__}")
print(f"sklearn: {sklearn.__version__}")

# Verify compatible versions
if np.__version__.startswith('2'):
    print("❌ ERROR: numpy 2.x detected! Downgrade first:")
    print("   pip install 'numpy<2.0' --force-reinstall")
    exit(1)

print("✅ Compatible versions!\n")

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/Robert/Downloads/elite-hangar-479017-m8-401b051a3b72.json'

# Load training data
data_file = "training_data_nba_20251122.csv"  # Update filename
print(f"Loading: {data_file}")

df = pd.read_csv(data_file)
print(f"✅ Loaded {len(df)} games\n")

# Define features
feature_columns = [
    'home_win_pct',
    'away_win_pct',
    'home_avg_points',
    'away_avg_points',
    'home_def_rating',
    'away_def_rating',
    'spread_abs',
    'total_line',
    'home_away_diff',
]

# BUILD 3 MODELS

print("="*60)
print("MODEL 1: Win Probability Predictor")
print("="*60)

X = df[feature_columns].values
y_win = df['home_win'].values

X_train, X_test, y_train, y_test = train_test_split(X, y_win, test_size=0.2, random_state=42)

model_win = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model_win.fit(X_train, y_train)
y_pred = model_win.predict(X_test)
y_proba = model_win.predict_proba(X_test)[:, 1]

acc_win = accuracy_score(y_test, y_pred)
auc_win = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc_win:.2%}")
print(f"AUC: {auc_win:.3f}")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))

print("\n" + "="*60)
print("MODEL 2: Spread Cover Predictor")
print("="*60)

y_cover = df['cover'].values
X_train, X_test, y_train, y_test = train_test_split(X, y_cover, test_size=0.2, random_state=42)

model_cover = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model_cover.fit(X_train, y_train)
y_pred = model_cover.predict(X_test)
y_proba = model_cover.predict_proba(X_test)[:, 1]

acc_cover = accuracy_score(y_test, y_pred)
auc_cover = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc_cover:.2%}")
print(f"AUC: {auc_cover:.3f}")
print(classification_report(y_test, y_pred, target_names=['No Cover', 'Cover']))

print("\n" + "="*60)
print("MODEL 3: Over/Under Predictor")
print("="*60)

y_over = df['over'].values
X_train, X_test, y_train, y_test = train_test_split(X, y_over, test_size=0.2, random_state=42)

model_over = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model_over.fit(X_train, y_train)
y_pred = model_over.predict(X_test)
y_proba = model_over.predict_proba(X_test)[:, 1]

acc_over = accuracy_score(y_test, y_pred)
auc_over = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc_over:.2%}")
print(f"AUC: {auc_over:.3f}")
print(classification_report(y_test, y_pred, target_names=['Under', 'Over']))

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Win Model)")
print("="*60)

importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_win.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df)

# Save models
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

models = {
    'win_model': model_win,
    'cover_model': model_cover,
    'over_model': model_over,
    'features': feature_columns
}

model_file = f'best_bet_model_{timestamp}.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(models, f)

print(f"\n✅ Saved models to: {model_file}")

# Metadata
metadata = {
    'created_date': datetime.now().isoformat(),
    'sklearn_version': sklearn.__version__,
    'numpy_version': np.__version__,
    'features': feature_columns,
    'models': {
        'win': {'accuracy': float(acc_win), 'auc': float(auc_win)},
        'cover': {'accuracy': float(acc_cover), 'auc': float(auc_cover)},
        'over': {'accuracy': float(acc_over), 'auc': float(auc_over)}
    },
    'training_samples': len(df),
    'feature_importance': importance_df.to_dict('records')
}

metadata_file = f'metadata_best_bet_{timestamp}.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

# Upload to GCS
bucket_name = "parlaydesk-ml-models"
model_path = "models/best-bet-finder-v1"

print(f"\n☁️ Uploading to Google Cloud Storage...")

client = storage.Client()
bucket = client.bucket(bucket_name)

blob = bucket.blob(f"{model_path}/model.pkl")
blob.upload_from_filename(model_file)
print(f"✅ Uploaded {model_file}")

blob = bucket.blob(f"{model_path}/metadata.json")
blob.upload_from_string(json.dumps(metadata, indent=2))
print(f"✅ Uploaded metadata")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"Model: gs://{bucket_name}/{model_path}")
print(f"Win Model: {acc_win:.2%} accuracy, {auc_win:.3f} AUC")
print(f"Cover Model: {acc_cover:.2%} accuracy, {auc_cover:.3f} AUC")
print(f"Over Model: {acc_over:.2%} accuracy, {auc_over:.3f} AUC")
print("\nNext: Register with Vertex AI")
print("Run: python register_best_bet_model.py")
