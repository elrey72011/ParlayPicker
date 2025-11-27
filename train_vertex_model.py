#!/usr/bin/env python3
"""
Train and Deploy XGBoost Model to GCP Vertex AI for Sports Betting Predictions

Usage:
    python train_vertex_model.py --data historical_games.csv --project-id your-project-id
    
Requirements:
    pip install google-cloud-aiplatform scikit-learn xgboost pandas joblib
"""

import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb

# Feature names expected by the app (must match exactly)
FEATURE_NAMES = [
    "home_win_pct",
    "away_win_pct", 
    "home_avg_points",
    "away_avg_points",
    "home_def_rating",
    "away_def_rating",
    "spread_normalized",
    "home_last_5",
    "away_last_5",
    "home_home_record",
    "away_away_record",
    "head_to_head",
    "rest_advantage",
    "injuries_impact",
    "weather_factor",
    "public_betting_pct",
    "sharp_money_indicator",
    "line_movement",
    "total_movement",
    "model_consensus",
    "theover_probability",
    "implied_home_prob",
    "home_streak",
    "away_streak",
    "division_game",
    "back_to_back",
    "primetime_game"
]

TARGET_COLUMN = "home_won"


def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and prepare features/target"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Check for target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Available: {list(df.columns)}")
    
    # Get available features
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    
    print(f"\nAvailable features: {len(available_features)}/{len(FEATURE_NAMES)}")
    if missing_features:
        print(f"Missing features (will be set to 0): {missing_features}")
    
    # Create feature matrix
    X = pd.DataFrame()
    for feature in FEATURE_NAMES:
        if feature in df.columns:
            X[feature] = df[feature].fillna(0)
        else:
            X[feature] = 0  # Default value for missing features
    
    y = df[TARGET_COLUMN].astype(int)
    
    # Handle any remaining NaN
    X = X.fillna(0)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[xgb.XGBClassifier, Dict]:
    """Train XGBoost model and return with metrics"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize model with tuned hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=20
    )
    
    # Train with early stopping
    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"ROC-AUC:  {auc_roc:.4f}")
    print(f"\nBreakeven at -110 odds: 52.4%")
    print(f"Model edge: {(accuracy - 0.524) * 100:+.2f}%")
    
    # Cross-validation
    print("\n5-Fold Cross-Validation:")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Away  Home")
    print(f"Actual Away   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Home   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Feature importance
    print(f"\nTop 10 Feature Importances:")
    importance_df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'roc_auc': auc_roc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': len(FEATURE_NAMES),
        'trained_at': datetime.now().isoformat()
    }
    
    return model, metrics


def save_model_local(model: xgb.XGBClassifier, metrics: Dict, output_dir: str = "model_artifacts"):
    """Save model locally"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save feature names for reference
    features_path = os.path.join(output_dir, "feature_names.json")
    with open(features_path, 'w') as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"Feature names saved to: {features_path}")
    
    return model_path


def deploy_to_vertex_ai(
    model_path: str,
    project_id: str,
    location: str = "us-central1",
    model_display_name: str = "parlaydesk-predictor",
    endpoint_display_name: str = "parlaydesk-endpoint"
) -> Tuple[str, str]:
    """Deploy model to GCP Vertex AI"""
    
    try:
        from google.cloud import aiplatform
        from google.cloud import storage
    except ImportError:
        print("\nError: google-cloud-aiplatform not installed")
        print("Run: pip install google-cloud-aiplatform google-cloud-storage")
        return None, None
    
    print(f"\n{'='*50}")
    print("DEPLOYING TO GCP VERTEX AI")
    print(f"{'='*50}")
    print(f"Project: {project_id}")
    print(f"Location: {location}")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Upload model to GCS first
    bucket_name = f"{project_id}-vertex-models"
    blob_name = f"parlaydesk/{datetime.now().strftime('%Y%m%d_%H%M%S')}/model.joblib"
    
    print(f"\nUploading model to gs://{bucket_name}/{blob_name}...")
    
    storage_client = storage.Client(project=project_id)
    
    # Create bucket if doesn't exist
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception:
        print(f"Creating bucket {bucket_name}...")
        bucket = storage_client.create_bucket(bucket_name, location=location)
    
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(model_path)
    
    model_gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Model uploaded to: {model_gcs_uri}")
    
    # Register model in Vertex AI
    print(f"\nRegistering model in Vertex AI...")
    
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=os.path.dirname(model_gcs_uri),
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    )
    
    print(f"Model registered: {model.resource_name}")
    model_id = model.name
    
    # Create endpoint
    print(f"\nCreating endpoint...")
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        project=project_id,
        location=location,
    )
    
    print(f"Endpoint created: {endpoint.resource_name}")
    endpoint_id = endpoint.name
    
    # Deploy model to endpoint
    print(f"\nDeploying model to endpoint...")
    print("(This may take 5-10 minutes)")
    
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{model_display_name}-v1",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=100,
    )
    
    print(f"\n{'='*50}")
    print("DEPLOYMENT COMPLETE!")
    print(f"{'='*50}")
    print(f"\nAdd these to your Streamlit app:")
    print(f"  gcp_project_id = \"{project_id}\"")
    print(f"  vertex_endpoint_id = \"{endpoint_id}\"")
    print(f"  gcp_location = \"{location}\"")
    
    return model_id, endpoint_id


def test_prediction(model: xgb.XGBClassifier):
    """Test prediction with sample data"""
    print(f"\n{'='*50}")
    print("TESTING PREDICTION")
    print(f"{'='*50}")
    
    # Sample game features
    sample_features = [
        0.65,   # home_win_pct
        0.48,   # away_win_pct
        112.5,  # home_avg_points
        108.2,  # away_avg_points
        105.0,  # home_def_rating
        108.5,  # away_def_rating
        0.35,   # spread_normalized (-3.5 spread normalized)
        0.60,   # home_last_5
        0.40,   # away_last_5
        0.70,   # home_home_record
        0.45,   # away_away_record
        0.50,   # head_to_head
        1.0,    # rest_advantage
        0.0,    # injuries_impact
        0.0,    # weather_factor
        0.55,   # public_betting_pct
        0.0,    # sharp_money_indicator
        0.0,    # line_movement
        0.0,    # total_movement
        0.50,   # model_consensus
        0.65,   # theover_probability
        0.60,   # implied_home_prob
        2.0,    # home_streak
        -1.0,   # away_streak
        0.0,    # division_game
        0.0,    # back_to_back
        0.0,    # primetime_game
    ]
    
    # Make prediction
    prob = model.predict_proba([sample_features])[0][1]
    
    print(f"\nSample Game:")
    print(f"  Home Win %: 65%, Away Win %: 48%")
    print(f"  Spread: -3.5 (home favored)")
    print(f"  Home Last 5: 60%, Away Last 5: 40%")
    print(f"\nModel Prediction:")
    print(f"  Home Win Probability: {prob:.1%}")
    print(f"  Confidence: {'High' if abs(prob - 0.5) > 0.15 else 'Medium' if abs(prob - 0.5) > 0.08 else 'Low'}")


def create_sample_training_data(output_path: str = "sample_training_data.csv", n_samples: int = 1000):
    """Create synthetic training data for testing"""
    print(f"Creating {n_samples} synthetic training samples...")
    
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        # Generate realistic feature values
        home_win_pct = np.random.beta(5, 5)  # Centered around 0.5
        away_win_pct = np.random.beta(5, 5)
        
        # Spread correlates with win percentages
        spread_diff = (home_win_pct - away_win_pct) * 20  # -10 to +10 range
        spread_normalized = (spread_diff + 15) / 30  # Normalize to 0-1
        
        home_avg_points = np.random.normal(105, 10)
        away_avg_points = np.random.normal(105, 10)
        
        # Generate other features
        row = {
            'home_win_pct': home_win_pct,
            'away_win_pct': away_win_pct,
            'home_avg_points': home_avg_points,
            'away_avg_points': away_avg_points,
            'home_def_rating': np.random.normal(105, 5),
            'away_def_rating': np.random.normal(105, 5),
            'spread_normalized': spread_normalized,
            'home_last_5': np.random.beta(5, 5),
            'away_last_5': np.random.beta(5, 5),
            'home_home_record': home_win_pct + np.random.normal(0.05, 0.05),
            'away_away_record': away_win_pct - np.random.normal(0.05, 0.05),
            'head_to_head': np.random.beta(5, 5),
            'rest_advantage': np.random.choice([-1, 0, 1, 2], p=[0.2, 0.5, 0.2, 0.1]),
            'injuries_impact': np.random.beta(1, 5),  # Usually low
            'weather_factor': 0,  # Indoor sports
            'public_betting_pct': np.random.beta(5, 5),
            'sharp_money_indicator': np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]),
            'line_movement': np.random.normal(0, 0.5),
            'total_movement': np.random.normal(0, 2),
            'model_consensus': np.random.beta(5, 5),
            'theover_probability': np.clip(home_win_pct + np.random.normal(0, 0.1), 0.2, 0.8),
            'implied_home_prob': np.clip(home_win_pct + np.random.normal(0, 0.05), 0.2, 0.8),
            'home_streak': np.random.randint(-5, 6),
            'away_streak': np.random.randint(-5, 6),
            'division_game': np.random.choice([0, 1], p=[0.7, 0.3]),
            'back_to_back': np.random.choice([0, 1], p=[0.85, 0.15]),
            'primetime_game': np.random.choice([0, 1], p=[0.8, 0.2]),
        }
        
        # Generate outcome based on features (with some randomness)
        # Home teams win ~54% on average
        win_prob = 0.5 + (home_win_pct - away_win_pct) * 0.5 + 0.04  # Home advantage
        win_prob += (row['home_last_5'] - row['away_last_5']) * 0.2
        win_prob += row['rest_advantage'] * 0.02
        win_prob = np.clip(win_prob, 0.2, 0.8)
        
        row['home_won'] = 1 if np.random.random() < win_prob else 0
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to: {output_path}")
    print(f"Home win rate: {df['home_won'].mean():.1%}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train and deploy sports betting ML model")
    parser.add_argument("--data", type=str, help="Path to training data CSV")
    parser.add_argument("--project-id", type=str, help="GCP Project ID for deployment")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP location")
    parser.add_argument("--model-name", type=str, default="parlaydesk-predictor", help="Model display name")
    parser.add_argument("--create-sample-data", action="store_true", help="Create synthetic training data")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Vertex AI after training")
    parser.add_argument("--output-dir", type=str, default="model_artifacts", help="Output directory for model")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        data_path = create_sample_training_data()
        if not args.data:
            args.data = data_path
    
    if not args.data:
        print("Error: --data required (or use --create-sample-data)")
        print("\nUsage examples:")
        print("  python train_vertex_model.py --data historical_games.csv")
        print("  python train_vertex_model.py --create-sample-data")
        print("  python train_vertex_model.py --data games.csv --project-id my-project --deploy")
        return
    
    # Load and prepare data
    X, y = load_and_prepare_data(args.data)
    
    # Train model
    model, metrics = train_model(X, y)
    
    # Test prediction
    test_prediction(model)
    
    # Save locally
    model_path = save_model_local(model, metrics, args.output_dir)
    
    # Deploy to Vertex AI if requested
    if args.deploy:
        if not args.project_id:
            print("\nError: --project-id required for deployment")
            return
        
        model_id, endpoint_id = deploy_to_vertex_ai(
            model_path=model_path,
            project_id=args.project_id,
            location=args.location,
            model_display_name=args.model_name,
        )
        
        if endpoint_id:
            print(f"\n✅ Deployment successful!")
            print(f"Configure your app with:")
            print(f"  gcp_project_id = \"{args.project_id}\"")
            print(f"  vertex_endpoint_id = \"{endpoint_id}\"")
    else:
        print(f"\n✅ Training complete!")
        print(f"To deploy to GCP Vertex AI, run:")
        print(f"  python train_vertex_model.py --data {args.data} --project-id YOUR_PROJECT_ID --deploy")


if __name__ == "__main__":
    main()
