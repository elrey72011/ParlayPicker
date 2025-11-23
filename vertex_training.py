"""
Vertex AI Training Pipeline for Sports Betting Models
Trains and deploys ML models for spread, moneyline, and totals predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
import json

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# Google Cloud
from google.cloud import aiplatform
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VertexAITrainingPipeline:
    """Training pipeline for sports betting models on Vertex AI"""
    
    def __init__(
        self,
        project_id: str,
        location: str,
        staging_bucket: str,
        sport: str
    ):
        """
        Initialize training pipeline
        
        Args:
            project_id: GCP project ID
            location: GCP region (e.g., 'us-central1')
            staging_bucket: GCS bucket for artifacts
            sport: Sport name
        """
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket
        self.sport = sport
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location,
            staging_bucket=staging_bucket
        )
        
        self.models = {}
        self.scalers = {}
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        drop_cols: List[str] = None,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training
        
        Args:
            df: Feature DataFrame
            target_col: Target variable column name
            drop_cols: Columns to drop
            test_size: Test set proportion
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preparing data for {target_col}")
        
        # Drop unnecessary columns
        drop_cols = drop_cols or [
            'DateTime', 'GameID', 'HomeTeam', 'AwayTeam', 'Matchup',
            'HomeScore', 'AwayScore'  # Don't leak actual results
        ]
        
        # Drop target columns we're not using
        target_cols = [
            'Target_HomeML', 'Target_AwayML', 'Target_HomeSpread', 
            'Target_AwaySpread', 'Target_Over', 'Target_Under',
            'Target_Margin', 'Target_Total'
        ]
        drop_cols.extend([col for col in target_cols if col != target_col])
        
        # Filter valid columns
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        # Prepare features and target
        X = df.drop(columns=drop_cols + [target_col])
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        
        # Time-based split (important for sports betting)
        if 'DateTime' in df.columns:
            df_sorted = df.sort_values('DateTime')
            split_idx = int(len(df_sorted) * (1 - test_size))
            train_idx = df_sorted.index[:split_idx]
            test_idx = df_sorted.index[split_idx:]
            
            X_train = X.loc[train_idx]
            X_test = X.loc[test_idx]
            y_train = y.loc[train_idx]
            y_test = y.loc[test_idx]
        else:
            # Random split if no date column
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        logger.info(f"Features: {len(X.columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost_classifier(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Train XGBoost classification model
        
        Args:
            X_train, X_test, y_train, y_test: Train/test splits
            model_name: Name for the model
            params: XGBoost parameters
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training XGBoost classifier: {model_name}")
        
        # Default parameters optimized for betting
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'early_stopping_rounds': 20
        }
        
        if params:
            default_params.update(params)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = XGBClassifier(**{k: v for k, v in default_params.items() 
                                if k != 'early_stopping_rounds'})
        
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate expected value (simplified)
        # Assumes -110 odds, can be refined with actual odds
        correct_bets = (y_pred == y_test).sum()
        total_bets = len(y_test)
        roi = ((correct_bets * 0.909) - (total_bets - correct_bets)) / total_bets * 100
        
        metrics = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'auc': auc,
            'roi_estimate': roi
        }
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Log Loss: {logloss:.4f}")
        logger.info(f"AUC: {auc:.4f}")
        logger.info(f"Estimated ROI: {roi:.2f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 features:\n{feature_importance.head(10)}")
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def train_xgboost_regressor(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Train XGBoost regression model (for margins/totals)
        
        Args:
            X_train, X_test, y_train, y_test: Train/test splits
            model_name: Name for the model
            params: XGBoost parameters
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training XGBoost regressor: {model_name}")
        
        default_params = {
            'objective': 'reg:squarederror',
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
        }
        
        if params:
            default_params.update(params)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = XGBRegressor(**default_params)
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        
        metrics = {
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"Model: {model_name}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics
        }
    
    def cross_validate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'classifier',
        n_splits: int = 5
    ) -> Dict:
        """
        Perform time-series cross-validation
        
        Args:
            X: Features
            y: Target
            model_type: 'classifier' or 'regressor'
            n_splits: Number of CV folds
            
        Returns:
            CV scores
        """
        logger.info(f"Running {n_splits}-fold time-series cross-validation")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        if model_type == 'classifier':
            model = XGBClassifier(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=100,
                random_state=42
            )
            scoring = 'roc_auc'
        else:
            model = XGBRegressor(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=100,
                random_state=42
            )
            scoring = 'neg_root_mean_squared_error'
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring=scoring)
        
        logger.info(f"CV Scores: {scores}")
        logger.info(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """
        Train models for all bet types
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Dictionary with all trained models
        """
        logger.info(f"Training all models for {self.sport}")
        
        results = {}
        
        # Moneyline models
        if 'Target_HomeML' in df.columns:
            logger.info("\n=== Training Moneyline Model ===")
            X_train, X_test, y_train, y_test = self.prepare_data(df, 'Target_HomeML')
            results['moneyline'] = self.train_xgboost_classifier(
                X_train, X_test, y_train, y_test, f'{self.sport}_moneyline'
            )
        
        # Spread models
        if 'Target_HomeSpread' in df.columns:
            logger.info("\n=== Training Spread Model ===")
            X_train, X_test, y_train, y_test = self.prepare_data(df, 'Target_HomeSpread')
            results['spread'] = self.train_xgboost_classifier(
                X_train, X_test, y_train, y_test, f'{self.sport}_spread'
            )
        
        # Over/Under models
        if 'Target_Over' in df.columns:
            logger.info("\n=== Training Over/Under Model ===")
            X_train, X_test, y_train, y_test = self.prepare_data(df, 'Target_Over')
            results['totals'] = self.train_xgboost_classifier(
                X_train, X_test, y_train, y_test, f'{self.sport}_totals'
            )
        
        # Margin prediction (regression)
        if 'Target_Margin' in df.columns:
            logger.info("\n=== Training Margin Regression Model ===")
            X_train, X_test, y_train, y_test = self.prepare_data(df, 'Target_Margin')
            results['margin'] = self.train_xgboost_regressor(
                X_train, X_test, y_train, y_test, f'{self.sport}_margin'
            )
        
        # Total points prediction (regression)
        if 'Target_Total' in df.columns:
            logger.info("\n=== Training Total Points Regression Model ===")
            X_train, X_test, y_train, y_test = self.prepare_data(df, 'Target_Total')
            results['total_points'] = self.train_xgboost_regressor(
                X_train, X_test, y_train, y_test, f'{self.sport}_total_points'
            )
        
        return results
    
    def save_models_locally(self, output_dir: str = './models'):
        """
        Save models and scalers to local directory
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            # Save model
            model_file = output_path / f'{model_name}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler
            scaler_file = output_path / f'{model_name}_scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers[model_name], f)
            
            logger.info(f"Saved {model_name} to {output_path}")
    
    def upload_to_gcs(self, local_dir: str, gcs_path: str):
        """
        Upload models to Google Cloud Storage
        
        Args:
            local_dir: Local directory with models
            gcs_path: GCS destination path
        """
        logger.info(f"Uploading models to gs://{self.staging_bucket}/{gcs_path}")
        
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.staging_bucket)
        
        local_path = Path(local_dir)
        for file_path in local_path.glob('**/*'):
            if file_path.is_file():
                blob_path = f"{gcs_path}/{file_path.name}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded {file_path.name}")


def main():
    """Example usage"""
    
    # Configuration
    PROJECT_ID = 'your-gcp-project-id'
    LOCATION = 'us-central1'
    STAGING_BUCKET = 'your-bucket-name'
    
    # Load processed data with features
    df = pd.read_csv('./training_data/nfl_ml_ready.csv')
    
    # Initialize pipeline
    pipeline = VertexAITrainingPipeline(
        project_id=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
        sport='NFL'
    )
    
    # Train all models
    results = pipeline.train_all_models(df)
    
    # Save models
    pipeline.save_models_locally('./models/nfl')
    
    # Upload to GCS
    pipeline.upload_to_gcs('./models/nfl', 'models/nfl')
    
    print("\n=== Training Complete ===")
    for bet_type, result in results.items():
        print(f"\n{bet_type.upper()}:")
        print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
