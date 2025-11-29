from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "sports_model_latest.joblib"

_model_artifact = None

def get_model():
    """
    Lazy-load the model artifact from disk.
    Returns (model, feature_names).
    """
    global _model_artifact
    if _model_artifact is None:
        _model_artifact = joblib.load(MODEL_PATH)
    return _model_artifact["model"], _model_artifact["feature_names"]

def score_games(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    features_df must contain all feature columns used in training.
    Returns a DataFrame copy with:
      - model_home_prob: P(home team wins) from the model
      - model_edge: model_home_prob − implied_home_prob (if that column exists)
    """
    model, feature_names = get_model()

    X = features_df[feature_names].values
    probs = model.predict_proba(X)[:, 1]  # probability home team wins

    out = features_df.copy()
    out["model_home_prob"] = probs

    if "implied_home_prob" in out.columns:
        out["model_edge"] = out["model_home_prob"] - out["implied_home_prob"]

    return out
