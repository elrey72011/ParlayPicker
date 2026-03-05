from __future__ import annotations

import pandas as pd

from core.schema.schema_validator import validate_schema
from parlaypicker.models.model_loader import load_model


def predict_proba(model_path: str, X):
    model = load_model(model_path)
    return model.predict_proba(X)[:, 1]


def predict_proba_dataframe(model_path: str, X: pd.DataFrame) -> pd.DataFrame:
    """Predict probabilities and validate the model prediction schema."""
    model = load_model(model_path)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({"ml_probability": probabilities})

    if "game_id" in X.columns:
        predictions["game_id"] = X["game_id"].values
    if "ai_probability" in X.columns:
        predictions["ai_probability"] = X["ai_probability"].values

    return validate_schema(predictions, "model_predictions")
