from __future__ import annotations

from parlaypicker.models.model_loader import load_model


def predict_proba(model_path: str, X):
    model = load_model(model_path)
    return model.predict_proba(X)[:, 1]
