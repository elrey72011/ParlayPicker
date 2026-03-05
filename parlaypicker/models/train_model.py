from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression


def train_model(X, y, output_path: str):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, output_path)
    return model
