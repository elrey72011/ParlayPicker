from __future__ import annotations

import joblib
import polars as pl
from sklearn.linear_model import LogisticRegression


def load_training_data(parquet_path: str, target_col: str):
    df = pl.read_parquet(parquet_path)
    y = df[target_col].to_numpy()
    X = df.drop(target_col).to_pandas()
    return X, y


def train_model(X, y, output_path: str):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, output_path)
    return model
