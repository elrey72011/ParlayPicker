from __future__ import annotations

from functools import lru_cache
import joblib


@lru_cache(maxsize=32)
def load_model(model_path: str):
    return joblib.load(model_path)
