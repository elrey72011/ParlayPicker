"""Compatibility wrapper that keeps pandas API while using Polars engine."""
from __future__ import annotations

import pandas as pd
import polars as pl

from core.schema.schema_utils import normalize_columns
from core.schema.schema_validator import validate_schema
from parlaypicker.data_pipeline.feature_engineering_polars import engineer_features_polars


def engineer_features(df: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    # workers kept for backward compatibility
    _ = workers
    normalized = normalize_columns(df.copy())
    pl_df = pl.from_pandas(normalized)
    engineered = engineer_features_polars(pl_df).to_pandas()
    engineered = normalize_columns(engineered)
    return validate_schema(engineered, "feature_engineering")
