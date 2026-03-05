"""Compatibility wrapper that keeps pandas API while using Polars engine."""
from __future__ import annotations

import pandas as pd
import polars as pl

from parlaypicker.data_pipeline.feature_engineering_polars import engineer_features_polars


def engineer_features(df: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    # workers kept for backward compatibility
    _ = workers
    pl_df = pl.from_pandas(df)
    return engineer_features_polars(pl_df).to_pandas()
