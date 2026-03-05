"""Feature engineering with vectorized operations."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd


def _chunk_transform(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_implied_prob"] = np.where(
        out["home_odds"] > 0,
        100 / (out["home_odds"] + 100),
        np.abs(out["home_odds"]) / (np.abs(out["home_odds"]) + 100),
    )
    out["away_implied_prob"] = np.where(
        out["away_odds"] > 0,
        100 / (out["away_odds"] + 100),
        np.abs(out["away_odds"]) / (np.abs(out["away_odds"]) + 100),
    )
    total = out["home_implied_prob"] + out["away_implied_prob"]
    out["market_prob"] = np.where(total > 0, out["home_implied_prob"] / total, out["home_implied_prob"])
    return out


def engineer_features(df: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    if workers <= 1 or len(df) < 1000:
        return _chunk_transform(df)
    chunks = np.array_split(df, workers)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_chunk_transform, chunks))
    return pd.concat(parts, ignore_index=True)
