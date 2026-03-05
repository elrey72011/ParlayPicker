"""General utility helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def standardize_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "game_id",
        "bet_type",
        "odds",
        "true_probability",
        "expected_value",
        "kelly_fraction",
        "bet_signal",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = None
    return df[required + [c for c in df.columns if c not in required]]


def export_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    standardize_export_columns(df).to_csv(path, index=False)
    return path
