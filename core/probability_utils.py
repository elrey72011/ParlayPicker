"""Utilities for detecting and normalizing probability columns."""

from __future__ import annotations

import pandas as pd


def detect_probability_column(df: pd.DataFrame) -> str | None:
    """Return the first known probability column present on a DataFrame."""
    possible_cols = [
        "ai_probability",
        "ai_prob",
        "model_probability",
        "vertex_probability",
        "probability",
        "ml_probability",
        "model_prob",
        "vertex_prob",
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    return None


def ensure_probability_column(
    df: pd.DataFrame,
    column_name: str = "ai_probability",
) -> pd.DataFrame:
    """Ensure `column_name` exists, copying from known aliases or defaulting to 0.5."""
    if column_name not in df.columns:
        prob_col = detect_probability_column(df)

        if prob_col:
            df[column_name] = df[prob_col]
        else:
            df[column_name] = 0.5

    return df

