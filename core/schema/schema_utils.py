"""Utilities for schema normalization."""
from __future__ import annotations

import re


def normalize_columns(df):
    """Normalize dataframe column names to snake_case lowercase keys."""
    normalized = []
    for col in df.columns:
        value = str(col).strip().lower()
        value = value.replace("%", "pct")
        value = re.sub(r"[^a-z0-9]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if value in {"league_name", "leagueid", "league_id"}:
            value = "league"
        normalized.append(value)

    df.columns = normalized
    return df
