"""Schema validation helpers for resilient dataframe pipelines."""
from __future__ import annotations

import logging
from typing import Any

from core.schema.schema_registry import SCHEMAS

logger = logging.getLogger(__name__)

STRICT_SCHEMA = False


def _default_for_column(column_name: str) -> Any:
    lowered = column_name.lower()
    if "probability" in lowered:
        return 0.5
    if "expected_value" in lowered:
        return 0.0
    if lowered in {"odds", "spread", "total", "point_diff", "off_rating", "def_rating", "market_probability"}:
        return 0.0
    return None


def validate_schema(df, schema_name: str):
    """Ensure a dataframe contains required columns for a pipeline stage."""
    required_columns = SCHEMAS.get(schema_name)

    if not required_columns:
        logger.warning("No schema registered for '%s'; skipping validation", schema_name)
        return df

    computed_columns = {"market_probability", "expected_value", "consensus_prob"}
    missing_columns: list[str] = []
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
            logger.warning("Missing column '%s' for schema '%s'", col, schema_name)
            if col in computed_columns:
                logger.info(
                    "Skipping placeholder for computed column '%s'; it must be calculated upstream",
                    col,
                )
                continue
            df[col] = _default_for_column(col)

    if STRICT_SCHEMA and missing_columns:
        raise ValueError(
            f"Schema validation failed for '{schema_name}'. Missing columns: {missing_columns}"
        )

    logger.info("Validated schema: %s", schema_name)
    return df
