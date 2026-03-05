"""High-performance feature engineering with Polars lazy execution."""
from __future__ import annotations

from pathlib import Path

import polars as pl


DATA_PARQUET_DIR = Path("data/parquet")


def _implied_probability_expr(col: str) -> pl.Expr:
    return (
        pl.when(pl.col(col) > 0)
        .then(100 / (pl.col(col) + 100))
        .otherwise(pl.col(col).abs() / (pl.col(col).abs() + 100))
    )


def engineer_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_columns(
        _implied_probability_expr("home_odds").alias("home_implied_prob"),
        _implied_probability_expr("away_odds").alias("away_implied_prob"),
    )
    return out.with_columns(
        pl.when((pl.col("home_implied_prob") + pl.col("away_implied_prob")) > 0)
        .then(pl.col("home_implied_prob") / (pl.col("home_implied_prob") + pl.col("away_implied_prob")))
        .otherwise(pl.col("home_implied_prob"))
        .alias("market_prob"),
        (pl.col("points_scored") - pl.col("points_allowed")).alias("point_diff")
        if {"points_scored", "points_allowed"}.issubset(set(out.columns))
        else pl.lit(None).alias("point_diff"),
    )


def lazy_team_scoring_by_season(parquet_path: str | Path, season: int) -> pl.DataFrame:
    """Example lazy query with pushdown optimization."""
    return (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("season") == season)
        .group_by("team")
        .agg(pl.col("points_scored").mean().alias("avg_points_scored"))
        .collect()
    )


def csv_to_parquet(csv_path: str | Path, parquet_name: str) -> Path:
    """Convert raw CSV input to parquet storage."""
    DATA_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PARQUET_DIR / parquet_name
    pl.read_csv(csv_path).write_parquet(out_path)
    return out_path
