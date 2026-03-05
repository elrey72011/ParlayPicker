"""High-performance end-to-end sports data pipeline using Polars + DuckDB."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from parlaypicker.data_pipeline.odds_collector import fetch_odds
from parlaypicker.data_pipeline.feature_engineering_polars import engineer_features_polars
from parlaypicker.core.duckdb_engine import engine


DATA_PARQUET_DIR = Path("data/parquet")


def _raw_odds_to_parquet(df: pl.DataFrame, sport: str, date_key: str) -> Path:
    DATA_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_PARQUET_DIR / f"odds_{sport.lower()}_{date_key}.parquet"
    df.write_parquet(path)
    latest = DATA_PARQUET_DIR / "latest_odds.parquet"
    df.write_parquet(latest)
    return path


def run_pipeline(sport: str, date_key: str) -> pl.DataFrame:
    payload = fetch_odds(sport, date_key)
    games = payload.get("games", [])
    if not games:
        return pl.DataFrame()

    df = pl.DataFrame(games)
    _raw_odds_to_parquet(df, sport, date_key)
    features_df = engineer_features_polars(df)

    feature_path = DATA_PARQUET_DIR / f"features_{sport.lower()}_{date_key}.parquet"
    features_df.write_parquet(feature_path)
    features_df.write_parquet(DATA_PARQUET_DIR / "features.parquet")
    return features_df


def historical_team_average_score(games_parquet_path: str | Path) -> pl.DataFrame:
    sql = f"""
    SELECT
        team,
        odds,
        AVG(score) as avg_score
    FROM '{games_parquet_path}'
    GROUP BY team, odds
    """
    return engine.query_parquet(sql)
