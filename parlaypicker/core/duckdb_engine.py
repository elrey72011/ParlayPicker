"""DuckDB analytics engine with Parquet and Arrow interoperability."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pyarrow as pa

DB_PATH = Path("data/parlaypicker.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class DuckDBEngine:
    """Thin wrapper around a persistent DuckDB connection."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)

    def query(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        """Run SQL and return a Polars DataFrame."""
        rel = self.conn.execute(sql, params or [])
        return pl.from_arrow(rel.arrow())

    def query_parquet(self, sql: str) -> pl.DataFrame:
        """Run SQL that references parquet files directly."""
        return self.query(sql)

    def register_arrow(self, name: str, table: pa.Table) -> None:
        self.conn.register(name, table)

    def register_polars(self, name: str, frame: pl.DataFrame) -> None:
        self.register_arrow(name, frame.to_arrow())

    def heavy_join(self, odds: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
        """Perform heavy joins in DuckDB for better performance."""
        self.register_polars("odds", odds)
        self.register_polars("stats", stats)
        sql = """
        SELECT
            odds.team,
            odds.odds,
            stats.off_rating
        FROM odds
        JOIN stats
            ON odds.team = stats.team
        """
        return self.query(sql)


engine = DuckDBEngine()


def query(sql: str, params: list[Any] | None = None) -> pl.DataFrame:
    """Convenience query function backed by the shared engine."""
    return engine.query(sql, params=params)
