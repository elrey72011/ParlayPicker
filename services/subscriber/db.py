"""Small PostgreSQL boundary; no research storage is reachable from here."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import uuid

import psycopg
from psycopg.rows import dict_row


ROOT = Path(__file__).resolve().parent


class Database:
    def __init__(self, url: str):
        if not url.startswith(("postgresql://", "postgres://")):
            raise ValueError("subscriber service requires PostgreSQL")
        self.url = url

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.url, row_factory=dict_row) as connection:
            with connection.transaction():
                yield connection

    def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with psycopg.connect(self.url, row_factory=dict_row) as connection:
            return connection.execute(sql, params).fetchone()

    def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with psycopg.connect(self.url, row_factory=dict_row) as connection:
            return list(connection.execute(sql, params).fetchall())

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(sql, params)
            return cursor.rowcount

    def migrate(self) -> None:
        migrations = sorted((ROOT / "migrations").glob("*.sql"))
        with psycopg.connect(self.url, autocommit=True) as connection:
            for migration in migrations:
                connection.execute(migration.read_text(encoding="utf-8"))


def new_id() -> uuid.UUID:
    return uuid.uuid4()
