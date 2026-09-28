"""Apply only versioned subscriber-schema migrations."""

from .db import Database
from .settings import Settings


def main() -> None:
    Database(Settings.from_env().database_url).migrate()


if __name__ == "__main__":
    main()
