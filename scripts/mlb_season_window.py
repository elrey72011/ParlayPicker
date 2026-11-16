"""Print whether a scheduled MLB receipt run has work to do today (the Actions season gate).

    python scripts/mlb_season_window.py        # prints "true" or "false"

A scheduled "MLB receipt reconciliation" run restores the whole receipt store, which takes
about 25 minutes, and outside the season there is no game to capture and no result to settle.
The window is deliberately wide: it opens before the earliest overseas opening series (those
have started as early as 18 March) and stays open to mid-November, past the latest World Series
finish (5 November), so the last results settle. Manual runs ignore it. Standard library only,
so it runs before the dependencies are installed.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

OPENS = (3, 15)    # (month, day) in UTC, inclusive
CLOSES = (11, 15)  # (month, day) in UTC, inclusive


def in_season(today: date) -> bool:
    return OPENS <= (today.month, today.day) <= CLOSES


if __name__ == "__main__":
    print("true" if in_season(datetime.now(timezone.utc).date()) else "false")
