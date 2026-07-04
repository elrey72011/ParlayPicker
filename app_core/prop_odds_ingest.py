"""Odds API player-prop ingestion. v1: MLB pitcher strikeouts.

Player props live on the per-event odds endpoint under market keys like
``pitcher_strikeouts``. Each outcome carries the player in ``description``, the side in
``name`` (Over/Under), the line in ``point`` and the price in ``price``. Parsing is split
from the HTTP call so it's unit-tested on a fixture (no network / no API key).
"""
from __future__ import annotations

from typing import Any

STRIKEOUT_MARKET_KEY = "pitcher_strikeouts"
# Pitcher-prop expansion (4 Jul): outs and walks are count stats projectable from
# the same game logs as strikeouts (inningsPitched -> outs, baseOnBalls -> walks).
# Hits/earned-runs are deliberately excluded: defense/BABIP-dominated, the model
# has no informational edge there.
PITCHER_PROP_MARKET_KEYS = ("pitcher_strikeouts", "pitcher_outs", "pitcher_walks")
_DEFAULT_BOOK_PRIORITY = ("novig", "draftkings", "fanduel", "betmgm")


def parse_strikeout_props(
    event_json: dict, book_priority: tuple[str, ...] = _DEFAULT_BOOK_PRIORITY
) -> list[dict]:
    """Back-compat wrapper: strikeout rows only (see parse_pitcher_props)."""
    return parse_pitcher_props(event_json, STRIKEOUT_MARKET_KEY, book_priority)


def parse_pitcher_props(
    event_json: dict,
    market_key: str = STRIKEOUT_MARKET_KEY,
    book_priority: tuple[str, ...] = _DEFAULT_BOOK_PRIORITY,
) -> list[dict]:
    """Extract per-pitcher over/under lines for ONE prop market from an event payload.

    Returns one row per pitcher: ``{pitcher, line, over_odds, under_odds, book,
    home_team, away_team, market_key}``. Uses the first book in ``book_priority`` that
    prices the market (falls back to any book that does). Pitchers missing a complete
    over+under pair at a single line are skipped.
    """
    if not isinstance(event_json, dict):
        return []
    home = event_json.get("home_team", "")
    away = event_json.get("away_team", "")
    books = {b.get("key"): b for b in event_json.get("bookmakers", []) if isinstance(b, dict)}

    ordered = [books[k] for k in book_priority if k in books]
    ordered += [b for k, b in books.items() if k not in book_priority]

    for book in ordered:
        market = next(
            (m for m in book.get("markets", []) if m.get("key") == market_key),
            None,
        )
        if not market:
            continue
        by_pitcher: dict[str, dict] = {}
        for oc in market.get("outcomes", []):
            pitcher = oc.get("description")
            side = str(oc.get("name", "")).strip().lower()
            point = oc.get("point")
            price = oc.get("price")
            if not pitcher or point is None or price is None:
                continue
            slot = by_pitcher.setdefault(pitcher, {"line": point})
            # Only pair sides quoted at the same line.
            if slot["line"] != point:
                continue
            if side.startswith("o"):
                slot["over_odds"] = price
            elif side.startswith("u"):
                slot["under_odds"] = price
        rows = [
            {
                "pitcher": p,
                "line": float(v["line"]),
                "over_odds": int(v["over_odds"]),
                "under_odds": int(v["under_odds"]),
                "book": book.get("key"),
                "home_team": home,
                "away_team": away,
                "market_key": market_key,
            }
            for p, v in by_pitcher.items()
            if "over_odds" in v and "under_odds" in v
        ]
        if rows:
            return rows
    return []


def fetch_strikeout_props(client: Any, sport_key: str, event_id: str) -> list[dict]:
    """Fetch + parse pitcher-strikeout props for one event via the Odds API event endpoint.

    Calls the per-event odds URL with ``markets=pitcher_strikeouts`` (the props endpoint is
    separate and per-event). Returns [] on any failure so a missing/unsupported market never
    breaks the slate.
    """
    rows = fetch_pitcher_props(client, sport_key, event_id, (STRIKEOUT_MARKET_KEY,))
    return [r for r in rows if r.get("market_key") == STRIKEOUT_MARKET_KEY]


def fetch_pitcher_props(
    client: Any,
    sport_key: str,
    event_id: str,
    market_keys: tuple[str, ...] = PITCHER_PROP_MARKET_KEYS,
) -> list[dict]:
    """Fetch + parse all requested pitcher-prop markets for one event.

    ONE HTTP call (comma-joined markets) per event, parsed per market; each row
    carries its ``market_key``. Returns [] on any failure so a missing/unsupported
    market never breaks the slate.
    """
    try:
        import requests

        url = f"{client.BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
        params = {
            "apiKey": client.api_key,
            "regions": getattr(client, "regions", "us2"),
            "markets": ",".join(market_keys),
            "bookmakers": getattr(client, "bookmakers", ",".join(_DEFAULT_BOOK_PRIORITY)),
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        payload = resp.json()
        rows: list[dict] = []
        for mk in market_keys:
            rows.extend(parse_pitcher_props(payload, mk))
        return rows
    except Exception:
        return []
