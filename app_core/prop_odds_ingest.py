"""Odds API player-prop ingestion for MLB and NFL over/under markets.

Player props live on the per-event odds endpoint under market keys like
``pitcher_strikeouts``. Each outcome carries the player in ``description``, the side in
``name`` (Over/Under), the line in ``point`` and the price in ``price``. Parsing is split
from the HTTP call so it's unit-tested on a fixture (no network / no API key).
"""
from __future__ import annotations

import time
from typing import Any

STRIKEOUT_MARKET_KEY = "pitcher_strikeouts"
# Pitcher-prop expansion (4 Jul): outs and walks are count stats projectable from
# the same game logs as strikeouts (inningsPitched -> outs, baseOnBalls -> walks).
# Hits/earned-runs are deliberately excluded: defense/BABIP-dominated, the model
# has no informational edge there.
PITCHER_PROP_MARKET_KEYS = ("pitcher_strikeouts", "pitcher_outs", "pitcher_walks")
BATTER_PROP_MARKET_KEYS = ("batter_hits", "batter_total_bases")
MLB_PLAYER_PROP_MARKET_KEYS = PITCHER_PROP_MARKET_KEYS + BATTER_PROP_MARKET_KEYS
# NFL v1 deliberately starts with repeatable volume markets. Touchdowns and
# interceptions are low-frequency binary/count outcomes and need a separately
# validated model before they can be ranked honestly.
NFL_PLAYER_PROP_MARKET_KEYS = (
    "player_pass_yds",
    "player_pass_attempts",
    "player_pass_completions",
    "player_rush_yds",
    "player_rush_attempts",
    "player_reception_yds",
    "player_receptions",
)
_DEFAULT_BOOK_PRIORITY = ("novig", "draftkings", "fanduel", "betmgm")
_PROP_FETCH_ATTEMPTS = 3
_USE_CLIENT_BOOKMAKERS = object()


class PropOddsFetchError(RuntimeError):
    """Raised when a player-prop request fails instead of returning an empty board."""


def _prop_regions_with_full_us_coverage(client: Any) -> str:
    """Return the US regions needed for broad player-prop bookmaker coverage."""
    configured = str(getattr(client, "regions", "") or "")
    regions = [
        value.strip().lower()
        for value in configured.split(",")
        if value.strip().lower() in {"us", "us2"}
    ]
    for required in ("us", "us2"):
        if required not in regions:
            regions.append(required)
    return ",".join(regions)


def _increment_diagnostic(diagnostics: dict | None, key: str) -> None:
    if isinstance(diagnostics, dict):
        diagnostics[key] = int(diagnostics.get(key, 0) or 0) + 1


def _request_prop_json(
    url: str,
    params: dict[str, object],
    *,
    error_prefix: str,
    raise_on_error: bool,
) -> object | None:
    """GET one prop endpoint with bounded retries and secret-safe errors."""
    import requests

    resp = None
    for attempt in range(_PROP_FETCH_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=15)
        except requests.RequestException as exc:
            if attempt + 1 >= _PROP_FETCH_ATTEMPTS:
                if raise_on_error:
                    raise PropOddsFetchError(f"{error_prefix}_network_error") from exc
                return None
            time.sleep(0.2 * (2 ** attempt))
            continue
        if resp.status_code == 200:
            break
        if (
            resp.status_code == 429 or resp.status_code >= 500
        ) and attempt + 1 < _PROP_FETCH_ATTEMPTS:
            time.sleep(0.2 * (2 ** attempt))
            continue
        if raise_on_error:
            raise PropOddsFetchError(
                f"{error_prefix}_http_{int(resp.status_code)}"
            )
        return None
    if resp is None or resp.status_code != 200:
        if raise_on_error:
            status = int(resp.status_code) if resp is not None else "unknown"
            raise PropOddsFetchError(f"{error_prefix}_http_{status}")
        return None
    try:
        return resp.json()
    except Exception as exc:
        if raise_on_error:
            raise PropOddsFetchError(f"{error_prefix}_response_error") from exc
        return None


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
    """Extract per-player over/under lines for ONE prop market from an event payload.

    Returns one row per player: ``{pitcher, line, over_odds, under_odds, book,
    home_team, away_team, market_key}``. For each player, uses the first complete
    over/under quote in ``book_priority`` and fills players absent from that book from
    later books. This preserves the preferred-book policy without allowing a partial
    NoVig board to hide players available at DraftKings/FanDuel/BetMGM. Players missing
    a complete over+under pair at a single line are skipped.
    """
    if not isinstance(event_json, dict):
        return []
    home = event_json.get("home_team", "")
    away = event_json.get("away_team", "")
    books = {b.get("key"): b for b in event_json.get("bookmakers", []) if isinstance(b, dict)}

    ordered = [books[k] for k in book_priority if k in books]
    ordered += [b for k, b in books.items() if k not in book_priority]

    selected_by_player: dict[str, dict] = {}
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
        if str(market_key).startswith("batter_"):
            participant_type = "batter"
        elif str(market_key).startswith("player_"):
            participant_type = "nfl_player"
        else:
            participant_type = "pitcher"
        for player, quote in by_pitcher.items():
            if "over_odds" not in quote or "under_odds" not in quote:
                continue
            player_key = " ".join(str(player).strip().lower().split())
            if not player_key or player_key in selected_by_player:
                continue
            selected_by_player[player_key] = {
                "player": player,
                "participant_type": participant_type,
                "pitcher": player if participant_type == "pitcher" else None,
                "batter": player if participant_type == "batter" else None,
                "line": float(quote["line"]),
                "over_odds": int(quote["over_odds"]),
                "under_odds": int(quote["under_odds"]),
                "book": book.get("key"),
                "home_team": home,
                "away_team": away,
                "market_key": market_key,
                "event_id": event_json.get("id"),
                "commence_time": event_json.get("commence_time"),
            }
    return list(selected_by_player.values())

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
    *,
    regions: str | None = None,
    bookmakers: object = _USE_CLIENT_BOOKMAKERS,
    raise_on_error: bool = False,
) -> list[dict]:
    """Fetch + parse all requested pitcher-prop markets for one event.

    ONE HTTP call (comma-joined markets) per event, parsed per market; each row
    carries its ``market_key``. By default failures return []; NFL callers can
    request ``PropOddsFetchError`` so transport failures are not mislabeled as
    an empty sportsbook board.
    """
    try:
        url = f"{client.BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
        params = {
            "apiKey": client.api_key,
            "regions": regions or getattr(client, "regions", "us2"),
            "markets": ",".join(market_keys),
            "oddsFormat": "american",
        }
        requested_bookmakers = (
            getattr(client, "bookmakers", ",".join(_DEFAULT_BOOK_PRIORITY))
            if bookmakers is _USE_CLIENT_BOOKMAKERS
            else bookmakers
        )
        if requested_bookmakers:
            params["bookmakers"] = str(requested_bookmakers)
        payload = _request_prop_json(
            url,
            params,
            error_prefix="player_prop",
            raise_on_error=raise_on_error,
        )
        if not isinstance(payload, dict):
            return []
        rows: list[dict] = []
        for mk in market_keys:
            rows.extend(parse_pitcher_props(payload, mk))
        return rows
    except PropOddsFetchError:
        raise
    except Exception as exc:
        if raise_on_error:
            raise PropOddsFetchError("player_prop_response_error") from exc
        return []


def fetch_event_player_prop_markets(
    client: Any,
    sport_key: str,
    event_id: str,
    *,
    regions: str | None = None,
) -> set[str]:
    """Discover recently open player-prop keys across bookmakers for one event."""
    url = f"{client.BASE_URL}/sports/{sport_key}/events/{event_id}/markets"
    params = {
        "apiKey": client.api_key,
        "regions": regions or _prop_regions_with_full_us_coverage(client),
        "dateFormat": "iso",
    }
    payload = _request_prop_json(
        url,
        params,
        error_prefix="player_prop_market_discovery",
        raise_on_error=True,
    )

    market_keys: set[str] = set()
    if isinstance(payload, dict):
        for book in payload.get("bookmakers", []):
            if not isinstance(book, dict):
                continue
            for market in book.get("markets", []):
                if isinstance(market, dict) and market.get("key"):
                    market_keys.add(str(market["key"]).strip())
    return {key for key in market_keys if key.startswith("player_")}


def fetch_mlb_player_props(
    client: Any,
    sport_key: str,
    event_id: str,
    market_keys: tuple[str, ...] = MLB_PLAYER_PROP_MARKET_KEYS,
) -> list[dict]:
    """Fetch the supported pitcher + batter markets in one event request."""
    return fetch_pitcher_props(client, sport_key, event_id, market_keys)


def fetch_nfl_player_props(
    client: Any,
    sport_key: str,
    event_id: str,
    market_keys: tuple[str, ...] = NFL_PLAYER_PROP_MARKET_KEYS,
    *,
    diagnostics: dict | None = None,
) -> list[dict]:
    """Fetch the supported NFL volume markets in one event request.

    The transport and quote shape are identical to MLB props; returned records
    carry ``participant_type=nfl_player`` so modeling, grading, and calibration
    remain league isolated downstream.
    """
    first_error: PropOddsFetchError | None = None
    try:
        rows = fetch_pitcher_props(
            client,
            sport_key,
            event_id,
            market_keys,
            raise_on_error=True,
        )
    except PropOddsFetchError as exc:
        first_error = exc
        rows = []
    if rows:
        return rows

    broad_regions = _prop_regions_with_full_us_coverage(client)
    try:
        open_market_keys = fetch_event_player_prop_markets(
            client,
            sport_key,
            event_id,
            regions=broad_regions,
        )
        _increment_diagnostic(
            diagnostics, "nfl_prop_market_discovery_success_count"
        )
    except PropOddsFetchError as exc:
        _increment_diagnostic(
            diagnostics, "nfl_prop_market_discovery_error_count"
        )
        # Keep the pre-discovery behavior as a bounded fallback when the free
        # market-keys endpoint is unavailable.
        recovered: list[dict] = []
        fallback_errors = 0
        for market_key in market_keys:
            try:
                recovered.extend(
                    fetch_pitcher_props(
                        client,
                        sport_key,
                        event_id,
                        (market_key,),
                        raise_on_error=True,
                    )
                )
            except PropOddsFetchError:
                fallback_errors += 1
        if recovered:
            return recovered
        if first_error is not None or fallback_errors == len(market_keys):
            raise first_error or exc
        return []

    available = tuple(key for key in market_keys if key in open_market_keys)
    if isinstance(diagnostics, dict) and open_market_keys:
        prior = {
            value
            for value in str(
                diagnostics.get("nfl_prop_available_market_keys", "") or ""
            ).split("|")
            if value
        }
        prior.update(open_market_keys)
        diagnostics["nfl_prop_available_market_keys"] = "|".join(sorted(prior))
    if not available:
        _increment_diagnostic(
            diagnostics, "nfl_prop_events_without_supported_markets"
        )
        return []

    _increment_diagnostic(
        diagnostics, "nfl_prop_events_with_supported_markets"
    )
    try:
        rows = fetch_pitcher_props(
            client,
            sport_key,
            event_id,
            available,
            regions=broad_regions,
            bookmakers=None,
            raise_on_error=True,
        )
    except PropOddsFetchError:
        rows = []
    if rows:
        _increment_diagnostic(
            diagnostics, "nfl_prop_broad_book_fallback_count"
        )
        return rows

    # A listed market can disappear between discovery and the odds request, or
    # one temporarily unavailable key can empty a multi-market response. Retry
    # only the discovered supported subset, one key at a time.
    recovered = []
    fetch_errors = 0
    for market_key in available:
        try:
            recovered.extend(
                fetch_pitcher_props(
                    client,
                    sport_key,
                    event_id,
                    (market_key,),
                    regions=broad_regions,
                    bookmakers=None,
                    raise_on_error=True,
                )
            )
        except PropOddsFetchError:
            fetch_errors += 1
    if recovered:
        _increment_diagnostic(
            diagnostics, "nfl_prop_broad_book_fallback_count"
        )
        return recovered
    if fetch_errors == len(available):
        raise PropOddsFetchError("player_prop_broad_fetch_failed")
    return []

