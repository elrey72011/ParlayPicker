"""Research-only ESPN/DraftKings fallback for same-day NCAAF FCS odds.

The primary source remains The Odds API (including Novig). ESPN's default
college-football scoreboard can omit FCS-only opening-day slates, so this module
queries ESPN's FCS group explicitly and supplies only games that are absent from
the primary response. Recovered prices are labeled separately downstream and
must never be treated as Novig execution prices or production-approved wagers.
"""

from __future__ import annotations

from datetime import datetime
import logging
import re
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import requests


logger = logging.getLogger(__name__)

ESPN_NCAAF_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/"
    "college-football/scoreboard"
)
ESPN_FCS_GROUP_ID = "81"
ESPN_FALLBACK_SOURCE = "espn_ncaaf_fcs_scoreboard"


def _number(value: Any) -> float | None:
    """Parse ESPN's signed numeric tokens such as ``o50.5`` or ``+125``."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def _close_value(container: Any, field: str) -> float | None:
    if not isinstance(container, dict):
        return None
    close = container.get("close")
    if isinstance(close, dict):
        parsed = _number(close.get(field))
        if parsed is not None:
            return parsed
    return _number(container.get(field))


def _provider_key(odds: dict[str, Any]) -> str | None:
    provider = odds.get("provider") or {}
    name = str(provider.get("name") or provider.get("displayName") or "").lower()
    compact = re.sub(r"[^a-z0-9]", "", name)
    if compact in {"draftkings", "draftking"}:
        return "draftkings"
    return None


def _team_name(competitor: dict[str, Any]) -> str:
    team = competitor.get("team") or {}
    return str(
        team.get("displayName")
        or team.get("shortDisplayName")
        or team.get("name")
        or ""
    ).strip()


def _market_outcomes(
    odds: dict[str, Any], home_team: str, away_team: str
) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []

    moneyline = odds.get("moneyline") or {}
    home_ml = _close_value(moneyline.get("home"), "odds")
    away_ml = _close_value(moneyline.get("away"), "odds")
    home_team_odds = odds.get("homeTeamOdds") or {}
    away_team_odds = odds.get("awayTeamOdds") or {}
    if home_ml is None:
        home_ml = _number(home_team_odds.get("moneyLine"))
    if away_ml is None:
        away_ml = _number(away_team_odds.get("moneyLine"))
    if home_ml is not None and away_ml is not None:
        markets.append(
            {
                "key": "h2h",
                "outcomes": [
                    {"name": home_team, "price": home_ml},
                    {"name": away_team, "price": away_ml},
                ],
            }
        )

    point_spread = odds.get("pointSpread") or {}
    home_spread = _close_value(point_spread.get("home"), "line")
    away_spread = _close_value(point_spread.get("away"), "line")
    home_spread_price = _close_value(point_spread.get("home"), "odds")
    away_spread_price = _close_value(point_spread.get("away"), "odds")
    if home_spread_price is None:
        home_spread_price = _number(home_team_odds.get("spreadOdds"))
    if away_spread_price is None:
        away_spread_price = _number(away_team_odds.get("spreadOdds"))

    # Support ESPN's older compact schema, where one unsigned spread is paired
    # with favorite flags rather than explicit home/away signed lines.
    if home_spread is None or away_spread is None:
        spread = _number(odds.get("spread"))
        if spread is not None:
            home_is_favorite = bool(home_team_odds.get("favorite"))
            away_is_favorite = bool(away_team_odds.get("favorite"))
            if home_is_favorite != away_is_favorite:
                home_spread = -abs(spread) if home_is_favorite else abs(spread)
                away_spread = -home_spread
    if all(
        value is not None
        for value in (home_spread, away_spread, home_spread_price, away_spread_price)
    ):
        markets.append(
            {
                "key": "spreads",
                "outcomes": [
                    {
                        "name": home_team,
                        "point": home_spread,
                        "price": home_spread_price,
                    },
                    {
                        "name": away_team,
                        "point": away_spread,
                        "price": away_spread_price,
                    },
                ],
            }
        )

    total = odds.get("total") or {}
    over_line = _close_value(total.get("over"), "line")
    under_line = _close_value(total.get("under"), "line")
    over_price = _close_value(total.get("over"), "odds")
    under_price = _close_value(total.get("under"), "odds")
    legacy_total = _number(odds.get("overUnder"))
    if over_line is None:
        over_line = legacy_total
    if under_line is None:
        under_line = legacy_total
    if over_price is None:
        over_price = _number(odds.get("overOdds"))
    if under_price is None:
        under_price = _number(odds.get("underOdds"))
    if all(
        value is not None
        for value in (over_line, under_line, over_price, under_price)
    ):
        markets.append(
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": over_line, "price": over_price},
                    {"name": "Under", "point": under_line, "price": under_price},
                ],
            }
        )

    return markets


def _canonical_team(value: Any) -> str:
    normalized = str(value or "").lower().replace("st. ", "saint ").replace("st ", "saint ")
    return re.sub(r"[^a-z0-9]", "", normalized)


def _game_key(game: dict[str, Any]) -> tuple[str, str]:
    teams = sorted(
        [_canonical_team(game.get("home_team")), _canonical_team(game.get("away_team"))]
    )
    return teams[0], teams[1]


def merge_missing_ncaaf_games(
    primary_games: Iterable[dict[str, Any]] | None,
    fallback_games: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Append missing FCS events without changing any primary event or quote."""

    merged = [game for game in (primary_games or []) if isinstance(game, dict)]
    seen = {_game_key(game) for game in merged}
    for game in fallback_games or []:
        if not isinstance(game, dict):
            continue
        key = _game_key(game)
        if not all(key) or key in seen:
            continue
        merged.append(game)
        seen.add(key)
    return merged


def fetch_espn_ncaaf_fcs_odds(target_date: str | None = None) -> list[dict[str, Any]]:
    """Return FCS games with complete DraftKings markets from ESPN's scoreboard."""

    date_text = str(target_date or "").strip()[:10]
    try:
        slate_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        slate_date = datetime.now(ZoneInfo("America/New_York")).date()

    params = {
        "dates": slate_date.strftime("%Y%m%d"),
        "groups": ESPN_FCS_GROUP_ID,
        "limit": 300,
    }
    try:
        response = requests.get(ESPN_NCAAF_SCOREBOARD_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("ESPN NCAAF FCS odds fallback failed closed: %s", exc)
        return []
    if not isinstance(payload, dict):
        logger.warning("ESPN NCAAF FCS odds fallback returned a non-object payload")
        return []

    games: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions") or []
        if not competitions or not isinstance(competitions[0], dict):
            continue
        competition = competitions[0]
        competitors = competition.get("competitors") or []
        home = next(
            (
                item
                for item in competitors
                if isinstance(item, dict) and item.get("homeAway") == "home"
            ),
            None,
        )
        away = next(
            (
                item
                for item in competitors
                if isinstance(item, dict) and item.get("homeAway") == "away"
            ),
            None,
        )
        if not home or not away:
            continue
        home_team = _team_name(home)
        away_team = _team_name(away)
        commence_time = str(event.get("date") or competition.get("date") or "").strip()
        if not home_team or not away_team or not commence_time:
            continue
        try:
            commence_datetime = datetime.fromisoformat(
                commence_time.replace("Z", "+00:00")
            )
            if commence_datetime.tzinfo is None:
                commence_datetime = commence_datetime.replace(tzinfo=ZoneInfo("UTC"))
            commence_et = commence_datetime.astimezone(ZoneInfo("America/New_York"))
            game_time_est = commence_et.strftime(
                "%Y-%m-%d %I:%M %p ET"
            ).replace(" 0", " ")
        except (TypeError, ValueError):
            game_time_est = ""

        bookmakers: list[dict[str, Any]] = []
        for odds in competition.get("odds") or []:
            if not isinstance(odds, dict) or _provider_key(odds) != "draftkings":
                continue
            markets = _market_outcomes(odds, home_team, away_team)
            if markets:
                bookmakers.append(
                    {"key": "draftkings", "title": "DraftKings", "markets": markets}
                )
                break
        if not bookmakers:
            continue

        event_id = str(event.get("id") or competition.get("id") or "").strip()
        if not event_id or event_id in seen_ids:
            continue
        seen_ids.add(event_id)
        normalized_teams = sorted([_canonical_team(home_team), _canonical_team(away_team)])
        games.append(
            {
                "id": f"espn-{event_id}",
                "matchup_id": (
                    "americanfootball_ncaaf:"
                    f"{normalized_teams[0]}:{normalized_teams[1]}:{slate_date.isoformat()}"
                ),
                "sport_key": "americanfootball_ncaaf",
                "sport_title": "NCAAF",
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence_time,
                "game_date": slate_date.isoformat(),
                "game_time_est": game_time_est,
                "odds_feed_source": ESPN_FALLBACK_SOURCE,
                "bookmakers": bookmakers,
            }
        )

    logger.info(
        "Recovered %s same-day NCAAF FCS games from ESPN/DraftKings for %s",
        len(games),
        slate_date,
    )
    return games
