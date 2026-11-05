"""The NHL's own official final scores, as an independent check for NHL settlement (roadmap D17).

The Odds API does not document how it reports a shootout game. NHL puck lines and
totals at US books settle on the official final score, including overtime and the
shootout (the shootout winner is credited with one goal), which is also the target in
``nhl_puck_line_evidence``. An NHL game is therefore settled only when the provider's
final score **equals** the NHL's own official final for that game
(``prospective_evidence.insert_score_settlement(..., official=...)``).

The source is the NHL's free public score feed (``api-web.nhle.com``): no key, no
paid credits. A game matches only when its start time is within 15 minutes and both
provider team names end with the NHL's team nicknames, and exactly one game matches.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json

SCORE_URL = "https://api-web.nhle.com/v1/score/{date}"
FINAL_STATES = frozenset({"OFF", "FINAL"})
START_TOLERANCE = timedelta(minutes=15)


def _time(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return result.astimezone(timezone.utc)


def _nickname(team):
    name = team.get("name") or team.get("commonName") or {}
    return str(name.get("default") if isinstance(name, dict) else name or "").strip()


def finals(payload):
    """Completed games in one score-feed response, with the raw game's hash."""
    out = []
    for game in (payload or {}).get("games") or []:
        if game.get("gameState") not in FINAL_STATES:
            continue
        home, away = game.get("homeTeam") or {}, game.get("awayTeam") or {}
        if type(home.get("score")) is not int or type(away.get("score")) is not int:
            continue
        raw = json.dumps(game, sort_keys=True, separators=(",", ":"))
        out.append({"game_id": game.get("id"), "start": _time(game["startTimeUTC"]),
                    "home_nickname": _nickname(home), "away_nickname": _nickname(away),
                    "home_score": home["score"], "away_score": away["score"],
                    "last_period_type": (game.get("gameOutcome") or {}).get("lastPeriodType"),
                    "source_hash": hashlib.sha256(raw.encode()).hexdigest()})
    return out


def match(event, games):
    """The single official game for a canonical event, or None (no match or ambiguous)."""
    start = _time(event["scheduled_start"])
    found = [g for g in games
             if abs(g["start"] - start) <= START_TOLERANCE and g["home_nickname"] and g["away_nickname"]
             and str(event["home_team"]).endswith(" " + g["home_nickname"])
             and str(event["away_team"]).endswith(" " + g["away_nickname"])]
    return found[0] if len(found) == 1 else None


def fetch(dates, request_get):
    """Official finals for the given ``YYYY-MM-DD`` dates (US Eastern game dates)."""
    games = []
    for date in sorted(set(dates)):
        response = request_get(SCORE_URL.format(date=date), timeout=20)
        response.raise_for_status()
        games.extend(finals(response.json()))
    return games


def game_dates(events):
    """Feed dates to request: the UTC start date and the day before (evening games)."""
    dates = set()
    for event in events:
        start = _time(event["scheduled_start"])
        dates.update({start.date().isoformat(), (start - timedelta(days=1)).date().isoformat()})
    return dates
