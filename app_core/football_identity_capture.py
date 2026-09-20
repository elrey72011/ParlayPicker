"""Pregame provider identity observation; no model or feature validation inferred."""
from copy import deepcopy
from datetime import datetime, timezone
import requests
from app_core.ncaaf_history import timestamp
from app_core.espn_results import _scoreboard_urls
from core.exposure_ledger import digest


def attach(games, sport, events, observed_at):
    out = deepcopy(games)
    observed = timestamp(observed_at)
    if sport not in {"NFL", "NCAAF"} or observed is None:
        return out
    if sport == "NFL":
        from app_core.nfl_identity import nfl_result_name as normalize
    else:
        from app_core.ncaaf_identity import normalize_ncaaf_team as normalize
    for game in out:
        start = timestamp(game.get("commence_time"))
        game["football_identity_status"] = "UNRESOLVED"
        if not start or observed >= start:
            game["football_identity_status"] = "NOT_PREGAME"
            continue
        matches = {}
        for event in events:
            for competition in event.get("competitions", []):
                kickoff = timestamp(competition.get("date") or event.get("date"))
                teams = competition.get("competitors", [])
                home = [t.get("team", {}) for t in teams if t.get("homeAway") == "home"]
                away = [t.get("team", {}) for t in teams if t.get("homeAway") == "away"]
                if len(home) != 1 or len(away) != 1 or kickoff != start:
                    continue
                def agrees(team, name):
                    return bool(name) and normalize(name) in {
                        normalize(team[k]) for k in ("displayName", "shortDisplayName", "name") if team.get(k)}
                if not agrees(home[0], game.get("home_team")) or not agrees(away[0], game.get("away_team")):
                    continue
                eid, hid, aid = event.get("id"), home[0].get("id"), away[0].get("id")
                if not all(str(x or "").isdigit() for x in (eid, hid, aid)) or hid == aid:
                    continue
                matches[(str(eid),str(hid),str(aid))] = event
        if len(matches) != 1:
            continue
        (eid,hid,aid), event = next(iter(matches.items()))
        existing = game.get("provider_ids") or {}
        if not isinstance(existing, dict) or (existing.get("espn") is not None and str(existing["espn"]) != eid):
            game["football_identity_status"] = "CONFLICT"
            continue
        home_id, away_id = f"espn:{sport.lower()}:{hid}", f"espn:{sport.lower()}:{aid}"
        if any(game.get(k) not in (None, "", v) for k,v in (("home_team_id",home_id),("away_team_id",away_id))):
            game["football_identity_status"] = "CONFLICT"
            continue
        game.update(home_team_id=home_id, away_team_id=away_id,
                    provider_ids={**existing, "espn": eid}, football_identity_status="MATCHED",
                    football_identity_observed_at=observed.isoformat(), football_identity_source_hash=digest(event))
    return out


def collect(games, sport):
    if sport not in {"NFL", "NCAAF"}:
        return games
    dates = sorted({t.strftime("%Y%m%d") for g in games if (t := timestamp(g.get("commence_time")))})
    # Bounded work per refresh. Missing dates stay unresolved, never guessed.
    events = []
    for day in dates[:3]:
        for url in _scoreboard_urls(sport, day):
            try:
                response = requests.get(url, timeout=(3, 5))
                response.raise_for_status()
                events.extend(response.json().get("events", []))
            except (requests.RequestException, ValueError, TypeError, AttributeError):
                continue
    return attach(games, sport, events, datetime.now(timezone.utc).isoformat())
