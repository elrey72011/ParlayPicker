"""Bounded, credential-safe CFBD historical access sampling."""
from datetime import datetime, timezone
import math
import requests

BASE_URL = "https://api.collegefootballdata.com"


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _completed(game, year):
    if not isinstance(game, dict):
        return False
    try:
        kickoff = datetime.fromisoformat(str(game.get("startDate", "")).replace("Z", "+00:00"))
    except ValueError:
        return False
    return (game.get("season") == year and game.get("completed") is True
            and game.get("startTimeTBD") is False and kickoff.tzinfo is not None
            and all(_number(game.get(k)) for k in ("id", "homeId", "awayId", "homePoints", "awayPoints"))
            and game["homeId"] != game["awayId"])


def probe_cfbd_access(token, *, current_year=None, get=None):
    """Sample week 1 of three prior years, at most six calls, without retries."""
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "status": "not_checked", "requests_made": 0, "seasons": [],
              "scope": "Week 1 regular-season FBS results and one team-stat sample per year; not a full-season completeness or model validation check."}
    token = str(token or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token or any(c.isspace() for c in token):
        report["status"] = "missing_or_invalid_key"
        return report
    get = get or requests.get
    year = current_year or datetime.now(timezone.utc).year

    def fetch(path, params):
        report["requests_made"] += 1
        try:
            response = get(BASE_URL + path, params=params,
                           headers={"Authorization": "Bearer " + token},
                           timeout=6, allow_redirects=False)
            if response.status_code != 200:
                return None, "http_" + str(int(response.status_code))
            payload = response.json()
            if not isinstance(payload, list):
                return None, "invalid_response_shape"
            return payload, None
        except requests.RequestException:
            return None, "connection_failed"
        except (ValueError, TypeError):
            return None, "invalid_json_response"

    for season in range(year - 1, year - 4, -1):
        row = {"season": season, "status": "not_checked", "returned_games": 0,
               "usable_completed_games": 0, "team_stats_accessible": False}
        report["seasons"].append(row)
        games, error = fetch("/games", {"year": season, "week": 1,
                             "seasonType": "regular", "classification": "fbs"})
        if error:
            row["status"] = error
            report["status"] = "incomplete"
            break
        row["returned_games"] = len(games)
        usable = [g for g in games if _completed(g, season)]
        row["usable_completed_games"] = len(usable)
        if not usable:
            row["status"] = "no_usable_completed_games"
            continue
        game = usable[0]
        row["sample_game_id"] = game["id"]
        stats, error = fetch("/games/teams", {"id": game["id"]})
        if error:
            row["status"] = error
            report["status"] = "incomplete"
            break
        matched = set()
        for item in stats:
            if not isinstance(item, dict) or item.get("id") != game["id"]:
                continue
            teams = item.get("teams")
            for team in teams if isinstance(teams, list) else []:
                if not isinstance(team, dict):
                    continue
                entries = team.get("stats")
                if (team.get("teamId") in (game["homeId"], game["awayId"])
                    and isinstance(entries, list) and any(
                        isinstance(s, dict) and s.get("category") and s.get("stat") is not None
                        and str(s["stat"]).strip() for s in entries)):
                    matched.add(team["teamId"])
        row["team_stats_accessible"] = len(matched) == 2
        row["status"] = "sample_accessible" if len(matched) == 2 else "team_stats_missing_or_unmatched"
    if report["status"] != "incomplete":
        report["status"] = ("samples_accessible" if all(
            r["status"] == "sample_accessible" for r in report["seasons"]) else "incomplete")
    return report
