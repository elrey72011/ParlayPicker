"""Retrospective actual-starter features; never evidence of pregame announcement."""
from collections import Counter, defaultdict
from app_core.mlb_history import timestamp

STATS = ("outs", "earnedRuns", "strikeOuts", "baseOnBalls", "hits")
PITCHER_FEATURES = [side + "_starter_" + stat for side in ("home", "away")
                    for stat in ("era", "k9", "bb9", "whip")]


def parse_boxscore(payload, game):
    result = {"game_id": game["game_id"], "appearances": [], "starters": {}}
    for side in ("home", "away"):
        team = payload["teams"][side]
        if team["team"]["id"] != game[side + "_id"]:
            raise ValueError("team_identity_mismatch")
        starters = []
        for player in team["players"].values():
            stats = player.get("stats", {}).get("pitching", {})
            if not stats:
                continue
            if any(isinstance(stats.get(k), bool) or not isinstance(stats.get(k), int)
                   or stats[k] < 0 for k in STATS):
                raise ValueError("invalid_pitching_stats")
            pid = player["person"]["id"]
            result["appearances"].append({"pitcher_id": pid, **{k: stats[k] for k in STATS}})
            if stats.get("gamesStarted") == 1:
                starters.append(pid)
        if len(starters) != 1:
            raise ValueError("ambiguous_actual_starter")
        result["starters"][side] = starters[0]
    if len({p["pitcher_id"] for p in result["appearances"]}) != len(result["appearances"]):
        raise ValueError("duplicate_pitcher")
    return result


def enrich(features, games, boxes):
    history = defaultdict(list)
    for gid, box in boxes.items():
        game = games[int(gid)]
        for appearance in box["appearances"]:
            history[(game["season"], appearance["pitcher_id"])].append((game, appearance))
    rows, excluded = [], Counter()
    for feature in features:
        gid = feature["game_id"]
        box = boxes.get(str(gid))
        if box is None:
            excluded["missing_boxscore"] += 1
            continue
        row = dict(feature)
        ready = True
        for side in ("home", "away"):
            pid = box["starters"][side]
            prior = [(g, a) for g, a in history[(feature["season"], pid)]
                     if timestamp(g["completed_at"]) < timestamp(feature["cutoff"])]
            prior.sort(key=lambda pair: (timestamp(pair[0]["completed_at"]), pair[0]["game_id"]))
            sums = {k: sum(a[k] for g, a in prior) for k in STATS}
            if len(prior) < 3 or sums["outs"] < 27:
                ready = False
                continue
            prefix = side + "_starter_"
            row[prefix + "id"] = pid
            row[prefix + "source_game_ids"] = [g["game_id"] for g, a in prior]
            row[prefix + "era"] = sums["earnedRuns"] * 27 / sums["outs"]
            row[prefix + "k9"] = sums["strikeOuts"] * 27 / sums["outs"]
            row[prefix + "bb9"] = sums["baseOnBalls"] * 27 / sums["outs"]
            row[prefix + "whip"] = (sums["hits"] + sums["baseOnBalls"]) * 3 / sums["outs"]
        if ready:
            rows.append(row)
        else:
            excluded["insufficient_pitcher_history"] += 1
    return rows, dict(excluded)
