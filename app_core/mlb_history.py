"""Research-only MLB history. No standings snapshots or betting promotion."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone


def timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timezone required")
    return parsed.astimezone(timezone.utc)


def normalize_game(feed):
    """Reject incomplete games and missing chronology instead of filling defaults."""
    data = feed["gameData"]
    if data["status"].get("abstractGameState") != "Final":
        raise ValueError("not_final")
    if data["game"].get("type") != "R":
        raise ValueError("not_regular_season")
    plays = feed["liveData"]["plays"]["allPlays"]
    if not plays or any(not p["about"].get("isComplete") for p in plays):
        raise ValueError("incomplete_play_history")
    # Actual first/last play, including resumed games; never use original date as end.
    start = min(timestamp(p["about"]["startTime"]) for p in plays)
    end = max(timestamp(p["about"]["endTime"]) for p in plays)
    scheduled = timestamp(data["datetime"]["dateTime"])
    cutoff = min(start, scheduled)
    if end <= start:
        raise ValueError("invalid_chronology")
    scores = feed["liveData"]["linescore"]["teams"]
    row = {"game_id": int(feed["gamePk"]), "season": int(data["game"]["season"]),
           "cutoff": cutoff.isoformat(), "started_at": start.isoformat(),
           "completed_at": end.isoformat()}
    for side in ("home", "away"):
        row[side + "_id"] = int(data["teams"][side]["id"])
        row[side + "_team"] = data["teams"][side]["name"]
        score = scores[side]["runs"]
        if isinstance(score, bool) or not isinstance(score, int) or score < 0:
            raise ValueError("invalid_score")
        row[side + "_score"] = score
    if row["home_id"] == row["away_id"] or row["home_score"] == row["away_score"]:
        raise ValueError("invalid_final")
    return row


def build_dataset(games, minimum_games=10):
    """Same-season prior completed games only; targets kept out of features."""
    if minimum_games < 1:
        raise ValueError("minimum_games must be positive")
    unique = {}
    for game in games:
        key = game["game_id"]
        if key in unique and unique[key] != game:
            raise ValueError("conflicting_game_id")
        unique[key] = game
    ordered = sorted(unique.values(), key=lambda g: (g["cutoff"], g["game_id"]))
    histories = {}
    for g in ordered:
        for team in (g["home_id"], g["away_id"]):
            histories.setdefault((g["season"], team), []).append(g)
    features, targets, excluded = [], [], Counter()
    for game in ordered:
        row = {k: game[k] for k in ("game_id", "season", "cutoff", "home_id", "away_id")}
        eligible = True
        for side in ("home", "away"):
            team = game[side + "_id"]
            prior = [g for g in histories[(game["season"], team)]
                     if timestamp(g["completed_at"]) < timestamp(game["cutoff"])]
            prior.sort(key=lambda g: (g["completed_at"], g["game_id"]))
            if len(prior) < minimum_games:
                eligible = False
            scored, allowed = [], []
            for g in prior:
                own = "home" if g["home_id"] == team else "away"
                other = "away" if own == "home" else "home"
                scored.append(g[own + "_score"]); allowed.append(g[other + "_score"])
            row[side + "_prior_game_ids"] = [g["game_id"] for g in prior]
            row[side + "_prior_games"] = len(prior)
            if prior:
                row[side + "_ppg"] = sum(scored) / len(prior)
                row[side + "_oppg"] = sum(allowed) / len(prior)
                row[side + "_win_pct"] = sum(a > b for a, b in zip(scored, allowed)) / len(prior)
        if not eligible:
            excluded["insufficient_prior_games"] += 1
            continue
        features.append(row)
        targets.append({k: game[k] for k in ("game_id", "home_score", "away_score")})
    return {"features": features, "targets": targets, "audit": {
        "games": len(ordered), "eligible_games": len(features), "excluded": dict(excluded),
        "season_counts": dict(Counter(g["season"] for g in ordered)),
        "research_only": True, "production_eligible": False,
        "limitation": "Historical feed corrections/publication times are unverified. No historical odds collected."}}
