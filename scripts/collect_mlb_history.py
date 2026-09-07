"""Collect a bounded, resumable MLB research checkpoint from official StatsAPI."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core.mlb_history import build_dataset, normalize_game


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def collect(output, seasons, max_requests=20):
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {
        "schema_version": 1, "schedules": {}, "games": {}, "excluded": {}, "requests": 0}
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    calls = 0

    def fetch(url, params=None):
        nonlocal calls
        calls += 1
        state["requests"] += 1
        response = requests.get(url, params=params, timeout=45)
        response.raise_for_status()
        return response.json()

    for season in seasons:
        key = str(season)
        if key not in state["schedules"] and calls < max_requests:
            payload = fetch("https://statsapi.mlb.com/api/v1/schedule", {
                "sportId": 1, "season": season, "gameType": "R"})
            if "dates" not in payload:
                raise ValueError("Invalid schedule response")
            state["schedules"][key] = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                                       "payload": payload}
            save(checkpoint, state)
        if key not in state["schedules"]:
            continue
        for day in state["schedules"][key]["payload"]["dates"]:
            for game in day["games"]:
                game_id = str(game["gamePk"])
                if game_id in state["games"] or game_id in state["excluded"]:
                    continue
                if calls >= max_requests:
                    break
                payload = fetch(f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live")
                try:
                    normalized = normalize_game(payload)
                    if normalized["game_id"] != int(game_id) or normalized["season"] != season:
                        raise ValueError("identity_mismatch")
                except (KeyError, ValueError, TypeError) as exc:
                    state["excluded"][game_id] = {"reason": str(exc), "season": season}
                else:
                    state["games"][game_id] = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                                                "source_url": f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live",
                                                "source_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                                                "record": normalized}
                save(checkpoint, state)
    save(checkpoint, state)
    selected = [g["record"] for g in state["games"].values() if g["record"]["season"] in seasons]
    result = build_dataset(selected)
    ids = {str(g["gamePk"]) for s in seasons for day in state["schedules"].get(str(s), {}).get("payload", {}).get("dates", []) for g in day["games"]}
    result["audit"].update({"requested_seasons": seasons, "scheduled_games": len(ids),
        "pending_games": len(ids - state["games"].keys() - state["excluded"].keys()),
        "missing_schedules": [s for s in seasons if str(s) not in state["schedules"]],
        "collection_exclusions": {k: v for k, v in state["excluded"].items() if v["season"] in seasons},
        "requests_this_batch": calls})
    result["audit"]["collection_complete"] = not result["audit"]["pending_games"] and not result["audit"]["missing_schedules"]
    for name, value in result.items():
        save(output / (name + ".json"), value)
    return result["audit"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 500 or any(s < 2010 or s >= datetime.now(timezone.utc).year for s in args.seasons):
        parser.error("Use completed seasons since 2010 and 1-500 requests per batch")
    print(json.dumps(collect(args.output, sorted(set(args.seasons)), args.max_requests), indent=2))


if __name__ == "__main__":
    main()
