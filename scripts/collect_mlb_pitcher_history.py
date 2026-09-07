"""Bounded actual-starter collection for 2023/2024 development only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core.mlb_history import build_dataset
from app_core.mlb_pitcher_history import parse_boxscore, enrich
from scripts.collect_mlb_history import save


def collect(checkpoint, output, max_requests=20):
    if not 1 <= max_requests <= 500:
        raise ValueError("Use 1-500 requests per batch")
    raw = checkpoint.read_bytes()
    source = json.loads(raw)
    games = {int(k): v["record"] for k, v in source["games"].items() if v["record"]["season"] in (2023, 2024)}
    if {g["season"] for g in games.values()} != {2023, 2024}:
        raise ValueError("Both development seasons are required")
    scheduled = {str(g["gamePk"]) for y in (2023, 2024) for d in source["schedules"][str(y)]["payload"]["dates"] for g in d["games"]}
    if scheduled - source["games"].keys() - source["excluded"].keys():
        raise ValueError("Complete source history first")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "pitcher-checkpoint.json"
    source_hash = hashlib.sha256(raw).hexdigest()
    state = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1, "source_hash": source_hash, "boxes": {}, "excluded": {}}
    if state.get("schema_version") != 1 or state["source_hash"] != source_hash:
        raise ValueError("Checkpoint source changed; use a separate output directory")
    calls = 0
    for gid, game in sorted(games.items()):
        key = str(gid)
        if key in state["boxes"] or key in state["excluded"]:
            continue
        if calls >= max_requests:
            break
        url = f"https://statsapi.mlb.com/api/v1/game/{gid}/boxscore"
        calls += 1
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        payload = response.json()
        try:
            box = parse_boxscore(payload, game)
        except (ValueError, KeyError, TypeError) as exc:
            state["excluded"][key] = str(exc)
        else:
            state["boxes"][key] = {"data": box, "source_url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
        save(path, state)
    dataset = build_dataset(list(games.values()))
    rows, exclusions = enrich(dataset["features"], games, {k:v["data"] for k,v in state["boxes"].items()})
    pending = len(games) - len(state["boxes"]) - len(state["excluded"])
    audit = {"pending": pending, "collection_complete": pending == 0, "accepted_boxscores": len(state["boxes"]),
             "eligible": len(rows), "feature_exclusions": exclusions, "boxscore_exclusions": state["excluded"],
             "research_only": True, "production_eligible": False,
             "limitation": "Actual starters are retrospective; pregame announcement and historical correction times are unverified."}
    save(output / "features.json", rows)
    save(output / "audit.json", audit)
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-requests", type=int, default=20)
    a = p.parse_args()
    print(json.dumps(collect(a.checkpoint, a.output, a.max_requests), indent=2))
