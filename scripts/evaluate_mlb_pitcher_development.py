"""Development comparison only; never reads the already evaluated 2025 season."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core.mlb_history import build_dataset
from app_core.mlb_pitcher_history import enrich, PITCHER_FEATURES
from app_core.mlb_research import FEATURES


def evaluate(source, pitchers):
    if pitchers["source_hash"] != hashlib.sha256(source).hexdigest():
        raise ValueError("Source mismatch")
    state = json.loads(source)
    games = {int(k):v["record"] for k,v in state["games"].items() if v["record"]["season"] in (2023,2024)}
    if set(map(str,games)) - pitchers["boxes"].keys() - pitchers["excluded"].keys():
        raise ValueError("Complete pitcher collection before comparison")
    rows, exclusions = enrich(build_dataset(list(games.values()))["features"], games,
                              {k:v["data"] for k,v in pitchers["boxes"].items()})
    train = [r for r in rows if r["season"] == 2023]
    validation = [r for r in rows if r["season"] == 2024]
    if min(len(train),len(validation)) < 100:
        raise ValueError("At least 100 matched games per development season required")
    metrics = {}
    for name, columns in [("team_only", FEATURES), ("team_and_starter", FEATURES + PITCHER_FEATURES)]:
        x = np.array([[r[k] for k in columns] for r in train])
        v = np.array([[r[k] for k in columns] for r in validation])
        mean, scale = x.mean(0), x.std(0)
        scale[scale == 0] = 1
        x, v = (x-mean)/scale, (v-mean)/scale
        metrics[name] = {}
        for target in ("margin", "total"):
            def outcomes(data):
                return np.array([games[r["game_id"]]["home_score"] + (1 if target == "total" else -1)*games[r["game_id"]]["away_score"] for r in data])
            y, actual = outcomes(train), outcomes(validation)
            coefficients = np.linalg.solve(x.T@x + 10*np.eye(len(columns)), x.T@(y-y.mean()))
            predictions = v@coefficients+y.mean()
            if target == "total": predictions = np.maximum(0,predictions)
            errors = predictions-actual
            metrics[name][target] = {"mae":float(np.abs(errors).mean()), "rmse":float(np.sqrt((errors**2).mean()))}
    return {"train_games":len(train), "development_games":len(validation), "metrics":metrics,
            "feature_exclusions":exclusions, "production_eligible":False,
            "interpretation":"Development comparison, not an untouched test. No win probabilities or betting returns estimated.",
            "protocol":"2023 fit; 2024 development; fixed ridge alpha 10; identical eligible games; no 2025 evaluation."}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("history",type=Path);p.add_argument("pitchers",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=evaluate(a.history.read_bytes(),json.loads(a.pitchers.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
