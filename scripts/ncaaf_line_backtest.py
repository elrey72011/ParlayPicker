"""Rebuild the frozen NCAAF research model from CFBD and backtest it on archived 2025 lines (D18 step 1).

    python scripts/ncaaf_line_backtest.py --output OUT.json

Needs CFBD_API_KEY. About 60 CFBD calls (games and team stats for 2023-2025, plus
2025 lines). Writes nothing to Drive or the evidence store.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from app_core import ncaaf_history as history  # noqa: E402
from app_core import ncaaf_line_backtest as bt  # noqa: E402
from app_core import ncaaf_line_model as lm  # noqa: E402
from app_core import ncaaf_research as research  # noqa: E402
from app_core.research_model_recovery import ARTIFACT as FROZEN_ARTIFACT  # noqa: E402


def collect(token, get=requests.get, max_batches=40):
    state = history.new_collection(2026)  # fixed protocol years 2023-2025
    for _ in range(max_batches):
        state, status = history.collect_batch(state, token, get=get)
        print(json.dumps({"collection": status, "batches": len(state["batches"])}), flush=True)
        if status == "requests_complete":
            return state
        if status != "batch_saved":
            raise SystemExit(f"CFBD collection stopped: {status}")
    raise SystemExit("CFBD collection did not finish within the batch limit")


def run(token, get=requests.get):
    state = collect(token, get)
    result = research.run_research(state)
    artifact = result["artifact"]
    _, features, targets = history.build_dataset(state)
    splits, _ = research.prepare(features, targets)
    holdout = splits[2025]
    lines = bt.select_lines(bt.fetch_lines(2025, token, get))
    # Market-anchored model (C5): fit on 2023, centre/scale on 2024, score 2025 once.
    all_lines = {**bt.select_lines(bt.fetch_lines(2023, token, get)),
                 **bt.select_lines(bt.fetch_lines(2024, token, get)), **lines}
    line_aware = {}
    for market in ("SPREAD", "TOTAL"):
        model = lm.fit(splits[2023], splits[2024], all_lines, market)
        line_aware[market] = lm.evaluate(model, holdout, all_lines) if model else {"games": 0, "blocker": "TOO_FEW_LINED_GAMES"}
        if model:
            line_aware[market].update(train_games=model["train_games"], calibration_games=model["calibration_games"],
                                      sigma=model["sigma"])
    return {"line_aware": line_aware,"artifact_hash": result["report"]["artifact_hash"],
            "artifact_matches_frozen": result["report"]["artifact_hash"] == FROZEN_ARTIFACT,
            "frozen_artifact_hash": FROZEN_ARTIFACT,
            "holdout_games": len(holdout),
            "games_with_lines": sum(1 for row in holdout if row["game_id"] in lines),
            "providers": sorted({v["provider"] for v in lines.values() if v["provider"]}),
            "score_metrics": result["report"]["holdout_metrics"],
            "backtest": bt.backtest(artifact, holdout, lines)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    token = os.getenv("CFBD_API_KEY", "").strip()
    if not token:
        raise SystemExit("MISSING_CREDENTIALS: CFBD_API_KEY")
    result = run(token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = bt.markdown(result)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
