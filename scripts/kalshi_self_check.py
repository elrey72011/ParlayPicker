"""Quick sanity check for Kalshi matching.

Run with:
    python scripts/kalshi_self_check.py [--games path/to/games.json]

The script reuses the same Kalshi matching logic as the Vertex master analysis
to surface a few rows that include the Kalshi market label, probability, and
edge vs. a provided model win probability.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd

from app_core.kalshi_integrator import KalshiIntegrator
from vertex_master_analyzer import VertexMasterAnalyzer


def _default_games() -> List[Dict[str, Any]]:
    today_iso = (datetime.utcnow() + timedelta(hours=2)).replace(microsecond=0).isoformat() + "Z"
    return [
        {
            "home_team": "Los Angeles Lakers",
            "away_team": "Golden State Warriors",
            "commence_time": today_iso,
            "win_prob": 0.55,
            "sport_key": "basketball_nba",
        },
        {
            "home_team": "New York Knicks",
            "away_team": "Boston Celtics",
            "commence_time": today_iso,
            "win_prob": 0.52,
            "sport_key": "basketball_nba",
        },
    ]


def load_games(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return _default_games()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = data.get("games", [])
        return list(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi market self-check")
    parser.add_argument("--games", help="Path to JSON list of games to evaluate", default=None)
    args = parser.parse_args()

    games = load_games(args.games)
    kalshi = KalshiIntegrator()
    analyzer = VertexMasterAnalyzer(kalshi_integrator=kalshi, use_kalshi=True)

    rows = []
    for game in games:
        feats = analyzer._get_kalshi_features(game)  # noqa: SLF001 - intentional reuse for debugging
        win_prob = game.get("win_prob") or game.get("implied_home_prob") or 0.5
        edge_vs_kalshi = None
        if feats.get("kalshi_available") and feats.get("kalshi_prob") is not None:
            edge_vs_kalshi = (float(win_prob) - float(feats["kalshi_prob"])) * 100
        rows.append(
            {
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "Kalshi": feats.get("kalshi_label") or feats.get("kalshi_match_debug"),
                "Kalshi %": float(feats.get("kalshi_prob") or 0) * 100,
                "Edge vs Kalshi %": edge_vs_kalshi,
                "Kalshi Edge": edge_vs_kalshi,
                "kalshi_match_debug": feats.get("kalshi_match_debug"),
            }
        )

    df = pd.DataFrame(rows)
    matched = df[df["Kalshi %"] > 0]
    print("=== Kalshi Self-Check ===")
    print(f"Games evaluated: {len(df)} | Matches: {len(matched)}")
    if not matched.empty:
        print(matched.head().to_string(index=False))
    else:
        print("No Kalshi matches found. Check API credentials or schedule.")


if __name__ == "__main__":
    main()
