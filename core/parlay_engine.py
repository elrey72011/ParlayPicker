from __future__ import annotations

from itertools import combinations

import pandas as pd

from core.parlay_safety import shares_game
from core.smart_parlay_engine import generate_smart_parlays


_OUTPUT_COLUMNS = [
    "parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev",
    "legs", "unique_game_count", "one_leg_per_game",
]


def _fallback_parlays(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df[df.get("expected_value", 0) > 0].copy()
    if candidates.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    records = []
    for i, j in combinations(candidates.index, 2):
        left = candidates.loc[i]
        right = candidates.loc[j]

        if shares_game(left, right):
            continue

        # simple anti-correlation guard for duplicate away teams
        if str(left.get("away_team", "")).lower() == str(right.get("away_team", "")).lower():
            continue

        left_label = left.get("best_pick") if pd.notna(left.get("best_pick")) and str(left.get("best_pick")).strip() else f"{left.get('away_team', 'leg')} vs {left.get('home_team', 'leg')}"
        right_label = right.get("best_pick") if pd.notna(right.get("best_pick")) and str(right.get("best_pick")).strip() else f"{right.get('away_team', 'leg')} vs {right.get('home_team', 'leg')}"
        labels = f"{left_label} | {right_label}"
        records.append(
            {
                "parlay_legs": labels,
                "combined_probability": 0.0,
                "combined_decimal_odds": 0.0,
                "parlay_ev": float(left.get("expected_value", 0)) + float(right.get("expected_value", 0)),
                "legs": 2,
                "unique_game_count": 2,
                "one_leg_per_game": True,
            }
        )

    return pd.DataFrame(records).sort_values("parlay_ev", ascending=False).reset_index(drop=True) if records else pd.DataFrame(columns=_OUTPUT_COLUMNS)


def generate_parlays(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper around the smart parlay engine."""
    out = generate_smart_parlays(df)
    if out.empty and "expected_value" in df.columns:
        return _fallback_parlays(df)
    return out
