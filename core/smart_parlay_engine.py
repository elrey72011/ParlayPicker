from __future__ import annotations

from itertools import combinations

import pandas as pd


def generate_smart_parlays(df: pd.DataFrame) -> pd.DataFrame:
    """Generate +EV parlays from the top edge candidates."""
    columns = ["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    needed = {"edge", "calibrated_probability", "decimal_odds"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    candidate_bets = df.nlargest(20, "edge").copy()
    candidate_bets = candidate_bets.dropna(subset=["calibrated_probability", "decimal_odds"])
    if candidate_bets.empty:
        return pd.DataFrame(columns=columns)

    label_cols = [c for c in ["best_pick", "team", "away_team", "home_team"] if c in candidate_bets.columns]
    records: list[dict[str, object]] = []

    for leg_count in (2, 3, 4):
        if len(candidate_bets) < leg_count:
            continue

        for combo in combinations(candidate_bets.index, leg_count):
            legs = candidate_bets.loc[list(combo)]
            combined_probability = float(legs["calibrated_probability"].prod())
            combined_decimal_odds = float(legs["decimal_odds"].prod())
            parlay_ev = (combined_probability * (combined_decimal_odds - 1)) - (1 - combined_probability)

            if parlay_ev <= 0:
                continue

            labels: list[str] = []
            for _, row in legs.iterrows():
                if "best_pick" in label_cols and pd.notna(row.get("best_pick")) and str(row.get("best_pick")).strip():
                    labels.append(str(row["best_pick"]))
                elif "team" in label_cols and pd.notna(row.get("team")):
                    labels.append(str(row["team"]))
                elif {"away_team", "home_team"}.issubset(label_cols):
                    labels.append(f"{row['away_team']} vs {row['home_team']}")
                elif "away_team" in label_cols:
                    labels.append(str(row["away_team"]))
                else:
                    labels.append("leg")

            records.append(
                {
                    "parlay_legs": " | ".join(labels),
                    "combined_probability": combined_probability,
                    "combined_decimal_odds": combined_decimal_odds,
                    "parlay_ev": parlay_ev,
                    "legs": leg_count,
                }
            )

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records).sort_values("parlay_ev", ascending=False).reset_index(drop=True)
