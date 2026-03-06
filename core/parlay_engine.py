from __future__ import annotations

from itertools import combinations

import pandas as pd


def _american_to_decimal(odds: float) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def generate_parlays(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["parlay_legs", "combined_probability", "combined_odds", "parlay_ev"])

    needed = {"expected_value", "model_probability", "market_probability"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=["parlay_legs", "combined_probability", "combined_odds", "parlay_ev"])

    candidate_bets = df[df["expected_value"] > 0].copy()
    if len(candidate_bets) > 25:
        candidate_bets = candidate_bets.nlargest(25, "expected_value")

    print("Candidate bets:", len(candidate_bets))
    print("Generating parlays...")

    if candidate_bets.empty:
        return pd.DataFrame(columns=["parlay_legs", "combined_probability", "combined_odds", "parlay_ev"])

    if "odds_american" in candidate_bets.columns:
        odds_col = "odds_american"
    elif "odds" in candidate_bets.columns:
        odds_col = "odds"
    else:
        return pd.DataFrame(columns=["parlay_legs", "combined_probability", "combined_odds", "parlay_ev"])

    candidate_bets["decimal_odds"] = pd.to_numeric(candidate_bets[odds_col], errors="coerce").apply(_american_to_decimal)
    candidate_bets = candidate_bets.dropna(subset=["decimal_odds", "model_probability"])
    if candidate_bets.empty:
        return pd.DataFrame(columns=["parlay_legs", "combined_probability", "combined_odds", "parlay_ev"])

    label_cols = [c for c in ["team", "away_team", "home_team"] if c in candidate_bets.columns]

    records = []
    idxs = list(candidate_bets.index)
    max_parlays = 10_000
    limit_reached = False

    for leg_count in (2, 3, 4):
        if len(idxs) < leg_count:
            continue
        for combo in combinations(idxs, leg_count):
            if len(records) >= max_parlays:
                limit_reached = True
                break

            legs = candidate_bets.loc[list(combo)]
            combined_probability = float(legs["model_probability"].prod())
            combined_odds = float(legs["decimal_odds"].prod())
            parlay_ev = (combined_probability * (combined_odds - 1)) - (1 - combined_probability)

            labels = []
            for _, row in legs.iterrows():
                if "team" in label_cols and pd.notna(row.get("team")):
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
                    "combined_odds": combined_odds,
                    "parlay_ev": parlay_ev,
                }
            )

        if limit_reached:
            break

    if not records:
        return pd.DataFrame(columns=["parlay_legs", "combined_probability", "combined_odds", "parlay_ev"])

    out = pd.DataFrame(records).sort_values("parlay_ev", ascending=False).reset_index(drop=True)
    return out
