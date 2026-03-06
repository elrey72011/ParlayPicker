from __future__ import annotations

import pandas as pd


def generate_parlays(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "expected_value" not in df.columns:
        return pd.DataFrame()

    df = df[df["expected_value"] > 0.02]
    df = df.sort_values("expected_value", ascending=False)

    picks = []
    used_teams = set()

    for _, row in df.iterrows():
        team = row["team"] if "team" in row.index else row.get("away_team")
        if team in used_teams:
            continue

        if {"home_team", "away_team"}.issubset(df.columns):
            if row["home_team"] in used_teams or row["away_team"] in used_teams:
                continue
            used_teams.add(row["home_team"])
            used_teams.add(row["away_team"])
        else:
            used_teams.add(team)

        picks.append(row)
        if len(picks) == 5:
            break

    return pd.DataFrame(picks)
