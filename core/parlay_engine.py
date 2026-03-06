from __future__ import annotations

import pandas as pd


def generate_parlays(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "expected_value" not in df.columns:
        return pd.DataFrame()

    df = df[df["expected_value"] > 0.02].copy()
    df = df.sort_values("expected_value", ascending=False)

    # Correlation guard: avoid duplicate team exposure.
    if "team" in df.columns:
        df = df.drop_duplicates(subset=["team"], keep="first")
    elif {"away_team", "home_team"}.issubset(df.columns):
        used = set()
        kept_rows = []
        for idx, row in df.iterrows():
            teams = {row["away_team"], row["home_team"]}
            if used.intersection(teams):
                continue
            used.update(teams)
            kept_rows.append(idx)
        df = df.loc[kept_rows]

    return df.head(20)
