import pandas as pd


def normalize_odds(df: pd.DataFrame) -> pd.DataFrame:
    possible_cols = [
        "odds_american",
        "price_american",
        "home_spread_price_american",
        "away_spread_price_american",
        "over_price_american",
        "under_price_american",
    ]

    for col in possible_cols:
        if col in df.columns:
            df["odds_american"] = df[col]
            return df

    df["odds_american"] = -110

    return df
