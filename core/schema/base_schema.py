BASE_GAME_COLUMNS = [
    "game_id",
    "league",
    "home_team",
    "away_team",
    "odds_american",
    "ai_probability",
    "ml_probability",
    "market_probability",
    "consensus_prob",
    "expected_value",
]


def ensure_base_schema(df):
    for col in BASE_GAME_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df
