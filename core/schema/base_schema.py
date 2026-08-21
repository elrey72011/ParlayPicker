BASE_GAME_COLUMNS = [
    "game_id",
    "league",
    "home_team",
    "away_team",
    "game_date",
    "game_time_est",
    "odds_american",
    "ai_probability",
    "ml_probability",
    "market_probability",
    "consensus_prob",
    "expected_value",
]


def ensure_base_schema(df):
    # Historical collectors use ``sport`` while the live pipeline uses
    # ``league``. Preserve the real identity before adding optional columns;
    # otherwise every historical row receives a blank league and cannot join
    # back to the live schedule.
    if "sport" in df.columns:
        if "league" not in df.columns:
            df["league"] = df["sport"]
        else:
            league = df["league"].astype("string").fillna("").str.strip()
            sport = df["sport"].astype("string").fillna("").str.strip()
            df["league"] = league.where(league.ne(""), sport)
    for col in BASE_GAME_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df
