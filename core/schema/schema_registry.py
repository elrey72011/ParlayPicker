"""Schema definitions for each ParlayPicker pipeline stage."""

SCHEMAS = {
    "raw_odds": [
        "game_id",
        "home_team",
        "away_team",
        "league",
        "odds",
        "spread",
        "total",
    ],
    "feature_engineering": [
        "game_id",
        "home_team",
        "away_team",
        "league",
        "point_diff",
        "off_rating",
        "def_rating",
    ],
    "model_predictions": [
        "game_id",
        "ai_probability",
        "ml_probability",
    ],
    "master_analysis": [
        "game_id",
        "ai_probability",
        "ml_probability",
        "market_probability",
        "expected_value",
    ],
}
