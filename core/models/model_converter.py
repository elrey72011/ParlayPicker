import logging
from typing import Iterable, List

import pandas as pd

from core.models.analysis_models import BettingAnalysis
from core.models.game_models import Game
from core.models.prediction_models import Prediction

logger = logging.getLogger(__name__)


def _normalize_game_row(row: dict) -> dict:
    """Normalize common game row aliases before model validation."""
    normalized = dict(row)

    if "odds_american" in normalized and "odds" not in normalized:
        normalized["odds"] = normalized.get("odds_american")

    if "home_team" not in normalized and "home" in normalized:
        normalized["home_team"] = normalized.get("home")

    if "away_team" not in normalized and "away" in normalized:
        normalized["away_team"] = normalized.get("away")

    if "game_id" not in normalized:
        home = str(normalized.get("home_team", ""))
        away = str(normalized.get("away_team", ""))
        league = str(normalized.get("league", ""))
        if home or away:
            normalized["game_id"] = f"{league}:{away}@{home}".strip(":")

    return normalized


def df_to_games(df: pd.DataFrame) -> List[Game]:
    games: List[Game] = []

    for row in df.to_dict(orient="records"):
        try:
            games.append(Game(**_normalize_game_row(row)))
        except Exception as exc:
            logger.warning("Invalid row skipped: %s", exc)

    return games


def games_to_df(games: Iterable[Game]) -> pd.DataFrame:
    return pd.DataFrame([game.model_dump() for game in games])


def df_to_predictions(df: pd.DataFrame) -> List[Prediction]:
    predictions: List[Prediction] = []

    for row in df.to_dict(orient="records"):
        try:
            predictions.append(Prediction(**row))
        except Exception as exc:
            logger.warning("Invalid prediction row skipped: %s", exc)

    return predictions


def predictions_to_df(predictions: Iterable[Prediction]) -> pd.DataFrame:
    return pd.DataFrame([prediction.model_dump() for prediction in predictions])


def analyses_to_df(analyses: Iterable[BettingAnalysis]) -> pd.DataFrame:
    return pd.DataFrame([analysis.model_dump() for analysis in analyses])
