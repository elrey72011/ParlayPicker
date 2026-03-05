from typing import Optional

from pydantic import BaseModel, ConfigDict


class Game(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    game_id: str
    league: Optional[str] = None

    home_team: str
    away_team: str

    odds: Optional[float] = None
    spread: Optional[float] = None
    total: Optional[float] = None
