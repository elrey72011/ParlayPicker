from pydantic import BaseModel, ConfigDict


class BettingAnalysis(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    game_id: str

    ai_probability: float
    ml_probability: float
    market_probability: float

    expected_value: float = 0
    bet_signal: bool = False
