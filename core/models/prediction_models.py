from pydantic import BaseModel, ConfigDict


class Prediction(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    game_id: str

    ai_probability: float = 0.5
    ml_probability: float = 0.5
    market_probability: float = 0.5
