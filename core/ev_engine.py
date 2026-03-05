from core.models.prediction_models import Prediction


def calculate_ev(probability, american_odds):
    if probability is None:
        probability = 0.5

    if american_odds > 0:
        payout = american_odds / 100
    else:
        payout = 100 / abs(american_odds)

    ev = probability * payout - (1 - probability)

    return ev


def compute_ev(prediction: Prediction) -> float:
    """Typed EV helper that avoids fragile DataFrame column access."""
    prob = (prediction.ai_probability + prediction.ml_probability) / 2
    return prob - (1 - prob)
