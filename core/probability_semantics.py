"""Explicit conversion of recorded push-aware probabilities for binary scoring."""
import math


def conditional_probabilities(row, probability_column="calibrated_probability", market_column="market_probability"):
    """Return conditional model/market probabilities, or None when unverified.

    Unconditional inputs require separately recorded model and market push mass.
    Never estimate push mass from odds, a line, or a settled outcome.
    """
    def value(column):
        try:
            result = float(row.get(column))
            return result if math.isfinite(result) and 0 <= result <= 1 else None
        except (TypeError, ValueError):
            return None
    model, market = value(probability_column), value(market_column)
    if model is None or market is None:
        return None
    semantics = str(row.get("probability_semantics", ""))
    if semantics == "win_conditional_on_decision":
        return model, market
    if semantics != "win_unconditional_with_push":
        return None
    push, market_push = value("push_probability"), value("market_push_probability")
    if push is None or market_push is None or push >= 1 or market_push >= 1:
        return None
    if model + push > 1 or market + market_push > 1:
        return None
    return model / (1 - push), market / (1 - market_push)
