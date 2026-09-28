"""Explicit conversion between conditional and unconditional push-aware mass."""
import math


def _probability(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0 <= result <= 1 else None


def unconditional_from_conditional(p_win_given_decided, p_push):
    """Return unconditional win/push/loss mass from supported conditional input.

    Push mass is always supplied per candidate.  This helper deliberately has no
    cohort default because an aggregate push rate is not a settlement contract.
    """
    decided_win = _probability(p_win_given_decided)
    push = _probability(p_push)
    if decided_win is None or push is None or push >= 1:
        return None
    decided_mass = 1.0 - push
    win = decided_mass * decided_win
    loss = decided_mass * (1.0 - decided_win)
    if not all(math.isfinite(value) for value in (win, push, loss)):
        return None
    return {"p_win": win, "p_push": push, "p_loss": loss}


def conditional_probabilities(row, probability_column="calibrated_probability", market_column="market_probability"):
    """Return conditional model/market probabilities, or None when unverified.

    Unconditional inputs require separately recorded model and market push mass.
    Never estimate push mass from odds, a line, or a settled outcome.
    """
    def value(column):
        return _probability(row.get(column))
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
