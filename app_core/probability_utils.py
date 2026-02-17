from typing import Optional, Any

def american_to_implied_prob(odds: Any) -> Optional[float]:
    """
    Convert American odds to implied probability with defensive caps for extreme values.

    Extreme odds (|odds| > 900) are capped to prevent unrealistic probabilities (>0.90 or <0.10).
    This helps NHL and other leagues with heavy favorites avoid probability collisions.

    Formula:
    - Positive odds (+X): probability = 100 / (X + 100)
    - Negative odds (-X): probability = X / (X + 100)

    Args:
        odds: American odds (e.g., -110, +150, 100)

    Returns:
        Probability between 0.0 and 1.0, or None if invalid
    """
    if odds is None:
        return None

    try:
        o = float(odds)
    except (ValueError, TypeError):
        return None

    # Handle edge cases
    if o == 0:
        return None  # Invalid odds

    # Cap extreme odds to prevent unrealistic probabilities
    # -900 converts to ~0.90, which is more reasonable than -990 -> 0.99
    if o < -900:
        o = -900
    elif o > 900:
        o = 900

    if o > 0:
        # Positive odds (underdog)
        return 100.0 / (o + 100.0)
    else:
        # Negative odds (favorite)
        # -110 means you bet $110 to win $100
        return abs(o) / (abs(o) + 100.0)

# Alias for compatibility with both naming conventions
american_to_implied = american_to_implied_prob
