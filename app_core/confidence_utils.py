
import math

def assign_confidence(prob):
    """
    Deterministic confidence assignment based on probability.
    >= 0.85 -> HIGH
    >= 0.70 -> MEDIUM
    < 0.70 -> LOW
    """
    if prob is None:
        return "LOW"
    try:
        p = float(prob)
    except:
        return "LOW"

    if p >= 0.85:
        return "HIGH"
    elif p >= 0.70:
        return "MEDIUM"
    else:
        return "LOW"
