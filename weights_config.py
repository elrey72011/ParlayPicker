# Global Weight Re-Allocation Configuration
# Context: Evolving ParlayPicker into a time-sensitive predictive engine.

# Base weights distribution
_BASE_WEIGHTS = {
    "kalshi_weight": 0.35,   # 35%
    "odds_weight": 0.30,     # 30% (Market)
    "ml_weight": 0.20,       # 20% (Reduced from 25%) (Model)
    "theover_weight": 0.10,  # 10%
    "sentiment_weight": 0.05 # 5% (Activated from 0%)
}

# Export separately for spread, total, and moneyline to allow for future divergence if needed,
# but currently they all share the same profile per instructions.
WEIGHTS_CONFIG = {
    "spread": _BASE_WEIGHTS.copy(),
    "total": _BASE_WEIGHTS.copy(),
    "moneyline": _BASE_WEIGHTS.copy()
}
