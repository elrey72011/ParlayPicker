# Hardcoded Global Weights for Probability Blending
# Two-tier system: Kalshi-heavy when Kalshi agrees, fallback when it doesn't

# Tier 1: Kalshi agrees (prob >= 55% for pick side)
KALSHI_WEIGHT = 0.55      # Prediction markets highest signal
MARKET_WEIGHT = 0.15      # Bookmaker odds
ML_MODEL_WEIGHT = 0.125    # Historical model
THEOVER_WEIGHT = 0.125     # TheOver consensus
SENTIMENT_WEIGHT = 0.05   # News sentiment

# Tier 2: Fallback weights (Kalshi disagrees or unavailable)
FALLBACK_MARKET_WEIGHT = 0.35
FALLBACK_ML_WEIGHT = 0.35
FALLBACK_THEOVER_WEIGHT = 0.20
FALLBACK_SENTIMENT_WEIGHT = 0.10
