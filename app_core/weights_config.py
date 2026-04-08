# Hardcoded Global Weights for Probability Blending
# Two-tier system: Kalshi-heavy when Kalshi agrees, fallback when it doesn't

# Tier 1: Kalshi agrees (prob >= 55% for pick side)
KALSHI_WEIGHT = 0.475      # Prediction markets highest signal
MARKET_WEIGHT = 0.20      # Bookmaker odds
ML_MODEL_WEIGHT = 0.15    # Historical model
THEOVER_WEIGHT = 0.125     # TheOver consensus
SENTIMENT_WEIGHT = 0.05   # News sentiment

# Market Maturity Overrides (MLB/NHL)
LOW_LIQUIDITY_KALSHI_WEIGHT = 0.30
LOW_LIQUIDITY_ML_MODEL_WEIGHT = 0.35

# Tier 2: Fallback weights (Kalshi disagrees or unavailable)
FALLBACK_MARKET_WEIGHT = 0.35
FALLBACK_ML_WEIGHT = 0.35
FALLBACK_THEOVER_WEIGHT = 0.20
FALLBACK_SENTIMENT_WEIGHT = 0.10

# Baseline Thresholds
BASELINE_MIN_EV = 0.01
BASELINE_MIN_EDGE = 0.02

# Stricter Total Over Thresholds
TOTAL_OVER_MIN_EV = 0.03
TOTAL_OVER_MIN_EDGE = 0.04

# NBA Star-Active Adjustments
NBA_STAR_ACTIVE_TOTAL_OVER_BOOST = 0.01
NBA_STAR_ACTIVE_TOTAL_UNDER_PENALTY = -0.01
