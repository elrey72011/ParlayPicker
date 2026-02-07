# Hardcoded Global Weights for Probability Blending
# User Instruction: "Stop the dynamic weight shifting. Use these static values."

KALSHI_WEIGHT = 0.55      # UP from 0.35 - Prediction markets highest signal
MARKET_WEIGHT = 0.15      # DOWN from 0.30 - Bookmaker odds include vig
ML_MODEL_WEIGHT = 0.15    # DOWN from 0.20 - Historical model
THEOVER_WEIGHT = 0.10     # SAME
SENTIMENT_WEIGHT = 0.05   # SAME
