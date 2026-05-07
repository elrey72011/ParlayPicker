# Hardcoded Global Weights for Probability Blending
# Two-tier system: Kalshi-heavy when Kalshi agrees, fallback when it doesn't

# Tier 1: Kalshi agrees (prob >= 55% for pick side)
KALSHI_WEIGHT = 0.40       # Prediction markets highest signal
MARKET_WEIGHT = 0.20       # Bookmaker odds
ML_MODEL_WEIGHT = 0.15     # Historical model
THEOVER_WEIGHT = 0.20      # TheOver consensus (raised; Kalshi over-dominance reduced)
SENTIMENT_WEIGHT = 0.05    # News sentiment

# Market Maturity Overrides — MLB only (moderate Kalshi liquidity)
# Reduces Kalshi weight, boosts ML to compensate for thinner market depth.
LOW_LIQUIDITY_KALSHI_WEIGHT = 0.30
LOW_LIQUIDITY_ML_MODEL_WEIGHT = 0.35

# NHL Tier-1 overrides — Kalshi is least reliable for hockey; ML + market dominate.
NHL_KALSHI_WEIGHT = 0.22
NHL_ML_MODEL_WEIGHT = 0.42
NHL_MARKET_WEIGHT = 0.22
NHL_THEOVER_WEIGHT = 0.09
NHL_SENTIMENT_WEIGHT = 0.05

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

# Total Win Probability Floors
TOTAL_MIN_WIN_PROB = 0.56
TOTAL_UNDER_MIN_WIN_PROB = 0.62
TOTAL_UNDER_MIN_EV = 0.18
TOTAL_UNDER_MIN_EDGE = 0.10
NHL_TOTAL_EXTRA_EDGE_PENALTY = 0.01
NHL_TOTAL_MIN_WIN_PROB = 0.57
NHL_TOTAL_MIN_WIN_PROB_STRICT = 0.58
NBA_TOTAL_MIN_WIN_PROB = 0.62
NBA_TOTAL_MIN_EV = 0.02
NBA_TOTAL_MIN_EDGE = 0.03
NHL_TOTAL_MIN_EV_STRICT = 0.01
NHL_TOTAL_MIN_EDGE_STRICT = 0.02
FALLBACK_HEAVY_TOTAL_EV_MULTIPLIER = 0.85

# Divergence guardrail thresholds — per league (ML vs Kalshi gap to trigger cap)
# Thresholds reflect market liquidity: liquid markets (NBA) trust Kalshi more,
# thin markets (NHL) tolerate larger gaps before capping as High Variance.
KALSHI_DIVERGENCE_THRESHOLD = 0.20          # Default for unlisted leagues
KALSHI_DIVERGENCE_THRESHOLD_NBA = 0.25      # Very liquid — tighter cap
KALSHI_DIVERGENCE_THRESHOLD_MLB = 0.22      # Moderate liquidity
KALSHI_DIVERGENCE_THRESHOLD_NHL = 0.30      # Thin market — ML more trusted

# Spread divergence override (allows Actionable despite divergence if strong signal)
SPREAD_DIVERGENCE_OVERRIDE_MIN_PROB = 0.55
SPREAD_DIVERGENCE_OVERRIDE_MIN_EV = 0.03
SPREAD_DIVERGENCE_OVERRIDE_MIN_EDGE = 0.04

# Divergent picks viability floor — min quality to surface as High Variance vs No Play
DIVERGENCE_HIGH_VARIANCE_MIN_EV = 0.03
DIVERGENCE_HIGH_VARIANCE_MIN_EDGE = 0.02
DIVERGENCE_HIGH_VARIANCE_MIN_PROB = 0.53

# Side Minimum Win Probability
SIDE_MIN_WIN_PROB = 0.52
MLB_SPREAD_MIN_WIN_PROB = 0.53
MLB_SPREAD_ACTIONABLE_BONUS = 0.00
MLB_SPREAD_EXTRA_ACTIONABLE_PENALTY = 0.01
MLB_SPREAD_ACTIONABLE_PENALTY = 0.03
MLB_SPREAD_FINALIST_SCORE_PENALTY = 0.05
NBA_SIDE_ACTIONABLE_BONUS = 0.01
NBA_OVER_ACTIONABLE_BONUS = 0.00
# Checked against POST-SHRINKAGE win probability (shrinkage is now applied in the
# gating loop before this threshold is evaluated). At MLB_TOTAL_OVER_PROB_SHRINK=0.70,
# this 0.62 post-shrink threshold requires a raw calibrated_probability of ~0.674+.
MLB_OVER_ACTIONABLE_MIN_PROB = 0.62
MLB_OVER_ACTIONABLE_MIN_EV = 0.07
MLB_OVER_ACTIONABLE_MIN_EDGE = 0.08

# Cold-Market Penalty Layer (by League + Market Type)
MLB_TOTAL_OVER_ACTIONABLE_PENALTY = 0.02
MLB_TOTAL_UNDER_ACTIONABLE_PENALTY = 0.03
NBA_TOTAL_OVER_ACTIONABLE_PENALTY = 0.02
NBA_TOTAL_UNDER_ACTIONABLE_PENALTY = 0.02
NHL_TOTAL_OVER_ACTIONABLE_PENALTY = 0.02
NHL_TOTAL_UNDER_ACTIONABLE_PENALTY = 0.03

# No-Kalshi totals are treated as lower confidence in selection stage
NO_KALSHI_TOTAL_EXTRA_PENALTY = 0.02
NO_KALSHI_TOTAL_UNDER_EXTRA_PENALTY = 0.03

# Cross-family selection nudge to avoid unders dominating finalists on EV/edge alone
TOTAL_UNDER_FINALIST_SCORE_PENALTY = 0.10

# Static empirical hooks (league + family) for later recap-driven calibration.
# Values are additive threshold bumps applied to both EV and edge in selection gating.
LEAGUE_MARKET_FAMILY_ACTIONABLE_PENALTIES = {
    ("MLB", "over"): 0.02,
    ("MLB", "under"): 0.02,
    ("MLB", "side"): 0.00,
    ("NBA", "over"): 0.01,
    ("NBA", "under"): 0.01,
    ("NBA", "side"): 0.00,
    ("NHL", "over"): 0.01,
    ("NHL", "under"): 0.02,
    ("NHL", "side"): 0.00,
}

# Model-health guardrail for noisy slates
FALLBACK_HEAVY_TOTAL_EXTRA_PENALTY = 0.01

# Profiles
BEST_PICKS_PROFILE = 'STANDARD'

# Consensus-aware Actionable overlays
NEUTRAL_ACTIONABLE_MIN_PROB = 0.60
NEUTRAL_ACTIONABLE_MIN_EV = 0.04
NEUTRAL_ACTIONABLE_MIN_EDGE = 0.05

DISAGREES_ACTIONABLE_MIN_PROB = 0.62
DISAGREES_ACTIONABLE_MIN_EV = 0.05
DISAGREES_ACTIONABLE_MIN_EDGE = 0.06

# Line Locking Configuration
LOCK_UPLOAD_LINES_FOR_MATCHED_ROWS = False
ALLOW_UPLOAD_TOTAL_FALLBACK_ACTIONABLE = False

# Production-card calibration guards (totals concentration + overconfidence control)
MAX_TOTAL_OVER_ACTIONABLE_SHARE = 0.60
MAX_TOTAL_OVER_ACTIONABLE_COUNT = 4
MAX_MLB_TOTAL_OVER_ACTIONABLE_COUNT = 3
TOTAL_OVER_PROB_SHRINK = 0.60
MLB_TOTAL_OVER_PROB_SHRINK = 0.70
MLB_TOTAL_OVER_MIN_PRODUCTION_WIN_PROB = 0.59
MLB_TOTAL_OVER_MIN_PRODUCTION_EV = 0.09
MLB_TOTAL_OVER_MIN_PRODUCTION_EDGE = 0.06
DEGRADED_FEATURE_KELLY_MULTIPLIER = 0.50
DEGRADED_FEATURE_MAX_SLATE_EXPOSURE_PCT = 0.12
DEGRADED_FEATURE_MAX_PICK_EXPOSURE_PCT = 0.02
ALLOW_EMPTY_CARD_RECOVERY = True
ENABLE_EMPTY_CARD_RECOVERY = True
EMPTY_CARD_RECOVERY_MAX_PICKS = 2
EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EV = 0.07
EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EDGE = 0.05
EMPTY_CARD_RECOVERY_MIN_PRODUCTION_WIN_PROB = 0.57
EMPTY_CARD_RECOVERY_EXCLUDE_MARKET_TYPES = []
EMPTY_CARD_RECOVERY_EXCLUDE_SOURCES = ["rejected_live"]
EMPTY_CARD_RECOVERY_MAX_KELLY_TOTAL_PCT = 0.05
EMPTY_CARD_RECOVERY_MAX_KELLY_PER_PICK_PCT = 0.025
ALLOW_MLB_TOTAL_OVER_EMPTY_CARD_RECOVERY = False

# Kelly Bet Sizing — Tiered Bankroll Allocation
# The 70/30 ratio means non-Actionable picks use 30% of the Kelly fraction
# that an equivalent Actionable pick would use (not a slate-level budget cap).
# This keeps per-pick amounts stable regardless of how many Actionable picks
# are on a given slate, while still expressing the confidence differential.
#
# Actionable:     0.25x fractional Kelly, 4% bankroll cap per pick
# High Variance:  0.075x fractional Kelly (30% of Actionable rate), 2% cap per pick
# Below Threshold:0.050x fractional Kelly (20% of Actionable rate), 1.5% cap per pick
# No Play:        $0
#
# Slate-level safety: non-Actionable total is capped at 30% of combined
# (Actionable + non-Actionable) if it would otherwise exceed that share.
ACTIONABLE_KELLY_SHARE = 0.70
NON_ACTIONABLE_KELLY_SHARE = 0.30
HIGH_VARIANCE_KELLY_FRACTION = 0.075     # 30% of Actionable's 0.25
BELOW_THRESHOLD_KELLY_FRACTION = 0.050  # 20% of Actionable's 0.25
NON_ACTIONABLE_MAX_PICK_PCT = 0.02      # 2% bankroll ceiling per non-Actionable pick
NON_ACTIONABLE_BELOW_THRESHOLD_MAX_PICK_PCT = 0.015  # 1.5% cap for Below Threshold

# Injury & Weather Adjustments
# Applied per key injured player to the side's model probability.
INJURY_PROB_PENALTY_PER_KEY_PLAYER = 0.015   # 1.5% per key player out
INJURY_KEY_PLAYER_THRESHOLD = 1              # minimum injuries to trigger adjustment
WEATHER_TOTAL_OVER_PENALTY = 0.025           # MLB outdoor bad weather suppresses overs by 2.5%
