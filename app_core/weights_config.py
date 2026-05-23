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

# MLB Totals blending overrides (Tier 1 and Tier 2).
# May-13 review: TheOver was double-counted (pre-mix + separate weight); overs 1-5.
# Fix: removed double-counting; reduced TheOver weight to 0.10 and raised market.
# May-16 correction: TheOver DOES incorporate pitcher data — the flat ~0.85 probs
# observed on May-13 were caused by team name cross-matching bugs (now fixed), not
# bad data quality. Restoring TheOver to 0.25; market eased back to 0.25.
MLB_TOTAL_THEOVER_WEIGHT = 0.25          # restored from 0.10; TheOver has pitcher data
MLB_TOTAL_ML_WEIGHT = 0.10               # unchanged
MLB_TOTAL_MARKET_WEIGHT = 0.25           # eased from 0.30; TheOver now shares signal
MLB_TOTAL_KALSHI_WEIGHT = 0.40           # unchanged
MLB_TOTAL_FALLBACK_THEOVER_WEIGHT = 0.30 # restored from 0.20; pitcher signal valuable in fallback
MLB_TOTAL_FALLBACK_ML_WEIGHT = 0.15      # unchanged
MLB_TOTAL_FALLBACK_MARKET_WEIGHT = 0.35  # eased from 0.40

# NBA Totals blending overrides (Tier 1 and Tier 2).
# TheOver incorporates pace, rest, defensive ratings the ML model lacks.
NBA_TOTAL_THEOVER_WEIGHT = 0.30          # up from standard 0.20
NBA_TOTAL_ML_WEIGHT = 0.12               # down from ML_MODEL_WEIGHT (0.15)
NBA_TOTAL_FALLBACK_THEOVER_WEIGHT = 0.40 # up from FALLBACK_THEOVER (0.20)
NBA_TOTAL_FALLBACK_ML_WEIGHT = 0.15      # down from FALLBACK_ML (0.35)

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
TOTAL_MIN_WIN_PROB = 0.54
TOTAL_UNDER_MIN_WIN_PROB = 0.62
TOTAL_UNDER_MIN_EV = 0.22           # raised 0.18→0.22 after May-16 review: both Actionable unders lost
TOTAL_UNDER_MIN_EDGE = 0.13         # raised 0.10→0.13 after May-16 review
# MLB-specific under win prob floor (MLB unders 0-2 Actionable on May-16; both lost badly)
MLB_TOTAL_UNDER_MIN_WIN_PROB = 0.66  # raised above general 0.62
NHL_TOTAL_EXTRA_EDGE_PENALTY = 0.01
NHL_TOTAL_MIN_WIN_PROB = 0.57
NHL_TOTAL_MIN_WIN_PROB_STRICT = 0.58
# Raised 0.62→0.65 after May-11 review: NBA Under 213.5 was Actionable and lost;
# NBA Unders are 0-3 across May 9-11 with no defensive-pace signal in feature set.
NBA_TOTAL_MIN_WIN_PROB = 0.65
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

# ML contradiction guardrail — totals only
# The XGBoost model predicts home win probability, which has no direct bearing on
# total runs. The old threshold (50%) was blocking good over picks when the ML
# simply predicted the away team to win (e.g., MIA/ATL: ML=40.8%, game went 12-0=12 total).
# Only block a total_over when ML probability is extremely low (< 35%), signalling the
# model strongly expects a low-scoring or blowout-type result.
# Unders are not subject to this guardrail — a low home win prob says nothing about
# whether the game stays under the total.
TOTAL_ML_CONTRADICTION_OVER_MAX_PROB = 0.35

# Side Minimum Win Probability
SIDE_MIN_WIN_PROB = 0.52
MLB_SPREAD_MIN_WIN_PROB = 0.53
# High-EV underdog override: when an MLB spread has very strong EV and edge signals
# (market mispricing the line), allow win_prob as low as MLB_SPREAD_HIGH_EV_MIN_WIN_PROB.
# TEX -1.5 on May-12 illustrates the case: win_prob=0.470, EV=0.274, edge=0.110, WIN.
# The market had Texas at 0.360 implied probability; the model/Kalshi saw 0.470/0.365.
MLB_SPREAD_HIGH_EV_OVERRIDE_MIN_EV = 0.20
MLB_SPREAD_HIGH_EV_OVERRIDE_MIN_EDGE = 0.08
MLB_SPREAD_HIGH_EV_MIN_WIN_PROB = 0.44
MLB_SPREAD_ACTIONABLE_BONUS = 0.00
MLB_SPREAD_EXTRA_ACTIONABLE_PENALTY = 0.01
MLB_SPREAD_ACTIONABLE_PENALTY = 0.03
MLB_SPREAD_FINALIST_SCORE_PENALTY = 0.05
NBA_SIDE_ACTIONABLE_BONUS = 0.01
NBA_OVER_ACTIONABLE_BONUS = 0.00
# Checked against POST-SHRINKAGE win probability (shrinkage is now applied in the
# gating loop before this threshold is evaluated).
# Raised from (0.62 / 0.07 / 0.08) after May-9 review: MLB Overs went 3-7 (30%).
# Further raised from (0.65 / 0.10 / 0.10) after May-10 review: MLB Overs went 2-6 (25%).
# Eased from (0.68 / 0.14 / 0.12) after May-12 recap to (0.56 / 0.07 / 0.03).
# Re-tightened to (0.63 / 0.09 / 0.05) after May-13 recap: overs went 1-5 (20%).
# Eased to (0.60 / 0.07 / 0.04) after May-17 recap: BT overs went 7-2 (78%) while
# team-name bugs are now fixed and TheOver pitcher data flows correctly. Lower the bar
# so strong overs can reach Actionable now that the signal is more reliable.
MLB_OVER_ACTIONABLE_MIN_PROB = 0.60
MLB_OVER_ACTIONABLE_MIN_EV = 0.07
MLB_OVER_ACTIONABLE_MIN_EDGE = 0.04
# Hard cap on calibrated probability for MLB overs — prevents TheOver from inflating
# blended probability above a reliable ceiling (even post double-counting fix).
MLB_OVER_CALIBRATED_PROB_CAP = 0.67

# MLB Under Actionable cap — MLB Unders have gone 0-4 at Actionable across May 16-17.
# Block them from reaching Actionable; cap at High Variance until the ML model's
# under bias is better understood. Set False to re-enable if performance recovers.
MLB_UNDER_ACTIONABLE_CAP = True

# NHL Under Actionable cap — CAR/MTL Under 5.5 went 8 total goals at Actionable on May 21.
# Same pattern as MLB unders: model overconfident on unders in lower-scoring sport contexts.
# Cap at High Variance until NHL under performance demonstrates reliability.
NHL_UNDER_ACTIONABLE_CAP = True

# Cold-Market Penalty Layer (by League + Market Type)
MLB_TOTAL_OVER_ACTIONABLE_PENALTY = 0.00
# MLB Under raised 0.03→0.05 after May-11 review: LA/SF Under 9.5 lost (12 runs scored).
# Further raised 0.05→0.07 after May-16 review: both Actionable unders lost (11 runs each).
MLB_TOTAL_UNDER_ACTIONABLE_PENALTY = 0.07
NBA_TOTAL_OVER_ACTIONABLE_PENALTY = 0.02
# NBA Under raised 0.02→0.05 after May-11 review: NBA Unders 0-3 across May 9-11.
NBA_TOTAL_UNDER_ACTIONABLE_PENALTY = 0.05
NHL_TOTAL_OVER_ACTIONABLE_PENALTY = 0.02
NHL_TOTAL_UNDER_ACTIONABLE_PENALTY = 0.03

# High total line penalty — MLB overs with a very high line (≥11.0) have consistently
# underperformed: COL/ARI Over 11.5 lost on both May-15 and May-16 (6 and 10 runs scored).
MLB_HIGH_TOTAL_LINE_THRESHOLD = 11.0
MLB_HIGH_TOTAL_LINE_OVER_PENALTY = 0.03  # added to req_ev and req_edge

# Mid-range total line penalty — MLB overs in the 9.5–10.9 range have underperformed.
# May 21: ARI/COL Over 9.5 went only 3 total runs at Actionable.
# Adds a smaller penalty tier between the base gate and the ≥11.0 extreme-line penalty.
MLB_MID_TOTAL_LINE_THRESHOLD = 9.5
MLB_MID_TOTAL_LINE_OVER_PENALTY = 0.02  # added to req_ev and req_edge

# Low total line floor — MLB overs with a line below 8.0 are pitcher-friendly games
# where the over rarely hits. May 20: CHC/MIL Over 6.5 (5 total), SD/LAD Over 7.5
# (4 total) both lost. May 22: MIA/NYM Over 7.5 (3 total), CHC/HOU Over 7.5 (6 total).
# Changed from High Variance to No Play — these kept appearing in the pick list and losing.
MLB_OVER_MIN_TOTAL_LINE = 8.0

# Low total line floor for MLB unders — Under 7.5 went 1-3 on May 22 (Angels, Boston,
# SF all lost badly with 13-15 total runs; only Yankees 7.5 under held). Sub-8.0 under
# lines sit in the dangerous mid-range where a few extra hits easily blow the total.
MLB_UNDER_MIN_TOTAL_LINE = 8.0

# No-Kalshi totals are treated as lower confidence in selection stage
NO_KALSHI_TOTAL_EXTRA_PENALTY = 0.02
NO_KALSHI_TOTAL_UNDER_EXTRA_PENALTY = 0.03

# Cross-family selection nudge to avoid unders dominating finalists on EV/edge alone
TOTAL_UNDER_FINALIST_SCORE_PENALTY = 0.10

# Static empirical hooks (league + family) for later recap-driven calibration.
# Values are additive threshold bumps applied to both EV and edge in selection gating.
LEAGUE_MARKET_FAMILY_ACTIONABLE_PENALTIES = {
    # MLB Over raised 0.02→0.04 after May-9 review (3-7, 30% hit rate).
    # Further raised 0.04→0.07 after May-10 review (2-6, 25% hit rate).
    # Eased 0.07→0.01 after May-12 review (Below Threshold MLB Overs 4-4, 50%);
    # TheOver is now wired into blending and carries the pitcher-quality signal.
    ("MLB", "over"): 0.01,
    # MLB Under raised 0.02→0.04 after May-11 review (LA/SF Under 9.5, High Variance, LOSS).
    # Further raised 0.04→0.06 after May-16 review: both Actionable unders lost (11 runs each).
    ("MLB", "under"): 0.06,
    ("MLB", "side"): 0.00,
    ("NBA", "over"): 0.01,
    # NBA Under raised 0.01→0.05 after May-11 review (0-3 across May 9-11).
    ("NBA", "under"): 0.05,
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
# May-9 review: MLB Overs 3-7 (30%). Cap reduced 3→2; shrink tightened 0.70→0.65;
# production thresholds raised to require stronger signal before an Over is Actionable.
# May-10 review: MLB Overs 2-6 (25%). Cap reduced 2→1; shrink tightened 0.65→0.55;
# production thresholds raised further; gating thresholds raised across the board.
# May-12 review: Below Threshold MLB Overs went 4-4 (50%); TheOver now wired into
# blending. MLB shrink eased 0.55→0.85 (double-penalizing post-TheOver). Cap raised
# 1→2 and production thresholds eased to allow legitimate winners through.
MAX_TOTAL_OVER_ACTIONABLE_SHARE = 0.50
MAX_TOTAL_OVER_ACTIONABLE_COUNT = 3
MAX_MLB_TOTAL_OVER_ACTIONABLE_COUNT = 2
TOTAL_OVER_PROB_SHRINK = 0.60
MLB_TOTAL_OVER_PROB_SHRINK = 0.85
MLB_TOTAL_OVER_MIN_PRODUCTION_WIN_PROB = 0.60
MLB_TOTAL_OVER_MIN_PRODUCTION_EV = 0.07
MLB_TOTAL_OVER_MIN_PRODUCTION_EDGE = 0.04
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
# MLB unders bypassed the Actionable cap via empty card recovery on May-18
# (NYY/TOR Under 8.5 and CHC/MIL Under 10.5 promoted despite MLB_UNDER_ACTIONABLE_CAP).
# Block MLB unders from recovery just as overs are blocked above.
ALLOW_MLB_TOTAL_UNDER_EMPTY_CARD_RECOVERY = False

# NHL Under Actionable cap bypass prevention — same pattern as MLB unders.
# Block NHL unders from being promoted by empty card recovery despite the cap.
ALLOW_NHL_TOTAL_UNDER_EMPTY_CARD_RECOVERY = False

# Kelly Bet Sizing — Tiered Bankroll Allocation
# The 70/30 ratio means non-Actionable picks use 30% of the Kelly fraction
# that an equivalent Actionable pick would use (not a slate-level budget cap).
# This keeps per-pick amounts stable regardless of how many Actionable picks
# are on a given slate, while still expressing the confidence differential.
#
# Actionable:     0.25x fractional Kelly, 4% bankroll cap per pick
# High Variance:  0.075x fractional Kelly (30% of Actionable rate), 2% cap per pick
# Below Threshold:0.025x fractional Kelly (10% of Actionable rate), 1% cap per pick
# No Play:        $0
#
# Below Threshold fraction halved (0.050→0.025) and cap tightened (1.5%→1%) after
# May-10 review: Below Threshold picks went 1-3; continuing to size them like
# meaningful wagers compounds losses on picks that already failed confidence gating.
#
# Slate-level safety: non-Actionable total is capped at 30% of combined
# (Actionable + non-Actionable) if it would otherwise exceed that share.
ACTIONABLE_KELLY_SHARE = 0.70
NON_ACTIONABLE_KELLY_SHARE = 0.30
HIGH_VARIANCE_KELLY_FRACTION = 0.075     # 30% of Actionable's 0.25
BELOW_THRESHOLD_KELLY_FRACTION = 0.025  # 10% of Actionable's 0.25
NON_ACTIONABLE_MAX_PICK_PCT = 0.02      # 2% bankroll ceiling per non-Actionable pick
NON_ACTIONABLE_BELOW_THRESHOLD_MAX_PICK_PCT = 0.010  # 1% cap for Below Threshold

# Injury & Weather Adjustments
# Applied per key injured player to the side's model probability.
INJURY_PROB_PENALTY_PER_KEY_PLAYER = 0.015   # 1.5% per key player out
INJURY_KEY_PLAYER_THRESHOLD = 1              # minimum injuries to trigger adjustment
WEATHER_TOTAL_OVER_PENALTY = 0.025           # MLB outdoor bad weather suppresses overs by 2.5%
