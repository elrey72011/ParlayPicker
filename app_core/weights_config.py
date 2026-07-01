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
# May-29 reweight: Kalshi and Market (Novig de-vig) are ~85-90% correlated — both are
# "what money thinks." Linear pooling of two correlated forecasters double-counts a
# single underlying source, so the old 0.40+0.30=0.70 on the market complex overstated
# its independent information. Shrunk Market 0.30→0.16 (market complex now 0.58,
# Kalshi-dominant) and redistributed to the two INDEPENDENT signals: TheOver (pitcher
# data) 0.25→0.30 and ML (team form) 0.05→0.12. Sum still 1.00. This is a variance-
# reduction / redundancy fix, not a backtest-fitted optimum — fit to Brier/log-loss on
# historical signal-vs-outcome data to prove the true optimum.
MLB_TOTAL_THEOVER_WEIGHT = 0.17          # market-trust reweight (16 Jun): cut 0.30->0.17
MLB_TOTAL_ML_WEIGHT = 0.08               # market-trust reweight (16 Jun): cut 0.12->0.08
MLB_TOTAL_MARKET_WEIGHT = 0.27           # market-trust reweight (16 Jun): raised 0.16->0.27
MLB_TOTAL_KALSHI_WEIGHT = 0.48           # market-trust reweight (16 Jun): raised 0.42->0.48
MLB_TOTAL_FALLBACK_THEOVER_WEIGHT = 0.17 # market-trust reweight (16 Jun): cut 0.30->0.17
MLB_TOTAL_FALLBACK_ML_WEIGHT = 0.08      # market-trust reweight (16 Jun): cut 0.15->0.08
MLB_TOTAL_FALLBACK_MARKET_WEIGHT = 0.55  # market-trust reweight (16 Jun): raised 0.35->0.55
# MARKET-TRUST REWEIGHT — MLB totals only (16 Jun). Evidence: across 13 graded slates
# (1-15 Jun, n=171, latest run/day) the STAKED tiers were the worst performers and the
# model's confidence was inverted — Actionable 32% / High Variance 39% / Below Threshold
# 54% — and within Over picks the relationship was the same (Actionable/HV overs 33-39%
# vs Below Threshold overs 59%). The gates promote the largest model-vs-market
# divergence (highest EV/edge); on the efficient MLB totals market that divergence is
# negatively predictive, so the model + TheOver signals were actively anti-predictive
# while the market complex (Kalshi + Market, which Below Threshold picks track) sat near
# the ~50% base rate. This reweight roughly HALVES the independent model+TheOver share
# (Tier 1: 0.42 -> 0.25; market complex 0.58 -> 0.75) so calibrated_probability tracks
# the market instead of chasing divergence. Expectation is "less bleed," NOT a winning
# edge: even market-tracking is ~50%, below the 52.4% break-even at -110. Reversible;
# revisit once scripts/fit_blend_weights.py is run on per-pick signal-vs-outcome data.

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
# Under win-prob floor lowered 0.62 -> 0.55 on 8 Jun. Graded MLB totals (n=212,
# scripts/edge_by_bucket.py): Unders hit 60.6% vs Overs 52.5%, and the eff_win [0.55,0.60)
# bucket hits 58.9% — i.e. Unders carry edge well below the old 0.62/0.63 floors. Relaxing
# OVERS bled (46.7%, kept strict at MLB_OVER_ACTIONABLE_MIN_PROB=0.65); this lowers only the
# UNDER floor. NBA/NHL unders are unaffected — NHL pins 0.58 (NHL_TOTAL_MIN_WIN_PROB_STRICT)
# and NBA pins 0.65 (NBA_TOTAL_MIN_WIN_PROB) via their own max() floors downstream. The high
# EV/edge floors below and the Agrees-only under gate stay, so this can only add high-quality
# Agrees unders that were blocked by a small prob gap, never weak picks.
TOTAL_UNDER_MIN_WIN_PROB = 0.55
TOTAL_UNDER_MIN_EV = 0.22           # raised 0.18→0.22 after May-16 review: both Actionable unders lost
TOTAL_UNDER_MIN_EDGE = 0.13         # raised 0.10→0.13 after May-16 review
# MLB-specific under win prob floor (MLB unders 0-2 Actionable on May-16; both lost badly)
# Lowered 0.66→0.63 on May-28: S/A-Tier Agrees picks at 64-65% were blocked by 1-2% gap;
# MLB_UNDER_ACTIONABLE_CAP already prevents these from reaching Actionable.
# Lowered 0.63→0.55 on 8 Jun (n=212): Unders are the edge side and hold at 0.55-0.63; the
# EV(0.22)/edge(0.13)/Agrees gates remain the quality backstop. See TOTAL_UNDER_MIN_WIN_PROB.
MLB_TOTAL_UNDER_MIN_WIN_PROB = 0.55  # was 0.63 (was 0.66)
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

# High-EV divergence override — preserve a divergent pick as High Variance/Speculative
# (rather than dropping it to No Play) when EV and edge are comfortably positive, even
# if win probability falls just short of DIVERGENCE_HIGH_VARIANCE_MIN_PROB. Mirrors the
# spread divergence/high-EV overrides already used for sides: when the market is clearly
# mispricing the line, the EV/edge signal outweighs a sub-0.53 raw win prob. Without this,
# clearly +EV picks like Houston Under 8.5 (EV +9.8%, edge +6.3%, prob 0.511) and
# St. Louis Under 8.5 (EV +7.0%, edge +5.6%, prob 0.530) were denied solely by the floor.
# The relaxed prob floor still requires the model to favor the pick (>= 0.50).
DIVERGENCE_HIGH_EV_OVERRIDE_MIN_EV = 0.05
DIVERGENCE_HIGH_EV_OVERRIDE_MIN_EDGE = 0.04
DIVERGENCE_HIGH_EV_OVERRIDE_MIN_PROB = 0.50

# ML contradiction guardrail — totals only
# The XGBoost model predicts home win probability, which has no direct bearing on
# total runs. The old threshold (50%) was blocking good over picks when the ML
# simply predicted the away team to win (e.g., MIA/ATL: ML=40.8%, game went 12-0=12 total).
# Only block a total_over when ML probability is extremely low (< 35%), signalling the
# model strongly expects a low-scoring or blowout-type result.
# Unders are not subject to this guardrail — a low home win prob says nothing about
# whether the game stays under the total.
# 18 Jun: raised 0.35 -> 0.45 so we don't stake an OVER our own ML model leans against.
# TheOver's coarse hit-rate was promoting overs to Actionable while ML sat below a coin
# flip (Seattle Over: ML 0.474). An over the model rates < 0.45 is now a No Play
# regardless of how hot TheOver runs.
TOTAL_ML_CONTRADICTION_OVER_MAX_PROB = 0.45

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
# History of reactive 1-2 slate tweaks (do not continue this): 0.62 (May-9) -> 0.65
# (May-10) -> 0.68 (May-12) -> 0.56 -> 0.63 (May-13) -> 0.60 (May-17).
# Jun-5 AGGREGATE calibration (n=191 graded picks, not one slate): effective_win_prob
# is well-calibrated at 0.65+ (predicts 68%, hits 64%) but OVERCONFIDENT in the 0.60-0.65
# band (predicts 62%, hits 52% = coin flip). That band was driving the Actionable
# "Hammers" that have been losing. Raise the gate to 0.65 so only the calibrated band gets
# big Kelly; 0.60-0.65 picks fall to High Variance (small stake). This is set from the
# aggregate, not a single slate -- judge future moves the same way, and prefer fixing
# calibration (the middle-band miscalibration is non-linear, so a flat shrink won't do it)
# over re-tuning this gate.
# 28 Jun: SYMMETRIZED with the under floor. The strict 0.65/0.07 over bar above was set in the
# "overs bled 46.7%" era; the current graded buckets (n=350) show overs are now ~51% —
# over:Agrees .509, over:Neutral .510 — no worse than under:Neutral (.500) or under:Disagrees
# (.457). Direction is NOT the signal; the bucket is (only under:Agrees .631 is truly good).
# The empirical-tier overlay already gates staking by PROVEN bucket, and no over bucket is
# proven yet, so this opens no floodgate: overs at ~51% are still below the -110 break-even
# and the overlay still blocks them. What it removes is the stale asymmetry — and it lets an
# over bucket actually stake IF it ever proves out (>0.55), instead of the 0.65 floor blocking
# it even then. (Per-line over guards — high/low total-line penalties — are kept; those are
# line-environment effects, not the over/under bucket asymmetry.)
MLB_OVER_ACTIONABLE_MIN_PROB = 0.55   # = MLB_TOTAL_UNDER_MIN_WIN_PROB (symmetric)
MLB_OVER_ACTIONABLE_MIN_EV = 0.03     # = TOTAL_OVER_MIN_EV (symmetric)
MLB_OVER_ACTIONABLE_MIN_EDGE = 0.04
# Earned relaxation of the strict MLB-over bar (17 Jun). The blanket 0.65 prob /
# 0.07 EV gate above made the over-heavy slates produce zero Actionable picks: the
# over-prob shrink + market-anchored debias pull every over's calibrated prob to
# ~0.48-0.51, ~15pts under the gate, so nothing qualified regardless of how the
# market/Kalshi/ML/TheOver signals lined up. These RELAXED requirements apply ONLY
# to over picks where Kalshi AGREES (directional consensus) AND the realized
# empirical bucket has EARNED trust (>=55% over >=25 graded picks, the same proof
# the empirical overlay's Actionable promotion requires). Non-Agrees overs, and
# Agrees overs in unproven/bleeding buckets, keep the strict 0.65/0.07/0.04 bar.
# This is edge-gated volume, not loosened discipline: the proven-bucket condition
# is the realized-performance backstop, and the empirical overlay still has the
# final say on staking.
# 0.55 (not 0.58): with the market-debias exemption for Kalshi-backed overs (below),
# an un-suppressed Agrees-over consensus lands ~0.55-0.56, so a 0.58 bar would still
# clip them. 0.55 vs the -110 break-even (0.524) is a thin but real +2.6% edge, and
# the proven-bucket condition + the empirical overlay remain the quality backstops.
MLB_OVER_AGREES_ACTIONABLE_MIN_PROB = 0.55
MLB_OVER_AGREES_ACTIONABLE_MIN_EV = 0.03
MLB_OVER_AGREES_ACTIONABLE_MIN_EDGE = 0.02
# Hard cap on calibrated probability for MLB overs — prevents residual TheOver
# inflation above a reliable ceiling. Raised from 0.67 → 0.72 (May-26): the 0.67
# cap was set when TheOver had a cross-matching bug producing flat ~0.85 probs.
# That bug was fixed May-16. The shrinkage factor (0.85) handles general calibration;
# the cap now only blocks genuine outliers, not normal high-confidence picks.
MLB_OVER_CALIBRATED_PROB_CAP = 0.72

# MLB total HV/Spec floor — May 27: HV/Spec MLB totals went 0-6 while BT went 6-2.
# Picks with effective_win_probability below this value are demoted from HV/Spec
# to Below Threshold (still visible, minimal Kelly sizing) rather than HV.
MLB_TOTAL_HV_MIN_WIN_PROB = 0.62

# MLB Under Actionable cap — was True after May 16-17 (0-4 Actionable record).
# Removed May-28: unders have hit at a higher rate than overs across May 22-27
# (BT unders 6-2 on May 27; Actionable unders 2-0; overs 0-4 on same slate).
# The cap was set before the TheOver conflict penalty and double-shrink fix.
# Replaced by a consensus gate in the pipeline: only "Agrees" MLB unders can be Actionable.
MLB_UNDER_ACTIONABLE_CAP = False

# NHL Under Actionable cap — CAR/MTL Under 5.5 went 8 total goals at Actionable on May 21.
# Same pattern as MLB unders: model overconfident on unders in lower-scoring sport contexts.
# Cap at High Variance until NHL under performance demonstrates reliability.
NHL_UNDER_ACTIONABLE_CAP = True

# Cold-Market Penalty Layer (by League + Market Type)
MLB_TOTAL_OVER_ACTIONABLE_PENALTY = 0.00
# MLB Under raised 0.03→0.05 after May-11 review: LA/SF Under 9.5 lost (12 runs scored).
# Further raised 0.05→0.07 after May-16 review: both Actionable unders lost (11 runs each).
# Lowered 0.07→0.03 on May-28: MLB_UNDER_ACTIONABLE_CAP=True already prevents Actionable;
# stacking a 7% penalty on top of the cap created a 28% combined edge requirement that
# blocked S-Tier Agrees picks with 19-20% edge — double-counting the same protection.
MLB_TOTAL_UNDER_ACTIONABLE_PENALTY = 0.03
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

# Edge-based no-stake gates from graded MLB totals (20 May-7 Jun, n=182; see
# scripts/edge_by_bucket.py). Two buckets bled below the -110 breakeven (52.4%) and are
# held out of the production card rather than tuned:
#   * Neutral-consensus totals — Over/Neutral hit 48.2% (n=56), the single largest losing
#     cell; Agrees (61.4%) and Disagrees (63.2%) totals keep their edge.
#   * Mid-line Overs (line in [MLB_OVER_MIN_TOTAL_LINE, MLB_MID_TOTAL_LINE_THRESHOLD), i.e.
#     8.0-9.5) — hit 46.5% (n=43), while the Under on those same lines hit 65.4%.
# Capped picks drop to Below Threshold (visible, unstaked). Flip either flag to False to
# restore prior behavior; re-evaluate as the graded sample grows.
MLB_TOTAL_NEUTRAL_NO_STAKE = True
MLB_OVER_MID_LINE_NO_STAKE = True

# Low total line floor — MLB overs with a line below 8.0 are pitcher-friendly games.
# May 20: CHC/MIL Over 6.5 (5 total), SD/LAD Over 7.5 (4 total) both lost. The
# low_line_over_guardrail is now CONSENSUS-AWARE (see core/streamlit_pipeline.py):
# Neutral low-line overs are held at Below Threshold, while Disagrees/Agrees low-line
# overs are surfaced at High Variance (the strong Agrees ones keep Actionable via the
# carve-out below). Sub-8.0 overs are not a uniformly weak bucket — see backtest.
MLB_OVER_MIN_TOTAL_LINE = 8.0

# Sub-8.0 MLB over escape hatch (conditioned carve-out for the low_line_over_guardrail
# above). Backtest (scripts/backtest_low_line_over.py, graded slates 20 May-5 Jun 2026,
# n=51 sub-8.0 overs):
#   sub-8.0 overs overall ...... 29-22 (56.9%), +19.2% ROI  (beats >=8.0 overs 53.8%)
#   sub-8.0 overs, Agrees ...... 8-3  (72.7%), +44.9% ROI   <- carve-out -> Actionable
#   sub-8.0 overs, Disagrees ... 12-8 (60.0%)               <- High Variance (profitable)
#   sub-8.0 overs, Neutral ..... 9-11 (45.0%)               <- held at Below Threshold
# The losses that originally motivated the veto (May-20 CHC/MIL Over 6.5, SD/LAD Over
# 7.5) were Neutral/Disagrees at effective win prob < 0.58. Only an already-Actionable,
# Kalshi-Agrees over with strong shrinkage-adjusted win prob and edge keeps Actionable.
#
# DISABLED 6 Jun: fresh slates contradicted the backtest above. Low-line MLB "Over 7.5"
# plays went 1-4 (5 Jun) then 0-4 (6 Jun) — the SAME matchups (SD/Mets, Texas/Cle,
# Miami/TB) were re-served and missed on back-to-back nights, landing 5, 5, 7, 6 runs.
# The model is still inflated on MLB overs (see the calibrated-prob cap in
# streamlit_pipeline), so the 0.62/0.08 floors below clear on overconfidence rather than
# real edge. Hold every sub-8.0 over at High Variance until the carve-out is re-tuned on
# the post-6-Jun graded sample. Flip back to True to restore the carve-out.
MLB_LOW_LINE_OVER_OVERRIDE_ENABLED = False
MLB_LOW_LINE_OVER_OVERRIDE_MIN_WIN_PROB = 0.62
MLB_LOW_LINE_OVER_OVERRIDE_MIN_EDGE = 0.08

# TheOver direction conflict penalty — when TheOver's probability clearly disagrees
# with the blended pick direction for an MLB total, apply this penalty to the
# conflicting pick's final_family_score so the TheOver-aligned direction wins selection.
# TheOver incorporates pitcher/rotation data that Kalshi + market can't fully price.
# May 22: Over picks on Miami 7.5, Cubs 7.5 and Under picks on Angels 7.5, BOS 7.5,
# SF 7.5 all lost; the model was following Kalshi/market while TheOver likely disagreed.
# May 23: All 5 MLB Over picks in High Variance lost (ATL, CHC, TOR, BOS, SD).
# Tightened threshold to 0.50 and raised penalty to 0.35 to flip more conflicting Overs.
MLB_THEOVER_CONFLICT_THRESHOLD = 0.50   # TheOver says other side has ≥50% probability
MLB_THEOVER_CONFLICT_PENALTY = 0.35     # Subtracted from final_family_score to flip selection

# Weight on Kalshi's DIRECTIONAL confidence relative to TheOver's when the two
# disagree on an MLB total's over/under direction (used by
# _mlb_total_direction_conflict). Motivation — the graded history (n=363,
# data/calibration/bucket_stats.json) shows the pick LOST whenever it fought Kalshi
# on an MLB total: over:Disagrees 46% (n=56), under:Disagrees 45% (n=38), while the
# only proven-profitable total bucket is Kalshi-agreeing under:Agrees 62% (n=66).
# Kalshi is therefore the more reliable direction signal, so it gets a moderate edge:
# at 1.5 TheOver must be >1.5x as confident (distance from 0.50) as Kalshi to flip the
# pick away from Kalshi's side. Genuinely strong TheOver pitcher reads (>1.5x) still
# win — this only reclaims the marginal cases where a mildly-more-confident TheOver was
# overriding Kalshi and losing. Governs DIRECTION selection only (which over/under row
# is penalized in the family sort); does NOT change calibrated_probability, EV, edge,
# or any staking threshold, so the calibration tables stay valid. 1.0 restores the
# prior symmetric most-confident-wins behavior (fully reversible). NOT a fitted
# optimum — a moderate, reversible lean toward the proven-more-reliable signal; fit to
# Brier/log-loss on per-pick signal-vs-outcome data to prove the true value.
KALSHI_DIRECTION_CONFIDENCE_WEIGHT = 1.5

# TheOver tags each WinProbability with a WinProbSource (set by our own M-code scraper):
#   model_hit_rate          -> TheOver's model picked the OVER; P(Over) = hit_rate
#   model_hit_rate_flipped  -> TheOver's model picked the UNDER; P(Over) = 1 - hit_rate
#   public_betting_pct      -> derived from public betting %
#   default_0.5             -> no read (handled separately by its 0.50 value)
# model_hit_rate_flipped is a GENUINE TheOver Under pick — the same signal as
# model_hit_rate, just the Under side — NOT a fallback or low-quality value. We FADE it:
# shrink P(Over) toward 0.50 so it pulls the blend/direction proportionally less. This is
# a tunable strategy bet (fade a cold model), not a data-quality filter.
#
# TUNING PROTOCOL — read before touching MLB_THEOVER_FADE_SHRINK:
#   * The fade ONLY affects flipped games (TheOver picked Under). Judge it by the
#     flipped-game counterfactual — on those games, did Over or Under actually hit? —
#     NOT by the aggregate consensus/ROI buckets in scripts/backtest_theover_direction.py.
#     Those buckets are STAKE-WEIGHTED and dominated by big Actionable Over hammers
#     (genuine model_hit_rate/public picks, not flipped games), so they conflate the fade
#     decision with hammer staking and will mislead you (they did once already).
#   * Evidence to date — fading (i.e. picking the market Over on a flipped game) was right
#     ~11-6 across 1-3 Jun (1 Jun 3-0, 2 Jun 5-3, 3 Jun 3-3). That supported the 0.75 setting.
#   * REVERSAL (5-6 Jun): the flipped-game signal inverted. On 6 Jun the pipeline faded
#     TheOver and forced the Over on 5 flipped games (Cubs/SF, Miami/TB, Texas/Cle, NYY/Bos,
#     Houston/Ath) and those Overs went 1-4 — i.e. TheOver's faded Unders would have gone
#     4-1. The 0.75 fade was now picking the losing side and producing all-Over, empty
#     production cards. Eased 0.75 -> 0.25 ("respect the signal" end; see
#     tests/test_mlb_total_direction_conflict.test_light_fade_still_lets_strong_theover_win)
#     so strong TheOver Under reads (conf >> Kalshi) flip direction to the Under again.
#   * Do NOT retune on one slate. Wait for >= ~6 graded slates from the current build, then
#     move the knob toward 0.0 (full trust) only if flipped-game Unders keep beating Overs,
#     or back toward 1.0 (full neutralize) if they start missing again. Empty
#     MLB_THEOVER_FADE_SOURCES to disable fading entirely.
# 18 Jun: added the OVER source ("model_hit_rate") alongside the flipped UNDER source.
# Both are coarse, repeated hit-rate fractions (e.g. 7/8, 25/32) used as if they were
# per-game probabilities, and fading only the under side structurally tilted the card
# toward overs — TheOver alone was promoting overs the ML model leaned against (18 Jun
# Seattle: ML 0.47 -> Actionable over on TheOver 0.88). Faded symmetrically now.
MLB_THEOVER_FADE_SOURCES = frozenset({"model_hit_rate_flipped", "model_hit_rate"})
# 26 Jun: the raw TheOver scrape (M-code) proves these are NOT genuine per-game reads —
# ModelHitRate is a hardcoded constant (0.875 = 7/8), so every Under pick emits P(Over)=0.125
# and every Over pick 0.875, collapsing to a single value across the slate (12/15 totals on
# the 26-Jun upload were identical). The pipeline already blanks this for the DIRECTION
# decision (see _mlb_total_direction_conflict) but left it in the MAGNITUDE blend at 0.25
# fade, where the constant was inflating Under win-probabilities with fake support (the
# Unders graded ~0.53 on the card yet kept losing). Neutralize it fully in the blend too:
# 1.0 sends the faded value to 0.50 (no opinion), so the win prob defers to the genuine
# Kalshi/Market/ML signals. There is no per-game information to preserve — the value is a
# placeholder. (PublicBettingPct, the one column that varies, is a separate untouched source.)
MLB_THEOVER_FADE_SHRINK = 1.0   # fraction of (P-0.5) removed; 1.0=neutralized, 0.0=untouched
# SCOPE: the shrink above is MLB-tuned and the blend callers apply it ONLY to MLB-total
# rows. Non-MLB totals (NBA/NHL) can also carry a faded WinProbSource, so they keep this
# separate legacy default — held at the prior 0.75 so the MLB-only reduction (0.75 -> 0.25,
# 6 Jun) does not silently change non-MLB calibrated probabilities. No evidence to retune
# non-MLB; revisit separately if/when graded non-MLB flipped-game data warrants it.
THEOVER_FADE_SHRINK_DEFAULT = 0.75

# 26 Jun: with the constant model_hit_rate neutralized, the only per-game value TheOver still
# emits is PublicBettingPct (the % of money/tickets on a side). Public money is a FADE signal,
# not a follow one — heavy-public sides win LESS — so we CONTRARIAN-fade it instead of letting
# the blend treat "88% on the Over" as "Over is 88% likely". The transform mirrors the
# deviation from 0.50: P(Over)_new = 0.50 - STRENGTH * (P(Over) - 0.50). STRENGTH 0 = neutral
# (ignore public), 1.0 = full mirror (1 - P), 0<s<1 = weak contrarian. Held weak: the
# public-fade edge is real but small, and this rides at 17% blend weight on the few games that
# even carry a public %. Scoped to MLB totals (the blend callers apply it only there).
MLB_PUBLIC_BETTING_FADE_SOURCES = frozenset({"public_betting_pct"})
MLB_PUBLIC_BETTING_FADE_STRENGTH = 0.5

# Degraded-feed down-weight (see _apply_analysis_calculations' degradation guard).
# When the slate's TheOver totals reads look degenerately clustered, shrink each
# totals read toward neutral 0.50 by this fraction instead of nulling TheOver
# outright. 1.0 = fully neutralized (equivalent to the old null behaviour),
# 0.0 = untouched. Set to 0.50 (halve the directional signal) so a false positive
# — TheOver legitimately rating many games at a common confidence — still
# contributes a damped, game-specific read to the blend, while a genuinely
# non-game-specific constant feed is still meaningfully discounted. Independent of
# MLB_THEOVER_FADE_SHRINK (the per-row flipped-source fade), which still applies.
THEOVER_DEGRADED_FADE_SHRINK = 0.50

# No-Kalshi totals are treated as lower confidence in selection stage
NO_KALSHI_TOTAL_EXTRA_PENALTY = 0.02
NO_KALSHI_TOTAL_UNDER_EXTRA_PENALTY = 0.03

# Cross-family selection nudge to avoid unders dominating finalists on EV/edge alone.
# Lowered 0.10 -> 0.05 on 7 Jun. The 0.10 blanket handicap had no evidentiary basis and
# was over-suppressing Unders: across 20 May-5 Jun graded MLB totals (n=182) Unders out-
# hit Overs 62.1% vs 55.2% (ROI +23.0% vs +11.1%), yet the finalist selection produced
# all-Over cards (7 Jun: 15/15 Overs) because every Under started 0.10 behind. Easing the
# handicap lets Unders compete on merit. Independent of the TheOver fade (see
# MLB_THEOVER_FADE_SHRINK), which governs the source-confidence flip, not this nudge.
TOTAL_UNDER_FINALIST_SCORE_PENALTY = 0.05

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
    # Raised 0.06→0.08 after May-27 review: HV/Spec MLB unders went 0-3.
    ("MLB", "under"): 0.08,
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

# Empirical tier overlay — final tier pass that reassigns Actionable / High
# Variance / Below Threshold from realized bucket performance (scripts/
# fit_bucket_stats.py) + isotonic-calibrated probability (scripts/
# fit_calibration.py), replacing model-vs-market EV/edge as the promotion
# signal. Jun 5-10 graded recaps: EV/edge-promoted tiers hit ~21% (Actionable
# 1-4, HV 3-11) while Below Threshold hit 59% — the 10 Jun slate went 10-5 and
# still lost money because stake followed the inverted tiers. Safety statuses
# (No Play / Missing Line) are never overridden. Flip False to restore the
# legacy EV/edge tiers. See core/empirical_tiers.py for thresholds.
EMPIRICAL_TIER_OVERLAY_ENABLED = True

# Market-anchored over-bias correction for MLB totals. 13 Jun: Kalshi and ML both
# sat 10-15 pts above the de-vig sportsbook market on P(over) for nearly every
# game, producing a 14-of-15 all-Over card while the market sat at coin-flip. The
# market is the sharp, unbiased anchor and graded MLB overs hit only ~52% (no edge
# over it), so the systematic model-vs-market over-gap is bias, not signal. The
# pipeline removes the slate-MEAN gap from each MLB total's blended P(over) (and
# adds it to P(under)) so direction selection rebalances — per-game relative leans
# and the market's own genuine lean are preserved. Flip False to disable.
# MAX_SHIFT caps how much the correction can move any single game's probability.
MLB_TOTAL_MARKET_DEBIAS_ENABLED = True
MLB_TOTAL_MARKET_DEBIAS_MAX_SHIFT = 0.15
# 17 Jun: exempt overs that Kalshi INDEPENDENTLY backs from the de-bias. The de-bias
# strips a model-only over-lean the sharp market contradicts; but when the prediction
# market (an independent source) ALSO leans over (kalshi P(over) >= this threshold),
# the lean is corroborated, not model bias to remove. Debiasing those was pulling a
# genuine Kalshi+ML+TheOver over-consensus down to the de-vig market (~0.48), turning
# +EV overs into -EV No Plays. Non-Agrees overs are still debiased. Set threshold high
# (>1.0) to disable the exemption and debias all overs as before.
MLB_TOTAL_MARKET_DEBIAS_EXEMPT_AGREES_OVER = True
MLB_TOTAL_MARKET_DEBIAS_AGREES_KALSHI_MIN = 0.52


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
# Same-direction UNDER Actionable caps — the mirror of the Over caps above.
# 12 Jun: the empirical overlay staked 4-5 MLB Actionable Unders (Disagrees bucket);
# the league went 10/15 games at 10+ runs and the Under block went 1-3 (am) / 1-4 (pm),
# -$61.60 on the Actionable tier, while the (mostly-Over) Below Threshold tier went 4-2.
# Same-direction totals share the night's run environment, so an unguarded Under block
# busts together exactly as the 6 Jun Over block did (0-4). Cap how many total_under
# (and MLB total_under) picks may be Actionable; the lowest-edge excess drops to High
# Variance. The empirical overlay re-applies these after promotion.
MAX_TOTAL_UNDER_ACTIONABLE_COUNT = 3
MAX_MLB_TOTAL_UNDER_ACTIONABLE_COUNT = 2
# Speculative (High Variance/Speculative) concentration caps. The Actionable caps above
# never protected the speculative surface, so the card could still collapse onto one
# league+direction there: 6 Jun the HV tier held 4 MLB "Over 7.5" plays that all lost
# together (0-4) while the benched Unders won. Cap how many total_over (and MLB
# total_over) picks may surface as High Variance; the lowest-ranked excess is pushed to
# Below Threshold so no single correlated Over bucket dominates the speculative card.
MAX_TOTAL_OVER_HIGH_VARIANCE_COUNT = 3
MAX_MLB_TOTAL_OVER_HIGH_VARIANCE_COUNT = 2
TOTAL_OVER_PROB_SHRINK = 0.60
MLB_TOTAL_OVER_PROB_SHRINK = 0.85

# Run-line (spread) model probability is the moneyline P(win), but a run-line pays on
# the MARGIN, not the win: a -1.5 favorite must win by 2+, a +1.5 dog covers unless it
# loses by 2+. The two outcomes are separated by the 1-run band (~28% of MLB games are
# decided by exactly 1 run), so P(win) systematically OVERSTATES a favorite's cover and
# understates a dog's. 19 Jun: Pittsburgh -1.5 carried ml P(win)=0.54 as its cover prob;
# it lost outright. Convert P(win) -> P(cover +-1.5) by shifting out the favorite's share
# of the 1-run band: fav cover = P(win) - band, dog cover = P(win) + band. The shift is
# symmetric so the two sides still sum to 1 (no push on a 1.5 line). MLB-only: NBA point
# spreads and NHL puck lines have different margin distributions.
MLB_RUNLINE_COVER_CONVERSION_ENABLED = True
MLB_RUNLINE_ONE_RUN_BAND = 0.135  # P(favorite wins by exactly 1 run); ~half the ~28% 1-run rate

# A MAIN total is priced near pick'em (both sides ~ -110), so its de-vigged implied
# probability sits near 0.50. A totals pick whose de-vig falls OUTSIDE this band is almost
# certainly an ALT line or a mis-scrape the live matcher latched onto instead of the main
# number -- value alone can't distinguish e.g. an Over 12.5 alt from a real 12.5 (20 Jun:
# NYY "Over 12.5" matched at +285 / de-vig 0.26 while the real line was ~9.5). Such lines
# are routed to the rejected-live -> uploaded-reference fallback rather than priced off
# garbage. Band kept generous so genuinely juiced-but-real main totals are still trusted.
MAIN_TOTAL_MIN_DEVIG_PROB = 0.35
MAIN_TOTAL_MAX_DEVIG_PROB = 0.65
MLB_TOTAL_OVER_MIN_PRODUCTION_WIN_PROB = 0.60
MLB_TOTAL_OVER_MIN_PRODUCTION_EV = 0.07
MLB_TOTAL_OVER_MIN_PRODUCTION_EDGE = 0.04
DEGRADED_FEATURE_KELLY_MULTIPLIER = 0.50
DEGRADED_FEATURE_MAX_SLATE_EXPOSURE_PCT = 0.12
DEGRADED_FEATURE_MAX_PICK_EXPOSURE_PCT = 0.02
ALLOW_EMPTY_CARD_RECOVERY = True
ENABLE_EMPTY_CARD_RECOVERY = True
EMPTY_CARD_RECOVERY_MAX_PICKS = 2
# Speculative-lean tuning (22 Jun, user-directed): when the card is otherwise empty, surface
# the best CLEAN positive-EV near-miss at SMALL size for daily action. The user accepted that
# these thin edges bleed slowly to the vig; harm is limited by (a) positive EV/edge required,
# (b) win-prob floor 0.50 (no outright dogs), (c) consensus must NOT be Disagrees -- never bet
# AGAINST Kalshi, the losing bucket, (d) small Kelly caps below, (e) overs still excluded
# (they bleed worst). Raise these back toward 0.07/0.05/0.57 to return to strict no-play.
EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EV = 0.02
EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EDGE = 0.02
EMPTY_CARD_RECOVERY_MIN_PRODUCTION_WIN_PROB = 0.50
EMPTY_CARD_RECOVERY_EXCLUDE_MARKET_TYPES = []
EMPTY_CARD_RECOVERY_EXCLUDE_SOURCES = ["rejected_live"]
# Stake kept small because these are thin/speculative edges, not vetted Actionable plays.
EMPTY_CARD_RECOVERY_MAX_KELLY_TOTAL_PCT = 0.03
EMPTY_CARD_RECOVERY_MAX_KELLY_PER_PICK_PCT = 0.015
# Consensus gate for speculative-lean recovery: only Agrees/Neutral picks. A "Disagrees"
# pick means Kalshi backs the other side (the graded-loser bucket), so it is never recovered.
EMPTY_CARD_RECOVERY_CONSENSUS = ("Agrees", "Neutral")
ALLOW_MLB_TOTAL_OVER_EMPTY_CARD_RECOVERY = False
# Unders re-allowed into speculative recovery 22 Jun (user-directed thin-edge action): the
# under side is the data-supported one (under:Agrees 61%), and the small-stake + positive-EV
# + not-Disagrees guards above limit the downside. Overs stay blocked (they bleed worst).
ALLOW_MLB_TOTAL_UNDER_EMPTY_CARD_RECOVERY = True

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
# 16 Jun: set to 0 — confine real production stakes to the PROVEN Actionable tier
# (Agrees-bucket, ~61% realized) and stop staking High Variance / Below Threshold.
# Across 13 graded slates (n=171) the non-Actionable tiers the portfolio was staking
# ran sub-break-even (staked Act+HV 37%); the totals market the system is fed is
# entirely Disagrees coin-flips, so any HV/BT stake is -EV. This also resolves an
# inconsistency: the portfolio optimizer was staking HV/BT while other paths
# (strategy lab, production-card guard) already zero non-Actionable rows. Raise above
# 0 only if a non-Actionable tier earns a proven, out-of-sample edge. 0 disables.
NON_ACTIONABLE_KELLY_SHARE = 0.0
HIGH_VARIANCE_KELLY_FRACTION = 0.075     # 30% of Actionable's 0.25
BELOW_THRESHOLD_KELLY_FRACTION = 0.025  # 10% of Actionable's 0.25
NON_ACTIONABLE_MAX_PICK_PCT = 0.02      # 2% bankroll ceiling per non-Actionable pick
NON_ACTIONABLE_BELOW_THRESHOLD_MAX_PICK_PCT = 0.010  # 1% cap for Below Threshold

# --- Force-deploy daily stake budget (17 Jun, user-directed) ------------------
# When DAILY_STAKE_FORCE_DEPLOY is True, the day's eligible card is staked to SUM to
# DAILY_STAKE_BUDGET, split by tier: Actionable gets ACTIONABLE_STAKE_SHARE of it and
# the viable non-Actionable picks (High Variance / Below Threshold) split the rest.
# Within a tier, bets are proportional to each pick's Kelly fraction (edge-weighted;
# equal-weight only if every Kelly is 0) and normalized to FILL the tier budget —
# OVERRIDING the per-pick (4%) and slate-% Kelly caps. A tier with no eligible picks
# deploys nothing (its budget is NOT pushed onto the other tier, so a card with no
# Actionable picks stakes only the 40%). Slates suspended by a health guard (e.g.
# slate_direction_imbalance) and picks without a clean live line still stake $0.
# This is an explicit, aggressive override of fractional-Kelly discipline; set
# DAILY_STAKE_FORCE_DEPLOY = False to restore normal Kelly sizing.
#
# Turned OFF 20 Jun: on no-edge slates it hunted for the least-bad row and force-staked
# it -- e.g. $750 on a corrupt-data "Cubs Over 13.5" -- which is exactly the posture that
# cost the 19 Jun card. With it off, empty/weak slates stake $0 and only genuine edges
# get normal fractional-Kelly stakes. Re-enable by setting this True.
DAILY_STAKE_FORCE_DEPLOY = False
DAILY_STAKE_BUDGET = 5000.0
ACTIONABLE_STAKE_SHARE = 0.60
# Concentration controls — keep a thin/weak slate from dumping a whole tier budget
# onto one marginal pick (17 Jun: a lone Below Threshold Under drew the full $2000).
#  - Stake the non-Actionable 40% ONLY on High Variance picks; Below Threshold picks
#    failed the thresholds outright and get no forced stake (True to include them).
#  - Cap any single force-deployed pick at FORCE_DEPLOY_MAX_PICK_PCT of DAILY_STAKE_BUDGET;
#    the excess is NOT redistributed, so a tier with too few picks UNDER-deploys by
#    design rather than concentrating. At 0.15 ($750 on a $5000 budget) the full $3000
#    Actionable share needs >=4 picks; fewer picks deploy proportionally less.
FORCE_DEPLOY_NONACTIONABLE_INCLUDE_BELOW_THRESHOLD = False
FORCE_DEPLOY_MAX_PICK_PCT = 0.15
# Consensus gate for the force-deploy non-Actionable (40%) tier (18 Jun). Never stake
# AGAINST Kalshi: a "Disagrees" pick means Kalshi backs the OTHER side, the bucket the
# graded history shows as a loser, so it is excluded from staking. "Neutral" (Kalshi
# undecided, our model/TheOver lead) and "Agrees" remain stakeable. Set to ("Agrees",)
# for the strict, market-confirmed-only posture (deploys far less while honest Kalshi
# sits near 0.50). The Actionable 60% tier is already Agrees-only via the empirical
# overlay (ACTIONABLE_PROVEN_CONSENSUS), so this only governs the speculative 40%.
FORCE_DEPLOY_NONACTIONABLE_CONSENSUS = ("Agrees", "Neutral")

# Injury & Weather Adjustments
# Applied per key injured player to the side's model probability.
INJURY_PROB_PENALTY_PER_KEY_PLAYER = 0.015   # 1.5% per key player out
INJURY_KEY_PLAYER_THRESHOLD = 1              # minimum injuries to trigger adjustment
WEATHER_TOTAL_OVER_PENALTY = 0.025           # MLB outdoor bad weather suppresses overs by 2.5%

# --- Kalshi extreme-price guard (21 Jun) --------------------------------------
# A pre-game TOTAL or run-line (SPREAD) Kalshi contract priced at a de-vigged near-
# certainty is a settled/illiquid/mis-scraped market, not a confident read. 20 Jun: a
# "Total Runs over 12.5" contract sat at yes_bid 0.99 / yes_ask 1.00 (mid 0.995) -- a
# TIGHT spread, so the >0.40 illiquidity guard missed it -- and that 0.995 inflated a
# blend to a fake 0.72 win prob behind the $750 Cubs landmine. Drop the Kalshi signal
# when a totals/spread price falls outside this band. Moneyline is EXEMPT: heavy
# favorites/dogs legitimately price near the edges.
KALSHI_TOTAL_SPREAD_MIN_PROB = 0.05
KALSHI_TOTAL_SPREAD_MAX_PROB = 0.95

# --- Moneyline parlay legs (21 Jun, user-directed) ----------------------------
# Moneyline is the model's NATIVE output (XGBoost predicts P(win) directly, no margin/
# cover conversion). Opened up as PARLAY-ONLY legs: they never surface as standalone
# single bets, only combine into parlays. Off by default -- a core-pipeline change, so it
# ships gated and is enabled deliberately. Moneyline has almost no graded history yet, so
# it starts uncalibrated; parlay-only keeps it from drawing single stakes while it builds
# a track record. Odds capped to a sane range (heavy favorites = low value / poor parlay
# legs; longshots = variance) and a real edge over the implied price is required.
ENABLE_MONEYLINE_PARLAY_LEGS = False
MONEYLINE_PARLAY_MIN_ODDS = -250
MONEYLINE_PARLAY_MAX_ODDS = 250
MONEYLINE_PARLAY_MIN_EDGE = 0.03

# ── Pitcher-strikeout props (production) ──
# MLB run totals are near-efficient — model and market agree to within a point or two, so
# most picks are honest "no edge / No Play". Strikeout props are a softer market, so the
# props slice is run daily to surface genuine edges. ENABLED to stake, but with a deliberate
# caveat: the prop model has NO graded track record yet, so it is uncalibrated. To avoid
# repeating the 19-Jun mistake (full stakes on an unproven signal) the per-pick and total
# stake fractions are held SMALL until a calibration record exists; raise them once the
# market has proven out. The +EV / min-edge discipline (PROP_MIN_EDGE) still gates every
# pick, so only real edges bet.
ENABLE_STRIKEOUT_PROPS_PRODUCTION = True
# Fraction of bankroll per prop pick / across all prop picks. Small because uncalibrated.
STRIKEOUT_PROP_KELLY_PER_PICK_PCT = 0.01
STRIKEOUT_PROP_KELLY_TOTAL_PCT = 0.03
# Kelly is staked fractionally (quarter-Kelly) on top of the cap — conservative sizing.
STRIKEOUT_PROP_KELLY_FRACTION = 0.25
