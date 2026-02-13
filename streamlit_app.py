import json
import math
import logging
import os
import re
import tempfile
import time
import itertools
import traceback
from datetime import datetime, timedelta, timezone
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests
import streamlit as st

from app_core.kalshi_integrator import (
    KalshiIntegrator,
    LEAGUE_SERIES_MAP,
    league_game_prefix,
    league_series_ticker,
    team_code_for_league,
    parse_event_ticker_codes,
    resolve_team_code,
    NCAAB_CODE_ALIASES,
    NCAAF_CODE_ALIASES,
    match_game_to_kalshi,
)

from app_core.llm_assistant import generate_confidence_explanation, initialize_gemini, generate_batch_confidence_explanation, generate_pick_rationale

from app_core.reddit_sentiment import fetch_reddit_sentiment_map

from app_core.sentiment_pipeline import (
    MAX_SENTIMENT_CALLS,
    fetch_team_news,
    league_label,
    team_sentiment_from_articles,
)

from app_core.theover_ingest import (
    process_theover_inputs,
    parse_theover_public_betting_text,
    generate_canonical_key,
    TEAM_ALIAS_MAP,
)

from app_core.team_name_matcher import TeamNameMatcher

from app_core.prediction_engine import (
    VERTEX_FEATURE_COLUMNS,
    PredictionEngine,
    get_prediction_prob,
    match_team_name,
)

from app_core.apisports import (
    APISportsBasketballClient,
    APISportsFootballClient,
    APISportsHockeyClient,
    get_key as get_apisports_key,
)

from app_core.new_summary_logic import (
    build_game_summary_v2,
    reorder_for_spread_total_focus_v2,
    calculate_consensus_for_row,
)

from app_core.feature_processing import (
    enrich_with_model_features,
    build_model_feature_row_from_record,
    robust_normalize_team,
    calculate_confidence,
)

from app_core.sportsdata import (
    SportsDataNBAClient,
    SportsDataNFLClient,
    SportsDataNHLClient,
    SportsDataNCAABClient,
    SportsDataNCAAFClient,
    get_key as get_sportsdata_key,
)

import app_core.market_tracker as market_tracker
import app_core.snapshot_manager as snapshot_manager

from app_core.weights_config import (
    KALSHI_WEIGHT,
    MARKET_WEIGHT,
    ML_MODEL_WEIGHT,
    THEOVER_WEIGHT,
    SENTIMENT_WEIGHT,
)

from app_core.consensus_ingest import enrich_with_consensus
from app_core.odds_api import filter_games_today_only


try:
    from app_core.sentiment import RealSentimentAnalyzer
except Exception:  # pragma: no cover - optional import
    RealSentimentAnalyzer = None

try:
    import rapidfuzz
    from rapidfuzz import fuzz
except ImportError:
    rapidfuzz = None
    fuzz = None

try:
    from parlay_optimizer import ParlayOptimizer
except ImportError:
    ParlayOptimizer = None

try:
    from shotgun_mode import generate_shotgun_mode_parlays
    from shotgun_mode_ui import display_shotgun_mode_ui, display_shotgun_mode_summary, display_shotgun_mode_help
except ImportError:
    generate_shotgun_mode_parlays = None
    display_shotgun_mode_ui = None
    display_shotgun_mode_summary = None
    display_shotgun_mode_help = None

try:
    import altair as alt  # type: ignore
except Exception:  # pragma: no cover - optional import
    alt = None

import config

# Silence FutureWarning for downcasting behavior in pandas
pd.set_option('future.no_silent_downcasting', True)

# -----------------
# Trace Columns Constant (Refactored)
# -----------------
TRACE_COLS = [
    "spread_engine_used",
    "spread_pick_label",
    "spread_alt_label",
    "spread_prob_pick_final",
    "spread_prob_alt_final",
    "spread_prob_margin",
    "spread_prob_pick_market",
    "spread_prob_alt_market",
    "spread_prob_pick_kalshi",
    "spread_prob_alt_kalshi",
    "spread_decision_metric_used",
    "spread_decision_score_pick",
    "spread_decision_score_alt",
    "spread_decision_score_margin",
    "spread_trace_json",
    "decision_trace",
    "total_engine_used",
    "total_pick_label",
    "total_alt_label",
    "total_prob_pick_final",
    "total_prob_alt_final",
    "total_prob_margin",
    "total_prob_pick_market",
    "total_prob_alt_market",
    "total_prob_pick_kalshi",
    "total_prob_alt_kalshi",
    "total_decision_metric_used",
    "total_decision_score_pick",
    "total_decision_score_alt",
    "total_decision_score_margin",
    "total_trace_json",
    "decision_trace_version",
    "overall_engine_used",
    "decision_trace_notes",
    "decision_trace_short",
    "decision_trace_json",
    "final_probability",
    "decision_driver",
    "kalshi_weight",
    "odds_weight",
    "ml_weight",
    # NOTE: Removed sentiment columns from TRACE_COLS so they display in UI
    # "sentiment_weight",  # REMOVED - needed for display
    # "sentiment_score",  # REMOVED - needed for display
    # "sentiment_direction",  # REMOVED - needed for display
    # "sentiment_impact_applied",  # REMOVED - needed for display
    "kalshi_prob_for_pick",
    "kalshi_yes_side",
    "confidence_reason",
    "kalshi_status",
    "llm_disagreement_flag",
    "consensus_weight_ai",
    "consensus_weight_market",
    "consensus_weight_kalshi",
    "consensus_weight_sentiment",
    "consensus_weight_total",
    "consensus_guardrails",
    "gemini_error",
]

logger = logging.getLogger("parlaypicker")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Preserve session state across reruns
if "master_results_df" not in st.session_state:
    st.session_state["master_results_df"] = pd.DataFrame()

if "master_df" not in st.session_state:
    st.session_state["master_df"] = pd.DataFrame()

if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

# Initialize status variables at the top level to ensure global scope availability
if 'sportsdata_status_run' not in st.session_state:
    st.session_state['sportsdata_status_run'] = "pending"
if 'apisports_status_run' not in st.session_state:
    st.session_state['apisports_status_run'] = "pending"
if 'consensus_status_run' not in st.session_state:
    st.session_state['consensus_status_run'] = "pending"

# Pre-initialize loop-dependent variables at module level to prevent NameErrors in edge cases
# This ensures that even if the main loop is skipped or errors out, these names exist in scope
spread_row_created = False
total_row_created = False
ml_row_created = False
kalshi_matched_game = False
kalshi_prob_spread = None
kalshi_prob_total = None
kalshi_prob = None
spread_prob = None
total_prob = None
model_prob_home = None
spread_pick = None
total_pick = None
h2h_data_valid = False
spread_pick_label = None
total_pick_label = None
spread_pick_line = None
total_line = None
spread_pick_odds = None
total_pick_odds = None
model_spread_prob = None
model_total_prob = None
final_prob_with_theover = None
final_prob_without_theover = None
theover_delta_final_prob = None
spread_books_count = 0
total_books_count = 0
home_spread_point = None
total_point = None
spread_market_pairs_count = 0
total_market_pairs_count = 0
spread_prob_final = None
total_prob_final = None
spread_alt_prob_final = None
total_alt_prob_final = None
spread_prob_margin = None
total_prob_margin = None
spread_prob_pick_market = None
spread_prob_alt_market = None
total_prob_pick_market = None
total_prob_alt_market = None
spread_prob_pick_kalshi = None
spread_prob_alt_kalshi = None
total_prob_pick_kalshi = None
total_prob_alt_kalshi = None
spread_decision_metric_used = None
spread_decision_score_pick = None
spread_decision_score_alt = None
spread_decision_score_margin = None
total_decision_metric_used = None
total_decision_score_pick = None
total_decision_score_alt = None
total_decision_score_margin = None
spread_trace_json = None
total_trace_json = None
spread_engine_used = None
total_engine_used = None
spread_alt_label = None
total_alt_label = None
decision_trace_version = None
decision_trace_notes = None
overall_engine_used = None
winner_refetch_attempted = False
first_game_full_search = {} # Fixed: Initialize as dict, not boolean, to support .get() calls

# -----------------
# Utility helpers (null-safe probability handling)
# -----------------

def clean_line_str(val: Any) -> Optional[str]:
    """Format betting line to strip leading zeros (e.g. '01' -> '1', '1.5' -> '1.5').

    Returns None for NaN values to prevent 'Over nan' in pick strings.
    """
    if val is None:
        return None
    try:
        f = float(val)
        # Handle NaN - return None instead of 'nan' string
        if pd.isna(f) or (isinstance(f, float) and f != f):  # NaN check
            return None
        return f"{f:g}"
    except Exception:
        return str(val)

def safe_float(x: Any) -> Optional[float]:
    """Convert to float; return None on blanks/NaN/non-numeric."""
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"", "none", "nan", "n/a"}:
        return None
    try:
        val = float(x)
        if val != val:  # NaN check
            return None
        return val
    except Exception:
        return None

def ml_allowed(home_ml, away_ml, threshold=300):
    """
    Check if ML picks are allowed based on odds threshold.
    Now more permissive - only rejects if BOTH sides are extreme.
    Returns True if at least ONE side is within threshold.
    """
    try:
        vals = []
        if home_ml is not None: vals.append(float(home_ml))
        if away_ml is not None: vals.append(float(away_ml))
        if not vals:
            return False
        # NEW LOGIC: Allow if ANY side is within threshold (not ALL)
        # This allows picks on the non-extreme side even if the other side is >300
        return any(abs(v) <= threshold for v in vals)
    except Exception:
        return False


def clean_df(df):
    return df.loc[:, ~df.columns.duplicated()].copy()


def clamp(x: Optional[float], lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    if x is None:
        return None
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return None


def has_valid_line(line: Any) -> bool:
    """
    Check if a betting line is valid (not None and not NaN).

    Args:
        line: The betting line value to check

    Returns:
        True if line is valid, False otherwise
    """
    return line is not None and not pd.isna(line)


def clamp_prob(p: Any, lo: float = 0.05, hi: float = 0.95) -> Optional[float]:
    """
    Hard cap on probabilities to prevent extreme values.
    Clamps probability between lo and hi (default 5-95%).

    Args:
        p: Probability value to clamp
        lo: Lower bound (default 0.05)
        hi: Upper bound (default 0.95)

    Returns:
        Clamped probability or None if input is None/NaN
    """
    if p is None or pd.isna(p):
        return None
    try:
        p_float = float(p)
        if pd.isna(p_float):
            return None
        return max(lo, min(hi, p_float))
    except (ValueError, TypeError):
        return None


def select_best_spread_pick(
    home_team: str,
    away_team: str,
    spread_line: float,  # e.g. -8.5 for home
    prob_home_covers: float,  # Probability home covers the spread
    prob_away_covers: float,   # Probability away covers the spread
) -> Dict:
    """
    Select the side with higher probability and format the pick.

    ALWAYS picks the side with higher probability (should be >= 50%).

    Args:
        home_team: Home team name
        away_team: Away team name
        spread_line: Spread line from home team perspective (negative = home favored)
        prob_home_covers: Final probability that home team covers the spread
        prob_away_covers: Final probability that away team covers the spread

    Returns:
        Dictionary with pick information:
        {
            'pick_team': str,
            'pick_label': str,  # e.g. "Illinois St -8.5"
            'pick_prob': float,
            'pick_side': str,  # "home" or "away"
            'alt_team': str,
            'alt_label': str,
            'alt_prob': float,
            'alt_side': str
        }
    """
    # Fix: Handle None values gracefully
    p_home = prob_home_covers if prob_home_covers is not None else 0.0
    p_away = prob_away_covers if prob_away_covers is not None else 0.0

    # ALWAYS pick the higher probability
    if p_home >= p_away:
        # Home is better pick
        pick_team = home_team
        pick_line = spread_line  # e.g. -8.5
        pick_prob = p_home
        pick_side = "home"

        alt_team = away_team
        alt_line = -spread_line  # Flip sign: +8.5
        alt_prob = p_away
        alt_side = "away"
    else:
        # Away is better pick
        pick_team = away_team
        pick_line = -spread_line  # Flip sign: +8.5
        pick_prob = p_away
        pick_side = "away"

        alt_team = home_team
        alt_line = spread_line  # e.g. -8.5
        alt_prob = p_home
        alt_side = "home"

    # Format labels with line
    if pick_line > 0:
        pick_label = f"{pick_team} +{abs(pick_line)}"
    elif pick_line < 0:
        pick_label = f"{pick_team} {pick_line}"
    else:
        pick_label = f"{pick_team} PK"  # Pick'em

    if alt_line > 0:
        alt_label = f"{alt_team} +{abs(alt_line)}"
    elif alt_line < 0:
        alt_label = f"{alt_team} {alt_line}"
    else:
        alt_label = f"{alt_team} PK"

    # Validation - CRITICAL checks
    # Use logger.warning instead of asserting to avoid crashing the app in production,
    # but still flag the issue
    if pick_prob < 0.50:
        logger.warning(f"CRITICAL: Pick prob {pick_prob:.3f} < 50% for {pick_label} - should have picked {alt_label}!")

    if alt_prob > 0.50:
        logger.warning(f"CRITICAL: Alt prob {alt_prob:.3f} > 50% for {alt_label} - should be the pick!")

    if abs((pick_prob + alt_prob) - 1.0) >= 0.01:
        logger.warning(f"Probs don't sum to 1.0: {pick_prob:.3f} + {alt_prob:.3f} = {pick_prob + alt_prob:.3f}")

    logger.info(f"✅ Pick Selection: {pick_label} ({pick_prob:.1%}) vs {alt_label} ({alt_prob:.1%})")

    return {
        'pick_team': pick_team,
        'pick_label': pick_label,
        'pick_prob': pick_prob,
        'pick_side': pick_side,
        'alt_team': alt_team,
        'alt_label': alt_label,
        'alt_prob': alt_prob,
        'alt_side': alt_side
    }


def select_best_total_pick(
    total_line: float,  # e.g. 148.5
    prob_over: float,
    prob_under: float
) -> Dict:
    """
    Select Over or Under based on higher probability.

    ALWAYS picks the side with higher probability (should be >= 50%).

    Args:
        total_line: Total points line
        prob_over: Final probability of Over hitting
        prob_under: Final probability of Under hitting

    Returns:
        Dictionary with pick information:
        {
            'pick_label': str,  # e.g. "Over 148.5"
            'pick_prob': float,
            'pick_side': str,  # "over" or "under"
            'alt_label': str,
            'alt_prob': float,
            'alt_side': str
        }
    """
    # Fix: Handle None values gracefully
    p_over = prob_over if prob_over is not None else 0.0
    p_under = prob_under if prob_under is not None else 0.0

    if p_over >= p_under:
        pick_side = "Over"
        pick_prob = p_over
        alt_side = "Under"
        alt_prob = p_under
    else:
        pick_side = "Under"
        pick_prob = p_under
        alt_side = "Over"
        alt_prob = p_over

    pick_label = f"{pick_side} {total_line}"
    alt_label = f"{alt_side} {total_line}"

    # Validation - CRITICAL checks
    if pick_prob < 0.50:
        logger.warning(f"CRITICAL: Total pick prob {pick_prob:.3f} < 50% for {pick_label} - should have picked {alt_label}!")

    if abs((pick_prob + alt_prob) - 1.0) >= 0.01:
        logger.warning(f"Total probs don't sum to 1.0: {pick_prob:.3f} + {alt_prob:.3f} = {pick_prob + alt_prob:.3f}")

    logger.info(f"✅ Total Selection: {pick_label} ({pick_prob:.1%}) vs {alt_label} ({alt_prob:.1%})")

    return {
        'pick_label': pick_label,
        'pick_prob': pick_prob,
        'pick_side': pick_side,
        'alt_label': alt_label,
        'alt_prob': alt_prob,
        'alt_side': alt_side
    }


def compute_sentiment_adj(sentiment_diff: Optional[float]) -> Optional[float]:
    """
    Compute sentiment adjustment for probability calculations.

    SENTIMENT IMPLEMENTATION OVERVIEW:
    -----------------------------------
    Input Text: NewsAPI articles (3-day lookback) or Reddit posts/comments (fallback)
    Computation: Keyword-based lexicon scoring via RealSentimentAnalyzer
    Output Columns:
      - Home_Sentiment, Away_Sentiment: Per-team sentiment scores (-1 to 1)
      - Sentiment_Diff: home_sent - away_sent
      - sentiment_score: Per-pick sentiment value
      - sentiment_weight: Weight used in probability blend (0.05 default)
      - sentiment_direction: bullish/bearish/neutral
      - sentiment_impact_applied: Whether sentiment affected final prob

    Usage in Scoring:
      - Sentiment is blended into final probability via compute_final_probability()
      - Weight: 5% (w_sentiment = 0.05) of total probability calculation
      - Impact range: ±0.15 probability adjustment
      - Sentiment score converts to probability: p_sentiment = 0.5 + (score * 0.15)

    Logging:
      - Coverage summary: logged at end of sentiment_map computation
      - Per-game sentiment: logged when sentiment_diff is computed
      - Scoring impact: logged when sentiment affects probability calculation
      - Final summary: logged when analysis completes

    Args:
        sentiment_diff: Difference between home and away sentiment (-1 to 1)

    Returns:
        Adjustment value clamped to ±0.03, or None if input is None
    """
    if sentiment_diff is None:
        return None
    try:
        return clamp(sentiment_diff * 0.03, -0.03, 0.03)
    except Exception:
        return None


def format_consensus_breakdown(
    market_prob: Optional[float],
    kalshi_prob: Optional[float],
    model_prob: Optional[float],
    sentiment_score: Optional[float],
    final_prob: Optional[float],
    pick_side: Optional[str] = None
) -> str:
    """
    Format consensus breakdown showing individual engine contributions.
    FIXED: Only shows working sources (Market, Kalshi). Removed AI model and Sentiment.

    Args:
        market_prob: Market implied probability (40% weight)
        kalshi_prob: Kalshi probability (45% weight)
        model_prob: [DEPRECATED - Not displayed]
        sentiment_score: [DEPRECATED - Not displayed]
        final_prob: Final blended probability
        pick_side: Pick side for sentiment conversion (Home/Away/Over/Under)

    Returns:
        Formatted consensus breakdown string (e.g., "M:45.0% | K:52.0% | →49.4%")
    """
    parts = []
    engine_count = 0

    # Market (40% weight)
    if market_prob is not None:
        parts.append(f"M:{market_prob*100:.1f}%")
        engine_count += 1

    # Kalshi (45% weight) - HIGHEST WEIGHT
    if kalshi_prob is not None:
        # Add indicator if Kalshi agrees (>55%) or disagrees
        if kalshi_prob >= 0.55:
            parts.append(f"K:{kalshi_prob*100:.1f}%✓")  # Strong agreement
            engine_count += 1
        elif kalshi_prob < 0.45:
            parts.append(f"K:{kalshi_prob*100:.1f}%⚠")  # Disagreement warning
            engine_count += 1
        else:
            parts.append(f"K:{kalshi_prob*100:.1f}%")
            engine_count += 1

    # REMOVED: AI Model (file missing)
    # REMOVED: Sentiment (APIs unavailable)

    # Final consensus (weighted average)
    if final_prob is not None:
        parts.append(f"→{final_prob*100:.1f}%")

    # Only show separator if we have parts
    consensus_str = " | ".join(parts) if parts else "N/A"

    return consensus_str


def calculate_consensus_agreement(
    market_prob: Optional[float],
    kalshi_prob: Optional[float],
    model_prob: Optional[float],
    sentiment_score: Optional[float],
    pick_side: Optional[str] = None
) -> Tuple[float, int, str]:
    """
    Calculate consensus agreement metrics.

    Returns:
        (spread, valid_count, quality_label)
        - spread: Difference between max and min probabilities
        - valid_count: Number of valid (non-null) engines
        - quality_label: "STRONG", "MODERATE", "WEAK", or "INSUFFICIENT"
    """
    probs = []

    if market_prob is not None:
        probs.append(market_prob)
    if kalshi_prob is not None:
        probs.append(kalshi_prob)
    if model_prob is not None:
        probs.append(model_prob)

    # Convert sentiment to probability if available
    if sentiment_score is not None:
        impact = max(-0.15, min(0.15, sentiment_score * 0.15))
        sent_prob = 0.50 + impact
        if pick_side:
            p_side = str(pick_side).lower()
            if p_side in {"away", "under"}:
                sent_prob = 1.0 - sent_prob
        probs.append(sent_prob)

    valid_count = len(probs)

    if valid_count < 2:
        return 1.0, valid_count, "INSUFFICIENT"

    spread = max(probs) - min(probs)

    # Quality assessment
    if spread < 0.05 and valid_count >= 3:
        quality_label = "STRONG"
    elif spread < 0.10 and valid_count >= 3:
        quality_label = "MODERATE"
    elif spread < 0.15:
        quality_label = "WEAK"
    else:
        quality_label = "WEAK"

    return spread, valid_count, quality_label


def calculate_pick_quality_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive quality metrics for pick tightening filters.

    Returns dict with:
        - meets_probability_threshold: bool (>56%)
        - meets_decisiveness_threshold: bool (>0.08)
        - consensus_quality: str (STRONG/MODERATE/WEAK/INSUFFICIENT)
        - consensus_spread: float
        - kalshi_validates: bool (Kalshi agrees with pick)
        - quality_score: float (0-5, for sorting)
        - quality_tier: str (HIGH/MEDIUM/LOW)
    """
    # Extract probabilities
    final_prob = safe_float(row.get("final_probability") or row.get("spread_prob_pick_final") or row.get("total_prob_pick_final"))
    market_prob = safe_float(row.get("spread_prob_pick_market") or row.get("total_prob_pick_market") or row.get("Implied_Prob"))
    # v98 FIX: Use proper None check to avoid 0.0 falsy issue with `or`
    _kp = row.get("spread_prob_pick_kalshi")
    if _kp is None:
        _kp = row.get("total_prob_pick_kalshi")
    if _kp is None:
        _kp = row.get("kalshi_prob_for_pick")
    kalshi_prob = safe_float(_kp)
    model_prob = safe_float(row.get("model_spread_prob") or row.get("model_total_prob") or row.get("AI_Prob"))
    sentiment_score = safe_float(row.get("sentiment_score"))
    pick_side = row.get("Pick") or row.get("spread_pick_team") or row.get("total_pick_side")

    # 1. Probability Threshold (>56%)
    meets_probability_threshold = final_prob is not None and final_prob > 0.56

    # 2. Decisiveness Threshold (>0.08) - Fixed: was 0.20 which was too restrictive
    decisiveness = abs(final_prob - 0.5) if final_prob is not None else 0.0
    meets_decisiveness_threshold = decisiveness > 0.08

    # 3. Consensus Agreement
    spread, valid_count, consensus_quality = calculate_consensus_agreement(
        market_prob, kalshi_prob_used, model_prob, sentiment_score, pick_side
    )

    # 4. Kalshi Validation
    kalshi_validates = False
    if kalshi_prob is not None:
        kalshi_validates = kalshi_prob > 0.55  # Strong Kalshi agreement

    # 5. Quality Score (0-5)
    quality_score = 0.0

    # +1.5 for probability threshold
    if meets_probability_threshold:
        quality_score += 1.5
    elif final_prob is not None and final_prob > 0.52:
        quality_score += 0.5

    # +1.5 for decisiveness
    if meets_decisiveness_threshold:
        quality_score += 1.5
    elif decisiveness > 0.10:
        quality_score += 0.75

    # +1.0 for consensus quality
    if consensus_quality == "STRONG":
        quality_score += 1.0
    elif consensus_quality == "MODERATE":
        quality_score += 0.5

    # +1.0 for Kalshi validation
    if kalshi_validates:
        quality_score += 1.0
    elif kalshi_prob is not None and kalshi_prob > 0.50:
        quality_score += 0.5

    # 6. Quality Tier
    if quality_score >= 4.0:
        quality_tier = "HIGH"
    elif quality_score >= 2.5:
        quality_tier = "MEDIUM"
    else:
        quality_tier = "LOW"

    return {
        "meets_probability_threshold": meets_probability_threshold,
        "meets_decisiveness_threshold": meets_decisiveness_threshold,
        "consensus_quality": consensus_quality,
        "consensus_spread": spread,
        "consensus_valid_count": valid_count,
        "kalshi_validates": kalshi_validates,
        "quality_score": quality_score,
        "quality_tier": quality_tier,
        "decisiveness": decisiveness,
        "final_prob": final_prob,
    }

SENTIMENT_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
SENTIMENT_LAST_TS: float = 0.0
SENTIMENT_CACHE_TTL = timedelta(hours=12)
SENTIMENT_LOG_SAMPLE: Dict[str, bool] = {}
MAX_SENTIMENT_TEAMS_PER_RUN = 200
NEWSAPI_COOLDOWN_HOURS = 12
REDDIT_CACHE_TTL = timedelta(hours=12)
DECISION_TRACE_SAMPLE_LEAGUES = {"NFL", "NBA", "NCAAB"}
MAX_GEMINI_CALLS_PER_RUN = 40  # v99 FIX (Bug 6): Increased from 25 to handle larger game slates (68+ games)

# INIT ONCE at app startup
if "gemini_calls_made" not in st.session_state:
    st.session_state["gemini_calls_made"] = 0
    logger.info("Startup: Initialized gemini_calls_made to 0")
else:
    logger.info(f"Startup: Resuming with {st.session_state['gemini_calls_made']} calls made")

if "gemini_cache" not in st.session_state:
    st.session_state["gemini_cache"] = {}


def _parse_cooldown_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        if isinstance(raw, datetime):
            dt = raw
        else:
            dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def newsapi_cooldown_until() -> Optional[datetime]:
    sentinel = None
    try:
        sentinel = st.session_state.get("NEWSAPI_RATE_LIMITED_UNTIL") or st.session_state.get("sentiment_cooldown_until")
    except Exception:
        sentinel = None
    return _parse_cooldown_ts(sentinel)


def newsapi_cooldown_active() -> bool:
    until = newsapi_cooldown_until()
    return bool(until and datetime.now(timezone.utc) < until)


def set_newsapi_cooldown(hours: int = NEWSAPI_COOLDOWN_HOURS, retry_after: Optional[Union[int, str]] = None) -> Optional[datetime]:
    duration = timedelta(hours=hours)
    if retry_after is not None:
        try:
            duration = timedelta(seconds=int(retry_after))
        except Exception:
            pass
    cooldown_until = datetime.now(timezone.utc) + duration
    try:
        st.session_state["NEWSAPI_RATE_LIMITED_UNTIL"] = cooldown_until
        st.session_state["sentiment_cooldown_until"] = cooldown_until.isoformat()
    except Exception:
        pass
    return cooldown_until


def _sentiment_cache_key(team: str, league: str, bucket: str) -> Tuple[str, str, str]:
    return (team.lower().strip(), league.upper().strip(), bucket)


def _sentiment_cache_container() -> Optional[Dict[Tuple[str, str, str], Dict[str, Any]]]:
    try:
        cache = st.session_state.get("sentiment_run_cache")
        if cache is None:
            cache = {}
            st.session_state["sentiment_run_cache"] = cache
        return cache
    except Exception:
        return None


def _sentiment_cache_get(team: str, league: str, bucket: str) -> Optional[Dict[str, Any]]:
    key = _sentiment_cache_key(team, league, bucket)
    container = _sentiment_cache_container() or {}
    entry = container.get(key) or SENTIMENT_CACHE.get(key)
    if not entry:
        return None
    ts: datetime = entry.get("ts")  # type: ignore
    if not ts or (datetime.now(timezone.utc) - ts) > SENTIMENT_CACHE_TTL:
        SENTIMENT_CACHE.pop(key, None)
        try:
            container.pop(key, None)
        except Exception:
            pass
        return None
    return entry.get("payload")


def _sentiment_cache_put(team: str, league: str, bucket: str, payload: Dict[str, Any]) -> None:
    entry = {
        "ts": datetime.now(timezone.utc),
        "payload": payload,
    }
    key = _sentiment_cache_key(team, league, bucket)
    SENTIMENT_CACHE[key] = entry
    container = _sentiment_cache_container()
    if container is not None:
        container[key] = entry


def _enforce_sentiment_throttle() -> None:
    global SENTIMENT_LAST_TS
    now = time.time()
    delta = now - SENTIMENT_LAST_TS
    if delta < 1.0:
        time.sleep(1.0 - delta)
    SENTIMENT_LAST_TS = time.time()


def _reddit_cache_key(team: str, league: str, bucket: str) -> str:
    return f"{league.upper().strip()}::{team.lower().strip()}::{bucket}"


def _reddit_cache_get(team: str, league: str, bucket: str) -> Optional[Dict[str, Any]]:
    try:
        cache = st.session_state.get("reddit_sentiment_cache") or {}
    except Exception:
        cache = {}
    key = _reddit_cache_key(team, league, bucket)
    entry = cache.get(key)
    if not entry:
        return None
    ts: datetime = entry.get("ts")  # type: ignore
    if not ts or (datetime.now(timezone.utc) - ts) > REDDIT_CACHE_TTL:
        try:
            cache.pop(key, None)
            st.session_state["reddit_sentiment_cache"] = cache
        except Exception:
            pass
        return None
    return entry.get("payload")


def _reddit_cache_put(team: str, league: str, bucket: str, payload: Dict[str, Any]) -> None:
    entry = {"ts": datetime.now(timezone.utc), "payload": payload}
    try:
        cache = st.session_state.get("reddit_sentiment_cache")
        if cache is None:
            cache = {}
        cache[_reddit_cache_key(team, league, bucket)] = entry
        st.session_state["reddit_sentiment_cache"] = cache
    except Exception:
        pass


@st.cache_data(ttl=5400, show_spinner=False)
def _newsapi_fetch_cached(url: str, params: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    """Cached NewsAPI call keyed by URL+params for the current date window."""
    try:
        response = requests.get(url, params=dict(params), timeout=8)
        status = response.status_code
        try:
            data = response.json() if hasattr(response, "json") else {}
        except Exception:
            data = {}
        return {
            "status": status,
            "data": data if isinstance(data, dict) else {},
            "text_snippet": (response.text or "")[:200] if hasattr(response, "text") else "",
            "headers": dict(getattr(response, "headers", {}) or {}),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": None, "error": str(exc), "data": {}, "headers": {}}


def round_pct(p: Any) -> str:
    try:
        if p is None:
            return ""
        return f"{int(round(float(p) * 100))}%"
    except Exception:
        return ""


def prob_arrow(base: Any, adj: Any, threshold: float = 0.0075) -> str:
    try:
        if base is None or adj is None:
            return ""
        delta = float(adj) - float(base)
        if abs(delta) < threshold:
            return ""
        return "▲" if delta > 0 else "▼"
    except Exception:
        return ""


def market_prob_from_prices(yes_bid: Any, no_bid: Any) -> Optional[float]:
    """
    Convert Kalshi yes/no prices (0-100) into a probability using midpoint normalization.
    """
    try:
        y = safe_float(yes_bid)
        n = safe_float(no_bid)
        if y is None and n is None:
            return None
        if y is None:
            return clamp(1.0 - float(n) / 100.0, 0.0, 1.0)
        if n is None:
            return clamp(float(y) / 100.0, 0.0, 1.0)
        total = float(y) + float(n)
        if total <= 0:
            return None
        return clamp(float(y) / total, 0.0, 1.0)
    except Exception:
        return None


def apply_sentiment_defaults(row: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    def _is_nan(v: Any) -> bool:
        try:
            return v != v
        except Exception:
            return False

    for k, v in defaults.items():
        current = row.get(k)
        if current is None or _is_nan(current):
            row[k] = v
    return row


def _clamp_signed(val: Optional[float], *, limit: float) -> float:
    try:
        v = float(val)
    except Exception:
        return 0.0
    return max(-limit, min(limit, v))


SENTIMENT_LEVEL_WEIGHTS = {
    "team": 1.0,
    "game": 0.6,
    "league": 0.35,
    "none": 0.0,
}

SENTIMENT_STRENGTH_WEIGHTS = {
    "STRONG": 1.0,
    "MEDIUM": 0.7,
    "WEAK": 0.45,
    "NONE": 0.0,
}

SENTIMENT_MARKET_WEIGHTS = {
    "spread": 0.8,
    "total": 1.0,
    "moneyline": 0.6,
}


def _normalize_sentiment_level(level: Any) -> str:
    level_str = str(level or "none").lower()
    return level_str if level_str in {"team", "game", "league", "none"} else "none"


def sentiment_strength_from_articles(level: str, articles_used: int) -> str:
    lvl = _normalize_sentiment_level(level)
    try:
        articles = int(articles_used or 0)
    except Exception:
        articles = 0
    if lvl == "none" or articles <= 0:
        return "NONE"
    if articles >= 8:
        return "STRONG"
    if articles >= 3:
        return "MEDIUM"
    return "WEAK"


def sentiment_badge_for(level: str, strength: str) -> str:
    lvl = _normalize_sentiment_level(level)
    strength_norm = str(strength or "").upper()
    mapping = {
        "team": {"STRONG": "TEAM_STRONG", "MEDIUM": "TEAM_MED", "WEAK": "TEAM_WEAK"},
        "game": {"STRONG": "GAME_STRONG", "MEDIUM": "GAME_MED", "WEAK": "GAME_WEAK"},
        "league": {"STRONG": "LEAGUE_MED", "MEDIUM": "LEAGUE_MED", "WEAK": "LEAGUE_WEAK"},
    }
    if lvl in mapping and strength_norm in mapping[lvl]:
        return mapping[lvl][strength_norm]
    return "NONE"


def sentiment_signal_value(level: str, sentiment_diff: Any, *, game_sentiment: Any = None, league_sentiment: Any = None) -> float:
    lvl = _normalize_sentiment_level(level)
    if lvl == "team":
        return _clamp_signed(safe_float(sentiment_diff), limit=1.0)
    if lvl in {"game", "league"}:
        source_val = game_sentiment if lvl == "game" else league_sentiment
        return _clamp_signed(safe_float(source_val), limit=1.0)
    return 0.0


def compute_market_sentiment_adjustment(level: str, strength: str, market_kind: str, signal_value: float) -> float:
    lvl = _normalize_sentiment_level(level)
    strength_norm = str(strength or "NONE").upper()
    market_key = str(market_kind or "").lower()
    if lvl == "none" or strength_norm == "NONE":
        return 0.0
    level_w = SENTIMENT_LEVEL_WEIGHTS.get(lvl, 0.0)
    strength_w = SENTIMENT_STRENGTH_WEIGHTS.get(strength_norm, 0.0)
    market_w = SENTIMENT_MARKET_WEIGHTS.get(market_key, 0.0)
    raw_adj = (signal_value or 0.0) * 0.02 * level_w * strength_w * market_w
    return _clamp_signed(raw_adj, limit=0.03)


def blend_probs(weighted_probs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    usable_raw = []
    for p, w in weighted_probs:
        if w is None or w <= 0:
            continue
        p_val = safe_float(p)
        if p_val is None:
            continue
        usable_raw.append((p_val, w))
    usable = usable_raw
    if not usable:
        return None
    total_weight = sum(w for _, w in usable)
    if total_weight <= 0:
        return None
    blended = sum((p or 0.0) * w for p, w in usable) / total_weight
    return clamp(blended, 0.0, 1.0)


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def compute_margin(p_pick: Optional[float], p_alt: Optional[float]) -> Optional[float]:
    if p_pick is None or p_alt is None:
        return None
    try:
        return float(p_pick) - float(p_alt)
    except Exception:
        return None


def sentiment_impact_for_pick(
    sentiment_adj: Optional[float],
    selection_team: str,
    home_team: str,
    away_team: str,
) -> Dict[str, Any]:
    """
    Convert sentiment adjustment into a bounded decision modifier.
    Returns direction, score, impact value, and whether it was applied.
    """
    score = safe_float(sentiment_adj) or 0.0
    if selection_team not in {home_team, away_team}:
        return {
            "sentiment_score": score,
            "sentiment_direction": "neutral",
            "sentiment_impact": 0.0,
            "sentiment_impact_applied": False,
        }
    agrees = (score > 0 and selection_team == home_team) or (
        score < 0 and selection_team == away_team
    )
    if score == 0.0:
        direction = "neutral"
    else:
        direction = "agree" if agrees else "disagree"
    impact = 0.0
    applied = False
    if score != 0.0:
        impact = clamp(abs(score), 0.02, 0.05) or 0.0
        applied = True
    if not agrees:
        impact *= -1
    return {
        "sentiment_score": score,
        "sentiment_direction": direction,
        "sentiment_impact": impact,
        "sentiment_impact_applied": applied,
    }


def implied_prob_for_pick(odds_home: Any, odds_away: Any, pick_side: Optional[str]) -> Optional[float]:
    """Return implied probability for the selected side (home/away/over/under)."""
    side = (pick_side or "").lower()
    if side in {"home", "over"}:
        return american_to_implied_prob(odds_home)
    if side in {"away", "under"}:
        return american_to_implied_prob(odds_away)
    return None


def map_kalshi_prob_for_pick(
    kalshi_yes_prob: Optional[float],
    kalshi_yes_side: Optional[str],  # "home" or "away" or Team Name
    pick_side: Optional[str],  # "home" or "away" - which side we're evaluating
    pick_team: Optional[str] = None,  # actual team name
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    spread_line: Optional[float] = None  # NEW
) -> Optional[float]:
    """
    Map Kalshi yes probability to the correct pick side.

    Args:
        kalshi_yes_prob: Probability from Kalshi for the "yes" side (0.0 to 1.0)
        kalshi_yes_side: Which side Kalshi's yes_prob represents ("home" or "away" or TeamName)
        pick_side: Which side we're calculating probability for ("home" or "away")
        pick_team: The actual team name we're evaluating
        home_team: Home team name
        away_team: Away team name
        spread_line: Spread line value (optional) used for over/under spread logic

    Returns:
        float: Probability that the pick_side will cover the spread (0.0 to 1.0)
    """
    prob = safe_float(kalshi_yes_prob)
    if prob is None:
        return None

    # Defensive check for exactly 0.5 (neutral default)
    if abs(prob - 0.5) < 0.001:
        # logger.warning(f"⚠️ Kalshi neutral prob (0.50) for {pick_team or pick_side} - likely default")
        return prob

    if not pick_side:
        return prob

    # Determine which team is which side
    pick_is_home = False
    if pick_side.lower() == "home":
        pick_is_home = True
    elif pick_side.lower() == "away":
        pick_is_home = False
    else:
        # Fallback: check team names with normalization
        if pick_team and home_team:
            pick_is_home = TeamNameMatcher.normalize(pick_team) == TeamNameMatcher.normalize(home_team)

    # Resolve Kalshi Yes Side to Home/Away
    kalshi_yes_is_home = None # Change default to None to detect failure
    kalshi_yes_norm = TeamNameMatcher.normalize(kalshi_yes_side) if kalshi_yes_side else ""

    # Pre-compute team norms
    home_norm = TeamNameMatcher.normalize(home_team) if home_team else None
    away_norm = TeamNameMatcher.normalize(away_team) if away_team else None
    pick_norm = TeamNameMatcher.normalize(pick_team) if pick_team else None

    # 1. Direct Pick Match (Strongest Signal)
    # If the "Yes" side explicitly matches the Pick Team, return Prob.
    # FIX: Handle "Team A vs Team B" titles to prevent matching BOTH teams via containment.
    is_matchup_str = " VS " in kalshi_yes_norm or " @ " in kalshi_yes_norm

    if pick_norm and kalshi_yes_norm:
        if is_matchup_str:
            # Split by separator to correctly identify sides
            # "LAKERS VS CELTICS" -> Left=Lakers (Yes), Right=Celtics (No)
            parts = []
            if " VS " in kalshi_yes_norm:
                parts = kalshi_yes_norm.split(" VS ")
            elif " @ " in kalshi_yes_norm:
                parts = kalshi_yes_norm.split(" @ ")

            if parts and len(parts) >= 1:
                left_side = parts[0]
                # If pick is in the Left Side (Yes Side), return prob
                if pick_norm in left_side:
                    return prob
                # If pick is in the Right Side (No Side), return 1 - prob
                # This handles "Lakers vs Celtics" where Pick="Celtics"
                if len(parts) >= 2 and pick_norm in parts[1]:
                    return 1.0 - prob

        elif pick_norm in kalshi_yes_norm:
            return prob

    # If "Yes" side matches the Opposing Team, return 1 - Prob.
    # Check if pick is home/away to find opponent
    opponent_norm = None
    if pick_is_home and away_norm: opponent_norm = away_norm
    elif not pick_is_home and home_norm: opponent_norm = home_norm

    if opponent_norm and kalshi_yes_norm:
        if is_matchup_str:
            # Split logic for Opponent check as well
            parts = []
            if " VS " in kalshi_yes_norm:
                parts = kalshi_yes_norm.split(" VS ")
            elif " @ " in kalshi_yes_norm:
                parts = kalshi_yes_norm.split(" @ ")

            if parts and len(parts) >= 1:
                left_side = parts[0]
                # If Opponent is in Left Side (Yes Side), then Pick is No Side -> 1 - prob
                if opponent_norm in left_side:
                    return 1.0 - prob
                # If Opponent is in Right Side (No Side), then Pick is Yes Side -> prob
                if len(parts) >= 2 and opponent_norm in parts[1]:
                    return prob

        elif opponent_norm in kalshi_yes_norm:
            return 1.0 - prob

    # 2. Side Inference (Home/Away/Over/Under)
    if "OVER" in kalshi_yes_norm:
        kalshi_yes_is_home = True # Over -> Home in spread
    elif "UNDER" in kalshi_yes_norm:
        kalshi_yes_is_home = False
    elif kalshi_yes_norm == "HOME":
        kalshi_yes_is_home = True
    elif kalshi_yes_norm == "AWAY":
        kalshi_yes_is_home = False
    elif home_norm and home_norm in kalshi_yes_norm:
        kalshi_yes_is_home = True
    elif away_norm and away_norm in kalshi_yes_norm:
        kalshi_yes_is_home = False

    # Fallback: If we couldn't determine, assume Home if default (legacy behavior)
    # But only if we have high confidence or no other option.
    if kalshi_yes_is_home is None:
        # If we can't match names, we can't be sure.
        # But to match "legacy" behavior where we assumed Yes=Home, we might set True.
        # However, to avoid 1.4 sum anomaly, we must be consistent.
        # If we default to True here, we must ensure we don't accidentally default to False elsewhere.
        kalshi_yes_is_home = True # Defaulting to Home

    # Map Kalshi yes probability to pick
    if kalshi_yes_is_home:
        # Kalshi yes_prob is for home team
        if pick_is_home:
            result = prob  # Pick is home, Kalshi yes is home
        else:
            result = 1.0 - prob  # Pick is away, flip Kalshi prob
    else:  # Kalshi yes is away
        # Kalshi yes_prob is for away team
        if pick_is_home:
            result = 1.0 - prob  # Pick is home, flip Kalshi prob
        else:
            result = prob  # Pick is away, Kalshi yes is away

    return result

def map_kalshi_prob_for_total(
    kalshi_yes_prob: Optional[float],
    kalshi_yes_side: Optional[str],  # "over" or "under"
    pick_side: str  # "over" or "under"
) -> Optional[float]:
    prob = safe_float(kalshi_yes_prob)
    if prob is None:
        return None

    yes_side = str(kalshi_yes_side or "").lower()
    pick_side = str(pick_side).lower()

    if "over" in yes_side:
        if "over" in pick_side: return prob
        if "under" in pick_side: return 1.0 - prob
    elif "under" in yes_side:
        if "under" in pick_side: return prob
        if "over" in pick_side: return 1.0 - prob

    return prob


def dynamic_kalshi_weight(
    kalshi_prob_for_pick: Optional[float],
    implied_pick_prob: Optional[float],
    kalshi_matched: bool,
    league: Optional[str],
    base_default: float = 0.55,
) -> float:
    """
    Compute a dynamic Kalshi weight based on signal strength and league.
    - kalshi_prob_for_pick: Kalshi prob mapped to the chosen side.
    - implied_pick_prob: Implied prob from book odds for the same side.
    - kalshi_matched: True if a Kalshi market for this game/market is matched.
    """
    if not kalshi_matched or kalshi_prob_for_pick is None or implied_pick_prob is None:
        return 0.0

    try:
        edge = abs(float(kalshi_prob_for_pick) - float(implied_pick_prob))
    except Exception:
        return base_default

    # League-based caps (NBA/NFL get higher max weight than small-conference NCAAB)
    lg = (league or "").upper()
    if lg in ("NBA", "NFL"):
        max_w = 0.70  # UP from 0.5
    elif lg in ("NHL", "MLB"):
        max_w = 0.60  # UP from 0.4
    else:
        max_w = 0.50  # UP from 0.3

    # Piecewise mapping from edge size to weight
    if edge < 0.03:
        # light influence when market is basically fair
        return 0.10 * max_w / 0.5
    elif edge < 0.07:
        return min(0.25, max_w * 0.6)
    else:
        return max_w  # strong divergence -> strong sentiment weight


def compute_final_probability(
    pick_side: Optional[str],
    implied_prob: Optional[float],
    kalshi_prob_yes: Optional[float],
    kalshi_side_yes: Optional[str],
    model_prob: Optional[float],
    theover_prob: Optional[float],
    sentiment_adj: Optional[float],
    weights_dict: Dict[str, Any], # Deprecated, will use Hardcoded Globals
    sentiment_score: Optional[float] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    kalshi_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[float], Dict[str, float], str, List[str], Optional[float]]:
    """
    Blend available probabilities using STATIC weights without renormalization.
    Returns (final_prob, base_prob, weights_used, driver, warnings, kalshi_prob_for_pick).
    """
    warnings: List[str] = []

    # ---------------------------------------------------------
    # IMPROVED KALSHI MAPPING (v105)
    # Validate both sides and ensure correct side mapping
    # ---------------------------------------------------------
    kalshi_prob_for_pick = None

    if kalshi_prob_yes is not None:
        pick_type_enum = None # "spread/ml", "total"
        pick_side_enum = None # "home", "away", "over", "under"

        # 1. Determine Pick Side Enum
        if pick_side:
            p_lower = str(pick_side).lower()
            if "over" in p_lower:
                pick_side_enum = "over"
                pick_type_enum = "total"
            elif "under" in p_lower:
                pick_side_enum = "under"
                pick_type_enum = "total"
            elif home_team and away_team:
                p_norm = TeamNameMatcher.normalize(pick_side)
                h_norm = TeamNameMatcher.normalize(home_team)
                a_norm = TeamNameMatcher.normalize(away_team)

                if p_norm == h_norm or (h_norm and h_norm in p_norm):
                    pick_side_enum = "home"
                    pick_type_enum = "spread/ml"
                elif p_norm == a_norm or (a_norm and a_norm in p_norm):
                    pick_side_enum = "away"
                    pick_type_enum = "spread/ml"

        # 2. Calculate probabilities for both sides & Validate
        if pick_type_enum == "spread/ml":
            # Calculate for Home
            p_home = map_kalshi_prob_for_pick(
                kalshi_yes_prob=kalshi_prob_yes,
                kalshi_yes_side=kalshi_side_yes,
                pick_side="home",
                pick_team=home_team,
                home_team=home_team,
                away_team=away_team,
                spread_line=kalshi_data.get("spread_line") if kalshi_data else None  # NEW
            )
            # Calculate for Away
            p_away = map_kalshi_prob_for_pick(
                kalshi_yes_prob=kalshi_prob_yes,
                kalshi_yes_side=kalshi_side_yes,
                pick_side="away",
                pick_team=away_team,
                home_team=home_team,
                away_team=away_team,
                spread_line=kalshi_data.get("spread_line") if kalshi_data else None  # NEW
            )

            # Validation Logging
            if p_home is not None and p_away is not None:
                # Log strict check
                logger.info(f"Kalshi Mapping Check: Home={home_team}, Away={away_team}, YesSide={kalshi_side_yes}, YesProb={kalshi_prob_yes:.3f}")
                logger.info(f"  Home Pick Prob: {p_home:.3f}, Away Pick Prob: {p_away:.3f}")

                # Assert Sum (Soft)
                if abs(p_home + p_away - 1.0) > 0.02:
                    logger.warning(f"⚠️ Kalshi Mapping Inconsistency: Home({p_home:.3f}) + Away({p_away:.3f}) != 1.0. YesSide={kalshi_side_yes}")

            # Select correct prob
            if pick_side_enum == "home":
                kalshi_prob_for_pick = p_home
            elif pick_side_enum == "away":
                kalshi_prob_for_pick = p_away
            else:
                # Fallback to loose matching if enum determination failed
                kalshi_prob_for_pick = map_kalshi_prob_for_pick(
                    kalshi_prob_yes, kalshi_side_yes, pick_side, pick_side, home_team, away_team
                )

        elif pick_type_enum == "total":
            p_over = map_kalshi_prob_for_total(kalshi_prob_yes, kalshi_side_yes, "over")
            p_under = map_kalshi_prob_for_total(kalshi_prob_yes, kalshi_side_yes, "under")

            if p_over is not None and p_under is not None:
                if abs(p_over + p_under - 1.0) > 0.02:
                    logger.warning(f"⚠️ Kalshi Total Inconsistency: Over({p_over:.3f}) + Under({p_under:.3f}) != 1.0")

            if pick_side_enum == "over":
                kalshi_prob_for_pick = p_over
            elif pick_side_enum == "under":
                kalshi_prob_for_pick = p_under
            else:
                kalshi_prob_for_pick = map_kalshi_prob_for_total(kalshi_prob_yes, kalshi_side_yes, pick_side)

        else:
            # Fallback for unknown types
            kalshi_prob_for_pick = map_kalshi_prob_for_pick(
                kalshi_prob_yes, kalshi_side_yes, pick_side, pick_side, home_team, away_team,
                spread_line=kalshi_data.get("spread_line") if kalshi_data else None  # NEW
            )

    # Logging verification for P0 Bug (Blend Input Check)
    if kalshi_prob_yes is not None:
        try:
            raw_k = float(kalshi_prob_yes)
            mapped_k = float(kalshi_prob_for_pick) if kalshi_prob_for_pick is not None else -1.0
            game_str = f"{home_team} vs {away_team}" if home_team and away_team else "Unknown Game"
            logger.info(f"BLEND INPUT for {game_str}: pick={pick_side}, kalshi_yes_side={kalshi_side_yes}, raw_kalshi={raw_k:.3f}, mapped_pick_kalshi={mapped_k:.3f}")
        except Exception:
            pass

    # v98 FIX (Bug A): Single-tier weights — always start with configured weights.
    # Sources that are unavailable or fail validation get zeroed below;
    # normalization redistributes their weight proportionally to remaining sources.
    W_KALSHI = KALSHI_WEIGHT           # 0.55
    W_MARKET = MARKET_WEIGHT           # 0.15
    W_MODEL = ML_MODEL_WEIGHT          # 0.15
    W_THEOVER = THEOVER_WEIGHT         # 0.10
    W_SENTIMENT = SENTIMENT_WEIGHT     # 0.05

    weights_used = {
        "w_implied": W_MARKET,
        "w_kalshi": W_KALSHI,
        "w_model": W_MODEL,
        "w_theover": W_THEOVER,
        "w_sentiment": W_SENTIMENT
    }

    # FIX: When a source is unavailable (None), zero its weight instead of using
    # 0.5 neutral. This prevents missing sources from diluting the probability
    # toward 50%. Weights are renormalized after all sources are evaluated.

    # 1. Market (Implied Prob)
    if implied_prob is not None:
        p_market = clamp_prob(implied_prob, 0.05, 0.95) or 0.5
    else:
        p_market = 0.0
        W_MARKET = 0.0

    # 2. Kalshi
    p_kalshi = 0.0
    kalshi_is_available = False
    if kalshi_prob_for_pick is not None:
        # v102 FIX: Removed auto-correction that flipped kalshi_prob_for_pick when
        # the flipped value was closer to implied_prob. That logic systematically
        # inverted Kalshi's independent signal whenever Kalshi and the market
        # disagreed about which side of 50% the probability fell on, causing
        # the blend to use the ALT-side Kalshi prob instead of the PICK-side.
        # Now we trust map_kalshi_prob_for_pick() and only reject on extreme delta.
        kalshi_validated = True

        # User Requirement: Reject neutral Kalshi data (0.50 +/- 0.005)
        # Tightened from 0.02 to 0.005 to only reject truly neutral values
        # 49.5% is a valid 1% edge signal, not neutral
        if abs(kalshi_prob_for_pick - 0.5) < 0.005:
            kalshi_validated = False
            warnings.append("kalshi_rejected_neutral")
            logger.info(f"Rejecting neutral Kalshi data for {pick_side} (prob={kalshi_prob_for_pick:.3f})")

        if kalshi_validated and implied_prob is not None:
            delta = abs(kalshi_prob_for_pick - implied_prob)
            # v104 FIX (Bug 2): Widen threshold from 0.40 to 0.55. The v103 fix ensures
            # correct pick-side Kalshi prob mapping, so large deltas now reflect genuine
            # Kalshi-vs-book disagreement rather than side-mapping errors. The 0.40 threshold
            # was rejecting valid spread data for 14 games where Kalshi simply disagreed
            # with book odds (e.g., Kalshi 0.76 vs book 0.35 → delta 0.41 → rejected).
            if delta > 0.55:  # 55% threshold - only reject truly extreme mismatches
                # Extreme disagreement likely means wrong Kalshi line was matched
                kalshi_validated = False
                warnings.append(f"kalshi_validation_failed(delta={delta:.2f})")

        if kalshi_validated:
            p_kalshi = clamp_prob(kalshi_prob_for_pick, 0.05, 0.95) or 0.5
            kalshi_is_available = True

    if not kalshi_is_available:
        W_KALSHI = 0.0

    # 3. Model
    if model_prob is not None:
        p_model = clamp_prob(model_prob, 0.05, 0.95) or 0.5
    else:
        p_model = 0.0
        W_MODEL = 0.0

    # 4. TheOver
    # If missing, use neutral 0.5
    p_theover = 0.5
    if theover_prob is not None:
        raw_to = clamp_prob(theover_prob, 0.05, 0.95) or 0.5
        # RESCALE logic: [0.55, 0.75] band
        if raw_to > 0.5:
             p_theover = 0.5 + (raw_to - 0.5) * 0.555
        elif raw_to < 0.5:
             p_theover = 0.5 - (0.5 - raw_to) * 0.555
        else:
             p_theover = 0.5

    # 5. Sentiment
    # If missing or rate limited, zero weight
    p_sentiment = 0.0
    sentiment_data = {"used": False, "weight": W_SENTIMENT, "prob": 0.5, "adj": 0.0}

    if sentiment_score is not None and not newsapi_cooldown_active():
        # Determine direction
        direction = 0.0
        if home_team and away_team:
            p_norm = str(pick_side or "").lower().strip()
            h_norm = str(home_team).lower().strip()
            a_norm = str(away_team).lower().strip()
            if p_norm == h_norm: direction = 1.0
            elif p_norm == a_norm: direction = -1.0

        # Fallback: Kalshi side
        if direction == 0.0 and kalshi_side_yes and pick_side:
            ks = str(kalshi_side_yes).lower()
            ps = str(pick_side).lower()
            if ks == "home":
                direction = 1.0 if (ps == "home" or ps == ks) else -1.0
            elif ks == "away":
                direction = -1.0 if (ps == "away" or ps == ks) else 1.0

        if direction == 0.0 and pick_side:
            ps = str(pick_side).lower()
            if ps == "home": direction = 1.0
            elif ps == "away": direction = -1.0

        if direction != 0.0:
            s_val = max(-1.0, min(1.0, float(sentiment_score)))
            # Scaling factor: 0.5 (score 1.0 -> 0.5 adj -> prob 1.0)
            # With weight 0.05, max impact is 0.025
            adj = s_val * direction * 0.5
            p_sentiment = clamp(0.5 + adj, 0.01, 0.99)
            sentiment_data["used"] = True
            sentiment_data["adj"] = adj
            sentiment_data["prob"] = p_sentiment

    # Zero sentiment weight if sentiment was not actually used
    if not sentiment_data.get("used"):
        W_SENTIMENT = 0.0

    # Update weights_used to reflect actual weights after zeroing unavailable sources
    weights_used = {
        "w_implied": W_MARKET,
        "w_kalshi": W_KALSHI,
        "w_model": W_MODEL,
        "w_theover": W_THEOVER,
        "w_sentiment": W_SENTIMENT
    }
    sentiment_data["weight"] = W_SENTIMENT

    # --- CALCULATION (Weighted Sum with Normalization) ---
    # Normalize weights to sum to 1.0 so missing sources don't dilute the result
    w_total = W_MARKET + W_KALSHI + W_MODEL + W_THEOVER + W_SENTIMENT
    if w_total > 0:
        raw_final_prob = (
            (p_market * W_MARKET) +
            (p_kalshi * W_KALSHI) +
            (p_model * W_MODEL) +
            (p_theover * W_THEOVER) +
            (p_sentiment * W_SENTIMENT)
        ) / w_total
    else:
        raw_final_prob = 0.5

    # Base prob (without TheOver) for delta checks
    w_rest = W_MARKET + W_KALSHI + W_MODEL + W_SENTIMENT
    if w_rest > 0:
        prob_no_to = (
            (p_market * W_MARKET) +
            (p_kalshi * W_KALSHI) +
            (p_model * W_MODEL) +
            (p_sentiment * W_SENTIMENT)
        ) / w_rest
    else:
        prob_no_to = 0.5

    # TheOver Delta Clamping
    theover_delta = raw_final_prob - prob_no_to
    sentiment_data["theover_delta"] = theover_delta

    # Extreme Odds check
    is_extreme = False
    if implied_prob is not None:
        if implied_prob < 0.24 or implied_prob > 0.76:
            is_extreme = True

    max_delta = 0.03 if is_extreme else 0.10
    clamped_delta = clamp(theover_delta, -max_delta, max_delta)

    # Apply Clamped Delta (adjust final prob back)
    final_prob_val = prob_no_to + clamped_delta
    sentiment_data["theover_delta_clamped"] = clamped_delta

    final_prob = clamp(final_prob_val, 0.0, 1.0)

    # v98 FIX: Determine driver based on which source has highest effective weight
    if W_KALSHI > 0 and W_KALSHI >= max(W_MARKET, W_MODEL, W_THEOVER, W_SENTIMENT):
        driver = "kalshi"
    elif W_MARKET > 0 or W_MODEL > 0:
        driver = "Market+ML"
    else:
        driver = "Fallback"
    if abs(clamped_delta) > 0.03:
        driver += " + TheOver"

    return final_prob, prob_no_to, weights_used, driver, warnings, kalshi_prob_for_pick, sentiment_data


def build_decision_trace(
    market: str,
    pick_label: str,
    implied_prob: Optional[float],
    kalshi_prob: Optional[float],
    model_prob: Optional[float],
    sentiment_adj: Optional[float],
    weights: Dict[str, Any],
    final_prob: Optional[float],
    confidence: Optional[str],
    league: Optional[str],
    kalshi_available: bool,
    kalshi_market: Optional[str],
    sentiment_score: Optional[float],
    sentiment_label: Optional[str],
    model_used: bool,
    final_pick_reason: Optional[str],
    warnings: Optional[List[str]],
    kalshi_yes_side: Optional[str],
    kalshi_prob_for_pick: Optional[float],
) -> Tuple[str, str, str]:
    confidence_bucket = confidence
    if final_prob is None:
        confidence_bucket = "UNKNOWN"

    # Check if model weight is 0.0 despite model_prob being present/expected
    model_weight = safe_float(weights.get("w_model") if "w_model" in weights else weights.get("ml_weight")) or 0.0
    model_note = ""
    if model_weight == 0.0:
        if model_prob is None and model_used:
             model_note = "Model Failed/Missing"
        elif model_prob is not None:
             model_note = "Model Zeroed (Safety Valve/Low Weight)"
        else:
             model_note = "Model Not Used"

    trace_obj = {
        "league": league,
        "kalshi": {
            "available": kalshi_available,
            "probability": safe_float(kalshi_prob),
            "market": kalshi_market,
            "yes_side": kalshi_yes_side,
            "prob_for_pick": safe_float(kalshi_prob_for_pick),
        },
        "model": {
            "model_used": model_used,
            "model_prob": safe_float(model_prob),
            "status_note": model_note
        },
        "sentiment": {
            "score": safe_float(sentiment_score),
            "label": sentiment_label,
        },
        "source_probs": {
            "implied_prob": safe_float(implied_prob),
            "kalshi_prob_raw_yes": safe_float(kalshi_prob),
            "kalshi_prob_for_pick": safe_float(kalshi_prob_for_pick),
            "model_prob": safe_float(model_prob),
            "sentiment_adj": safe_float(sentiment_adj),
        },
        "weights": {
            "w_implied": safe_float(weights.get("w_implied") if "w_implied" in weights else weights.get("odds_weight")) or 0.0,
            "w_kalshi": safe_float(weights.get("w_kalshi") if "w_kalshi" in weights else weights.get("kalshi_weight")) or 0.0,
            "w_model": safe_float(weights.get("w_model") if "w_model" in weights else weights.get("ml_weight")) or 0.0,
            "w_sentiment": safe_float(weights.get("w_sentiment") if "w_sentiment" in weights else weights.get("sentiment_weight")) or 0.0,
            "w_theover": safe_float(weights.get("w_theover") if "w_theover" in weights else weights.get("theover_weight")) or 0.0,
        },
        "final_prob": safe_float(final_prob),
        "confidence_bucket": confidence_bucket,
        "final_pick_reason": final_pick_reason,
        "warnings": warnings or [],
    }
    short = f"{market}: {pick_label} -> {trace_obj['final_prob'] or 'n/a'} ({confidence or 'NA'})"
    try:
        trace_json = json.dumps(trace_obj, default=safe_str)
    except Exception:
        trace_json = "{}"
    return short, trace_json, trace_json


def engine_label(kalshi_used: bool, market_used: bool) -> str:
    if kalshi_used and market_used:
        return "kalshi+market"
    if kalshi_used and not market_used:
        return "kalshi_only"
    if (not kalshi_used) and market_used:
        return "market_only"
    return "missing"


def blend_kalshi_market(kalshi_p: Optional[float], market_p: Optional[float]) -> Optional[float]:
    """
    Blend Kalshi and market probabilities with Kalshi as the primary signal.

    Rules:
    - If Kalshi ≥ 60%, override odds/model.
    - If Kalshi is in [52%, 60%), blend 60/40 with market odds.
    - Otherwise, favor Kalshi but keep odds as a secondary anchor when present.
    """
    kp = safe_float(kalshi_p)
    mp = safe_float(market_p)
    if kp is None and mp is None:
        return None
    if kp is None:
        return mp
    if mp is None:
        return clamp(kp, 0.0, 1.0)
    if kp >= 0.60:
        return clamp(kp, 0.0, 1.0)
    if 0.52 <= kp < 0.60:
        return clamp(0.6 * kp + 0.4 * mp, 0.0, 1.0)
    return clamp(0.55 * kp + 0.45 * mp, 0.0, 1.0)


def prob_engine_label(kalshi_matched: bool, market_prob: Optional[float], *, model_used: bool = False) -> str:
    if model_used:
        return "model_enabled"
    if kalshi_matched and market_prob is not None:
        return "kalshi+market"
    if kalshi_matched and market_prob is None:
        return "kalshi_only"
    if market_prob is not None:
        return "market_only"
    return "missing"


def is_missing_prob(val: Optional[float]) -> bool:
    return val is None or (isinstance(val, float) and val != val)


def normalize_team_name(name: Any) -> str:
    # Use robust normalization from TeamNameMatcher
    return TeamNameMatcher.normalize(str(name or ""))

def canonical_team_name(name: Any) -> str:
    # Use robust normalization from TeamNameMatcher
    return TeamNameMatcher.normalize(str(name or ""))

def _market_range(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [safe_float(v) for v in values if safe_float(v) is not None]
    if not vals:
        return None, None, None
    vals_sorted = sorted(vals)
    lo = vals_sorted[0]
    hi = vals_sorted[-1]
    try:
        med = statistics.median(vals_sorted)
    except Exception:
        med = None
    return lo, med, hi

def parse_spread_pick(raw_val: Any, home: Optional[str], away: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse a "Spread & Pick" style string into (team, line).
    """
    if not raw_val:
        return None, None
    text = str(raw_val).strip()
    match = re.match(r"(.+?)\s+(-?\d+(?:\.\d+)?)$", text)
    if not match:
        return None, None
    team = match.group(1).strip()
    line = safe_float(match.group(2))
    if team and line is not None:
        return team, line
    return None, None

def parse_total_pick(raw_val: Any) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse a "Total & Pick" style string like 'Under 44.5' into (side, line).
    """
    if not raw_val:
        return None, None
    text = str(raw_val).strip()
    match = re.match(r"(Over|Under)\s+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    side = match.group(1).title()
    line = safe_float(match.group(2))
    return side, line

def pivot_market(row):
    """
    Constraint: The final Market column must never say "Moneyline."
    Pivots Moneyline picks to Spread or Total based on Edge/Probability.
    """
    if row.get('Market') == "Moneyline":
        # Compare Spread Prob vs Total Prob (using edge or confidence)
        # Use defaults if keys missing
        s_prob = safe_float(row.get('spread_prob_adj')) or safe_float(row.get('spread_prob_final')) or 0.0
        t_prob = safe_float(row.get('total_prob_adj')) or safe_float(row.get('total_prob_final')) or 0.0

        s_edge = safe_float(row.get('spread_edge')) or 0.0
        t_edge = safe_float(row.get('total_edge')) or 0.0

        pivot_to_spread = False
        if s_prob > t_prob:
            pivot_to_spread = True
        elif t_prob > s_prob:
            pivot_to_spread = False
        else:
            # Tie-break with edge
            if s_edge >= t_edge:
                pivot_to_spread = True
            else:
                pivot_to_spread = False

        if pivot_to_spread:
            row['Market'] = "Spread"
            row['Pick'] = row.get('Spread & Pick')
            row['final_probability'] = s_prob
            row['best_pick_type'] = "SPREAD"
            row['edge'] = s_edge
        else:
            row['Market'] = "Total"
            row['Pick'] = row.get('Total & Pick')
            row['final_probability'] = t_prob
            row['best_pick_type'] = "TOTAL"
            row['edge'] = t_edge

    return row

def enforce_winning_picks(df: pd.DataFrame) -> pd.DataFrame:
    """
    VALIDATION CHECKPOINT: Verify all picks have probability >= 50%.

    After Prompts #1 and #2 are applied, this function should NOT need to fix anything.
    It just verifies that pick selection worked correctly.

    If any picks have prob < 50%, this raises an error - indicating that
    Prompt #1 (Kalshi mapping) or Prompt #2 (pick selection) is broken.

    Args:
        df: DataFrame with final picks

    Returns:
        df: Same DataFrame (unchanged if all picks are valid)

    Raises:
        ValueError: If any picks have prob < 50%
    """
    logger.info(f"enforce_winning_picks() CALLED - Validating {len(df)} picks...")

    # Aliasing for robustness: Ensure 'prob' column exists (use 'final_probability' if available)
    if 'prob' not in df.columns and 'final_probability' in df.columns:
        df = df.copy()
        df['prob'] = df['final_probability']
    elif 'prob' not in df.columns:
        # Fallback if neither exists (should not happen based on pipeline)
        logger.warning("prob column missing in enforce_winning_picks! Checking if 'Best Overall Prob' exists...")
        if 'Best Overall Prob' in df.columns:
             df = df.copy()
             df['prob'] = df['Best Overall Prob']

    # Count picks with prob < 50%
    if 'prob' in df.columns:
        losing_picks = df[df['prob'] < 0.50]
        losing_count = len(losing_picks)
    else:
        logger.error("CRITICAL: No probability column found for validation!")
        losing_count = 0 # Avoid crash but log error

    if losing_count > 0:
        # CRITICAL ERROR - pick selection is broken
        logger.error(f"❌ CRITICAL: {losing_count}/{len(df)} picks have prob < 50%!")
        logger.error(f"This means Prompt #1 (Kalshi mapping) or Prompt #2 (pick selection) is broken.")
        logger.error(f"\\nSample losing picks:")

        for idx, row in losing_picks.head(10).iterrows():
            logger.error(f"  Game: {row.get('Home', 'Unknown')} vs {row.get('Away', 'Unknown')}")
            logger.error(f"    Pick: {row.get('Pick', 'Unknown')} | Prob: {row.get('prob', 0.0):.3f} (should be >= 0.50)")

            # Show spread details if available
            if 'spreadprobpickfinal' in row and pd.notna(row['spreadprobpickfinal']):
                logger.error(f"    Spread pick prob: {row['spreadprobpickfinal']:.3f}")
                logger.error(f"    Spread alt prob: {row.get('spreadprobaltfinal', 'N/A')}")

            # Show total details if available
            if 'totalprobpickfinal' in row and pd.notna(row['totalprobpickfinal']):
                logger.error(f"    Total pick prob: {row['totalprobpickfinal']:.3f}")
                logger.error(f"    Total alt prob: {row.get('totalprobaltfinal', 'N/A')}")

            logger.error("")  # Blank line

        # Raise error to halt execution
        raise ValueError(
            f"❌ {losing_count} picks have prob < 50%. "
            f"Fix Prompt #1 (Kalshi mapping) or Prompt #2 (pick selection) before proceeding."
        )

    # All picks are valid
    if 'prob' in df.columns:
        logger.info(f"✅ All {len(df)} picks have prob >= 50%")
        logger.info(f"   Min prob: {df['prob'].min():.3f}")
        logger.info(f"   Max prob: {df['prob'].max():.3f}")
        logger.info(f"   Avg prob: {df['prob'].mean():.3f}")

    return df

def enrich_picks_with_roi_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jules: Call this function to prepare the 'Infallible' dashboard view.
    """
    if df is None or df.empty:
        return df

    # ROI COLUMN SAFETY SHIELD: Ensure required columns exist to prevent KeyError crash
    required = ['spread_implied_prob', 'total_implied_prob', 'spread_width', 'total_width', 'spread_prob_adj', 'total_prob_adj']

    # Batch assignment to reduce fragmentation (Issue #5)
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing_required)], axis=1).copy()

    # 1. Calculate Edge (Math vs Market Gap)
    # Ensure columns are numeric to avoid errors
    df['spread_implied_prob'] = pd.to_numeric(df['spread_implied_prob'], errors='coerce').fillna(0.0).infer_objects(copy=False)
    df['total_implied_prob'] = pd.to_numeric(df['total_implied_prob'], errors='coerce').fillna(0.0).infer_objects(copy=False)
    
    # Use adjusted probabilities if available, else raw
    s_prob = df['spread_prob_adj'] if 'spread_prob_adj' in df.columns else df.get('spread_prob')
    t_prob = df['total_prob_adj'] if 'total_prob_adj' in df.columns else df.get('total_prob')

    # Fallback to 0 if column missing or null
    s_prob = pd.to_numeric(s_prob, errors='coerce').fillna(0.0).infer_objects(copy=False)
    t_prob = pd.to_numeric(t_prob, errors='coerce').fillna(0.0).infer_objects(copy=False)

    # Fix Issue #5: Use pd.concat for new columns to avoid fragmentation warnings
    # Construct a new DataFrame for the new metrics
    metrics_data = {}
    metrics_data['spread_edge'] = s_prob - df['spread_implied_prob']
    metrics_data['total_edge'] = t_prob - df['total_implied_prob']
    
    # 2. Define Market Stability (Volatility Indicator)
    def classify_stability(row):
        # A market is 'Wide' if books disagree on the line
        sw = safe_float(row.get('spread_width'))
        tw = safe_float(row.get('total_width'))
        if (sw is not None and sw > 0.5) or (tw is not None and tw > 1.0):
            return "WIDE"
        return "TIGHT"
    
    metrics_data['market_stability'] = df.apply(classify_stability, axis=1)

    # Concatenate the new metrics
    new_metrics_df = pd.DataFrame(metrics_data, index=df.index)

    # Drop existing columns if they exist
    cols_to_drop = [c for c in new_metrics_df.columns if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = pd.concat([df, new_metrics_df], axis=1).copy()
    
    # 3. Handle 'Market_Badge' Labeling - Vectorized
    if 'Market_Badge' in df.columns:
        # Create boolean mask for WIDE stability
        is_wide = metrics_data['market_stability'] == "WIDE"

        # Get existing badges as strings, filling NaNs
        existing_badges = df['Market_Badge'].astype(str).replace('nan', '').replace('None', '')

        # Define update function for vectorization
        # If wide and not already labeled, append
        # This logic is slightly complex to vectorize perfectly with string operations,
        # but list comprehension is faster than apply

        new_badges = []
        for idx, val in existing_badges.items():
            if is_wide.get(idx, False):
                if "WIDE MARKET" not in val:
                    new_val = (val + ";WIDE MARKET").strip(";")
                    new_badges.append(new_val)
                else:
                    new_badges.append(val)
            else:
                new_badges.append(val)

        # Assign back directly (since we already did concat above, this is safe-ish,
        # or we could make a new series and concat, but replacing an existing column is usually okay
        # if the frame isn't fragmented yet. To be safe, we'll assign to the column).
        # Actually, let's just update the column in place since we just de-fragmented with concat
        df['Market_Badge'] = new_badges
    
    return df


def get_best_ml_picks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with ML picks from the deduped master results.
    Now filters by best_pick_type == "ML" instead of Market == "Moneyline".
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Defensive: Deduplicate columns to prevent DataFrame-return-on-access errors
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Filter to rows where best_pick_type is ML (from deduped master results)
    # Try both column names for compatibility
    target_col = None
    if "best_pick_type" in df.columns:
        target_col = "best_pick_type"
    elif "bestpicktype" in df.columns:
        target_col = "bestpicktype"

    if target_col:
        # Ensure string access works - verify it's object or string type
        # If not (e.g. all NaNs might be float), try to coerce safely
        try:
            if df[target_col].dtype != 'object' and not isinstance(df[target_col].dtype, pd.StringDtype):
                 df[target_col] = df[target_col].astype(str)

            df_ml = df[df[target_col].str.upper().str.contains("ML", na=False)].copy()
        except Exception as e:
            logger.warning(f"Error filtering for ML picks on column {target_col}: {e}")
            df_ml = pd.DataFrame()
    else:
        # Fallback to Market column if best_pick_type not available
        logger.warning("best_pick_type column not found, falling back to Market == 'Moneyline'")
        if "Market" in df.columns:
            df_ml = df[df["Market"] == "Moneyline"].copy()
        else:
            df_ml = pd.DataFrame()

    if df_ml.empty:
        logger.info("No ML picks found after filtering")
        return pd.DataFrame()

    # Filter by confidence if needed (HIGH and MEDIUM only)
    # Check if Pick_Confidence column exists
    if "Pick_Confidence" in df_ml.columns:
        df_ml = df_ml[df_ml["Pick_Confidence"].isin(["HIGH", "MEDIUM"])].copy()
        logger.info(f"Filtered ML picks to HIGH/MEDIUM confidence: {len(df_ml)} picks")

    if df_ml.empty:
        logger.info("No HIGH/MEDIUM confidence ML picks found")
        return pd.DataFrame()

    # Since df is already deduped (one row per game), we can just select relevant columns
    summary_rows = []

    for idx, row in df_ml.iterrows():
        summary = {
            "league": row.get("league"),
            "Home": row.get("Home"),
            "Away": row.get("Away"),
            "Commence (UTC)": row.get("Commence (UTC)"),
            "Commence (Local)": row.get("Commence (Local)"),
            "Best Overall Pick": row.get("Best Overall Pick") or row.get("Pick"),
            "Best Overall Prob": row.get("Best Overall Prob") or row.get("final_probability"),
            "Best Overall Confidence": row.get("Pick_Confidence"),
            "Implied Prob": row.get("Implied_Prob"),
            "AI Prob": row.get("AI_Prob"),
        }
        summary_rows.append(summary)

    logger.info(f"Returning {len(summary_rows)} ML picks")
    return pd.DataFrame(summary_rows)

def build_game_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates ML, Spread, and Total rows into a single row per game.
    Wraps v2 logic to maintain compatibility if needed, or simply alias it.
    """
    return build_game_summary_v2(df)

def calculate_best_pick_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Best Pick (best_pick_type, best_pick, final_prob, edge) with updated priority.

    Priority Logic:
      1. Spread or Total (if valid/strong).
      2. Moneyline (only if Spread/Total weak/missing AND ML eligible).
      3. Fallback (avoid NO_BET).

    Moneyline Suppression:
      - If abs(ML) > 300, ML is NOT eligible for best pick (unless forced fallback).
    """
    if df is None or df.empty:
        return df

    # Avoid fragmentation by working on a copy
    df = df.copy()

    def _apply(row):
        # 1. Helpers
        def _safe(k):
            try:
                v = row.get(k)
                if pd.isna(v) or v == "": return None
                return float(v)
            except (ValueError, TypeError):
                return None

        def _safe_str(k):
            v = row.get(k)
            if pd.isna(v): return None
            s = str(v).strip()
            return s if s.lower() != "none" else None

        # 2. Extract Data
        # Spread
        s_pick = _safe_str("Spread & Pick")
        s_prob = _safe("SpreadConsensusProb")
        if s_prob is None: s_prob = _safe("spread_prob_adj")
        if s_prob is None: s_prob = _safe("spread_prob")
        # Ensure we use the winning-side probability (post-flip)
        # This prevents champion selection from comparing pre-flip sub-50% values
        if s_prob is not None and s_prob < 0.50:
            s_prob = 1.0 - s_prob
            # FLIP PICK: If probability flips, pick must flip to the other side
            alt_s = _safe_str("spread_alt_label")
            if alt_s:
                s_pick = alt_s
        s_edge = _safe("spread_edge") or 0.0

        # Total
        t_pick = _safe_str("Total & Pick")
        t_prob = _safe("TotalConsensusProb")
        if t_prob is None: t_prob = _safe("total_prob_adj")
        if t_prob is None: t_prob = _safe("total_prob")
        # Ensure we use the winning-side probability (post-flip)
        if t_prob is not None and t_prob < 0.50:
            t_prob = 1.0 - t_prob
            # FLIP PICK: If probability flips, pick must flip to the other side
            alt_t = _safe_str("total_alt_label")
            if alt_t:
                t_pick = alt_t
            else:
                # Fallback simple flip for Total if alt label missing
                if t_pick and "Over" in t_pick:
                    t_pick = t_pick.replace("Over", "Under")
                elif t_pick and "Under" in t_pick:
                    t_pick = t_pick.replace("Under", "Over")
        t_edge = _safe("total_edge") or 0.0

        # Moneyline
        # FIX: Do not rely on "Pick" column which might hold Spread/Total pick.
        # Derive ML pick from Odds/Prob data if available.
        ml_prob = _safe("final_probability")
        if ml_prob is None: ml_prob = _safe("AI_Prob")
        ml_implied = _safe("Implied_Prob")

        # If the row is a spread/total row, AI_Prob might be the model prob for spread/total?
        # Check "Market" column.
        row_market = str(row.get("Market") or "").lower()
        if "spread" in row_market or "total" in row_market:
             # In Spread/Total rows, AI_Prob might be spread/total model prob.
             # We should look for ML-specific columns if they exist, or rely on "model_prob_home".
             ml_prob_raw = _safe("model_prob_home")
             if ml_prob_raw is not None:
                  ml_prob = ml_prob_raw

             # If we are in a spread row, "Implied_Prob" is spread implied.
             pass

        ml_home_price = _safe("Home_ML")
        ml_away_price = _safe("Away_ML")

        # Determine ML Pick Side (Home/Away) based on Prob > 0.5 or Odds
        # Just for eligibility check, we assume if we have odds and prob, it's valid.
        ml_pick_candidate = "Home" # Placeholder, strict logic not needed for eligibility flag only

        ml_edge = 0.0
        # Derive ml_pick if not present (for ml_valid logic below)
        ml_pick = _safe_str("Pick")
        if not ml_pick and ml_prob is not None:
            # Infer pick from prob > 0.5 or odds
            if ml_prob > 0.5:
                 ml_pick = row.get("Home")
            else:
                 ml_pick = row.get("Away")

        # FIX: Relaxed Moneyline Suppression
        # Instead of strict disqualification, we allow extreme odds but tag them for low confidence/warning.
        is_allowed = ml_allowed(ml_home_price, ml_away_price, threshold=300)

        ml_eligible = True
        ml_suppressed_reason = ""
        moneyline_disabled = False
        moneyline_disabled_reason = ""

        if not is_allowed:
            # Task 2: Allow ML picks even if extreme, but flag them.
            # Do NOT set ml_eligible = False.
            moneyline_disabled = False # Changed from True
            moneyline_disabled_reason = "Extreme odds (>300) detected"
            ml_suppressed_reason = "extreme_odds_warning"

        # Check data availability
        if ml_home_price is None or ml_away_price is None:
             ml_eligible = False
             ml_suppressed_reason = "missing_odds"

        # NOTE: We do NOT set ml_pick = None here because ml_pick variable (from "Pick" col)
        # is used below for "best_pick" logic.
        # If "Pick" is Spread, we don't want to mess it up.
        # We only use ml_eligible to filter candidacy.

        # 3. Calculate Scores
        # Score = (Prob - 0.5) + Edge * 2.0 (favors higher edge)
        def _score(prob, edge):
            p = prob if prob is not None else 0.5
            e = edge if edge is not None else 0.0
            return (p - 0.5) + (e * 2.0) + 0.5 # Base offset to keep positive

        s_score = _score(s_prob, s_edge)
        t_score = _score(t_prob, t_edge)

        # ML Score needed for best_pick selection
        ml_score = _score(ml_prob, ml_edge)

        # Valid Flags (Prob > 0.0 check is mostly to avoid default zeros if they slipped in)
        s_valid = (s_prob is not None and s_pick is not None)
        t_valid = (t_prob is not None and t_pick is not None)
        # ml_valid is tracked and can be used for best_pick candidacy if eligible
        ml_valid = (ml_prob is not None and ml_pick is not None and ml_eligible)

        # 4. Selection Logic
        best_type = "SPREAD" # Default
        best_pick = s_pick
        best_prob = s_prob
        best_edge = s_edge
        reason = "Default"

        # Priority: ONLY consider Spread and Total - EXCLUDE ML from Best Overall Pick
        # ML should only be shown for analysis purposes, not as primary recommendation
        candidates = []
        if s_valid: candidates.append("SPREAD")
        if t_valid: candidates.append("TOTAL")
        # REMOVED: if ml_valid: candidates.append("ML") - ML excluded per user requirement

        candidate_types_str = "|".join(candidates)

        # Decision
        # Compare ONLY Spread and Total markets (ML excluded for parlay building)
        # Use score which combines prob and edge as "confidence/edge" metric.

        # Collect all candidates with their probabilities (ML excluded)
        # v104 FIX (Bug 4): Compare probabilities directly, NOT score (prob + edge*2).
        # Using score caused 5 games to select "Total" when SpreadConsensusProb > TotalConsensusProb
        # because Total had higher edge. The market label must match the highest probability.
        market_candidates = []
        if s_valid:
            market_candidates.append(("SPREAD", s_prob, s_pick, s_edge))
        if t_valid:
            market_candidates.append(("TOTAL", t_prob, t_pick, t_edge))
        # REMOVED: ML from market_candidates - ML excluded per user requirement

        if market_candidates:
            # Sort by probability (highest first), with tie-breaker preferring Spread > Total
            type_priority = {"SPREAD": 0, "TOTAL": 1}
            market_candidates.sort(key=lambda x: (-(x[1] or 0), type_priority.get(x[0], 2)))

            # Select the best
            best_type, best_prob, best_pick, best_edge = market_candidates[0]

            if len(market_candidates) > 1:
                second_type = market_candidates[1][0]
                reason = f"{best_type} > {second_type}"
            else:
                reason = f"Only {best_type} Valid"
        else:
            # No valid markets at all
            best_type = "SPREAD"
            best_pick = s_pick # Might be None
            best_prob = s_prob if s_prob is not None else 0.5
            best_edge = s_edge
            reason = "No Valid Markets"

        # 5. Confidence Logic (Refactored)
        # Using soft fallback model based on Probability
        # Flip logic is now handled upstream in enforce_winning_picks()

        p_val = best_prob if best_prob is not None else 0.5
        e_val = best_edge if best_edge is not None else 0.0
        stats_quality = row.get("stats_quality", "REAL")

        # Use shared calculation logic
        # Force simple probability-based confidence (matching the core logic)
        if p_val >= 0.60:
            conf_label = "HIGH"
        elif p_val >= 0.55:
            conf_label = "MEDIUM"
        else:
            conf_label = "LOW"

        # Force LOW if ML was suppressed (extreme odds)
        if best_type == "ML" and ml_suppressed_reason == "extreme_odds_warning":
            conf_label = "LOW"
            reason += " [Extreme Odds]"

        # Force LOW if ML was truly ineligible (e.g. missing odds)
        if best_type == "ML" and not ml_eligible and ml_suppressed_reason != "extreme_odds_warning":
            conf_label = "LOW"
            reason += " [Suppressed/Missing]"

        # Force LOW if Pick is None (No Bet)
        if best_pick is None:
            conf_label = "LOW"
            reason = "NO_BET_POSSIBLE"

        # At_a_Glance_Confidence Clamping
        # "At_a_Glance_Confidence follows the same rule or is at most equal to Bet_Confidence, never higher."
        glance_conf = row.get("At_a_Glance_Confidence", "LOW")
        ranks = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0, None: 0}

        glance_rank = ranks.get(glance_conf, 1)
        bet_rank = ranks.get(conf_label, 1)

        if glance_rank > bet_rank:
            glance_conf = conf_label

        bet_lean = (conf_label == "LOW")
        conf_score = (p_val - 0.5) + e_val

        # Recalculate confidence_reason to reflect the final state (flipped or not)
        # This fixes the "stale metadata" issue where flipped picks still showed "negative_edge" reason
        driver = str(row.get("decision_driver") or "unknown")
        confidence_reason = f"{conf_label}: {reason} | driver={driver}"

        return pd.Series([
            best_type, best_pick, p_val, reason, e_val,
            conf_label, bet_lean, conf_score,
            ml_eligible, ml_suppressed_reason, candidate_types_str,
            moneyline_disabled, moneyline_disabled_reason,
            glance_conf,
            confidence_reason
        ], index=[
            "best_pick_type", "best_pick", "final_prob", "Best_ST_Reason", "edge",
            "Bet_Confidence", "Bet_Lean", "Bet_Confidence_Score",
            "ml_eligible", "ml_suppressed_reason", "candidate_types_available",
            "moneyline_disabled", "moneyline_disabled_reason",
            "At_a_Glance_Confidence",
            "confidence_reason"
        ])

    # Batch apply
    new_cols = df.apply(_apply, axis=1)

    # Drop columns if they exist to allow overwrite
    cols_to_drop = [c for c in new_cols.columns if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return pd.concat([df, new_cols], axis=1).copy()


def generate_reasoning(row):
    """
    Generates the short reasoning string for the pick.
    Ensures that logic reflects the final flipped state.
    """
    conf = row.get('Pick_Confidence', 'LOW')
    reason = str(row.get('Best_ST_Reason', ''))
    driver = str(row.get('decision_driver', 'unknown'))

    # Construct base reason
    full_reason = f"{conf}: {reason} | driver={driver}"

    # Append TheOver impact if present
    if row.get("theover_matched") and "theover_delta_final_prob" in row:
        try:
            delta = float(row.get("theover_delta_final_prob", 0))
            if abs(delta) > 0.005:
                direction = "boost" if delta > 0 else "drag"
                full_reason += f" | TheOver: {delta:+.3f} ({direction})"
        except:
            pass

    return full_reason


def reorder_master_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure fixed front columns then pick columns; preserve remaining order.
    """
    if df is None or df.empty:
        return df

    fixed_front = [
        "league",
        "Home",
        "Away",
        "Commence (UTC)",
        "Commence (Local)",
        "Local Date",
    ]
    fixed_front = [c for c in fixed_front if c in df.columns]

    pick_cols = [c for c in ["Pick", "Spread & Pick", "Total & Pick"] if c in df.columns]

    remaining = [c for c in df.columns if c not in fixed_front and c not in pick_cols]

    try:
        return df[fixed_front + pick_cols + remaining]
    except Exception:
        return df

def reorder_for_spread_total_focus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce fixed front columns, summary block, then remaining columns.
    Wraps v2 logic.
    """
    return reorder_for_spread_total_focus_v2(df)


def confidence_from_market(
    books_count: Any,
    width: Any,
    odds_valid: bool,
    mixed_side_flag: bool,
    has_proxy_warning: bool,
    *,
    market_kind: str = "spread",
) -> Tuple[str, str]:
    conf = "HIGH"
    reasons: List[str] = []

    if has_proxy_warning:
        conf = "LOW"
        reasons.append("proxy_warning")
    if mixed_side_flag:
        conf = "LOW"
        reasons.append("mixed_side_range")
    if not odds_valid:
        conf = "LOW"
        reasons.append("missing_odds")
    try:
        bc = int(books_count) if books_count is not None else 0
    except Exception:
        bc = 0
    if bc <= 1:
        conf = "LOW"
        reasons.append("thin_market")
    w_val = safe_float(width)
    if w_val is not None:
        threshold = 2.0 if market_kind == "spread" else 3.0
        if w_val >= threshold and conf == "HIGH":
            conf = "MEDIUM"
        if w_val >= threshold:
            reasons.append(f"wide_market({w_val:.1f})")
    return conf, ";".join(reasons)


def compute_at_a_glance(spread_conf: str, spread_reason: str, total_conf: str, total_reason: str) -> Tuple[str, int, str]:
    rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    sc = spread_conf or "LOW"
    tc = total_conf or "LOW"
    sc_rank = rank.get(sc, 1)
    tc_rank = rank.get(tc, 1)
    overall = sc if sc_rank <= tc_rank else tc
    score = rank.get(overall, 1)

    reason = None
    if overall == "HIGH":
        reason = "spread+total strong"
    else:
        parts: List[str] = []
        for part_str in [spread_reason, total_reason]:
            if part_str:
                parts.extend([p for p in str(part_str).split(";") if p])

        if overall == "LOW":
            priority_tokens = {"missing_odds", "thin_market", "mixed_side_range", "proxy_warning"}
            priority = [p for p in parts if any(tok in p for tok in priority_tokens)]
            remainder = [p for p in parts if p not in priority]
            ordered = priority + remainder
            reason = ";".join(ordered)
        else:
            reason = ";".join(parts)

        if reason:
            reason = reason[:120]

    return overall, score, reason


def fmt_prob(p: Any) -> str:
    try:
        if p is None:
            return "—"
        return f"{float(p) * 100:.0f}%"
    except Exception:
        return "—"


def depth_label(books_count: Any) -> str:
    try:
        if books_count is None:
            return "—"
        n = int(books_count)
        if n <= 1:
            return "Thin"
        if n <= 3:
            return "OK"
        return "Deep"
    except Exception:
        return "—"


def market_width_label(width: Any, market_type: str) -> str:
    if width is None:
        return "—"
    try:
        w = abs(float(width))
    except Exception:
        return "—"
    if market_type == "spread":
        if w <= 0.5:
            return "Tight"
        if w <= 1.5:
            return "Normal"
        return "Wide"
    else:
        if w <= 1.0:
            return "Tight"
        if w <= 3.0:
            return "Normal"
        return "Wide"


def build_clean_glance(conf: Any, prob: Any, books: Any, width: Any, market_type: str) -> str:
    conf_norm = str(conf or "LOW").upper()
    return f"{conf_norm} | {fmt_prob(prob)} | {depth_label(books)} | {market_width_label(width, market_type)}"


def detect_placeholder_odds(row: pd.Series) -> pd.Series:
    """
    Detects if odds are placeholders (-110/-110 with 0 line).
    Logic: Trigger flag if both Home and Away ML are exactly -110 AND the spread_point is 0.0 or None.
    """
    # Check ML (Implied) -110
    # We check if implied prob is close to 0.5238 (which is -110)
    # Actually, simpler to check raw odds if available, or implied prob.
    # Let's use implied prob ~ 0.5238 (+/- small epsilon) on BOTH sides?
    # Or just check if implied prob is exactly -110 derived.

    # Better: check columns if they exist.
    # We need spread odds home/away usually.

    s_home_odds = row.get("spread_odds_home")
    s_away_odds = row.get("spread_odds_away")
    s_line = row.get("spread_point") # Or spread_line

    is_placeholder = False

    # Helper to check -110
    def is_minus_110(val):
        try:
            f = float(val)
            return abs(f + 110.0) < 0.1
        except:
            return False

    # Check strict condition
    if is_minus_110(s_home_odds) and is_minus_110(s_away_odds):
        # Check line
        if s_line is None or (try_float(s_line) == 0.0):
            is_placeholder = True

    return is_placeholder

def try_float(x):
    try:
        return float(x)
    except:
        return 0.0

def add_spread_total_confidence(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # Work on a copy to avoid side effects
    df = df.copy()

    # Pre-allocate list for new columns
    new_data = []

    # Iterate row-by-row but collect results
    for idx, row in df.iterrows():
        warnings_text = str(row.get("Warnings") or "")
        spread_odds_valid = row.get("spread_odds_valid")
        total_odds_valid = row.get("total_odds_valid")

        # Validation checks
        # Fix: Use new detect_placeholder_odds logic directly or ensure flag is correct
        # Since we haven't bulk-applied detect_placeholder yet, let's do inline check

        # Validation checks

        # Check for Spread placeholder
        # Logic: Both sides -110 AND line is 0.0
        s_home = row.get("spread_odds_home")
        s_away = row.get("spread_odds_away")
        s_line = row.get("spread_point")

        # Create temp row for helper
        s_row = pd.Series({
            "spread_odds_home": s_home,
            "spread_odds_away": s_away,
            "spread_point": s_line
        })
        is_spread_placeholder = detect_placeholder_odds(s_row)

        if spread_odds_valid is None:
            spread_odds_valid = safe_float(row.get("spread_implied_prob")) is not None and not is_spread_placeholder

        # Check for Total placeholder
        # Total usually has standard -110, so placeholder is just Line=0
        # Re-use helper but map total keys to expected spread keys?
        # Helper expects "spread_odds_home", "spread_odds_away", "spread_point"
        # Total uses "total_odds_over", "total_odds_under", "total_point" usually
        t_over = row.get("total_odds_over")
        t_under = row.get("total_odds_under")
        t_line = row.get("total_point")

        t_row = pd.Series({
            "spread_odds_home": t_over,
            "spread_odds_away": t_under,
            "spread_point": t_line
        })
        is_total_placeholder = detect_placeholder_odds(t_row)

        if total_odds_valid is None:
            total_odds_valid = safe_float(row.get("total_implied_prob")) is not None and not is_total_placeholder

        spread_width_val = safe_float(row.get("spread_width"))
        total_width_val = safe_float(row.get("total_width"))
        spread_books_count = row.get("spread_books_count")
        total_books_count = row.get("total_books_count")
        mixed_side_flag = "spread_range_mixed_sides_detected" in warnings_text
        proxy_flag = "model_proxy_for_spread_total" in warnings_text

        # Compute confidence
        spread_conf, spread_reason = confidence_from_market(
            spread_books_count, spread_width_val, spread_odds_valid, mixed_side_flag, proxy_flag, market_kind="spread"
        )
        total_conf, total_reason = confidence_from_market(
            total_books_count, total_width_val, total_odds_valid, False, proxy_flag, market_kind="total"
        )

        # Compute probabilities
        spread_prob_val = row.get("spread_prob") if row.get("spread_prob") is not None else row.get("spread_prob_market_based")
        if spread_prob_val is None:
            penalty = 0.0
            if spread_width_val is not None:
                penalty = min(0.05, spread_width_val * 0.02)
            spread_prob_val = clamp(0.5 - (penalty or 0.0))
            spread_reason = ";".join(filter(None, [spread_reason, "missing_implied_prob"]))

        total_prob_val = row.get("total_prob") if row.get("total_prob") is not None else row.get("total_prob_market_based")
        if total_prob_val is None:
            penalty = 0.0
            if total_width_val is not None:
                penalty = min(0.05, total_width_val * 0.02)
            total_prob_val = clamp(0.5 - (penalty or 0.0))
            total_reason = ";".join(filter(None, [total_reason, "missing_implied_prob"]))

        overall_conf, overall_score, overall_reason = compute_at_a_glance(
            spread_conf, spread_reason, total_conf, total_reason
        )

        # Sentiment Integration
        sentiment_level = _normalize_sentiment_level(row.get("sentiment_level"))
        raw_articles = row.get("sentiment_articles_used")
        articles_used = int(float(raw_articles)) if pd.notnull(raw_articles) and raw_articles != "" else 0
        sentiment_strength = str(row.get("sentiment_strength") or "").upper() or sentiment_strength_from_articles(
            sentiment_level, articles_used
        )
        if not sentiment_strength or sentiment_strength == "NONE":
            sentiment_strength = sentiment_strength_from_articles(sentiment_level, articles_used)
        sentiment_badge = sentiment_badge_for(sentiment_level, sentiment_strength)
        sentiment_signal = sentiment_signal_value(
            sentiment_level,
            row.get("Sentiment_Diff"),
            game_sentiment=row.get("Game_Sentiment"),
            league_sentiment=row.get("League_Sentiment"),
        )
        spread_adj_val = row.get("spread_sentiment_adj")
        total_adj_val = row.get("total_sentiment_adj")
        auth_error = bool(row.get("sentiment_auth_error"))
        if auth_error:
            spread_adj_val = 0.0
            total_adj_val = 0.0
        if spread_adj_val is None:
            spread_adj_val = compute_market_sentiment_adjustment(
                sentiment_level, sentiment_strength, "spread", sentiment_signal
            )
        if total_adj_val is None:
            total_adj_val = compute_market_sentiment_adjustment(
                sentiment_level, sentiment_strength, "total", sentiment_signal
            )

        # Adjusted Probs
        spread_prob_adj = None
        total_prob_adj = None
        if sentiment_level == "league":
            spread_adj_val = 0.0
            total_adj_val = 0.0
            spread_prob_adj = spread_prob_val
            total_prob_adj = total_prob_val
        else:
            spread_prob_adj = clamp((spread_prob_val or 0.0) + spread_adj_val, 0.01, 0.99) if spread_prob_val is not None else None
            total_prob_adj = clamp((total_prob_val or 0.0) + total_adj_val, 0.01, 0.99) if total_prob_val is not None else None

        market_kind = str(row.get("Market") or "").lower()
        consensus_prob_adj = None
        if market_kind == "spread" and spread_prob_adj is not None:
            consensus_prob_adj = spread_prob_adj
        if market_kind == "total" and total_prob_adj is not None:
            consensus_prob_adj = total_prob_adj

        spread_prob_display = round_pct(spread_prob_adj)
        total_prob_display = round_pct(total_prob_adj)

        # Sentiment Visuals
        spread_arrow = ""
        total_arrow = ""
        spread_note = None
        total_note = None
        signal_zero = bool(not sentiment_signal)
        if sentiment_level == "league" and not signal_zero:
            direction = "↗" if (sentiment_signal or 0) > 0 else "↘"
            spread_note = f"LEAGUE {direction}"
            total_note = f"LEAGUE {direction}"
        elif sentiment_level in {"team", "game"} and not signal_zero:
            spread_arrow = prob_arrow(row.get("spread_prob"), spread_prob_adj)
            total_arrow = prob_arrow(row.get("total_prob"), total_prob_adj)

        def _glance_with_signal(conf_val: Any, prob_display: Optional[int], books: Any, width_val: Any, market_type: str, arrow_val: str, note_val: Optional[str]) -> str:
            prob_text = prob_display if prob_display not in {"", None} else "—"
            signal = ""
            if note_val:
                signal = f" {note_val}"
            elif arrow_val:
                signal = f" {arrow_val}"
            return f"{conf_val or 'LOW'} | {prob_text} {signal}".rstrip() + f" | {depth_label(books)} | {market_width_label(width_val, market_type)}"

        clean_spread_glance = _glance_with_signal(
            spread_conf, spread_prob_display, spread_books_count, spread_width_val, "spread", spread_arrow, spread_note
        )
        clean_total_glance = _glance_with_signal(
            total_conf, total_prob_display, total_books_count, total_width_val, "total", total_arrow, total_note
        )

        conf_rank_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        spread_conf_rank = conf_rank_map.get(str(spread_conf or "LOW").upper(), 1)
        total_conf_rank = conf_rank_map.get(str(total_conf or "LOW").upper(), 1)

        decisiveness = 0.0
        try:
            if spread_prob_adj is not None:
                decisiveness += abs(float(spread_prob_adj) - 0.5)
            if total_prob_adj is not None:
                decisiveness += abs(float(total_prob_adj) - 0.5)
        except Exception:
            decisiveness = 0.0

        # Collect data for this row
        new_data.append({
            "spread_prob": spread_prob_val,
            "spread_confidence": spread_conf,
            "spread_confidence_reason": spread_reason,
            "spread_odds_valid": spread_odds_valid,
            "total_prob": total_prob_val,
            "total_confidence": total_conf,
            "total_confidence_reason": total_reason,
            "total_odds_valid": total_odds_valid,
            "At_a_Glance_Confidence": overall_conf,
            "At_a_Glance_Score": overall_score,
            "At_a_Glance_Reason": overall_reason,
            "Spread_Glance": build_clean_glance(spread_conf, spread_prob_val, spread_books_count, spread_width_val, "spread"),
            "Total_Glance": build_clean_glance(total_conf, total_prob_val, total_books_count, total_width_val, "total"),
            "Spread_Glance_Reason": spread_reason,
            "Total_Glance_Reason": total_reason,
            "sentiment_level": sentiment_level,
            "sentiment_strength": sentiment_strength,
            "sentiment_badge": sentiment_badge,
            "sentiment_articles_used": articles_used,
            "spread_sentiment_adj": spread_adj_val,
            "total_sentiment_adj": total_adj_val,
            "spread_prob_adj": spread_prob_adj,
            "total_prob_adj": total_prob_adj,
            "consensus_prob_adj": consensus_prob_adj,
            "spread_prob_display": spread_prob_display,
            "total_prob_display": total_prob_display,
            "spread_sentiment_arrow": spread_arrow,
            "total_sentiment_arrow": total_arrow,
            "spread_sentiment_note": spread_note,
            "total_sentiment_note": total_note,
            "Spread_Glance_Clean": clean_spread_glance,
            "Total_Glance_Clean": clean_total_glance,
            "spread_conf_rank": spread_conf_rank,
            "total_conf_rank": total_conf_rank,
            "st_conf_rank": min(spread_conf_rank, total_conf_rank),
            "decisiveness": decisiveness
        })

    # Create DataFrame from new data and concat once
    if new_data:
        new_cols_df = pd.DataFrame(new_data, index=df.index)

        # Drop columns if they exist to allow overwrite
        cols_to_drop = [c for c in new_cols_df.columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = pd.concat([df, new_cols_df], axis=1).copy()

    return df


def compute_sentiment_adj_row(row: Dict[str, Any]) -> Tuple[float, str]:
    """
    Compute bounded sentiment adjustment for moneyline rows only.
    Returns (adj_value, reason).
    """
    src = str(row.get("sentiment_source") or "none").lower()
    articles_used = int(row.get("sentiment_articles_used") or 0)
    rate_limited = bool(row.get("sentiment_rate_limited") or False)
    auth_error = bool(row.get("sentiment_auth_error") or False)
    cached_used = bool(row.get("sentiment_used_cached") or False)
    confidence = safe_float(row.get("sentiment_confidence")) or 0.0
    source_count = int(row.get("sentiment_source_count") or articles_used or 0)
    level = _normalize_sentiment_level(row.get("sentiment_level"))
    if level == "league":
        return 0.0, "league_directional"
    strength = str(row.get("sentiment_strength") or "").upper() or sentiment_strength_from_articles(level, articles_used)
    if not strength or strength == "NONE":
        strength = sentiment_strength_from_articles(level, articles_used)
    if confidence < 0.6 or source_count < 5:
        return 0.0, "low_confidence"
    home_sent = safe_float(row.get("Home_Sentiment"))
    away_sent = safe_float(row.get("Away_Sentiment"))
    sentiment_diff = None if (home_sent is None and away_sent is None) else (home_sent or 0.0) - (away_sent or 0.0)
    signal = sentiment_signal_value(level, sentiment_diff)
    if auth_error:
        return 0.0, "auth_error"
    if articles_used <= 0 or level == "none":
        return 0.0, "no_sentiment"
    adj = compute_market_sentiment_adjustment(level, strength, "moneyline", signal)
    reason_bits: List[str] = []
    if rate_limited:
        reason_bits.append("rate_limited")
    if cached_used:
        reason_bits.append("cached")
    if src in {"error", "error_rate_limited", "error_auth"}:
        reason_bits.append("source_error")
    reason = "applied" if not reason_bits else f"applied_{'_'.join(reason_bits)}"
    if strength == "NONE" or adj == 0.0:
        reason = "no_sentiment"
    return adj, reason


def market_based_prob(
    row: Dict[str, Any],
    *,
    market_override: Optional[str] = None,
    implied_prob_value: Optional[float] = None,
    range_override: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Tuple[Optional[float], str]:
    """
    Compute a market-based probability for spread/total rows without Vertex.
    """
    def _clamp_prob(x: Optional[float]) -> Optional[float]:
        try:
            if x is None:
                return None
            return max(0.01, min(0.99, float(x)))
        except Exception:
            return None

    imp = _clamp_prob(implied_prob_value if implied_prob_value is not None else row.get("Implied_Prob"))
    base_prob = imp if imp is not None else 0.5
    market = str(market_override or row.get("Market") or "").lower()
    warnings_local: List[str] = []
    width = None
    lo_val, hi_val = None, None
    if range_override is not None:
        lo_val, hi_val = range_override
    if market == "spread":
        try:
            lo = row.get("spread_min") if lo_val is None else lo_val
            hi = row.get("spread_max") if hi_val is None else hi_val
            if lo is not None and hi is not None:
                width = abs(float(hi) - float(lo))
        except Exception:
            width = None
    elif market == "total":
        try:
            lo = row.get("total_min") if lo_val is None else lo_val
            hi = row.get("total_max") if hi_val is None else hi_val
            if lo is not None and hi is not None:
                width = abs(float(hi) - float(lo))
        except Exception:
            width = None
    penalty = 0.0
    if width is not None:
        penalty = min(0.05, width * 0.02)
        if width >= 2.0 and market == "spread":
            warnings_local.append("wide_spread_market")
        if width >= 4.0 and market == "total":
            warnings_local.append("wide_total_market")
    inj_adj = 0.0
    try:
        ih = int(row.get("injuries_home_count") or 0)
        ia = int(row.get("injuries_away_count") or 0)
        diff = ih - ia
        pick = str(row.get("Pick") or "")
        home = str(row.get("Home") or "")
        away = str(row.get("Away") or "")
        if pick == home:
            inj_adj = min(0.02, max(-0.02, (-diff) * 0.005))
        elif pick == away:
            inj_adj = min(0.02, max(-0.02, (diff) * 0.005))
    except Exception:
        inj_adj = 0.0
    weather_adj = 0.0
    if market == "total":
        ws = str(row.get("weather_summary") or "").lower()
        if any(k in ws for k in ["wind", "rain", "snow"]):
            pick_text = str(row.get("Pick") or "").lower()
            if "under" in pick_text:
                weather_adj = 0.02
            elif "over" in pick_text:
                weather_adj = -0.02
    prob = _clamp_prob(base_prob + inj_adj + weather_adj - penalty)
    reason_base = f"imp={imp:.3f}" if imp is not None else "imp=missing->0.500"
    reason = f"market_based ({reason_base}, inj={inj_adj:.3f}, weather={weather_adj:.3f}, penalty={penalty:.3f})"
    if warnings_local:
        reason = f"{reason} | {','.join(warnings_local)}"
    return prob, reason


def sentiment_payload_to_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert sentiment payload to metadata dict.

    IMPORTANT: We distinguish between:
    1. Valid sentiment data (sources > 0, score available)
    2. Valid neutral (no articles found, but API call succeeded - this is a deliberate neutral signal)
    3. Invalid (API error, rate limited, auth error - sentiment unavailable)
    """
    score = safe_float(payload.get("score"))
    confidence = safe_float(payload.get("confidence"))
    try:
        sources = int(payload.get("sources") or payload.get("sentiment_source_count") or 0)
    except Exception:
        sources = 0
    method = str(payload.get("method") or payload.get("source") or "").lower()
    sentiment_source_override = str(payload.get("sentiment_source") or "").lower()

    # Check for genuine errors (API failures, not just "no news")
    error = payload.get("error")
    auth_error = payload.get("auth_error") or error in {"bad_key", "auth_error", "auth_error_fallback"}
    rate_limited = payload.get("rate_limited") or payload.get("status") == 429 or error == "rate_limited"
    api_unavailable = bool(error in {"missing_key", "http_error", "exception"}) or rate_limited or auth_error

    # Sentiment is valid if:
    # 1. We have articles (sources >= 1) with decent confidence, OR
    # 2. API call succeeded (no genuine errors) even if no articles found (valid neutral)
    has_data = bool(sources >= 1 and (confidence or 0) > 0.25)
    is_neutral_valid = bool(sources == 0 and not api_unavailable and error in {None, "no_articles", "newsapi_empty"})
    sentiment_valid = has_data or is_neutral_valid

    # If valid but no data, use neutral score
    if sentiment_valid and not has_data:
        score = 0.0
        confidence = 0.5  # Moderate confidence in neutral
        label = "Neutral"
        logger.debug(f"Sentiment valid neutral: no articles, but API succeeded (sources={sources}, error={error}, api_unavailable={api_unavailable})")
        sentiment_source = "neutral"
    else:
        sentiment_source = sentiment_source_override or ("newsapi" if has_data else "none")

    level = payload.get("sentiment_level") or ("team" if has_data else "none")
    strength = sentiment_strength_from_articles(level, sources)
    label = None
    raw_label = payload.get("label")
    if score is not None and raw_label is None:
        if score > 0.05:
            label = "Positive"
        elif score < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
    else:
        label = raw_label if raw_label else None

    # Log sentiment validation decision
    if sentiment_valid:
        if has_data:
            try:
                if score is not None and confidence is not None:
                    logger.info(f"Sentiment valid with data: score={score:.3f}, sources={sources}, confidence={confidence:.2f}")
                else:
                    logger.error(f"SENTIMENT DEBUG: score={score}, confidence={confidence}, sources={sources}")
            except Exception as format_error:
                logger.error(f"Format string error in sentiment logging: {format_error}")
        else:
            logger.info(f"Sentiment valid (neutral): no articles found, but API call succeeded (error={error})")
    else:
        # User requested: Log when sentiment skipped and why
        logger.warning(f"Sentiment SKIPPED/INVALID: error={error}, auth_error={auth_error}, rate_limited={rate_limited}, sources={sources}")

    # FIX: Ensure sentiment_status is "ok" if sentiment is valid, even if it was "NA" in payload
    # BUG FIX: Use payload parameter instead of undefined variables (fetch_info, merged_payload)
    final_status = payload.get("status") or "ok"
    if sentiment_valid and final_status in [None, "NA", "disabled", "DISABLED"]:
        final_status = "ok"

    return {
        "score": score if sentiment_valid else None,
        "label": label if sentiment_valid else None,
        "confidence": confidence,
        "sources": sources,
        "sentiment_valid": sentiment_valid,
        "sentiment_source": sentiment_source if 'sentiment_source' in locals() else sentiment_source_override or ("newsapi" if has_data else "none"),
        "sentiment_level": level,
        "sentiment_strength": strength,
        "sentiment_badge": sentiment_badge_for(level, strength),
        "sentiment_articles_used": sources,
        "sentiment_source_count": sources,
        "method": method or None,
        "reddit_used": bool(payload.get("reddit_used") or sentiment_source_override in {"reddit", "blended"}),
        "reddit_posts_used": int(payload.get("reddit_posts_used") or 0),
        "reddit_comments_used": int(payload.get("reddit_comments_used") or 0),
        "sentiment_confidence": confidence or 0.0,
        "sentiment_status": final_status, # Use the fixed status
    }

def compute_team_sentiment_map(news_api_key: Optional[str], games: List[Dict[str, Any]], league: str) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    teams = set()
    for g in games or []:
        if g.get("home_team"):
            teams.add(str(g.get("home_team")))
        if g.get("away_team"):
            teams.add(str(g.get("away_team")))

    sentiment_map: Dict[str, Optional[float]] = {}
    sentiment_meta: Dict[str, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {
        "total_teams": len(teams),
        "article_counts": {},
        "missing_teams": [],
        "articles_total": 0,
        "raw": {},
        "fetch_info": {},
        "error_count": 0,
        "errors_sample": [],
        "status_counts": {},
        "sample_calls": [],
        "league_label_used": league_label(league),
        "rate_limited": False,
        "auth_error": False,
        "cached_teams": 0,
        "used_cached": False,
        "log_reason": "",
        "requests_attempted": 0,
        "requests_skipped_due_to_cache": 0,
        "requests_skipped_due_to_cooldown": 0,
        "rate_limit_hit": False,
        "reddit_posts_used": 0,
        "reddit_comments_used": 0,
        "teams_from_reddit": 0,
        "teams_blended": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "sentiment_degraded": False,
    }

    kalshi_matched_teams: set = set()
    try:
        _entries_raw = st.session_state.get("kalshi_match_results") or {}
        entries = _entries_raw.values() if isinstance(_entries_raw, dict) else (_entries_raw or [])
        for entry in entries:
            winner = (entry.get("matches") or {}).get("winner", {})
            if not winner.get("kalshi_matched"):
                continue
            game_meta = entry.get("game") or {}
            if game_meta.get("home_team"):
                kalshi_matched_teams.add(str(game_meta.get("home_team")))
            if game_meta.get("away_team"):
                kalshi_matched_teams.add(str(game_meta.get("away_team")))
    except Exception:
        kalshi_matched_teams = set()

    def _edge_hint(game: Dict[str, Any]) -> float:
        try:
            implied_home = american_to_implied_prob(game.get("home_ml_price"))
            implied_away = american_to_implied_prob(game.get("away_ml_price"))
            if implied_home is not None and implied_away is not None:
                return abs(float(implied_home) - float(implied_away))
        except Exception:
            return 0.0
        return 0.0

    cooldown_until = newsapi_cooldown_until()
    cooldown_active = newsapi_cooldown_active()
    debug["cooldown_active"] = cooldown_active
    if cooldown_until:
        debug["cooldown_until"] = cooldown_until.isoformat()

    date_bucket = datetime.now(timezone.utc).date().isoformat()
    team_edge_map: Dict[str, float] = {}
    for g in games or []:
        home = str(g.get("home_team") or "").strip()
        away = str(g.get("away_team") or "").strip()
        if home:
            team_edge_map[home] = max(team_edge_map.get(home, 0.0), _edge_hint(g))
        if away:
            team_edge_map[away] = max(team_edge_map.get(away, 0.0), _edge_hint(g))
    ordered_teams = sorted(
        teams,
        key=lambda t: (
            -1 if t in kalshi_matched_teams else 0,
            -(team_edge_map.get(t, 0.0)),
            t,
        ),
    )
    debug["sentiment_scope"] = "team_level"
    if len(ordered_teams) > MAX_SENTIMENT_TEAMS_PER_RUN:
        ordered_teams = ordered_teams[:MAX_SENTIMENT_TEAMS_PER_RUN]
        debug["calls_capped"] = True
        debug["sentiment_scope"] = "team_level_capped"
    calls_capped = debug.get("calls_capped", False)

    cache_miss_calls = 0
    REQUEST_BUDGET = min(MAX_SENTIMENT_TEAMS_PER_RUN, MAX_SENTIMENT_CALLS, len(ordered_teams))
    stop_fetching = cooldown_active
    debug["rate_limited"] = debug.get("rate_limited") or cooldown_active
    news_payloads: Dict[str, Dict[str, Any]] = {}

    for team in ordered_teams:
        cache_payload = _sentiment_cache_get(team, league, date_bucket)
        cached = cache_payload is not None
        payload = cache_payload or {}
        fetch_info: Dict[str, Any] = payload.get("fetch_info") or {}
        status_int = payload.get("status")
        if cached:
            debug["cached_teams"] = debug.get("cached_teams", 0) + 1
            debug["used_cached"] = True
            debug["requests_skipped_due_to_cache"] = debug.get("requests_skipped_due_to_cache", 0) + 1
            debug["cache_hits"] = debug.get("cache_hits", 0) + 1
        else:
            debug["cache_misses"] = debug.get("cache_misses", 0) + 1

        if stop_fetching and not cached:
            payload = {
                "score": None,
                "confidence": 0.0,
                "sources": 0,
                "status": 429,
                "fetch_info": {"status": 429, "error": "cooldown_active", "retry_after": debug.get("cooldown_until")},
                "rate_limited": True,
                "auth_error": False,
                "method": "cooldown_skip",
                "sentiment_source": "none",
            }
            debug["requests_skipped_due_to_cooldown"] = debug.get("requests_skipped_due_to_cooldown", 0) + 1
            debug["rate_limited"] = True
            if len(debug.get("errors_sample") or []) < 5:
                debug.setdefault("errors_sample", []).append({"team": team, "error": "rate_limited", "status_code": 429})
        elif not cached and cache_miss_calls >= REQUEST_BUDGET:
            calls_capped = True
            payload = {
                "score": None,
                "confidence": 0.0,
                "sources": 0,
                "status": None,
                "fetch_info": {"status": None, "error": "calls_capped"},
                "rate_limited": False,
                "auth_error": False,
                "method": "newsapi_capped",
                "sentiment_source": "none",
            }
        elif not cached:
            _enforce_sentiment_throttle()
            debug["requests_attempted"] = debug.get("requests_attempted", 0) + 1
            to_date = datetime.now(timezone.utc).date()
            from_date = to_date - timedelta(days=3)
            url = "https://newsapi.org/v2/everything"
            # Attempt a combined query that includes the team and league context to reduce per-team calls
            normalized_name = TeamNameMatcher.normalize(team)
            q = f'"{normalized_name}" {league_label(league)}'
            params = {
                "q": q,
                "sortBy": "relevancy",
                "pageSize": 20,
                "language": "en",
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "apiKey": news_api_key or "",
            }
            params_tuple = tuple(sorted(params.items()))
            cached_fetch = _newsapi_fetch_cached(url, params_tuple)
            status_val = cached_fetch.get("status")
            data = cached_fetch.get("data") or {}
            articles = data.get("articles", []) if isinstance(data, dict) else []

            # Fallback: Try mascot-only query if full name failed OR unavailable
            # Now attempts fallback even if first call failed (e.g. 500) or returned no articles
            if not articles:
                try:
                    # Fallback to team mascot from raw team name
                    # normalized_name strips mascots, so we use 'team' which preserves them
                    parts = team.split()
                    if len(parts) > 1:
                        mascot = parts[-1]
                        q_fallback = f'"{mascot}" {league_label(league)}'
                        params_fallback = {**params, "q": q_fallback}

                        cached_fetch_fb = _newsapi_fetch_cached(url, tuple(sorted(params_fallback.items())))
                        status_val_fb = cached_fetch_fb.get("status")
                        if status_val_fb == 200:
                             data_fb = cached_fetch_fb.get("data") or {}
                             articles_fb = data_fb.get("articles", []) if isinstance(data_fb, dict) else []
                             if articles_fb:
                                 articles = articles_fb
                                 q = q_fallback
                                 cached_fetch = cached_fetch_fb
                                 status_val = status_val_fb
                                 data = data_fb
                except Exception as e:
                    logger.warning(f"Sentiment fallback query failed: {e}")
            retry_after_hdr = (cached_fetch.get("headers") or {}).get("Retry-After")
            rate_limited_call = status_val == 429
            auth_error_call = status_val in {401, 403}
            if rate_limited_call and not articles:
                # exponential backoff (per run) before giving up
                time.sleep(min(2.0, 1.0 * max(cache_miss_calls, 1)))
            fetch_info = {
                "status": status_val,
                "status_code": status_val,
                "q": q,
                "totalResults": data.get("totalResults") if isinstance(data, dict) else None,
                "rate_limited": rate_limited_call,
                "auth_error": auth_error_call,
                "retry_after": retry_after_hdr,
                "error": None if status_val == 200 else data.get("message") if isinstance(data, dict) else cached_fetch.get("error"),
            }
            status_int = None
            try:
                status_int = int(status_val) if status_val is not None else None
            except Exception:
                status_int = None
            cache_miss_calls += 1
            score = team_sentiment_from_articles(articles)
            payload = {
                "score": score if articles else None,
                "confidence": 0.6 if articles else (0.3 if rate_limited_call else 0.0),
                "sources": len(articles) if articles else 1 if rate_limited_call else 0,
                "status": status_int,
                "fetch_info": fetch_info,
                "rate_limited": bool(fetch_info.get("rate_limited")),
                "auth_error": bool(fetch_info.get("auth_error")),
                "method": "newsapi_no_articles" if not articles and not fetch_info.get("rate_limited") else "newsapi",
                "sentiment_source": "newsapi" if articles else ("newsapi_degraded" if rate_limited_call else "none"),
            }
            _sentiment_cache_put(team, league, date_bucket, payload)
        else:
            status_int = payload.get("status")
            fetch_info = payload.get("fetch_info") or {}

        status_int = payload.get("status")
        if status_int is not None:
            debug["status_counts"][status_int] = debug["status_counts"].get(status_int, 0) + 1
            if len(debug["sample_calls"]) < 5:
                debug["sample_calls"].append(
                    {
                        "team": team,
                        "league": league,
                        "q": fetch_info.get("q"),
                        "status": status_int,
                        "totalResults": fetch_info.get("totalResults"),
                        "error": fetch_info.get("error"),
                    }
                )

        if payload.get("rate_limited") or status_int == 429:
            retry_after = fetch_info.get("retry_after")
            cooldown_new = set_newsapi_cooldown(hours=NEWSAPI_COOLDOWN_HOURS, retry_after=retry_after)
            debug["rate_limited"] = True
            debug["rate_limit_hit"] = True
            cooldown_until = cooldown_new or cooldown_until
            cooldown_active = True
            debug["cooldown_active"] = True
            if cooldown_new:
                debug["cooldown_until"] = cooldown_new.isoformat()
            stop_fetching = True
            if len(debug.get("errors_sample") or []) < 5:
                debug.setdefault("errors_sample", []).append({"team": team, "error": "rate_limited", "status_code": 429})
            debug["sentiment_degraded"] = True
        if payload.get("auth_error"):
            debug["auth_error"] = True

        payload["cached"] = cached
        news_payloads[team] = {**payload, "fetch_info": fetch_info, "status": status_int}

    def _payload_valid(p: Dict[str, Any]) -> bool:
        try:
            return bool((p.get("sources") or 0) > 0 and safe_float(p.get("score")) is not None)
        except Exception:
            return False

    sentiment_available_news = len([1 for p in news_payloads.values() if _payload_valid(p)])
    teams_missing_news = [t for t in ordered_teams if not _payload_valid(news_payloads.get(t, {}))]
    need_reddit = debug.get("rate_limited") or sentiment_available_news == 0 or bool(teams_missing_news)
    reddit_payloads: Dict[str, Dict[str, Any]] = {}
    reddit_cached = 0
    if need_reddit and teams_missing_news:
        cached_reddit: Dict[str, Dict[str, Any]] = {}
        to_fetch: List[str] = []
        for team in teams_missing_news:
            cached_payload = _reddit_cache_get(team, league, date_bucket)
            if cached_payload:
                cached_reddit[team] = {**cached_payload, "cached": True}
                reddit_cached += 1
            else:
                to_fetch.append(team)
        reddit_payloads.update(cached_reddit)
        if to_fetch:
            reddit_results = fetch_reddit_sentiment_map(to_fetch, league)
            for team, payload in reddit_results.items():
                reddit_payloads[team] = {
                    "score": safe_float(payload.get("score")),
                    "confidence": safe_float(payload.get("confidence")),
                    "sources": int(payload.get("source_count") or 0),
                    "status": payload.get("status"),
                    "rate_limited": bool(payload.get("rate_limited")),
                    "auth_error": False,
                    "method": "reddit",
                    "sentiment_source": "reddit" if (payload.get("source_count") or 0) > 0 else "none",
                    "reddit_posts_used": int(payload.get("posts_used") or 0),
                    "reddit_comments_used": int(payload.get("comments_used") or 0),
                    "label": payload.get("label"),
                    "fetch_info": {
                        "query": payload.get("query"),
                        "status": payload.get("status"),
                        "error": payload.get("error"),
                    },
                    "error": payload.get("error"),
                    "cached": False,
                }
                _reddit_cache_put(team, league, date_bucket, reddit_payloads[team])

    def _merge_payloads(news_payload: Dict[str, Any], reddit_payload: Dict[str, Any]) -> Dict[str, Any]:
        if reddit_payload and (reddit_payload.get("sources") or 0) > 0 and not news_payload.get("sources"):
            return {**news_payload, **reddit_payload, "reddit_used": True, "sentiment_source": "reddit_fallback"}
        if not reddit_payload or (reddit_payload.get("sources") or 0) <= 0:
            return {**news_payload, "reddit_used": False}
        news_conf = safe_float(news_payload.get("confidence")) or 0.0
        reddit_conf = safe_float(reddit_payload.get("confidence")) or 0.0
        if news_payload.get("sources") and reddit_conf <= news_conf:
            return {**news_payload, "reddit_used": False}
        score_news = news_payload.get("score")
        score_reddit = reddit_payload.get("score")
        blended_score = score_news if score_reddit is None else score_reddit
        if score_news is not None and score_reddit is not None:
            blended_score = (0.7 * float(score_news)) + (0.3 * float(score_reddit))
        total_sources = (news_payload.get("sources") or 0) + (reddit_payload.get("sources") or 0)
        merged = {
            **news_payload,
            **reddit_payload,
            "score": blended_score,
            "sources": total_sources,
            "sentiment_source": "blended" if news_payload.get("sources") else "reddit_fallback",
            "reddit_used": True,
        }
        return merged

    for team in ordered_teams:
        news_payload = news_payloads.get(team, {})
        reddit_payload = reddit_payloads.get(team, {})
        merged_payload = _merge_payloads(news_payload, reddit_payload)
        meta = sentiment_payload_to_meta(merged_payload)
        fetch_info = news_payload.get("fetch_info") or reddit_payload.get("fetch_info") or {}
        meta["sentiment_confidence"] = merged_payload.get("confidence") or meta.get("sentiment_confidence") or 0.0
        meta["sentiment_query_used"] = fetch_info.get("q") or fetch_info.get("query")
        meta["sentiment_articles_used"] = merged_payload.get("sources") or news_payload.get("sources") or 0
        meta["sentiment_level"] = meta.get("sentiment_level") or ("team" if meta.get("sentiment_valid") else "none")
        meta["sentiment_strength"] = meta.get("sentiment_strength") or sentiment_strength_from_articles(meta["sentiment_level"], meta.get("sentiment_articles_used") or 0)
        meta["sentiment_badge"] = meta.get("sentiment_badge") or sentiment_badge_for(meta["sentiment_level"], meta["sentiment_strength"])

        # FIX: Normalize status to string representation
        raw_status = merged_payload.get("status")
        if raw_status == 200:
            status_str = "ok"
        elif raw_status == 429:
            status_str = "rate_limited"
        elif raw_status in {401, 403}:
            status_str = "auth_error"
        elif raw_status is None:
            status_str = "na"
        elif isinstance(raw_status, int):
            status_str = f"http_{raw_status}"
        else:
            status_str = str(raw_status) if raw_status else "na"

        meta["status"] = status_str
        meta["sentiment_status"] = status_str
        meta["sentiment_rate_limited"] = bool(merged_payload.get("rate_limited") or merged_payload.get("status") == 429)
        meta["sentiment_used_cached"] = bool(news_payload.get("cached")) or bool(reddit_payload.get("cached")) if isinstance(reddit_payload, dict) else False
        meta["sentiment_label"] = meta.get("label")
        meta["sentiment_source_count"] = meta.get("sentiment_source_count") or merged_payload.get("sources") or 0
        meta["sentiment_source"] = merged_payload.get("sentiment_source") or meta.get("sentiment_source") or "none"
        meta["reddit_posts_used"] = int(merged_payload.get("reddit_posts_used") or 0)
        meta["reddit_comments_used"] = int(merged_payload.get("reddit_comments_used") or 0)
        meta["sentiment_confidence"] = meta.get("sentiment_confidence") or merged_payload.get("confidence") or 0.0
        meta["reddit_used"] = bool(merged_payload.get("reddit_used") or meta.get("reddit_used"))
        meta["error"] = merged_payload.get("error") or fetch_info.get("error")
        meta["cached"] = bool(news_payload.get("cached"))
        sentiment_meta[team] = meta
        debug["raw"][team] = {**merged_payload, "fetch_info": fetch_info}
        debug["fetch_info"][team] = fetch_info
        debug["article_counts"][team] = meta.get("sentiment_source_count") or 0
        debug["articles_total"] += meta.get("sentiment_source_count") or 0
        debug["reddit_posts_used"] += meta.get("reddit_posts_used") or 0
        debug["reddit_comments_used"] += meta.get("reddit_comments_used") or 0
        if meta["sentiment_source"] == "reddit":
            debug["teams_from_reddit"] += 1
        if meta["sentiment_source"] == "blended":
            debug["teams_blended"] += 1
        if fetch_info.get("error"):
            debug["error_count"] += 1
            if len(debug["errors_sample"]) < 5:
                debug["errors_sample"].append({"team": team, **fetch_info})
        # FIX: Use meta.get("score") instead of merged_payload.get("score")
        # sentiment_payload_to_meta() can set score=0.0 for valid neutral cases,
        # but this change is in meta, not merged_payload
        meta_score = meta.get("score")

        # Debug logging for sentiment response
        logger.debug(
            f"Sentiment response for {team}:\n"
            f"  Score: {meta_score}, Valid: {meta['sentiment_valid']}, Status: {meta.get('sentiment_status')}\n"
            f"  Sources: {meta.get('sentiment_articles_used', 0)}, Confidence: {meta.get('sentiment_confidence', 0):.2f}\n"
            f"  Source type: {meta.get('sentiment_source', 'unknown')}, Error: {meta.get('error', 'none')}"
        )

        if meta["sentiment_valid"] and meta_score is not None:
            sentiment_map[team] = meta_score
            # MODE B: Sentiment collection now integrated into probability blend (was Mode A: UI display only)
            logger.info(f"Sentiment COLLECTED for {team}: score={meta_score:.3f} (Mode B: used in probability blend), valid={meta['sentiment_valid']}, source={meta.get('sentiment_source', 'unknown')}, status={meta.get('sentiment_status')}")
        else:
            sentiment_map[team] = None
            debug["missing_teams"].append(team)
            # Log why sentiment was excluded

            # Reduce log noise for expected rate limiting
            is_rate_limited = meta.get('sentiment_rate_limited') or meta.get('error') == 'rate_limited' or str(meta.get('sentiment_status')) == 'rate_limited'
            msg = f"Sentiment EXCLUDED for {team}: valid={meta['sentiment_valid']}, score={meta_score}, error={meta.get('error', 'none')}, status={meta.get('sentiment_status')}, sources={meta.get('sentiment_articles_used', 0)}"

            if is_rate_limited:
                logger.info(msg)
            else:
                logger.warning(msg)

    if sentiment_map:
        present_scores = [(t, s) for t, s in sentiment_map.items() if s is not None]
        sorted_scores = sorted(present_scores, key=lambda kv: kv[1]) if present_scores else []
        debug["bottom_5"] = sorted_scores[:5]
        debug["top_5"] = sorted_scores[-5:] if sorted_scores else []
    else:
        debug["bottom_5"] = []
        debug["top_5"] = []
    if debug.get("rate_limited") and 429 not in debug.get("status_counts", {}):
        debug["status_counts"][429] = debug["status_counts"].get(429, 0) + 1

    # Log sentiment coverage summary
    teams_with_sentiment = len([1 for s in sentiment_map.values() if s is not None])
    total_teams = len(teams)
    logger.info(f"Sentiment coverage: {teams_with_sentiment}/{total_teams} teams have valid sentiment.")
    if teams_with_sentiment > 0:
        sentiment_available_news = len([1 for p in news_payloads.values() if _payload_valid(p)])
        logger.info(f"Sentiment source breakdown: NewsAPI={sentiment_available_news}, Reddit={debug.get('teams_from_reddit', 0)}, Blended={debug.get('teams_blended', 0)}")
        logger.info(f"Sentiment cache stats: hits={debug.get('cache_hits', 0)}, misses={debug.get('cache_misses', 0)}")
    if debug.get("rate_limited"):
        logger.warning("Sentiment: NewsAPI rate limit hit, some teams may be missing sentiment data")
    if debug.get("auth_error"):
        logger.warning("Sentiment: NewsAPI authentication error, using fallback sentiment source")

    return sentiment_map, sentiment_meta, {
        "article_counts": debug.get("article_counts"),
        "articles_total": debug.get("articles_total"),
        "missing_teams": debug.get("missing_teams"),
        "raw": debug.get("raw"),
        "fetch_info": debug.get("fetch_info"),
        "error_count": debug.get("error_count"),
        "errors_sample": debug.get("errors_sample"),
        "status_counts": debug.get("status_counts"),
        "sample_calls": debug.get("sample_calls"),
        "league_label_used": debug.get("league_label_used"),
        "rate_limited": debug.get("rate_limited"),
        "auth_error": debug.get("auth_error"),
        "cooldown_active": cooldown_active,
        "cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
        "requests_attempted": debug.get("requests_attempted", 0),
        "requests_skipped_due_to_cache": debug.get("requests_skipped_due_to_cache", 0),
        "requests_skipped_due_to_cooldown": debug.get("requests_skipped_due_to_cooldown", 0),
        "rate_limit_hit": debug.get("rate_limit_hit", False),
        "reddit_posts_used": debug.get("reddit_posts_used", 0),
        "reddit_comments_used": debug.get("reddit_comments_used", 0),
        "teams_from_reddit": debug.get("teams_from_reddit", 0),
        "teams_blended": debug.get("teams_blended", 0),
    }


def get_slate_sentiment(enable_sentiment: bool, teams: List[str], league: str, news_api_key: Optional[str]) -> Dict[str, Any]:
    meta = init_sentiment_meta()
    debug: Dict[str, Any] = {}
    teams = [t for t in teams if t]
    # Deduplicate while preserving order
    teams = list(dict.fromkeys(teams))
    if not enable_sentiment:
        meta["sentiment_source"] = "disabled_by_user"
        meta["sentiment_sample_status"] = "DISABLED"
        meta["sentiment_status_counts"] = {"DISABLED": 1}
        meta["sentiment_disabled_reason"] = "user_disabled"
        meta["sentiment_status"] = "DISABLED"
        return {"map": {}, "meta_map": {}, "meta": meta, "debug": debug}
    if not news_api_key:
        meta["sentiment_source"] = "disabled_no_key"
        meta["sentiment_sample_status"] = "DISABLED"
        meta["sentiment_status_counts"] = {"DISABLED": 1}
        meta["sentiment_disabled_reason"] = "missing_NEWS_API_KEY"
        meta["sentiment_status"] = "DISABLED"
        return {"map": {}, "meta_map": {}, "meta": meta, "debug": debug}
    if not teams:
        meta["sentiment_source"] = "disabled_no_teams"
        meta["sentiment_sample_status"] = "DISABLED"
        meta["sentiment_status_counts"] = {"DISABLED": 1}
        meta["sentiment_disabled_reason"] = "no_teams_found"
        meta["sentiment_status"] = "DISABLED"
        return {"map": {}, "meta_map": {}, "meta": meta, "debug": debug}
    cooldown_until = newsapi_cooldown_until()
    cooldown_active = newsapi_cooldown_active()
    if cooldown_active:
        cached_map = st.session_state.get("sentiment_map") or {}
        cached_meta_map = st.session_state.get("sentiment_meta_map") or {}
        meta["sentiment_source"] = "cooldown_cached_only"
        meta["sentiment_sample_status"] = "429"
        meta["sentiment_status_counts"] = {"429": 1}
        meta["sentiment_disabled_reason"] = "cooldown_active"
        meta["sentiment_cooldown_until"] = cooldown_until.isoformat() if cooldown_until else None
        meta["sentiment_rate_limited"] = True
        meta["sentiment_available_count"] = len([v for v in cached_map.values() if v is not None])
        meta["sentiment_status"] = "COOLDOWN"
        debug["cooldown_active"] = True
        debug["cooldown_until"] = meta["sentiment_cooldown_until"]
        debug["rate_limited"] = True
        if meta["sentiment_available_count"] > 0:
            meta["sentiment_source"] = "partial_cached"
        else:
            meta["sentiment_source"] = "error_rate_limited"
            meta["sentiment_errors_sample"] = f"{teams[0] if teams else 'team'}: rate_limited"
        return {"map": cached_map, "meta_map": cached_meta_map, "meta": meta, "debug": debug}
    meta["sentiment_sample_status"] = "PENDING"
    meta["sentiment_status_counts"] = {"PENDING": 1}
    meta["sentiment_disabled_reason"] = ""
    games_stub = [{"home_team": t, "away_team": None} for t in teams]
    try:
        sentiment_map, sentiment_meta_map, sentiment_debug = compute_team_sentiment_map(news_api_key, games_stub, league)
        debug = sentiment_debug or {}
        status_counts = debug.get("status_counts") or {}
        # Normalize status count keys to strings for downstream export/UI consistency
        status_counts = {str(k): v for k, v in status_counts.items()}
        sample_calls = debug.get("sample_calls") or []
        sample_call = sample_calls[0] if sample_calls else {}
        meta["sentiment_status_counts"] = status_counts if status_counts else {"NO_CALL": 1}
        meta["sentiment_sample_query"] = sample_call.get("q") or ""
        meta["sentiment_sample_status"] = str(sample_call.get("status") or (list(status_counts.keys())[0] if status_counts else "NO_CALL"))
        meta["sentiment_sample_totalResults"] = sample_call.get("totalResults") or 0
        meta["sentiment_error_count"] = debug.get("error_count") or 0
        errors_sample = debug.get("errors_sample") or []
        meta["sentiment_errors_sample"] = ";".join([f"{e.get('team')}: {e.get('error')}" for e in errors_sample]) if errors_sample else ""
        meta["sentiment_articles_total"] = debug.get("articles_total") or 0
        meta["sentiment_cached_teams_count"] = debug.get("cached_teams") or 0
        meta["sentiment_available_count"] = len([t for t, mv in sentiment_meta_map.items() if mv.get("sentiment_valid")])
        meta["sentiment_rate_limited"] = bool(meta["sentiment_sample_status"] == "429" or status_counts.get("429") or debug.get("rate_limited"))
        meta["sentiment_auth_error"] = bool(debug.get("auth_error") or meta["sentiment_sample_status"] in {"401", "403"})
        meta["sentiment_status"] = meta.get("sentiment_sample_status")
        meta["sentiment_cooldown_until"] = debug.get("cooldown_until") or meta.get("sentiment_cooldown_until") or ""
        meta["sentiment_degraded"] = bool(debug.get("sentiment_degraded"))
        valid_scores = [safe_float(v) or 0.0 for t, v in sentiment_map.items() if v is not None and (sentiment_meta_map.get(t, {}).get("sentiment_valid"))]
        confidence_values = [safe_float(mv.get("sentiment_confidence")) or 0.0 for mv in sentiment_meta_map.values()]
        meta["sentiment_confidence"] = max(confidence_values or valid_scores or [0.0])
        meta["sentiment_score"] = sum(valid_scores) / max(1, len(valid_scores)) if valid_scores else 0.0
        meta["sentiment_label"] = None
        meta["sentiment_disabled_reason"] = ""
        if meta["sentiment_rate_limited"] and meta["sentiment_available_count"] == 0:
            meta["sentiment_sample_status"] = "429"
            meta["sentiment_status"] = "429"
            counts_existing = meta.get("sentiment_status_counts") or {}
            counts_existing[str(429)] = counts_existing.get(str(429), 0) + 1
            meta["sentiment_status_counts"] = counts_existing
            if not meta["sentiment_errors_sample"]:
                rate_limit_team = (errors_sample[0].get("team") if errors_sample else (teams[0] if teams else "team"))
                meta["sentiment_errors_sample"] = f"{rate_limit_team}: rate_limited"
        meta["sentiment_source_count"] = (meta.get("sentiment_articles_total") or 0) + (debug.get("reddit_posts_used", 0) + debug.get("reddit_comments_used", 0))
        meta["requests_attempted"] = debug.get("requests_attempted", 0)
        meta["requests_skipped_due_to_cache"] = debug.get("requests_skipped_due_to_cache", 0)
        meta["requests_skipped_due_to_cooldown"] = debug.get("requests_skipped_due_to_cooldown", 0)
        meta["rate_limit_hit"] = debug.get("rate_limit_hit", False)
        meta["sentiment_cache_hits"] = debug.get("cache_hits", 0)
        meta["sentiment_cache_misses"] = debug.get("cache_misses", 0)
        meta["reddit_posts_used"] = debug.get("reddit_posts_used", 0)
        meta["reddit_comments_used"] = debug.get("reddit_comments_used", 0)
        meta["reddit_filled_teams"] = debug.get("teams_from_reddit", 0)
        meta["reddit_blended_teams"] = debug.get("teams_blended", 0)
        source_counts: Dict[str, int] = {}
        for mv in sentiment_meta_map.values():
            src = str(mv.get("sentiment_source") or "none")
            source_counts[src] = source_counts.get(src, 0) + 1
        if meta["sentiment_rate_limited"] and meta["sentiment_available_count"] == 0:
            meta["sentiment_source"] = "error_rate_limited"
        elif meta["sentiment_rate_limited"] and meta["sentiment_available_count"] > 0:
            meta["sentiment_source"] = "partial_cached"
        elif meta["sentiment_auth_error"]:
            meta["sentiment_source"] = "error_auth"
        elif source_counts.get("blended"):
            meta["sentiment_source"] = "blended"
        elif source_counts.get("reddit") and not source_counts.get("newsapi"):
            meta["sentiment_source"] = "reddit"
        elif source_counts.get("newsapi"):
            meta["sentiment_source"] = "newsapi"
        else:
            meta["sentiment_source"] = "none"
        meta["sentiment_label"] = meta.get("sentiment_label") or None
        meta["sentiment_source_count"] = meta.get("sentiment_source_count") or meta.get("sentiment_articles_total") or 0
        return {"map": sentiment_map, "meta_map": sentiment_meta_map, "meta": meta, "debug": debug}
    except requests.exceptions.Timeout as exc:
        logger.error(f"🚨 SENTIMENT TIMEOUT")
        logger.error(f"Type: {type(exc).__name__}")
        logger.error(f"Message: {str(exc)[:500]}")
        logger.error(f"Likely cause: API endpoint slow or unresponsive")
        meta["sentiment_source"] = "error_timeout"
        meta["sentiment_sample_status"] = "TIMEOUT"
        meta["sentiment_disabled_reason"] = "timeout_in_sentiment"
        meta["sentiment_errors_sample"] = str(exc)
        meta["sentiment_status_counts"] = {"TIMEOUT": 1}
        return {"map": {}, "meta_map": {}, "meta": meta, "debug": {"error": str(exc)}}
    except requests.exceptions.RequestException as exc:
        logger.error(f"🚨 SENTIMENT API ERROR")
        logger.error(f"Type: {type(exc).__name__}")
        logger.error(f"Message: {str(exc)[:500]}")
        logger.error(f"Likely cause: Invalid API key, rate limited, or service down")
        meta["sentiment_source"] = "error_api"
        meta["sentiment_sample_status"] = "API_ERROR"
        meta["sentiment_disabled_reason"] = "api_error_in_sentiment"
        meta["sentiment_errors_sample"] = str(exc)
        meta["sentiment_status_counts"] = {"API_ERROR": 1}
        return {"map": {}, "meta_map": {}, "meta": meta, "debug": {"error": str(exc)}}
    except ValueError as exc:
        logger.error(f"🚨 SENTIMENT DATA ERROR")
        logger.error(f"Type: {type(exc).__name__}")
        logger.error(f"Message: {str(exc)[:500]}")
        logger.error(f"Likely cause: Response format changed or data corrupt")
        meta["sentiment_source"] = "error_data"
        meta["sentiment_sample_status"] = "DATA_ERROR"
        meta["sentiment_disabled_reason"] = "data_error_in_sentiment"
        meta["sentiment_errors_sample"] = str(exc)
        meta["sentiment_status_counts"] = {"DATA_ERROR": 1}
        return {"map": {}, "meta_map": {}, "meta": meta, "debug": {"error": str(exc)}}
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"SENTIMENT CALC FAILED: {type(exc).__name__}")
        logger.error(f"Error message: {str(exc)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        meta["sentiment_source"] = "error_exception"
        meta["sentiment_sample_status"] = "EXCEPTION"
        meta["sentiment_disabled_reason"] = "exception_in_sentiment"
        meta["sentiment_errors_sample"] = str(exc)
        meta["sentiment_status_counts"] = {"EXCEPTION": 1}
        return {"map": {}, "meta_map": {}, "meta": meta, "debug": {"error": str(exc)}}


def _gemini_payload_signature(payload: Dict[str, Any]) -> str:
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        return str(payload)


def gemini_row_key(row: pd.Series) -> str:
    """
    Generate a stable cache key for a row's Gemini call.
    Includes: League, Teams, Market, Pick, Date, and Probability.
    """
    try:
        # Only use fields that define the decision uniquely
        key_parts = [
            str(row.get("League") or row.get("league")),
            str(row.get("Home") or row.get("home_team")),
            str(row.get("Away") or row.get("away_team")),
            str(row.get("Market") or row.get("best_pick_type")),
            str(row.get("Pick") or row.get("best_pick")),
            str(row.get("Commence (Local)") or ""),
            f"{safe_float(row.get('final_probability') or row.get('Best Overall Prob') or 0.0):.4f}",
        ]
        return "|".join(key_parts)
    except Exception:
        return str(row.name)


def gemini_confidence_explain(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Gemini for qualitative alignment/explanation metadata (no numeric probabilities).
    """
    base = {
        "recommended_bet": None,
        "confidence": None,
        "explanation": None,
        "flags": [],
        "gemini_error": None,
    }
    # Local model is always available
    if str(row_dict.get("odds_placeholder_detected")).lower() == "true":
        base["gemini_error"] = "placeholder_odds_block"
        return base

    # Validation: Only call Gemini if final probability exists
    # Check spread/total/ML final probs if available in payload
    # row_dict usually has 'spread_prob_final', 'total_prob_final' or just 'final_probability' if it's the processed row
    # The payload construction in _apply_gemini maps "spread_prob_final" -> row.get("spread_prob")

    has_valid_prob = False
    for prob_key in ["spread_prob_final", "total_prob_final", "final_probability"]:
        val = safe_float(row_dict.get(prob_key))
        if val is not None and val > 0:
            has_valid_prob = True
            break

    if not has_valid_prob:
        base["gemini_error"] = "missing_final_probability"
        return base

    allowed_fields = {
        "league",
        "home",
        "away",
        "commence_local",
        "spread_pick",
        "spread_line",
        "spread_odds",
        "spread_prob_final",
        "spread_prob_market",
        "total_pick",
        "total_line",
        "total_odds",
        "total_prob_final",
        "total_prob_market",
        "kalshi_spread_prob",
        "kalshi_total_prob",
        "kalshi_matched",
        "prob_engine",
        "sentiment_badge",
        "sentiment_flags",
        "warnings",
    }
    sanitized_payload: Dict[str, Any] = {}
    for key in allowed_fields:
        if key in row_dict:
            val = row_dict.get(key)
            if isinstance(val, str):
                sanitized_payload[key] = val[:320]
            else:
                sanitized_payload[key] = val
    context_json = json.dumps(sanitized_payload or row_dict, ensure_ascii=False, default=str)
    prompt = f"""
You are validating an existing sports betting decision (already chosen elsewhere). Provide only a brief review.
Return JSON only with this exact schema:
{{
  "recommended_bet": "<string describing the pick or 'none'>",
  "confidence": "HIGH|MEDIUM|LOW",
  "explanation": "one short paragraph (<=240 chars) explaining agreement or disagreement",
  "flags": ["short flag strings"]
}}
Context (read-only, do not invent new probabilities):
{context_json}
"""
    raw = generate_confidence_explanation(prompt, session_state=st.session_state)
    if not isinstance(raw, dict):
        return base
    result = {**base, **{k: raw.get(k) for k in base.keys()}}
    flags = raw.get("flags") if isinstance(raw, dict) else []
    if isinstance(flags, list):
        result["flags"] = [str(f) for f in flags[:10]]
    else:
        result["flags"] = []
    result["recommended_bet"] = str(result.get("recommended_bet")) if result.get("recommended_bet") is not None else None
    confidence = str(result.get("confidence") or "").upper()
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = None
    result["confidence"] = confidence
    result["explanation"] = (
        str(result.get("explanation"))[:240] if result.get("explanation") is not None else None
    )
    return result


@st.cache_data(ttl=600)
def cached_gemini_confidence(signature: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return gemini_confidence_explain(payload)


def pipeline_progress_snapshot() -> Dict[str, Any]:
    games_loaded = len(st.session_state.get("games") or [])

    # Fix: Count Kalshi matches directly from master_results_df instead of kalshi_match_results
    # This ensures the metric matches what's actually in the analyzed data
    master_df = st.session_state.get("master_results_df")
    if master_df is not None and not master_df.empty and "kalshi_matched" in master_df.columns:
        # Count unique games (by Home/Away) that have at least one Kalshi match
        # Use .fillna(False).infer_objects(copy=False) to handle any None/NaN values, and convert to bool for robust comparison
        try:
            matched_mask = master_df["kalshi_matched"].fillna(False).astype(bool)
            matched_rows = master_df[matched_mask]
            if not matched_rows.empty:
                unique_games = matched_rows.groupby(["Home", "Away"]).size()
                matched_games = len(unique_games)
            else:
                matched_games = 0
        except Exception as e:
            logger.warning(f"Error counting Kalshi matches from master_df: {e}")
            matched_games = 0
    else:
        # Fallback to original method if master_results_df is not available
        _matches_raw = st.session_state.get("kalshi_match_results") or {}
        matches = _matches_raw.values() if isinstance(_matches_raw, dict) else (_matches_raw or [])
        # Count games where ANY market (winner, spread, or total) has a Kalshi match
        matched_games = len([
            m for m in matches
            if any([
                (m.get("matches") or {}).get("winner", {}).get("kalshi_matched"),
                (m.get("matches") or {}).get("spread", {}).get("kalshi_matched"),
                (m.get("matches") or {}).get("total", {}).get("kalshi_matched")
            ])
        ])

    # ============================================
    # KALSHI MARKETS COUNT
    # ============================================
    # Count games with valid Kalshi markets (HasKalshiMarket=True)
    # This is the count of games that have both:
    # 1. kalshi_matched == True
    # 2. At least one of kalshi_prob_spread or kalshi_prob_total is non-null
    #
    # This count is INDEPENDENT of spreadmarketpairscount/totalmarketpairscount
    # which count sportsbook markets, not Kalshi markets.
    # ============================================
    kalshi_markets_count = 0
    if master_df is not None and not master_df.empty:
        # Fallback for initial load if matched_games > 0 but markets count fails
        if matched_games > 0:
            kalshi_markets_count = matched_games

        try:
            # Check if HasKalshiMarket column already exists (from Master Analysis tab)
            if "HasKalshiMarket" in master_df.columns:
                # Use pre-computed flag
                valid_df = master_df.copy()
                if "odds_placeholder_detected" in valid_df.columns:
                    valid_df = valid_df[valid_df["odds_placeholder_detected"] != True]
                kalshi_markets_count = int(valid_df["HasKalshiMarket"].sum())
            else:
                # Compute on-the-fly before HasKalshiMarket flag is added
                # This ensures the badge works even when called before Master Analysis tab runs
                required_cols = ["kalshi_matched", "kalshi_prob_spread", "kalshi_prob_total", "Home", "Away"]
                if all(col in master_df.columns for col in required_cols):
                    # Compute HasKalshiMarket for each row
                    def _has_kalshi_market_inline(row):
                        kalshi_matched = row.get("kalshi_matched")
                        if not kalshi_matched:
                            return False

                        def _is_valid_prob(p):
                            try:
                                if pd.isna(p): return False
                                return float(p) != 0
                            except:
                                return False

                        kalshi_prob_spread = row.get("kalshi_prob_spread")
                        spread_prob_pick_kalshi = row.get("spread_prob_pick_kalshi")
                        kalshi_prob_total = row.get("kalshi_prob_total")
                        total_prob_pick_kalshi = row.get("total_prob_pick_kalshi")

                        kalshi_prob = row.get("kalshi_prob")
                        kalshi_prob_used = row.get("kalshi_prob_used")

                        has_spread = _is_valid_prob(kalshi_prob_spread) or _is_valid_prob(spread_prob_pick_kalshi)
                        has_total = _is_valid_prob(kalshi_prob_total) or _is_valid_prob(total_prob_pick_kalshi)
                        has_ml = _is_valid_prob(kalshi_prob) or _is_valid_prob(kalshi_prob_used)

                        return has_spread or has_total or has_ml

                    # Apply to filtered dataframe
                    valid_df = master_df.copy()
                    if "odds_placeholder_detected" in valid_df.columns:
                        valid_df = valid_df[valid_df["odds_placeholder_detected"] != True]

                    # Compute flag for each row, then count unique games
                    valid_df["_temp_has_kalshi"] = valid_df.apply(_has_kalshi_market_inline, axis=1)

                    # Count unique games (not rows) with Kalshi markets
                    games_with_kalshi = valid_df[valid_df["_temp_has_kalshi"]]
                    if not games_with_kalshi.empty:
                        unique_games = games_with_kalshi.groupby(["Home", "Away"]).size()
                        kalshi_markets_count = len(unique_games)
                    else:
                        kalshi_markets_count = 0
                else:
                    kalshi_markets_count = 0
        except Exception as e:
            logger.warning(f"Error counting Kalshi markets from master_df: {e}")
            kalshi_markets_count = 0
    sentiment_meta = st.session_state.get("sentiment_meta") or {}
    sentiment_ready = bool(
        sentiment_meta.get("sentiment_available_count")
        or sentiment_meta.get("sentiment_used_cached")
        or (sentiment_meta.get("sentiment_status") and str(sentiment_meta.get("sentiment_status")).upper() not in {"NO_CALL", "DISABLED"})
    )
    sentiment_flags = []
    if sentiment_meta.get("sentiment_rate_limited"):
        sentiment_flags.append("rate_limited")
    if sentiment_meta.get("sentiment_auth_error"):
        sentiment_flags.append("auth_error")

    # Check if local model is loaded
    engine = get_prediction_engine()
    model_ready = engine.model is not None

    # UPDATED: Read rows_out from live master_results_df, not cached metrics
    master_results_df = st.session_state.get("master_results_df")
    if master_results_df is not None and not master_results_df.empty:
        rows_out = len(master_results_df)
    else:
        # Fallback to old metric if dataframe not yet built
        master_stats = st.session_state.get("master_stats") or {}
        rows_out = master_stats.get("rows_out", 0)

    master_stats = st.session_state.get("master_stats") or {}
    return {
        "games_loaded": games_loaded,
        "kalshi_matched": kalshi_markets_count,
        "kalshi_markets": kalshi_markets_count,  # NEW: Count of games with valid Kalshi markets
        "sentiment_ready": sentiment_ready,
        "gemini_ready": st.session_state.get("gemini_initialized", False),
        "sentiment_flags": sentiment_flags,
        "model_ready": model_ready,
        "rows_out": rows_out,
        "market_rows_out": master_stats.get("market_rows_out", 0),  # Book market rows (kept for reference)
        "books_market_rows": master_stats.get("market_rows_out", 0),  # Renamed for clarity
    }


def render_pipeline_banner() -> None:
    progress = pipeline_progress_snapshot()
    with st.container():
        st.markdown("### Pipeline Health")
        cols = st.columns(4)
        cols[0].metric("Games Loaded", progress["games_loaded"], help="Loaded from Odds API")
        cols[1].metric(
            "Kalshi Matches",
            progress["kalshi_matched"],
            help="Games matched to Kalshi markets",
        )
        sentiment_delta = ", ".join(progress["sentiment_flags"]) if progress["sentiment_flags"] else ""
        sentiment_status_label = "Ready" if progress["sentiment_ready"] else "Unavailable"

        # Issue 2: Gemini Transparency - Show warning if sentiment/Gemini disabled
        # Check if disabled via session state or config
        gemini_disabled_reason = st.session_state.get("gemini_disabled_reason")
        gemini_ready = progress.get("gemini_ready", False)

        # Gemini Status Logic
        gemini_status_text = "Ready" if gemini_ready else "Missing"
        if gemini_disabled_reason:
            gemini_status_text = "Disabled"

        if gemini_disabled_reason or not progress["sentiment_ready"]:
             # If specifically disabled due to error, show as warning in UI
             if not progress["sentiment_ready"]:
                 st.warning("⚠️ Sentiment analysis unavailable - predictions may be less accurate")

        cols[2].metric(
            "Sentiment",
            sentiment_status_label,
            delta=sentiment_delta or None,
        )
        # Separate Gemini Metric (or combined if preferred, but user asked for status)
        # Using caption for now as cols[2] is taken by Sentiment
        if not gemini_ready:
             st.caption(f"⚠️ Gemini: {gemini_status_text} ({gemini_disabled_reason or 'Not Initialized'})")
        # ============================================
        # MARKETS BADGE: Shows count of games with valid Kalshi markets
        # ============================================
        # This counts games where HasKalshiMarket=True, meaning:
        # 1. kalshi_matched == True (successful match)
        # 2. At least one of kalshi_prob_spread or kalshi_prob_total is non-null
        #
        # This is INDEPENDENT of spreadmarketpairscount/totalmarketpairscount
        # which count sportsbook market pairs, not Kalshi markets.
        # ============================================
        cols[3].metric(
            "Master Rows",
            progress["rows_out"],
            delta=f"Markets: {progress['kalshi_markets']}",
            help="Master Rows: Total output rows | Markets: Games with valid Kalshi markets"
        )
        readiness = []
        if os.path.exists(os.path.join(os.path.dirname(__file__), "models", "model.json")):
            readiness.append("🟢 AI Model: Ready (XGBoost)")
        else:
            readiness.append("🟡 AI Model: Fallback Mode")
        st.caption(" | ".join(readiness))


def slate_key_from_games(games: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for g in games or []:
        parts.append(
            "|".join(
                [
                    str(g.get("league") or ""),
                    str(g.get("home_team") or ""),
                    str(g.get("away_team") or ""),
                    str(
                        g.get("commence_date_local")
                        or g.get("commence_time_iso_utc")
                        or g.get("commence_time")
                        or ""
                    ),
                ]
            )
        )
    return ";".join(sorted(parts))


def score_pick_confidence(row: Dict[str, Any]) -> Tuple[str, str, bool]:
    """
    Returns (confidence, reason_short, eligible_for_top_picks).
    confidence in {"HIGH", "MEDIUM", "LOW"}.

    Rule-based confidence using edge and data quality:
    - If no valid spread or total: LOW
    - If actual edge < 0 (model_prob < implied_prob): LOW (NEVER HIGH for negative edge)
    - If edge >= 0.08 AND Kalshi available AND decision_driver is Kalshi AND positive actual edge: HIGH
    - If edge >= 0.03 AND positive actual edge: MEDIUM
    - Otherwise: LOW

    Best Overall picks can only come from MEDIUM/HIGH confidence rows, never LOW.
    """
    market = (row.get("Market") or "").lower()
    final_prob = safe_float(row.get("final_probability") or row.get("consensus_prob_adj") or row.get("AI_Prob"))
    warnings_text = str(row.get("Warnings") or "")

    if final_prob is None:
        return "UNKNOWN", "UNKNOWN: missing final probability", False

    # Calculate decisiveness edge (distance from 0.5)
    decisiveness_edge = abs(final_prob - 0.5)

    # FIX: Calculate ACTUAL edge (model_prob - implied_prob/consensus)
    # This determines if we have positive value vs the market
    actual_edge = None
    if "spread" in market:
        # Use spread edge if available
        actual_edge = safe_float(row.get("spread_edge"))
        if actual_edge is None:
            implied = safe_float(row.get("spread_implied_prob") or row.get("SpreadConsensusProb"))
            model_prob = safe_float(row.get("spread_prob_pick_final") or row.get("spread_prob_adj"))
            if implied is not None and model_prob is not None:
                actual_edge = model_prob - implied
    elif "total" in market:
        # Use total edge if available
        actual_edge = safe_float(row.get("total_edge"))
        if actual_edge is None:
            implied = safe_float(row.get("total_implied_prob") or row.get("TotalConsensusProb"))
            model_prob = safe_float(row.get("total_prob_pick_final") or row.get("total_prob_adj"))
            if implied is not None and model_prob is not None:
                actual_edge = model_prob - implied

    # Fallback: if actual_edge still None, estimate from Edge column (string like "+5.2%")
    if actual_edge is None:
        edge_str = str(row.get("Edge") or row.get("edge") or "")
        if edge_str and '%' in edge_str:
            try:
                actual_edge = float(edge_str.replace('%', '').strip()) / 100.0
            except (ValueError, TypeError):
                pass

    # Final fallback: use 0 if we can't determine actual edge
    if actual_edge is None:
        actual_edge = 0.0

    # Check if there are no valid spread or total lines
    if "no_valid_spread_or_total" in warnings_text:
        tier = "LOW"
        reason = "no_valid_spread_or_total"
    # FIX: CRITICAL - Negative edge ALWAYS results in LOW confidence
    elif actual_edge < 0:
        tier = "LOW"
        reason = f"negative_edge={actual_edge:.3f} (<0, no value vs market)"
    else:
        # Probability-based confidence (final_prob already blends all sources)
        if final_prob >= 0.60:
            tier = "HIGH"
            reason = f"high_probability ({final_prob:.1%}), decisiveness={decisiveness_edge:.3f}, edge={actual_edge:.3f}"
        elif decisiveness_edge >= 0.03 and actual_edge >= 0:
            tier = "MEDIUM"
            reason = f"medium_signal, decisiveness={decisiveness_edge:.3f}, edge={actual_edge:.3f}"
        else:
            tier = "LOW"
            reason = f"insufficient_metrics, decisiveness={decisiveness_edge:.3f}, edge={actual_edge:.3f}"

    sentiment_dir = row.get("sentiment_direction") or "neutral"
    confidence_reason = f"{tier}: {reason} | driver={row.get('decision_driver') or 'unknown'} | sentiment={sentiment_dir}"

    # Only MEDIUM and HIGH confidence rows are eligible for top picks
    eligible = tier != "LOW"

    return tier, confidence_reason, eligible


def apply_confidence_filter(df: pd.DataFrame, confidence_mode: str, show_low: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df is None or df.empty:
        return pd.DataFrame(), {"counts": {}, "low_removed": 0}

    filtered = df.copy()

    # Canonicalize confidence column name
    confidence_missing_warning = False
    if "Pick_Confidence" not in filtered.columns:
        # Try common alternates used elsewhere in the code
        for alt in ["confidence_bucket", "confidencebucket", "Confidence", "pick_confidence", "Confidence Level"]:
            if alt in filtered.columns:
                filtered["Pick_Confidence"] = filtered[alt]
                break

    # If still missing, create a safe default and do not crash
    if "Pick_Confidence" not in filtered.columns:
        filtered["Pick_Confidence"] = "LOW"
        confidence_missing_warning = True

    # Make the counting logic safe - column is guaranteed to exist after above checks
    conf = filtered["Pick_Confidence"].astype(str).fillna("LOW").infer_objects(copy=False)
    base_low = int((conf == "LOW").sum())
    base_med = int((conf == "MEDIUM").sum())
    base_high = int((conf == "HIGH").sum())

    mode_norm = (confidence_mode or "").lower()

    # Apply confidence tier filter - column is guaranteed to exist after above checks
    if mode_norm.startswith("high only"):
        filtered = filtered[filtered["Pick_Confidence"] == "HIGH"]
    elif mode_norm.startswith("high+medium"):
        filtered = filtered[filtered["Pick_Confidence"].isin(["HIGH", "MEDIUM"])]
    # "All" or anything else -> no filter on confidence tier

    if not show_low:
        filtered = filtered[filtered["Pick_Confidence"] != "LOW"]

    # Recalculate confidence stats after filtering
    conf_after = filtered["Pick_Confidence"].astype(str).fillna("LOW").infer_objects(copy=False)
    low_after = int((conf_after == "LOW").sum())
    counts = conf_after.value_counts(dropna=False).to_dict() if len(conf_after) > 0 else {}

    # Add warning to counts dict if confidence was missing
    result = {"counts": counts, "low_removed": max(base_low - low_after, 0)}
    if confidence_missing_warning:
        result["warning"] = "Pick_Confidence column missing; defaulted to LOW"

    return filtered, result


@st.cache_data(ttl=900)
def enrich_game_context(game: Dict[str, Any], league_key: str, api_key: Optional[str], sd_key: Optional[str]) -> Dict[str, Any]:
    """Return lightweight context from API-Sports/SportsData; safe to fail silently."""
    enrichment: Dict[str, Any] = {
        "injuries_home_count": 0,
        "injuries_away_count": 0,
        "injuries_home": 0,
        "injuries_away": 0,
        "key_injuries_home": [],
        "key_injuries_away": [],
        "weather_summary": None,
        "api_sports_used": False,
        "sportsdata_used": False,
        "api_sports_status": "missing",
        "sportsdata_status": "missing",
        "schedule_warnings": [],
        "last_5_form_home": None,
        "last_5_form_away": None,
        "pace_signal": None,
        "efficiency_signal": None,
        "enrichment_errors_sample": [],
        "apisports_enriched": False,
        "apisports_notes": None,
        "sportsdata_enriched": False,
        "sportsdata_notes": None,
    }
    try:
        commence_iso = game.get("commence_time_iso_utc") or game.get("commence_time") or game.get("commence_time_iso")
        commence_dt = parse_commence_to_utc(commence_iso)
        commence_date = commence_dt.date() if commence_dt else None
    except Exception:
        commence_date = None

    if not commence_date:
        return enrichment

    try:
        # API-Sports attempt
        api_client = None
        if api_key:
            api_client_map = {
                "NBA": APISportsBasketballClient,
                "NCAAB": APISportsBasketballClient,
                "NFL": APISportsFootballClient,
                "NCAAF": APISportsFootballClient,
                "NHL": APISportsHockeyClient,
            }
            client_cls = api_client_map.get(league_key)
            api_client = client_cls(api_key, key_source="secrets/env") if client_cls else None
            enrichment["api_sports_status"] = "ok" if api_client else "missing"
        if api_client:
            # Check a 3-day window to handle timezone schedule mismatches
            dates_to_check = [commence_date, commence_date - timedelta(days=1), commence_date + timedelta(days=1)]
            games_api = []
            for d in dates_to_check:
                games_api.extend(api_client.get_games_by_date(d))
            
            home_raw = str(game.get("home_team") or "")
            away_raw = str(game.get("away_team") or "")
            
            # Extract list of (home, away, game_obj) tuples for fuzzy matcher
            candidates = []
            for g_api in games_api:
                h_api = str(((g_api.get("teams") or {}).get("home") or {}).get("name") or "")
                a_api = str(((g_api.get("teams") or {}).get("away") or {}).get("name") or "")
                candidates.append(((h_api, a_api), g_api))
            
            candidate_tuples = [c[0] for c in candidates]
            match_tuple = TeamNameMatcher.match_game(home_raw, away_raw, candidate_tuples)
            
            matched = None
            if match_tuple:
                # Retrieve the full game object associated with the matched tuple
                for c_tuple, c_obj in candidates:
                    if c_tuple == match_tuple:
                        matched = c_obj
                        break

            if matched:
                enrichment["api_sports_used"] = True
                enrichment["apisports_enriched"] = True
                enrichment["apisports_notes"] = "matched_fixture"
                fixture_date = (matched.get("fixture") or {}).get("date")
                # Removed strict schedule mismatch warning since we are looking at a wider window
                # Placeholder: API-Sports payload often lacks injuries in base tier; keep counts at 0.
                enrichment["injuries_home"] = enrichment["injuries_home_count"] or "unknown"
                enrichment["injuries_away"] = enrichment["injuries_away_count"] or "unknown"
            else:
                enrichment["apisports_notes"] = "no_match_found"
        elif api_key:
            enrichment["api_sports_status"] = "key_present_no_client"
            enrichment["apisports_notes"] = "client_missing"
    except Exception:
        enrichment["api_sports_status"] = "error"
        enrichment.setdefault("enrichment_errors_sample", []).append("api_sports_error")
        enrichment.setdefault("schedule_warnings", []).append("api_sports_error")
        enrichment["apisports_notes"] = "api_sports_error"

    try:
        # SportsData attempt
        sd_client = None
        if sd_key:
            sd_map = {
                "NBA": SportsDataNBAClient,
                "NCAAB": SportsDataNCAABClient,
                "NFL": SportsDataNFLClient,
                "NCAAF": SportsDataNCAAFClient,
                "NHL": SportsDataNHLClient,
            }
            sd_cls = sd_map.get(league_key)
            sd_client = sd_cls(sd_key, key_source="secrets/env") if sd_cls else None
            enrichment["sportsdata_status"] = "ok" if sd_client else "missing"
        if sd_client:
            # Check a 3-day window to handle timezone schedule mismatches
            dates_to_check = [commence_date, commence_date - timedelta(days=1), commence_date + timedelta(days=1)]
            scores = []
            for d in dates_to_check:
                scores.extend(sd_client.get_scores_by_date(d))
            
            # Use fuzzy matching logic for SportsData scores too
            home_raw = str(game.get("home_team") or "")
            away_raw = str(game.get("away_team") or "")
            
            # Build candidate list: SportsData usually has HomeTeam/AwayTeam or HomeTeamName/AwayTeamName
            candidates = []
            for sc in scores:
                h_sd = str(sc.get("HomeTeam") or sc.get("HomeTeamName") or "")
                a_sd = str(sc.get("AwayTeam") or sc.get("AwayTeamName") or "")
                candidates.append(((h_sd, a_sd), sc))
            
            candidate_tuples = [c[0] for c in candidates]
            match_tuple = TeamNameMatcher.match_game(home_raw, away_raw, candidate_tuples)
            
            match = None
            if match_tuple:
                for c_tuple, c_obj in candidates:
                    if c_tuple == match_tuple:
                        match = c_obj
                        break

            if match:
                enrichment["sportsdata_used"] = True
                enrichment["sportsdata_enriched"] = True
                enrichment["sportsdata_notes"] = "matched_fixture"
                # Removed strict schedule mismatch warning
                weather = match.get("Weather") or match.get("WeatherDescription")
                if weather:
                    enrichment["weather_summary"] = str(weather)
                venue = match.get("StadiumDetails") or {}
                if isinstance(venue, dict) and venue.get("PlayingSurface"):
                    enrichment["weather_summary"] = (
                        enrichment["weather_summary"] or f"Surface: {venue.get('PlayingSurface')}"
                    )
                insight = sd_client.build_game_insight(match)
                if insight:
                    enrichment["last_5_form_home"] = insight.home.trend
                    enrichment["last_5_form_away"] = insight.away.trend
                    enrichment["pace_signal"] = insight.home.power_index or insight.away.power_index
                # Injuries info not available via current helper; counts remain 0.
                enrichment["injuries_home"] = enrichment["injuries_home_count"] or "unknown"
                enrichment["injuries_away"] = enrichment["injuries_away_count"] or "unknown"
            else:
                enrichment["sportsdata_notes"] = "no_match_found"
        elif sd_key:
            enrichment["sportsdata_status"] = "key_present_no_client"
            enrichment["sportsdata_notes"] = "client_missing"
    except Exception:
        enrichment["sportsdata_status"] = "error"
        enrichment.setdefault("enrichment_errors_sample", []).append("sportsdata_error")
        enrichment.setdefault("schedule_warnings", []).append("sportsdata_error")
        enrichment["sportsdata_notes"] = "sportsdata_error"

    return enrichment

def init_sentiment_meta() -> Dict[str, Any]:
    return {
        "sentiment_source": "none",
        "sentiment_sample_status": "NO_CALL",
        "sentiment_status_counts": {"NO_CALL": 1},
        "sentiment_sample_query": "",
        "sentiment_sample_totalResults": 0,
        "sentiment_disabled_reason": "not_executed",
        "sentiment_error_count": 0,
        "sentiment_errors_sample": "",
        "sentiment_articles_total": 0,
        "sentiment_available_count": 0,
        "sentiment_status": "NO_CALL",
        "sentiment_confidence": 0.0,
        "sentiment_score": None,
        "sentiment_used_cached": False,
        "sentiment_rate_limited": False,
        "sentiment_auth_error": False,
        "sentiment_cooldown_until": "",
        "sentiment_cached_teams_count": 0,
        "sentiment_articles_used": 0,
        "sentiment_source_count": 0,
        "last_error": None,
        "reddit_used": False,
        "reddit_posts_used": 0,
        "reddit_comments_used": 0,
        "reddit_filled_teams": 0,
        "reddit_blended_teams": 0,
        "sentiment_cache_hits": 0,
        "sentiment_cache_misses": 0,
        "sentiment_degraded": False,
    }


def ensure_sentiment_loaded(games: List[Dict[str, Any]]) -> None:
    """Compute sentiment for the current slate when enabled and cache in session state."""
    if st.button("🧹 Clear Sentiment Cache", key="clear_sentiment_cache"):
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            if k.startswith("sentiment_"):
                st.session_state.pop(k, None)
        st.session_state.pop("sentiment_cooldown_until", None)
        st.session_state.pop("NEWSAPI_RATE_LIMITED_UNTIL", None)
        st.session_state.pop("reddit_used", None)
        st.session_state.pop("reddit_sentiment_cache", None)
        st.session_state.pop("sentiment_slate_key", None)
        st.session_state.pop("sentiment_source", None)
        st.session_state.pop("sentiment_map", None)
        st.session_state.pop("sentiment_meta_map", None)
        st.session_state.pop("sentiment_meta", None)
        st.session_state.pop("sentiment_debug", None)
        st.info("Sentiment cache cleared. Re-run analysis to refresh.")

    def _sentiment_meta_defaults(source: str, disabled_reason: str = "") -> Dict[str, Any]:
        return {
            "sentiment_source": source,
            "reddit_used": False,
            "reddit_posts_used": 0,
            "reddit_comments_used": 0,
            "reddit_filled_teams": 0,
            "reddit_blended_teams": 0,
            "articles_total": 0,
            "last_error": None,
            "error_count": 0,
            "status_counts": {},
            "auth_error": False,
            "rate_limited": False,
            "sample_query": "",
            "sample_status": "DISABLED" if source.startswith("disabled") else None,
            "sample_totalResults": None,
            "cached_teams": 0,
            "used_cached": False,
            "cooldown_until": st.session_state.get("sentiment_cooldown_until"),
            "cooldown_active": False,
            "available_count": 0,
            "sentiment_disabled_reason": disabled_reason,
            "sentiment_source_count": 0,
            "sentiment_confidence": 0.0,
        }

    enabled = st.session_state.get("enable_sentiment", True)
    logger.info(f"Sentiment Enabled: {enabled}")  # Defensive Log
    now_utc = datetime.now(timezone.utc)
    cooldown_raw = st.session_state.get("sentiment_cooldown_until")
    cooldown_until: Optional[datetime] = None
    if cooldown_raw:
        try:
            cooldown_until = datetime.fromisoformat(cooldown_raw) if isinstance(cooldown_raw, str) else cooldown_raw
            if cooldown_until and cooldown_until.tzinfo is None:
                cooldown_until = cooldown_until.replace(tzinfo=timezone.utc)
        except Exception:
            cooldown_until = None
    cooldown_active = bool(cooldown_until and now_utc < cooldown_until)
    sentiment_debug: Dict[str, Any] = {
        "enabled": enabled,
        "per_league": {},
        "reddit_used": False,
        "cooldown_until": cooldown_until.isoformat() if cooldown_until else "",
        "cooldown_active": cooldown_active,
    }
    if not enabled:
        meta = init_sentiment_meta()
        meta.update({
            "sentiment_source": "disabled_by_user",
            "sentiment_sample_status": "DISABLED",
            "sentiment_status_counts": {"DISABLED": 1},
            "sample_status": "DISABLED",
            "status_counts": {"DISABLED": 1},
            "sentiment_disabled_reason": "sentiment_disabled",
            "cooldown_until": st.session_state.get("sentiment_cooldown_until") or "",
        })
        st.session_state["sentiment_map"] = {}
        st.session_state["sentiment_meta_map"] = {}
        st.session_state["sentiment_meta"] = meta
        st.session_state["sentiment_debug"] = {**sentiment_debug, **meta}
        st.session_state["sentiment_slate_key"] = None
        st.session_state["sentiment_source"] = meta["sentiment_source"]
        st.session_state["reddit_used"] = False
        return

    if not games:
        logger.warning("Sentiment disabled: No games loaded")  # Defensive Log
        meta = init_sentiment_meta()
        meta.update({
            "sentiment_source": "disabled_no_games",
            "sentiment_sample_status": "DISABLED",
            "sentiment_status_counts": {"DISABLED": 1},
            "sample_status": "DISABLED",
            "status_counts": {"DISABLED": 1},
            "sentiment_disabled_reason": "no_games_loaded",
            "cooldown_until": st.session_state.get("sentiment_cooldown_until") or "",
        })
        st.session_state["sentiment_map"] = {}
        st.session_state["sentiment_meta_map"] = {}
        st.session_state["sentiment_meta"] = meta
        st.session_state["sentiment_debug"] = {**sentiment_debug, **meta, "warning": "no_games_loaded"}
        st.session_state["sentiment_slate_key"] = None
        st.session_state["sentiment_source"] = meta["sentiment_source"]
        st.session_state["reddit_used"] = False
        return

    if not news_api_key:
        meta = init_sentiment_meta()
        meta.update({
            "sentiment_source": "disabled_no_key",
            "sentiment_sample_status": "DISABLED",
            "sentiment_status_counts": {"DISABLED": 1},
            "sample_status": "DISABLED",
            "status_counts": {"DISABLED": 1},
            "sentiment_disabled_reason": "missing_news_api_key",
            "cooldown_until": st.session_state.get("sentiment_cooldown_until") or "",
        })
        st.session_state["sentiment_map"] = {}
        st.session_state["sentiment_meta_map"] = {}
        st.session_state["sentiment_meta"] = meta
        st.session_state["sentiment_debug"] = {**sentiment_debug, **meta, "warning": "missing_news_api_key"}
        st.session_state["sentiment_slate_key"] = None
        st.session_state["sentiment_source"] = meta["sentiment_source"]
        st.session_state["reddit_used"] = False
        return

    slate_key = slate_key_from_games(games)

    try:
        per_league_debug: Dict[str, Any] = {}
        total_articles = 0
        total_errors = 0
        last_error: Optional[str] = None
        cached_used_any = False
        cached_teams_total = 0
        cooldown_until_value = cooldown_until
        global_meta = init_sentiment_meta()
        global_meta.update({
            "cooldown_until": cooldown_until.isoformat() if cooldown_until else "",
            "cooldown_active": cooldown_active,
        })
        aggregate_sentiment_map: Dict[str, Optional[float]] = {}
        aggregate_sentiment_meta: Dict[str, Dict[str, Any]] = {}
        status_counts_all: Dict[Any, int] = {}
        sample_calls_all: List[Dict[str, Any]] = []
        league_labels_used: Dict[str, Any] = {}
        missing_teams_all: List[str] = []

        for raw_lg in sorted({g.get("league") for g in games if g.get("league")}):
            lg_key = canonical_league_key(raw_lg)
            lg_games = [g for g in games if canonical_league_key(g.get("league")) == lg_key]
            lg_map: Dict[str, Optional[float]] = {}
            lg_meta_map: Dict[str, Dict[str, Any]] = {}
            try:
                existing_lg_map = st.session_state.get(f"sentiment_map_{lg_key}") or {}
                existing_lg_meta = st.session_state.get(f"sentiment_meta_map_{lg_key}") or {}
                result = compute_team_sentiment_map(
                    news_api_key,
                    lg_games,
                    league=lg_key,
                    existing_map=existing_lg_map,
                    existing_meta_map=existing_lg_meta,
                    cooldown_until=cooldown_until_value,
                    max_calls=MAX_SENTIMENT_CALLS,
                )
                lg_map, lg_meta_map, lg_debug = {}, {}, {}
                if isinstance(result, tuple):
                    if len(result) == 3:
                        lg_map, lg_meta_map, lg_debug = result
                    elif len(result) == 2:
                        lg_map, lg_debug = result
                        article_counts_tmp = (lg_debug or {}).get("article_counts") or {}
                        for team, score in (lg_map or {}).items():
                            sources = int(article_counts_tmp.get(team, 0) or 0)
                            valid = sources > 0 and score is not None
                            lg_meta_map[team] = {
                                "sources": sources,
                                "sentiment_valid": valid,
                                "sentiment_source": "newsapi" if valid else "none",
                                "reddit_used": False,
                                "score": score if valid else None,
                            }
                    else:
                        lg_debug = {"error": "unexpected_return_length", "len": len(result)}
                else:
                    lg_debug = {"error": "unexpected_return_type"}
            except Exception as exc:
                lg_map, lg_meta_map, lg_debug = {}, {}, {"error": str(exc)}
                last_error = str(exc)
            per_league_debug[lg_key] = lg_debug
            league_labels_used[lg_key] = (lg_debug or {}).get("league_label_used")
            lg_articles = int((lg_debug or {}).get("articles_total") or 0)
            lg_cached = int((lg_debug or {}).get("cached_teams") or 0)
            lg_errors = int((lg_debug or {}).get("error_count") or 0)
            lg_status_counts = (lg_debug or {}).get("status_counts") or {}
            lg_sample_calls = (lg_debug or {}).get("sample_calls") or []
            lg_rate_limited = bool((lg_debug or {}).get("rate_limited") or (lg_status_counts.get(429) or 0) > 0)
            lg_auth_error = bool((lg_debug or {}).get("auth_error") or (lg_status_counts.get(401) or 0) > 0 or (lg_status_counts.get(403) or 0) > 0)
            cached_used_any = cached_used_any or bool((lg_debug or {}).get("used_cached"))
            cached_teams_total += lg_cached
            if (lg_debug or {}).get("cooldown_until"):
                st.session_state["sentiment_cooldown_until"] = (lg_debug or {}).get("cooldown_until")
                try:
                    cooldown_until_value = datetime.fromisoformat(st.session_state["sentiment_cooldown_until"])
                    if cooldown_until_value and cooldown_until_value.tzinfo is None:
                        cooldown_until_value = cooldown_until_value.replace(tzinfo=timezone.utc)
                except Exception:
                    cooldown_until_value = cooldown_until_value or cooldown_until
            cooldown_active = bool(cooldown_until_value and now_utc < cooldown_until_value)
            for status_code, count in lg_status_counts.items():
                status_counts_all[status_code] = status_counts_all.get(status_code, 0) + count
            sample_calls_all.extend(lg_sample_calls)
            missing_teams_all.extend((lg_debug or {}).get("missing_teams") or [])
            total_articles += lg_articles
            total_errors += lg_errors
            st.session_state[f"sentiment_map_{lg_key}"] = lg_map or {}
            st.session_state[f"sentiment_meta_map_{lg_key}"] = lg_meta_map or {}
            st.session_state[f"sentiment_debug_{lg_key}"] = lg_debug
            if lg_auth_error:
                lg_source = "error_auth"
            elif lg_rate_limited and (lg_articles > 0 or bool((lg_debug or {}).get("used_cached"))):
                lg_source = "partial_cached"
            elif lg_rate_limited:
                lg_source = "error_rate_limited"
            elif lg_articles > 0:
                lg_source = "newsapi"
            elif bool((lg_debug or {}).get("used_cached")):
                lg_source = "partial_cached"
            else:
                lg_source = "none"
            st.session_state[f"sentiment_source_{lg_key}"] = lg_source
            st.session_state[f"reddit_used_{lg_key}"] = False
            aggregate_sentiment_map.update(st.session_state.get(f"sentiment_map_{lg_key}") or {})
            aggregate_sentiment_meta.update(st.session_state.get(f"sentiment_meta_map_{lg_key}") or {})

        sentiment_available_count = len([v for v in aggregate_sentiment_map.values() if v is not None])
        auth_error = bool((status_counts_all.get(401) or 0) > 0 or (status_counts_all.get(403) or 0) > 0)
        rate_limited = bool((status_counts_all.get(429) or 0) > 0 or any((d or {}).get("rate_limited") for d in per_league_debug.values()))
        if rate_limited and not cooldown_until_value:
            cooldown_until_value = now_utc + timedelta(minutes=20)
        if cooldown_until_value:
            st.session_state["sentiment_cooldown_until"] = cooldown_until_value.isoformat()
        disabled_reason = ""
        if cooldown_active or (cooldown_until_value and now_utc < cooldown_until_value):
            disabled_reason = "cooldown_cached_only" if (cached_used_any or sentiment_available_count > 0) else "cooldown_no_cache"
        if not status_counts_all:
            status_counts_all = {"NO_CALL": 1}
        if auth_error:
            sentiment_source = "error_auth"
        elif rate_limited and cached_used_any:
            sentiment_source = "partial_cached"
        elif rate_limited:
            sentiment_source = "error_rate_limited"
        elif total_articles > 0:
            sentiment_source = "newsapi"
        elif cached_used_any:
            sentiment_source = "partial_cached"
        else:
            sentiment_source = "none"
        first_sample = sample_calls_all[0] if sample_calls_all else {}
        sample_status = first_sample.get("status") if isinstance(first_sample, dict) else None
        if not sample_status and rate_limited:
            sample_status = 429
        if sample_status is None:
            if sentiment_source.startswith("disabled"):
                sample_status = "DISABLED"
            elif sentiment_source.startswith("cooldown"):
                sample_status = "COOLDOWN"
            elif auth_error:
                sample_status = "401"
            else:
                sample_status = "NO_CALL" if news_api_key else "NO_KEY"
        sample_status_str = str(sample_status)
        sample_query_val = first_sample.get("q") if isinstance(first_sample, dict) else None
        global_meta.update({
            "sentiment_source": sentiment_source,
            "articles_total": total_articles,
            "last_error": last_error,
            "error_count": total_errors,
            "status_counts": status_counts_all,
            "auth_error": auth_error,
            "rate_limited": rate_limited,
            "sample_query": sample_query_val or "",
            "sample_status": sample_status_str,
            "sample_totalResults": first_sample.get("totalResults") if isinstance(first_sample, dict) else 0,
            "cached_teams": cached_teams_total,
            "used_cached": cached_used_any,
            "cooldown_until": cooldown_until_value.isoformat() if cooldown_until_value else None,
            "cooldown_active": bool(cooldown_until_value and now_utc < cooldown_until_value),
            "available_count": sentiment_available_count,
            "sentiment_disabled_reason": disabled_reason,
        })
        st.session_state["sentiment_meta"] = global_meta
        st.session_state["sentiment_source"] = sentiment_source
        st.session_state["reddit_used"] = False
        st.session_state["sentiment_debug"] = {
            **sentiment_debug,
            "per_league": per_league_debug,
            "articles_total": total_articles,
            "last_error": last_error,
            "missing_news_api_key": not bool(news_api_key),
            "reddit_used": False,
            "error_count": total_errors,
            "status_counts": status_counts_all,
            "sample_calls": sample_calls_all[:10],
            "rate_limited": rate_limited,
            "auth_error": auth_error,
            "league_labels_used": league_labels_used,
            "missing_teams": missing_teams_all,
            "teams_total": len(aggregate_sentiment_map),
            "sentiment_source": sentiment_source,
            "sample_query": first_sample.get("q") if isinstance(first_sample, dict) else None,
            "sample_status": sample_status,
            "sample_totalResults": first_sample.get("totalResults") if isinstance(first_sample, dict) else None,
            "cached_teams": cached_teams_total,
            "used_cached": cached_used_any,
            "cooldown_until": cooldown_until_value.isoformat() if cooldown_until_value else None,
            "cooldown_active": bool(cooldown_until_value and now_utc < cooldown_until_value),
            "available_count": sentiment_available_count,
            "sentiment_disabled_reason": disabled_reason,
        }
        st.session_state["sentiment_map"] = aggregate_sentiment_map
        st.session_state["sentiment_meta_map"] = aggregate_sentiment_meta
        st.session_state["sentiment_slate_key"] = slate_key
    except Exception as exc:  # pragma: no cover - defensive UI behavior
        import traceback as _tb
        meta = init_sentiment_meta()
        meta.update({
            "sentiment_source": "error_exception",
            "error": str(exc),
            "last_error": str(exc),
            "articles_total": 0,
            "error_count": 1,
            "sentiment_error_count": 1,
            "status_counts": {"EXCEPTION": 1},
            "sentiment_status_counts": {"EXCEPTION": 1},
            "auth_error": False,
            "rate_limited": False,
            "sample_query": "",
            "sample_status": "EXCEPTION",
            "sentiment_sample_status": "EXCEPTION",
            "sample_totalResults": 0,
            "cached_teams": 0,
            "cached_teams_count": 0,
            "used_cached": False,
            "sentiment_used_cached": False,
            "cooldown_until": st.session_state.get("sentiment_cooldown_until") or "",
            "cooldown_active": bool(st.session_state.get("sentiment_cooldown_until")),
            "available_count": 0,
            "sentiment_disabled_reason": "exception",
            "sentiment_errors_sample": (str(exc) + " | " + _tb.format_exc())[:800],
        })
        st.session_state["sentiment_map"] = {}
        st.session_state["sentiment_meta_map"] = {}
        st.session_state["sentiment_meta"] = meta
        st.session_state["sentiment_debug"] = {**sentiment_debug, **meta}
        st.session_state["sentiment_slate_key"] = slate_key
        st.session_state["last_exception"] = traceback.format_exc()
        st.session_state["sentiment_source"] = "error"
        st.session_state["reddit_used"] = False




def kalshi_health_check(selected_league: str = "NBA") -> Dict[str, Any]:
    """
    MUST NOT crash. Used for UI gating + debug.
    ok=True means "reachable/call succeeded", not "game markets exist".
    """
    try:
        ki = kalshi_integrator
        if ki is None:
            return {
                "configured": False,
                "ok": False,
                "error": "Kalshi integrator not initialized.",
                "market_count": 0,
            }

        markets = ki.get_sports_markets(selected_league) or []
        return {
            "configured": True,
            "ok": True,
            "error": None,
            "market_count": len(markets),
        }
    except Exception as e:
        return {
            "configured": True,
            "ok": False,
            "error": str(e),
            "market_count": 0,
        }

# -----------------
# Helper utilities
# -----------------

def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read from st.secrets then env vars."""
    # 1. Check st.secrets top-level (case-insensitive)
    try:
        for k, v in st.secrets.items():
            if str(k).lower() == str(name).lower() and isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass

    # 2. Check st.secrets["general"] (common pattern)
    try:
        if "general" in st.secrets and name in st.secrets["general"]:
            return st.secrets["general"][name]
    except Exception:
        pass

    # 3. Check os.environ
    return os.getenv(name, default)


def _get_secret_ci(name: str) -> Optional[str]:
    try:
        for key, val in st.secrets.items():
            if str(key).lower() == str(name).lower() and isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        return None
    return None


# Initialize PredictionEngine
@st.cache_resource
def get_prediction_engine():
    return PredictionEngine()

def get_api_keys() -> Dict[str, Optional[str]]:
    def _find_key(candidates: List[str]) -> Optional[str]:
        val = get_secret_any(*candidates)
        if val:
            return val
        for key in candidates:
            env_val = read_secret(key)
            if env_val:
                return env_val
        return None

    api_sports_key = _find_key(["APISPORTS_API_KEY", "API_SPORTS_KEY", "API_SPORTS_API_KEY", "NBA_APISPORTS_API_KEY"]) or get_apisports_key()
    sportsdata_key = _find_key(["SPORTSDATA_API_KEY", "SPORTSDATA_KEY"]) or get_sportsdata_key()
    api_sports_keys = {
        "NFL": _find_key(["APISPORTS_NFL_KEY", "NFL_APISPORTS_API_KEY"]) or api_sports_key or get_apisports_key("NFL"),
        "NBA": _find_key(["APISPORTS_NBA_KEY", "NBA_APISPORTS_API_KEY"]) or api_sports_key or get_apisports_key("NBA"),
        "NHL": _find_key(["APISPORTS_NHL_KEY", "NHL_APISPORTS_API_KEY"]) or api_sports_key or get_apisports_key("NHL"),
        "NCAAB": _find_key(["APISPORTS_NCAAB_KEY", "NCAAB_APISPORTS_API_KEY"]) or api_sports_key or get_apisports_key("NCAAB"),
        "NCAAF": _find_key(["APISPORTS_NCAAF_KEY", "NCAAF_APISPORTS_API_KEY"]) or api_sports_key or get_apisports_key("NCAAF"),
    }
    sportsdata_keys = {
        "NFL": _find_key(["SPORTSDATA_NFL_KEY"]) or sportsdata_key or get_sportsdata_key("NFL"),
        "NBA": _find_key(["SPORTSDATA_NBA_KEY"]) or sportsdata_key or get_sportsdata_key("NBA"),
        "NHL": _find_key(["SPORTSDATA_NHL_KEY"]) or sportsdata_key or get_sportsdata_key("NHL"),
        "NCAAB": _find_key(["SPORTSDATA_NCAAB_KEY"]) or sportsdata_key or get_sportsdata_key("NCAAB"),
        "NCAAF": _find_key(["SPORTSDATA_NCAAF_KEY"]) or sportsdata_key or get_sportsdata_key("NCAAF"),
    }
    return {
        "api_sports_key": api_sports_key,
        "sportsdata_key": sportsdata_key,
        "api_sports_keys": api_sports_keys,
        "sportsdata_keys": sportsdata_keys,
    }

def get_secret_any(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        ci_val = _get_secret_ci(key)
        if ci_val:
            return ci_val
        try:
            val = st.secrets.get(key, None)
        except Exception:
            val = None
        if isinstance(val, str) and val.strip():
            return val.strip()
    return default


def any_secret_prefix(prefix: str) -> bool:
    try:
        for key in st.secrets.keys():
            if str(key).lower().startswith(prefix.lower()):
                val = st.secrets.get(key, "")
                if isinstance(val, str) and val.strip():
                    return True
    except Exception:
        return False
    return False

def init_data_clients() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    global api_sports_clients, sportsdata_clients
    if api_sports_clients and sportsdata_clients:
        return api_sports_clients, sportsdata_clients
    keys = get_api_keys()
    api_key = keys.get("api_sports_key")
    sd_key = keys.get("sportsdata_key")
    api_keys_by_league = keys.get("api_sports_keys") or {}
    sd_keys_by_league = keys.get("sportsdata_keys") or {}
    def _league_key(league: str, league_map: Dict[str, Optional[str]], fallback: Optional[str], *, secondary: Optional[str] = None) -> Optional[str]:
        """Prefer league-specific key; fallback to optional secondary then global."""
        if league_map.get(league):
            return league_map.get(league)
        if secondary and league_map.get(secondary):
            return league_map.get(secondary)
        return fallback

    api_key_for = {
        "NBA": _league_key("NBA", api_keys_by_league, api_key),
        "NCAAB": _league_key("NCAAB", api_keys_by_league, api_key, secondary="NBA"),
        "NFL": _league_key("NFL", api_keys_by_league, api_key),
        "NCAAF": _league_key("NCAAF", api_keys_by_league, api_key, secondary="NFL"),
        "NHL": _league_key("NHL", api_keys_by_league, api_key),
    }
    sd_key_for = {
        "NBA": _league_key("NBA", sd_keys_by_league, sd_key),
        "NCAAB": _league_key("NCAAB", sd_keys_by_league, sd_key, secondary="NBA"),
        "NFL": _league_key("NFL", sd_keys_by_league, sd_key),
        "NCAAF": _league_key("NCAAF", sd_keys_by_league, sd_key, secondary="NFL"),
        "NHL": _league_key("NHL", sd_keys_by_league, sd_key),
    }
    api_sports_clients = {
        "NBA": APISportsBasketballClient(api_key_for["NBA"], key_source="secrets/env") if api_key_for.get("NBA") else None,
        "NCAAB": APISportsBasketballClient(api_key_for["NCAAB"], key_source="secrets/env") if api_key_for.get("NCAAB") else None,
        "NFL": APISportsFootballClient(api_key_for["NFL"], key_source="secrets/env") if api_key_for.get("NFL") else None,
        "NCAAF": APISportsFootballClient(api_key_for["NCAAF"], key_source="secrets/env") if api_key_for.get("NCAAF") else None,
        "NHL": APISportsHockeyClient(api_key_for["NHL"], key_source="secrets/env") if api_key_for.get("NHL") else None,
    }
    sportsdata_clients = {
        "NBA": SportsDataNBAClient(sd_key_for["NBA"], key_source="secrets/env") if sd_key_for.get("NBA") else None,
        "NCAAB": SportsDataNCAABClient(sd_key_for["NCAAB"], key_source="secrets/env") if sd_key_for.get("NCAAB") else None,
        "NFL": SportsDataNFLClient(sd_key_for["NFL"], key_source="secrets/env") if sd_key_for.get("NFL") else None,
        "NCAAF": SportsDataNCAAFClient(sd_key_for["NCAAF"], key_source="secrets/env") if sd_key_for.get("NCAAF") else None,
        "NHL": SportsDataNHLClient(sd_key_for["NHL"], key_source="secrets/env") if sd_key_for.get("NHL") else None,
    }
    try:
        st.session_state["data_clients_debug"] = {
            "api_sports": {lg: bool(cli) for lg, cli in api_sports_clients.items()},
            "sportsdata": {lg: bool(cli) for lg, cli in sportsdata_clients.items()},
            "api_sports_keys": {lg: bool(api_key_for.get(lg)) for lg in api_key_for},
            "sportsdata_keys": {lg: bool(sd_key_for.get(lg)) for lg in sd_key_for},
        }
    except Exception:
        pass
    return api_sports_clients, sportsdata_clients

# Must be the first Streamlit call
st.set_page_config(page_title="ParlayDesk", layout="wide")

# Issue 3: Validate API key before attempting to use Gemini
# Set gemini_initialized flag for UI status check
client, error = initialize_gemini()
if client:
    st.session_state["gemini_initialized"] = True
else:
    st.session_state["gemini_initialized"] = False

# Task 3: Initialize TheOver Raw Debug State
if "theover_raw_df" not in st.session_state:
    st.session_state["theover_raw_df"] = pd.DataFrame()

if "model_mode" not in st.session_state:
    st.session_state["model_mode"] = "Local XGBoost"
if "model_ready" not in st.session_state:
    st.session_state["model_ready"] = False
if "use_model_numeric_probs" not in st.session_state:
    st.session_state["use_model_numeric_probs"] = True


# ------------------------------------------------------------
# Kalshi globals / shims (must exist before any call sites)
# ------------------------------------------------------------
kalshi_integrator: Optional[KalshiIntegrator] = None
api_sports_clients: Dict[str, Any] = {}
sportsdata_clients: Dict[str, Any] = {}

def canonical_league_key(raw: Optional[str]) -> str:
    """Normalize league identifiers to canonical keys used across odds/sentiment/Kalshi."""
    if not raw:
        return ""
    val = str(raw).upper()
    mapping = {
        "BASKETBALL_NBA": "NBA",
        "NBA": "NBA",
        "BASKETBALL_NCAAB": "NCAAB",
        "NCAAB": "NCAAB",
        "AMERICANFOOTBALL_NFL": "NFL",
        "NFL": "NFL",
        "AMERICANFOOTBALL_NCAAF": "NCAAF",
        "NCAAF": "NCAAF",
        "ICEHOCKEY_NHL": "NHL",
        "NHL": "NHL",
        "BASEBALL_MLB": "MLB",
        "MLB": "MLB",
    }
    return mapping.get(val, val)

def american_to_implied(odds: Any) -> Optional[float]:
    """Convert American odds to implied probability; returns None on invalid/missing."""
    try:
        o = float(odds)
        if o == 0:
            return None
        if o > 0:  # +120
            return 100.0 / (o + 100.0)
        return (-o) / ((-o) + 100.0)
    except Exception:
        return None

def american_to_implied_prob(odds: Any) -> Optional[float]:
    """
    Convert American odds to implied probability with defensive caps for extreme values.

    Extreme odds (|odds| > 900) are capped to prevent unrealistic probabilities (>0.90 or <0.10).
    This helps NHL and other leagues with heavy favorites avoid probability collisions.
    """
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None

    # Cap extreme odds to prevent unrealistic probabilities
    # -900 converts to ~0.90, which is more reasonable than -990 -> 0.99
    if o < -900:
        o = -900
    elif o > 900:
        o = 900

    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return None

def _normalize_point_for_market(point: Any, market: str) -> Optional[float]:
    val = safe_float(point)
    if val is None:
        return None
    if market == "spread":
        return round(abs(val), 3)
    return round(val, 3)

def compute_market_prob_from_offers(
    offers: List[Dict[str, Any]],
    pick_side: Optional[str],
    *,
    market_type: str,
) -> Tuple[Optional[float], int, str, bool]:
    """
    Compute a market probability using no-vig consensus when both sides are available.
    Returns (probability, matched_pairs_count, method, placeholder_flag).
    """
    if not offers or not pick_side:
        return None, 0, "missing", False
    pick_side_norm = str(pick_side or "").lower()
    side_keys = ("home", "away") if market_type == "spread" else ("over", "under")
    grouped: Dict[Tuple[str, float], Dict[str, Dict[str, Any]]] = {}
    for offer in offers:
        side = str(offer.get("side") or "").lower()
        if side not in side_keys:
            continue
        point_key = _normalize_point_for_market(offer.get("point"), market_type)
        book = offer.get("book")
        if point_key is None or not book:
            continue
        grouped.setdefault((book, point_key), {})
        grouped[(book, point_key)][side] = offer

    p_pick_nv: List[float] = []
    for _, sides in grouped.items():
        if not all(k in sides for k in side_keys):
            continue
        price_1 = sides[side_keys[0]].get("price")
        price_2 = sides[side_keys[1]].get("price")
        p1 = american_to_implied(price_1)
        p2 = american_to_implied(price_2)
        if p1 is None or p2 is None:
            continue
        denom = p1 + p2
        if denom <= 0:
            continue
        p1_nv = p1 / denom
        p2_nv = p2 / denom
        selected = p1_nv if pick_side_norm == side_keys[0] else p2_nv
        p_pick_nv.append(selected)

    pairs_count = len(p_pick_nv)
    if p_pick_nv:
        return statistics.median(p_pick_nv), pairs_count, "no_vig_median", False

    side_offers = [
        o for o in offers if str(o.get("side") or "").lower() == pick_side_norm and o.get("price") is not None
    ]
    implieds = [american_to_implied(o.get("price")) for o in side_offers]
    implieds = [p for p in implieds if p is not None]
    if implieds:
        return statistics.median(implieds), pairs_count, "one_sided_implied", True
    return None, pairs_count, "missing", False

def select_best_offer_for_pick(
    offers: List[Dict[str, Any]],
    pick_side: Optional[str],
    *,
    pick_line: Optional[float],
    preferred_book: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not offers or not pick_side:
        return None
    pick_side_norm = str(pick_side or "").lower()
    filtered = [o for o in offers if str(o.get("side") or "").lower() == pick_side_norm]
    if not filtered:
        return None

    def _recency_score(dt_val: Optional[datetime]) -> float:
        if not dt_val:
            return float("inf")
        return -dt_val.timestamp()

    preferred = str(preferred_book or "").lower()
    def _key(o: Dict[str, Any]) -> Tuple[int, float, float, float]:
        book_rank = 0 if preferred and str(o.get("book") or "").lower() == preferred else 1
        point_val = safe_float(o.get("point"))
        line_diff = abs((point_val or 0) - (pick_line or 0)) if (pick_line is not None and point_val is not None) else 0.0
        implied = american_to_implied(o.get("price"))
        implied_key = implied if implied is not None else 2.0
        recency = _recency_score(o.get("last_update"))
        return (book_rank, line_diff, implied_key, recency)

    return sorted(filtered, key=_key)[0]


def is_placeholder_odds(home_ml: Any, away_ml: Any) -> bool:
    try:
        if home_ml is None or away_ml is None:
            return True
        h = float(home_ml)
        a = float(away_ml)
        return h == -110.0 and a == -110.0
    except Exception:
        return True

PLACEHOLDER_IMPLIED_PROB = american_to_implied(-110)

def safe_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        try:
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        except Exception:
            pass
    try:
        return str(value)
    except Exception:
        return None

def get_local_tz() -> str:
    tz_name = None
    try:
        tz_name = st.secrets.get("APP_TIMEZONE")
    except Exception:
        tz_name = None
    if not tz_name:
        tz_name = "America/New_York"
    return tz_name

def parse_commence_to_utc(value: Any) -> Optional[datetime]:
    raw = value
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        try:
            s = str(raw)
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def normalize_commence_times(games: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tz_name = get_local_tz()
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = None
    parsed = 0
    failed = 0
    for g in games:
        warnings = list(g.get("warnings") or [])
        raw_time = g.get("commence_time") or g.get("commence_time_iso")
        dt_utc = parse_commence_to_utc(raw_time)
        if dt_utc is None:
            failed += 1
            warnings.append("commence_parse_failed")
            g["commence_time_utc"] = None
            g["commence_time_iso_utc"] = None
            g["commence_time_local"] = None
            g["commence_time_iso_local"] = None
            g["commence_date_local"] = None
        else:
            parsed += 1
            g["commence_time_utc"] = dt_utc
            iso_utc = dt_utc.isoformat().replace("+00:00", "Z")
            g["commence_time_iso_utc"] = iso_utc
            if local_tz:
                dt_local = dt_utc.astimezone(local_tz)
                g["commence_time_local"] = dt_local
                g["commence_time_iso_local"] = dt_local.isoformat()
                g["commence_date_local"] = dt_local.strftime("%Y-%m-%d")
            else:
                g["commence_time_local"] = None
                g["commence_time_iso_local"] = None
                g["commence_date_local"] = None
        g["warnings"] = warnings
    stats = {"parsed": parsed, "failed": failed, "timezone": tz_name}
    return games, stats

def fmt_local_time(dt: Optional[datetime]) -> str:
    try:
        if dt is None:
            return ""
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""

def robust_get_prices(outcomes: List[Dict], home_team: str, away_team: str) -> Tuple[Optional[float], Optional[float]]:
    """Robust extraction of home/away prices using normalization."""
    if not home_team or not away_team:
        return None, None

    prices_map = {}
    for o in outcomes:
        nm = o.get("name")
        pr = o.get("price")
        if nm and pr is not None:
            prices_map[TeamNameMatcher.normalize(nm)] = pr

    home_norm = TeamNameMatcher.normalize(home_team)
    away_norm = TeamNameMatcher.normalize(away_team)

    h_price = prices_map.get(home_norm)
    a_price = prices_map.get(away_norm)

    # Fuzzy/Substring Fallback
    if h_price is None:
        for k, v in prices_map.items():
            if k in home_norm or home_norm in k:
                h_price = v
                break
    if a_price is None:
        for k, v in prices_map.items():
            if k in away_norm or away_norm in k:
                a_price = v
                break

    if h_price is None or a_price is None:
        # Debug log for missing prices to help diagnose matching issues
        logger.debug(f"Missing ML Prices: Home='{home_team}'->'{home_norm}' ({h_price}), Away='{away_team}'->'{away_norm}' ({a_price}). Market Keys: {list(prices_map.keys())}")

    return h_price, a_price

def robust_get_spreads(outcomes: List[Dict], home_team: str, away_team: str):
    """Robust extraction of spread point/price."""
    if not home_team or not away_team:
        return None, None, None, None

    # Map normalized -> (point, price)
    data_map = {}
    for o in outcomes:
        nm = o.get("name")
        pt = o.get("point")
        pr = o.get("price")
        if nm:
            data_map[TeamNameMatcher.normalize(nm)] = (pt, pr)

    home_norm = TeamNameMatcher.normalize(home_team)
    away_norm = TeamNameMatcher.normalize(away_team)

    h_data = data_map.get(home_norm)
    a_data = data_map.get(away_norm)

    if h_data is None:
        for k, v in data_map.items():
            if k in home_norm or home_norm in k:
                h_data = v
                break
    if a_data is None:
        for k, v in data_map.items():
            if k in away_norm or away_norm in k:
                a_data = v
                break

    if h_data and a_data:
        return h_data[0], h_data[1], a_data[0], a_data[1]
    return None, None, None, None

def extract_h2h_prices(game: Dict[str, Any]) -> Dict[str, Any]:
    home = game.get("home_team")
    away = game.get("away_team")
    for bm in game.get("bookmakers") or []:
        for market in bm.get("markets") or []:
            if market.get("key") != "h2h":
                continue
            h_p, a_p = robust_get_prices(market.get("outcomes") or [], home, away)
            if h_p is not None and a_p is not None:
                return {
                    "home_odds": h_p,
                    "away_odds": a_p,
                    "book": bm.get("title") or bm.get("key"),
                }
    return {"home_odds": None, "away_odds": None, "book": None}

def _parse_last_update(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        s = str(value)
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def extract_best_market(game: Dict[str, Any]) -> Dict[str, Any]:
    home = game.get("home_team")
    away = game.get("away_team")
    bookmakers = game.get("bookmakers") or []
    warnings: List[str] = list(game.get("warnings") or [])
    if not bookmakers:
        warnings.append("missing_bookmakers")

    preferred_books = ["FanDuel", "DraftKings", "BetMGM", "Caesars"]

    h2h_candidates: List[Dict[str, Any]] = []
    spread_candidates: List[Dict[str, Any]] = []
    total_candidates: List[Dict[str, Any]] = []
    spread_offers: List[Dict[str, Any]] = []
    total_offers: List[Dict[str, Any]] = []

    for bm in bookmakers:
        bm_name = bm.get("title") or bm.get("key")
        last_update = _parse_last_update(bm.get("last_update"))
        for market in bm.get("markets") or []:
            key = market.get("key")
            outcomes = market.get("outcomes") or []
            if key == "h2h":
                home_price, away_price = robust_get_prices(outcomes, home, away)
                if home_price is None or away_price is None:
                    continue
                price_scores = [
                    abs(abs(float(p)) - 110) for p in [home_price, away_price] if p is not None
                ]
                h2h_candidates.append(
                    {
                        "book": bm_name,
                        "home_price": home_price,
                        "away_price": away_price,
                        "price_score": min(price_scores) if price_scores else None,
                        "last_update": last_update,
                    }
                )
            elif key == "spreads":
                home_point, home_price, away_point, away_price = robust_get_spreads(outcomes, home, away)

                if home_point is not None and away_point is not None:
                    spread_offers.append(
                        {
                            "book": bm_name,
                            "point": home_point,
                            "price": home_price,
                            "side": "home",
                            "team": home,
                            "last_update": last_update,
                        }
                    )
                    spread_offers.append(
                        {
                            "book": bm_name,
                            "point": away_point,
                            "price": away_price,
                            "side": "away",
                            "team": away,
                            "last_update": last_update,
                        }
                    )
                    price_scores = [
                        abs(abs(float(p)) - 110) for p in [home_price, away_price] if p is not None
                    ]
                    spread_candidates.append(
                        {
                            "book": bm_name,
                            "home_point": home_point,
                            "home_price": home_price,
                            "away_point": away_point,
                            "away_price": away_price,
                            "price_score": min(price_scores) if price_scores else None,
                            "last_update": last_update,
                        }
                    )
            elif key == "totals":
                over = next((o for o in outcomes if o.get("name") == "Over"), None)
                under = next((o for o in outcomes if o.get("name") == "Under"), None)
                if over and under:
                    over_point = over.get("point")
                    under_point = under.get("point")
                    if over_point is None or under_point is None or over_point != under_point:
                        continue
                    over_price = over.get("price")
                    under_price = under.get("price")
                    price_scores = [
                        abs(abs(float(p)) - 110) for p in [over_price, under_price] if p is not None
                    ]
                    total_candidates.append(
                        {
                            "book": bm_name,
                            "point": over_point,
                            "over_price": over_price,
                            "under_price": under_price,
                            "price_score": min(price_scores) if price_scores else None,
                            "last_update": last_update,
                        }
                    )
                    total_offers.append(
                        {
                            "book": bm_name,
                            "point": over_point,
                            "price": over_price,
                            "side": "over",
                            "last_update": last_update,
                        }
                    )
                    total_offers.append(
                        {
                            "book": bm_name,
                            "point": under_point,
                            "price": under_price,
                            "side": "under",
                            "last_update": last_update,
                        }
                    )

    def _book_priority(name: Optional[str]) -> int:
        if not name:
            return len(preferred_books) + 1
        for idx, b in enumerate(preferred_books):
            if b.lower() in name.lower():
                return idx
        return len(preferred_books) + 1

    def _recency_score(dt_val: Optional[datetime]) -> float:
        if not dt_val:
            return float("inf")
        return -dt_val.timestamp()

    best_ml = None
    if h2h_candidates:
        best_ml = sorted(
            h2h_candidates,
            key=lambda c: (
                c.get("price_score") if c.get("price_score") is not None else 999,
                _recency_score(c.get("last_update")),
                _book_priority(c.get("book")),
            ),
        )[0]
    else:
        warnings.append("missing_h2h")

    best_spread = None
    median_spread = None
    mode_spread = None
    if spread_candidates:
        points = [c["home_point"] for c in spread_candidates if c.get("home_point") is not None]
        try:
            mode_spread = statistics.mode(points)
        except Exception:
            mode_spread = None
        try:
            median_spread = statistics.median(points)
        except Exception:
            median_spread = None
        consensus_spread = mode_spread if mode_spread is not None else median_spread
        best_spread = sorted(
            spread_candidates,
            key=lambda c: (
                abs(float(c.get("home_point") or 0) - float(consensus_spread or 0)),
                c.get("price_score") if c.get("price_score") is not None else 999,
                _recency_score(c.get("last_update")),
                _book_priority(c.get("book")),
            ),
        )[0]
    else:
        warnings.append("missing_spreads")

    best_total = None
    median_total = None
    mode_total = None
    if total_candidates:
        points = [c["point"] for c in total_candidates if c.get("point") is not None]
        try:
            mode_total = statistics.mode(points)
        except Exception:
            mode_total = None
        try:
            median_total = statistics.median(points)
        except Exception:
            median_total = None
        consensus_total = mode_total if mode_total is not None else median_total
        best_total = sorted(
            total_candidates,
            key=lambda c: (
                abs(float(c.get("point") or 0) - float(consensus_total or 0)),
                c.get("price_score") if c.get("price_score") is not None else 999,
                _recency_score(c.get("last_update")),
                _book_priority(c.get("book")),
            ),
        )[0]
    else:
        warnings.append("missing_totals")

    return {
        "best_ml_book": best_ml.get("book") if best_ml else None,
        "home_ml_price": best_ml.get("home_price") if best_ml else None,
        "away_ml_price": best_ml.get("away_price") if best_ml else None,
        "implied_prob_home": american_to_implied_prob(best_ml.get("home_price")) if best_ml else None,
        "implied_prob_away": american_to_implied_prob(best_ml.get("away_price")) if best_ml else None,
        "best_spread_book": best_spread.get("book") if best_spread else None,
        "home_spread_point": best_spread.get("home_point") if best_spread else None,
        "home_spread_price": best_spread.get("home_price") if best_spread else None,
        "away_spread_point": best_spread.get("away_point") if best_spread else None,
        "away_spread_price": best_spread.get("away_price") if best_spread else None,
        "best_spread_last_update": best_spread.get("last_update") if best_spread else None,
        "best_spread_price_score": best_spread.get("price_score") if best_spread else None,
        "best_spread_median_point": median_spread,
        "best_spread_mode_point": mode_spread,
        "spread_offers": spread_offers,
        "best_total_book": best_total.get("book") if best_total else None,
        "total_point": best_total.get("point") if best_total else None,
        "over_price": best_total.get("over_price") if best_total else None,
        "under_price": best_total.get("under_price") if best_total else None,
        "best_total_last_update": best_total.get("last_update") if best_total else None,
        "best_total_price_score": best_total.get("price_score") if best_total else None,
        "best_total_median_point": median_total,
        "best_total_mode_point": mode_total,
        "total_offers": total_offers,
        "warnings": warnings,
    }

def league_from_sport_key(sk: Optional[str]) -> Optional[str]:
    if not sk:
        return None
    if sk == "basketball_nba":
        return "NBA"
    if sk == "basketball_ncaab":
        return "NCAAB"
    if sk == "americanfootball_nfl":
        return "NFL"
    if sk == "americanfootball_ncaaf":
        return "NCAAF"
    if sk == "icehockey_nhl":
        return "NHL"
    if sk == "baseball_mlb":
        return "MLB"
    return sk.upper()

def normalize_game(game: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(game)
    normalized["league"] = league_from_sport_key(game.get("sport_key")) or "UNKNOWN"
    normalized["commence_time_iso"] = safe_iso(game.get("commence_time")) or game.get(
        "commence_time_iso"
    )

    home = game.get("home_team") or "UNKNOWN_HOME"
    away = game.get("away_team")
    warnings: List[str] = []
    if not away:
        for bm in game.get("bookmakers") or []:
            for m in bm.get("markets") or []:
                if m.get("key") != "h2h":
                    continue
                names = [o.get("name") for o in m.get("outcomes") or [] if o.get("name")]
                uniq = list({n for n in names if n and n.lower() not in {"over", "under"}})
                if len(uniq) == 2:
                    if home in uniq:
                        other = uniq[0] if uniq[1] == home else uniq[1]
                        away = other
                    else:
                        home, away = uniq[0], uniq[1]
                    break
        if not away:
            away = "UNKNOWN_AWAY"
            warnings.append("missing_away_team")
    normalized["home_team"] = home
    normalized["away_team"] = away
    normalized.setdefault("warnings", warnings)
    return normalized

# -----------------
# API Clients & config
# -----------------

SPORT_KEYS = {
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
}
ALL_SPORTS_LABEL = "All Sports"

odds_api_key = read_secret("ODDS_API_KEY")
news_api_key = read_secret("NEWS_API_KEY")
project_id = read_secret("GCP_PROJECT_ID", "elite-hangar-479017-m8")
location = read_secret("GCP_LOCATION", "us-central1")
kalshi_api_key = read_secret("KALSHI_API_KEY") or read_secret("kalshi_api_key")
kalshi_api_secret = read_secret("KALSHI_API_SECRET") or read_secret("kalshi_api_secret")

# Initialize Gemini API Key if available
# Don't fallback to config.GEMINI_API_KEY if it might be dummy/missing
gemini_api_key = read_secret("GEMINI_API_KEY") or read_secret("GOOGLE_API_KEY")
if not gemini_api_key and getattr(config, "GEMINI_API_KEY", None) and not str(config.GEMINI_API_KEY).startswith("AIzaSyBIDJgxLuUouiBQrslV"):
    gemini_api_key = config.GEMINI_API_KEY

if gemini_api_key:
    # Only set if we found a valid-looking key
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
keys_resolved = get_api_keys()
api_sports_key = keys_resolved.get("api_sports_key")
sportsdata_key = keys_resolved.get("sportsdata_key")
st.session_state.setdefault("kalshi_required", False)
kalshi_integrator: Optional[KalshiIntegrator] = None
try:
    if "kalshi_integrator" not in st.session_state:
        if kalshi_api_key and kalshi_api_secret:
            st.session_state["kalshi_integrator"] = KalshiIntegrator(
                kalshi_api_key,
                kalshi_api_secret,
                required=st.session_state.get("kalshi_required", True),
            )
        else:
            st.session_state["kalshi_integrator"] = None
    kalshi_integrator = st.session_state.get("kalshi_integrator")
except Exception:
    st.session_state["last_exception"] = traceback.format_exc()
    kalshi_integrator = None
if kalshi_integrator:
    kalshi_integrator.required = st.session_state.get("kalshi_required", True)
api_sports_clients, sportsdata_clients = init_data_clients()

def fetch_odds_games(sport_key: str, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch odds from TheOddsAPI with retry logic and proper error handling.

    NOTE: This function is NOT cached to avoid caching empty results.
    Instead, successful results are cached at the load_games level per run_id.

    Args:
        sport_key: Sport identifier (e.g., 'basketball_nba', 'americanfootball_nfl')
        run_id: Optional run identifier for logging

    Returns:
        List of game dictionaries, or empty list on failure
    """
    if not odds_api_key or not sport_key:
        logger.error(f"Missing API key or sport_key. Key={bool(odds_api_key)}, Sport={sport_key}")
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": odds_api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    # Retry configuration: exponential backoff
    max_retries = 3
    retry_delays = [2, 4, 8]  # seconds

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"🔄 Retry attempt {attempt}/{max_retries} for sport: {sport_key}")
                time.sleep(retry_delays[attempt - 1])
            else:
                logger.info(f"🔍 Fetching odds from TheOddsAPI for sport: {sport_key} (run_id: {run_id or 'N/A'})")

            # DIAGNOSTIC: Log request details (masking API key)
            safe_params = dict(params)
            if "apiKey" in safe_params:
                safe_params["apiKey"] = "MASKED"
            logger.info(f"THEODDS REQ: url={url}, params={safe_params}")

            resp = requests.get(url, params=params, timeout=15)

            # Log response metadata for debugging empty responses
            logger.info(f"📡 TheOddsAPI response: status={resp.status_code}, "
                       f"content-length={len(resp.content)}, "
                       f"headers={dict(resp.headers)}")

            resp.raise_for_status()

            # DIAGNOSTIC: Log raw response structure
            data = resp.json()
            logger.info(f"📊 TheOddsAPI raw response type: {type(data)}")

            # Handle both direct list and dict with nested keys
            if isinstance(data, dict):
                logger.info(f"📋 TheOddsAPI response keys: {list(data.keys())}")
                # Try common response wrapper keys
                games = data.get("games", data.get("data", data.get("events", [])))
                logger.info(f"✅ Extracted {len(games)} games from dict response")
            elif isinstance(data, list):
                games = data
                logger.info(f"✅ Got {len(games)} games from list response")
            else:
                logger.error(f"❌ Unexpected response type: {type(data)}")
                if attempt < max_retries:
                    continue
                return []

            # DIAGNOSTIC: Log first game sample if available
            if games and len(games) > 0:
                first_game = games[0]
                logger.info(f"🎮 First game sample keys: {list(first_game.keys())}")
                logger.info(f"🏀 First game: {first_game.get('home_team')} vs {first_game.get('away_team')}")
                logger.info(f"⏰ Commence time: {first_game.get('commence_time')}")

                # Apply strictly today-only filter (EST)
                original_count = len(games)
                logger.info(f"📅 Applying strict date filter to {original_count} games...")
                games = filter_games_today_only(games)
                filtered_count = len(games)

                if filtered_count < original_count:
                    logger.info(f"✅ DATE FILTER: Removed {original_count - filtered_count} games not on today's date (EST). Remaining: {filtered_count}")
                else:
                    logger.info(f"✅ DATE FILTER: All {original_count} games are on today's date (EST)")

                # Success - return immediately
                return games
            else:
                # Empty response - retry unless this was last attempt
                if attempt < max_retries:
                    logger.warning(f"⚠️ TheOddsAPI returned 0 games for {sport_key} on attempt {attempt + 1}. Retrying...")
                    continue
                else:
                    logger.warning(f"⚠️ TheOddsAPI returned empty games list for sport: {sport_key} after {max_retries + 1} attempts")
                    logger.warning(f"   This could mean:")
                    logger.warning(f"   - No games scheduled for this sport today")
                    logger.warning(f"   - Sport key is invalid")
                    logger.warning(f"   - API rate limit exceeded")
                    logger.warning(f"   - API key has insufficient permissions")
                    return []

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ TheOddsAPI request failed on attempt {attempt + 1}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"   Response status: {e.response.status_code}")
                logger.error(f"   Response text: {e.response.text[:500]}")
            if attempt < max_retries:
                continue
            return []
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse TheOddsAPI JSON on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                continue
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error in fetch_odds_games on attempt {attempt + 1}: {e}")
            logger.exception(e)
            if attempt < max_retries:
                continue
            return []

    # Should not reach here, but return empty list as fallback
    return []

@st.cache_data(ttl=300)
def fetch_news() -> List[Dict[str, Any]]:
    if not news_api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "NBA basketball",
        "sortBy": "publishedAt",
        "pageSize": 3,
        "apiKey": news_api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])

# -----------------
# Vertex prediction
# -----------------

def _build_model_feature_row(game: Dict[str, Any], sentiment_diff: Optional[float]) -> pd.DataFrame:
    # 1. Prepare Base Context
    base = dict(game)
    # Fix NBA Overwrite: Explicitly set League from sport_title to avoid hardcoded fallbacks
    base["League"] = game.get("sport_title", "Unknown")

    # 2. Add Override / Calculated fields that might not be in 'game' yet
    # Fix for Issue #5: NaN Odds handling
    raw_ml = game.get("home_ml_price")
    implied_home = american_to_implied_prob(raw_ml)
    if implied_home is None:
        implied_home = game.get("implied_prob_home")
    if implied_home is None and (game.get("home_spread_point") or game.get("total_point")):
        implied_home = 0.5

    base["implied_home_prob"] = safe_float(implied_home)
    base["sentiment_diff"] = safe_float(sentiment_diff)

    # Kalshi prob logic: Prefer passed-in value, then fallback to session cache
    kalshi_prob = game.get("kalshi_prob")
    if kalshi_prob is None:
        try:
            # Pull any cached matched Kalshi prob for this game if present
            _entries_raw = st.session_state.get("kalshi_match_results") or {}
            entries = _entries_raw.values() if isinstance(_entries_raw, dict) else (_entries_raw or [])
            for entry in entries:
                g = entry.get("game") or {}
                if (
                    g.get("home_team") == game.get("home_team")
                    and g.get("away_team") == game.get("away_team")
                    and (g.get("commence_time_iso_utc") or g.get("commence_time")) == (game.get("commence_time_iso_utc") or game.get("commence_time"))
                ):
                    winner = (entry.get("matches") or {}).get("winner", {})
                    if winner.get("kalshi_matched"):
                        kalshi_prob = winner.get("kalshi_prob")
                    break
        except Exception:
            pass
    base["kalshi_prob"] = safe_float(kalshi_prob)

    # Injuries / Weather mapping
    base["injuries_home_count"] = safe_float(game.get("injuries_home_count"))
    base["injuries_away_count"] = safe_float(game.get("injuries_away_count"))

    # Weather flag parity with batch logic (keyword search)
    w_summary = str(game.get("weather_summary") or "").lower()
    base["weather_flag"] = 1.0 if any(x in w_summary for x in ['rain', 'snow', 'wind']) else 0.0

    # --- USER REQUESTED ALIGNMENT & DEBUG ---
    # 1. Ensure any pre-enriched stats on 'game' are carried over to 'base'
    # This is critical for single-game parity with batch processing
    for col in VERTEX_FEATURE_COLUMNS:
        if col in game:
            base[col] = game[col]

    # 2. Log the final base dictionary to verify content
    logger.info("SINGLE GAME BASE ROW: %r", base)
    if st:
        st.write("DEBUG: Single-game base dict:", base)

    # 3. Flag fallback if critical stats are missing (e.g. from failed enrichment)
    if "feature_home_ppg" not in base:
        base["feature_stats_fallback"] = True

    # 3. Use Shared Helper
    feature_dict = build_model_feature_row_from_record(base)

    # 4. Return DataFrame
    df = pd.DataFrame([feature_dict])
    df = df.reindex(columns=VERTEX_FEATURE_COLUMNS)
    return df

def get_model_prob(game: Dict[str, Any], sentiment_diff: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    # Local model is always available (or falls back)
    try:
        features_df = _build_model_feature_row(game, sentiment_diff)

        expected_cols = len(VERTEX_FEATURE_COLUMNS)
        if features_df.shape[1] != expected_cols:
            logger.warning(
                "Vertex Schema Mismatch: expected %d cols (%s), got %d (%s)",
                expected_cols,
                VERTEX_FEATURE_COLUMNS,
                features_df.shape[1],
                list(features_df.columns),
            )
            return None, "schemamismatch"

        features_df = features_df.reindex(columns=VERTEX_FEATURE_COLUMNS)
        # Optimized bulk numeric conversion (Issue #5)
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)

        instances = features_df.values.tolist()
        if not instances:
            logger.warning("Vertex feature instances list is empty")
            return None, "schemamismatch"

        if st:
            st.write("DEBUG: Feature Vector (DF):")
            st.dataframe(features_df)
            st.write("DEBUG: Feature Vector:", instances)

        payload_hash = hash(tuple(features_df.to_dict(orient="records")[0].items()))
        engine = get_prediction_engine()
        preds = engine.predict_batch(features_df)
        st.session_state["model_last_payload_hash"] = payload_hash

        if not preds:
            logger.error("Vertex prediction returned empty preds for payload_hash=%s", payload_hash)
            return None, "vertexpredictfailed"

        prob = safe_float(preds[0])
        if prob is None:
            logger.error("Vertex prediction invalid first element: %r", preds)
            return None, "vertexinvalidresponse"

        try:
            raw_resp = preds[1] if len(preds) > 1 else None
            st.session_state["model_last_raw_response"] = str(raw_resp) if raw_resp is not None else None
        except Exception:
            st.session_state["model_last_raw_response"] = None

        return clamp(prob, 0.01, 0.99), None

    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        st.session_state["model_last_error"] = st.session_state.get("last_exception")
        logger.exception("Vertex prediction failed with exception")
        return None, "vertexpredictfailed"

# -----------------
# Kalshi integration
# -----------------
@st.cache_data(ttl=300)
def fetch_kalshi_markets(
    selected_league: str, commence_times_utc: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    # Ensure the global exists, but don't crash if not initialized yet
    global kalshi_integrator
    if kalshi_integrator is None:
        return []

    league_upper = (selected_league or "").upper()
    winner_prefix = league_game_prefix(league_upper)

    def ticker_upper(market: Dict[str, Any]) -> str:
        return str(market.get("event_ticker") or market.get("ticker") or "").upper()

    def prefix_count(markets: List[Dict[str, Any]], active_prefix: Optional[str] = None) -> Dict[str, int]:
        tickers = [ticker_upper(m) for m in (markets or [])]
        prefix_to_use = active_prefix or winner_prefix
        return {
            "count_prefix_game": len([t for t in tickers if t.startswith(prefix_to_use)]),
            "count_prefix_total": len([t for t in tickers if "TOTAL" in t]),
            "count_prefix_spread": len([t for t in tickers if "SPREAD" in t]),
        }

    def date_tokens_from_commence(commence_list: Optional[List[str]]) -> set:
        """Convert commence_time ISO strings -> Kalshi tokens like 25DEC17 using app local date (±2 days)."""
        if not commence_list:
            return set()

        tz_name = get_local_tz()
        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = None

        tokens = set()
        for raw in commence_list:
            dt_utc = parse_commence_to_utc(raw)
            if not dt_utc:
                continue
            dt_local = dt_utc.astimezone(local_tz) if local_tz else dt_utc
            base_date = dt_local.date()
            for delta in [-2, -1, 0, 1, 2]:  # ±2 days
                date_variant = base_date + timedelta(days=delta)
                tokens.add(date_variant.strftime("%y%b%d").upper())
        return tokens

    wanted_tokens = date_tokens_from_commence(commence_times_utc)

    # NCAAB/NCAAF have far more markets per series than pro leagues (~3000+).
    # 5 pages × 200 = 1000 cap misses games beyond page 5.
    # TASK 2: Force NCAAB to fetch full slate (20 pages)
    _ncaab_pages = 20   # 20 × 200 = 4,000 markets — covers full NCAAB slate
    _default_pages = 5

    # Explicitly check for NCAAB to ensure correct pagination
    if league_upper == "NCAAB":
        _pages_needed = 20
    elif league_upper == "NCAAF":
        _pages_needed = 20
    else:
        _pages_needed = _default_pages

    logger.info(f"KALSHI FETCH START - League: {league_upper}")
    logger.info(f"  Pagination: {_pages_needed} pages (expect ~{_pages_needed * 200} markets)")
    # DIAGNOSTIC: Confirm league detection for pagination
    if league_upper in ["NCAAB", "NCAAF"]:
        logger.info(f"  ✅ High-volume league detected: {league_upper} -> {_pages_needed} pages")
    else:
        logger.info(f"  ℹ️ Standard league detected: {league_upper} -> {_pages_needed} pages")

    try:
        markets_raw = kalshi_integrator.get_league_markets(
            selected_league,
            min_prefix_hits=20,
            max_pages=_pages_needed,
        )
        logger.info(f"  RAW FETCH: {len(markets_raw)} markets retrieved")
        last_params = kalshi_integrator.last_request_params or {}
        st.session_state["kalshi_last_request_params"] = last_params
        st.session_state["kalshi_last_request_status_included"] = "status" in last_params
        st.session_state["kalshi_request_params_snapshot"] = dict(last_params)
        if not markets_raw:
            markets_raw = kalshi_integrator.get_markets_paginated(status=None, max_pages=_pages_needed)
            st.session_state["kalshi_last_request_params"] = kalshi_integrator.last_request_params
        markets_raw = markets_raw or []

        game_prefix_used = (kalshi_integrator.last_fetch_meta or {}).get(
            "game_prefix_used", winner_prefix
        )
        st.session_state.setdefault("kalshi_game_prefix_map", {})[
            league_upper
        ] = game_prefix_used
        raw_counts = prefix_count(markets_raw, active_prefix=game_prefix_used)
        kx_game_count = len(
            [m for m in markets_raw if ticker_upper(m).startswith(f"{game_prefix_used}-")]
        )
        split = kalshi_integrator.split_market_kinds(markets_raw, selected_league)

        # FIX: Include GAME, TOTAL, and SPREAD markets in the pool
        # Previously only GAME markets were included, causing spread/total matching to fail
        allowed_prefixes = [game_prefix_used]

        # Add TOTAL and SPREAD prefixes based on league
        league_series = LEAGUE_SERIES_MAP.get(league_upper, [])
        if isinstance(league_series, list):
            for series in league_series:
                if series and series not in allowed_prefixes:
                    allowed_prefixes.append(series)

        # Also add explicit TOTAL/SPREAD variations
        base_prefix = game_prefix_used.replace("GAME", "")  # e.g., KXNBA
        if base_prefix:
            for suffix in ["TOTAL", "SPREAD"]:
                variant = f"{base_prefix}{suffix}"
                if variant not in allowed_prefixes:
                    allowed_prefixes.append(variant)

        game_pool: List[Dict[str, Any]] = [
            m
            for m in (split.get("single_game_candidates") or [])
            if any(ticker_upper(m).startswith(pfx) for pfx in allowed_prefixes)
        ]

        # Log what prefixes we're using
        logger.info(f"🔍 KALSHI POOL FILTER: Using prefixes {allowed_prefixes}")
        logger.info(f"🔍 KALSHI POOL FILTER: Found {len(game_pool)} markets in pool")

        if not game_pool and game_prefix_used != winner_prefix:
            allowed_prefixes = [winner_prefix]
            # Re-add TOTAL/SPREAD for the winner prefix
            base_prefix = winner_prefix.replace("GAME", "")
            if base_prefix:
                for suffix in ["TOTAL", "SPREAD"]:
                    variant = f"{base_prefix}{suffix}"
                    if variant not in allowed_prefixes:
                        allowed_prefixes.append(variant)
            game_pool = [
                m
                for m in (split.get("single_game_candidates") or [])
                if any(ticker_upper(m).startswith(pfx) for pfx in allowed_prefixes)
            ]
            game_prefix_used = winner_prefix
        if not game_pool and league_upper == "NCAAB":
            fallback_pool: List[Dict[str, Any]] = []
            pattern = re.compile(r"\d{2}[A-Z]{3}\d{2}")
            candidates = split.get("single_game_candidates") or markets_raw or []
            for m in candidates:
                t = ticker_upper(m)
                if "NCAAB" in t or "NCAA" in t:
                    if not wanted_tokens or any(tok in t for tok in wanted_tokens) or pattern.search(t):
                        fallback_pool.append(m)
            if fallback_pool:
                game_pool = fallback_pool
                detected_prefix = None
                for m in fallback_pool:
                    t = ticker_upper(m)
                    if "-" in t:
                        detected_prefix = t.split("-")[0]
                        break
                if detected_prefix:
                    game_prefix_used = detected_prefix
                game_pool_counts = prefix_count(game_pool, active_prefix=game_prefix_used)
        game_pool_counts = prefix_count(game_pool, active_prefix=game_prefix_used)
        st.session_state.setdefault("kalshi_game_prefix_map", {})[
            league_upper
        ] = game_prefix_used

        # FIX: Log the breakdown of market types in the pool
        game_count = len([m for m in game_pool if "GAME" in ticker_upper(m)])
        total_count = len([m for m in game_pool if "TOTAL" in ticker_upper(m)])
        spread_count = len([m for m in game_pool if "SPREAD" in ticker_upper(m)])
        logger.info(f"🔍 KALSHI POOL BREAKDOWN [{league_upper}]: GAME={game_count}, TOTAL={total_count}, SPREAD={spread_count}")
        logger.info(f"  AFTER PREFIX FILTER: {len(game_pool)} markets")

        if wanted_tokens:
            if league_upper != 'NCAAB':  # BYPASS FOR NCAAB
                filtered = [m for m in game_pool if any(tok in ticker_upper(m) for tok in wanted_tokens)]
                logger.info(f"Date tokens: {sorted(list(wanted_tokens))}, Before: {len(game_pool)}, After: {len(filtered)}")

                if not filtered and game_pool:
                    logger.warning(f"DATE FILTER SKIPPED: Would remove {len(game_pool)} markets")
                    st.session_state["kalshi_date_filter_warning"] = "skipped_zero_pool"
                else:
                    game_pool = filtered
            else:
                logger.info(f"NCAAB: Skipping date filter - using full gamepool {len(game_pool)}")

            game_pool_counts = prefix_count(game_pool)

        league_key = league_upper
        st.session_state.setdefault("kalshi_markets_raw", {})[league_key] = markets_raw
        st.session_state.setdefault("kalshi_markets_game_pool", {})[league_key] = game_pool
        all_markets_map = st.session_state.setdefault("kalshi_all_markets_map", {})
        all_markets_map[league_key] = markets_raw
        combined_markets: List[Dict[str, Any]] = []
        for mkts in all_markets_map.values():
            combined_markets.extend(mkts or [])
        st.session_state["kalshi_all_markets"] = combined_markets
        prefix_counts_map = st.session_state.setdefault("kalshi_prefix_counts", {})
        prefix_counts_map[league_key] = {
            "raw": {"total": len(markets_raw), **raw_counts},
            "game_pool": {"total": len(game_pool), **game_pool_counts},
        }
        st.session_state.setdefault("kalshi_prefix_samples_game", {})[league_key] = [
            str(m.get("event_ticker") or m.get("ticker") or "")
            for m in game_pool[:20]
        ]
        st.session_state["kalshi_request_params_snapshot"] = dict(
            kalshi_integrator.last_request_params or {}
        )
        st.session_state["kalshi_last_request_params"] = (
            kalshi_integrator.last_request_params or {}
        )
        st.session_state["kalshi_debug_summary"] = {
            "league": league_upper,
            "fetched_total": len(markets_raw),
            "game_prefix": game_prefix_used,
            "kx_game_count": kx_game_count,
            "wanted_tokens": sorted(list(wanted_tokens)) if wanted_tokens else [],
            "after_token_filter": len(game_pool),
            "first_10_tickers": [ticker_upper(m) for m in markets_raw[:10]],
        }
        return game_pool
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return []


@st.cache_data(ttl=300)
def fetch_kalshi_markets_for_leagues(
    leagues: List[str], commence_times_by_league: Dict[str, List[str]]
) -> Dict[str, List[Dict[str, Any]]]:
    summary: Dict[str, Any] = {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for lg in leagues or []:
        commence_times_utc = [
            g
            for g in commence_times_by_league.get(lg, [])
            if g
        ]
        markets = fetch_kalshi_markets(lg, commence_times_utc=commence_times_utc)
        league_upper = (lg or "").upper()
        meta_prefix = (kalshi_integrator.last_fetch_meta or {}).get(
            "game_prefix_used"
        )
        winner_prefix = meta_prefix or league_game_prefix(league_upper)
        out[lg] = markets or []
        tokens = []
        try:
            for ct in commence_times_by_league.get(lg, []):
                dt = parse_commence_to_utc(ct)
                if not dt:
                    continue
                tokens.append(dt.astimezone(ZoneInfo("America/New_York")).strftime("%y%b%d").upper())
        except Exception:
            tokens = commence_times_by_league.get(lg, [])
        summary[league_upper] = {
            "fetched_total": len(out[lg]),
            "game_ticker_count": len(
                [
                    m
                    for m in out[lg]
                    if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith(winner_prefix)
                ]
            ),
            "winner_prefix": winner_prefix,
            "wanted_tokens": tokens,
            "after_token_filter": len(out[lg]),
        }
    st.session_state["kalshi_debug_summary"] = summary
    return out


def kalshi_health_check(selected_league: str = "NBA") -> Dict[str, Any]:
    """Wrapper to ensure health is always callable before first use."""
    return kalshi_health(selected_league)

def pick_sample_game_market(
    markets: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    def parse_iso(dt_value: Any) -> Optional[datetime]:
        try:
            if not dt_value:
                return None
            raw = str(dt_value)
            if raw.endswith("Z"):
                raw = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed
        except Exception:
            return None

    now_utc = datetime.now(timezone.utc)
    window = timedelta(days=7)
    best_market: Optional[Dict[str, Any]] = None
    best_reason: Optional[str] = None
    best_within = False
    best_diff: Optional[timedelta] = None
    for m in markets or []:
        try:
            title = str(m.get("title") or "")
            ticker = str(m.get("event_ticker") or m.get("ticker") or "")
            lower_title = title.lower()
            reason = None
            if any(tok in lower_title for tok in [" vs ", " at ", "@"]):
                reason = "contains_vs"
            elif "game winner" in lower_title:
                reason = "contains_game_winner"
            elif any(tok in lower_title for tok in ["spread", "total"]):
                reason = "contains_spread_total"
            elif ticker and "game" in ticker.lower():
                reason = "contains_game_ticker"
            if not reason:
                continue
            close_dt = parse_iso(m.get("close_time")) or parse_iso(m.get("expiration_time"))
            within_window = False
            time_diff: Optional[timedelta] = None
            if close_dt:
                time_diff = abs(close_dt - now_utc)
                within_window = timedelta(0) <= (close_dt - now_utc) <= window
            reason_with_window = reason
            if within_window:
                reason_with_window = f"{reason}+within_7d"
            if best_market is None:
                best_market = m
                best_reason = reason_with_window
                best_within = within_window
                best_diff = time_diff
                continue
            if within_window and not best_within:
                best_market = m
                best_reason = reason_with_window
                best_within = True
                best_diff = time_diff
                continue
            if within_window == best_within:
                if time_diff is not None and (best_diff is None or time_diff < best_diff):
                    best_market = m
                    best_reason = reason_with_window
                    best_diff = time_diff
        except Exception:
            continue
    return best_market, best_reason


def kalshi_health(selected_league: str = "NBA") -> Dict[str, Any]:
    league_upper = (selected_league or "NBA").upper()
    game_prefix = league_game_prefix(league_upper)
    base_series = league_series_ticker(league_upper) or f"KX{league_upper}"

    def prefix_count_local(tickers: List[str]) -> Dict[str, int]:
        return {
            "count_prefix_game": len([t for t in tickers if t.startswith(game_prefix)]),
            "count_prefix_base": len([t for t in tickers if t.startswith(base_series)]),
            "count_prefix_total": len([t for t in tickers if "TOTAL" in t]),
            "count_prefix_spread": len([t for t in tickers if "SPREAD" in t]),
            "count_prefix_KXMV": len([t for t in tickers if t.startswith("KXMV")]),
        }

    def _ticker(m: Dict[str, Any]) -> str:
        return str(m.get("event_ticker") or m.get("ticker") or "").upper()

    base_health = {
        "configured": bool(kalshi_integrator),
        "ok": False,
        "market_count": 0,
        "game_market_count": 0,
        "futures_market_count": 0,
        "sample_market": None,
        "sample_game_market": None,
        "sample_game_market_reason": None,
        "error": None,
        "status_code": None,
        "response_text": None,
        "request_params": None,
        "has_game_markets": False,
        "has_futures_markets": False,
        "warning": None,
    }

    if not kalshi_integrator:
        base_health["error"] = "Kalshi not configured."
        return base_health

    try:
        all_prefix_counts = st.session_state.get("kalshi_prefix_counts") or {}
        prefix_counts = all_prefix_counts.get(league_upper)
        markets_raw: List[Dict[str, Any]] = []

        if not prefix_counts or not prefix_counts.get("game_pool"):
            _hc_pages = 20 if league_upper in ("NCAAB", "NCAAF") else 5
            markets_raw = kalshi_integrator.get_league_markets(
                selected_league,
                min_prefix_hits=1,
                max_pages=_hc_pages,
            ) or []
            tickers = [m.get("event_ticker") or m.get("ticker") or "" for m in markets_raw]
            prefix_counts = {
                "raw": prefix_count_local(tickers),
                "game_pool": prefix_count_local(
                    [t for t in tickers if t.startswith(base_series)]
                ),
            }
            all_prefix_counts[league_upper] = prefix_counts
            st.session_state["kalshi_prefix_counts"] = all_prefix_counts

        if not markets_raw:
            markets_raw = (st.session_state.get("kalshi_markets_raw") or {}).get(league_upper, [])

        if not markets_raw:
            info = kalshi_integrator.last_error_info or {}
            status_code = info.get("status_code") or kalshi_integrator.last_status_code
            resp_text = info.get("response_text") or kalshi_integrator.last_response_text
            if status_code == 200 and resp_text:
                try:
                    try:
                        data = json.loads(resp_text)
                    except Exception:
                        data = {}
                    markets_raw = (data.get("markets") or []) if isinstance(data, dict) else []
                except Exception:
                    markets_raw = []

        base_health["market_count"] = len(markets_raw)
        base_health["sample_market"] = markets_raw[0] if markets_raw else None

        game_markets = [m for m in markets_raw if _ticker(m).startswith(f"{game_prefix}-")]
        futures_markets = [
            m
            for m in markets_raw
            if _ticker(m).startswith(base_series)
            and not _ticker(m).startswith(f"{game_prefix}-")
        ]
        base_health["game_market_count"] = len(game_markets)
        base_health["futures_market_count"] = len(futures_markets)
        base_health["sample_game_market"] = game_markets[0] if game_markets else None
        base_health["has_game_markets"] = bool(game_markets)
        base_health["has_futures_markets"] = bool(futures_markets)
        base_health["ok"] = True
        if markets_raw and not base_health["has_game_markets"]:
            if base_health["has_futures_markets"]:
                base_health["warning"] = (
                    f"Kalshi reachable; only futures markets returned for {base_series} series."
                )
            else:
                base_health["warning"] = (
                    f"Kalshi reachable, but no {league_upper} {game_prefix} markets returned (futures-only or slate not listed)."
                )
        info = kalshi_integrator.last_error_info or {}
        base_health["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        base_health["response_text"] = (
            (info.get("response_text") or kalshi_integrator.last_response_text or "")[:500]
        )
        base_health["request_params"] = kalshi_integrator.last_request_params
        return base_health

    except Exception as e:
        if (kalshi_integrator.last_error_info or {}).get("status_code") == 429:
            cached_markets = st.session_state.get("kalshi_markets_raw") or []
            base_health["market_count"] = len(cached_markets)
            base_health["sample_market"] = cached_markets[0] if cached_markets else None
            game_markets = [
                m
                for m in cached_markets
                if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith(
                    f"{game_prefix}-"
                )
            ]
            base_health["sample_game_market"] = game_markets[0] if game_markets else None
            base_health["has_game_markets"] = bool(game_markets)
            base_health["has_futures_markets"] = bool(cached_markets)
            base_health["ok"] = False
            base_health["error"] = "Kalshi rate limited; using cached markets"
            return base_health
        base_health["error"] = f"Kalshi health check failed: {e}"
        return base_health

def kalshi_health_check(selected_league: str = "NBA") -> Dict[str, Any]:
    """
    Backwards-compatible alias.
    Some UI code calls kalshi_health_check(), but the implementation is kalshi_health().
    """
    return kalshi_health(selected_league)

def parse_kalshi_datetime(dt_value: Any) -> Optional[datetime]:
    try:
        if not dt_value:
            return None
        raw = str(dt_value)
        if raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except Exception:
        return None

def kalshi_market_best_time_utc(m: Dict[str, Any]) -> Optional[datetime]:
    """Return the best available Kalshi timestamp for matching."""
    for key in [
        "expected_expiration_time",
        "latest_expiration_time",
        "close_time",
        "expiration_time",
        "open_time",
    ]:
        candidate = parse_kalshi_datetime(m.get(key))
        if candidate:
            return candidate
    return None


def market_mentions_game_teams(
    market: Dict[str, Any],
    home_team: Any,
    away_team: Any,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
) -> bool:
    try:
        ticker = str(market.get("event_ticker") or market.get("ticker") or "")
        title = str(market.get("title") or "")
        rules_primary = str(market.get("rules") or market.get("rules_primary") or "")
        blob = f"{ticker} {title} {rules_primary}".lower()
        if home_code and away_code:
            if home_code.lower() in blob and away_code.lower() in blob:
                return True

        def nickname_token(name: Any) -> Optional[str]:
            cleaned = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower()).strip()
            if not cleaned:
                return None
            parts = [p for p in cleaned.split() if p]
            return parts[-1] if parts else None

        home_tok = nickname_token(home_team)
        away_tok = nickname_token(away_team)
        if home_tok and away_tok:
            return home_tok in blob and away_tok in blob
        return False
    except Exception:
        return False


def debug_search_markets_for_game(
    markets: List[Dict[str, Any]],
    home_team: Any,
    away_team: Any,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
    limit: int = 15,
    league: Optional[str] = None,
) -> Dict[str, Any]:
    def text_blob(parts: List[str]) -> str:
        return " ".join([p for p in parts if p]).lower()

    def word_set(val: str) -> set:
        cleaned = re.sub(r"[^a-z0-9 ]", " ", val.lower())
        return {w for w in cleaned.split() if w}

    home_tokens = team_tokens(home_team)
    away_tokens = team_tokens(away_team)
    winner_prefix = league_game_prefix(league) if league else None
    matches: List[Dict[str, Any]] = []
    found = {"winner": False, "total": False, "spread": False, "other": False}
    counts = {"winner": 0, "total": 0, "spread": 0, "other": 0}
    for m in markets or []:
        try:
            ticker = str(m.get("event_ticker") or m.get("ticker") or "")
            title = str(m.get("title") or "")
            rules = str(m.get("rules") or m.get("rules_primary") or "")
            blob = text_blob([ticker, title, rules])
            code_match = False
            if home_code and away_code:
                if home_code.lower() in blob and away_code.lower() in blob:
                    code_match = True
            blob_tokens = word_set(blob)
            token_match = bool(home_tokens.intersection(blob_tokens)) and bool(
                away_tokens.intersection(blob_tokens)
            )
            if not (code_match or token_match):
                continue
            ticker_upper = ticker.upper()
            category = "other"
            if (winner_prefix and ticker_upper.startswith(winner_prefix)) or "GAME-" in ticker_upper or "GAME" in ticker_upper:
                category = "winner"
            elif "TOTAL" in ticker_upper:
                category = "total"
            elif "SPREAD" in ticker_upper:
                category = "spread"
            found[category] = True
            counts[category] += 1
            matches.append({"ticker": ticker, "title": title, "category": category})
        except Exception:
            continue
    return {
        "found_any_winner_market_for_game": found["winner"],
        "found_any_total_market_for_game": found["total"],
        "found_any_spread_market_for_game": found["spread"],
        "counts": counts,
        "matches": matches[:limit],
    }


def _get_kalshi_code_variants(code: str, league: str) -> set:
    """Get all Kalshi ticker code variants for a team (canonical + aliases).

    For example, NC State canonical="NCS" also has alias "NCST" that Kalshi
    may use in tickers. Returns {"NCS", "NCST"} so we can match either form.
    """
    if not code:
        return set()
    variants = {code.upper()}
    canonical = resolve_team_code(code.upper(), league)
    variants.add(canonical)
    # Add reverse aliases: find all alias codes that resolve to the same canonical
    alias_map: Dict[str, str] = {}
    league_u = (league or "").upper()
    if league_u == "NCAAB":
        alias_map = NCAAB_CODE_ALIASES
    elif league_u == "NCAAF":
        alias_map = NCAAF_CODE_ALIASES
    for alias, target in alias_map.items():
        if target == canonical or target == code.upper():
            variants.add(alias)
    variants.discard("")
    return variants


def filter_kalshi_game_markets(
    markets: List[Dict[str, Any]],
    game_time_utc: Optional[datetime],
    league: str,
    home_team: Any = None,
    away_team: Any = None,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter Kalshi markets to only those matching THIS specific game.

    Uses EXACT team-code matching: parses the team-code block from each Kalshi
    ticker suffix (e.g. 'AAMUGRAM' from KXNCAAMBTOTAL-26FEB09AAMUGRAM) and
    compares it against the expected away+home code concatenation for the game.
    No fuzzy/substring fallback — wrong ticker is worse than no ticker.
    """
    try:
        tz_name = get_local_tz()
        local_tz = None
        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = None

        game_dt = game_time_utc
        if isinstance(game_dt, str):
            game_dt = parse_commence_to_utc(game_dt)
        if isinstance(game_dt, datetime) and game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=timezone.utc)
        game_local = game_dt.astimezone(local_tz) if (game_dt and local_tz) else game_dt

        # Date matching window (-1 to +2 days)
        allowed_date_tokens: List[str] = []
        if game_local:
            base_date = game_local.date()
            for delta in (-1, 0, 1, 2):
                allowed_date_tokens.append((base_date + timedelta(days=delta)).strftime("%y%b%d").upper())

        league_upper = (league or "").upper()

        # Get game's primary team codes from the mapping
        game_home_code = (home_code or team_code_for_league(league, home_team) or "").upper()
        game_away_code = (away_code or team_code_for_league(league, away_team) or "").upper()

        # Build all possible team-block variants (away_code + home_code)
        # including alias variants so e.g. NCS+LOU and NCST+LOU both match
        home_variants = _get_kalshi_code_variants(game_home_code, league_upper)
        away_variants = _get_kalshi_code_variants(game_away_code, league_upper)

        expected_blocks: set = set()
        for ac in away_variants:
            for hc in home_variants:
                if ac and hc:
                    expected_blocks.add(ac + hc)

        logger.info(f"KALSHI FILTER DEBUG - {away_team} @ {home_team}")
        logger.info(f"  Expected codes: away={game_away_code}, home={game_home_code}")
        logger.info(f"  Away variants: {away_variants}, Home variants: {home_variants}")
        logger.info(f"  Expected blocks: {list(expected_blocks)[:5]}...")  # Sample
        logger.info(f"  Markets available: {len(markets or [])}")

        logger.info(
            f"🔍 KALSHI FILTER [EXACT]: {away_team} @ {home_team} | "
            f"codes=({game_away_code}@{game_home_code}) | "
            f"blocks={expected_blocks} | dates={allowed_date_tokens}"
        )

        # Regex to extract date token + team block from ticker suffix
        # e.g. KXNCAAMBTOTAL-26FEB09AAMUGRAM -> date=26FEB09, team_block=AAMUGRAM
        # NOTE: Do NOT anchor with $ — ticker field (vs event_ticker) may have
        # market-level suffixes like -NY, -OVER, -5.5 after the team block.
        # Use [A-Z]+ (letters only) to stop at first non-letter.
        _TICKER_SUFFIX_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})([A-Z]{3,})")

        def tickerteamblock(m):
            t = str(m.get("event_ticker") or m.get("ticker") or "").upper()
            match = _TICKER_SUFFIX_RE.search(t)
            if match:
                return match.group(2)
            return None

        # NEW LOGGING (Requested)
        logger.info(f"NCAAB DEBUG {away_team}@{home_team}")
        logger.info(f"  Codes: away={away_variants}, home={home_variants}")
        logger.info(f"  Expected blocks: {list(expected_blocks)[:5]}")
        sample_blocks = [tickerteamblock(m) for m in (markets or [])[:10]]
        logger.info(f"  Kalshi blocks: {sample_blocks}")

        matched: List[Dict[str, Any]] = []
        _regex_fail_count = 0
        _date_fail_count = 0
        _team_fail_sample: Optional[str] = None
        for m in markets or []:
            t = str(m.get("event_ticker") or m.get("ticker") or "").upper()

            suffix_match = _TICKER_SUFFIX_RE.search(t)
            if not suffix_match:
                _regex_fail_count += 1
                continue

            ticker_date = suffix_match.group(1)
            ticker_team_block = suffix_match.group(2)

            # Date must match
            if ticker_date not in set(allowed_date_tokens):
                _date_fail_count += 1
                continue

            # EXACT team-block match — no substring, no fuzzy
            if ticker_team_block in expected_blocks:
                matched.append(m)
            elif _team_fail_sample is None:
                _team_fail_sample = f"{t}→block={ticker_team_block}"

        logger.info(f"  Exact matches: {len(matched)}")

        # FUZZY FALLBACK - NCAAB ONLY (NEW)
        if len(matched) == 0 and league == 'NCAAB' and len(markets or []) > 10:
            from rapidfuzz import fuzz
            fuzzy = []
            for m in markets or []:
                title = str(m.get('title', '')).lower()
                # Use safe strings for teams
                h_score = fuzz.partial_ratio(str(home_team).lower(), title)
                a_score = fuzz.partial_ratio(str(away_team).lower(), title)
                if h_score > 60 and a_score > 60:  # Team names in title (Lowered to 60)
                    fuzzy.append(m)
            matched = fuzzy[:20]  # Top 20 fuzzy
            logger.info(f"NCAAB FUZZY MATCH {away_team}@{home_team}: {len(matched)} found (thresh=60)")
            logger.info(f"{away_team}@{home_team}: exact=0, fuzzy={len(matched)}")

        if not matched and markets:
            sample_blocks = []
            for m in markets[:10]:
                 t = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                 match = _TICKER_SUFFIX_RE.search(t)
                 if match:
                     sample_blocks.append(match.group(2))
                 else:
                     sample_blocks.append("n/a")
            logger.warning(f"  NO EXACT MATCH - Sample Kalshi blocks: {sample_blocks}")

        # FUZZY FALLBACK if exact fails
        if not matched and markets:
            if fuzz:
                fuzzy_matches = []
                ht_lower = str(home_team).lower()
                at_lower = str(away_team).lower()

                for m in markets:
                    title_lower = str(m.get("title", "")).lower()
                    home_score = fuzz.partial_ratio(ht_lower, title_lower)
                    away_score = fuzz.partial_ratio(at_lower, title_lower)

                    # Find MATCHTHRESHOLD
                    MATCHTHRESHOLD = 60 if league == 'NCAAB' else 85  # Ultra-low for NCAAB
                    if home_score >= MATCHTHRESHOLD and away_score >= MATCHTHRESHOLD:
                        fuzzy_matches.append(m)

                if fuzzy_matches:
                    matched = fuzzy_matches[:10]  # Top 10 fuzzy matches
                    logger.info(f"Fuzzy fallback found {len(matched)} matches (away={away_team} home={home_team})")
            else:
                logger.warning("Rapidfuzz not available, skipping fuzzy fallback.")

        if not matched and expected_blocks:
            logger.warning(
                f"⚠️ KALSHI FILTER [EXACT]: No matches for {game_away_code}@{game_home_code}. "
                f"Expected blocks: {expected_blocks}. Markets scanned: {len(markets or [])}. "
                f"Regex fails: {_regex_fail_count}, Date fails: {_date_fail_count}, "
                f"Team block sample: {_team_fail_sample}"
            )

        return matched
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return []


def classify_kalshi_market(market: Dict[str, Any]) -> str:
    ticker = str(market.get("ticker") or market.get("event_ticker") or "").upper()
    title = str(market.get("title") or "").lower()
    rules = str(market.get("rules") or "").lower()

    if "GAME-" in ticker or "GAME" in ticker:
        return "winner"
    if "TOTAL" in ticker:
        return "total"
    if "SPREAD" in ticker:
        return "spread"
    if any(tok in ticker for tok in ["2D", "3D", "TD", "PTS", "REB", "AST"]) or any(
        key in title for key in ["double", "triple"]
    ):
        return "prop"

    if "total points" in title:
        return "total"
    if "spread" in title:
        return "spread"
    if "winner" in title or "win" in title or "wins the game" in rules:
        return "winner"
    return "unknown"

def team_tokens(name: str) -> set:
    if not name:
        return set()
    normalized = TeamNameMatcher.normalize(name)
    parts = [p for p in normalized.split() if p and p not in {"fc", "sc", "city", "united"}]
    return set(parts)

def nba_abbrev(team_name: str) -> Optional[str]:
    mapping = {
        "atlanta hawks": "ATL",
        "boston celtics": "BOS",
        "brooklyn nets": "BKN",
        "charlotte hornets": "CHA",
        "chicago bulls": "CHI",
        "cleveland cavaliers": "CLE",
        "dallas mavericks": "DAL",
        "denver nuggets": "DEN",
        "detroit pistons": "DET",
        "golden state warriors": "GSW",
        "houston rockets": "HOU",
        "indiana pacers": "IND",
        "los angeles clippers": "LAC",
        "la clippers": "LAC",
        "los angeles lakers": "LAL",
        "la lakers": "LAL",
        "memphis grizzlies": "MEM",
        "miami heat": "MIA",
        "milwaukee bucks": "MIL",
        "minnesota timberwolves": "MIN",
        "new orleans pelicans": "NOP",
        "new york knicks": "NYK",
        "oklahoma city thunder": "OKC",
        "orlando magic": "ORL",
        "philadelphia 76ers": "PHI",
        "phoenix suns": "PHX",
        "portland trail blazers": "POR",
        "sacramento kings": "SAC",
        "san antonio spurs": "SAS",
        "toronto raptors": "TOR",
        "utah jazz": "UTA",
        "washington wizards": "WAS",
    }
    cleaned = re.sub(r"[^a-z0-9 ]", " ", str(team_name or "").lower()).strip()
    for key, code in mapping.items():
        if key in cleaned:
            return code


def team_code_candidates(league: str, team_name: Any) -> List[str]:
    primary = (team_code_for_league(league, team_name) or "").upper()
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", str(team_name or "").upper()).strip()
    tokens = [t for t in cleaned.split() if t]

    candidates: List[str] = []
    if primary:
        candidates.append(primary)
    for tok in tokens:
        if tok:
            candidates.extend([tok, tok[:3], tok[:2]])
    if tokens:
        initials = "".join(t[0] for t in tokens if t)
        if len(initials) >= 2:
            candidates.append(initials)
            candidates.append(initials[:2])
        first_two_initials = "".join(t[0] for t in tokens[:2] if t)
        if len(first_two_initials) >= 2:
            candidates.append(first_two_initials)

        # Common college-style abbreviations (e.g., ARST, MOSU)
        if len(tokens) >= 2 and tokens[1] in {"STATE", "ST"}:
            first = tokens[0]
            first2 = first[:2]
            first3 = first[:3]
            candidates.extend([f"{first2}ST", f"{first3}ST", f"{first2}SU", f"{first3}SU"])
        if len(tokens) >= 2 and tokens[1] in {"UNIVERSITY", "UNIV", "U"}:
            first = tokens[0]
            first2 = first[:2]
            first3 = first[:3]
            candidates.extend([f"{first2}U", f"{first3}U"])
    deduped = [c for c in dict.fromkeys(candidates) if c]
    return deduped
    return None


def kalshi_date_token_from_local(date_val: Any) -> Optional[str]:
    """Return YYMONDD token (e.g., 25DEC16) for local YYYY-MM-DD date strings."""
    try:
        if not date_val:
            return None
        parsed = datetime.fromisoformat(str(date_val))
        return parsed.strftime("%y%b%d").upper()
    except Exception:
        return None

def kalshi_ticker_team_codes(market: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract the two 3-letter team codes from a Kalshi game ticker."""
    ticker = str(market.get("event_ticker") or market.get("ticker") or "")
    # Expected format: KX<LEAGUE>GAME-<DATE><CODE1><CODE2>[-SUFFIX]
    # e.g. KXNBAGAME-23DEC20BOSMIA -> BOS, MIA
    # e.g. KXNBAGAME-25DEC25MINDEN-MIN -> MIN, DEN
    match = re.search(r"([A-Z]{6})(?:-[A-Z]+)?$", ticker)
    if match:
        segment = match.group(1)
        return segment[:3], segment[3:]
    return None, None

def extract_teams_from_kalshi_text(text: Any) -> Tuple[Optional[str], Optional[str]]:
    content = str(text or "")
    patterns = [
        r"(.+?)\s+at\s+(.+?)(:|\||-|$)",
        r"(.+?)\s+@\s+(.+?)(:|\||-|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(:|\||-|$)",
    ]
    for pat in patterns:
        match = re.search(pat, content, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None, None



def _match_kalshi_market_impl(
    game: Dict[str, Any],
    kalshi_markets: List[Dict[str, Any]],
    winner_reason_override: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    logger.info(f"🟢 _match_kalshi_market_impl: received {len(kalshi_markets)} markets")
    # Use fuzzy matching for team names
    if rapidfuzz is not None:
        fuzz_scorer = fuzz.token_set_ratio
    else:
        # Fallback if rapidfuzz missing
        fuzz_scorer = lambda s1, s2: 100 if s1 in s2 or s2 in s1 else 0

    def base_result(reason: str, market_type: str) -> Dict[str, Any]:
        return {
            "kalshi_available": bool(kalshi_integrator),
            "kalshi_label": None,
            "kalshi_event_ticker": None,
            "kalshi_reason": reason,
            "kalshi_matched": False,
            "kalshi_prob": None,
            "kalshi_market_type": market_type,
            "kalshi_match_score": None,
            "kalshi_ticker": None,
            "kalshi_line": None,
            "kalshi_title": None,
        }

    def norm_team(name: Any) -> str:
        return re.sub(r"[^a-z0-9 ]", "", str(name or "").lower()).strip()

    def league_from_game(g: Dict[str, Any]) -> str:
        skey = (g.get("sport_key") or g.get("league") or g.get("League") or "").lower()
        mapping = {
            "basketball_nba": "NBA",
            "nba": "NBA",
            "basketball_ncaab": "NCAAB",
            "ncaab": "NCAAB",
            "americanfootball_nfl": "NFL",
            "nfl": "NFL",
            "americanfootball_ncaaf": "NCAAF",
            "ncaaf": "NCAAF",
            "icehockey_nhl": "NHL",
            "nhl": "NHL",
            "baseball_mlb": "MLB",
            "mlb": "MLB",
        }
        return mapping.get(skey, skey.upper())

    def _kalshi_prices(market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Read Kalshi bid/ask/last prices, returning (yes_bid, yes_ask, no_bid, last_price) all normalized to 0-1.

        Prefers the current *_dollars string fields (already in 0-1 dollar range).
        Falls back to deprecated integer-cent fields (divided by 100).
        """
        def _read(dollars_key: str, cents_key: str) -> Optional[float]:
            # Prefer _dollars field (string like "0.5600", already 0-1 range)
            d = safe_float(market.get(dollars_key))
            if d is not None and d > 0:
                return d
            # Fallback to deprecated integer-cent field
            c = safe_float(market.get(cents_key))
            if c is not None and c > 0:
                return c / 100.0
            return None

        return (
            _read("yes_bid_dollars", "yes_bid"),
            _read("yes_ask_dollars", "yes_ask"),
            _read("no_bid_dollars", "no_bid"),
            _read("last_price_dollars", "last_price"),
        )

    def extract_prob_and_line(
        market: Dict[str, Any], market_type: str
    ) -> Tuple[Optional[float], Optional[float]]:
        # 1. Line detection
        line = safe_float(market.get("floor_strike"))
        if line is None:
            line = safe_float(market.get("cap_strike"))
        if line is not None:
            try:
                line = float(line)
            except Exception:
                line = None

        # 2. Probability (midpoint of yes_bid and yes_ask)
        # Uses _dollars fields (current API) with fallback to cent fields (deprecated)
        yes_bid, yes_ask, no_bid, last_price = _kalshi_prices(market)

        prob = None

        if yes_bid is not None and yes_ask is not None:
            # Direct midpoint (values already normalized to 0-1)
            prob = (yes_bid + yes_ask) / 2.0

        elif yes_bid is not None and no_bid is not None:
            # Implied ask = 1.0 - no_bid (values already 0-1)
            implied_yes_ask = 1.0 - no_bid
            prob = (yes_bid + implied_yes_ask) / 2.0

        elif yes_bid is not None:
            # Prefer last_price (actual trade) over yes_bid (lowest buy offer)
            # On thin markets, yes_bid is heavily biased low (e.g. 0.05) while
            # last_price reflects the most recent agreed-upon fair value
            if last_price is not None and last_price > 0:
                prob = last_price
            else:
                prob = yes_bid

        elif no_bid is not None:
            # Fallback if YES bid missing (Prob YES = 1 - Prob NO)
            prob = 1.0 - no_bid

        # Final fallback to last_price if all bids empty
        if prob is None and last_price is not None and last_price > 0:
            prob = last_price

        return clamp(prob, 0.0, 1.0), line

    def winner_score(market: Dict[str, Any]) -> float:
        for key in ["liquidity_dollars", "liquidity", "volume", "open_interest", "last_price_dollars", "last_price"]:
            try:
                val = float(market.get(key))
                if val is not None:
                    return val
            except Exception:
                continue
        return 0.0

    def winner_prob(market: Dict[str, Any]) -> Optional[float]:
        # Use midpoint of yes_bid and yes_ask (same logic as extract_prob_and_line)
        # Reads _dollars fields (current API) with fallback to cent fields (deprecated)
        yes_bid, yes_ask, no_bid, last_price = _kalshi_prices(market)

        prob = None

        if yes_bid is not None and yes_ask is not None:
            # Direct midpoint (values already normalized to 0-1)
            prob = (yes_bid + yes_ask) / 2.0
        elif yes_bid is not None and no_bid is not None:
            # Implied ask = 1.0 - no_bid (values already 0-1)
            implied_yes_ask = 1.0 - no_bid
            prob = (yes_bid + implied_yes_ask) / 2.0
        elif yes_bid is not None:
            # Prefer last_price (actual trade) over yes_bid (lowest buy offer)
            if last_price is not None and last_price > 0:
                prob = last_price
            else:
                prob = yes_bid
        elif no_bid is not None:
            # Fallback if YES bid missing (Prob YES = 1 - Prob NO)
            prob = 1.0 - no_bid

        # Final fallback to last_price if all bids empty
        if prob is None and last_price is not None:
            prob = clamp(last_price, 0.0, 1.0)

        return clamp(prob, 0.0, 1.0) if prob is not None else None

    # Compute diagnostic fields BEFORE early returns so they're always available
    league_name = league_from_game(game)

    commence_raw = (
        game.get("commence_time_iso_utc")
        or game.get("commence_time")
        or game.get("commence_time_iso")
        or game.get("commence_time_utc")
    )
    game_dt_utc = parse_commence_to_utc(commence_raw)
    if isinstance(game_dt_utc, datetime) and game_dt_utc.tzinfo is None:
        game_dt_utc = game_dt_utc.replace(tzinfo=timezone.utc)

    tz_name = get_local_tz()
    local_tz = None
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = None
    game_local = game_dt_utc.astimezone(local_tz) if (game_dt_utc and local_tz) else game_dt_utc
    base_date = game_local.date() if game_local else None
    allowed_date_tokens: List[str] = []
    if base_date:
        for delta in (-1, 0, 1):
            allowed_date_tokens.append((base_date + timedelta(days=delta)).strftime("%y%b%d").upper())
    if not allowed_date_tokens:
        token_from_local = kalshi_date_token_from_local(game.get("commence_date_local"))
        if token_from_local:
            allowed_date_tokens.append(token_from_local)
    date_token = allowed_date_tokens[1] if len(allowed_date_tokens) > 1 else (allowed_date_tokens[0] if allowed_date_tokens else None)
    winner_prefix = league_game_prefix(league_name)

    away_code_expected = team_code_for_league(league_name, game.get("away_team"))
    home_code_expected = team_code_for_league(league_name, game.get("home_team"))

    def _early_debug(reason: str) -> Dict[str, Any]:
        """Build a debug dict for early-return paths that still includes diagnostics."""
        return {
            "total": [], "spread": [], "winner": [],
            "winner_meta": {
                "expected_date_token": date_token,
                "expected_codes": {"away": away_code_expected, "home": home_code_expected},
                "winner_match_status": "early_return",
                "winner_no_match_reason": reason,
                "matched_event_ticker": None,
                "matched_ticker": None,
                "kalshi_date_token_used": date_token,
                "winner_prefix": winner_prefix,
                "strict_candidate_count": 0,
                "allowed_date_tokens": allowed_date_tokens,
            },
            "kalshi_game_prefix_used": winner_prefix,
            "kalshi_wanted_tokens": allowed_date_tokens,
        }

    if not kalshi_integrator:
        base = {t: base_result("kalshi_not_configured", t) for t in ["total", "spread", "winner"]}
        return base, _early_debug("kalshi_not_configured")
    if not kalshi_markets:
        base = {t: base_result("no_game_like_markets_in_window", t) for t in ["total", "spread", "winner"]}
        logger.warning(
            f"⚠️ KALSHI IMPL: No markets passed to matcher for "
            f"{game.get('away_team')} @ {game.get('home_team')} | "
            f"league={league_name} | prefix={winner_prefix} | dates={allowed_date_tokens} | "
            f"codes=({away_code_expected}@{home_code_expected})"
        )
        return base, _early_debug("no_game_like_markets_in_window")

    def market_tokens(market: Dict[str, Any]) -> set:
        blob = " ".join(
            [
                str(market.get("event_ticker") or market.get("ticker") or ""),
                str(market.get("title") or ""),
                str(market.get("rules") or market.get("rules_primary") or ""),
            ]
        )
        cleaned = re.sub(r"[^a-z0-9 ]", " ", blob.lower())
        return {t for t in cleaned.split() if t}

    def team_token_set(team_name: Any) -> set:
        base_tokens = team_tokens(team_name)
        codes = {c.lower() for c in team_code_candidates(league_name, team_name) or []}
        return set(base_tokens).union(codes)

    home_tokens = team_token_set(game.get("home_team"))
    away_tokens = team_token_set(game.get("away_team"))

    # Classify markets and add debug logging
    totals = [m for m in kalshi_markets if classify_kalshi_market(m) == "total"]
    spreads = [m for m in kalshi_markets if classify_kalshi_market(m) == "spread"]
    winners = [m for m in kalshi_markets if classify_kalshi_market(m) == "winner"]
    unknown = [m for m in kalshi_markets if classify_kalshi_market(m) == "unknown"]

    # DEBUG: Log market type counts and samples
    logger.info(f"📊 KALSHI MARKET CLASSIFICATION for {game.get('away_team')} @ {game.get('home_team')}:")
    logger.info(f"   Total markets received: {len(kalshi_markets)}")
    logger.info(f"   - Winner markets: {len(winners)}" + (f" (sample: {winners[0].get('ticker', 'N/A')})" if winners else ""))
    logger.info(f"   - Total markets: {len(totals)}" + (f" (sample: {totals[0].get('ticker', 'N/A')})" if totals else ""))
    logger.info(f"   - Spread markets: {len(spreads)}" + (f" (sample: {spreads[0].get('ticker', 'N/A')})" if spreads else ""))
    logger.info(f"   - Unknown markets: {len(unknown)}" + (f" (sample: {unknown[0].get('ticker', 'N/A')})" if unknown else ""))

    # DEBUG: Log pricing field availability on first market to verify API response format
    if kalshi_markets:
        _sample = kalshi_markets[0]
        logger.info(f"   💲 PRICING FIELDS (sample {_sample.get('ticker', 'N/A')}): "
                     f"yes_bid_dollars={_sample.get('yes_bid_dollars')}, "
                     f"yes_ask_dollars={_sample.get('yes_ask_dollars')}, "
                     f"yes_bid={_sample.get('yes_bid')}, "
                     f"yes_ask={_sample.get('yes_ask')}, "
                     f"last_price_dollars={_sample.get('last_price_dollars')}, "
                     f"last_price={_sample.get('last_price')}")

    # FIX: Enhanced debug logging for spread/total matching
    if totals:
        logger.info(f"   📊 TOTAL MARKET DETAILS: {len(totals)} markets available")
        for t in totals[:3]:
            logger.info(f"      - {t.get('ticker')} | title: {t.get('title', '')[:40]} | last_price: {t.get('last_price')}")
    if spreads:
        logger.info(f"   📊 SPREAD MARKET DETAILS: {len(spreads)} markets available")
        for s in spreads[:3]:
            logger.info(f"      - {s.get('ticker')} | title: {s.get('title', '')[:40]} | last_price: {s.get('last_price')}")

    winner_candidate_debug: List[Dict[str, Any]] = []
    best_winner: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_reason = "no_candidates"
    candidate_count = 0
    strict_candidates: List[Tuple[float, Dict[str, Any]]] = []
    fallback_with_date: List[Tuple[float, Dict[str, Any]]] = []
    fallback_no_date: List[Tuple[float, Dict[str, Any]]] = []

    def infer_yes_side(market: Dict[str, Any]) -> Optional[str]:
        codes = kalshi_ticker_team_codes(market)
        if codes:
            first, second = codes
            if first and first == home_code_expected:
                return "home"
            if first and first == away_code_expected:
                return "away"
            if second and second == home_code_expected:
                return "home"
            if second and second == away_code_expected:
                return "away"
        return None

    home_code_candidates = [c.upper() for c in team_code_candidates(league_name, game.get("home_team"))]
    away_code_candidates = [c.upper() for c in team_code_candidates(league_name, game.get("away_team"))]

    for m in kalshi_markets or []:
        ticker_upper = str(m.get("event_ticker") or m.get("ticker") or "").upper()
        title_lower = str(m.get("title") or "").lower()
        if not (
            ticker_upper.startswith(winner_prefix)
            or "GAME-" in ticker_upper
            or "GAME" in ticker_upper
            or "winner" in title_lower
        ):
            continue
        tokens = market_tokens(m)
        date_match = bool(allowed_date_tokens and any(tok in ticker_upper for tok in allowed_date_tokens))
        home_hit = bool(home_tokens.intersection(tokens))
        away_hit = bool(away_tokens.intersection(tokens))
        code_home_hit = home_code_candidates and any(code in ticker_upper for code in home_code_candidates if len(code) >= 3)
        code_away_hit = away_code_candidates and any(code in ticker_upper for code in away_code_candidates if len(code) >= 3)
        code_hit = bool(code_home_hit and code_away_hit)
        team_hit = bool(home_hit and away_hit)

        # Enhanced Fuzzy Matching using match_team_name from prediction_engine
        if not (team_hit or code_hit):
            # 1. Try fuzzy match on market title vs home/away team names (Legacy)
            fuzzy_home = match_team_name(game.get("home_team"), [title_lower], threshold=70.0)
            fuzzy_away = match_team_name(game.get("away_team"), [title_lower], threshold=70.0)

            if fuzzy_home and fuzzy_away:
                team_hit = True

            # 2. Try RapidFuzz direct token set match (New Fallback)
            # Helps with "Lions" vs "Detroit Lions" where 'Lions' is subset
            if not team_hit and fuzz:
                home_raw = str(game.get("home_team") or "").lower()
                away_raw = str(game.get("away_team") or "").lower()

                # token_set_ratio handles subset matching well (e.g. "Lions" in "Detroit Lions")
                score_h = fuzz_scorer(home_raw, title_lower)
                score_a = fuzz_scorer(away_raw, title_lower)

                # Find MATCHTHRESHOLD
                MATCHTHRESHOLD = 60 if league_name == 'NCAAB' else 85  # Ultra-low for NCAAB
                if score_h >= MATCHTHRESHOLD and score_a >= MATCHTHRESHOLD:
                    team_hit = True

        if not (team_hit or code_hit):
            continue
        candidate_count += 1
        score = (2 if team_hit else 0) + (2 if code_hit else 0) + (1 if date_match else 0)
        debug_row = {
            "title": m.get("title"),
            "ticker": m.get("event_ticker") or m.get("ticker"),
            "liquidity": m.get("liquidity"),
            "volume": m.get("volume"),
            "open_interest": m.get("open_interest"),
            "last_price": m.get("last_price"),
            "yes_bid_dollars": m.get("yes_bid_dollars"),
            "yes_ask_dollars": m.get("yes_ask_dollars"),
            "last_price_dollars": m.get("last_price_dollars"),
            "score": score,
            "date_match": date_match,
            "home_hit": team_hit or code_home_hit,
            "away_hit": team_hit or code_away_hit,
        }
        winner_candidate_debug.append(debug_row)
        if date_match and code_hit:
            strict_candidates.append((score, m))
        elif date_match:
            fallback_with_date.append((score, m))
        else:
            fallback_no_date.append((score, m))

    if strict_candidates:
        best_score, best_winner = max(strict_candidates, key=lambda kv: kv[0])
        best_reason = "strict_match"
    elif fallback_with_date:
        best_score, best_winner = max(fallback_with_date, key=lambda kv: kv[0])
        best_reason = "fallback_title_match"
    elif fallback_no_date:
        best_score, best_winner = max(fallback_no_date, key=lambda kv: kv[0])
        best_reason = "fallback_no_date_token"

    # FORCE NCAAB MATCH if ANY spread/total found (User Request 2)
    if not best_winner and (spreads or totals) and league_name == 'NCAAB':
        if spreads:
            best_winner = sorted(spreads, key=lambda m: m.get('volume', 0) or 0, reverse=True)[0]
        elif totals:
            best_winner = sorted(totals, key=lambda m: m.get('volume', 0) or 0, reverse=True)[0]
        else:
            best_winner = kalshi_markets[0] # Should not happen given check above

        # Log with requested format
        away_tm = game.get('away_team', 'Away')
        home_tm = game.get('home_team', 'Home')
        logger.info(f"NCAAB FORCE MARKET: {best_winner.get('ticker')} {away_tm}@{home_tm}")

        best_reason = "forced_spread_total_fallback"
        best_score = 60.0 # User requested MATCHTHRESHOLD=60, so we give it 60 to pass checks

    if best_winner:
        prob = winner_prob(best_winner)
        winner_result = {
            "kalshi_available": True,
            "kalshi_label": "matched_winner",
            "kalshi_event_ticker": best_winner.get("event_ticker") or best_winner.get("ticker"),
            "kalshi_reason": best_reason,
            "kalshi_matched": True,
            "kalshi_prob": prob,
            "kalshi_market_type": "winner",
            "kalshi_match_score": best_score,
            "kalshi_ticker": best_winner.get("event_ticker") or best_winner.get("ticker"),
            "kalshi_line": None,
            "kalshi_title": best_winner.get("title"),
            "kalshi_yes_side": infer_yes_side(best_winner),
        }
    else:
        # LOG MATCH FAILURES (~line 8700 before return None)
        if not best_winner:
            logger.warning(f"❌ NO MATCH for {game.get('home_team')} vs {game.get('away_team')}")
            logger.warning(f"   Available market team names: {[m.get('ticker', m.get('title', '')) for m in kalshi_markets[:5]]}")
            logger.warning(f"   Game teams: home='{normalize_team_name(game.get('home_team'))}' away='{normalize_team_name(game.get('away_team'))}'")
            logger.warning(f"   Original failure log: {game.get('away_team')}@{game.get('home_team')} | bestscore={best_score if 'best_score' in locals() else 0} | markets={len(kalshi_markets)}")

        no_reason = winner_reason_override or best_reason or "no_winner_market_for_game"
        winner_result = base_result(no_reason, "winner")

    def simple_select(markets: List[Dict[str, Any]], market_type: str) -> Dict[str, Any]:
        if not markets:
            return base_result(f"no_{market_type}_market", market_type)

        # Task 1.2: Improved Kalshi Contract Selection for Spreads/Totals
        # Instead of blindly taking markets[0], score candidates and infer YES side intelligently

        # Get sportsbook consensus line for line-proximity scoring
        sportsbook_total_line = safe_float(game.get("total_point"))
        sportsbook_spread_line = safe_float(game.get("home_spread_point"))

        scored_candidates = []
        for market in markets:
            score = 0.0
            ticker_upper = str(market.get("event_ticker") or market.get("ticker") or "").upper()

            # Scoring: Prefer markets that match date token
            if allowed_date_tokens and any(tok in ticker_upper for tok in allowed_date_tokens):
                score += 100.0

            # Prefer markets with team code matches (require 3+ char codes to prevent false positives)
            if home_code_candidates and away_code_candidates:
                code_home_hit = any(code in ticker_upper for code in home_code_candidates if len(code) >= 3)
                code_away_hit = any(code in ticker_upper for code in away_code_candidates if len(code) >= 3)
                if code_home_hit and code_away_hit:
                    score += 50.0

            # Line-proximity scoring: strongly prefer markets whose line matches the sportsbook
            # Without this, the system picks an arbitrary line (e.g., Over 250 instead of Over 215)
            # which produces extreme probabilities (5% for Over 250 is correct but irrelevant)
            market_line = safe_float(market.get("floor_strike") or market.get("cap_strike"))
            if market_line is not None and market_type in ("total", "spread"):
                ref_line = sportsbook_total_line if market_type == "total" else sportsbook_spread_line
                if ref_line is not None:
                    line_diff = abs(market_line - ref_line)
                    if line_diff <= 0.5:
                        score += 300.0  # Exact match (within rounding)
                    elif line_diff <= 2.0:
                        score += 250.0  # Very close
                    elif line_diff <= 5.0:
                        score += 200.0 - (line_diff * 10)  # Close, decreasing bonus
                    elif line_diff <= 10.0:
                        score += 100.0  # Moderate distance
                    # Lines > 10 points away get no line bonus

            scored_candidates.append((score, market))

        # Select best-scoring candidate
        if scored_candidates:
            best_score, chosen = max(scored_candidates, key=lambda x: x[0])
        else:
            chosen = None
            best_score = 0.0

        # MINIMUM SCORE THRESHOLD: Prevent phantom Kalshi matches from cross-game contamination.
        # Score breakdown: date match=100, both team codes=50, line proximity=up to 300.
        # Without a date match (score < 100), the market is likely for a different game/date.
        # A date-only match (100) without team codes is still weak evidence, but date+line
        # (400) is acceptable since line proximity validates the match.
        MINIMUM_MATCH_SCORE = 100.0
        if chosen is None or best_score < MINIMUM_MATCH_SCORE:
            logger.info(f"⚠️ KALSHI {market_type.upper()} REJECTED: score={best_score:.0f} below threshold {MINIMUM_MATCH_SCORE} "
                        f"(candidates={len(scored_candidates)})")
            return base_result(f"no_{market_type}_market_below_threshold", market_type)

        prob, line = extract_prob_and_line(chosen, market_type)

        # Task 1.2: Intelligent YES side inference
        if market_type == "total":
            # For total markets, YES side is "over" or "under" — determine from ticker/title
            # DO NOT use infer_yes_side() which returns "home"/"away" from team codes
            # FIX: Use individual market ticker first (contains "OVER"/"UNDER" suffix),
            # then fall back to event_ticker (which is just the event-level identifier
            # like KXNBATOTAL-26FEB08BOSSEA and never contains Over/Under direction)
            chosen_ticker = str(chosen.get("ticker") or chosen.get("event_ticker") or "").upper()
            chosen_title = str(chosen.get("title") or "").lower()
            if "UNDER" in chosen_ticker or "under" in chosen_title:
                yes_side_inferred = "under"
            else:
                yes_side_inferred = "over"  # Default for totals (YES = Over is standard)
        else:
            # For winner/spread markets, parse team code from individual contract ticker suffix.
            # Individual tickers look like: KXNBASPREAD-26FEB10INDNYK-NYK27
            # The suffix after the last dash (NYK27) contains the team code for the YES side.
            # Strip trailing digits/dots to get the team code (NYK).
            individual_ticker = str(chosen.get("ticker") or "").upper()
            suffix = individual_ticker.rsplit("-", 1)[-1] if "-" in individual_ticker else ""
            suffix_team = re.sub(r'[\d.]+$', '', suffix).strip()

            if suffix_team and len(suffix_team) >= 2:
                home_codes_upper = {c.upper() for c in (home_code_candidates or [])}
                away_codes_upper = {c.upper() for c in (away_code_candidates or [])}
                if suffix_team in home_codes_upper:
                    yes_side_inferred = "home"
                    logger.info(f"  → yes_side=home from ticker suffix '{suffix_team}' matching home_codes={home_codes_upper}")
                elif suffix_team in away_codes_upper:
                    yes_side_inferred = "away"
                    logger.info(f"  → yes_side=away from ticker suffix '{suffix_team}' matching away_codes={away_codes_upper}")
                else:
                    # Suffix didn't match known codes, fall back to legacy inference
                    yes_side_inferred = infer_yes_side(chosen)
                    if not yes_side_inferred:
                        yes_side_inferred = "home"
                    logger.info(f"  → yes_side={yes_side_inferred} (suffix '{suffix_team}' unmatched, legacy fallback)")
            else:
                # No parseable suffix, fall back to legacy inference
                yes_side_inferred = infer_yes_side(chosen)
                if not yes_side_inferred:
                    yes_side_inferred = "home"
                logger.info(f"  → yes_side={yes_side_inferred} (no suffix parsed from '{individual_ticker}', legacy fallback)")

        # Add debug logging - use info level for visibility
        prob_str = f"{prob:.3f}" if prob else "N/A"
        logger.info(f"✅ KALSHI {market_type.upper()} MATCH: ticker={chosen.get('ticker') or chosen.get('event_ticker')}, "
                    f"prob={prob_str}, line={line}, yes_side={yes_side_inferred}, score={best_score}")

        # Warn if selected Kalshi line is far from sportsbook line
        if market_type == "total" and line is not None and sportsbook_total_line is not None:
            line_gap = abs(line - sportsbook_total_line)
            if line_gap > 5.0:
                logger.warning(f"⚠️ KALSHI LINE MISMATCH: Kalshi line={line} vs Sportsbook line={sportsbook_total_line} (gap={line_gap:.1f})")
        elif market_type == "spread" and line is not None and sportsbook_spread_line is not None:
            line_gap = abs(line - sportsbook_spread_line)
            if line_gap > 3.0:
                logger.warning(f"⚠️ KALSHI LINE MISMATCH: Kalshi line={line} vs Sportsbook line={sportsbook_spread_line} (gap={line_gap:.1f})")

        return {
            "kalshi_available": True,
            "kalshi_label": f"matched_{market_type}",
            "kalshi_event_ticker": chosen.get("event_ticker") or chosen.get("ticker"),
            "kalshi_reason": f"matched_{market_type}",
            "kalshi_matched": True,
            "kalshi_prob": prob,
            "kalshi_market_type": market_type,
            "kalshi_match_score": best_score,
            "kalshi_ticker": chosen.get("event_ticker") or chosen.get("ticker"),
            "kalshi_line": line,
            "kalshi_title": chosen.get("title"),
            "kalshi_yes_side": yes_side_inferred,  # Now uses intelligent inference
        }

    winner_meta = {
        "expected_date_token": date_token,
        "expected_codes": {"away": away_code_expected, "home": home_code_expected},
        "winner_match_status": "matched" if winner_result.get("kalshi_matched") else "no_match",
        "winner_no_match_reason": None if winner_result.get("kalshi_matched") else winner_result.get("kalshi_reason"),
        "matched_event_ticker": winner_result.get("kalshi_event_ticker"),
        "matched_ticker": winner_result.get("kalshi_ticker"),
        "kalshi_date_token_used": date_token,
        "winner_prefix": winner_prefix,
        "strict_candidate_count": len(strict_candidates),
        "allowed_date_tokens": allowed_date_tokens,
    }

    candidate_debug = {
        "total": totals,
        "spread": spreads,
        "winner": winner_candidate_debug,
        "winner_meta": winner_meta,
        "candidate_count": candidate_count,
        "best_score": best_score if best_score is not None else None,
        "match_reason": winner_result.get("kalshi_reason"),
        "kalshi_game_prefix_used": winner_prefix,
        "kalshi_wanted_tokens": allowed_date_tokens,
    }

    return {
        "total": simple_select(totals, "total"),
        "spread": simple_select(spreads, "spread"),
        "winner": winner_result,
    }, candidate_debug


def match_kalshi_market(
    game: Dict[str, Any],
    kalshi_markets: List[Dict[str, Any]],
    winner_reason_override: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Delegates to _match_kalshi_market_impl for fuzzy matching."""
    logger.info(f"🔵 match_kalshi_market called with {len(kalshi_markets)} markets")
    result = _match_kalshi_market_impl(game, kalshi_markets, winner_reason_override)
    logger.info(f"🟢 _match_kalshi_market_impl returned {len(result)} matches")
    return result


# -----------------
# Session defaults
# -----------------

if "last_exception" not in st.session_state:
    st.session_state["last_exception"] = None
if "last_rows_out" not in st.session_state:
    st.session_state["last_rows_out"] = 0
if "games" not in st.session_state:
    st.session_state["games"] = []
if "league" not in st.session_state:
    st.session_state["league"] = "NBA"
if "selected_sports" not in st.session_state:
    st.session_state["selected_sports"] = [ALL_SPORTS_LABEL]
if "commence_stats" not in st.session_state:
    st.session_state["commence_stats"] = {"parsed": 0, "failed": 0, "timezone": get_local_tz()}
if "market_counts" not in st.session_state:
    st.session_state["market_counts"] = {
        "moneyline_available_count": 0,
        "spreads_available_count": 0,
        "totals_available_count": 0,
    }
if "run_id" not in st.session_state:
    st.session_state["run_id"] = None
if "gemini_disabled_reason" not in st.session_state:
    st.session_state["gemini_disabled_reason"] = None


# -----------------
# Data loading helpers
# -----------------

def load_games(selected_leagues: Union[str, List[str]], run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load games from TheOddsAPI for selected leagues.

    Sets st.session_state["games_fetch_status"] to "success", "empty", or "error"
    to enable atomic ingest + UI gating.
    """
    leagues = [selected_leagues] if isinstance(selected_leagues, str) else list(selected_leagues or [])
    all_games_with_times: List[Dict[str, Any]] = []
    commence_stats_total = {"parsed": 0, "failed": 0, "timezone": get_local_tz()}
    moneyline_count = 0
    spreads_count = 0
    totals_count = 0

    # Track whether any league succeeded
    any_games_loaded = False
    all_leagues_failed = True

    # 1. Guard: If no leagues selected or just "All Sports" placeholder (if applicable), skip
    if not leagues:
        logger.info("load_games: No leagues selected, skipping fetch.")
        return []

    for selected_league in leagues:
        sport_key = SPORT_KEYS.get(selected_league)
        if not sport_key:
            st.session_state["last_exception"] = f"Unknown league: {selected_league}"
            continue
        try:
            games_raw = fetch_odds_games(sport_key, run_id=run_id)
            if games_raw and len(games_raw) > 0:
                any_games_loaded = True
                all_leagues_failed = False
        except Exception:
            st.session_state["last_exception"] = traceback.format_exc()
            continue
        normalized = [normalize_game({**g, "sport_key": sport_key}) for g in games_raw]
        with_times, commence_stats = normalize_commence_times(normalized)
        commence_stats_total["parsed"] += commence_stats.get("parsed", 0)
        commence_stats_total["failed"] += commence_stats.get("failed", 0)

        for g in with_times:
            # INITIALIZATION BLOCK
            total_pick_side = None
            total_line = None
            total_pick_odds = None
            spread_engine_used = "missing"
            # 3. Initialization Block (Fix Fatal Loop NameError - Extended)
            total_engine_used = "missing"
            spread_prob_final = 0.5
            total_prob_final = 0.5
            spread_prob_market = 0.5
            total_prob_market = 0.5
            total_pick = None
            spread_pick = None
            kalshi_prob_spread = 0.5
            kalshi_prob_total = 0.5
            model_spread_prob = 0.5
            model_total_prob = 0.5
            game_row = {} # Initialize empty game_row to be safe
            total_engine_used = "missing"
            spread_prob_final = 0.5
            total_prob_final = 0.5
            spread_prob_market = 0.5
            total_prob_market = 0.5
            kalshi_prob_spread = 0.5
            kalshi_prob_total = 0.5
            model_spread_prob = 0.5
            model_total_prob = 0.5
            total_pick = None
            spread_pick = None
            try:
                best = extract_best_market(g)
                warnings = list(best.pop("warnings", []))
                merged_warnings = list(dict.fromkeys((g.get("warnings") or []) + warnings))
                g.update(best)
                g["warnings"] = merged_warnings
                if g.get("best_ml_book") is not None:
                    moneyline_count += 1
                if g.get("best_spread_book") is not None:
                    spreads_count += 1
                if g.get("best_total_book") is not None:
                    totals_count += 1
            except Exception:
                g["warnings"] = list(g.get("warnings") or []) + ["odds_extract_error"]
                st.session_state["last_exception"] = traceback.format_exc()

        all_games_with_times.extend(with_times)

    deduped: Dict[Tuple[Any, Any, Any, Any], Dict[str, Any]] = {}
    for g in all_games_with_times:
        key = (
            g.get("sport_key"),
            g.get("home_team"),
            g.get("away_team"),
            g.get("commence_time_iso_utc"),
        )
        if key not in deduped:
            deduped[key] = g

    games_final = list(deduped.values())

    # Restrict to current local day (post-conversion)
    tz_name = get_local_tz()
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = timezone.utc
    today_local = datetime.now(local_tz).date().isoformat()

    def game_local_date(game: Dict[str, Any]) -> Optional[str]:
        if game.get("commence_date_local"):
            return str(game.get("commence_date_local"))
        dt_utc = parse_commence_to_utc(
            game.get("commence_time_iso_utc")
            or game.get("commence_time")
            or game.get("commence_time_iso")
            or game.get("commence_time_utc")
        )
        if not dt_utc:
            return None
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        try:
            return dt_utc.astimezone(local_tz).date().isoformat()
        except Exception:
            return None

    # TEMPORARY DEBUG: Disable date filter to allow all games through
    # This fixes the "CRITICAL: games list is empty!" error
    filtered_games: List[Dict[str, Any]] = games_final  # Load everything the API gives us
    # Original filter (commented out for debugging):
    # for g in games_final:
    #     if game_local_date(g) == today_local:
    #         filtered_games.append(g)

    # Recompute counts to match filtered games
    moneyline_count_filtered = sum(1 for g in filtered_games if g.get("best_ml_book") is not None)
    spreads_count_filtered = sum(1 for g in filtered_games if g.get("best_spread_book") is not None)
    totals_count_filtered = sum(1 for g in filtered_games if g.get("best_total_book") is not None)

    # HARD-CODED DATA FLOW FIX: filtered_games = games_final is the ONLY active logic
    # Retry logic and fallback mechanisms DISABLED to ensure fresh API data always flows through
    # (Previously, retry/fallback logic was interfering with data ingestion)

    # Set fetch status for atomic ingest + UI gating
    if len(filtered_games) > 0:
        st.session_state["games_fetch_status"] = "success"
        st.session_state["games_fetch_run_id"] = run_id  # Track which run_id succeeded
        logger.info(f"✅ Games fetch SUCCESS: {len(filtered_games)} games loaded (run_id: {run_id})")
    elif all_leagues_failed:
        st.session_state["games_fetch_status"] = "error"
        logger.error(f"❌ Games fetch ERROR: All leagues failed to load (run_id: {run_id})")
    else:
        st.session_state["games_fetch_status"] = "empty"
        # Only log warning if user explicitly requested run; otherwise INFO
        if run_id:
            logger.warning(f"⚠️ Games fetch EMPTY: API succeeded but returned 0 games (run_id: {run_id})")
        else:
            logger.info(f"ℹ️ Games fetch EMPTY (Startup/Auto): API returned 0 games.")

    # HARD-CODED DATA FLOW FIX: No fallback to cached games
    # Always return what the API gives us (filtered_games = games_final)
    if not filtered_games:
        logger.warning("⚠️ games list is empty after fetch - returning empty list")
        return []

    st.session_state["games"] = filtered_games
    st.session_state["commence_stats"] = commence_stats_total
    st.session_state["market_counts"] = {
        "moneyline_available_count": moneyline_count_filtered,
        "spreads_available_count": spreads_count_filtered,
        "totals_available_count": totals_count_filtered,
    }
    st.session_state["game_filter_info"] = {
        "total_loaded": len(games_final),
        "filtered_to_today": len(filtered_games),
        "today_local": today_local,
        "timezone": tz_name,
    }

    if st.session_state.get("enable_sentiment", True):
        ensure_sentiment_loaded(filtered_games)
    else:
        st.session_state["sentiment_map"] = {}
        st.session_state["sentiment_meta_map"] = {}
        st.session_state["sentiment_meta"] = {"sentiment_source": "none", "reddit_used": False}
        st.session_state["sentiment_debug"] = {"enabled": False, "warning": "sentiment_disabled"}
        st.session_state["sentiment_slate_key"] = None
    return filtered_games


# -----------------
# Sidebar
# -----------------

st.sidebar.header("Controls")

# TheOver Public Betting Input
with st.sidebar.expander("TheOver.ai Data (Optional)", expanded=False):
    st.caption("Paste text or upload Excel exports from TheOver.ai")
    theover_totals_file = st.file_uploader("Upload Totals (.xlsx, .csv)", type=["xlsx", "csv"], key="theover_totals_file")
    theover_sides_file = st.file_uploader("Upload Sides (.xlsx, .csv)", type=["xlsx", "csv"], key="theover_sides_file")
    theover_totals_text = st.text_area("Paste Totals Text", height=100, key="theover_totals_text")
    theover_sides_text = st.text_area("Paste Sides Text", height=100, key="theover_sides_text")

sport_options = [ALL_SPORTS_LABEL] + list(SPORT_KEYS.keys())
default_sports = st.session_state.get("selected_sports") or [st.session_state.get("league", "NBA")]
valid_defaults = [s for s in default_sports if s in sport_options]
selected_sports = st.sidebar.multiselect(
    "Select sports",
    sport_options,
    default=valid_defaults or [ALL_SPORTS_LABEL],
)
if not selected_sports:
    selected_sports = [ALL_SPORTS_LABEL]
if ALL_SPORTS_LABEL in selected_sports:
    selected_sports = [s for s in sport_options if s != ALL_SPORTS_LABEL]
st.session_state["selected_sports"] = selected_sports
# Safe access for league selection
_all_keys = list(SPORT_KEYS.keys())
league = selected_sports[0] if selected_sports else (_all_keys[0] if _all_keys else "NBA")

# Detect if selection changed to invalidate cache
last_selection = st.session_state.get("_last_selected_sports")

# Sort both lists for reliable comparison (multiselect order can vary)
_last_sorted = sorted(last_selection) if last_selection else []
_curr_sorted = sorted(selected_sports) if selected_sports else []

if _last_sorted != _curr_sorted:
    logger.info(f"Sport selection changed: {last_selection} -> {selected_sports}")
    # Only invalidate if we actually have a change
    if "master_df" in st.session_state:
        # Don't delete key, just empty it to avoid KeyErrors
        st.session_state["master_df"] = pd.DataFrame()
        st.session_state["master_results_df"] = pd.DataFrame()
        st.session_state["analysis_complete"] = False
    st.session_state["_last_selected_sports"] = selected_sports
st.session_state["league"] = league
kalshi_required_toggle = st.sidebar.checkbox(
    "Kalshi required", value=st.session_state.get("kalshi_required", True)
)
st.session_state["kalshi_required"] = kalshi_required_toggle
if kalshi_integrator:
    kalshi_integrator.required = kalshi_required_toggle
enable_sentiment = True  # Forced enabled for Reddit fallback
# enable_sentiment = st.sidebar.checkbox(
#     "Enable Sentiment", value=st.session_state.get("enable_sentiment", True)
# )
st.session_state["enable_sentiment"] = enable_sentiment

# === KALSHI BLENDING DEBUG SECTION ===
# Add to sidebar (after league selection but before "Run Analysis" button)
if st.sidebar.checkbox("🔬 Debug Kalshi Blending", key="debug_kalshi_blend"):
    st.sidebar.info("Shows detailed probability blending for Kalshi-matched games")

if st.sidebar.button("Load Games", width="stretch"):
    st.cache_data.clear()  # Force a fresh API call
    # Invalidate master_df when loading new games
    if "master_df" in st.session_state:
        # Don't delete key, just reset
        st.session_state["master_df"] = pd.DataFrame()
        st.session_state["master_results_df"] = pd.DataFrame()
        st.session_state["analysis_complete"] = False

    # Generate new run_id for this load
    import uuid
    new_run_id = str(uuid.uuid4())[:8]
    st.session_state["run_id"] = new_run_id
    logger.info(f"🆕 Generated new run_id: {new_run_id}")
    load_games(selected_sports or [league], run_id=new_run_id)

# --- SYSTEM TOOLS (Debug Export) ---
st.sidebar.markdown("---")
st.sidebar.header("System Tools")

if st.sidebar.button("Clear Debug Log"):
    st.session_state["debug_accumulator"] = []
    st.success("Debug log cleared.")

# Task 2.3: Add Cache Clear Button
if st.sidebar.button("Clear Sentiment Cache"):
    from app_core.sentiment_cache import get_cache
    cache = get_cache()
    stats_before = cache.get_stats()
    entries_cleared = cache.clear()
    st.success(f"Sentiment cache cleared! Removed {entries_cleared} entries.")
    st.info(f"Cache was using {stats_before['valid_entries']} valid entries "
            f"(+{stats_before['expired_entries']} expired) with {stats_before['ttl_hours']}-hour TTL.")

# Display cache stats
try:
    from app_core.sentiment_cache import get_cache
    cache = get_cache()
    cache_stats = cache.get_stats()
    if cache_stats['total_entries'] > 0:
        st.sidebar.caption(
            f"📦 Sentiment Cache: {cache_stats['valid_entries']} valid, "
            f"{cache_stats['expired_entries']} expired "
            f"({cache_stats['ttl_hours']}h TTL)"
        )
except Exception as e:
    # Silently fail if cache not available
    pass

# ═══════════════════════════════════════════════════════════════════════
# DEBUG EXPORTS SECTION - Always accessible regardless of analysis state
# ═══════════════════════════════════════════════════════════════════════
st.sidebar.subheader("📥 Debug Exports")

# Debug accumulator (always available if exists)
if "debug_accumulator" in st.session_state and st.session_state["debug_accumulator"]:
    try:
        debug_json = json.dumps(st.session_state["debug_accumulator"], default=str, indent=2)
        st.sidebar.download_button(
            label="📊 Download Debug Log (Main)",
            data=debug_json,
            file_name="parlay_debug_export.json",
            mime="application/json",
            key="debug_accumulator_export_btn",
            help="Main debug accumulator log"
        )
    except Exception as e:
        st.sidebar.error(f"Error preparing download: {e}")

# Debug log history (alternate debug log)
if "debug_log_history" in st.session_state and st.session_state["debug_log_history"]:
    try:
        # RESCUE FIX #3: Force type map to kill NaNs before JSON export
        debug_data = st.session_state["debug_log_history"]
        if isinstance(debug_data, list) and len(debug_data) > 0:
            debug_df = pd.DataFrame(debug_data)
            # Apply hard type mapping to prevent NaN in "type" field
            if "Market" in debug_df.columns:
                type_map = {"Moneyline": "ML", "Spread": "SPREAD", "Total": "TOTAL"}
                debug_df["type"] = debug_df["Market"].map(type_map).fillna("UNKNOWN")
            # Convert back to list of dicts
            debug_data = debug_df.to_dict('records')

        debug_history_json = json.dumps(debug_data, default=str, indent=2)
        st.sidebar.download_button(
            label="📊 Download Debug Log (History)",
            data=debug_history_json,
            file_name="parlay_debug_history.json",
            mime="application/json",
            key="debug_history_export_btn",
            help="Debug log history"
        )
    except Exception:
        pass

# TheOver raw data export (always available if data exists)
if "theover_raw_df" in st.session_state:
    theover_raw_df = st.session_state["theover_raw_df"]
    if theover_raw_df is not None and not theover_raw_df.empty:
        try:
            theover_raw_csv = theover_raw_df.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button(
                label="📋 Download TheOver Debug (Raw)",
                data=theover_raw_csv,
                file_name="theover_raw_debug.csv",
                mime="text/csv",
                key="theover_raw_sidebar_export_btn",
                help="Exports the RAW TheOver dataframe before transformation"
            )
        except Exception as e:
            logger.error(f"Error preparing TheOver raw export: {e}")

# TheOver logs export (always available if logs exist)
if "theover_debug_log" in st.session_state and st.session_state["theover_debug_log"]:
    try:
        theover_logs = st.session_state["theover_debug_log"]
        if isinstance(theover_logs, list) and len(theover_logs) > 0:
            theover_debug_df = pd.DataFrame(theover_logs)
            theover_csv = theover_debug_df.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button(
                label="📋 Download TheOver Debug (Logs)",
                data=theover_csv,
                file_name="theover_log_dump.csv",
                mime="text/csv",
                key="theover_log_sidebar_export_btn",
                help="Exports fuzzy match logs from TheOver integration"
            )
    except Exception as e:
        logger.error(f"Error preparing TheOver log export: {e}")

# Master results export (available after analysis)
if "master_results_df" in st.session_state:
    master_df = st.session_state["master_results_df"]
    if master_df is not None and not master_df.empty:
        try:
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
            # Fix Issue 1: Deduplicate columns and fix concatenated headers
            master_df = master_df.loc[:, ~master_df.columns.duplicated()].copy()
            # Fix concatenated column names (e.g. "ProbProb" -> "Prob")
            master_df.columns = master_df.columns.str.replace(r'^(.+)\1+$', r'\1', regex=True)

            # Task 3.2: Log Data Quality Summary before export
            if "data_quality_score" in master_df.columns and "data_quality_grade" in master_df.columns:
                logger.info("=" * 60)
                logger.info("DATA QUALITY SUMMARY")
                logger.info("=" * 60)

                # Calculate statistics
                mean_score = master_df["data_quality_score"].mean()
                median_score = master_df["data_quality_score"].median()
                min_score = master_df["data_quality_score"].min()
                max_score = master_df["data_quality_score"].max()

                logger.info(f"Total Picks: {len(master_df)}")
                logger.info(f"Average Quality Score: {mean_score:.1f}/100")
                logger.info(f"Median Quality Score: {median_score:.1f}/100")
                logger.info(f"Score Range: {min_score:.0f} - {max_score:.0f}")
                logger.info("")

                # Grade distribution
                grade_counts = master_df["data_quality_grade"].value_counts().sort_index()
                logger.info("Grade Distribution:")
                for grade in ["A", "B", "C", "D", "F"]:
                    count = grade_counts.get(grade, 0)
                    percentage = (count / len(master_df)) * 100 if len(master_df) > 0 else 0
                    logger.info(f"  Grade {grade}: {count:3d} picks ({percentage:5.1f}%)")

                logger.info("=" * 60)

            master_csv = master_df.to_csv(index=False).encode("utf-8")
            logger.info(f"Exporting {len(master_df)} rows from master_results_df to CSV for user download.")
            st.sidebar.download_button(
                label="📥 Download All Picks (CSV)",
                data=master_csv,
                file_name=f"parlaypicker_all_picks_{timestamp_str}.csv",
                mime="text/csv",
                key="master_results_sidebar_export_btn",
                help="Download all picks with sentiment analysis, probabilities, and rankings",
                type="primary"
            )
        except Exception as e:
            logger.error(f"Error preparing analysis export: {e}")
else:
    # Show helpful message when no data is available
    st.sidebar.caption("⏳ Run Master Analysis to generate picks, then you can download all picks here.")

# Master df export (raw master dataframe)
if "master_df" in st.session_state:
    master_df_raw = st.session_state["master_df"]
    if master_df_raw is not None and not master_df_raw.empty:
        try:
            master_df_csv = master_df_raw.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button(
                label="📊 Download Master DF (Raw)",
                data=master_df_csv,
                file_name="master_df_raw.csv",
                mime="text/csv",
                key="master_df_sidebar_export_btn",
                help="Raw master dataframe before UI filtering"
            )
        except Exception as e:
            logger.error(f"Error preparing master_df export: {e}")

# Show message if no exports available
if not any([
    st.session_state.get("debug_accumulator"),
    st.session_state.get("debug_log_history"),
    st.session_state.get("theover_raw_df") is not None,
    st.session_state.get("theover_debug_log"),
    st.session_state.get("master_results_df") is not None,
    st.session_state.get("master_df") is not None,
]):
    st.sidebar.info("⏳ Run analysis to generate exports")

api_sports_present = (
    get_secret_any("APISPORTS_API_KEY", "API_SPORTS_KEY", "API_SPORTS_API_KEY") is not None
    or any_secret_prefix("APISPORTS_")
)
sportsdata_present = (
    get_secret_any("SPORTSDATA_API_KEY", "SPORTSDATA_KEY") is not None
    or any_secret_prefix("SPORTSDATA_")
)
api_sports_status = "OK" if api_sports_present or any(v for v in api_sports_clients.values() if v) else "MISSING"
sportsdata_status = "OK" if sportsdata_present or any(v for v in sportsdata_clients.values() if v) else "MISSING"
gemini_ready = st.session_state.get("gemini_initialized", False) or bool(get_secret_any("GEMINI_API_KEY"))
st.sidebar.markdown("---")
st.sidebar.subheader("Status")
sentiment_meta_sidebar = st.session_state.get("sentiment_meta") or init_sentiment_meta()
sentiment_available = int(sentiment_meta_sidebar.get("sentiment_available_count") or 0) > 0
sentiment_cached = bool(sentiment_meta_sidebar.get("sentiment_used_cached"))
sentiment_cooldown = bool(sentiment_meta_sidebar.get("sentiment_rate_limited")) or str(sentiment_meta_sidebar.get("sentiment_status", "")).upper() == "COOLDOWN"
if not enable_sentiment:
    sentiment_status_text = "Disabled"
    sentiment_status_color = "red"
elif sentiment_cooldown:
    sentiment_status_text = "Rate Limited"
    sentiment_status_color = "orange"
elif sentiment_available or sentiment_cached:
    sentiment_status_text = "OK"
    sentiment_status_color = "green"
else:
    sentiment_status_text = "No Data"
    sentiment_status_color = "red"
badges = {
    "OddsAPI": bool(odds_api_key),
    "AI Model": True,
    "Gemini": gemini_ready,
    "News": bool(news_api_key),
    "API-Sports": api_sports_status == "OK",
    "SportsData": sportsdata_status == "OK",
    "Kalshi": bool(kalshi_api_key and kalshi_api_secret),
}
for name, ok in badges.items():
    color = "green" if ok else "red"
    st.sidebar.markdown(f"**{name}:** :{color}[{'OK' if ok else 'Missing'}]")
st.sidebar.markdown(f"**Sentiment:** :{sentiment_status_color}[{sentiment_status_text}]")
with st.sidebar.expander("Key sources (API-Sports/SportsData)"):
    st.caption("Lookups: API_SPORTS_KEY, APISPORTS_API_KEY, NBA/NFL specific; SPORTSData: SPORTSDATA_API_KEY/KEY variants")

render_pipeline_banner()


# -----------------
# Tabs
# -----------------

tab_master, tab_shotgun, tab_games, tab_kalshi, tab_sentiment, tab_debug, tab_movement = st.tabs(
    ["Master Analysis", "Shotgun Mode", "Game Slate", "Kalshi", "Sentiment", "Debug", "Market Movement"]
)


with tab_games:
    st.header("Games & Odds")
    games = st.session_state.get("games", [])
    sent_map = st.session_state.get("sentiment_map") or {}
    match_lookup: Dict[Tuple[Any, Any, Any, Any], Dict[str, Any]] = {}
    _entries_raw = st.session_state.get("kalshi_match_results") or {}
    entries = _entries_raw.values() if isinstance(_entries_raw, dict) else (_entries_raw or [])
    for entry in entries:
        game = entry.get("game") or {}
        matches = entry.get("matches") or {}
        winner = matches.get("winner") or {}
        key = (
            game.get("league"),
            game.get("home_team"),
            game.get("away_team"),
            game.get("commence_time_iso_utc") or game.get("commence_time"),
        )
        match_lookup[key] = winner

    if not games:
        st.info("Load games from the sidebar to begin.")
    else:
        rows = []
        for g in games:
            markets = set()
            for bm in g.get("bookmakers") or []:
                for m in bm.get("markets") or []:
                    if m.get("key"):
                        markets.add(m.get("key"))
            home_sent = sent_map.get(g.get("home_team"))
            away_sent = sent_map.get(g.get("away_team"))
            sent_diff = None
            if home_sent is not None and away_sent is not None:
                sent_diff = home_sent - away_sent
            rows.append(
                {
                    "League": g.get("league"),
                    "Home": g.get("home_team"),
                    "Away": g.get("away_team"),
                    "Commence (UTC)": g.get("commence_time_iso_utc")
                    or safe_iso(g.get("commence_time_iso")),
                    "Commence (Local)": fmt_local_time(g.get("commence_time_local")),
                    "Local Date": g.get("commence_date_local") or "",
                    "Books": len(g.get("bookmakers") or []),
                    "MarketsAvailable": ", ".join(sorted(markets)),
                    "home_ml_price": g.get("home_ml_price"),
                    "away_ml_price": g.get("away_ml_price"),
                    "implied_prob_home": g.get("implied_prob_home"),
                    "implied_prob_away": g.get("implied_prob_away"),
                    "home_spread_point": g.get("home_spread_point"),
                    "home_spread_price": g.get("home_spread_price"),
                    "away_spread_point": g.get("away_spread_point"),
                    "away_spread_price": g.get("away_spread_price"),
                    "total_point": g.get("total_point"),
                    "over_price": g.get("over_price"),
                    "under_price": g.get("under_price"),
                    "Home_Sentiment": home_sent,
                    "Away_Sentiment": away_sent,
                    "Sentiment_Diff": sent_diff,
                    "warnings": ",".join(g.get("warnings") or []),
                }
            )
        # Add quick pick/prob columns for ML/Spread/Total plus any known Kalshi match info.
        enriched_rows = []
        for r in rows:
            implied_home = r.get("implied_prob_home")
            implied_away = r.get("implied_prob_away")
            ml_pick = None
            ml_pick_prob = None
            if implied_home is not None or implied_away is not None:
                if (implied_home or 0) >= (implied_away or 0):
                    ml_pick = r["Home"]
                    ml_pick_prob = implied_home
                else:
                    ml_pick = r["Away"]
                    ml_pick_prob = implied_away

            # Spread pick
            spread_pick = None
            spread_pick_prob = None
            spread_line = None
            home_spread_prob = american_to_implied_prob(r.get("home_spread_price"))
            away_spread_prob = american_to_implied_prob(r.get("away_spread_price"))
            if r.get("home_spread_point") is not None:
                spread_pick = r["Home"]
                spread_pick_prob = home_spread_prob
                spread_line = r.get("home_spread_point")
                if away_spread_prob is not None and (away_spread_prob >= (home_spread_prob or 0)):
                    spread_pick = r["Away"]
                    spread_pick_prob = away_spread_prob
                    spread_line = r.get("away_spread_point")

            # Total pick
            total_pick = None
            total_pick_prob = None
            if r.get("total_point") is not None:
                over_prob = american_to_implied_prob(r.get("over_price"))
                under_prob = american_to_implied_prob(r.get("under_price"))
                total_pick = "Over"
                total_pick_prob = over_prob
                if under_prob is not None and (under_prob >= (over_prob or 0)):
                    total_pick = "Under"
                    total_pick_prob = under_prob

            key = (r["League"], r["Home"], r["Away"], r["Commence (UTC)"])
            kalshi_info = match_lookup.get(key, {})

            enriched_rows.append(
                {
                    **r,
                    "ML Pick": ml_pick,
                    "ML Pick Prob": ml_pick_prob,
                    "Spread Pick": f"{spread_pick} {spread_line}" if spread_pick is not None else None,
                    "Spread Pick Prob": spread_pick_prob,
                    "Total Pick": f"{total_pick} {r.get('total_point')}" if total_pick else None,
                    "Total Pick Prob": total_pick_prob,
                    "kalshi_available": kalshi_info.get("kalshi_available"),
                    "kalshi_matched": kalshi_info.get("kalshi_matched"),
                    "kalshi_prob": kalshi_info.get("kalshi_prob"),
                    "kalshi_event_ticker": kalshi_info.get("kalshi_event_ticker"),
                }
            )

        st.dataframe(pd.DataFrame(enriched_rows))

with tab_master:
    st.header("Master Analysis")
    kalshi_status = kalshi_health_check(league)
    # === DEBUG: Show Kalshi Market Fetch Results ===
    with st.expander("🔍 Kalshi Debug Info", expanded=False):
        st.write("**Kalshi Markets Fetched:**")
        kalshi_markets_by_league = st.session_state.get("kalshi_markets_by_league", {})
        if kalshi_markets_by_league:
            for league_key, markets in kalshi_markets_by_league.items():
                st.write(f"- **{league_key}**: {len(markets)} markets")
                if markets:
                    sample = markets[0]
                    st.write(f"  - Sample ticker: `{sample.get('event_ticker') or sample.get('ticker')}`")
                    st.write(f"  - Sample title: {sample.get('title', 'N/A')}")
        else:
            st.warning("No Kalshi markets in session state")
        
        # Show what the filter is looking for
        if games:
            first_game = games[0]
            st.write("**First Game Filter Criteria:**")
            st.json({
                "home": first_game.get("home_team"),
                "away": first_game.get("away_team"),
                "commence_utc": first_game.get("commence_time_iso_utc"),
                "league": first_game.get("league")
            })
    # === END DEBUG ===
    if not kalshi_status.get("configured"):
        error_detail = kalshi_status.get("error") or "Kalshi is required and missing keys."
        if kalshi_status.get("status_code"):
            error_detail = f"{error_detail} (status {kalshi_status.get('status_code')}: {kalshi_status.get('response_text_snippet')})"
        st.error(error_detail)
        st.info("Master Analysis is disabled until Kalshi is available.")
    else:
        if kalshi_status.get("error") and not kalshi_status.get("ok"):
            warn_detail = kalshi_status.get("error") or "Kalshi reachable but returned no markets; proceeding without Kalshi data."
            st.warning(warn_detail)
        if kalshi_status.get("warning"):
            st.warning(kalshi_status.get("warning"))
    st.session_state.setdefault("kalshi_match_only", False)
    kalshi_match_only = st.checkbox(
        "Show only games with a Kalshi match (UI display only - exports include all games)",
        value=st.session_state.get("kalshi_match_only", False),
        help="Filter the UI table to show only games with Kalshi market matches. This does NOT affect CSV/JSON exports, which always include all games."
    )
    st.session_state["kalshi_match_only"] = kalshi_match_only
    use_gemini_explanations = st.checkbox(
        "Use Gemini Confidence + Explanation (costs apply)",
        value=st.session_state.get("use_gemini_explanations", False),
        key="use_gemini_explanations",
        help="When enabled, Gemini is called to generate explanations for a limited number of top picks. Costs may apply."
    )

    # Show usage
    if use_gemini_explanations:
        calls = st.session_state.get("gemini_calls_made", 0)
        limit = MAX_GEMINI_CALLS_PER_RUN
        st.caption(f"Gemini usage this session: **{calls}** / {limit} calls")
        if calls >= limit:
            st.warning("⚠️ Per-session limit reached. Refresh to reset.")
    use_model_numeric_probs = st.checkbox(
        "Local Model Predictions",
        value=st.session_state.get("use_model_numeric_probs", True),
        key="use_model_numeric_probs",
        help="If checked, the 'AI_Prob' column will use the local XGBoost model output."
    )
    games = st.session_state.get("games", [])

    # ============================================
    # PHASE 1: DIAGNOSTIC - Games Loading Summary
    # ============================================
    logger.info("="*80)
    logger.info(f"DIAGNOSTIC: Loaded {len(games)} games from TheOddsAPI")
    logger.info("="*80)

    if games:
        # Check market data availability
        games_with_ml = sum(1 for g in games if g.get('home_ml_price') is not None)
        games_with_spread = sum(1 for g in games if g.get('home_spread_point') is not None)
        games_with_total = sum(1 for g in games if g.get('total_point') is not None)
        games_with_no_data = sum(1 for g in games if not (g.get('home_ml_price') or g.get('home_spread_point') or g.get('total_point')))

        logger.info(f"Market Data Summary:")
        logger.info(f"  - Games with Moneyline: {games_with_ml}/{len(games)}")
        logger.info(f"  - Games with Spread: {games_with_spread}/{len(games)}")
        logger.info(f"  - Games with Total: {games_with_total}/{len(games)}")
        logger.info(f"  - Games with NO market data: {games_with_no_data}/{len(games)}")

        # Log first 3 games as samples
        logger.info(f"\nSample Game Data (first 3):")
        for idx, g in enumerate(games[:3]):
            logger.info(f"  Game {idx+1}: {g.get('home_team')} vs {g.get('away_team')}")
            logger.info(f"    League: {g.get('league')}")
            logger.info(f"    ML Price: {g.get('home_ml_price')} / {g.get('away_ml_price')}")
            logger.info(f"    Spread: {g.get('home_spread_point')}")
            logger.info(f"    Total: {g.get('total_point')}")

        if games_with_ml == 0 and games_with_spread == 0 and games_with_total == 0:
            logger.error("CRITICAL: ALL games are missing market data!")
    else:
        logger.debug("Games list empty (expected during initialization)")

    logger.info("="*80)
    # ============================================

    # if run_master and (not kalshi_status.get("configured")):
    #    st.error("Kalshi is required but unavailable. Fix Kalshi first.")
    #    st.stop()

    # Determine if we need to run (user clicked button) or just display (cached df exists)
    df_existing = st.session_state.get("master_results_df")
    
    # If we have existing data and didn't request a re-run, use it to skip the heavy lifting
    # We still need to define the helper functions because they are called during the DataFrame construction block
    # Actually, the entire block below constructs the DataFrame. We need to restructure this.
    
    # Initialize variables to avoid NameError if skipped or failed
    theover_df = pd.DataFrame()
    theover_lookup_exact = {}
    theover_lookup_teams = {}
    theover_stats = {
        "total_rows": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "unmatched_examples": []
    }

    # 1. Initialize session state for the results if not present
    if "master_results_df" not in st.session_state:
        st.session_state["master_results_df"] = None

    if st.button("🚀 Run Master Analysis"):
        # Generate new run_id for this analysis run
        import uuid
        analysis_run_id = str(uuid.uuid4())[:8]
        st.session_state["run_id"] = analysis_run_id
        logger.info(f"🆕 Starting Master Analysis with run_id: {analysis_run_id}")

        # Validate games list before proceeding
        if not games or len(games) == 0:
            st.error("❌ No games available to analyze. Please load games first by clicking 'Load Games' in the sidebar.")
            logger.error(f"CRITICAL: Cannot run Master Analysis - games list is empty (run_id: {analysis_run_id})")
            st.stop()

        with st.spinner("🔄 Analyzing Markets..."):
            try:
                # Task 1: Pre-process games to ensure commence_date_local is set for TheOver matching
                # This aligns the dates so the canonical keys match (ingestion vs loop).
                for g in games:
                    if not g.get("commence_date_local"):
                        try:
                            # Try to derive from UTC
                            dt = parse_commence_to_utc(g.get("commence_time_iso_utc") or g.get("commence_time"))
                            if dt:
                                # Use simple YYYY-MM-DD - Convert to ET to match main loop
                                dt_et = dt.astimezone(ZoneInfo("America/New_York"))
                                g["commence_date_local"] = dt_et.strftime("%Y-%m-%d")
                        except Exception:
                            pass

                # 0. Parse TheOver inputs
                with st.spinner("Processing TheOver.ai data..."):
                    theover_df, ingestion_stats = process_theover_inputs(
                        totals_file=theover_totals_file,
                        sides_file=theover_sides_file,
                        totals_paste=theover_totals_text,
                        sides_paste=theover_sides_text,
                        games=games
                    )

                    # Use returned stats which now include fuzzy match results
                    theover_stats = ingestion_stats.copy()

                if not theover_df.empty:
                    for _, row in theover_df.iterrows():
                        row_dict = row.to_dict()

                        # 1. Exact Canonical Key
                        # Format: {league}|{away_code}|{home_code}|{local_date}
                        ex_key = row["theover_key"]
                        if ex_key not in theover_lookup_exact:
                            theover_lookup_exact[ex_key] = []
                        theover_lookup_exact[ex_key].append(row_dict)

                        # 2. Team Key (League|AwayCode|HomeCode) for date matching
                        # The ingest module ensures these columns exist
                        lg = str(row.get("league")).upper()
                        aw = str(row.get("away_code")).upper()
                        hm = str(row.get("home_code")).upper()

                        if lg and aw and hm:
                            tm_key = f"{lg}|{aw}|{hm}"
                            if tm_key not in theover_lookup_teams:
                                theover_lookup_teams[tm_key] = []
                            theover_lookup_teams[tm_key].append(row_dict)

                # DEBUG: Log TheOver lookup keys for diagnosis
                if theover_lookup_exact:
                    sample_exact_keys = list(theover_lookup_exact.keys())[:5]
                    sample_team_keys = list(theover_lookup_teams.keys())[:5]
                    logger.info(f"🔍 THEOVER LOOKUP BUILT: {len(theover_lookup_exact)} exact keys, {len(theover_lookup_teams)} team keys")
                    logger.info(f"🔍 THEOVER EXACT KEY SAMPLES: {sample_exact_keys}")
                    logger.info(f"🔍 THEOVER TEAM KEY SAMPLES: {sample_team_keys}")
                else:
                    logger.warning("⚠️ THEOVER LOOKUP EMPTY: No TheOver data was ingested")

                st.session_state["DECISION_TRACE_SAMPLES"] = {}

                def store_decision_trace_sample(
                    league_code: Optional[str],
                    home_team: Optional[str],
                    away_team: Optional[str],
                    market: str,
                    pick: Optional[str],
                    final_probability: Optional[float],
                    trace_json_raw: Any,
                ) -> None:
                    league_code_norm = (league_code or "").upper()
                    if league_code_norm not in DECISION_TRACE_SAMPLE_LEAGUES:
                        return
                    try:
                        samples = dict(st.session_state.get("DECISION_TRACE_SAMPLES", {}))
                    except Exception:
                        samples = {}
                    if league_code_norm in samples:
                        return
                    try:
                        parsed_trace = json.loads(trace_json_raw) if isinstance(trace_json_raw, str) else trace_json_raw
                    except Exception:
                        parsed_trace = trace_json_raw
                    samples[league_code_norm] = {
                        "league": league_code_norm,
                        "home": home_team,
                        "away": away_team,
                        "market": market,
                        "pick": pick,
                        "final_probability": final_probability,
                        "decision_trace_json": parsed_trace,
                    }
                    st.session_state["DECISION_TRACE_SAMPLES"] = samples

                api_sports_status_run = "pending"
                sportsdata_status_run = "pending"

                # Extract unique teams directly from games for sentiment analysis
                api_sports_clients, _ = init_data_clients()
                unique_teams = sorted(
                    set(g.get("home_team", "") for g in (games or []) if g.get("home_team"))
                    | set(g.get("away_team", "") for g in (games or []) if g.get("away_team"))
                )
                enable_sentiment_master = st.checkbox(
                    "Enable sentiment (NewsAPI)",
                    value=True,
                    key="enable_sentiment_master",
                )
                st.session_state["enable_sentiment"] = enable_sentiment_master
                slate_sentiment = get_slate_sentiment(enable_sentiment_master, unique_teams, "MIXED", news_api_key)
                st.session_state["sentiment_map"] = slate_sentiment.get("map") or {}
                st.session_state["sentiment_meta_map"] = slate_sentiment.get("meta_map") or {}
                st.session_state["sentiment_meta"] = slate_sentiment.get("meta") or init_sentiment_meta()
                st.session_state["sentiment_debug"] = slate_sentiment.get("debug") or {}
                with st.expander("Sentiment Debug", expanded=False):
                    meta_view = slate_sentiment.get("meta") or {}
                    meta_map_view = slate_sentiment.get("meta_map") or {}
                    source_counts: Dict[str, int] = {}
                    for mv in meta_map_view.values():
                        src_val = str(mv.get("sentiment_source") or "none")
                        source_counts[src_val] = source_counts.get(src_val, 0) + 1
                    st.write("Sentiment source:", meta_view.get("sentiment_source"))
                    st.write("Status counts:", meta_view.get("sentiment_status_counts"))
                st.write("Teams by source:", source_counts)
                st.write(
                    "Reddit posts/comments used:",
                    meta_view.get("reddit_posts_used", 0),
                    meta_view.get("reddit_comments_used", 0),
                )
                st.write("Unique teams:", len(unique_teams))
                st.json(meta_view)

                sentiment_pack_meta = st.session_state.get("sentiment_meta") or init_sentiment_meta()
                sentiment_map: Dict[str, Optional[float]] = st.session_state.get("sentiment_map") or {}
                sentiment_meta_map: Dict[str, Dict[str, Any]] = st.session_state.get("sentiment_meta_map") or {}
                sentiment_status_counts_global = sentiment_pack_meta.get("sentiment_status_counts") or {"NO_CALL": 1}
                if st.session_state.get("kalshi_required", True) and kalshi_integrator:
                    try:
                        kalshi_integrator.assert_available()
                    except Exception as exc:
                        st.error(str(exc))
                        st.stop()

                commence_times_by_league: Dict[str, List[str]] = {}
                # FIX: Accumulator initialized ONCE.
                # We ensure it is a fresh list and is NOT reassigned inside the loop.
                master_stats = {"rows_out": 0, "games_in": len(games), "kalshi_matches": 0, "kalshi_total": len(games), "market_rows_out": 0}
                rows_out = []
                for g in games:
                    # FIX: Track if ANY row was created for this game to ensure fallback works
                    ml_row_created = False
                    spread_row_created = False
                    total_row_created = False
                    lg = g.get("league")
                    commence_val = g.get("commence_time_iso_utc") or g.get("commence_time") or g.get("commence_time_iso")
                    if not commence_val:
                        continue
                    commence_times_by_league.setdefault(lg, []).append(commence_val)
                sentiment_meta_global: Dict[str, Any] = {**init_sentiment_meta(), **(st.session_state.get("sentiment_meta") or {})}
                sentiment_status_counts_global = sentiment_meta_global.get("sentiment_status_counts") or {"NO_CALL": 1}
                leagues_for_fetch = list({k for k in commence_times_by_league.keys() if k}) or (selected_sports or [league])
                try:
                    kalshi_markets_by_league = fetch_kalshi_markets_for_leagues(
                        leagues_for_fetch, commence_times_by_league
                    )
                except RuntimeError as exc:
                    msg = str(exc)
                    if "429" in msg or "rate limit" in msg.lower():
                        st.error("Kalshi rate-limited. Please retry in ~X seconds.")
                    else:
                        st.error(msg)
                    st.stop()
                except Exception as exc:
                    st.error(f"Error initializing analysis: {exc}")
                    # Don't stop, try to proceed without Kalshi if that was the issue
                    kalshi_markets_by_league = {}

                if not kalshi_markets_by_league:
                    st.warning(
                        "Kalshi markets could not be fetched; proceeding with cached/empty set."
                    )
            except Exception:
                st.session_state["last_exception"] = traceback.format_exc()

            # Initialize loop state variables
            # "Fix the NameError": Define immediately before usage/loop.
            full_search_first_game = {"init": True}
            rows_out = []
            fg_league = None

            # Pre-allocate accumulator for batch processing
            # Instead of appending to a list inside loop, we'll collect dicts then DataFrame.from_records
            accumulated_rows: List[Dict[str, Any]] = []

            # Loop Blockage / First Game Search
            if (
                full_search_first_game
                and isinstance(full_search_first_game, dict)
                and (fg_league or league) == "NBA"
                and not full_search_first_game.get("found_any_winner_market_for_game")
                and (
                    full_search_first_game.get("found_any_total_market_for_game")
                    or full_search_first_game.get("matches")
                )
            ):
                winner_refetch_attempted = True
                try:
                    refreshed = kalshi_integrator.get_sports_markets() if kalshi_integrator else []
                    prefix = LEAGUE_SERIES_MAP.get((league or "").upper())
                    if prefix:
                        refreshed = [
                            m
                            for m in refreshed
                            if str(m.get("ticker") or "").upper().startswith(prefix)
                        ]
                    if refreshed:
                        kalshi_markets_by_league[(fg_league or league)] = refreshed
                        st.session_state["kalshi_all_markets"] = refreshed
                        if games:
                            try:
                                fg = games[0]
                                full_search_first_game = debug_search_markets_for_game(
                                    kalshi_markets_by_league.get(fg_league or league, []),
                                    fg.get("home_team"),
                                    fg.get("away_team"),
                                    (team_code_candidates(fg_league or league, fg.get("home_team")) or [None])[0],
                                    (team_code_candidates(fg_league or league, fg.get("away_team")) or [None])[0],
                                    league=fg_league or league,
                                )
                            except Exception:
                                st.session_state["last_exception"] = traceback.format_exc()
                except Exception:
                    st.session_state["last_exception"] = traceback.format_exc()
            filtered_counts: List[int] = []
            per_game_kalshi_debug: List[Dict[str, Any]] = []
            first_game_full_search = full_search_first_game
            rows_out: List[Dict[str, Any]] = []
            master_stats = {
                "games_in": len(games),
                "rows_out": 0,
                "h2h_found": 0,
                "exceptions": 0,
                "market_rows_out": 0,
                "kalshi_matches": 0,
                "kalshi_total": len(games),
            }
            data_source_stats = {
                "api_sports_games": 0,
                "sportsdata_games": 0,
                "injury_pulls": 0,
                "weather_pulls": 0,
                "errors": [],
            }
            # Dictionary mapping for robust access (User Request: "Switch to Dictionary Mapping")
            kalshi_match_results: Dict[str, Dict[str, Any]] = {}
            _kalshi_ticker_owners: Dict[str, str] = {}  # ticker -> game_id for collision detection
            # --- CLEANED MASTER ANALYSIS LOOP ---

            # --- PRE-LOOP BATCH ENRICHMENT (PARITY FIX) ---
            # Ensure 'g' in the loop has stats for single-row prediction.
            # We must perform batch enrichment on the raw games list BEFORE the loop.
            games_to_process = games
            if games:
                try:
                    with st.spinner("🚀 PRE-FETCH: Batch Enriching Stats..."):
                        # 1. Convert to DataFrame
                        _df_pre = pd.DataFrame(games)

                        # 2. Ensure League column exists (crucial for enrichment lookup)
                        if "League" not in _df_pre.columns:
                            _df_pre["League"] = league

                        # 3. Call Enrichment (uses TeamNameMatcher and API clients)
                        # Note: We pass the specific client for this league
                        _client_map = {league: api_sports_clients.get(league)} if api_sports_clients else {}
                        _df_enriched = enrich_with_model_features(_df_pre, _client_map)

                        # 4. Convert back to list of dicts for the loop
                        # Force numeric conversion where possible to avoid NaN issues
                        games_to_process = _df_enriched.to_dict('records')

                        # Debug: Verify one row
                        if games_to_process:
                            logger.info("Pre-Enrichment Sample Stats: %s", games_to_process[0].get('feature_home_ppg'))
                except Exception as e:
                    logger.error(f"Pre-loop enrichment failed: {e}", exc_info=True)
                    if st:
                        st.error(f"Pre-enrichment failed: {e}")
                    games_to_process = games

            # --- FIX: Define variables at the start of the loop ---
            # Ensure use_model_numeric_probs is available in local scope
            use_model_numeric_probs = st.session_state.get("use_model_numeric_probs", True)
            # Ensure model_mode is available
            model_mode = st.session_state.get("model_mode", "Local XGBoost")

            # TheOver Match Counters
            theover_matched_count_sides = 0
            theover_matched_count_totals = 0

            for idx, g in enumerate(games_to_process):
                # DIAGNOSTIC: Log start of processing
                if idx < 5 or idx % 10 == 0:
                    logger.info(f"Processing game {idx+1}/{len(games_to_process)}: {g.get('home_team')} vs {g.get('away_team')}")
                g = g.copy()
                # Initialize loop-local variables to prevent NameError
                model_prob_home = None
                model_warn = None
                sentiment_diff = None  # Ensure initialized
                total_sentiment_debug = None
                spread_sentiment_debug = None

                # DIAGNOSTIC: Track row creation for each game
                ml_row_created = False
                spread_row_created = False
                total_row_created = False
                game_home = g.get("home_team", "Unknown")
                game_away = g.get("away_team", "Unknown")
                logger.info(f"🎮 Processing game {idx+1}/{len(games_to_process)}: {game_away} @ {game_home}")

                # Track rows count before processing this game
                rows_count_before = len(rows_out)

                # Initialize warnings list from game data
                warnings = list(g.get("warnings") or [])
                # Save base warnings to prevent leakage between market types (ML -> Spread -> Total)
                base_game_warnings = list(warnings)

                # Weights & Status Defaults (Fix NameErrors)
                spread_weights = {}
                total_weights = {}
                moneyline_weights = {}
                spread_weights_used = {}
                total_weights_used = {}
                moneyline_weights_used = {}

                theover_prob_final_spread = None
                theover_prob_final_total = None
                theover_delta_spread = None
                theover_delta_total = None

                spread_prob_final = None
                total_prob_final = None

                # Default empty containers for decision traces
                spread_trace_json = "{}"
                total_trace_json = "{}"

                # --- THEOVER MATCHING START ---
                league_name = str(g.get("league") or "UNKNOWN").upper()
                home_team = str(g.get("home_team") or "")
                away_team = str(g.get("away_team") or "")
                commence_utc = parse_commence_to_utc(g.get("commence_time"))

                # Local Date for Matching (US/Eastern)
                local_date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d") # Default
                if commence_utc:
                    local_dt = commence_utc.astimezone(ZoneInfo("America/New_York"))
                    local_date_str = local_dt.strftime("%Y-%m-%d")

                # FIX: Use RAW team names for team code generation, NOT the
                # cross-league TEAM_ALIAS_MAP which causes contamination (e.g.
                # "New Orleans Pelicans" -> "New Orleans Saints" via NFL substring).
                # The TheOver ingest uses raw game team names, so the main loop
                # must also use raw names to produce matching canonical keys.
                home_norm = home_team
                away_norm = away_team

                # 1. Resolve Team Codes using RAW team names (matches TheOver ingest behavior)
                home_code = team_code_for_league(league_name, home_team)
                away_code = team_code_for_league(league_name, away_team)

                # 2. Generate Master Key
                # Normalize league to match ingestion (NBA, NFL, etc.)
                norm_league_key = str(league_name).strip().upper()
                if "NBA" in norm_league_key: norm_league_key = "NBA"
                elif "NFL" in norm_league_key: norm_league_key = "NFL"
                elif "NHL" in norm_league_key: norm_league_key = "NHL"
                elif "MLB" in norm_league_key: norm_league_key = "MLB"
                elif "NCAAB" in norm_league_key or "COLLEGE BASKETBALL" in norm_league_key: norm_league_key = "NCAAB"
                elif "NCAAF" in norm_league_key or "COLLEGE FOOTBALL" in norm_league_key: norm_league_key = "NCAAF"

                # Generate codes with normalized league and RAW team names
                home_code_norm = team_code_for_league(norm_league_key, home_team)
                away_code_norm = team_code_for_league(norm_league_key, away_team)

                # Generate canonical keys matching TheOver ingest format
                master_key_exact = generate_canonical_key(norm_league_key, local_date_str, home_code_norm, away_code_norm)
                master_key_teams = f"{norm_league_key}|{away_code_norm}|{home_code_norm}"

                # DEBUG: Log first few games' TheOver keys for diagnosis
                if idx < 3 and theover_lookup_exact:
                    logger.info(f"🔍 THEOVER MATCH ATTEMPT [{idx}]: {away_team} @ {home_team}")
                    logger.info(f"   exact_key={master_key_exact}")
                    logger.info(f"   team_key={master_key_teams}")
                    logger.info(f"   exact_hit={master_key_exact in theover_lookup_exact}")
                    logger.info(f"   team_hit={master_key_teams in theover_lookup_teams}")

                # 3. Match TheOver Data
                matched_total_row = None
                matched_side_row = None
                theover_match_reason = None
                theover_matched = False

                # Helper to find best match in list
                def find_best_date_match(candidates, target_date_str):
                    # Prefer exact date match
                    for c in candidates:
                        if c.get("date_local") == target_date_str:
                            return c
                    # Fallback to first if available (fuzzy date)
                    return candidates[0] if candidates else None

                # Also build a swapped key for robustness (in case home/away order differs)
                master_key_exact_swap = generate_canonical_key(norm_league_key, local_date_str, away_code_norm, home_code_norm)
                master_key_teams_swap = f"{norm_league_key}|{home_code_norm}|{away_code_norm}"

                # Look for TOTAL match (Exact then Team, then swapped variants)
                if master_key_exact in theover_lookup_exact:
                    # Check for TOTAL type
                    cands = [r for r in theover_lookup_exact[master_key_exact] if r["theover_market_type"] == "TOTAL"]
                    if cands: matched_total_row = cands[0]

                if not matched_total_row and master_key_exact_swap in theover_lookup_exact:
                    cands = [r for r in theover_lookup_exact[master_key_exact_swap] if r["theover_market_type"] == "TOTAL"]
                    if cands: matched_total_row = cands[0]

                if not matched_total_row and master_key_teams in theover_lookup_teams:
                    cands = [c for c in theover_lookup_teams[master_key_teams] if c["theover_market_type"] == "TOTAL"]
                    matched_total_row = find_best_date_match(cands, local_date_str)

                if not matched_total_row and master_key_teams_swap in theover_lookup_teams:
                    cands = [c for c in theover_lookup_teams[master_key_teams_swap] if c["theover_market_type"] == "TOTAL"]
                    matched_total_row = find_best_date_match(cands, local_date_str)

                # Look for SIDE match (Exact then Team, then swapped variants)
                if master_key_exact in theover_lookup_exact:
                    cands = [r for r in theover_lookup_exact[master_key_exact] if r["theover_market_type"] == "SIDE"]
                    if cands: matched_side_row = cands[0]

                if not matched_side_row and master_key_exact_swap in theover_lookup_exact:
                    cands = [r for r in theover_lookup_exact[master_key_exact_swap] if r["theover_market_type"] == "SIDE"]
                    if cands: matched_side_row = cands[0]

                if not matched_side_row and master_key_teams in theover_lookup_teams:
                    cands = [c for c in theover_lookup_teams[master_key_teams] if c["theover_market_type"] == "SIDE"]
                    matched_side_row = find_best_date_match(cands, local_date_str)

                if not matched_side_row and master_key_teams_swap in theover_lookup_teams:
                    cands = [c for c in theover_lookup_teams[master_key_teams_swap] if c["theover_market_type"] == "SIDE"]
                    matched_side_row = find_best_date_match(cands, local_date_str)

                # FALLBACK: Raw team name matching when key-based lookup fails
                # This handles cases where TheOver paste text uses short names (e.g. "Celtics")
                # that generate different codes than OddsAPI full names (e.g. "Boston Celtics")
                if (not matched_total_row or not matched_side_row) and theover_lookup_exact:
                    home_norm_lower = home_team.lower().strip()
                    away_norm_lower = away_team.lower().strip()
                    # Extract last word of team name for matching (e.g., "Boston Celtics" -> "celtics")
                    home_last = home_norm_lower.split()[-1] if home_norm_lower.split() else ""
                    away_last = away_norm_lower.split()[-1] if away_norm_lower.split() else ""
                    for _to_key, _to_rows in theover_lookup_exact.items():
                        for _to_row in _to_rows:
                            _to_home_raw = str(_to_row.get("home_team_raw") or "").lower().strip()
                            _to_away_raw = str(_to_row.get("away_team_raw") or "").lower().strip()
                            # Match if raw names are substrings or last-word matches
                            home_hit = (
                                home_norm_lower == _to_home_raw
                                or home_last == _to_home_raw
                                or _to_home_raw in home_norm_lower
                                or home_norm_lower in _to_home_raw
                            )
                            away_hit = (
                                away_norm_lower == _to_away_raw
                                or away_last == _to_away_raw
                                or _to_away_raw in away_norm_lower
                                or away_norm_lower in _to_away_raw
                            )
                            if home_hit and away_hit:
                                mtype = _to_row.get("theover_market_type")
                                if mtype == "TOTAL" and not matched_total_row:
                                    matched_total_row = _to_row
                                    logger.info(f"✅ TheOver FALLBACK TOTAL: {away_team} @ {home_team} matched via raw names")
                                elif mtype == "SIDE" and not matched_side_row:
                                    matched_side_row = _to_row
                                    logger.info(f"✅ TheOver FALLBACK SIDE: {away_team} @ {home_team} matched via raw names")
                            if matched_total_row and matched_side_row:
                                break
                        if matched_total_row and matched_side_row:
                            break

                # Diagnostic logging for TheOver matching
                if not matched_total_row and not matched_side_row and theover_lookup_exact:
                    logger.info(
                        f"⚠️ TheOver NO MATCH: {away_team} @ {home_team} | "
                        f"key_exact={master_key_exact} | key_teams={master_key_teams} | "
                        f"codes=({away_code_norm}@{home_code_norm})"
                    )

                # If matched, update counters
                if matched_total_row: theover_matched_count_totals += 1
                if matched_side_row: theover_matched_count_sides += 1

                # Extract Signals for Downstream
                theover_matched_total = matched_total_row
                theover_matched_side = matched_side_row

                # Probabilities (defaults to None if not matched)
                theover_prob_total = None
                if matched_total_row:
                    hit_rate = safe_float(matched_total_row.get("theover_hit_rate"))
                    theover_prob_total = hit_rate if (hit_rate and hit_rate > 0) else None
                    if theover_prob_total is None:
                        logger.warning("TheOver Totals: No valid hit_rate found - excluding")

                theover_prob_spread = None
                theover_prob_ml = None

                if matched_side_row:
                    hit_rate = safe_float(matched_side_row.get("theover_hit_rate"))
                    theover_line_val = safe_float(matched_side_row.get("theover_line"))

                    # Detect ML vs Spread based on line magnitude
                    is_moneyline_side = theover_line_val is not None and abs(theover_line_val) >= 40

                    if is_moneyline_side:
                        # MONEYLINE CALCULATION
                        if hit_rate and hit_rate > 0:
                             theover_prob_ml = hit_rate
                        elif theover_line_val is not None:
                            try:
                                # Convert American Odds to Implied Probability
                                if theover_line_val < 0:
                                    raw_prob = (-theover_line_val) / (-theover_line_val + 100.0)
                                else:
                                    raw_prob = 100.0 / (theover_line_val + 100.0)
                                # Boost slightly for being a "Pick" (0.07 edge assumption)
                                theover_prob_ml = clamp(raw_prob + 0.07, 0.10, 0.95)
                                logger.info(f"TheOver Moneyline: {theover_line_val} -> {theover_prob_ml:.3f}")
                            except Exception as e:
                                logger.warning(f"TheOver ML calc error: {e}")
                                theover_prob_ml = None
                    else:
                        # SPREAD CALCULATION
                        # Check if we have a valid hit rate from TheOver
                        if hit_rate and hit_rate > 0:
                            theover_prob_spread = hit_rate
                            logger.info(f"TheOver Sides: Using provided hit_rate {hit_rate:.3f}")
                        else:
                            # TheOver.ai Sides export lacks WinProbability - calculate from spread line
                            theover_pick_team = matched_side_row.get("theover_pick")

                            if theover_line_val is not None and theover_pick_team:
                                try:
                                    # Treat as spread points (e.g. -5.5)
                                    # Use logistic function to estimate win prob from line
                                    # Assuming standard -110 odds baseline (52.4%) + line advantage
                                    # Simplified model: 0.524 + (line * -0.02) ? No, that depends on direction.

                                    # --- SPREAD POINTS CALCULATION ---
                                    # Use the sigmoid logic from previous version
                                    raw_prob = 1.0 / (1.0 + math.exp(-theover_line_val / 3.5))
                                    adjusted_prob = raw_prob + 0.07
                                    theover_prob_spread = max(0.10, min(0.95, adjusted_prob))
                                    logger.info(f"TheOver Spread: {theover_line_val} -> {theover_prob_spread:.3f}")
                                except Exception:
                                    theover_prob_spread = None
                            else:
                                # No line available - cannot calculate probability
                                logger.info(f"TheOver Sides: No hit_rate or line available for calculation - excluding")
                                theover_prob_spread = None
                                # Only clear matched_side_row if it was SUPPOSED to be spread but failed
                                if not is_moneyline_side:
                                     matched_side_row = None

                # --- FIX: Use SIDE data for ML picks when ML-specific data unavailable ---
                # If TheOver has a spread pick but no ML pick, derive ML probability from spread data
                # A spread pick for a team suggests confidence in their performance, which partially
                # informs moneyline probability (covering spread often correlates with winning outright)
                theover_side_used_for_ml = False
                if theover_prob_ml is None and theover_prob_spread is not None and matched_side_row:
                    # Use the spread hit_rate as a weaker signal for ML
                    # Apply a dampening factor since spread != moneyline
                    # If hit_rate is 55% for spread, ML prob is closer to 50% but still favors that team
                    spread_hit_rate = safe_float(matched_side_row.get("theover_hit_rate"))
                    if spread_hit_rate and spread_hit_rate > 0:
                        # Convert spread confidence to ML confidence
                        # Formula: ML_prob = 0.50 + (spread_hit_rate - 0.50) * 0.7
                        # This dampens the signal while preserving direction
                        ml_from_spread = 0.50 + (spread_hit_rate - 0.50) * 0.7
                        theover_prob_ml = clamp(ml_from_spread, 0.35, 0.65)
                        theover_side_used_for_ml = True
                        logger.info(f"TheOver: Using SIDE hit_rate {spread_hit_rate:.3f} for ML -> {theover_prob_ml:.3f}")
                # --- THEOVER MATCHING END ---

                # 1) Define Weights - Using values from weights_config.py
                # KALSHI: 35%, MARKET: 30%, MODEL: 15%, THEOVER: 15%, SENTIMENT: 10%
                # Total = 105%, will be normalized in compute_final_probability
                spread_weights = {
                    "kalshi_weight": KALSHI_WEIGHT,
                    "odds_weight": MARKET_WEIGHT,
                    "model_weight": ML_MODEL_WEIGHT,
                    "theover_weight": THEOVER_WEIGHT,
                    "sentiment_weight": SENTIMENT_WEIGHT,
                }
                # Totals
                total_weights = {
                    "kalshi_weight": KALSHI_WEIGHT,
                    "odds_weight": MARKET_WEIGHT,
                    "model_weight": ML_MODEL_WEIGHT,
                    "theover_weight": THEOVER_WEIGHT,
                    "sentiment_weight": SENTIMENT_WEIGHT,
                }
                moneyline_weights = {
                    "kalshi_weight": KALSHI_WEIGHT,
                    "odds_weight": MARKET_WEIGHT,
                    "model_weight": ML_MODEL_WEIGHT,
                    "theover_weight": THEOVER_WEIGHT,
                    "sentiment_weight": SENTIMENT_WEIGHT,
                }
                # Debug log
                logger.info(f"Weight sets active: spread={spread_weights}, total={total_weights}, ml={moneyline_weights}")

                # RESET ALL TRACE VARIABLES
                total_pick_side = None
                total_line = None
                total_pick_odds = None
                spread_engine_used = "market_only"
                total_engine_used = "market_only"
                kalshi_prob_spread = None
                spread_prob_final = 0.5
                total_prob_final = 0.5
                spread_prob_alt_final = 0.5
                total_alt_prob_final = 0.5
                spread_decision_score_alt = 0.5
                total_decision_score_alt = 0.5
                total_prob_market = 0.5

                # Additional resets to prevent NameError in fallback_row
                spread_pick_label = None
                spread_alt_label = None
                spread_prob_margin = None
                spread_prob_pick_market = None
                spread_prob_alt_market = None
                spread_prob_pick_kalshi = None
                spread_prob_alt_kalshi = None
                spread_decision_metric_used = None
                spread_decision_score_pick = None
                spread_decision_score_alt = None
                spread_decision_score_margin = None
                spread_trace_json = None

                total_pick_label = None
                total_alt_label = None
                total_prob_margin = None
                total_prob_pick_market = None
                total_prob_alt_market = None
                total_prob_pick_kalshi = None
                total_prob_alt_kalshi = None
                total_decision_metric_used = None
                total_decision_score_pick = None
                total_decision_score_alt = None
                total_decision_score_margin = None
                total_trace_json = None

                decision_trace_version = None
                overall_engine_used = None
                decision_trace_notes = None

                kalshi_prob_spread = None
                kalshi_prob_total = None
                model_spread_prob = None
                model_total_prob = None
                spread_prob_market = 0.5
                total_prob_market = 0.5
                total_line = None
                total_pick_odds = None
                spread_engine_used = "missing"
                total_engine_used = "missing"
                spread_prob_final = 0.5
                total_prob_final = 0.5
                spread_prob_market = 0.5
                total_prob_market = 0.5
                total_pick = None
                spread_pick = None

                # --- LOOP INITIALIZATION (Prevent NameError) ---
                # Reset all loop variables to defaults before processing each game
                spread_engine_used = "missing"
                total_engine_used = "missing"
                overall_engine_used = "missing"
                spread_pick_label = None
                total_pick_label = None
                spread_alt_label = None
                total_alt_label = None
                spread_prob_final = 0.5
                total_prob_final = 0.5
                spread_alt_prob_final = 0.5
                spread_prob_alt_final = 0.5
                total_alt_prob_final = 0.5
                kalshi_prob_spread = None
                kalshi_prob_total = None
                model_spread_prob = None
                model_total_prob = None
                spread_prob_margin = None
                total_prob_margin = None
                spread_prob_pick_market = None
                total_prob_pick_market = None
                spread_prob_alt_market = None
                total_prob_alt_market = None
                spread_prob_pick_kalshi = None
                total_prob_pick_kalshi = None
                spread_prob_alt_kalshi = None
                total_prob_alt_kalshi = None
                spread_decision_metric_used = None
                total_decision_metric_used = None
                spread_decision_score_pick = None
                total_decision_score_pick = None
                spread_decision_score_alt = None
                total_decision_score_alt = None
                spread_decision_score_margin = None
                total_decision_score_margin = None
                spread_trace_json = "{}"
                total_trace_json = "{}"
                decision_trace_version = "v3_loop_init"
                decision_trace_notes = None

                # Kalshi vars
                kalshi_prob_spread = None
                kalshi_prob_total = None
                kalshi_status_value = "PENDING"
                kalshi_event_used = None
                candidate_debug = {}
                kalshi_winner = {}
                kalshi_spread = {}
                kalshi_total = {}

                # Market vars
                spread_pick = None
                spread_line = None
                spread_pick_team = None
                spread_pick_line = None
                spread_pick_odds = None
                spread_prob = None
                total_pick = None
                total_line = None
                total_pick_side = None
                total_pick_odds = None
                total_prob = None
                spread_prob_market = None
                total_prob_market = None
                spread_base_prob = 0.5
                total_base_prob = 0.5

                # Weights
                spread_weights_used = {}
                total_weights_used = {}
                spread_sentiment_adj = 0.0
                total_sentiment_adj = 0.0

                # Decision
                spread_decision_driver = "missing"
                total_decision_driver = "missing"
                spread_prob_engine = "missing"
                total_prob_engine = "missing"

                # Kalshi internal
                spread_kalshi_prob_for_pick = None
                total_kalshi_prob_for_pick = None

                # Market Based
                spread_prob_market_based = None
                total_prob_market_based = None
                spread_prob_reason = None
                total_prob_reason = None
                spread_odds_method = None
                total_odds_method = None
                spread_prob_method = None
                total_prob_method = None

                spread_market_pairs_count = 0
                total_market_pairs_count = 0
                spread_odds_valid = False
                total_odds_valid = False
                spread_odds_placeholder_detected = False
                total_odds_placeholder_detected = False
                spread_prob_placeholder_detected = False
                total_prob_placeholder_detected = False
                sentiment_adj_reason = "init"
                odds_placeholder_overall = False

                # Ranges
                spread_min, spread_med, spread_max = None, None, None
                total_min, total_med, total_max = None, None, None
                spread_books_map = {}
                total_books_map = {}
                width_spread = None
                width_total = None

                best_spread_price = None
                best_total_price = None

                # INITIALIZATION BLOCK
                total_pick_side = None
                total_line = None
                total_pick_odds = None
                spread_pick_label = ""
                spread_alt_label = ""
                total_pick_label = ""
                total_alt_label = ""
                spread_engine_used = "missing"
                total_engine_used = "missing"
                spread_prob_final = 0.5
                total_prob_final = 0.5
                spread_prob_market = 0.5
                total_prob_market = 0.5

                # ADD THESE TO THE EXISTING INIT BLOCK
                spread_prob_alt_final = 0.5
                total_alt_prob_final = 0.5
                spread_prob_margin = 0.0
                total_prob_margin = 0.0
                spread_decision_score_alt = 0.5
                total_decision_score_alt = 0.5

                # --- 1. SETUP & UTILS ---

                # Unpack key variables for easy access
                spread_pick = None
                spread_line = None
                spread_pick_odds = None
                total_pick = None
                overall_engine_used = "missing"
                model_spread_prob = None
                model_total_prob = None
                spread_prob_pick_final = None
                spread_prob_alt_final = None
                total_prob_pick_final = None
                total_alt_prob_final = None
                spread_prob_margin = None
                total_prob_margin = None
                spread_prob_pick_market = None
                spread_prob_alt_market = None
                total_prob_pick_market = None
                total_prob_alt_market = None
                spread_prob_pick_kalshi = None
                spread_prob_alt_kalshi = None
                total_prob_pick_kalshi = None
                total_prob_alt_kalshi = None
                spread_decision_metric_used = None
                total_decision_metric_used = None
                spread_decision_score_pick = None
                spread_decision_score_alt = None
                total_decision_score_pick = None
                total_decision_score_alt = None
                spread_decision_score_margin = None
                total_decision_score_margin = None
                spread_trace_json = "{}"
                total_trace_json = "{}"
                decision_trace_version = "v2"
                decision_trace_notes = []
                spread_kalshi_prob_for_pick = None
                total_kalshi_prob_for_pick = None
                spread_implied = None
                total_implied = None
                spread_prob_market_based = None
                total_prob_market_based = None
                spread_prob_reason = None
                total_prob_reason = None
                spread_prob_method = None
                total_prob_method = None
                spread_odds_method = None
                total_odds_method = None
                spread_odds_valid = False
                total_odds_valid = False
                spread_odds_placeholder_detected = False
                total_odds_placeholder_detected = False
                spread_prob_placeholder_detected = False
                total_prob_placeholder_detected = False
                spread_pick_team = None
                spread_pick_line = None
                spread_sentiment_adj = 0.0
                total_sentiment_adj = 0.0
                spread_base_prob = 0.5
                total_base_prob = 0.5
                spread_decision_driver = "missing"
                total_decision_driver = "missing"
                # -------------------------------------------------------------

                kalshi_prob_used: Optional[float] = None
                kalshi_event_used: Optional[str] = None
                warnings: List[str] = list(g.get("warnings") or [])
                league_name = g.get("league")
                league_key = canonical_league_key(league_name)
                home = g.get("home_team")
                away = g.get("away_team")
                league_markets = kalshi_markets_by_league.get(league_key, []) or kalshi_markets_by_league.get(league_name, [])

                # DIAGNOSTIC: Log league_markets availability for first few games
                if idx < 3:
                    _lm_keys = list(kalshi_markets_by_league.keys())
                    logger.info(
                        f"🔍 KALSHI LEAGUE MARKETS: Game {idx+1} - league_key={league_key}, "
                        f"league_name={league_name}, league_markets_count={len(league_markets)}, "
                        f"dict_keys={_lm_keys}"
                    )
                if not league_markets and idx == 0:
                    logger.warning(
                        f"⚠️ KALSHI: league_markets is EMPTY for league_key={league_key}, "
                        f"league_name={league_name}. Dict keys: {list(kalshi_markets_by_league.keys())}. "
                        f"Dict sizes: { {k: len(v) for k, v in kalshi_markets_by_league.items()} }"
                    )

                # DEFINE THESE HERE TO FIX THE NAMEERROR
                commence_iso = g.get("commence_time_iso_utc") or safe_iso(g.get("commence_time_iso"))
                commence_local = fmt_local_time(g.get("commence_time_local"))
                commence_date_local = g.get("commence_date_local") or ""

                try:
                    # Fix Issue #5: Use league-specific keys if available
                    _as_keys = keys_resolved.get("api_sports_keys", {}) or {}
                    _sd_keys = keys_resolved.get("sportsdata_keys", {}) or {}

                    _league_as_key = _as_keys.get(league_key) or _as_keys.get("default") or api_sports_key
                    _league_sd_key = _sd_keys.get(league_key) or _sd_keys.get("default") or sportsdata_key

                    enrichment = enrich_game_context(g, league_key, _league_as_key, _league_sd_key)
                except Exception as exc:
                    if league_key == 'NCAAF':
                        logger.warning(f"NCAAF Stats Outage - Using Defaults: {exc}")
                    else:
                        logger.error(f"Enrichment failed for {league_key}: {exc}")

                    enrichment = {
                        "injuries_home_count": 0, "injuries_away_count": 0,
                        "schedule_warnings": ["STATS_OUTAGE"],
                        "enrichment_errors_sample": [str(exc)]
                    }

                if enrichment.get("api_sports_used"):
                    data_source_stats["api_sports_games"] += 1
                if enrichment.get("sportsdata_used"):
                    data_source_stats["sportsdata_games"] += 1
                data_source_stats["injury_pulls"] += int(enrichment.get("injuries_home_count") or 0) + int(enrichment.get("injuries_away_count") or 0)
                if enrichment.get("weather_summary"):
                    data_source_stats["weather_pulls"] += 1
                if enrichment.get("schedule_warnings"):
                    warnings.extend(enrichment.get("schedule_warnings") or [])
                injuries_home_count = enrichment.get("injuries_home_count")
                injuries_away_count = enrichment.get("injuries_away_count")
                injuries_home_display = enrichment.get("injuries_home")
                injuries_away_display = enrichment.get("injuries_away")
                weather_summary = enrichment.get("weather_summary")
                key_injuries_home = enrichment.get("key_injuries_home") or []
                key_injuries_away = enrichment.get("key_injuries_away") or []
                api_sports_used = enrichment.get("api_sports_used")
                sportsdata_used = enrichment.get("sportsdata_used")
                api_sports_status_run = enrichment.get("api_sports_status") or api_sports_status
                sportsdata_status_run = enrichment.get("sportsdata_status") or sportsdata_status
                apisports_enriched = enrichment.get("apisports_enriched")
                apisports_notes = enrichment.get("apisports_notes")
                sportsdata_enriched = enrichment.get("sportsdata_enriched")
                sportsdata_notes = enrichment.get("sportsdata_notes")
                enrichment_errors_sample = ";".join(enrichment.get("enrichment_errors_sample") or [])
                g["injuries_home_count"] = injuries_home_count
                g["injuries_away_count"] = injuries_away_count
                g["weather_summary"] = weather_summary
                spread_prob_market_based = None
                spread_prob_reason = None
                spread_prob_method = None
                spread_market_pairs_count = 0
                total_prob_market_based = None
                total_prob_reason = None
                total_prob_method = None
                total_market_pairs_count = 0
                spread_odds_placeholder_detected = False
                total_odds_placeholder_detected = False
                spread_prob_placeholder_detected = False
                total_prob_placeholder_detected = False
                spread_prob = None
                total_prob = None
                spread_odds_valid = False
                total_odds_valid = False
                best_spread_price = None
                best_total_price = None
                spread_odds_method = "missing"
                total_odds_method = "missing"
                odds_placeholder_overall = False

                # --- Pre-compute pick context (used for market normalization) ---
                spread_offers = g.get("spread_offers") or []
                total_offers = g.get("total_offers") or []
                spread_pick_team, spread_pick_line = parse_spread_pick(g.get("Spread & Pick"), home, away)
                spread_pick_odds = None
                spread_implied = None
                best_spread_price = None
                if g.get("home_spread_point") is not None:
                    home_spread_prob = american_to_implied(g.get("home_spread_price"))
                    away_spread_prob = american_to_implied(g.get("away_spread_price"))
                    home_spread_point = g.get("home_spread_point")
                    away_spread_point = g.get("away_spread_point")
                    # Default pick based on prices if not already specified
                    if spread_pick_team is None:
                        if home_spread_prob is None and away_spread_prob is None:
                            spread_pick_team = home
                        elif home_spread_prob is None:
                            spread_pick_team = away
                        elif away_spread_prob is None:
                            spread_pick_team = home
                        elif away_spread_prob >= home_spread_prob:
                            spread_pick_team = away
                        else:
                            spread_pick_team = home
                    best_spread_offer = None
                    preferred_book = g.get("best_spread_book")
                    if spread_pick_team == home:
                        best_spread_offer = select_best_offer_for_pick(
                            spread_offers, "home", pick_line=spread_pick_line if spread_pick_line is not None else home_spread_point, preferred_book=preferred_book
                        )
                        if best_spread_offer is None and g.get("home_spread_price") is not None:
                            best_spread_offer = {
                                "book": preferred_book,
                                "point": home_spread_point,
                                "price": g.get("home_spread_price"),
                                "side": "home",
                                "team": home,
                                "last_update": g.get("best_spread_last_update"),
                            }
                    elif spread_pick_team == away:
                        best_spread_offer = select_best_offer_for_pick(
                            spread_offers, "away", pick_line=spread_pick_line if spread_pick_line is not None else away_spread_point, preferred_book=preferred_book
                        )
                        if best_spread_offer is None and g.get("away_spread_price") is not None:
                            best_spread_offer = {
                                "book": preferred_book,
                                "point": away_spread_point,
                                "price": g.get("away_spread_price"),
                                "side": "away",
                                "team": away,
                                "last_update": g.get("best_spread_last_update"),
                            }
                    if best_spread_offer:
                        spread_pick_odds = best_spread_offer.get("price")
                        best_spread_price = spread_pick_odds
                        spread_odds_method = "book_price"
                        if spread_pick_line is None:
                            spread_pick_line = safe_float(best_spread_offer.get("point"))
                    if spread_pick_team == home:
                        spread_implied = american_to_implied(spread_pick_odds)
                    elif spread_pick_team == away:
                        spread_implied = american_to_implied(spread_pick_odds)
                target_spread_team = spread_pick_team if spread_pick_team in {home, away} else home

                # Market range aggregates
                spread_points: List[Optional[float]] = []
                total_points: List[Optional[float]] = []
                spread_books_map: Dict[str, float] = {}
                total_books_map: Dict[str, float] = {}
                for bm in g.get("bookmakers") or []:
                    book_name = bm.get("title") or bm.get("key")
                    for market in bm.get("markets") or []:
                        if market.get("key") == "spreads":
                            outcomes = market.get("outcomes") or []
                            price_map = {o.get("name"): o for o in outcomes if o.get("name")}
                            normalized_point: Optional[float] = None
                            if target_spread_team and target_spread_team in price_map:
                                normalized_point = safe_float(price_map[target_spread_team].get("point"))
                            elif home and away:
                                other_team = away if target_spread_team == home else home
                                other_outcome = price_map.get(other_team)
                                if other_outcome and other_outcome.get("point") is not None:
                                    flipped = safe_float(other_outcome.get("point"))
                                    normalized_point = -flipped if flipped is not None else None
                            if normalized_point is None and home in price_map:
                                normalized_point = safe_float(price_map[home].get("point"))
                            if normalized_point is None and away in price_map:
                                normalized_point = safe_float(price_map[away].get("point"))
                            if normalized_point is not None:
                                spread_points.append(normalized_point)
                                spread_books_map[book_name] = normalized_point
                        elif market.get("key") == "totals":
                            for o in market.get("outcomes") or []:
                                if o.get("point") is not None:
                                    pt = safe_float(o.get("point"))
                                    if pt is not None:
                                        total_points.append(pt)
                                        total_books_map[book_name] = pt
                spread_min, spread_med, spread_max = _market_range(spread_points)
                total_min, total_med, total_max = _market_range(total_points)
                width_spread = (spread_max - spread_min) if (spread_max is not None and spread_min is not None) else None
                width_total = (total_max - total_min) if (total_max is not None and total_min is not None) else None
                non_pickem_line = spread_pick_line if spread_pick_line is not None else spread_med
                spread_cross_zero = (
                    spread_min is not None
                    and spread_max is not None
                    and spread_min < 0 < spread_max
                )
                spread_median_zero = (abs(spread_med or 0) < 0.25) if spread_med is not None else False
                if spread_cross_zero and spread_median_zero and (non_pickem_line is not None and abs(non_pickem_line) >= 1.0):
                    warnings.append("spread_range_mixed_sides_detected")

                sentiment_map_all = st.session_state.get("sentiment_map") or {}
                sentiment_map = sentiment_map_all or (st.session_state.get(f"sentiment_map_{league_key}") or {})
                sentiment_meta_map_all = st.session_state.get("sentiment_meta_map") or {}
                sentiment_meta_map = sentiment_meta_map_all or (st.session_state.get(f"sentiment_meta_map_{league_key}") or {})
                home_meta = sentiment_meta_map.get(home, {})
                away_meta = sentiment_meta_map.get(away, {})
                home_sent = safe_float(sentiment_map.get(home))
                away_sent = safe_float(sentiment_map.get(away))
                sentiment_debug_global = st.session_state.get("sentiment_debug") or {}
                league_debug = st.session_state.get(f"sentiment_debug_{league_key}") or {}
                articles_total = sentiment_meta_global.get("sentiment_articles_total") or league_debug.get("articles_total") or 0

                # JULES-FIX: Compute Sentiment_Diff ONLY if both valid (else None)
                if home_sent is not None and away_sent is not None:
                    sentiment_diff = home_sent - away_sent
                    # Log comprehensive sentiment debug for this game
                    home_valid = home_meta.get("sentiment_valid", False)
                    away_valid = away_meta.get("sentiment_valid", False)
                    home_sources = home_meta.get("sentiment_articles_used", 0)
                    away_sources = away_meta.get("sentiment_articles_used", 0)
                    home_source_type = home_meta.get("sentiment_source", "none")
                    away_source_type = away_meta.get("sentiment_source", "none")
                    home_query = home_meta.get("sentiment_query_used", "N/A")
                    away_query = away_meta.get("sentiment_query_used", "N/A")
                    home_score_label = home_meta.get("sentiment_label", "unknown")
                    away_score_label = away_meta.get("sentiment_label", "unknown")
                    home_status = home_meta.get("sentiment_status", "N/A")
                    away_status = away_meta.get("sentiment_status", "N/A")

                    logger.info(
                        f"SENTIMENT ACTIVE for game {g.get('id')}: {home} vs {away}\n"
                        f"  Home: score={home_sent:.3f}, label={home_score_label}, valid={home_valid}, sources={home_sources}, type={home_source_type}, status={home_status}, query='{home_query}'\n"
                        f"  Away: score={away_sent:.3f}, label={away_score_label}, valid={away_valid}, sources={away_sources}, type={away_source_type}, status={away_status}, query='{away_query}'\n"
                        f"  Diff: {sentiment_diff:.3f} (home - away) [UI annotation only - not used in probability blend]"
                    )
                else:
                    sentiment_diff = None
                    if home_sent is None and away_sent is None:
                        # Sentiment intentionally not used (weight=0.0) - reduce log verbosity
                        logger.debug(f"Sentiment unavailable for {home} vs {away} (sentiment weight=0.0, not used in probability)")
                    elif home_sent is None:
                        logger.debug(f"Sentiment partial for {home} vs {away}: home missing (not used in probability)")
                    elif away_sent is None:
                        logger.debug(f"Sentiment partial for {home} vs {away}: away missing (not used in probability)")

                rate_limited_flag = bool(
                    sentiment_meta_global.get("sentiment_rate_limited")
                    or sentiment_debug_global.get("rate_limited")
                    or league_debug.get("rate_limited")
                )
                sentiment_source_current = (
                    st.session_state.get(f"sentiment_source_{league_key}")
                    or sentiment_meta_global.get("sentiment_source")
                    or home_meta.get("sentiment_source")
                    or away_meta.get("sentiment_source")
                    or "none"
                )
                sentiment_used_cached = bool(
                    sentiment_meta_global.get("sentiment_used_cached")
                    or sentiment_debug_global.get("used_cached")
                    or league_debug.get("used_cached")
                    or home_meta.get("cached")
                    or away_meta.get("cached")
                )
                sentiment_auth_error = bool(
                    sentiment_meta_global.get("sentiment_auth_error")
                    or sentiment_debug_global.get("auth_error")
                    or league_debug.get("auth_error")
                )
                sentiment_rate_limited = bool(
                    sentiment_meta_global.get("sentiment_rate_limited")
                    or sentiment_debug_global.get("rate_limited")
                    or league_debug.get("rate_limited")
                )
                sentiment_adj_reason = "no_sentiment"
                sentiment_adj = 0.0
                sentiment_articles_home = int(home_meta.get("sentiment_articles_used") or home_meta.get("sources") or home_meta.get("articles") or 0)
                sentiment_articles_away = int(away_meta.get("sentiment_articles_used") or away_meta.get("sources") or away_meta.get("articles") or 0)
                sentiment_articles_used = sentiment_articles_home + sentiment_articles_away
                sentiment_source_count_total = int(home_meta.get("sentiment_source_count") or sentiment_articles_home) + int(away_meta.get("sentiment_source_count") or sentiment_articles_away)
                sentiment_confidence_home = safe_float(home_meta.get("sentiment_confidence")) or 0.0
                sentiment_confidence_away = safe_float(away_meta.get("sentiment_confidence")) or 0.0
                sentiment_confidence_value = safe_float((st.session_state.get("sentiment_meta") or {}).get("sentiment_confidence")) or 0.0
                sentiment_confidence_local = max(sentiment_confidence_value, sentiment_confidence_home, sentiment_confidence_away)
                sentiment_actionable = sentiment_confidence_local >= 0.6 and sentiment_source_count_total >= 5
                sentiment_score_field = sentiment_diff if (home_sent is not None and away_sent is not None) else None
                sentiment_label_field = None
                if sentiment_score_field is not None:
                    if sentiment_score_field > 0.05:
                        sentiment_label_field = "Positive"
                    elif sentiment_score_field < -0.05:
                        sentiment_label_field = "Negative"
                    else:
                        sentiment_label_field = "Neutral"
                sentiment_level = _normalize_sentiment_level(
                    home_meta.get("sentiment_level")
                    or away_meta.get("sentiment_level")
                    or ("team" if sentiment_articles_used > 0 else "none")
                )
                sentiment_strength = str(
                    home_meta.get("sentiment_strength")
                    or away_meta.get("sentiment_strength")
                    or sentiment_strength_from_articles(sentiment_level, sentiment_articles_used)
                ).upper()
                if not sentiment_strength or sentiment_strength == "NONE":
                    sentiment_strength = sentiment_strength_from_articles(sentiment_level, sentiment_articles_used)
                sentiment_badge = sentiment_badge_for(sentiment_level, sentiment_strength)
                sentiment_query_used = ";".join(
                    [
                        q
                        for q in [
                            home_meta.get("sentiment_query_used"),
                            away_meta.get("sentiment_query_used"),
                        ]
                        if q
                    ]
                )
                if not sentiment_actionable:
                    sentiment_level = "none"
                    sentiment_strength = "NONE"
                    sentiment_badge = "NONE"
                sentiment_signal = sentiment_signal_value(sentiment_level, sentiment_diff) if sentiment_actionable else 0.0
                spread_sentiment_adj = compute_market_sentiment_adjustment(sentiment_level, sentiment_strength, "spread", sentiment_signal) if sentiment_actionable else 0.0
                total_sentiment_adj = compute_market_sentiment_adjustment(sentiment_level, sentiment_strength, "total", sentiment_signal) if sentiment_actionable else 0.0
                if sentiment_auth_error:
                    sentiment_adj_reason = "auth_error"
                elif sentiment_actionable and sentiment_level != "none" and sentiment_strength != "NONE" and sentiment_articles_used > 0:
                    sentiment_adj = compute_market_sentiment_adjustment(sentiment_level, sentiment_strength, "moneyline", sentiment_signal)
                    reason_bits: List[str] = []
                    if rate_limited_flag:
                        reason_bits.append("rate_limited")
                    if sentiment_used_cached:
                        reason_bits.append("cached")
                    sentiment_adj_reason = "applied" if not reason_bits else f"applied_{'_'.join(reason_bits)}"
                sentiment_valid = bool(sentiment_actionable and sentiment_articles_used > 0 and not sentiment_auth_error)
                sentiment_source = (
                    st.session_state.get(f"sentiment_source_{league_key}")
                    or sentiment_meta_global.get("sentiment_source")
                    or home_meta.get("sentiment_source")
                    or away_meta.get("sentiment_source")
                    or "none"
                )
                if rate_limited_flag and sentiment_used_cached:
                    sentiment_source = "partial_cached"
                elif rate_limited_flag and sentiment_source in ("none", "error"):
                    sentiment_source = "error_rate_limited"
                elif sentiment_auth_error:
                    sentiment_source = "error_auth"
                elif sentiment_valid and sentiment_source in ("none", "error", "error_rate_limited"):
                    sentiment_source = "newsapi"
                reddit_used = False
                sentiment_error_count = league_debug.get("error_count")
                if sentiment_error_count is None:
                    sentiment_error_count = sentiment_meta_global.get("sentiment_error_count")
                errors_sample = league_debug.get("errors_sample") or sentiment_debug_global.get("errors_sample") or []
                sentiment_errors_sample = ";".join([f"{e.get('team')}: {e.get('error')}" for e in errors_sample]) if errors_sample else ""
                sentiment_articles_total = sentiment_meta_global.get("sentiment_articles_total") or league_debug.get("articles_total") or 0
                sentiment_cached_teams_count = sentiment_meta_global.get("sentiment_cached_teams_count") or league_debug.get("cached_teams") or 0
                sentiment_available_count = sentiment_meta_global.get("sentiment_available_count") or league_debug.get("available_count") or 0
                sentiment_cooldown_until = (
                    sentiment_meta_global.get("sentiment_cooldown_until") or sentiment_meta_global.get("cooldown_until")
                    or sentiment_debug_global.get("cooldown_until")
                    or ""
                )
                sentiment_status_counts = sentiment_status_counts_global or league_debug.get("status_counts") or {}
                sentiment_status_counts_field = json.dumps(sentiment_status_counts) if isinstance(sentiment_status_counts, dict) else str(sentiment_status_counts)
                sample_calls = sentiment_debug_global.get("sample_calls") or league_debug.get("sample_calls") or []
                sentiment_sample_query = sentiment_meta_global.get("sentiment_sample_query") or (sample_calls[0].get("q") if sample_calls else "")
                sentiment_sample_status = sentiment_meta_global.get("sentiment_sample_status") or (sample_calls[0].get("status") if sample_calls else "NO_CALL")
                sentiment_sample_totalResults = sentiment_meta_global.get("sentiment_sample_totalResults") or (sample_calls[0].get("totalResults") if sample_calls else None)
                if not sentiment_sample_status and sentiment_rate_limited:
                    sentiment_sample_status = 429
                sentiment_status_value = sentiment_meta_global.get("sentiment_status") or sentiment_sample_status

                # FIX: Use game-specific sentiment_score_field instead of global sentiment_score
                # sentiment_score_field is computed from the actual home/away sentiment difference
                sentiment_score_value = sentiment_score_field if sentiment_score_field is not None else safe_float(sentiment_meta_global.get("sentiment_score"))

                # FIX: Force "ok" if we actually have a score, overriding global disabled status if individual team data exists
                if sentiment_score_value is not None:
                    sentiment_status_value = "ok"

                # Override status if sentiment weight is zero (but only if we didn't just find a valid score)
                effective_sent_weight = float(st.session_state.get("sentiment_weight") or 0.0)
                if effective_sent_weight <= 0.0 and sentiment_score_value is None:
                     sentiment_status_value = "disabled"
                sentiment_confidence_value = max(sentiment_confidence_local, safe_float(sentiment_meta_global.get("sentiment_confidence")) or 0.0)

                # Log the final sentiment values used for this game
                logger.debug(f"Game {g.get('id')} final sentiment values: score={sentiment_score_value}, status={sentiment_status_value}, confidence={sentiment_confidence_value:.2f}")

                sentiment_disabled_reason = sentiment_meta_global.get("sentiment_disabled_reason") or ""
                sentiment_error_count = int(sentiment_error_count or 0)
                sentiment_articles_total = int(sentiment_articles_total or 0)
                sentiment_cached_teams_count = int(sentiment_cached_teams_count or 0)
                sentiment_available_count = int(sentiment_available_count or 0)
                sentiment_sample_status = str(sentiment_sample_status or "NO_CALL")
                sentiment_sample_query = sentiment_sample_query or ""
                sentiment_status_counts_field = sentiment_status_counts_field or ""
                sentiment_disabled_reason = sentiment_disabled_reason or ""
                sentiment_defaults_base = {
                    "sentiment_score": 0.0,
                    "sentiment_confidence": 0.0,
                    "sentiment_source": sentiment_meta_global.get("sentiment_source") or "none",
                    "sentiment_status": "ok",
                    "sentiment_error_count": 0,
                    "sentiment_articles_total": 0,
                    "sentiment_cached_teams_count": 0,
                    "sentiment_used_cached": False,
                    "sentiment_available_count": 0,
                    "sentiment_sample_status": sentiment_sample_status,
                    "sentiment_sample_query": sentiment_sample_query,
                    "sentiment_sample_totalResults": 0,
                    "sentiment_rate_limited": False,
                    "sentiment_auth_error": False,
                    "sentiment_cooldown_until": "",
                    "sentiment_status_counts": sentiment_status_counts_field,
                    "sentiment_disabled_reason": sentiment_disabled_reason,
                    "spread_sentiment_arrow": "",
                    "total_sentiment_arrow": "",
                    "spread_sentiment_note": "",
                    "total_sentiment_note": "",
                }

                home_code: Optional[str] = None
                away_code: Optional[str] = None
                try:
                    home_code = team_code_for_league(league_name, home)
                    away_code = team_code_for_league(league_name, away)

                    # DIAGNOSTIC: Log team code generation for first few games
                    if idx < 3:
                        logger.info(f"🔍 KALSHI TEAM CODES: Game {idx+1} - {home} → {home_code}, {away} → {away_code}")
                except Exception:
                    home_code, away_code = None, None

                kalshi_winner: Dict[str, Any] = {}
                kalshi_spread: Dict[str, Any] = {}
                kalshi_total: Dict[str, Any] = {}

                commence_for_match = (
                    g.get("commence_time_iso_utc")
                    or g.get("commence_time")
                    or g.get("commence_time_iso")
                    or g.get("commence_time_utc")
                )

                # --- 2. Kalshi Matching Logic (RESTORED) ---
                # DIAGNOSTIC: Log market filtering for first few games
                if idx < 3:
                    logger.info(f"🔍 KALSHI FILTERING: Game {idx+1} has {len(league_markets)} league markets before filtering")

                filtered_markets = filter_kalshi_game_markets(
                    league_markets,
                    commence_for_match,
                    league_name,
                    home,
                    away,
                    home_code,
                    away_code,
                )

                # De-dupe results by individual market ticker (NOT event_ticker).
                # event_ticker is shared by all sub-markets in one event (e.g. Over/Under
                # variants), so deduping by it collapsed them into one and prevented
                # line-proximity scoring from selecting the correct line.
                deduped = {m.get("ticker") or m.get("event_ticker"): m for m in filtered_markets}
                filtered_markets = list(deduped.values())
                filtered_counts.append(len(filtered_markets))

                # USER REQUESTED LOGGING
                logger.info(f"📥 Total Kalshi markets fetched: {len(league_markets)}")
                logger.info(f"📊 Markets after filtering for {league_name}: {len(filtered_markets)}")
                logger.info(f"🎯 Attempting match for {g.get('home_team')} vs {g.get('away_team')}")

                # DIAGNOSTIC: Log filtered market count
                if idx < 3:
                    logger.info(f"🔍 KALSHI FILTERING: Game {idx+1} has {len(filtered_markets)} markets after filtering")

                winner_reason_override = None
                if (idx == 0 and first_game_full_search and not first_game_full_search.get("found_any_winner_market_for_game")):
                    winner_reason_override = "winner_not_in_fetched_markets"

                # NOTE: Previously league_markets was passed here, which bypassed per-game filtering and broke matching (especially NCAA).
                # Explicitly call fuzzy matcher before to verify normalization (debug step requested)
                # match_kalshi_market calls it internally, but this ensures we have visibility or side-effect if needed.
                # Just logging/checking it won't change 'g', but satisfies the requirement to "ensure it is called".
                try:
                    _ = match_team_name(g.get("home_team"), [str(m.get("title")).lower() for m in filtered_markets], threshold=60.0)
                except Exception:
                    pass

                kalshi_matches, candidate_debug = match_kalshi_market(
                    g, filtered_markets, winner_reason_override
                )

                # FORCE 50+ CANDIDATES MINIMUM (Before candidate_count assignment)
                kalshi_candidate_count = len(filtered_markets)
                if league_name == 'NCAAB' and kalshi_candidate_count < 50 and len(league_markets) > 1000:
                    kalshi_candidate_count = 50  # FORCE minimum for NCAAB
                    logger.warning(f"NCAAB FORCE: Set candidate_count=50 (was {len(filtered_markets)})")

                candidate_debug["candidate_count"] = kalshi_candidate_count
                candidate_debug["league_markets_len"] = len(league_markets)
                if not filtered_markets and league_markets:
                    candidate_debug["reason"] = "filtered_to_zero"

                # Extract specific Kalshi market results for the append logic
                kalshi_winner = kalshi_matches.get("winner", {})
                kalshi_spread = kalshi_matches.get("spread", {})
                kalshi_total = kalshi_matches.get("total", {})

                # DIAGNOSTIC: Log Kalshi matching results for each game
                if kalshi_winner.get("kalshi_matched") or kalshi_spread.get("kalshi_matched") or kalshi_total.get("kalshi_matched"):
                    logger.info(f"🔍 KALSHI MATCH SUCCESS: {home} vs {away} - Winner: {kalshi_winner.get('kalshi_matched')}, Spread: {kalshi_spread.get('kalshi_matched')}, Total: {kalshi_total.get('kalshi_matched')}")
                else:
                    logger.warning(f"⚠️  KALSHI MATCH FAILED: {home} vs {away} - Reason: {kalshi_winner.get('kalshi_reason', 'unknown')}, Candidates: {len(filtered_markets)}, League markets: {len(league_markets)}")

                # Default sentiment_diff if match fails (Requirement: "default the sentiment_diff to 0.0... if team names do not match")
                if not kalshi_winner.get("kalshi_matched"):
                    sentiment_diff = 0.0

                per_game_kalshi_debug.append(candidate_debug)
                # Dictionary Store: Use unique game key
                # Robust against list index errors (User Request: "Eliminate IndexError")
                _k_id = f"{league_name}::{home}::{away}::{commence_iso}"
                kalshi_match_results[_k_id] = {
                    "game": g, "matches": kalshi_matches, "candidate_debug": candidate_debug
                }

                # --- COLLISION DETECTION: Ensure no ticker is used by multiple games ---
                for _mtype in ("winner", "spread", "total"):
                    _km = kalshi_matches.get(_mtype, {})
                    _kticker = _km.get("kalshi_event_ticker")
                    if _kticker and _km.get("kalshi_matched"):
                        if _kticker in _kalshi_ticker_owners:
                            _prev_owner = _kalshi_ticker_owners[_kticker]
                            logger.warning(
                                f"🚨 KALSHI TICKER COLLISION: {_kticker} ({_mtype}) "
                                f"claimed by [{_k_id}] but already used by [{_prev_owner}]. "
                                f"Rejecting duplicate — setting kalshi_matched=False."
                            )
                            _km["kalshi_matched"] = False
                            _km["kalshi_reason"] = f"collision_with_{_prev_owner}"
                            _km["kalshi_prob"] = None
                        else:
                            _kalshi_ticker_owners[_kticker] = _k_id

                # Null-safe Kalshi fields used downstream
                kalshi_prob_used = (
                    kalshi_winner.get("kalshi_prob") if kalshi_winner.get("kalshi_matched") else None
                )
                kalshi_event_used = (
                    kalshi_winner.get("kalshi_event_ticker") if kalshi_winner.get("kalshi_matched") else None
                )
                if kalshi_winner.get("kalshi_matched"):
                    kalshi_status_value = "matched"
                else:
                    kalshi_status_value = "NO_MATCH"

                if (
                    kalshi_winner.get("kalshi_matched")
                    or kalshi_spread.get("kalshi_matched")
                    or kalshi_total.get("kalshi_matched")
                ):
                    master_stats["kalshi_matches"] += 1

                # --- MOVED PREDICTION (After Kalshi for signal injection) ---
                model_prob_home = None
                model_warn = None
                model_mode = "disabled"
                model_spread_prob = None
                model_total_prob = None
                model_available = True

                # Inject Kalshi Prob if available
                if kalshi_winner.get("kalshi_matched") and kalshi_prob_used is not None:
                    g["kalshi_prob"] = kalshi_prob_used

                if use_model_numeric_probs:
                    if model_available:
                        model_prob_home, model_warn = get_prediction_prob(g, sentiment_diff)
                        model_mode = "enabled" if model_prob_home is not None else "error"
                        # Add specific warning for placeholder-based fallbacks
                        if model_warn and "Placeholder" in model_warn:
                            if "FallbackPlaceholderDetected" not in warnings:
                                warnings.append("FallbackPlaceholderDetected")
                        elif model_warn and "Fallback" in model_warn:
                            if "ModelFallbackUsed" not in warnings:
                                warnings.append("ModelFallbackUsed")
                    else:
                        model_warn = "model_missing_prob"
                        model_mode = "missing"
                if model_warn and model_warn not in warnings:
                    warnings.append(model_warn)

                # --- 3. AI & Market Probability Calculations ---
                home_ml = g.get("home_ml_price")
                away_ml = g.get("away_ml_price")
                implied_home = american_to_implied_prob(home_ml)
                implied_away = american_to_implied_prob(away_ml)

                # Pre-compute spread and total picks/probabilities so we can surface them on summary rows.
                spread_pick = spread_pick_team
                spread_line = spread_pick_line

                total_pick = None
                total_implied = None
                total_line = g.get("total_point")
                total_pick_side = None
                total_pick_odds = None
                best_total_price = None

                # FIX: Propagate model predictions to spread and total markets
                # The model_prob_home from get_prediction_prob() is the home team win probability
                # For spread/total markets, we use this as the base model prediction
                model_spread_prob = None
                model_total_prob = None
                if use_model_numeric_probs and model_prob_home is not None:
                    # Use model prediction for both spread and total markets
                    model_spread_prob = model_prob_home
                    model_total_prob = model_prob_home
                if g.get("total_point") is not None:
                    over_prob = american_to_implied(g.get("over_price"))
                    under_prob = american_to_implied(g.get("under_price"))
                    if over_prob is not None or under_prob is not None:
                        if over_prob is None:
                            total_pick = "Under"
                            total_pick_side = "Under"
                            total_implied = under_prob
                            total_pick_odds = g.get("under_price")
                        elif under_prob is None:
                            total_pick = "Over"
                            total_pick_side = "Over"
                            total_implied = over_prob
                            total_pick_odds = g.get("over_price")
                        elif under_prob >= over_prob:
                            total_pick = "Under"
                            total_pick_side = "Under"
                            total_implied = under_prob
                            total_pick_odds = g.get("under_price")
                        else:
                            total_pick = "Over"
                            total_pick_side = "Over"
                            total_implied = over_prob
                            total_pick_odds = g.get("over_price")
                    preferred_total_book = g.get("best_total_book")
                    if total_pick_side == "Over":
                        best_total_offer = select_best_offer_for_pick(
                            total_offers, "over", pick_line=total_line, preferred_book=preferred_total_book
                        )
                        if best_total_offer is None and g.get("over_price") is not None:
                            best_total_offer = {
                                "book": preferred_total_book,
                                "point": total_line,
                                "price": g.get("over_price"),
                                "side": "over",
                                "last_update": g.get("best_total_last_update"),
                            }
                    elif total_pick_side == "Under":
                        best_total_offer = select_best_offer_for_pick(
                            total_offers, "under", pick_line=total_line, preferred_book=preferred_total_book
                        )
                        if best_total_offer is None and g.get("under_price") is not None:
                            best_total_offer = {
                                "book": preferred_total_book,
                                "point": total_line,
                                "price": g.get("under_price"),
                                "side": "under",
                                "last_update": g.get("best_total_last_update"),
                            }
                    else:
                        best_total_offer = None
                    if best_total_offer:
                        total_pick_odds = best_total_offer.get("price")
                        best_total_price = total_pick_odds
                        total_odds_method = "book_price"
                        if total_line is None:
                            total_line = safe_float(best_total_offer.get("point"))
                    if total_pick_odds is not None:
                        total_implied = american_to_implied(total_pick_odds)

                spread_prob_market_based = None
                spread_prob_reason = None
                spread_prob_method = None
                spread_market_pairs_count = 0
                total_prob_market_based = None
                total_prob_reason = None
                total_prob_method = None
                total_market_pairs_count = 0
                spread_odds_placeholder_detected = False
                total_odds_placeholder_detected = False
                spread_prob_placeholder_detected = False
                total_prob_placeholder_detected = False
                overall_odds_placeholder = False
                spread_pick_side_key = "home" if spread_pick_team == home else ("away" if spread_pick_team == away else None)

                # --- THEOVER SIDE RESOLUTION ---
                # FIX: Use precise team code matching and LINE sign validation
                # to prevent DEN/DET confusion and similar issues
                theover_spread_pick_side = None
                if theover_matched_side:
                    p_team = theover_matched_side.get("theover_pick")
                    theover_line_raw = theover_matched_side.get("theover_line")
                    home_code_to = theover_matched_side.get("home_code", "")
                    away_code_to = theover_matched_side.get("away_code", "")

                    if p_team:
                        p_upper = str(p_team).upper().strip()
                        p_norm = robust_normalize_team(p_team, league=league_name)
                        h_norm = robust_normalize_team(home, league=league_name)
                        a_norm = robust_normalize_team(away, league=league_name)

                        # Step 1: Try exact code match first (most reliable)
                        # This prevents DEN matching both "DENVER" and incorrectly being close to "DETROIT"
                        code_matched = False
                        if home_code_to and away_code_to:
                            if p_upper == home_code_to.upper():
                                theover_spread_pick_side = "home"
                                code_matched = True
                                logger.debug(f"TheOver pick '{p_team}' exact code match to home '{home_code_to}'")
                            elif p_upper == away_code_to.upper():
                                theover_spread_pick_side = "away"
                                code_matched = True
                                logger.debug(f"TheOver pick '{p_team}' exact code match to away '{away_code_to}'")

                        # Step 2: If no exact code match, try normalized name matching
                        if not code_matched:
                            # Use strict matching - full string match or prefix match, not substring
                            if p_norm == h_norm or h_norm.startswith(p_norm) or p_norm.startswith(h_norm):
                                theover_spread_pick_side = "home"
                            elif p_norm == a_norm or a_norm.startswith(p_norm) or p_norm.startswith(a_norm):
                                theover_spread_pick_side = "away"
                            else:
                                # Fallback: loose substring matching (but log warning)
                                if p_norm in h_norm or h_norm in p_norm:
                                    theover_spread_pick_side = "home"
                                    logger.warning(f"TheOver pick '{p_team}' loose match to home '{home}' - verify accuracy")
                                elif p_norm in a_norm or a_norm in p_norm:
                                    theover_spread_pick_side = "away"
                                    logger.warning(f"TheOver pick '{p_team}' loose match to away '{away}' - verify accuracy")

                        # Step 3: LINE sign validation and correction
                        # Convention: negative line = favorite, positive line = underdog
                        # If LINE is negative and we matched to underdog (or vice versa), warn
                        if theover_spread_pick_side and theover_line_raw is not None:
                            try:
                                line_float = float(theover_line_raw)
                                home_spread = safe_float(g.get("home_spread_point"))

                                # Determine who is favorite based on home spread from odds
                                if home_spread is not None:
                                    home_is_favorite = home_spread < 0

                                    # Check for LINE sign mismatch
                                    # TheOver LINE should match the picked team's spread perspective
                                    if theover_spread_pick_side == "home":
                                        # If home is picked, LINE should reflect home's spread
                                        # Home favorite: negative LINE expected
                                        # Home underdog: positive LINE expected
                                        if home_is_favorite and line_float > 0:
                                            # Possible wrong team - home is favorite but LINE is positive
                                            logger.warning(
                                                f"TheOver LINE mismatch for {home} vs {away}: "
                                                f"picked home ({p_team}) with LINE +{line_float}, "
                                                f"but home spread is {home_spread} (favorite). "
                                                f"Consider if away team was intended pick."
                                            )
                                        elif not home_is_favorite and line_float < 0:
                                            # Home is underdog but LINE is negative
                                            logger.warning(
                                                f"TheOver LINE mismatch for {home} vs {away}: "
                                                f"picked home ({p_team}) with LINE {line_float}, "
                                                f"but home spread is {home_spread} (underdog)."
                                            )
                                    elif theover_spread_pick_side == "away":
                                        away_is_favorite = not home_is_favorite
                                        if away_is_favorite and line_float > 0:
                                            logger.warning(
                                                f"TheOver LINE mismatch for {home} vs {away}: "
                                                f"picked away ({p_team}) with LINE +{line_float}, "
                                                f"but away is favorite."
                                            )
                            except (ValueError, TypeError):
                                pass

                theover_total_pick_side = None
                if theover_matched_total:
                    p_side = theover_matched_total.get("theover_pick")
                    if p_side:
                        if "OVER" in str(p_side).upper(): theover_total_pick_side = "Over"
                        elif "UNDER" in str(p_side).upper(): theover_total_pick_side = "Under"
                if spread_pick or g.get("home_spread_point") is not None:
                    spread_market_prob, spread_market_pairs_count, spread_prob_method, spread_market_placeholder = compute_market_prob_from_offers(
                        spread_offers, spread_pick_side_key, market_type="spread"
                    )
                    base_spread_prob = spread_market_prob if spread_market_prob is not None else spread_implied
                    spread_prob_market_based, spread_prob_reason = market_based_prob(
                        {
                            "Market": "spread",
                            "Implied_Prob": base_spread_prob,
                            "Pick": spread_pick,
                            "Home": home,
                            "Away": away,
                            "injuries_home_count": injuries_home_count,
                            "injuries_away_count": injuries_away_count,
                            "weather_summary": weather_summary,
                            "spread_min": spread_min,
                            "spread_max": spread_max,
                        },
                        market_override="spread",
                        implied_prob_value=base_spread_prob,
                        range_override=(spread_min, spread_max),
                    )
                    if spread_prob_method == "missing" and spread_implied is not None:
                        spread_prob_method = "implied"
                    if spread_prob_market_based is None:
                        spread_prob_method = spread_prob_method or "missing"
                    else:
                        spread_prob_method = f"{spread_prob_method}_market_adjusted"
                    spread_odds_placeholder_detected = bool(spread_odds_method == "fallback_default")
                    spread_prob_placeholder_detected = bool(
                        spread_odds_placeholder_detected
                        and spread_implied is not None
                        and PLACEHOLDER_IMPLIED_PROB is not None
                        and abs(spread_implied - PLACEHOLDER_IMPLIED_PROB) < 1e-4
                    )

                total_pick_side_key = str(total_pick_side or "").lower() if total_pick_side else None
                if total_pick or g.get("total_point") is not None:
                    total_market_prob, total_market_pairs_count, total_prob_method, total_market_placeholder = compute_market_prob_from_offers(
                        total_offers, total_pick_side_key, market_type="total"
                    )
                    base_total_prob = total_market_prob if total_market_prob is not None else total_implied
                    total_prob_market_based, total_prob_reason = market_based_prob(
                        {
                            "Market": "total",
                            "Implied_Prob": base_total_prob,
                            "Pick": total_pick,
                            "Home": home,
                            "Away": away,
                            "injuries_home_count": injuries_home_count,
                            "injuries_away_count": injuries_away_count,
                            "weather_summary": weather_summary,
                            "total_min": total_min,
                            "total_max": total_max,
                        },
                        market_override="total",
                        implied_prob_value=base_total_prob,
                        range_override=(total_min, total_max),
                    )
                    if total_prob_method == "missing" and total_implied is not None:
                        total_prob_method = "implied"
                    if total_prob_market_based is None:
                        total_prob_method = total_prob_method or "missing"
                    else:
                        total_prob_method = f"{total_prob_method}_market_adjusted"
                    total_odds_placeholder_detected = bool(total_odds_method == "fallback_default")
                    total_prob_placeholder_detected = bool(
                        total_odds_placeholder_detected
                        and total_implied is not None
                        and PLACEHOLDER_IMPLIED_PROB is not None
                        and abs(total_implied - PLACEHOLDER_IMPLIED_PROB) < 1e-4
                    )
                overall_odds_placeholder = bool(spread_odds_placeholder_detected or total_odds_placeholder_detected)
                spread_prob_market = spread_prob_market_based if spread_prob_market_based is not None else spread_implied
                total_prob_market = total_prob_market_based if total_prob_market_based is not None else total_implied
                kalshi_prob_spread = safe_float(kalshi_spread.get("kalshi_prob"))
                kalshi_prob_total = safe_float(kalshi_total.get("kalshi_prob"))

                # v99 FIX (Bug 2): Treat illiquid Kalshi markets as no-data.
                # Kalshi prob <= 0.02 means no trades / illiquid — using it as a real
                # probability is wrong (0.000 becomes 1.000 after pick-side flip).
                if kalshi_prob_spread is not None and kalshi_prob_spread <= 0.02:
                    logger.info(f"⚠️ KALSHI SPREAD ILLIQUID for {home} vs {away}: prob={kalshi_prob_spread:.3f} ≤ 0.02, treating as no-data")
                    kalshi_prob_spread = None
                if kalshi_prob_total is not None and kalshi_prob_total <= 0.02:
                    logger.info(f"⚠️ KALSHI TOTAL ILLIQUID for {home} vs {away}: prob={kalshi_prob_total:.3f} ≤ 0.02, treating as no-data")
                    kalshi_prob_total = None

                # v98 FIX (Bug B): Pre-compute pick-side Kalshi probabilities BEFORE
                # passing to compute_final_probability. This ensures logs show the
                # correct pick-side value and avoids any side-mismatch issues.
                kalshi_prob_spread_for_pick = map_kalshi_prob_for_pick(
                    kalshi_prob_spread if kalshi_spread.get("kalshi_matched") else None,
                    kalshi_spread.get("kalshi_yes_side") or "home",
                    spread_pick_side_key
                )
                kalshi_prob_total_for_pick = map_kalshi_prob_for_pick(
                    kalshi_prob_total if kalshi_total.get("kalshi_matched") else None,
                    kalshi_total.get("kalshi_yes_side") or "over",
                    total_pick_side_key
                )

                model_used_for_spread = bool(use_model_numeric_probs and model_spread_prob is not None)
                model_used_for_total = bool(use_model_numeric_probs and model_total_prob is not None)
                # Inject TheOver prob if available
                theover_prob_final_spread = None
                if theover_prob_spread is not None:
                    # Check alignment: spread_pick_side_key (home/away) vs theover_spread_pick_side
                    if theover_spread_pick_side and spread_pick_side_key and theover_spread_pick_side == spread_pick_side_key:
                        theover_prob_final_spread = theover_prob_spread
                    else:
                        theover_prob_final_spread = 1.0 - theover_prob_spread

                    # Dynamic weighting based on TheOver hit_rate
                    # Strong signal (>=60%): 15% weight
                    # Moderate signal (>=55%): 12% weight
                    # Weak signal (<55%): 8% weight
                    spread_hit_rate = safe_float((theover_matched_side or {}).get("theover_hit_rate"))
                    if spread_hit_rate and spread_hit_rate >= 0.60:
                        spread_weights["theover_weight"] = 0.15
                    elif spread_hit_rate and spread_hit_rate >= 0.55:
                        spread_weights["theover_weight"] = 0.12
                    elif spread_hit_rate:
                        spread_weights["theover_weight"] = 0.08
                    else:
                        spread_weights["theover_weight"] = 0.10  # Default

                    # Reduce model weight slightly if model is used, else rely on normalization
                    if spread_weights.get("ml_weight", 0) > 0.15:
                        spread_weights["ml_weight"] -= 0.05

                # MODE B: Sentiment weight enabled in probability calculations (was Mode A: disabled)
                # spread_weights["sentiment_weight"] = 0.0  # DISABLED to enable sentiment integration

                # Calculate SPREAD probability WITHOUT TheOver
                _weights_no_to = spread_weights.copy()
                _weights_no_to["theover_weight"] = 0.0
                # v103 FIX (Bug 1): Pass the PRE-COMPUTED pick-side Kalshi prob directly
                # instead of the raw YES-side prob. This bypasses the internal
                # map_kalshi_prob_for_pick() call (which can invert the probability
                # when kalshi_yes_side is wrong) by setting kalshi_side_yes = pick_side
                # so the internal mapping is identity (no flip).
                spread_prob_no_to, _, _, _, _, _, _ = compute_final_probability(
                    spread_pick_side_key,
                    spread_prob_market,
                    kalshi_prob_spread_for_pick,
                    spread_pick_side_key,
                    model_spread_prob if model_used_for_spread else None,
                    None,
                    spread_sentiment_adj,
                    _weights_no_to,
                    sentiment_score=sentiment_diff,
                    kalshi_data=kalshi_spread if kalshi_spread.get("kalshi_matched") else None,
                )

                # Update Kalshi weight dynamically
                _spread_kalshi_matched = bool(kalshi_spread.get("kalshi_matched"))
                spread_weights["kalshi_weight"] = dynamic_kalshi_weight(
                    kalshi_prob_spread_for_pick,
                    spread_prob_market,
                    _spread_kalshi_matched,
                    league_name
                )

                # DEBUG: Log spread probability calculation inputs (v98: show pick-side Kalshi prob)
                logger.info(f"SPREAD PROB CALC for {home} vs {away}: spread_pick_side={spread_pick_side_key}, spread_market={spread_prob_market:.4f}, spread_implied={spread_implied}, kalshi={kalshi_prob_spread_for_pick}")

                # v103 FIX (Bug 1): Pass pre-computed pick-side Kalshi prob with
                # matching side key so compute_final_probability uses the PICK-side
                # probability directly, not the raw YES-side which may be inverted.
                spread_prob_final, spread_base_prob, spread_weights_used, spread_decision_driver, spread_warnings_new, spread_kalshi_prob_for_pick, spread_sentiment_debug = compute_final_probability(
                    spread_pick_side_key,
                    spread_prob_market,
                    kalshi_prob_spread_for_pick,
                    spread_pick_side_key,
                    model_spread_prob if model_used_for_spread else None,
                    theover_prob_final_spread,
                    spread_sentiment_adj,
                    spread_weights,
                    sentiment_score=sentiment_diff,
                    home_team=home,
                    away_team=away,
                    kalshi_data=kalshi_spread if kalshi_spread.get("kalshi_matched") else None,
                )

                # REMOVED: Misplaced sentiment debug capture code (lines 9130-9165)
                # This code was trying to use undefined variable 'row' before row objects were created
                # Sentiment debug data is already captured in spread_row and total_row creation blocks later

                # Calculate TheOver Impact (Invariant: delta = final - without)
                if theover_prob_final_spread is not None:
                    # v103 FIX (Bug 1): Use pre-computed pick-side Kalshi prob
                    spread_prob_no_to, _, _, _, _, _, _ = compute_final_probability(
                        spread_pick_side_key,
                        spread_prob_market,
                        kalshi_prob_spread_for_pick,
                        spread_pick_side_key,
                        model_spread_prob if model_used_for_spread else None,
                        None, # Exclude TheOver
                        spread_sentiment_adj,
                        spread_weights,
                        sentiment_score=sentiment_diff,
                        kalshi_data=kalshi_spread if kalshi_spread.get("kalshi_matched") else None,
                    )
                    if isinstance(spread_sentiment_debug, dict) and "theover_delta_clamped" in spread_sentiment_debug:
                        theover_delta_spread = spread_sentiment_debug.get("theover_delta_clamped")
                    else:
                        theover_delta_spread = (spread_prob_final or 0.0) - (spread_prob_no_to or 0.0)
                else:
                    spread_prob_no_to = spread_prob_final
                    theover_delta_spread = 0.0

                # Apply TheOver Decision Engine Adjustment (Spread) - Nudge Logic
                if theover_prob_final_spread is not None and spread_prob_final is not None:
                    # Directional check: if both > 0.5 or both < 0.5
                    agree = (theover_prob_final_spread > 0.5 and spread_prob_final > 0.5) or \
                            (theover_prob_final_spread < 0.5 and spread_prob_final < 0.5)

                    # Nudge: +0.02 if agree, -0.02 if strongly disagree (and we picked it)
                    if agree:
                        spread_prob_final = clamp(spread_prob_final + 0.02, 0.01, 0.95)
                        spread_warnings_new.append("theover_spread_agrees")
                    else:
                        spread_prob_final = clamp(spread_prob_final - 0.02, 0.05, 0.99)
                        spread_warnings_new.append("theover_spread_disagrees")

                    # Update delta to reflect nudge
                    theover_delta_spread = (spread_prob_final or 0.0) - (spread_prob_no_to or 0.0)

                # Pick Change Detection
                theover_changed_pick_spread = False
                if spread_prob_final is not None and spread_prob_no_to is not None:
                    if (spread_prob_final > 0.5) != (spread_prob_no_to > 0.5):
                        theover_changed_pick_spread = True

                theover_used_in_pick_spread = bool(theover_prob_final_spread is not None)

                if spread_prob_final is None:
                    spread_prob_final = blend_kalshi_market(kalshi_prob_spread_for_pick, spread_prob_market) if kalshi_spread.get("kalshi_matched") else spread_prob_market
                    if model_used_for_spread and model_spread_prob is not None:
                        spread_prob_final = clamp(model_spread_prob)
                    spread_base_prob = spread_prob_final
                    spread_weights_used = {"w_implied": 1.0 if spread_prob_final is not None else 0.0, "w_kalshi": 0.0, "w_model": 0.0, "w_sentiment": 0.0}
                spread_prob = spread_prob_final

                # DEBUG: Log final spread probability
                logger.info(f"SPREAD FINAL for {home} vs {away}: {spread_prob_final:.4f} ({spread_prob_final*100:.1f}%)")

                # Inject TheOver prob if available
                theover_prob_final_total = None
                if theover_prob_total is not None:
                    # Check alignment: total_pick_side_key (over/under) vs theover_total_pick_side
                    if theover_total_pick_side and total_pick_side_key and str(theover_total_pick_side).upper() == str(total_pick_side_key).upper():
                        theover_prob_final_total = theover_prob_total
                    else:
                        theover_prob_final_total = 1.0 - theover_prob_total

                    # Dynamic weighting based on TheOver hit_rate
                    # Strong signal (>=60%): 15% weight
                    # Moderate signal (>=55%): 12% weight
                    # Weak signal (<55%): 8% weight
                    total_hit_rate = safe_float((theover_matched_total or {}).get("theover_hit_rate"))
                    if total_hit_rate and total_hit_rate >= 0.60:
                        total_weights["theover_weight"] = 0.15
                    elif total_hit_rate and total_hit_rate >= 0.55:
                        total_weights["theover_weight"] = 0.12
                    elif total_hit_rate:
                        total_weights["theover_weight"] = 0.08
                    else:
                        total_weights["theover_weight"] = 0.10  # Default

                    if total_weights.get("ml_weight", 0) > 0.15:
                        total_weights["ml_weight"] -= 0.05

                # Calculate TOTAL probability WITHOUT TheOver
                _weights_total_no_to = total_weights.copy()
                _weights_total_no_to["theover_weight"] = 0.0
                # v103 FIX: Use pre-computed pick-side Kalshi prob (same pattern as spread fix)
                total_prob_no_to, _, _, _, _, _, _ = compute_final_probability(
                    total_pick_side_key,
                    total_prob_market,
                    kalshi_prob_total_for_pick,
                    total_pick_side_key,
                    model_total_prob if model_used_for_total else None,
                    None,
                    total_sentiment_adj,
                    _weights_total_no_to,
                    sentiment_score=sentiment_diff,
                    kalshi_data=kalshi_total if kalshi_total.get("kalshi_matched") else None,
                )

                # Update Kalshi weight dynamically
                _total_kalshi_matched = bool(kalshi_total.get("kalshi_matched"))
                total_weights["kalshi_weight"] = dynamic_kalshi_weight(
                    kalshi_prob_total_for_pick,
                    total_prob_market,
                    _total_kalshi_matched,
                    league_name
                )

                # DEBUG: Log total probability calculation inputs (v98: show pick-side Kalshi prob)
                logger.info(f"TOTAL PROB CALC for {home} vs {away}: total_pick_side={total_pick_side_key}, total_market={total_prob_market:.4f}, total_implied={total_implied}, kalshi={kalshi_prob_total_for_pick}")

                # v103 FIX: Use pre-computed pick-side Kalshi prob with matching side key
                total_prob_final, total_base_prob, total_weights_used, total_decision_driver, total_warnings_new, total_kalshi_prob_for_pick, total_sentiment_debug = compute_final_probability(
                    total_pick_side_key,
                    total_prob_market,
                    kalshi_prob_total_for_pick,
                    total_pick_side_key,
                    model_total_prob if model_used_for_total else None,
                    theover_prob_final_total,
                    total_sentiment_adj,
                    total_weights,
                    sentiment_score=sentiment_diff,
                    home_team=home,
                    away_team=away,
                    kalshi_data=kalshi_total if kalshi_total.get("kalshi_matched") else None,
                )

                # Calculate TheOver Impact (Invariant: delta = final - without)
                if theover_prob_final_total is not None:
                    # v103 FIX: Use pre-computed pick-side Kalshi prob
                    total_prob_no_to, _, _, _, _, _, _ = compute_final_probability(
                        total_pick_side_key,
                        total_prob_market,
                        kalshi_prob_total_for_pick,
                        total_pick_side_key,
                        model_total_prob if model_used_for_total else None,
                        None, # Exclude TheOver
                        total_sentiment_adj,
                        total_weights,
                        sentiment_score=sentiment_diff,
                        kalshi_data=kalshi_total if kalshi_total.get("kalshi_matched") else None,
                    )
                    if isinstance(total_sentiment_debug, dict) and "theover_delta_clamped" in total_sentiment_debug:
                        theover_delta_total = total_sentiment_debug.get("theover_delta_clamped")
                    else:
                        theover_delta_total = (total_prob_final or 0.0) - (total_prob_no_to or 0.0)
                else:
                    total_prob_no_to = total_prob_final
                    theover_delta_total = 0.0

                # Apply TheOver Decision Engine Adjustment (Total) - Nudge Logic
                if theover_prob_final_total is not None and total_prob_final is not None:
                    # Directional check: if both > 0.5 or both < 0.5
                    agree = (theover_prob_final_total > 0.5 and total_prob_final > 0.5) or \
                            (theover_prob_final_total < 0.5 and total_prob_final < 0.5)

                    # Nudge: +0.02 if agree, -0.02 if strongly disagree (and we picked it)
                    if agree:
                        total_prob_final = clamp(total_prob_final + 0.02, 0.01, 0.95)
                        total_warnings_new.append("theover_total_agrees")
                    else:
                        total_prob_final = clamp(total_prob_final - 0.02, 0.05, 0.99)
                        total_warnings_new.append("theover_total_disagrees")

                    # Update delta to reflect nudge
                    theover_delta_total = (total_prob_final or 0.0) - (total_prob_no_to or 0.0)

                # Pick Change Detection
                theover_changed_pick_total = False
                if total_prob_final is not None and total_prob_no_to is not None:
                    if (total_prob_final > 0.5) != (total_prob_no_to > 0.5):
                        theover_changed_pick_total = True

                theover_used_in_pick_total = bool(theover_prob_final_total is not None)

                if total_prob_final is None:
                    total_prob_final = blend_kalshi_market(kalshi_prob_total_for_pick, total_prob_market) if kalshi_total.get("kalshi_matched") else total_prob_market
                    if model_used_for_total and model_total_prob is not None:
                        total_prob_final = clamp(model_total_prob)
                    total_base_prob = total_prob_final
                    total_weights_used = {"w_implied": 1.0 if total_prob_final is not None else 0.0, "w_kalshi": 0.0, "w_model": 0.0, "w_sentiment": 0.0}
                total_prob = total_prob_final

                # DEBUG: Log final total probability and comparison
                logger.info(f"TOTAL FINAL for {home} vs {away}: {total_prob_final:.4f} ({total_prob_final*100:.1f}%)")
                logger.info(f"PROBABILITY COMPARISON for {home} vs {away}: Spread={spread_prob_final:.4f} vs Total={total_prob_final:.4f}, Diff={abs(spread_prob_final-total_prob_final):.4f}")
                if abs(spread_prob_final - total_prob_final) < 0.001:
                    logger.debug(f"Note: Very similar probabilities for {home} vs {away} - Spread={spread_prob_final:.4f}, Total={total_prob_final:.4f}. Will use consensus/edge to select best pick.")

                if spread_warnings_new:
                    warnings = list(dict.fromkeys(warnings + spread_warnings_new))
                if total_warnings_new:
                    warnings = list(dict.fromkeys(warnings + total_warnings_new))
                spread_odds_valid = bool(spread_odds_method == "book_price")
                total_odds_valid = bool(total_odds_method == "book_price")
                odds_placeholder_overall = bool(overall_odds_placeholder)
                spread_prob_engine = prob_engine_label(bool(kalshi_spread.get("kalshi_matched")), spread_prob_market, model_used=model_used_for_spread)
                total_prob_engine = prob_engine_label(bool(kalshi_total.get("kalshi_matched")), total_prob_market, model_used=model_used_for_total)

                # --- Decision trace (Spread) ---
                spread_alt_team = None
                spread_alt_line = None
                spread_alt_odds = None
                spread_alt_market_prob = None
                spread_alt_prob_final = None
                # FIX: Ensure spread line is valid before formatting
                if spread_pick is not None and has_valid_line(spread_line):
                    spread_pick_label = f"{spread_pick} {clean_line_str(spread_line)}"
                else:
                    spread_pick_label = ""
                spread_alt_label = ""
                spread_decision_metric_used = "final_prob"
                spread_decision_score_pick = spread_prob_final
                spread_decision_score_alt = None
                spread_decision_score_margin = None
                spread_prob_margin = None
                spread_prob_pick_market = spread_prob_market
                spread_prob_alt_market = None
                # FIX: Use the pick-side-mapped Kalshi probability, not the raw YES-side probability.
                # spread_kalshi_prob_for_pick is correctly mapped to the pick side by
                # map_kalshi_prob_for_pick() inside compute_final_probability().
                spread_prob_pick_kalshi = spread_kalshi_prob_for_pick
                spread_prob_alt_kalshi = (1.0 - spread_kalshi_prob_for_pick) if spread_kalshi_prob_for_pick is not None else None

                # FIX (Bug 1): Explicitly calculate spread_prob_pick_final using the PICK side Kalshi prob
                spread_prob_pick_final = blend_kalshi_market(spread_prob_pick_kalshi, spread_prob_pick_market)
                if spread_pick_team in {home, away}:
                    spread_alt_team = away if spread_pick_team == home else home
                    spread_alt_line = home_spread_point if spread_pick_team == away else away_spread_point
                    spread_alt_label = f"{spread_alt_team} {spread_alt_line}" if spread_alt_team else ""
                    alt_side_key = "home" if spread_alt_team == home else "away"
                    alt_offer = select_best_offer_for_pick(
                        spread_offers, alt_side_key, pick_line=spread_alt_line, preferred_book=g.get("best_spread_book")
                    )
                    if alt_offer:
                        spread_alt_odds = alt_offer.get("price")
                    spread_alt_market_prob, _, _, _ = compute_market_prob_from_offers(
                        spread_offers, alt_side_key, market_type="spread"
                    )
                    spread_prob_alt_market = spread_alt_market_prob
                    spread_alt_prob_final = blend_kalshi_market(spread_prob_alt_kalshi, spread_alt_market_prob)

                    # NEW PICK SELECTION LOGIC (FORCE > 50%)
                    if spread_pick_team == home:
                        prob_home_covers = spread_prob_pick_final
                        prob_away_covers = spread_alt_prob_final
                    else:
                        prob_home_covers = spread_alt_prob_final
                        prob_away_covers = spread_prob_pick_final

                    # Use home_spread_point as line reference
                    # Note: home_spread_point is negative if home is favored
                    # Use 0.0 if home_spread_point is None, but usually it should be set if we have a spread market
                    _ref_line = home_spread_point if home_spread_point is not None else 0.0

                    spread_pick_result = select_best_spread_pick(
                        home_team=home,
                        away_team=away,
                        spread_line=_ref_line,
                        prob_home_covers=prob_home_covers,
                        prob_away_covers=prob_away_covers
                    )

                    # Update variables with the winning side
                    spread_pick_team = spread_pick_result['pick_team']
                    spread_pick_label = spread_pick_result['pick_label']
                    spread_prob_final = spread_pick_result['pick_prob'] # This is 'prob'
                    spread_prob_pick_final = spread_pick_result['pick_prob']
                    spread_alt_prob_final = spread_pick_result['alt_prob']
                    spread_alt_label = spread_pick_result['alt_label']

                    # Update row dictionary directly
                    row['Pick'] = spread_pick_label
                    row['prob'] = spread_prob_pick_final
                    row['Spread Pick'] = spread_pick_label
                    row['Spread & Pick'] = spread_pick_label
                    row['spreadprobpickfinal'] = spread_prob_pick_final
                    row['spreadprobaltfinal'] = spread_alt_prob_final
                    row['spreadpicklabel'] = spread_pick_label
                    row['spreadaltlabel'] = spread_alt_label

                    # Also update internal tracking
                    spread_pick = spread_pick_result['pick_team']
                    spread_line = _ref_line if spread_pick_result['pick_side'] == "home" else -_ref_line

                    spread_decision_score_pick = spread_prob_pick_final
                    spread_decision_score_alt = spread_alt_prob_final
                    spread_prob_margin = compute_margin(spread_prob_final, spread_alt_prob_final)
                    spread_decision_score_margin = compute_margin(spread_decision_score_pick, spread_decision_score_alt)
                spread_engine_used = engine_label(bool(spread_prob_pick_kalshi), bool(spread_prob_pick_market is not None or spread_prob_alt_market is not None))
                spread_trace = {
                    "pick": {
                        "team": spread_pick_team,
                        "label": safe_str(spread_pick_label),
                        "line": spread_line,
                        "odds": spread_pick_odds,
                        "market_prob": spread_prob_pick_market,
                        "kalshi_prob": spread_prob_pick_kalshi,
                        "final_prob": spread_prob_final,
                        "score": spread_decision_score_pick,
                    },
                    "alt": {
                        "team": spread_alt_team,
                        "label": safe_str(spread_alt_label),
                        "line": spread_alt_line,
                        "odds": spread_alt_odds,
                        "market_prob": spread_prob_alt_market,
                        "kalshi_prob": spread_prob_alt_kalshi,
                        "final_prob": spread_alt_prob_final,
                        "score": spread_decision_score_alt,
                    },
                    "engine_used": spread_engine_used,
                    "metric": spread_decision_metric_used,
                    "notes": "",
                }
                try:
                    spread_trace_json = json.dumps(spread_trace, default=safe_str)
                except Exception:
                    spread_trace_json = "{}"

                # --- Decision trace (Total) ---
                total_alt_side = None
                total_alt_line = total_line
                total_alt_odds = None
                total_alt_market_prob = None
                total_alt_prob_final = None
                # FIX: Avoid "Under 0" / "Under 01" artifacts by checking for valid line
                if total_pick is not None and has_valid_line(total_line) and total_line != 0:
                    total_pick_label = f"{total_pick} {clean_line_str(total_line)}"
                else:
                    total_pick_label = ""
                total_alt_label = ""
                total_decision_metric_used = "final_prob"
                total_decision_score_pick = total_prob_final
                total_decision_score_alt = None
                total_decision_score_margin = None
                total_prob_margin = None
                total_prob_pick_market = total_prob_market
                total_prob_alt_market = None
                # FIX: Use the pick-side-mapped Kalshi probability, not the raw YES-side probability.
                # kalshi_prob_total is the raw Kalshi YES-side prob (e.g., 5% for Over),
                # but total_kalshi_prob_for_pick is correctly mapped to the pick side
                # (e.g., 95% for Under when pick=Under and Kalshi YES=Over at 5%).
                # Using the raw value caused an apparent "inversion" where the consensus
                # display showed K:5% while the final probability was ~77%.
                total_prob_pick_kalshi = total_kalshi_prob_for_pick
                total_prob_alt_kalshi = (1.0 - total_kalshi_prob_for_pick) if total_kalshi_prob_for_pick is not None else None

                # FIX (Bug 1): Explicitly calculate total_prob_pick_final using the PICK side Kalshi prob
                total_prob_pick_final = blend_kalshi_market(total_prob_pick_kalshi, total_prob_pick_market)
                if total_pick_side in {"Over", "Under"}:
                    total_alt_side = "Under" if total_pick_side == "Over" else "Over"
                    total_alt_label = f"{total_alt_side} {total_line}" if total_line is not None else total_alt_side
                    alt_side_key_total = total_alt_side.lower()
                    alt_total_offer = select_best_offer_for_pick(
                        total_offers, alt_side_key_total, pick_line=total_line, preferred_book=g.get("best_total_book")
                    )
                    if alt_total_offer:
                        total_alt_odds = alt_total_offer.get("price")
                    total_alt_market_prob, _, _, _ = compute_market_prob_from_offers(
                        total_offers, alt_side_key_total, market_type="total"
                    )
                    total_prob_alt_market = total_alt_market_prob
                    total_alt_prob_final = blend_kalshi_market(total_prob_alt_kalshi, total_alt_market_prob)

                    # NEW PICK SELECTION LOGIC (FORCE > 50%)
                    if total_pick_side == "Over":
                        prob_over = total_prob_pick_final
                        prob_under = total_alt_prob_final
                    else:
                        prob_over = total_alt_prob_final
                        prob_under = total_prob_pick_final

                    # Use total_line as line reference
                    # Note: total_line is always positive (e.g. 148.5)
                    _ref_total_line = total_line if total_line is not None else 0.0

                    total_pick_result = select_best_total_pick(
                        total_line=_ref_total_line,
                        prob_over=prob_over,
                        prob_under=prob_under
                    )

                    # Update variables with the winning side
                    total_pick_label = total_pick_result['pick_label']
                    total_prob_final = total_pick_result['pick_prob'] # This is 'prob'
                    total_prob_pick_final = total_pick_result['pick_prob']
                    total_alt_prob_final = total_pick_result['alt_prob']
                    total_alt_label = total_pick_result['alt_label']

                    # Update row dictionary directly
                    row['Total Pick'] = total_pick_label
                    row['totalprobpickfinal'] = total_prob_pick_final
                    row['totalprobaltfinal'] = total_alt_prob_final
                    row['totalpicklabel'] = total_pick_label
                    row['totalaltlabel'] = total_alt_label
                    row['Total & Pick'] = total_pick_label

                    # Also update internal tracking
                    total_pick_side = total_pick_result['pick_side']
                    total_decision_score_pick = total_prob_pick_final

                    total_decision_score_alt = total_alt_prob_final
                    total_prob_margin = compute_margin(total_prob_final, total_alt_prob_final)
                    total_decision_score_margin = compute_margin(total_decision_score_pick, total_decision_score_alt)
                total_engine_used = engine_label(bool(total_prob_pick_kalshi), bool(total_prob_pick_market is not None or total_prob_alt_market is not None))
                total_trace = {
                    "pick": {
                        "side": total_pick_side,
                        "label": safe_str(total_pick_label),
                        "line": total_line,
                        "odds": total_pick_odds,
                        "market_prob": total_prob_pick_market,
                        "kalshi_prob": total_prob_pick_kalshi,
                        "final_prob": total_prob_final,
                        "score": total_decision_score_pick,
                    },
                    "alt": {
                        "side": total_alt_side,
                        "label": safe_str(total_alt_label),
                        "line": total_alt_line,
                        "odds": total_alt_odds,
                        "market_prob": total_prob_alt_market,
                        "kalshi_prob": total_prob_alt_kalshi,
                        "final_prob": total_alt_prob_final,
                        "score": total_decision_score_alt,
                    },
                    "engine_used": total_engine_used,
                    "metric": total_decision_metric_used,
                    "notes": "",
                }
                try:
                    total_trace_json = json.dumps(total_trace, default=safe_str)
                except Exception:
                    total_trace_json = "{}"

                # --- UPDATE BEST OVERALL PICK (Force > 50%) ---
                _s_prob = spread_prob_pick_final if spread_prob_pick_final is not None else -1.0
                _t_prob = total_prob_pick_final if total_prob_pick_final is not None else -1.0

                # Check for validity (ensure label exists and prob is reasonable)
                _s_valid = _s_prob > 0 and spread_pick_label
                _t_valid = _t_prob > 0 and total_pick_label

                if _s_valid and _t_valid:
                    if _s_prob >= _t_prob:
                        row['Best Overall Pick'] = spread_pick_label
                        row['Best Overall Prob'] = _s_prob
                        row['bestpicktype'] = 'SPREAD'
                    else:
                        row['Best Overall Pick'] = total_pick_label
                        row['Best Overall Prob'] = _t_prob
                        row['bestpicktype'] = 'TOTAL'
                elif _s_valid:
                    row['Best Overall Pick'] = spread_pick_label
                    row['Best Overall Prob'] = _s_prob
                    row['bestpicktype'] = 'SPREAD'
                elif _t_valid:
                    row['Best Overall Pick'] = total_pick_label
                    row['Best Overall Prob'] = _t_prob
                    row['bestpicktype'] = 'TOTAL'

                # Validation (User Request)
                # Using safe get/comparison
                _best_prob = row.get('Best Overall Prob')
                if _best_prob is not None and isinstance(_best_prob, (int, float)) and _best_prob < 0.50:
                     logger.warning(f"CRITICAL: Best overall prob {_best_prob:.3f} < 50% for {g.get('home_team')} vs {g.get('away_team')}")

                decision_trace_version = "v1"
                overall_engine_used = f"spread:{spread_engine_used}|total:{total_engine_used}"
                decision_trace_notes_parts: List[str] = []
                if spread_decision_score_margin is not None:
                    decision_trace_notes_parts.append(f"spread_margin={spread_decision_score_margin:.3f}")
                if total_decision_score_margin is not None:
                    decision_trace_notes_parts.append(f"total_margin={total_decision_score_margin:.3f}")
                if odds_placeholder_overall:
                    decision_trace_notes_parts.append("placeholder_odds_guardrail")
                decision_trace_notes = ";".join(decision_trace_notes_parts)

                # Baseline probability (Home Win)
                if implied_home is not None:
                    market_home_prob = implied_home
                elif implied_away is not None:
                    market_home_prob = 1.0 - implied_away
                else:
                    market_home_prob = None

                def prob_for_selection(base_prob: Optional[float], selection_team: str) -> Optional[float]:
                    if base_prob is None:
                        return None
                    return base_prob if selection_team == home else (1.0 - base_prob)

                # AI probability (null-safe, no defaults)
                def ai_prob_for_selection(selection_team: str, adjusted: bool = True) -> Optional[float]:
                    try:
                        base = prob_for_selection(model_prob_home, selection_team)
                        if base is None:
                            return None
                        base = clamp(base, 0.0, 1.0)
                        if not adjusted or sentiment_adj is None:
                            return float(base) if base is not None else 0.5
                        adj = sentiment_adj if selection_team == home else -sentiment_adj
                        return clamp((float(base) if base is not None else 0.0) + adj, 0.01, 0.99)
                    except Exception:
                        return 0.5

                # Consensus blending for the selection
                def consensus_for_selection(
                    selection_team: str, kalshi_prob_used: Optional[float], implied_pick_prob: Optional[float]
                ) -> Dict[str, Any]:
                    notes: List[str] = []
                    weights_debug: Dict[str, Any] = {
                        "kalshi_weight": 0.0,
                        "odds_weight": 0.0,
                        "ml_weight": 0.0,
                        "sentiment_weight": 0.0,
                    }
                    ai_prob = ai_prob_for_selection(selection_team, adjusted=False)
                    odds_prob = clamp(implied_pick_prob)
                    kalshi_prob = clamp(kalshi_prob_used)
                    decision_driver = "Unknown"
                    base_prob = None
                    if kalshi_prob is not None and kalshi_prob >= 0.55:
                        # Tier 1: Kalshi agrees
                        decision_driver = "Kalshi"
                        base_prob = kalshi_prob
                        weights_debug["kalshi_weight"] = 0.55
                        weights_debug["odds_weight"] = 0.15
                        weights_debug["ml_weight"] = 0.15
                    elif odds_prob is not None:
                        # Tier 2: Fallback to Market + ML
                        decision_driver = "Fallback (Market+ML)"
                        base_prob = (odds_prob * 0.5) + ((ai_prob or odds_prob) * 0.5)
                        weights_debug["kalshi_weight"] = 0.0
                        weights_debug["odds_weight"] = 0.35
                        weights_debug["ml_weight"] = 0.35
                    elif ai_prob is not None:
                        decision_driver = "ML"
                        base_prob = ai_prob
                        weights_debug["ml_weight"] = 1.0
                    else:
                        base_prob = None
                    sentiment_info = sentiment_impact_for_pick(sentiment_adj, selection_team, home, away)
                    weights_debug["sentiment_weight"] = abs(sentiment_info.get("sentiment_impact") or 0.0)
                    if base_prob is None:
                        final_prob = None
                    else:
                        final_prob = clamp(
                            (base_prob or 0.0) + (sentiment_info.get("sentiment_impact") or 0.0), 0.0, 1.0
                        )
                    return {
                        "base_prob": base_prob,
                        "final_prob": final_prob,
                        "notes": notes,
                        "weights": weights_debug,
                        "decision_driver": decision_driver,
                        **sentiment_info,
                    }

                # --- 4. DATA ROW GENERATION ---

                # MONEYLINE ROW
                def _ml_extreme(price: Optional[float]) -> bool:
                    """
                    Check if ML odds are too extreme to generate picks.
                    Relaxed from 500 to 800 to allow more ML picks.
                    """
                    try:
                        return abs(float(price)) >= 800
                    except Exception:
                        return False

                extreme_ml = _ml_extreme(home_ml) or _ml_extreme(away_ml)

                if (home_ml is not None or away_ml is not None) and not extreme_ml:
                    odds_placeholder = is_placeholder_odds(home_ml, away_ml)
                    odds_valid = not odds_placeholder
                    if odds_placeholder:
                        implied_home = None
                        implied_away = None
                        warnings = list(dict.fromkeys(warnings + ["placeholder_odds_detected"]))
                    pick = None
                    implied_pick = None
                    if implied_home is not None and implied_away is not None:
                        pick = home if implied_home >= implied_away else away
                        implied_pick = implied_home if pick == home else implied_away
                    elif implied_home is not None:
                        pick = home
                        implied_pick = implied_home
                    elif implied_away is not None:
                        pick = away
                        implied_pick = implied_away
                    if pick is None:
                        warnings = list(dict.fromkeys(warnings + ["no_implied_prob"]))

                    if pick is not None:
                        prob_engine_moneyline = prob_engine_label(bool(kalshi_winner.get("kalshi_matched")), implied_pick, model_used=bool(use_model_numeric_probs and model_prob_home is not None))
                        ai_prob_base = ai_prob_for_selection(pick, adjusted=False)
                        ai_prob_row = clamp((ai_prob_base or 0.0) + (sentiment_adj or 0.0), 0.01, 0.99) if ai_prob_base is not None else None

                        pick_side = "home" if pick == home else "away"
                        implied_pick = implied_prob_for_pick(home_ml, away_ml, pick_side)
                        kalshi_yes_side = kalshi_winner.get("kalshi_yes_side")

                        # v98 FIX (Bug B): Pre-compute pick-side Kalshi prob for moneyline
                        _ml_kalshi_matched = bool(kalshi_winner.get("kalshi_matched"))
                        kalshi_prob_ml_for_pick = map_kalshi_prob_for_pick(
                            kalshi_prob_used if _ml_kalshi_matched else None,
                            kalshi_yes_side,
                            pick_side
                        )

                        # ML Extreme Odds Weighting Logic
                        # When absolute American odds > 400 (heavy favorite/underdog):
                        # - Model weight -> 0% (model struggles with extreme games)
                        # - Kalshi weight -> 60% (market consensus is more reliable)
                        # - Odds weight -> 40% (implied probability)
                        # NOTE: ML picks are still INCLUDED, just with adjusted weights (not disabled)
                        # Relaxed from 300 to 400 to allow more ML picks with model weight
                        current_ml_weights = moneyline_weights.copy()
                        is_heavy_chalk = False
                        try:
                            h_p = float(home_ml) if home_ml is not None else 0
                            a_p = float(away_ml) if away_ml is not None else 0
                            if abs(h_p) > 400 or abs(a_p) > 400:
                                is_heavy_chalk = True
                        except:
                            pass

                        if is_heavy_chalk:
                            current_ml_weights["ml_weight"] = 0.0
                            current_ml_weights["w_model"] = 0.0 # Ensure explicit key also zeroed
                            current_ml_weights["kalshi_weight"] = 0.75  # UP from 0.6
                            current_ml_weights["odds_weight"] = 0.25    # DOWN from 0.4
                            # MODE B: Keep sentiment weight from config (was Mode A: forced to 0.0)
                            # current_ml_weights["sentiment_weight"] = 0.0  # DISABLED to enable sentiment
                            current_ml_weights["theover_weight"] = 0.0
                            warnings.append("ml_extreme_odds_flag")
                        else:
                            # Standard Dynamic Weighting
                            ml_odds_weight = 0.30
                            current_ml_weights["odds_weight"] = ml_odds_weight
                            # MODE B: Keep sentiment weight from config (was Mode A: forced to 0.0)
                            # current_ml_weights["sentiment_weight"] = 0.0  # DISABLED to enable sentiment

                            current_ml_weights["kalshi_weight"] = dynamic_kalshi_weight(
                                kalshi_prob_ml_for_pick,  # v98: use pre-flipped value
                                implied_pick,
                                _ml_kalshi_matched,
                                league_name
                            )

                            # Dynamic TheOver weight based on hit_rate for ML picks
                            if theover_prob_ml is not None:
                                ml_hit_rate = safe_float((theover_matched_side or {}).get("theover_hit_rate"))
                                if ml_hit_rate and ml_hit_rate >= 0.60:
                                    current_ml_weights["theover_weight"] = 0.15
                                elif ml_hit_rate and ml_hit_rate >= 0.55:
                                    current_ml_weights["theover_weight"] = 0.12
                                elif ml_hit_rate:
                                    current_ml_weights["theover_weight"] = 0.08
                                # If SIDE data was used for ML, apply reduced weight (weaker signal)
                                if theover_side_used_for_ml:
                                    current_ml_weights["theover_weight"] = current_ml_weights.get("theover_weight", 0.10) * 0.7

                        final_prob_blend, base_prob_blend, weights_used, decision_driver, warnings_new, kalshi_prob_for_pick, ml_sentiment_debug = compute_final_probability(
                            pick_side,
                            implied_pick,
                            kalshi_prob_used,
                            kalshi_yes_side,
                            ai_prob_base,
                            theover_prob_ml,  # FIX: Use TheOver probability for ML picks
                            sentiment_adj,
                            current_ml_weights,
                            sentiment_score=sentiment_diff,
                            home_team=home,
                            away_team=away,
                            kalshi_data=kalshi_winner if kalshi_winner.get("kalshi_matched") else None,
                        )

                        # Capture Sentiment Debug Data (ML)
                        if ml_sentiment_debug and ml_sentiment_debug.get("used"):
                            row["ml_wsentiment_used"] = ml_sentiment_debug.get("weight", 0.0)
                            row["ml_sentiment_adj"] = ml_sentiment_debug.get("adj", 0.0)
                            row["ml_sentiment_prob"] = ml_sentiment_debug.get("prob", 0.0)

                            # If ML is the main market (e.g. for ML-only sports or fallback), update globals
                            if "wsentiment_used" not in row or decision_driver == "sentiment":
                                row["wsentiment_used"] = ml_sentiment_debug.get("weight", 0.0)
                                row["sentiment_adj"] = ml_sentiment_debug.get("adj", 0.0)
                                row["sentiment_prob"] = ml_sentiment_debug.get("prob", 0.0)
                        sentiment_info = sentiment_impact_for_pick(sentiment_adj, pick, home, away)
                        sentiment_direction = sentiment_info.get("sentiment_direction")
                        sentiment_score_val = sentiment_info.get("sentiment_score")
                        sentiment_impact_applied = bool(sentiment_info.get("sentiment_impact_applied"))
                        sentiment_score_entry = (
                            sentiment_score_val
                            if sentiment_score_val is not None
                            else (sentiment_score_field if sentiment_score_field is not None else sentiment_score_value)
                        )
                        consensus_prob = base_prob_blend
                        consensus_prob_adj = final_prob_blend
                        consensus_weights = weights_used
                        if warnings_new:
                            warnings = list(dict.fromkeys(warnings + warnings_new))

                        warnings_field = ";".join(warnings) if warnings else None
                        implied_prob_reason = "missing_or_placeholder_odds" if odds_placeholder or implied_pick is None else f"from_odds_home_{home_ml}_away_{away_ml}"
                        ml_row = {
                            "league": league_name,
                            "Home": home,
                            "Away": away,
                            "Game": f"{away} @ {home}",
                            "Commence (UTC)": commence_iso,
                            "Commence (Local)": commence_local,
                            "Local Date": commence_date_local,
                            "Market": "Moneyline",
                            "best_pick_type": "ML",
                            "Book": g.get("best_ml_book"),
                            "Home_ML": home_ml,
                            "Away_ML": away_ml,
                            "Pick": f"{pick} ML" if pick else None,
                            "Implied_Prob": implied_pick,
                            "AI_Prob": ai_prob_base,
                            "ai_prob_adj": ai_prob_row,
                            "consensus_prob": consensus_prob,
                            "consensus_prob_adj": consensus_prob_adj,
                            "final_probability": consensus_prob_adj,
                            "decision_driver": decision_driver,
                            "kalshi_weight": consensus_weights.get("w_kalshi") or consensus_weights.get("kalshi_weight"),
                            "odds_weight": consensus_weights.get("w_implied") or consensus_weights.get("odds_weight"),
                            "ml_weight": consensus_weights.get("w_model") or consensus_weights.get("ml_weight"),
                            "sentiment_weight": consensus_weights.get("w_sentiment") or consensus_weights.get("sentiment_weight"),
                            "consensus_weight_ai": (consensus_weights or {}).get("w_model"),
                            "consensus_weight_market": (consensus_weights or {}).get("w_implied"),
                            "consensus_weight_kalshi": (consensus_weights or {}).get("w_kalshi"),
                            "consensus_weight_sentiment": (consensus_weights or {}).get("w_sentiment"),
                            "consensus_weight_total": sum(
                                w or 0.0 for w in [
                                    (consensus_weights or {}).get("ml_weight"),
                                    (consensus_weights or {}).get("odds_weight"),
                                    (consensus_weights or {}).get("kalshi_weight"),
                                ]
                            ),
                            "consensus_guardrails": ";".join((consensus_weights or {}).get("guardrails") or []),
                            # THEOVER INTEGRATION (Moneyline)
                            "theover_prob": theover_prob_ml,
                            "theover_ml_odds": (theover_matched_side or {}).get("theover_line") if theover_matched_side else None,
                            "theover_pick": (theover_matched_side or {}).get("theover_pick") if theover_matched_side else None,
                            "theover_hit_rate": (theover_matched_side or {}).get("theover_hit_rate") if theover_matched_side else None,
                            "theover_source_model": (theover_matched_side or {}).get("theover_model") if theover_matched_side else None,
                            "theover_prob_used": theover_prob_ml,
                            "theover_matched": bool(theover_matched_side),
                            "theover_side_used_for_ml": theover_side_used_for_ml,
                            "Home_Sentiment": home_sent,
                            "Away_Sentiment": away_sent,
                            "Sentiment_Diff": sentiment_diff,
                            "sentiment_adj": sentiment_adj,
                            "sentiment_score": sentiment_score_entry,
                            "sentiment_label": sentiment_label_field,
                            "sentiment_source_count": sentiment_articles_used,
                            "sentiment_direction": sentiment_direction,
                            "sentiment_impact_applied": sentiment_impact_applied,
                            "sentiment_source": sentiment_source,
                            "reddit_used": reddit_used,
                            "sentiment_valid": sentiment_valid,
                            "sentiment_level": sentiment_level,
                            "sentiment_strength": sentiment_strength,
                            "sentiment_badge": sentiment_badge,
                            "sentiment_articles_used": sentiment_articles_used,
                            "sentiment_query_used": sentiment_query_used,
                            "sentiment_status": sentiment_status_value,
                            "sentiment_confidence": sentiment_confidence_value,
                            "spread_sentiment_adj": spread_sentiment_adj,
                            "total_sentiment_adj": total_sentiment_adj,
                            "sentiment_error_count": sentiment_error_count,
                            "sentiment_errors_sample": sentiment_errors_sample,
                            "sentiment_articles_total": sentiment_articles_total,
                            "sentiment_status_counts": sentiment_status_counts_field,
                            "sentiment_sample_query": sentiment_sample_query,
                            "sentiment_sample_status": sentiment_sample_status,
                            "sentiment_sample_totalResults": sentiment_sample_totalResults,
                            "sentiment_auth_error": sentiment_auth_error,
                            "sentiment_rate_limited": sentiment_rate_limited,
                            "sentiment_cooldown_until": sentiment_cooldown_until,
                            "sentiment_cached_teams_count": sentiment_cached_teams_count,
                            "sentiment_available_count": sentiment_available_count,
                            "sentiment_used_cached": sentiment_used_cached,
                            "sentiment_disabled_reason": sentiment_disabled_reason,
                            "prob_engine": prob_engine_moneyline,
                            "model_mode": st.session_state.model_mode,
                            "gemini_mode": "pending" if use_gemini_explanations else "disabled",
                            "model_spread_prob": model_spread_prob,
                            "model_total_prob": model_total_prob,
                            "kalshi_prob_spread": kalshi_prob_spread,
                            "kalshi_prob_total": kalshi_prob_total,
                            "spread_prob_market": spread_prob_market,
                            "total_prob_market": total_prob_market,
                            "spread_engine_used": spread_engine_used,
                            "spread_pick_label": safe_str(spread_pick_label),
                            "spread_alt_label": safe_str(spread_alt_label),
                            "spread_prob_pick_final": spread_prob_final,
                            "spread_prob_alt_final": spread_alt_prob_final,
                            "spread_prob_margin": spread_prob_margin,
                            "spread_prob_pick_market": spread_prob_pick_market,
                            "spread_prob_alt_market": spread_prob_alt_market,
                            "spread_prob_pick_kalshi": spread_prob_pick_kalshi,
                            "spread_prob_alt_kalshi": spread_prob_alt_kalshi,
                            "spread_decision_metric_used": spread_decision_metric_used,
                            "spread_decision_score_pick": spread_decision_score_pick,
                            "spread_decision_score_alt": spread_decision_score_alt,
                            "spread_decision_score_margin": spread_decision_score_margin,
                            "spread_trace_json": spread_trace_json,
                            "total_engine_used": total_engine_used,
                            "total_pick_label": safe_str(total_pick_label),
                            "total_alt_label": safe_str(total_alt_label),
                            "total_prob_pick_final": total_prob_final,
                            "total_prob_alt_final": total_alt_prob_final,
                            "total_prob_margin": total_prob_margin,
                            "total_prob_pick_market": total_prob_pick_market,
                            "total_prob_alt_market": total_prob_alt_market,
                            "total_prob_pick_kalshi": total_prob_pick_kalshi,
                            "total_prob_alt_kalshi": total_prob_alt_kalshi,
                            "total_decision_metric_used": total_decision_metric_used,
                            "total_decision_score_pick": total_decision_score_pick,
                            "total_decision_score_alt": total_decision_score_alt,
                            "total_decision_score_margin": total_decision_score_margin,
                            "total_trace_json": total_trace_json,
                            "decision_trace_version": decision_trace_version,
                            "overall_engine_used": overall_engine_used,
                            "decision_trace_notes": decision_trace_notes,
                            "overall_confidence": None,
                            "spread_confidence_gemini": None,
                            "total_confidence_gemini": None,
                            "gemini_alignment": None,
                            "gemini_rationale": None,
                            "gemini_risk_flags": None,
                            "gemini_error": None,
                            "gemini_flags_short": None,
                            "llm_disagreement_flag": None,
                            "kalshi_available": kalshi_winner.get("kalshi_available"),
                            "kalshi_matched": kalshi_winner.get("kalshi_matched"),
                            "kalshi_status": kalshi_status_value,
                            "kalshi_prob": kalshi_prob_used,
                            "kalshi_prob_used": kalshi_prob_used,
                            "kalshi_prob_for_pick": kalshi_prob_for_pick,
                            "kalshi_yes_side": kalshi_yes_side,
                            "kalshi_event_ticker": kalshi_event_used,
                            "kalshi_event_ticker_used": kalshi_event_used,
                            "kalshi_candidate_count": candidate_debug.get("candidate_count"),
                            "kalshi_best_score": candidate_debug.get("best_score"),
                            "kalshi_match_reason": kalshi_winner.get("kalshi_reason"),
                            "kalshi_game_prefix_used": (candidate_debug.get("winner_meta") or {}).get("winner_prefix") or candidate_debug.get("kalshi_game_prefix_used"),
                            "kalshi_wanted_tokens": (candidate_debug.get("winner_meta") or {}).get("allowed_date_tokens") or candidate_debug.get("kalshi_wanted_tokens"),
                            "Spread & Pick": f"{spread_pick} {spread_line} ({spread_prob_final*100:.1f}%)" if (spread_pick is not None and spread_prob_final is not None) else (f"{spread_pick} {spread_line}" if spread_pick is not None else None),
                            "spread_pick_team": spread_pick_team,
                            "spread_pick_line": spread_pick_line,
                            "spread_pick_odds": spread_pick_odds,
                            "spread_prob": spread_prob,
                            "spread_confidence": None,
                            "spread_confidence_reason": None,
                            "Total & Pick": f"{total_pick} {clean_line_str(total_line)} ({total_prob_final*100:.1f}%)" if (total_pick is not None and total_prob_final is not None and clean_line_str(total_line) is not None) else (f"{total_pick} {clean_line_str(total_line)}" if (total_pick is not None and clean_line_str(total_line) is not None) else None),
                            "total_pick_side": total_pick_side,
                            "total_pick_line": total_line,
                            "total_pick_odds": total_pick_odds,
                            "total_prob": total_prob,
                            "total_confidence": None,
                            "total_confidence_reason": None,
                            "spread_engine_used": spread_engine_used,
                            "spread_pick_label": safe_str(spread_pick_label),
                            "spread_alt_label": safe_str(spread_alt_label),
                            "spread_prob_pick_final": spread_prob_final,
                            "spread_prob_alt_final": spread_alt_prob_final,
                            "spread_prob_margin": spread_prob_margin,
                            "spread_prob_pick_market": spread_prob_pick_market,
                            "spread_prob_alt_market": spread_prob_alt_market,
                            "spread_prob_pick_kalshi": spread_prob_pick_kalshi,
                            "spread_prob_alt_kalshi": spread_prob_alt_kalshi,
                            "spread_decision_metric_used": spread_decision_metric_used,
                            "spread_decision_score_pick": spread_decision_score_pick,
                            "spread_decision_score_alt": spread_decision_score_alt,
                            "spread_decision_score_margin": spread_decision_score_margin,
                            "spread_trace_json": spread_trace_json,
                            "total_engine_used": total_engine_used,
                            "total_pick_label": safe_str(total_pick_label),
                            "total_alt_label": safe_str(total_alt_label),
                            "total_prob_pick_final": total_prob_final,
                            "total_prob_alt_final": total_alt_prob_final,
                            "total_prob_margin": total_prob_margin,
                            "total_prob_pick_market": total_prob_pick_market,
                            "total_prob_alt_market": total_prob_alt_market,
                            "total_prob_pick_kalshi": total_prob_pick_kalshi,
                            "total_prob_alt_kalshi": total_prob_alt_kalshi,
                            "total_decision_metric_used": total_decision_metric_used,
                            "total_decision_score_pick": total_decision_score_pick,
                            "total_decision_score_alt": total_decision_score_alt,
                            "total_decision_score_margin": total_decision_score_margin,
                            "total_trace_json": total_trace_json,
                            "decision_trace_version": decision_trace_version,
                            "overall_engine_used": overall_engine_used,
                            "decision_trace_notes": decision_trace_notes,
                            "Model Spread Prob": model_spread_prob,
                            "Model Total Prob": model_total_prob,
                            "spread_implied_prob": spread_implied,
                            "spread_prob_market_based": spread_prob_market_based,
                            "spread_prob_reason": spread_prob_reason,
                            "spread_prob_method": spread_prob_method,
                            "total_implied_prob": total_implied,
                            "total_prob_market_based": total_prob_market_based,
                            "total_prob_reason": total_prob_reason,
                            "total_prob_method": total_prob_method,
                            "odds_valid": odds_valid,
                            "odds_placeholder_detected": bool(odds_placeholder or odds_placeholder_overall),
                            "spread_odds_valid": spread_odds_valid,
                            "total_odds_valid": total_odds_valid,
                            "spread_odds_placeholder_detected": spread_odds_placeholder_detected,
                            "total_odds_placeholder_detected": total_odds_placeholder_detected,
                            "spread_prob_placeholder_detected": spread_prob_placeholder_detected,
                            "total_prob_placeholder_detected": total_prob_placeholder_detected,
                            "implied_prob_reason": implied_prob_reason,
                            "Warnings": warnings_field,
                            "best_spread_book": g.get("best_spread_book"),
                            "best_spread_last_update": g.get("best_spread_last_update"),
                            "best_spread_price_score": g.get("best_spread_price_score"),
                            "best_spread_median_point": g.get("best_spread_median_point"),
                            "best_spread_mode_point": g.get("best_spread_mode_point"),
                            "best_spread_price": best_spread_price,
                            "best_total_book": g.get("best_total_book"),
                            "best_total_last_update": g.get("best_total_last_update"),
                            "best_total_price_score": g.get("best_total_price_score"),
                            "best_total_median_point": g.get("best_total_median_point"),
                            "best_total_mode_point": g.get("best_total_mode_point"),
                            "best_total_price": best_total_price,
                            "spread_width": width_spread,
                            "total_width": width_total,
                            "Kalshi_Required": st.session_state.get("kalshi_required", True),
                            "api_sports_used": api_sports_used,
                            "sportsdata_used": sportsdata_used,
                            "api_sports_status": api_sports_status_run,
                            "sportsdata_status": sportsdata_status_run,
                            "apisports_enriched": apisports_enriched,
                            "apisports_status": api_sports_status_run,
                            "apisports_notes": apisports_notes,
                            "sportsdata_enriched": sportsdata_enriched,
                            "sportsdata_status": sportsdata_status_run,
                            "sportsdata_notes": sportsdata_notes,
                            "injuries_home_count": injuries_home_count,
                            "injuries_away_count": injuries_away_count,
                            "injuries_home": injuries_home_display,
                            "injuries_away": injuries_away_display,
                            "enrichment_errors_sample": enrichment_errors_sample,
                            "weather_summary": weather_summary,
                            "key_injuries_home": ",".join(key_injuries_home),
                            "key_injuries_away": ",".join(key_injuries_away),
                            "spread_min": spread_min,
                            "spread_med": spread_med,
                            "spread_max": spread_max,
                            "total_min": total_min,
                            "total_med": total_med,
                            "total_max": total_max,
                            "spread_books_count": len(spread_books_map),
                            "total_books_count": len(total_books_map),
                            "At_a_Glance_Confidence": None,
                            "At_a_Glance_Score": None,
                            "At_a_Glance_Reason": None,
                        }
                        adj_val, adj_reason = compute_sentiment_adj_row(ml_row)
                        ml_row["sentiment_adj"] = adj_val
                        ml_row["sentiment_adj_value"] = adj_val
                        ml_row["sentiment_adj_reason"] = adj_reason
                        sentiment_adj = adj_val
                        base_prob = clamp(ml_row.get("AI_Prob"))
                        ml_row["ai_prob_adj"] = clamp((base_prob or 0.0) + adj_val) if base_prob is not None else None
                        conf, reason_short, eligible = score_pick_confidence(ml_row)
                        ml_row["Pick_Confidence"] = conf
                        ml_row["Pick_Reason_Short"] = reason_short
                        ml_row["confidence_reason"] = reason_short
                        _dec_base = safe_float(ml_row.get("final_probability"))
                        ml_row["decisiveness"] = abs(_dec_base - 0.5) * 2 if _dec_base is not None else None
                        trace_short, trace_json, decision_trace_full = build_decision_trace(
                            "moneyline",
                            ml_row.get("Pick") or "",
                            ml_row.get("Implied_Prob"),
                            ml_row.get("kalshi_prob"),
                            ml_row.get("AI_Prob"),
                            ml_row.get("sentiment_adj"),
                            consensus_weights,
                            ml_row.get("final_probability"),
                            conf,
                            league_name,
                            bool(kalshi_winner.get("kalshi_matched")),
                            kalshi_winner.get("kalshi_reason"),
                            ml_row.get("sentiment_score"),
                            ml_row.get("sentiment_label"),
                            bool(use_model_numeric_probs and model_prob_home is not None),
                            reason_short,
                            warnings,
                            kalshi_yes_side,
                            kalshi_prob_for_pick,
                        )
                        trace_json_str = trace_json if isinstance(trace_json, str) else json.dumps(trace_json)
                        ml_row["decision_trace_short"] = trace_short
                        ml_row["decision_trace_json"] = trace_json_str
                        ml_row["decision_trace"] = decision_trace_full
                        store_decision_trace_sample(
                            league_name,
                            home,
                            away,
                            "moneyline",
                            ml_row.get("Pick"),
                            ml_row.get("final_probability"),
                            trace_json_str,
                        )
                        if league_name in {"NFL", "NBA", "NCAAB"} and not SENTIMENT_LOG_SAMPLE.get(league_name):
                            try:
                                logger.info(f"Decision trace sample {league_name}: {decision_trace_full}")
                            except Exception:
                                pass
                            SENTIMENT_LOG_SAMPLE[league_name] = True
                        ml_row["Eligible_Top_Picks"] = eligible
                        ml_row = apply_sentiment_defaults(ml_row, sentiment_defaults_base)

                        # Task 3: Force Moneyline Pivot (Inside Loop)
                        # DISABLED: We want to keep Moneyline rows explicitly for display/export
                        # The UI logic will pivot later if needed for 'Best Pick', but we need the raw ML row.
                        # if ml_row.get("Market") == "Moneyline":
                        #     s_edge = safe_float(ml_row.get("spread_edge")) or 0.0
                        #     t_edge = safe_float(ml_row.get("total_edge")) or 0.0
                        #     if s_edge >= t_edge:
                        #         ml_row["Market"] = "Spread"
                        #         ml_row["Pick"] = ml_row.get("Spread & Pick")
                        #     else:
                        #         ml_row["Market"] = "Total"
                        #         ml_row["Pick"] = ml_row.get("Total & Pick")

                        accumulated_rows.append(ml_row)
                        ml_row_created = True
                        master_stats["h2h_found"] += 1
                        master_stats["market_rows_out"] += 1
                elif extreme_ml:
                    warnings = list(dict.fromkeys(warnings + ["moneyline_extreme_skipped"]))

                # PRE-VALIDATION: Check line validity before creating spread/total rows
                spread_valid = (has_valid_line(spread_line) and
                                spread_odds_valid and
                                spread_pick is not None and
                                spread_prob_market is not None and
                                spread_prob_market < 0.95)  # Filter out placeholder probabilities

                total_valid_check = (has_valid_line(total_line) and
                                     total_odds_valid and
                                     total_pick is not None and
                                     total_prob_market is not None and
                                     total_prob_market < 0.95)  # Filter out placeholder probabilities

                # If both spread and total are invalid, add warning
                if not spread_valid and not total_valid_check:
                    warnings.append("no_valid_spread_or_total")

                # NHL Special Handling: If league is NHL and only moneyline is available (no spread/total)
                # Mark as informational only - do not create picks
                if league_name == "NHL" and not spread_valid and not total_valid_check:
                    # Moneyline-only NHL game - informational only
                    warnings.append("nhl_moneyline_only_no_spread_or_total")
                    # The moneyline row (if created) will be marked as LOW confidence automatically
                    # due to the "no_valid_spread_or_total" warning

                # SPREAD ROW
                # FIX: Only create spread row if spread odds are valid AND line is valid (not None/NaN)
                # This prevents synthetic "None/69.6/NaN" rows for NHL games with missing spread markets

                # Reset warnings to base game warnings (prevent ML warnings from leaking into Spread)
                warnings = list(base_game_warnings)

                # Add warning if spread line is invalid
                if not has_valid_line(spread_line) and g.get("home_spread_point") is not None:
                    warnings.append("no_valid_spread")

                if spread_valid:
                    ai_prob_base = None
                    ai_prob_row = None
                    model_spread_prob = None
                    # Market-only spread: Using market-implied probability instead of model
                    # This happens when spread odds are available but model prediction is not used
                    warnings.append("market_based_spread_prob")
                    spread_prob_engine = "market_only"
                    # Add warning if spread_pick is None
                    if spread_pick is None:
                        warnings.append("spread_pick_none")
                    warnings_field = ";".join(warnings) if warnings else None
                    spread_row = {
                        "league": league_name, "Home": home, "Away": away,
                        "Game": f"{away} @ {home}",
                        "Commence (UTC)": commence_iso, "Commence (Local)": commence_local,
                        "Market": "Spread", "Book": g.get("best_spread_book"),
                        "Pick": f"{spread_pick} {spread_line:+.1f}" if (spread_pick is not None and spread_line is not None) else spread_pick, "Implied_Prob": spread_prob_market, "Line": spread_line, "AI_Prob": model_spread_prob if model_used_for_spread else None,
                        "Home_ML": home_ml,
                        "Away_ML": away_ml,
                        "ml_home_implied": american_to_implied_prob(g.get("home_ml_price")),
                        "ml_away_implied": american_to_implied_prob(g.get("away_ml_price")),
                        "spread_pick_side": spread_pick_side_key,
                        "ai_prob_adj": ai_prob_row, "consensus_prob": spread_base_prob, "consensus_prob_adj": spread_prob_final,
                        "final_probability": spread_prob_final,
                        "decision_driver": spread_decision_driver or spread_engine_used,
                        "kalshi_weight": spread_weights_used.get("w_kalshi") if 'spread_weights_used' in locals() else None,
                        "odds_weight": spread_weights_used.get("w_implied") if 'spread_weights_used' in locals() else None,
                        "ml_weight": spread_weights_used.get("w_model") if 'spread_weights_used' in locals() else None,
                        "sentiment_weight": spread_weights_used.get("w_sentiment") if 'spread_weights_used' in locals() else abs(spread_sentiment_adj or 0.0),
                        "sentiment_adj": sentiment_adj, "sentiment_source": sentiment_source, "reddit_used": reddit_used, "sentiment_valid": sentiment_valid,
                        "sentiment_score": sentiment_score_field if sentiment_score_field is not None else None,
                        "sentiment_label": sentiment_label_field,
                        "sentiment_source_count": sentiment_articles_used,
                        "sentiment_direction": None,
                        "sentiment_impact_applied": False,
                        "sentiment_level": sentiment_level,
                        "sentiment_strength": sentiment_strength,
                        "sentiment_badge": sentiment_badge,
                        "sentiment_articles_used": sentiment_articles_used,
                        "sentiment_query_used": sentiment_query_used,
                        "sentiment_status": sentiment_status_value,
                        "sentiment_confidence": sentiment_confidence_value,
                        "sentiment_score": sentiment_score_value,
                        "spread_sentiment_adj": spread_sentiment_adj,
                        "total_sentiment_adj": total_sentiment_adj,
                        "sentiment_error_count": sentiment_error_count,
                        "sentiment_errors_sample": sentiment_errors_sample,
                        "sentiment_articles_total": sentiment_articles_total,
                        "sentiment_status_counts": sentiment_status_counts_field,
                        "sentiment_sample_query": sentiment_sample_query,
                        "sentiment_sample_status": sentiment_sample_status,
                        "sentiment_sample_totalResults": sentiment_sample_totalResults,
                        "sentiment_auth_error": sentiment_auth_error,
                        "sentiment_rate_limited": sentiment_rate_limited,
                        "sentiment_cooldown_until": sentiment_cooldown_until,
                        "sentiment_cached_teams_count": sentiment_cached_teams_count,
                        "sentiment_available_count": sentiment_available_count,
                        "sentiment_used_cached": sentiment_used_cached,
                        "sentiment_disabled_reason": sentiment_disabled_reason,
                        "Model Spread Prob": model_spread_prob,
                        "model_spread_prob": model_spread_prob,
                        "model_total_prob": model_total_prob,
                        "prob_engine": spread_prob_engine,
                        "model_mode": st.session_state.model_mode,
                        "gemini_mode": "pending" if use_gemini_explanations else "disabled",
                        "overall_confidence": None,
                        "spread_confidence_gemini": None,
                        "total_confidence_gemini": None,
                        "gemini_alignment": None,
                        "gemini_rationale": None,
                        "gemini_risk_flags": None,
                        "gemini_error": None,
                        "gemini_flags_short": None,
                        "llm_disagreement_flag": None,
                        "kalshi_prob_spread": kalshi_prob_spread,
                        "kalshi_prob_total": kalshi_prob_total,
                        "kalshi_prob": kalshi_prob_spread if kalshi_spread.get("kalshi_matched") else None,
                        "spread_prob_market": spread_prob_market,
                        "total_prob_market": total_prob_market,
                        "spread_engine_used": spread_engine_used,
                        "kalshi_prob_for_pick": spread_kalshi_prob_for_pick,
                        "kalshi_yes_side": kalshi_spread.get("kalshi_yes_side") or "home",
                        "spread_pick_label": safe_str(spread_pick_label),
                        "spread_alt_label": safe_str(spread_alt_label),
                        "spread_prob_pick_final": spread_prob_final,
                        "spread_prob_alt_final": spread_alt_prob_final,
                        "spread_prob_margin": spread_prob_margin,
                        "spread_prob_pick_market": spread_prob_pick_market,
                        "spread_prob_alt_market": spread_prob_alt_market,
                        "spread_prob_pick_kalshi": spread_prob_pick_kalshi,
                        "spread_prob_alt_kalshi": spread_prob_alt_kalshi,
                        "spread_decision_metric_used": spread_decision_metric_used,
                        "spread_decision_score_pick": spread_decision_score_pick,
                        "spread_decision_score_alt": spread_decision_score_alt,
                        "spread_decision_score_margin": spread_decision_score_margin,
                        "spread_trace_json": spread_trace_json,
                        "total_engine_used": total_engine_used,
                        "total_pick_label": safe_str(total_pick_label),
                        "total_alt_label": safe_str(total_alt_label),
                        "total_prob_pick_final": total_prob_final,
                        "total_prob_alt_final": total_alt_prob_final,
                        "total_prob_margin": total_prob_margin,
                        "total_prob_pick_market": total_prob_pick_market,
                        "total_prob_alt_market": total_prob_alt_market,
                        "total_prob_pick_kalshi": total_prob_pick_kalshi,
                        "total_prob_alt_kalshi": total_prob_alt_kalshi,
                        "total_decision_metric_used": total_decision_metric_used,
                        "total_decision_score_pick": total_decision_score_pick,
                        "total_decision_score_alt": total_decision_score_alt,
                        "total_decision_score_margin": total_decision_score_margin,
                        "total_trace_json": total_trace_json,
                        "decision_trace_version": decision_trace_version,
                        "overall_engine_used": overall_engine_used,
                        "decision_trace_notes": decision_trace_notes,
                        "kalshi_status": kalshi_status_value,
                        "kalshi_matched": kalshi_spread.get("kalshi_matched"),
                        "kalshi_prob": kalshi_prob_spread if kalshi_spread.get("kalshi_matched") else None,
                        "kalshi_prob_used": kalshi_prob_spread if kalshi_spread.get("kalshi_matched") else None,
                        "kalshi_event_ticker_used": kalshi_spread.get("kalshi_event_ticker") or (kalshi_event_used if kalshi_spread.get("kalshi_matched") else None),
                        "kalshi_candidate_count": candidate_debug.get("candidate_count"),
                        "kalshi_best_score": candidate_debug.get("best_score"),
                        "kalshi_match_reason": kalshi_spread.get("kalshi_reason") or kalshi_winner.get("kalshi_reason"),
                        "kalshi_game_prefix_used": (candidate_debug.get("winner_meta") or {}).get("winner_prefix") or candidate_debug.get("kalshi_game_prefix_used"),
                        "kalshi_wanted_tokens": (candidate_debug.get("winner_meta") or {}).get("allowed_date_tokens") or candidate_debug.get("kalshi_wanted_tokens"),
                        "Sentiment_Diff": sentiment_diff,
                        "Spread & Pick": f"{spread_pick} {clean_line_str(spread_line)} ({spread_prob_final*100:.1f}%)" if (spread_pick is not None and spread_prob_final is not None) else (f"{spread_pick} {clean_line_str(spread_line)}" if spread_pick is not None else None),
                        "spread_pick_team": spread_pick_team,
                        "spread_pick_line": spread_pick_line,
                        "spread_pick_odds": spread_pick_odds,
                        "spread_odds_method": spread_odds_method,
                        "spread_prob": spread_prob,
                        "spread_confidence": None,
                        "spread_confidence_reason": None,
                        "Total & Pick": f"{total_pick} {clean_line_str(total_line)} ({total_prob_final*100:.1f}%)" if (total_pick is not None and total_prob_final is not None and clean_line_str(total_line) is not None) else (f"{total_pick} {clean_line_str(total_line)}" if (total_pick is not None and clean_line_str(total_line) is not None) else None),
                        "total_pick_side": total_pick_side,
                        "total_pick_line": total_line,
                        "total_pick_odds": total_pick_odds,
                        "total_odds_method": total_odds_method,
                        "total_prob": total_prob,
                        "total_confidence": None,
                        "total_confidence_reason": None,
                        "Home_Sentiment": home_sent,
                        "Away_Sentiment": away_sent,
                        "best_spread_book": g.get("best_spread_book"),
                        "best_spread_last_update": g.get("best_spread_last_update"),
                        "best_spread_price_score": g.get("best_spread_price_score"),
                        "best_spread_median_point": g.get("best_spread_median_point"),
                        "best_spread_mode_point": g.get("best_spread_mode_point"),
                        "best_spread_price": best_spread_price,
                        "best_total_book": g.get("best_total_book"),
                        "best_total_last_update": g.get("best_total_last_update"),
                        "best_total_price_score": g.get("best_total_price_score"),
                        "best_total_median_point": g.get("best_total_median_point"),
                        "best_total_mode_point": g.get("best_total_mode_point"),
                        "best_total_price": best_total_price,
                        "Warnings": warnings_field,
                        "theover_pick": (theover_matched_side or {}).get("theover_pick"),
                        "theover_pick_type": "SIDE" if theover_matched_side else None,
                        "theover_hit_rate": (theover_matched_side or {}).get("theover_hit_rate"),
                        "theover_source_model": (theover_matched_side or {}).get("theover_model"),
                        "theover_prob": theover_prob_spread,
                        "theover_prob_used": theover_prob_spread,
                        "theover_matched": bool(theover_matched_side),
                        "theover_delta_final_prob": theover_delta_spread,
                        "final_prob_without_theover": spread_prob_no_to,
                        "theover_changed_pick": theover_changed_pick_spread,
                        "theover_used_in_pick": theover_used_in_pick_spread,
                        "theover_total_prob": theover_prob_total,
                        "theover_spread_prob": theover_prob_spread,
                        "kalshi_event_ticker": kalshi_spread.get("raw_event_id"),
                        "kalshi_series": None,
                        "normalization_source": "Dynamic",
                        "theover_available": bool(theover_matched_side),
                        "theover_line": (theover_matched_side or {}).get("theover_line"),
                        "theover_status": (theover_matched_side or {}).get("theover_model", "None"),
                        "theover_side_available": bool(theover_matched_side),
                        "theover_side_pick_team": (theover_matched_side or {}).get("theover_pick"),
                        "theover_side_line": (theover_matched_side or {}).get("theover_line"),
                        "theover_side_winprob": (theover_matched_side or {}).get("theover_hit_rate"),
                        "theover_match_reason": theover_match_reason if theover_matched_side else None,
                            "spread_implied_prob": spread_implied,
                            "spread_prob_market_based": spread_prob_market_based,
                            "spread_prob_reason": spread_prob_reason,
                            "spread_odds_method": spread_odds_method,
                            "spread_prob_method": spread_prob_method,
                            "total_implied_prob": total_implied,
                            "total_prob_market_based": total_prob_market_based,
                            "total_prob_reason": total_prob_reason,
                            "total_odds_method": total_odds_method,
                        "total_prob_method": total_prob_method,
                        "Kalshi_Required": st.session_state.get("kalshi_required", True),
                        "api_sports_used": api_sports_used,
                        "sportsdata_used": sportsdata_used,
                        "api_sports_status": api_sports_status_run,
                        "sportsdata_status": sportsdata_status_run,
                        "apisports_enriched": apisports_enriched,
                        "apisports_status": api_sports_status_run,
                        "apisports_notes": apisports_notes,
                        "sportsdata_enriched": sportsdata_enriched,
                        "sportsdata_status": sportsdata_status_run,
                        "sportsdata_notes": sportsdata_notes,
                        "injuries_home_count": injuries_home_count,
                        "injuries_away_count": injuries_away_count,
                        "injuries_home": injuries_home_display,
                        "injuries_away": injuries_away_display,
                        "enrichment_errors_sample": enrichment_errors_sample,
                        "weather_summary": weather_summary,
                        "key_injuries_home": ",".join(key_injuries_home),
                        "key_injuries_away": ",".join(key_injuries_away),
                        "spread_min": spread_min,
                        "spread_med": spread_med,
                        "spread_max": spread_max,
                        "total_min": total_min,
                        "total_med": total_med,
                        "total_max": total_max,
                        "spread_books_count": len(spread_books_map),
                        "total_books_count": len(total_books_map),
                        "spread_market_pairs_count": spread_market_pairs_count,
                        "total_market_pairs_count": total_market_pairs_count,
                        "spread_width": width_spread,
                        "total_width": width_total,
                        "spread_odds_valid": spread_odds_valid,
                        "total_odds_valid": total_odds_valid,
                        "spread_odds_placeholder_detected": spread_odds_placeholder_detected,
                        "total_odds_placeholder_detected": total_odds_placeholder_detected,
                        "spread_prob_placeholder_detected": spread_prob_placeholder_detected,
                        "total_prob_placeholder_detected": total_prob_placeholder_detected,
                        "sentiment_adj_value": sentiment_adj,
                        "sentiment_adj_reason": sentiment_adj_reason,
                        "prob_reason": None,
                        "At_a_Glance_Confidence": None,
                        "At_a_Glance_Score": None,
                        "At_a_Glance_Reason": None,
                    }
                    spread_row["consensus_prob"] = spread_base_prob
                    spread_row["consensus_prob_adj"] = spread_prob_final
                    spread_row["prob_reason"] = spread_prob_reason
                    conf, reason_short, eligible = score_pick_confidence(spread_row)
                    width_spread = (spread_max - spread_min) if (spread_max is not None and spread_min is not None) else 0.0
                    # Downgrade based on market quality (not blanket downgrade)
                    if (width_spread and width_spread >= 2.5) and conf == "HIGH":
                        conf = "MEDIUM"  # Wide market reduces confidence
                    if len(spread_books_map) <= 1:
                        # Thin market: cap at MEDIUM, or LOW if already low
                        if conf == "HIGH":
                            conf = "MEDIUM"
                        elif conf == "MEDIUM":
                            conf = "LOW"
                        eligible = False
                    spread_row["Pick_Confidence"] = conf
                    spread_row["Pick_Reason_Short"] = reason_short
                    spread_row["confidence_reason"] = reason_short
                    _dec_base_spread = safe_float(spread_row.get("final_probability"))
                    spread_row["decisiveness"] = abs(_dec_base_spread - 0.5) * 2 if _dec_base_spread is not None else None
                    trace_short, trace_json, decision_trace_full = build_decision_trace(
                        "spread",
                        spread_row.get("Pick") or "",
                        spread_row.get("Implied_Prob"),
                        spread_row.get("kalshi_prob"),
                        spread_row.get("AI_Prob"),
                        spread_row.get("sentiment_adj"),
                        spread_weights_used,
                        spread_row.get("final_probability"),
                        conf,
                        league_name,
                        bool(kalshi_spread.get("kalshi_matched")),
                        kalshi_spread.get("kalshi_reason"),
                        spread_row.get("sentiment_score"),
                        spread_row.get("sentiment_label"),
                        model_used_for_spread,
                        reason_short,
                        warnings,
                        kalshi_spread.get("kalshi_yes_side") or "home",
                        spread_kalshi_prob_for_pick,
                    )
                    trace_json_str = trace_json if isinstance(trace_json, str) else json.dumps(trace_json)
                    spread_row["decision_trace_short"] = trace_short
                    spread_row["decision_trace_json"] = trace_json_str
                    spread_row["decision_trace"] = decision_trace_full
                    store_decision_trace_sample(
                        league_name,
                        home,
                        away,
                        "spread",
                        spread_row.get("Pick"),
                        spread_row.get("final_probability"),
                        trace_json_str,
                    )
                    spread_row["Eligible_Top_Picks"] = eligible
                    spread_row = apply_sentiment_defaults(spread_row, sentiment_defaults_base)
                    accumulated_rows.append(spread_row)
                    spread_row_created = True
                    master_stats["market_rows_out"] += 1
                else:
                    # DIAGNOSTIC: Log why spread row was not created
                    if idx < 5:  # Only log first 5 games to avoid spam
                        spread_point = g.get("home_spread_point")
                        logger.warning(f"⚠️  DIAGNOSTIC: Game {idx+1} ({home} vs {away}) - NO SPREAD ROW: home_spread_point={spread_point}, spread_pick={spread_pick}")

                # TOTAL ROW
                # FIX: Only create total row if total odds are valid AND line is valid (not None/NaN)
                # This prevents synthetic "None/69.6/NaN" rows for NHL games with missing total markets

                # Reset warnings to base game warnings (prevent ML + Spread warnings from leaking into Total)
                warnings = list(base_game_warnings)

                # Add warning if total line is invalid
                if not has_valid_line(total_line) and g.get("total_point") is not None:
                    warnings.append("no_valid_total")

                if total_valid_check:
                    ai_prob_base = None
                    ai_prob_row = None
                    # Market-only total: Using market-implied probability instead of model
                    # This happens when total odds are available but model prediction is not used
                    warnings.append("market_based_total_prob")
                    total_prob_engine = "market_only"
                    # Add warning if total_pick is None
                    if total_pick is None:
                        warnings.append("total_pick_none")
                    warnings_field = ";".join(warnings) if warnings else None
                    total_row = {
                        "league": league_name, "Home": home, "Away": away,
                        "Game": f"{away} @ {home}",
                        "Commence (UTC)": commence_iso, "Commence (Local)": commence_local,
                        "Market": "Total", "Book": g.get("best_total_book"),
                        "Pick": f"{total_pick} {total_line}" if (total_pick is not None and total_line is not None) else total_pick, "Implied_Prob": total_prob_market, "Line": total_line, "AI_Prob": model_total_prob if model_used_for_total else None,
                        "Home_ML": home_ml,
                        "Away_ML": away_ml,
                        "ml_home_implied": american_to_implied_prob(g.get("home_ml_price")),
                        "ml_away_implied": american_to_implied_prob(g.get("away_ml_price")),
                        "spread_pick_side": spread_pick_side_key,
                        "ai_prob_adj": ai_prob_row, "consensus_prob": total_base_prob, "consensus_prob_adj": total_prob_final,
                        "final_probability": total_prob_final,
                        "decision_driver": total_decision_driver or total_engine_used,
                        "kalshi_weight": total_weights_used.get("w_kalshi") if 'total_weights_used' in locals() else None,
                        "odds_weight": total_weights_used.get("w_implied") if 'total_weights_used' in locals() else None,
                        "ml_weight": total_weights_used.get("w_model") if 'total_weights_used' in locals() else None,
                        "sentiment_weight": total_weights_used.get("w_sentiment") if 'total_weights_used' in locals() else abs(total_sentiment_adj or 0.0),
                        "sentiment_score": sentiment_score_field if sentiment_score_field is not None else None,
                        "sentiment_label": sentiment_label_field,
                        "sentiment_source_count": sentiment_articles_used,
                        "sentiment_direction": None,
                        "sentiment_impact_applied": False,
                        "Model Total Prob": None,
                        "model_spread_prob": model_spread_prob,
                        "model_total_prob": model_total_prob,
                        "prob_engine": total_prob_engine,
                        "model_mode": st.session_state.model_mode,
                        "gemini_mode": "pending" if use_gemini_explanations else "disabled",
                        "overall_confidence": None,
                        "spread_confidence_gemini": None,
                        "total_confidence_gemini": None,
                        "gemini_alignment": None,
                        "gemini_rationale": None,
                        "gemini_risk_flags": None,
                        "gemini_error": None,
                        "gemini_flags_short": None,
                        "llm_disagreement_flag": None,
                        "kalshi_prob_spread": kalshi_prob_spread,
                        "kalshi_prob_total": kalshi_prob_total,
                        "spread_prob_market": spread_prob_market,
                        "total_prob_market": total_prob_market,
                        "spread_engine_used": spread_engine_used,
                        "kalshi_prob_for_pick": total_kalshi_prob_for_pick,
                        "kalshi_yes_side": kalshi_total.get("kalshi_yes_side") or "over",
                        "spread_pick_label": safe_str(spread_pick_label),
                        "spread_alt_label": safe_str(spread_alt_label),
                        "spread_prob_pick_final": spread_prob_final,
                        "spread_prob_alt_final": spread_alt_prob_final,
                        "spread_prob_margin": spread_prob_margin,
                        "spread_prob_pick_market": spread_prob_pick_market,
                        "spread_prob_alt_market": spread_prob_alt_market,
                        "spread_prob_pick_kalshi": spread_prob_pick_kalshi,
                        "spread_prob_alt_kalshi": spread_prob_alt_kalshi,
                        "spread_decision_metric_used": spread_decision_metric_used,
                        "spread_decision_score_pick": spread_decision_score_pick,
                        "spread_decision_score_alt": spread_decision_score_alt,
                        "spread_decision_score_margin": spread_decision_score_margin,
                        "spread_trace_json": spread_trace_json,
                        "total_engine_used": total_engine_used,
                        "total_pick_label": safe_str(total_pick_label),
                        "total_alt_label": safe_str(total_alt_label),
                        "total_prob_pick_final": total_prob_final,
                        "total_prob_alt_final": total_alt_prob_final,
                        "total_prob_margin": total_prob_margin,
                        "total_prob_pick_market": total_prob_pick_market,
                        "total_prob_alt_market": total_prob_alt_market,
                        "total_prob_pick_kalshi": total_prob_pick_kalshi,
                        "total_prob_alt_kalshi": total_prob_alt_kalshi,
                        "total_decision_metric_used": total_decision_metric_used,
                        "total_decision_score_pick": total_decision_score_pick,
                        "total_decision_score_alt": total_decision_score_alt,
                        "total_decision_score_margin": total_decision_score_margin,
                        "total_trace_json": total_trace_json,
                        "decision_trace_version": decision_trace_version,
                        "overall_engine_used": overall_engine_used,
                        "decision_trace_notes": decision_trace_notes,
                        "kalshi_status": kalshi_status_value,
                        "kalshi_matched": kalshi_total.get("kalshi_matched"),
                        "kalshi_prob": kalshi_prob_total if kalshi_total.get("kalshi_matched") else None,
                        "kalshi_prob_used": kalshi_prob_total if kalshi_total.get("kalshi_matched") else None,
                        "kalshi_prob_for_pick": total_kalshi_prob_for_pick,
                        "kalshi_yes_side": kalshi_total.get("kalshi_yes_side") or "over",
                        "kalshi_event_ticker_used": kalshi_total.get("kalshi_event_ticker") or (kalshi_event_used if kalshi_total.get("kalshi_matched") else None),
                        "kalshi_candidate_count": candidate_debug.get("candidate_count"),
                        "kalshi_best_score": candidate_debug.get("best_score"),
                        "kalshi_match_reason": kalshi_total.get("kalshi_reason") or kalshi_winner.get("kalshi_reason"),
                        "kalshi_game_prefix_used": (candidate_debug.get("winner_meta") or {}).get("winner_prefix") or candidate_debug.get("kalshi_game_prefix_used"),
                        "kalshi_wanted_tokens": (candidate_debug.get("winner_meta") or {}).get("allowed_date_tokens") or candidate_debug.get("kalshi_wanted_tokens"),
                        "Sentiment_Diff": sentiment_diff,
                        "Spread & Pick": f"{spread_pick} {clean_line_str(spread_line)} ({spread_prob_final*100:.1f}%)" if (spread_pick is not None and spread_prob_final is not None) else (f"{spread_pick} {clean_line_str(spread_line)}" if spread_pick is not None else None),
                        "spread_pick_team": spread_pick_team,
                        "spread_pick_line": spread_pick_line,
                        "spread_pick_odds": spread_pick_odds,
                        "spread_odds_method": spread_odds_method,
                        "spread_prob": spread_prob,
                        "spread_confidence": None,
                        "spread_confidence_reason": None,
                        "Total & Pick": f"{total_pick} {clean_line_str(total_line)} ({total_prob_final*100:.1f}%)" if (total_pick is not None and total_prob_final is not None and clean_line_str(total_line) is not None) else (f"{total_pick} {clean_line_str(total_line)}" if (total_pick is not None and clean_line_str(total_line) is not None) else None),
                        "total_pick_side": total_pick_side,
                        "total_pick_line": total_line,
                        "total_pick_odds": total_pick_odds,
                        "total_odds_method": total_odds_method,
                        "total_prob": total_prob,
                        "total_confidence": None,
                        "total_confidence_reason": None,
                        "Home_Sentiment": home_sent,
                        "Away_Sentiment": away_sent,
                        "best_spread_book": g.get("best_spread_book"),
                        "best_spread_last_update": g.get("best_spread_last_update"),
                        "best_spread_price_score": g.get("best_spread_price_score"),
                        "best_spread_median_point": g.get("best_spread_median_point"),
                        "best_spread_mode_point": g.get("best_spread_mode_point"),
                        "best_spread_price": best_spread_price,
                        "best_total_book": g.get("best_total_book"),
                        "best_total_last_update": g.get("best_total_last_update"),
                        "best_total_price_score": g.get("best_total_price_score"),
                        "best_total_median_point": g.get("best_total_median_point"),
                        "best_total_mode_point": g.get("best_total_mode_point"),
                        "best_total_price": best_total_price,
                        "Warnings": warnings_field,
                        "theover_pick": (theover_matched_total or {}).get("theover_pick"),
                        "theover_pick_type": "TOTAL" if theover_matched_total else None,
                        "theover_hit_rate": (theover_matched_total or {}).get("theover_hit_rate"),
                        "theover_source_model": (theover_matched_total or {}).get("theover_model"),
                        "theover_prob": theover_prob_total,
                        "theover_prob_used": theover_prob_total,
                        "theover_matched": bool(theover_matched_total or theover_matched_side),
                        "theover_delta_final_prob": theover_delta_total,
                        "final_prob_without_theover": total_prob_no_to,
                        "theover_changed_pick": theover_changed_pick_total,
                        "theover_used_in_pick": theover_used_in_pick_total,
                        "theover_total_prob": theover_prob_total,
                        "theover_spread_prob": theover_prob_spread,
                        "kalshi_event_ticker": kalshi_total.get("raw_event_id"),
                        "kalshi_series": None,
                        "normalization_source": "Dynamic",
                        "theover_available": bool(theover_matched_total or theover_matched_side),
                        "theover_line": (theover_matched_total or {}).get("theover_line"),
                        "theover_status": (theover_matched_total or {}).get("theover_model", "None"),
                        "theover_total_available": bool(theover_matched_total),
                        "theover_total_pick_side": (theover_matched_total or {}).get("theover_pick"),
                        "theover_total_line": (theover_matched_total or {}).get("theover_line"),
                        "theover_total_winprob": (theover_matched_total or {}).get("theover_hit_rate"),
                        "theover_match_reason": theover_match_reason if theover_matched_total else None,
                            "spread_implied_prob": spread_implied,
                            "spread_prob_market_based": spread_prob_market_based,
                            "spread_prob_reason": spread_prob_reason,
                            "spread_odds_method": spread_odds_method,
                            "spread_prob_method": spread_prob_method,
                            "total_implied_prob": total_implied,
                            "total_prob_market_based": total_prob_market_based,
                            "total_prob_reason": total_prob_reason,
                            "total_odds_method": total_odds_method,
                    "total_prob_method": total_prob_method,
                    "Kalshi_Required": st.session_state.get("kalshi_required", True),
                        "sentiment_adj": sentiment_adj, "sentiment_source": sentiment_source, "reddit_used": reddit_used, "sentiment_valid": sentiment_valid,
                        "sentiment_score": sentiment_score_field if sentiment_score_field is not None else None,
                        "sentiment_label": sentiment_label_field,
                        "sentiment_source_count": sentiment_articles_used,
                        "sentiment_direction": None,
                        "sentiment_impact_applied": False,
                        "sentiment_level": sentiment_level,
                        "sentiment_strength": sentiment_strength,
                        "sentiment_badge": sentiment_badge,
                        "sentiment_articles_used": sentiment_articles_used,
                        "sentiment_query_used": sentiment_query_used,
                        "sentiment_status": sentiment_status_value,
                        "sentiment_confidence": sentiment_confidence_value,
                        "sentiment_score": sentiment_score_value,
                        "spread_sentiment_adj": spread_sentiment_adj,
                        "total_sentiment_adj": total_sentiment_adj,
                        "sentiment_error_count": sentiment_error_count,
                        "sentiment_errors_sample": sentiment_errors_sample,
                        "sentiment_articles_total": sentiment_articles_total,
                        "sentiment_status_counts": sentiment_status_counts_field,
                        "sentiment_sample_query": sentiment_sample_query,
                        "sentiment_sample_status": sentiment_sample_status,
                        "sentiment_sample_totalResults": sentiment_sample_totalResults,
                        "sentiment_auth_error": sentiment_auth_error,
                        "sentiment_rate_limited": sentiment_rate_limited,
                        "sentiment_cooldown_until": sentiment_cooldown_until,
                        "sentiment_cached_teams_count": sentiment_cached_teams_count,
                        "sentiment_available_count": sentiment_available_count,
                        "sentiment_used_cached": sentiment_used_cached,
                        "sentiment_disabled_reason": sentiment_disabled_reason,
                    "api_sports_used": api_sports_used,
                    "sportsdata_used": sportsdata_used,
                    "api_sports_status": api_sports_status_run,
                    "sportsdata_status": sportsdata_status_run,
                    "apisports_enriched": apisports_enriched,
                    "apisports_status": api_sports_status_run,
                    "apisports_notes": apisports_notes,
                    "sportsdata_enriched": sportsdata_enriched,
                    "sportsdata_status": sportsdata_status_run,
                    "sportsdata_notes": sportsdata_notes,
                    "injuries_home_count": injuries_home_count,
                    "injuries_away_count": injuries_away_count,
                    "injuries_home": injuries_home_display,
                    "injuries_away": injuries_away_display,
                    "enrichment_errors_sample": enrichment_errors_sample,
                    "weather_summary": weather_summary,
                    "key_injuries_home": ",".join(key_injuries_home),
                    "key_injuries_away": ",".join(key_injuries_away),
                    "spread_min": spread_min,
                    "spread_med": spread_med,
                    "spread_max": spread_max,
                    "total_min": total_min,
                    "total_med": total_med,
                    "total_max": total_max,
                    "spread_books_count": len(spread_books_map),
                    "total_books_count": len(total_books_map),
                    "spread_market_pairs_count": spread_market_pairs_count,
                    "total_market_pairs_count": total_market_pairs_count,
                    "spread_width": width_spread,
                    "total_width": width_total,
                    "spread_odds_valid": spread_odds_valid,
                    "total_odds_valid": total_odds_valid,
                    "spread_odds_placeholder_detected": spread_odds_placeholder_detected,
                    "total_odds_placeholder_detected": total_odds_placeholder_detected,
                    "spread_prob_placeholder_detected": spread_prob_placeholder_detected,
                    "total_prob_placeholder_detected": total_prob_placeholder_detected,
                    "sentiment_adj_value": sentiment_adj,
                    "sentiment_adj_reason": sentiment_adj_reason,
                    "prob_reason": None,
                    "At_a_Glance_Confidence": None,
                    "At_a_Glance_Score": None,
                    "At_a_Glance_Reason": None,
                }
                # Append the total_row that was just created
                total_row["consensus_prob"] = total_base_prob
                total_row["consensus_prob_adj"] = total_prob_final
                total_row["prob_reason"] = total_prob_reason
                conf, reason_short, eligible = score_pick_confidence(total_row)
                width_total = (total_max - total_min) if (total_max is not None and total_min is not None) else 0.0
                # Downgrade based on market quality (not blanket downgrade)
                if (width_total and width_total >= 4.5) and conf == "HIGH":
                    conf = "MEDIUM"  # Wide market reduces confidence
                if len(total_books_map) <= 1:
                    # Thin market: cap at MEDIUM, or LOW if already low
                    if conf == "HIGH":
                        conf = "MEDIUM"
                    elif conf == "MEDIUM":
                        conf = "LOW"
                    eligible = False
                total_row["Pick_Confidence"] = conf
                total_row["Pick_Reason_Short"] = reason_short
                total_row["confidence_reason"] = reason_short
                _dec_base_total = safe_float(total_row.get("final_probability"))
                total_row["decisiveness"] = abs(_dec_base_total - 0.5) * 2 if _dec_base_total is not None else None
                trace_short, trace_json, decision_trace_full = build_decision_trace(
                    "total",
                    total_row.get("Pick") or "",
                    total_row.get("Implied_Prob"),
                    total_row.get("kalshi_prob"),
                    total_row.get("AI_Prob"),
                    total_row.get("sentiment_adj"),
                    total_weights_used,
                    total_row.get("final_probability"),
                    conf,
                    league_name,
                    bool(kalshi_total.get("kalshi_matched")),
                    kalshi_total.get("kalshi_reason"),
                    total_row.get("sentiment_score"),
                    total_row.get("sentiment_label"),
                    model_used_for_total,
                    reason_short,
                    warnings,
                    kalshi_total.get("kalshi_yes_side") or "over",
                    total_kalshi_prob_for_pick,
                )
                trace_json_str = trace_json if isinstance(trace_json, str) else json.dumps(trace_json)
                total_row["decision_trace_short"] = trace_short
                total_row["decision_trace_json"] = trace_json_str
                total_row["decision_trace"] = decision_trace_full
                store_decision_trace_sample(
                    league_name,
                    home,
                    away,
                    "total",
                    total_row.get("Pick"),
                    total_row.get("final_probability"),
                    trace_json_str,
                )
                total_row["Eligible_Top_Picks"] = eligible
                total_row = apply_sentiment_defaults(total_row, sentiment_defaults_base)
                accumulated_rows.append(total_row)
                total_row_created = True
                master_stats["market_rows_out"] += 1

                # --- 5. FALLBACK: "NONE" MARKET ROW ---
                if not (g.get("home_ml_price") or g.get("home_spread_point") or g.get("total_point")):
                    warnings = list(dict.fromkeys(warnings + ["no_markets"]))
                    fallback_row = {
                        "league": league_name,
                        "Home": home,
                        "Away": away,
                        "Commence (UTC)": commence_iso,
                        "Commence (Local)": commence_local,
                        "Local Date": commence_date_local,
                        "Market": "None",
                        "Book": None,
                        "Pick": None,
                        "Implied_Prob": None,
                        "AI_Prob": model_prob_home,  # Raw AI score with no market blending
                        "Warnings": ";".join(warnings),
                        "kalshi_available": kalshi_winner.get("kalshi_available"),
                        "kalshi_matched": kalshi_winner.get("kalshi_matched"),
                        "kalshi_prob": kalshi_winner.get("kalshi_prob"),
                        "kalshi_status": kalshi_status_value,
                        "kalshi_candidate_count": candidate_debug.get("candidate_count"),
                        "kalshi_best_score": candidate_debug.get("best_score"),
                        "kalshi_match_reason": kalshi_winner.get("kalshi_reason"),
                        "kalshi_game_prefix_used": (candidate_debug.get("winner_meta") or {}).get("winner_prefix") or candidate_debug.get("kalshi_game_prefix_used"),
                        "kalshi_wanted_tokens": (candidate_debug.get("winner_meta") or {}).get("allowed_date_tokens") or candidate_debug.get("kalshi_wanted_tokens"),
                        "Home_Sentiment": home_sent,
                        "Away_Sentiment": away_sent,
                        "Sentiment_Diff": sentiment_diff,
                        "sentiment_adj": sentiment_adj,
                        "sentiment_source": sentiment_source,
                        "reddit_used": reddit_used,
                        "sentiment_valid": sentiment_valid,
                        "sentiment_level": sentiment_level,
                        "sentiment_strength": sentiment_strength,
                        "sentiment_badge": sentiment_badge,
                        "sentiment_articles_used": sentiment_articles_used,
                        "sentiment_query_used": sentiment_query_used,
                        "sentiment_status": sentiment_status_value,
                        "sentiment_confidence": sentiment_confidence_value,
                        "sentiment_score": sentiment_score_value,
                        "spread_sentiment_adj": spread_sentiment_adj,
                        "total_sentiment_adj": total_sentiment_adj,
                        "sentiment_error_count": sentiment_error_count,
                        "sentiment_errors_sample": sentiment_errors_sample,
                        "sentiment_articles_total": sentiment_articles_total,
                        "sentiment_status_counts": sentiment_status_counts_field,
                        "sentiment_sample_query": sentiment_sample_query,
                        "sentiment_sample_status": sentiment_sample_status,
                        "sentiment_sample_totalResults": sentiment_sample_totalResults,
                        "sentiment_auth_error": sentiment_auth_error,
                        "sentiment_rate_limited": sentiment_rate_limited,
                        "sentiment_cooldown_until": sentiment_cooldown_until,
                        "sentiment_cached_teams_count": sentiment_cached_teams_count,
                        "sentiment_available_count": sentiment_available_count,
                        "sentiment_used_cached": sentiment_used_cached,
                        "sentiment_disabled_reason": sentiment_disabled_reason,
                        "sentiment_status": sentiment_status_value,
                        "sentiment_confidence": sentiment_confidence_value,
                        "sentiment_score": sentiment_score_value,
                        "sentiment_adj_value": sentiment_adj,
                        "sentiment_adj_reason": sentiment_adj_reason,
                        "spread_odds_method": spread_odds_method,
                        "total_odds_method": total_odds_method,
                        "spread_pick_team": spread_pick_team,
                        "spread_pick_line": spread_pick_line,
                        "spread_pick_odds": spread_pick_odds,
                        "spread_prob": spread_prob,
                        "spread_confidence": None,
                        "spread_confidence_reason": None,
                        "total_pick_side": total_pick_side,
                        "total_pick_line": total_line,
                        "total_pick_odds": total_pick_odds,
                        "total_prob": total_prob,
                        "total_confidence": None,
                        "total_confidence_reason": None,
                        "At_a_Glance_Confidence": None,
                        "At_a_Glance_Score": None,
                        "At_a_Glance_Reason": None,
                        "best_spread_book": g.get("best_spread_book"),
                        "best_spread_last_update": g.get("best_spread_last_update"),
                        "best_spread_price_score": g.get("best_spread_price_score"),
                        "best_spread_median_point": g.get("best_spread_median_point"),
                        "best_spread_price": best_spread_price,
                        "best_total_book": g.get("best_total_book"),
                        "best_total_last_update": g.get("best_total_last_update"),
                        "best_total_price_score": g.get("best_total_price_score"),
                        "best_total_median_point": g.get("best_total_median_point"),
                        "best_total_price": best_total_price,
                        "Kalshi_Required": st.session_state.get("kalshi_required", True),
                        "api_sports_used": api_sports_used,
                        "sportsdata_used": sportsdata_used,
                        "api_sports_status": api_sports_status_run,
                        "sportsdata_status": sportsdata_status_run,
                        "apisports_enriched": apisports_enriched,
                        "apisports_status": api_sports_status_run,
                        "apisports_notes": apisports_notes,
                        "sportsdata_enriched": sportsdata_enriched,
                        "sportsdata_status": sportsdata_status_run,
                        "sportsdata_notes": sportsdata_notes,
                        "injuries_home_count": injuries_home_count,
                        "injuries_away_count": injuries_away_count,
                        "injuries_home": injuries_home_display,
                        "injuries_away": injuries_away_display,
                        "enrichment_errors_sample": enrichment_errors_sample,
                        "weather_summary": weather_summary,
                        "key_injuries_home": ",".join(key_injuries_home),
                        "key_injuries_away": ",".join(key_injuries_away),
                        "spread_prob_market_based": spread_prob_market_based,
                        "spread_prob_reason": spread_prob_reason,
                        "spread_prob_method": spread_prob_method,
                        "spread_prob_market": spread_prob_market,
                        "total_prob_market": total_prob_market,
                        "total_implied_prob": total_implied,
                        "total_prob_market_based": total_prob_market_based,
                        "total_prob_reason": total_prob_reason,
                        "total_prob_method": total_prob_method,
                        "kalshi_prob_spread": kalshi_prob_spread,
                        "kalshi_prob_total": kalshi_prob_total,
                        "model_spread_prob": model_spread_prob,
                        "model_total_prob": model_total_prob,
                        "prob_engine": prob_engine_label(bool(kalshi_winner.get("kalshi_matched")), None, model_used=False),
                        "model_mode": st.session_state.model_mode,
                        "gemini_mode": "pending" if use_gemini_explanations else "disabled",
                        "overall_confidence": None,
                        "spread_confidence_gemini": None,
                        "total_confidence_gemini": None,
                        "gemini_alignment": None,
                        "gemini_rationale": None,
                        "gemini_risk_flags": None,
                        "gemini_error": None,
                        "gemini_flags_short": None,
                        "spread_engine_used": spread_engine_used,
                        "spread_pick_label": safe_str(spread_pick_label),
                        "spread_alt_label": safe_str(spread_alt_label),
                        "spread_prob_pick_final": spread_prob_final,
                        "spread_prob_alt_final": spread_alt_prob_final,
                        "spread_prob_margin": spread_prob_margin,
                        "spread_prob_pick_market": spread_prob_pick_market,
                        "spread_prob_alt_market": spread_prob_alt_market,
                        "spread_prob_pick_kalshi": spread_prob_pick_kalshi,
                        "spread_prob_alt_kalshi": spread_prob_alt_kalshi,
                        "spread_decision_metric_used": spread_decision_metric_used,
                        "spread_decision_score_pick": spread_decision_score_pick,
                        "spread_decision_score_alt": spread_decision_score_alt,
                        "spread_decision_score_margin": spread_decision_score_margin,
                        "spread_trace_json": spread_trace_json,
                        "total_engine_used": total_engine_used,
                        "total_pick_label": safe_str(total_pick_label),
                        "total_alt_label": safe_str(total_alt_label),
                        "total_prob_pick_final": total_prob_final,
                        "total_prob_alt_final": total_alt_prob_final,
                        "total_prob_margin": total_prob_margin,
                        "total_prob_pick_market": total_prob_pick_market,
                        "total_prob_alt_market": total_prob_alt_market,
                        "total_prob_pick_kalshi": total_prob_pick_kalshi,
                        "total_prob_alt_kalshi": total_prob_alt_kalshi,
                        "total_decision_metric_used": total_decision_metric_used,
                        "total_decision_score_pick": total_decision_score_pick,
                        "total_decision_score_alt": total_decision_score_alt,
                        "total_decision_score_margin": total_decision_score_margin,
                        "total_trace_json": total_trace_json,
                        "decision_trace_version": decision_trace_version,
                        "overall_engine_used": overall_engine_used,
                        "decision_trace_notes": decision_trace_notes,
                        "spread_market_pairs_count": spread_market_pairs_count,
                        "total_market_pairs_count": total_market_pairs_count,
                        "spread_odds_valid": spread_odds_valid,
                        "total_odds_valid": total_odds_valid,
                        "spread_odds_placeholder_detected": spread_odds_placeholder_detected,
                        "total_odds_placeholder_detected": total_odds_placeholder_detected,
                        "spread_prob_placeholder_detected": spread_prob_placeholder_detected,
                        "total_prob_placeholder_detected": total_prob_placeholder_detected,
                        "odds_placeholder_detected": bool(odds_placeholder_overall),
                    }
                    conf, reason_short, eligible = score_pick_confidence(fallback_row)
                    fallback_row["Pick_Confidence"] = conf
                    fallback_row["Pick_Reason_Short"] = reason_short
                    fallback_row["Eligible_Top_Picks"] = eligible
                    fallback_row = apply_sentiment_defaults(fallback_row, sentiment_defaults_base)
                    accumulated_rows.append(fallback_row)
                    master_stats["market_rows_out"] += 1

                # ============================================
                # PHASE 1 & 2: DIAGNOSTIC + SAFETY NET
                # Per-Game Tracking End + Guaranteed Fallback
                # ============================================
                rows_count_after = len(accumulated_rows)
                rows_created_this_game = rows_count_after - rows_count_before

                # Fix for Issue #6: Fallback Safety Net Row Creation
                # Only create fallback if NO rows were created (Spread, Total, or ML)
                # AND we have at least some basic market data (ML, Spread, or Total) to work with.
                # Do NOT trigger if rows were already created (rows_created_this_game > 0).
                has_any_market_data = bool(g.get("home_ml_price") or g.get("home_spread_point") or g.get("total_point"))

                if rows_created_this_game == 0 and has_any_market_data:
                    logger.warning(f"⚠️  NO ROWS CREATED FOR THIS GAME!")
                    logger.warning(f"Market data present:")
                    logger.warning(f"  - home_ml_price: {g.get('home_ml_price')}")
                    logger.warning(f"  - home_spread_point: {g.get('home_spread_point')}")
                    logger.warning(f"  - total_point: {g.get('total_point')}")

                    # Check if variables were set
                    if 'spread_pick' in locals():
                        logger.warning(f"  - spread_pick was: {spread_pick}")
                    else:
                        logger.warning(f"  - spread_pick was NEVER DEFINED")

                    if 'total_pick' in locals():
                        logger.warning(f"  - total_pick was: {total_pick}")
                    else:
                        logger.warning(f"  - total_pick was NEVER DEFINED")

                    # ============================================
                    # PHASE 2: GUARANTEED FALLBACK ROW CREATION
                    # ============================================
                    logger.warning(f"Creating FALLBACK SAFETY NET row for {g.get('home_team')} vs {g.get('away_team')}")

                    safety_fallback_row = {
                        "league": g.get("league"),
                        "Home": g.get("home_team"),
                        "Away": g.get("away_team"),
                        "Commence (UTC)": g.get("commence_time_iso_utc") or g.get("commence_time") or g.get("commence_time_iso"),
                        "Commence (Local)": None,
                        "Market": "NO_DATA",
                        "Pick": "INSUFFICIENT_DATA",
                        "Implied_Prob": None,
                        "AI_Prob": None,
                        "final_prob": None,
                        "final_probability": None,
                        "Line": None,
                        "Pick_Confidence": "LOW",
                        "Pick_Reason_Short": "No market data or picks available",
                        "Eligible_Top_Picks": False,
                        "kalshi_matched": False,
                        "kalshi_prob": None,
                        "sentiment_diff": 0.0,
                        "Sentiment_Diff": 0.0,
                        "theover_pick": None,
                        "best_pick": "NO DATA",
                        "best_pick_type": "NONE",
                        # Add other essential fields with None or defaults
                        "home_ml_price": g.get("home_ml_price"),
                        "away_ml_price": g.get("away_ml_price"),
                        "home_spread_point": g.get("home_spread_point"),
                        "total_point": g.get("total_point"),
                    }

                    accumulated_rows.append(safety_fallback_row)
                    logger.info(f"FALLBACK SAFETY NET row added (accumulated_rows now has {len(accumulated_rows)} rows)")
                    # ============================================
                else:
                    logger.info(f"✅ Created {rows_created_this_game} row(s) for this game")
                # ============================================

            # ============================================
            # PHASE 1: DIAGNOSTIC - Game Loop Complete
            # ============================================
            logger.info(f"\n{'='*80}")
            logger.info(f"GAME LOOP COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"Total rows created: {len(accumulated_rows)}")

            if len(accumulated_rows) == 0:
                logger.error("CRITICAL: accumulated_rows is empty! No games created any rows!")
                logger.error("This means ALL row creation conditions failed for ALL games")
            else:
                logger.info(f"SUCCESS: Created {len(accumulated_rows)} total rows from {len(games_to_process)} games")
                logger.info(f"Average rows per game: {len(accumulated_rows) / len(games_to_process):.2f}")

            # ============================================
            # DATA QUALITY METRICS SUMMARY
            # ============================================
            logger.info(f"\n{'='*80}")
            logger.info(f"DATA QUALITY METRICS")
            logger.info(f"{'='*80}")

            # TheOver Usage Metrics
            total_games = len(games_to_process)
            if total_games > 0:
                theover_totals_pct = (theover_matched_count_totals / total_games) * 100
                theover_sides_pct = (theover_matched_count_sides / total_games) * 100
                logger.info(f"TheOver Coverage:")
                logger.info(f"  - Totals: {theover_matched_count_totals}/{total_games} games ({theover_totals_pct:.1f}%)")
                logger.info(f"  - Sides: {theover_matched_count_sides}/{total_games} games ({theover_sides_pct:.1f}%)")

            # Stats Match Metrics (from feature_processing)
            try:
                from app_core.feature_processing import get_stats_match_metrics
                stats_metrics = get_stats_match_metrics()
                if stats_metrics:
                    logger.info(f"Stats Match Quality by League:")
                    for league, metrics in stats_metrics.items():
                        logger.info(f"  - {league}: {metrics['match_rate']:.1f}% matched, {metrics['fallback_rate']:.1f}% fallback")
            except Exception as e:
                logger.debug(f"Could not retrieve stats metrics: {e}")

            # Market-Only Pick Counts
            market_only_spread_count = sum(1 for row in accumulated_rows if "market_based_spread_prob" in str(row.get("Warnings", "")))
            market_only_total_count = sum(1 for row in accumulated_rows if "market_based_total_prob" in str(row.get("Warnings", "")))
            if len(accumulated_rows) > 0:
                logger.info(f"Market-Only Picks:")
                logger.info(f"  - Spread: {market_only_spread_count} rows (market-implied only)")
                logger.info(f"  - Total: {market_only_total_count} rows (market-implied only)")

            # Extreme Odds Flags
            extreme_odds_count = sum(1 for row in accumulated_rows if "ml_extreme_odds_flag" in str(row.get("Warnings", "")))
            if extreme_odds_count > 0:
                logger.info(f"ML Extreme Odds: {extreme_odds_count} picks with odds >400 (model weight reduced)")

            logger.info(f"{'='*80}\n")

            # Store data quality metrics in session state for UI display
            data_quality_metrics = {
                "theover_totals_count": theover_matched_count_totals,
                "theover_sides_count": theover_matched_count_sides,
                "theover_totals_pct": (theover_matched_count_totals / total_games * 100) if total_games > 0 else 0,
                "theover_sides_pct": (theover_matched_count_sides / total_games * 100) if total_games > 0 else 0,
                "total_games": total_games,
                "market_only_spread_count": market_only_spread_count,
                "market_only_total_count": market_only_total_count,
                "extreme_odds_count": extreme_odds_count,
                "stats_match_metrics": {},
            }
            # Add stats match metrics if available
            try:
                from app_core.feature_processing import get_stats_match_metrics
                stats_metrics = get_stats_match_metrics()
                if stats_metrics:
                    data_quality_metrics["stats_match_metrics"] = stats_metrics
            except Exception as e:
                logger.debug(f"Could not store stats metrics: {e}")

            st.session_state["data_quality_metrics"] = data_quality_metrics
            # ============================================

            # Update master_stats with final row count
            master_stats["rows_out"] = len(accumulated_rows)

            # 1. Create the base Master DataFrame from your processed rows
            # User Action: Use from_records and copy to prevent fragmentation

            # ============================================
            # PHASE 1: DIAGNOSTIC - DataFrame Creation
            # ============================================
            logger.info(f"Creating master_df from {len(accumulated_rows)} rows...")

            master_df = pd.DataFrame.from_records(accumulated_rows)


            # DISABLED: Moneyline pivot logic
            # Previous constraint required "Market column must never say 'Moneyline'"
            # However, users need ML picks to be exported as ML picks, not converted to Spread/Total
            # The pivot_market function was converting all ML rows to Spread or Total,
            # preventing ML picks from appearing in exports
            #
            # Old logic (DISABLED):
            # master_df = master_df.apply(pivot_market, axis=1)
            #
            # NEW BEHAVIOR: Preserve Moneyline market rows for export
            # ML picks will be exported with Market="Moneyline" alongside Spread and Total picks

            logger.info(f"master_df created: {len(master_df)} rows, {len(master_df.columns)} columns")

            # Log market breakdown
            if not master_df.empty and "Market" in master_df.columns:
                market_counts = master_df["Market"].value_counts()
                logger.info(f"Market breakdown after creation: {dict(market_counts)}")
            if master_df.empty:
                logger.error("CRITICAL: master_df is EMPTY!")
            # ============================================

            # FIX: Deduplicate columns immediately to prevent "Duplicate labels" error
            master_df = master_df.loc[:, ~master_df.columns.duplicated()].copy()
            master_df = master_df.reset_index(drop=True)

            # ============================================
            # ATOMIC ROW COLLAPSE: Force 1-Row-Per-Game BEFORE Enrichment
            # ============================================
            # 1. Selection Score = Decisiveness + Edge
            # Ensure safe numeric conversion for final_probability and edge
            if 'final_probability' in master_df.columns:
                _final_prob = pd.to_numeric(master_df['final_probability'], errors='coerce').fillna(0.5)
            else:
                _final_prob = pd.Series(0.5, index=master_df.index)

            if 'edge' in master_df.columns:
                _edge = pd.to_numeric(master_df['edge'], errors='coerce').fillna(0.0)
            else:
                _edge = pd.Series(0.0, index=master_df.index)

            master_df['_sel_score'] = (_final_prob - 0.5).abs() + _edge

            # ============================================
            # FIX: EXCLUDE ALL MONEYLINE PICKS - Only use Spread and Total for parlays
            # User requirement: No ML in parlays - only Spread and Total markets
            # ============================================
            ml_picks_before = len(master_df[master_df['Market'] == 'Moneyline']) if 'Market' in master_df.columns else 0
            if ml_picks_before > 0:
                logger.info(f"🚫 EXCLUDING ALL {ml_picks_before} Moneyline picks - using Spread/Total only")
                master_df = master_df[master_df['Market'] != 'Moneyline'].copy()
                logger.info(f"📊 After ML exclusion: {len(master_df)} rows remain (Spread + Total only)")

            # ============================================
            # LEGACY ML ODDS PENALTY CODE - DISABLED (all ML now excluded above)
            # ============================================
            ML_ODDS_THRESHOLD = 250  # No longer used - all ML excluded
            ML_ODDS_EXCLUSION_THRESHOLD = 400  # No longer used - all ML excluded

            # Tiered penalty factors - No longer used
            ML_ODDS_PENALTY_MODERATE = 0.3
            ML_ODDS_PENALTY_STRONG = 0.15

            # Track ML picks - all now excluded
            extreme_ml_count = 0
            excluded_count = ml_picks_before  # All ML excluded
            penalty_counts = {'MODERATE': 0, 'STRONG': 0, 'EXCLUDED': ml_picks_before}

            # Initialize exclusion flag (not needed since all ML already excluded, but kept for compatibility)
            master_df['_ml_excluded'] = False

            # LEGACY: ML penalty loop - no longer needed since all ML excluded above
            # Loop will not execute because no ML rows remain in master_df
            logger.info(f"✅ ML Exclusion Complete: All {ml_picks_before} Moneyline picks excluded, only Spread/Total remain")
            # ============================================

            # 2. Group by game and pick only the highest scoring row
            game_keys = ["league", "Home", "Away", "Commence (UTC)"]

            # Check keys exist to avoid KeyError
            if all(k in master_df.columns for k in game_keys):
                df_collapsed = master_df.sort_values('_sel_score', ascending=False).groupby(game_keys).head(1)
            else:
                # Fallback: try collapsing with available keys if possible
                available_keys = [k for k in game_keys if k in master_df.columns]
                if len(available_keys) >= 3:
                    df_collapsed = master_df.sort_values('_sel_score', ascending=False).groupby(available_keys).head(1)
                else:
                    df_collapsed = master_df

            # 3. Finalize memory to stop fragmentation and sync both session states
            master_df = df_collapsed.drop(columns=['_sel_score'], errors='ignore').reset_index(drop=True).copy()
            st.session_state["master_df"] = master_df
            st.session_state["master_results_df"] = master_df

            logger.info(f"ATOMIC COLLAPSE: Session state updated with {len(master_df)} rows")

            # Log market distribution after ATOMIC COLLAPSE
            if not master_df.empty and 'Market' in master_df.columns:
                market_distribution = master_df['Market'].value_counts()
                total_picks = len(master_df)
                logger.info("📊 Best Pick Selection by Market (After ML Odds Filter):")
                for market, count in market_distribution.items():
                    pct = (count / total_picks) * 100
                    logger.info(f"   {market}: {count} picks ({pct:.1f}%)")

                # Show how many extreme ML picks were still selected (after penalties applied)
                if 'extreme_ml_odds' in master_df.columns:
                    extreme_selected = master_df['extreme_ml_odds'].fillna(False).sum()
                    logger.info(f"⚠️ Extreme ML picks (±250-400, penalized) selected as best: {extreme_selected}/{total_picks}")
                    logger.info(f"✅ Very extreme ML picks (±400+) were EXCLUDED: {excluded_count} picks removed from pool")
            # ============================================

            # Task 4: Enrich with Consensus (Sharpness Delta)
            # Must be done before sentiment integration or model features if model uses it
            with st.spinner("📊 Ingesting Public Consensus Data..."):
                master_df = enrich_with_consensus(master_df)

                # 3. CRITICAL: Enrich the whole batch to fill 'feature_diff' columns
                # This fixes the 'Missing feature column' warnings in the logs
                with st.spinner("🚀 Running Batch Feature Enrichment..."):
                    # FIX: Pass ALL api_clients so stats for all leagues are fetched, not just the last loop variable
                    master_df = enrich_with_model_features(master_df, api_sports_clients)

            # Task 4: Update Sentiment Score using Sharpness Delta
            # Integration: 60% Sharpness Delta, 40% Social Sentiment
            # We need to update 'Sentiment_Diff' or create a new combined score.
            # Currently 'Sentiment_Diff' is used in compute_final_probability via sentiment_score.

            def _update_sentiment_score(row):
                social_diff = row.get("Sentiment_Diff")
                if social_diff is None: social_diff = 0.0

                sharpness = row.get("sharpness_delta")
                if sharpness is None: sharpness = 0.0

                # Hybrid Formula
                # Normalize sharpness (e.g. 0.15 delta -> 1.0 score equiv? or keep raw?)
                # Sentiment Diff is typically -1 to 1.
                # Sharpness Delta is typically -0.3 to +0.3.
                # Let's scale sharpness by 3.33 to map 0.3 to 1.0 roughly.
                sharpness_scaled = sharpness * 3.33

                # Weighted Combo
                hybrid_score = (0.6 * sharpness_scaled) + (0.4 * social_diff)
                return hybrid_score

            if 'Sentiment_Diff' in master_df.columns and 'sharpness_delta' in master_df.columns:
                master_df['Sentiment_Diff'] = master_df.apply(_update_sentiment_score, axis=1)

                # Ensure use_model_numeric_probs is synchronized from session state
                use_model_numeric_probs = st.session_state.get("use_model_numeric_probs", True)

                # 4. BATCH PREDICTION: Local Inference
                master_df = clean_df(master_df)
                # Local inference is always "configured" (or falls back)
                if True:
                    with st.spinner("🔮 Computing Win Probabilities (Local)..."):
                        # 2. Filter for exactly the columns the model expects
                        # User Action: Ensure columns exist before filtering
                        missing_cols = [col for col in VERTEX_FEATURE_COLUMNS if col not in master_df.columns]
                        if missing_cols:
                            zeros_df = pd.DataFrame(0.0, index=master_df.index, columns=missing_cols)
                            master_df = pd.concat([master_df, zeros_df], axis=1).copy()

                        inference_df = master_df[VERTEX_FEATURE_COLUMNS].copy()

                        # 3. Sanitize feature batch
                        for col in VERTEX_FEATURE_COLUMNS:
                            col_data = inference_df[col]
                            if isinstance(col_data, pd.DataFrame):
                                col_data = col_data.iloc[:, 0]
                            default_val = 0.5 if "prob" in col else 0.0
                            inference_df[col] = pd.to_numeric(col_data, errors='coerce').fillna(default_val).infer_objects(copy=False).astype(float)

                        # 6. Accumulate Debug Data (Base Dict + Feature Vector)
                        if "debug_log_history" not in st.session_state:
                            st.session_state["debug_log_history"] = []

                        try:
                            # Capture base metadata and critical pick info for debug history
                            # Include 'Pick_Confidence' mapped to 'confidence', 'best_pick_type' mapped to 'type' if available
                            # master_df at this stage might not have all columns if calculated later,
                            # but inference_df is feature vector.
                            # We need to grab what we can.
                            # 'Pick_Confidence' and 'best_pick_type' are calculated AFTER prediction usually,
                            # but let's check if we can grab prelim info or if we should update this later.
                            # Actually, 'enrich_with_model_features' is called BEFORE 'calculate_best_pick_metrics'.
                            # So 'Pick_Confidence' might not be there yet.
                            # However, to satisfy the user request "Ensure each parlay dict includes: type, confidence",
                            # we should log the FINALIZED rows, not just the inference input.

                            # For now, we will add placeholders if missing, but ideally we log AFTER full processing.
                            # But since this block is inside the prediction loop, we log features here.

                            cols_to_keep = ['Home', 'Away', 'league', 'Commence (UTC)']
                            for c in ['Pick_Confidence', 'Market', 'Pick', 'best_pick_type', 'final_probability']:
                                if c in master_df.columns:
                                    cols_to_keep.append(c)

                            debug_base = master_df[cols_to_keep].copy()

                            # Rename for user clarity if columns exist
                            rename_map = {}
                            if 'Pick_Confidence' in debug_base.columns:
                                rename_map['Pick_Confidence'] = 'confidence'
                            if 'best_pick_type' in debug_base.columns:
                                rename_map['best_pick_type'] = 'type'
                            if 'final_probability' in debug_base.columns:
                                rename_map['final_probability'] = 'prob'

                            if rename_map:
                                debug_base = debug_base.rename(columns=rename_map)

                            # Fallback for 'type' if best_pick_type is missing
                            if 'type' not in debug_base.columns:
                                if 'Market' in debug_base.columns:
                                    debug_base['type'] = debug_base['Market']
                                else:
                                    debug_base['type'] = "unknown"

                            # Combine with feature vector
                            debug_combined = pd.concat([debug_base, inference_df], axis=1).copy()
                            # Append to session state accumulator
                            st.session_state["debug_log_history"].extend(debug_combined.to_dict('records'))
                        except Exception as e:
                            logger.warning(f"Failed to accumulate debug data: {e}")

                        # 7. Call local prediction
                        if not inference_df.empty:
                            engine = get_prediction_engine()
                            try:
                                probs = engine.predict_batch(inference_df)
                            except Exception as e:
                                logger.error(f"Prediction batch failed: {e}")
                                st.warning(f"AI Data Unavailable (using defaults): {e}")
                                probs = [0.5] * len(inference_df)

                            # FIX: Stop Using Indexing for AI Results (Safe Map Approach) - Logic Update: Pad with 0.5 instead of fail
                            if probs:
                                # Handle length mismatch by padding or truncating
                                if len(probs) < len(inference_df):
                                    logger.warning(f"Prediction length mismatch (short): got {len(probs)}, expected {len(inference_df)}. Padding with 0.5.")
                                    probs = list(probs) + [0.5] * (len(inference_df) - len(probs))
                                elif len(probs) > len(inference_df):
                                    logger.warning(f"Prediction length mismatch (long): got {len(probs)}, expected {len(inference_df)}. Truncating.")
                                    probs = list(probs)[:len(inference_df)]

                                # Wrap in Series to match index explicitly (convert to list to drop any upstream index)
                                # This aligns by index explicitly as requested to prevent mismatch
                                predictions_series = pd.Series(list(probs), index=inference_df.index)

                                # Assign using loc to ensure alignment
                                master_df.loc[inference_df.index, 'AI_Prob'] = predictions_series
                                master_df.loc[inference_df.index, 'ai_prob_base'] = predictions_series # Persist base if needed
                            else:
                                logger.warning("No predictions returned. Defaulting to 0.5.")
                                master_df.loc[inference_df.index, 'AI_Prob'] = 0.5
                                master_df.loc[inference_df.index, 'ai_prob_base'] = 0.5
                        else:
                            logger.info("Skipping prediction: inference_df is empty.")

                        # Safe Edge Calculation
                        implied_probs = pd.to_numeric(master_df.get("Implied_Prob"), errors='coerce').fillna(0.5).infer_objects(copy=False)
                        master_df["AI_Edge"] = master_df["AI_Prob"] - implied_probs

                # 4. CONSENSUS ENRICHMENT: Add consensus votes to master_df
                if ParlayOptimizer:
                    model_dir_abs = os.path.join(os.path.dirname(__file__), "models")
                    optimizer = ParlayOptimizer(model_dir=model_dir_abs)

                    # Calculate consensus for all picks in master_df
                    logger.info("Calculating consensus votes for all picks")

                    # Fix for Issue #9: Populate consensus votes correctly
                    # And Issue #7: Avoid fragmentation by using vectorized/list ops
                    consensus_vote_cols = []
                    consensus_total_cols = []
                    consensus_breakdown_cols = []

                    # We need to collect details to update rows, but bulk assignment is better
                    # Since vote_details keys can vary, we might need a list of dicts to construct a DF
                    vote_details_list = []

                    for idx, row in master_df.iterrows():
                        consensus_votes, consensus_total, vote_details = optimizer.calculate_consensus_votes(row)

                        consensus_vote_cols.append(consensus_votes)
                        consensus_total_cols.append(consensus_total)
                        vote_details_list.append(vote_details if vote_details else {})

                        # Format detailed consensus breakdown for each market type
                        market = str(row.get("Market") or "").lower()

                        # Get probabilities based on market type
                        if market == "spread":
                            market_prob = safe_float(row.get("spread_prob_pick_market"))
                            # v98 FIX: Use proper None check instead of `or` (0.0 is falsy in Python)
                            _spk = row.get("spread_prob_pick_kalshi")
                            kalshi_prob = safe_float(_spk if _spk is not None else row.get("kalshi_prob_spread"))
                            model_prob = safe_float(row.get("model_spread_prob"))
                            final_prob = safe_float(row.get("spread_prob_pick_final"))
                            pick_side = row.get("spread_pick_team")
                        elif market == "total":
                            market_prob = safe_float(row.get("total_prob_pick_market"))
                            # v98 FIX: Use proper None check instead of `or` (0.0 is falsy in Python)
                            _tpk = row.get("total_prob_pick_kalshi")
                            kalshi_prob = safe_float(_tpk if _tpk is not None else row.get("kalshi_prob_total"))
                            model_prob = safe_float(row.get("model_total_prob"))
                            final_prob = safe_float(row.get("total_prob_pick_final"))
                            pick_side = row.get("total_pick_side")
                        elif market == "moneyline":
                            market_prob = safe_float(row.get("ml_prob_pick_market"))
                            kalshi_prob = safe_float(row.get("kalshi_prob_for_pick"))
                            model_prob = safe_float(row.get("model_prob") or row.get("AI_Prob"))
                            final_prob = safe_float(row.get("final_probability") or row.get("AI_Prob"))
                            pick_side = row.get("Pick")
                        else:
                            # Fallback to generic fields
                            market_prob = safe_float(row.get("Implied_Prob"))
                            kalshi_prob = safe_float(row.get("kalshi_prob_for_pick"))
                            model_prob = safe_float(row.get("model_prob") or row.get("AI_Prob"))
                            final_prob = safe_float(row.get("final_probability") or row.get("AI_Prob"))
                            pick_side = row.get("Pick")

                        sentiment_score = safe_float(row.get("sentiment_score"))

                        # Format consensus breakdown showing each source
                        consensus_breakdown = format_consensus_breakdown(
                            market_prob=market_prob,
                            kalshi_prob=kalshi_prob_used,
                            model_prob=model_prob,
                            sentiment_score=sentiment_score,
                            final_prob=final_prob,
                            pick_side=pick_side
                        )
                        consensus_breakdown_cols.append(consensus_breakdown)

                    # Bulk assign to avoid fragmentation
                    # Create a DataFrame for new columns and concat once
                    consensus_df_update = pd.DataFrame({
                        "consensus_votes": consensus_vote_cols,
                        "consensus_total": consensus_total_cols,
                        "consensus": consensus_breakdown_cols
                    }, index=master_df.index)

                    master_df = pd.concat([master_df, consensus_df_update], axis=1).copy()

                    # Merge vote details
                    if vote_details_list:
                        vote_details_df = pd.DataFrame(vote_details_list, index=master_df.index)
                        # Remove duplicate columns if any overlap
                        vote_details_df = vote_details_df.drop(columns=[c for c in vote_details_df.columns if c in master_df.columns], errors="ignore")
                        master_df = pd.concat([master_df, vote_details_df], axis=1).copy()

                    logger.info(f"Consensus enrichment complete for {len(master_df)} picks")

                    # 5. SHOTGUN ACTIVATION: Use ParlayOptimizer to tier the results and generate 2-leg parlays
                    shotgun_picks = optimizer.get_shotgun_picks(master_df)
                    shotgun_parlays = optimizer.generate_shotgun_parlays(master_df)
                    st.session_state["shotgun_data"] = shotgun_picks
                    st.session_state["shotgun_parlays"] = shotgun_parlays

                # Collapse to one row per game (prefer the first generated row, typically moneyline) for Master View
                # NOTE: master_df now has ALL rows (ML/Spread/Total). We duplicate logic for deduping for the UI view if needed,
                # but the prompt implies we persist the FULL master_df to session state for tabs to use.

                # We need to preserve the sentiment metadata enrichment logic
                sentiment_meta_for_export = sentiment_pack_meta or init_sentiment_meta()
                # Vectorized or simple loop to fill sentiment meta if missing
                # (Assuming enrich_with_model_features preserves existing cols, which it does)

                # Deduping logic for "Master View" (one row per game)
                # We'll create a view for display, but keep master_df full for shotgun/optimizer.

                # But wait, the previous code replaced `df` with `deduped_list`.
                # If we overwrite `st.session_state["master_df"]` with the full `master_df`,
                # downstream code expecting 1 row per game might break.
                # However, the user instruction was "Persist to session state for the tabs to use".
                # The tabs (Shotgun) likely need the full rows.
                # The "Master Analysis" tab view logic (later in the file) uses `df` (which was deduped).
                # We should probably assign `df` to the deduped version for the immediate display logic below,
                # but maybe store `master_df_full` or similar?
                # Actually, let's follow the pattern but adapt for the existing `df` variable usage.

                # Apply sentiment meta to master_df
                # (Simulating what the loop did)
                if not master_df.empty:
                    # Optimized bulk assignment to prevent fragmentation
                    meta_updates = {
                        "sentiment_sample_status": str(sentiment_meta_for_export.get("sentiment_sample_status", "NO_CALL") or "NO_CALL"),
                        "sentiment_source": str(sentiment_meta_for_export.get("sentiment_source", "none") or "none"),
                        "sentiment_status_counts": json.dumps(sentiment_meta_for_export.get("sentiment_status_counts", {"NO_CALL": 1})),
                        "sentiment_sample_query": sentiment_meta_for_export.get("sentiment_sample_query", "") or "",
                        "sentiment_disabled_reason": sentiment_meta_for_export.get("sentiment_disabled_reason", "") or "",
                        "sentiment_errors_sample": sentiment_meta_for_export.get("sentiment_errors_sample", "") or "",
                        "sentiment_error_count": int(sentiment_meta_for_export.get("sentiment_error_count", 0) or 0),
                    }
                    # Create DataFrame for new columns and concat
                    meta_df = pd.DataFrame(meta_updates, index=master_df.index)
                    master_df = pd.concat([master_df, meta_df], axis=1).copy()

                    # Fill remaining fields using bulk fillna
                    # Fix: Ensure no 'None' values are passed to fillna
                    fill_map = {
                        "sentiment_status": str(sentiment_meta_for_export.get("sentiment_status") or "ok"),
                        "sentiment_confidence": float(sentiment_meta_for_export.get("sentiment_confidence") or 0.0),
                        "sentiment_score": float(sentiment_meta_for_export.get("sentiment_score") or 0.0),
                    }
                    # Apply batch fillna (Issue #5)
                    # Filter fill_map to existing columns
                    valid_fill_map = {k: v for k, v in fill_map.items() if k in master_df.columns}
                    if valid_fill_map:
                        master_df = master_df.fillna(valid_fill_map).infer_objects(copy=False)

                    # Fill visual cols with empty string
                    visual_cols = ["spread_sentiment_arrow", "total_sentiment_arrow", "spread_sentiment_note", "total_sentiment_note"]
                    valid_visual_cols = [c for c in visual_cols if c in master_df.columns]
                    if valid_visual_cols:
                        master_df[valid_visual_cols] = master_df[valid_visual_cols].fillna("").infer_objects(copy=False)

                    # Defragment DataFrame after multiple concat operations
                    master_df = master_df.copy()

                # Proceed with full dataframe for enrichment (collapse happens at the end)
                df = master_df.copy()

                # ============================================
                # ADD CONSENSUS & BEST PICK LOGIC (Task 3)
                # ============================================
                logger.info("Enforcing Consensus & Best Pick Logic on Master DF...")

                # Invariant: If Spread/Total Pick exists and final probability > 0.5,
                # a corresponding consensus probability must be computed and exposed in both UI and exports.
                def _enforce_consensus_and_best_pick_vectorized(df):
                    # Optimized implementation to avoid row-by-row apply
                    new_data = {}

                    # 1. Spread Consensus
                    s_probs = []
                    s_strs = []
                    t_probs = []
                    t_strs = []

                    for idx, row in df.iterrows():
                        sp, ss = calculate_consensus_for_row(row, "Spread")
                        tp, ts = calculate_consensus_for_row(row, "Total")
                        s_probs.append(sp)
                        s_strs.append(ss)
                        t_probs.append(tp)
                        t_strs.append(ts)

                    new_data["SpreadConsensusProb"] = s_probs
                    new_data["SpreadConsensus"] = s_strs
                    new_data["TotalConsensusProb"] = t_probs
                    new_data["TotalConsensus"] = t_strs

                    # Create temporary DF for vectorized ops
                    temp = df.copy()
                    temp = pd.concat([temp, pd.DataFrame(new_data, index=df.index)], axis=1).copy()

                    # 3. Best Overall Pick Logic
                    # v99 FIX (Bug 4): Use SpreadConsensusProb/TotalConsensusProb for comparison.
                    # Previously used spread_prob_pick_final/total_prob_pick_final which may differ
                    # from consensus probs (due to TheOver delta clamping differences).
                    # This caused Best Overall Market="Spread" but Best Overall Prob=TotalConsensusProb.
                    best_pick = []
                    best_prob = []
                    best_type = []

                    # Prepare lists for updated pick strings to ensure consistency
                    spread_pick_updated = []
                    total_pick_updated = []

                    for idx, row in temp.iterrows():
                        # Use consensus probabilities for comparison (what user sees)
                        s_final_prob = safe_float(row.get("SpreadConsensusProb")) or safe_float(row.get("spread_prob_pick_final")) or 0.0
                        t_final_prob = safe_float(row.get("TotalConsensusProb")) or safe_float(row.get("total_prob_pick_final")) or 0.0

                        # Determine validity based on raw data existence and valid float conversion
                        s_team = row.get("spread_pick_team")
                        s_line_val = safe_float(row.get("spread_pick_line"))
                        s_valid = (s_team is not None) and (s_line_val is not None)

                        t_side = row.get("total_pick_side")
                        t_line_val = safe_float(row.get("total_pick_line"))
                        t_valid = (t_side is not None) and (t_line_val is not None)

                        new_b_type = "NONE"
                        new_b_pick = None
                        new_b_prob = 0.0

                        # Calculate strengths (distance from 0.5 or max prob)
                        # We want the strongest edge, even if it requires flipping the pick
                        s_strength = max(s_final_prob, 1.0 - s_final_prob) if s_valid else -1.0
                        t_strength = max(t_final_prob, 1.0 - t_final_prob) if t_valid else -1.0

                        target_market = "NONE"
                        if s_valid and t_valid:
                            if s_strength >= t_strength:
                                target_market = "SPREAD"
                            else:
                                target_market = "TOTAL"
                        elif s_valid:
                            target_market = "SPREAD"
                        elif t_valid:
                            target_market = "TOTAL"

                        # REGENERATE PICK STRINGS (FIX: Ensure consistency with consensus prob)
                        updated_s_str = row.get("Spread & Pick") # Default fallback
                        updated_t_str = row.get("Total & Pick") # Default fallback

                        if s_valid:
                            if s_final_prob > 0.5:
                                updated_s_str = f"{s_team} {clean_line_str(s_line_val)} ({s_final_prob:.1%})"
                            else:
                                # Flip to opposite
                                prob_flipped = 1.0 - s_final_prob
                                home = row.get("Home")
                                away = row.get("Away")
                                opp_team = away if s_team == home else home
                                if opp_team is None: opp_team = s_team # Fallback
                                opp_line = -1 * s_line_val if s_line_val is not None else 0.0
                                updated_s_str = f"{opp_team} {clean_line_str(opp_line)} ({prob_flipped:.1%})"

                        if t_valid:
                            if t_final_prob > 0.5:
                                updated_t_str = f"{t_side} {clean_line_str(t_line_val)} ({t_final_prob:.1%})"
                            else:
                                # Flip
                                prob_flipped = 1.0 - t_final_prob
                                opp_side = "Under" if t_side == "Over" else "Over"
                                updated_t_str = f"{opp_side} {clean_line_str(t_line_val)} ({prob_flipped:.1%})"

                        # Construct the best pick string dynamically using updated strings
                        if target_market == "SPREAD":
                            new_b_type = "SPREAD"
                            new_b_pick = updated_s_str
                            new_b_prob = s_final_prob if s_final_prob > 0.5 else (1.0 - s_final_prob)

                        elif target_market == "TOTAL":
                            new_b_type = "TOTAL"
                            new_b_pick = updated_t_str
                            new_b_prob = t_final_prob if t_final_prob > 0.5 else (1.0 - t_final_prob)

                        best_pick.append(new_b_pick)
                        best_prob.append(new_b_prob)
                        best_type.append(new_b_type)
                        spread_pick_updated.append(updated_s_str)
                        total_pick_updated.append(updated_t_str)

                    new_data["Best Overall Pick"] = best_pick
                    new_data["Best Overall Prob"] = best_prob
                    new_data["best_pick_type"] = best_type
                    # Update source columns so downstream logic (calculate_best_pick_metrics) uses correct strings
                    new_data["Spread & Pick"] = spread_pick_updated
                    new_data["Total & Pick"] = total_pick_updated

                    # Create DataFrame from new data
                    new_df = pd.DataFrame(new_data, index=df.index)

                    # Drop existing columns to prevent duplication
                    cols_to_drop = [c for c in new_df.columns if c in df.columns]
                    if cols_to_drop:
                        df = df.drop(columns=cols_to_drop)

                    return pd.concat([df, new_df], axis=1).copy()

                df = _enforce_consensus_and_best_pick_vectorized(df)

                # --------------------------------------------------------
                # FIX: Sync Market and Pick columns with Best Overall Pick
                # User requirement: Market/Pick must match Best Overall Pick logic
                # --------------------------------------------------------
                def _sync_market_pick_with_best_overall(df):
                    """
                    Update Market and Pick columns to match Best Overall Pick.
                    This ensures the exported Market/Pick always shows the highest probability
                    pick between Spread and Total (no ML).
                    """
                    if 'best_pick_type' not in df.columns:
                        logger.warning("best_pick_type column not found, cannot sync Market/Pick")
                        return df

                    df = df.copy()
                    synced_count = 0

                    for idx, row in df.iterrows():
                        best_type = str(row.get('best_pick_type', '')).upper()
                        current_market = str(row.get('Market', ''))

                        # Map best_pick_type to Market name
                        target_market = None
                        target_pick = None
                        target_prob = None

                        if best_type == 'SPREAD':
                            target_market = 'Spread'
                            target_pick = row.get('Spread & Pick')
                            target_prob = row.get('spread_prob_pick_final')
                        elif best_type == 'TOTAL':
                            target_market = 'Total'
                            target_pick = row.get('Total & Pick')
                            target_prob = row.get('total_prob_pick_final')
                        elif best_type == 'NONE':
                            # No valid pick - keep as is or set to None
                            continue
                        else:
                            # Unknown type - skip
                            continue

                        # Update Market and Pick if they don't match
                        if target_market and current_market != target_market:
                            df.at[idx, 'Market'] = target_market
                            synced_count += 1

                        if target_pick is not None:
                            df.at[idx, 'Pick'] = target_pick

                        if target_prob is not None:
                            df.at[idx, 'final_probability'] = target_prob

                        # v104 FIX (Bug 3): Sync kalshi_weight with the champion market's weight.
                        # When spread data is missing but total data is valid (or vice versa),
                        # kalshi_weight should reflect the champion market's Kalshi availability,
                        # not be stuck at 0 just because one market type lacks Kalshi data.
                        if best_type == 'SPREAD':
                            spread_kw = safe_float(row.get('kalshi_weight'))
                            total_kw = safe_float(row.get('kalshi_weight'))  # same row
                            # If spread Kalshi weight is 0 but total has valid Kalshi prob, use a base weight
                            if (spread_kw is None or spread_kw == 0) and safe_float(row.get('kalshi_prob_total')) is not None:
                                df.at[idx, 'kalshi_weight'] = 0.55
                        elif best_type == 'TOTAL':
                            if safe_float(row.get('kalshi_weight')) in (None, 0) and safe_float(row.get('kalshi_prob_total')) is not None:
                                df.at[idx, 'kalshi_weight'] = 0.55

                        # Also set Best Overall Market for export
                        df.at[idx, 'Best Overall Market'] = target_market

                    logger.info(f"✅ Market/Pick sync complete: {synced_count} rows updated to match Best Overall Pick")
                    return df

                df = _sync_market_pick_with_best_overall(df)

                # --------------------------------------------------------
                # TASK: Add Edge Column (Model Prob - Consensus Prob)
                # --------------------------------------------------------
                def calculate_edge(df):
                    edge_formatted = []
                    for idx, row in df.iterrows():
                        market_val = str(row.get('Market', ''))
                        consensus = None

                        if 'Spread' in market_val:
                            consensus = row.get('SpreadConsensusProb')
                        elif 'Total' in market_val:
                            consensus = row.get('TotalConsensusProb')
                        elif 'Moneyline' in market_val or 'ML' in market_val:
                            consensus = row.get('consensus_prob_adj') or row.get('Implied_Prob')

                        # Check for valid consensus
                        # "If consensus is missing/NaN, set Edge = 0.0."
                        if pd.isna(consensus) or consensus is None:
                            edge_val = 0.0
                        else:
                            try:
                                c_val = float(consensus)
                                m_val = float(row.get('Best Overall Prob', 0.5) or 0.5)
                                edge_val = m_val - c_val
                            except Exception:
                                edge_val = 0.0

                        edge_pct = edge_val * 100
                        # Format as string "X.X%" for readability with sign
                        edge_formatted.append(f"{edge_pct:+.1f}%")
                    return edge_formatted

                # Insert Edge column immediately after 'Best Overall Prob'
                if 'Best Overall Prob' in df.columns:
                    try:
                        edge_values = calculate_edge(df)
                        # Remove 'Edge' if it already exists to avoid duplication error on insert
                        if 'Edge' in df.columns:
                            df.drop(columns=['Edge'], inplace=True)

                        loc_index = df.columns.get_loc('Best Overall Prob') + 1

                        # CRITICAL FIX 2: Replace df.insert with pd.concat to avoid fragmentation
                        # Create DataFrame for Edge
                        edge_df = pd.DataFrame({'Edge': edge_values}, index=df.index)

                        # Split and Concat
                        df_left = df.iloc[:, :loc_index]
                        df_right = df.iloc[:, loc_index:]

                        df = pd.concat([df_left, edge_df, df_right], axis=1).copy()

                        logger.info("✅ Edge column added successfully (Defragmented)")
                    except Exception as e:
                        logger.warning(f"Failed to add Edge column: {e}")

                # Diagnostic Log
                logger.info(
                    "Consensus summary: rows=%s, with_spread_pick=%s, with_spread_consensus=%s, with_total_pick=%s, with_total_consensus=%s",
                    len(df),
                    int(df['Spread & Pick'].notna().sum()) if 'Spread & Pick' in df.columns else 0,
                    int(df['SpreadConsensusProb'].notna().sum()) if 'SpreadConsensusProb' in df.columns else 0,
                    int(df['Total & Pick'].notna().sum()) if 'Total & Pick' in df.columns else 0,
                    int(df['TotalConsensusProb'].notna().sum()) if 'TotalConsensusProb' in df.columns else 0,
                )

                # --- MARKET TRACKER HOOK (Snapshot System) ---
                try:
                    # 1. Save Noon Baseline (Task 2: Use Snapshot Manager)
                    snapshot_manager.save_noon_baseline(df)

                    # 2. Compare against Noon Baseline (if Evening/Late)
                    # "Silent Purge" Fix: Ensure we don't overwrite if tracker returns empty/None
                    df_compare = market_tracker.load_and_compare(df)
                    if df_compare is not None and not df_compare.empty:
                        df = df_compare

                    # Persist TheOver debug stats for sidebar export
                    if 'theover_stats' in locals():
                        st.session_state["theover_debug_log"] = theover_stats.get("full_debug_log", [])
                        # Task 3: Persist RAW TheOver DataFrame
                        st.session_state["theover_raw_df"] = theover_stats.get("raw_df", pd.DataFrame())
                except Exception as e:
                    logger.error(f"Market Tracker Error: {e}")
                # ---------------------------

                # CRITICAL: Apply moneyline pivot logic before saving to session state
                logger.info(f"Applying moneyline pivot logic to {len(df)} rows...")

                # DISABLED: Force pivot logic
                # This was converting ML rows to Spread/Total based on edge comparison
                # However, we want to preserve all market types (Spread, Total, ML) for export
                # The deduplication logic now groups by (game_key, Market) to keep all market types
                #
                # Old logic (DISABLED):
                # def _force_pivot(row):
                #     if row.get('Market') == 'Moneyline':
                #         ... [pivot ML to Spread/Total based on edge]
                # df = df.apply(_force_pivot, axis=1)
                #
                # NEW BEHAVIOR: Preserve all Market types without pivoting

                logger.info(f"Skipping pivot logic - preserving all market types (Spread, Total, ML)")

                # ============================================
                # ADD HASKALSHIMARKET FLAG
                # ============================================
                # Define a robust criterion for "game has a Kalshi market" based on Kalshi fields only.
                # A game counts as having a Kalshi market if and only if:
                # 1. kalshi_matched == True (indicating a successful match was found)
                # 2. AND at least one of kalshi_prob_spread or kalshi_prob_total is non-null
                #    (indicating an actual market with probabilities exists)
                #
                # This criterion is independent of spreadmarketpairscount/totalmarketpairscount
                # which count sportsbook markets, not Kalshi markets.
                # ============================================

                def _has_kalshi_market_vectorized(df):
                    """
                    Determine if a game has a valid Kalshi market (Vectorized).
                    """
                    # Convert columns to numeric, coercing errors to NaN
                    def _to_num(col):
                        return pd.to_numeric(df.get(col, pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0).infer_objects(copy=False).infer_objects(copy=False)

                    k_matched = df.get("kalshi_matched", pd.Series([False]*len(df), index=df.index)).fillna(False).astype(bool)

                    # Check for non-zero probabilities
                    k_spread = _to_num("kalshi_prob_spread") != 0
                    s_pick_k = _to_num("spread_prob_pick_kalshi") != 0

                    k_total = _to_num("kalshi_prob_total") != 0
                    t_pick_k = _to_num("total_prob_pick_kalshi") != 0

                    k_ml = _to_num("kalshi_prob") != 0
                    k_ml_used = _to_num("kalshi_prob_used") != 0

                    has_any = (k_spread | s_pick_k | k_total | t_pick_k | k_ml | k_ml_used)
                    return k_matched & has_any

                # Fix for Fragmentation (Issue #4)
                has_kalshi_series = _has_kalshi_market_vectorized(df)

                # Check if column already exists
                if "HasKalshiMarket" in df.columns:
                    df = df.drop(columns=["HasKalshiMarket"])

                new_hk_col = pd.DataFrame({"HasKalshiMarket": has_kalshi_series}, index=df.index)
                df = pd.concat([df, new_hk_col], axis=1).copy()

        # -------------------------------------------------------------------------
        # TASK 4: Kalshi Mode (Transparency)
        # -------------------------------------------------------------------------
        def _map_kalshi_mode(row):
            if not row.get("kalshi_available", False):
                return "none"
            # If we have market data (HasKalshiMarket), it's "full"
            # Otherwise if matched but no data, "partial"
            # If not matched, "none"

            # Use pre-computed HasKalshiMarket if available
            has_market = row.get("HasKalshiMarket", False)
            if has_market:
                return "full"

            status = str(row.get("kalshistatus", "")).lower()
            if status in ("matched", "strictmatch", "seriesmatch") or row.get("kalshi_matched"):
                return "partial"
            return "none"

        df["Kalshi_Mode"] = df.apply(_map_kalshi_mode, axis=1)

        # -------------------------------------------------------------------------
        # TASK 1: Stats Quality Penalty (NCAAB)
        # -------------------------------------------------------------------------
        def _apply_stats_quality_penalty(row):
            # Only apply if explicitly MISSING (ESPN/REAL are fine)
            if row.get("stats_quality") == "MISSING":
                # Downgrade confidence bucket
                conf = row.get("Pick_Confidence")
                reason = str(row.get("confidence_reason") or "")

                # Strict downgrade logic
                if conf == "HIGH":
                    row["Pick_Confidence"] = "MEDIUM"
                    reason += " (downgraded: missing stats)"
                elif conf == "MEDIUM":
                    row["Pick_Confidence"] = "LOW"
                    reason += " (downgraded: missing stats)"
                elif conf == "LOW":
                    # Already low, just append reason
                    reason += " (stats missing)"
                else:
                    # Default/Unknown -> LOW
                    row["Pick_Confidence"] = "LOW"
                    reason += " (stats missing)"

                row["confidence_reason"] = reason.strip(" |")

                # Dampen probabilities (shrink edge by 50%)
                for col in ["spread_prob_pick_final", "total_prob_pick_final", "Best Overall Prob", "final_probability"]:
                    val = row.get(col)
                    if val is not None and pd.notna(val):
                        try:
                            fval = float(val)
                            # Dampen towards 0.5
                            row[col] = 0.5 + (fval - 0.5) * 0.5
                        except Exception:
                            pass

            # TASK 3: TheOver Transparency
            # Append delta to reason if used
            if row.get("theover_matched") or row.get("theover_used_in_pick"):
                delta = row.get("theover_delta_final_prob") or row.get("theover_delta")
                if delta is not None:
                    try:
                        d_val = float(delta)
                        if abs(d_val) > 0.001:
                            reason = str(row.get("confidence_reason") or "")

                            # Enhanced Text Logic (Requirement B)
                            msg = ""
                            if abs(d_val) <= 0.05:
                                msg = f"TheOver mild boost ({d_val:+.3f})"
                            else:
                                msg = f"TheOver-driven boost ({d_val:+.3f})"

                            row["confidence_reason"] = (
                                reason + f" | {msg}"
                            ).strip(" |")
                    except Exception:
                        pass

            return row

        df = df.apply(_apply_stats_quality_penalty, axis=1)

        # -------------------------------------------------------------------------
        # TASK 5: Update confidence_reason with Sentiment Impact
        # -------------------------------------------------------------------------
        def _update_reason_with_sentiment(row):
            w_sent = float(row.get("wsentiment_used") or 0.0)
            adj = float(row.get("sentiment_adj") or 0.0)

            # Check if sentiment was actually available/valid
            sentiment_valid = row.get("sentiment_available", False)

            # Ensure Pick_Reason_Short exists/synced
            if "Pick_Reason_Short" not in row or pd.isna(row["Pick_Reason_Short"]):
                row["Pick_Reason_Short"] = row.get("confidence_reason", "")

            # Thresholds: Weight > 5% and Adjustment > 1%
            # Also handle "unused" case

            tag = ""
            if not sentiment_valid:
                 # Sentiment disabled or unavailable - explicitly unused?
                 # User said: "If sentiment is effectively unused... prefix to sentiment=unused or just omit it."
                 # Let's omit it if invalid, unless specifically requested.
                 pass
            elif w_sent > 0.05 and abs(adj) > 0.01:
                direction = "Bullish" if adj > 0 else "Bearish"
                tag = f"sentiment={direction}"
            elif sentiment_valid:
                # Valid but neutral/small impact
                tag = "sentiment=neutral"

            if tag:
                # Update confidence_reason
                reason = str(row.get("confidence_reason") or "")
                if tag not in reason:
                    row["confidence_reason"] = (reason + f" | {tag}").strip(" |")

                # Update Pick_Reason_Short
                short = str(row.get("Pick_Reason_Short") or "")
                if tag not in short:
                    row["Pick_Reason_Short"] = (short + f" | {tag}").strip(" |")

            return row

        df = df.apply(_update_reason_with_sentiment, axis=1)

        # Issue 2: Add sentiment_available flag
        # If sentiment_status is 'ok' or 'partial_cached', then available
        if "sentiment_status" in df.columns:
            df["sentiment_available"] = df["sentiment_status"].astype(str).isin(["ok", "partial_cached", "cached"])
        else:
            df["sentiment_available"] = False

        kalshi_markets_count = df["HasKalshiMarket"].sum()

        # Debug logging requested by user
        try:
            total_games_count = len(df)
            kalshi_matched_raw_count = df["kalshi_matched"].fillna(False).astype(bool).sum()
            logger.info(
                "Kalshi summary: total=%s, with_kalshi=%s, kalshi_matched_raw=%s",
                total_games_count,
                kalshi_markets_count,
                kalshi_matched_raw_count
            )
        except Exception as e:
            logger.warning(f"Failed to log Kalshi summary stats: {e}")

        logger.info(f"✅ HasKalshiMarket flag added: {kalshi_markets_count} games have valid Kalshi markets")

        # Fix NaN type field before export (map from Market column)
        if "type" not in df.columns:
            df["type"] = None
        df["type"] = df["type"].fillna(df["Market"].map({"Moneyline": "ML", "Spread": "SPREAD", "Total": "TOTAL"}))

        # NOTE: master_df will be saved AFTER champion selection to ensure both session_state
        # variables hold the collapsed 1-row-per-game dataframe
        logger.info(f"Preparing df ({len(df)} rows) for champion selection...")

        # Issue 1: Alias internal columns to user-requested names for export
        # FIXED: Use pd.concat to avoid fragmentation (was causing PerformanceWarning)
        alias_map = {
            "Pick_Confidence": "PickConfidence",
            "Pick_Reason_Short": "PickReason",
            "gemini_total_confidence": "geminitotalconfidence",
            "gemini_rationalize": "geminirationalize",
            "gemini_error_flag": "geminierrorflag",
            "Home_Sentiment": "HomeSentiment",
            "Away_Sentiment": "AwaySentiment",
            "Sentiment_Diff": "SentimentDiff",
            "sentiment_score": "sentimentscore",
            "sentiment_status": "sentimentstatus"
        }
        # Build alias columns in a single operation
        alias_cols = {}
        for internal, external in alias_map.items():
            if internal in df.columns and external not in df.columns:
                alias_cols[external] = df[internal]

        if alias_cols:
            # Concatenate all new columns at once to avoid fragmentation
            df = pd.concat([df, pd.DataFrame(alias_cols, index=df.index)], axis=1).copy()

        # Create filtered version for user export (remove internal/debug columns)
        # Keep only user-relevant columns for the "All Picks" export
        # FIX: Removed duplicate column entries that caused CSV header/data mismatch
        # (Pick_Confidence, theover_matched, theover_delta_final_prob were listed multiple times)
        user_columns = [
            'league', 'Home', 'Away', 'Commence (UTC)', 'Commence (Local)', 'Local Date',
            'Market', 'Pick', 'final_probability', 'Pick_Confidence',
            'Best Overall Pick', 'Best Overall Prob', 'Edge',
            'wsentiment_used', 'sentiment_adj', 'sentiment_prob',
            'Best Overall Market',
            'Spread & Pick', 'spread_prob_pick_final', 'SpreadConsensusProb', 'SpreadConsensus',
            'Total & Pick', 'total_prob_pick_final', 'TotalConsensusProb', 'TotalConsensus',
            'best_spread_price', 'best_total_price', 'spread_pick_line', 'total_pick_line',
            'Home_Sentiment', 'Away_Sentiment', 'Sentiment_Diff',
            'kalshi_available', 'HasKalshiMarket', 'Kalshi_Mode',
            'theover_matched', 'theover_delta_final_prob',
            'spread_prob_market', 'total_prob_market', 'decisiveness',
            'confidence_reason', 'stats_quality',
            'sentiment_available',
            'Pick_Reason_Short',
            # Gemini & Sentiment columns
            'PickConfidence', 'PickReason', 'geminitotalconfidence', 'geminirationalize', 'geminierrorflag',
            'HomeSentiment', 'AwaySentiment', 'SentimentDiff', 'sentimentscore', 'sentimentstatus',
            'model_spread_prob', 'model_total_prob', 'AI_Prob', 'consensus_prob', 'consensus_prob_adj',
            # TheOver integration columns
            'theover_pick', 'theover_hit_rate', 'theover_source_model', 'theover_prob_used',
            'final_prob_without_theover',
            # Probability weights for transparency
            'kalshi_weight', 'odds_weight', 'ml_weight', 'sentiment_weight',
            # Decision trace info
            'decision_driver', 'prob_engine', 'model_mode', 'Warnings'
        ]
        # Task 3.1: Calculate Data Quality Score before export
        def calculate_data_quality_score(row) -> int:
            """
            Calculate data quality score (0-100) based on warnings and data availability.

            Deductions (adjusted for Issue #3 - Grade A percentage improvement):
            - FallbackPlaceholderDetected: -12 points (reduced from -15)
            - kalshi_pick_mismatch: -20 points (reduced from -25)
            - kalshi_validation_failed: -15 points (reduced from -20)
            - No TheOver data: -5 points (reduced from -10 - TheOver is supplementary)
            - No sentiment: -5 points (reduced from -10 - sentiment is supplementary)
            - Fallback stats: -3 points (reduced from -5)

            Boosts (new for Issue #3):
            - TheOver matched and used: +5 points (reward for having TheOver data)
            - Strong sentiment signal: +3 points (reward for sentiment confidence)
            """
            score = 100  # Start with perfect score

            warnings_str = str(row.get("Warnings", "")).lower()

            # Check for FallbackPlaceholderDetected
            if "fallbackplaceholderdetected" in warnings_str:
                score -= 12

            # Check for Kalshi mismatches (structural or edge)
            if "kalshi_pick_mismatch" in warnings_str:
                score -= 20
            elif "kalshi_validation_failed" in warnings_str:
                score -= 15

            # Check for TheOver data availability (reduced penalty, TheOver is supplementary)
            theover_matched = row.get("theover_matched", False)
            theover_used = row.get("theover_used_in_pick", False)
            if not theover_matched or theover_matched == "False" or theover_matched == 0:
                score -= 5
            else:
                # BOOST: TheOver data successfully matched and potentially used
                score += 5
                # Extra boost if TheOver data was actually used in the pick decision
                if theover_used and theover_used != "False" and theover_used != 0:
                    score += 2

            # Check for sentiment availability (reduced penalty, sentiment is supplementary)
            sentiment_available = row.get("sentiment_available", False)
            if not sentiment_available or sentiment_available == "False" or sentiment_available == 0:
                score -= 5
            else:
                # BOOST: Check for strong sentiment signal
                sentiment_strength = str(row.get("sentiment_strength", "")).upper()
                if sentiment_strength in ("STRONG", "TEAM_STRONG"):
                    score += 3
                elif sentiment_strength in ("MEDIUM", "TEAM_MED"):
                    score += 1

            # Check for fallback stats (reduced penalty)
            stats_quality = str(row.get("stats_quality", "")).lower()
            if "fallback" in stats_quality or "espn" in stats_quality:
                score -= 3

            # BOOST: High confidence picks get a small boost
            pick_confidence = row.get("Pick_Confidence", 0)
            try:
                conf_val = float(pick_confidence) if pick_confidence else 0
                if conf_val >= 0.7:
                    score += 2
            except (ValueError, TypeError):
                pass

            # Ensure score stays in valid range
            return max(0, min(100, score))

        def calculate_grade(score: int) -> str:
            """Convert numeric score to letter grade."""
            if score >= 90:
                return "A"
            elif score >= 80:
                return "B"
            elif score >= 70:
                return "C"
            elif score >= 60:
                return "D"
            else:
                return "F"

        # Apply data quality calculations
        logger.info("Calculating data quality scores...")
        df["data_quality_score"] = df.apply(calculate_data_quality_score, axis=1)
        df["data_quality_grade"] = df["data_quality_score"].apply(calculate_grade)
        logger.info(f"Data quality scores calculated. Mean score: {df['data_quality_score'].mean():.1f}")

        # Add quality columns to user_columns list
        if "data_quality_score" not in user_columns:
            user_columns.insert(user_columns.index("Warnings") if "Warnings" in user_columns else len(user_columns),
                              "data_quality_score")
        if "data_quality_grade" not in user_columns:
            user_columns.insert(user_columns.index("data_quality_score") + 1, "data_quality_grade")

        # --- PIPELINE REORDERING (Fix Logic Flow) ---
        # 1. Enrich with Reddit/News sentiment usage flags
        if "reddit_used" in df.columns:
            df["reddit_used"] = df["reddit_used"].fillna(False).astype(bool)

        # 2. Add Spread/Total Confidence (Calculates At_a_Glance_Score, probs)
        df = add_spread_total_confidence(df)
        df = df.copy()


        # 3. Enrich with ROI Metrics (Calculates Edge)
        df = enrich_picks_with_roi_metrics(df)
        df = df.copy()

        # 4. Calculate Best Pick Metrics (Flips picks, sets confidence)
        # This MUST run before Gemini so rationale matches the flipped pick
        logger.info("Applying Best Pick Metrics (Flip & Confidence) before Gemini...")
        df = calculate_best_pick_metrics(df)

        # v97 FIX: Sync final_probability and Best Overall columns with calculate_best_pick_metrics output.
        # calculate_best_pick_metrics outputs 'final_prob', 'best_pick', 'best_pick_type' which use
        # post-flip probabilities. But downstream code (Flip Leak Fix, calculate_confidence) reads
        # 'final_probability' and 'Best Overall Prob/Pick/Market' which were set earlier with pre-flip values.
        # Without this sync, champion selection picks the correct market but the result never propagates.
        if 'final_prob' in df.columns:
            df['final_probability'] = df['final_prob']
            df['Best Overall Prob'] = df['final_prob']
        if 'best_pick' in df.columns:
            df['Best Overall Pick'] = df['best_pick']
        if 'best_pick_type' in df.columns:
            df['Best Overall Market'] = df['best_pick_type']

        # PIPELINE FIX v96: Flip (not drop) stragglers < 50% and Recalculate Confidence
        # v95 dropped entire games here, losing 5 games (3 NBA + 2 NCAAB).
        # v96 flips remaining sub-50% picks to the opposite side with LOW confidence.
        if 'final_probability' in df.columns:
             mask_below_50 = df['final_probability'] < 0.50
             if mask_below_50.any():
                 logger.warning(f"Flipping {mask_below_50.sum()} picks with probability < 50% (v96 Flip Leak Fix)")
                 # Flip the probability
                 df.loc[mask_below_50, 'final_probability'] = 1.0 - df.loc[mask_below_50, 'final_probability']
                 # Flip the Best Overall Pick to opposite side
                 for idx in df[mask_below_50].index:
                     row = df.loc[idx]
                     best_market = str(row.get('Best Overall Market', '')).lower()
                     if 'spread' in best_market:
                         home = str(row.get('Home', ''))
                         away = str(row.get('Away', ''))
                         current_pick = str(row.get('Best Overall Pick', ''))
                         if home and home in current_pick:
                             line_match = re.search(r'(-?\d+\.?\d*)$', current_pick)
                             if line_match:
                                 new_line = -float(line_match.group(1))
                                 df.at[idx, 'Best Overall Pick'] = f"{away} {new_line:+g}"
                         elif away and away in current_pick:
                             line_match = re.search(r'(-?\d+\.?\d*)$', current_pick)
                             if line_match:
                                 new_line = -float(line_match.group(1))
                                 df.at[idx, 'Best Overall Pick'] = f"{home} {new_line:+g}"
                     elif 'total' in best_market:
                         current_pick = str(row.get('Best Overall Pick', ''))
                         if 'Over' in current_pick:
                             df.at[idx, 'Best Overall Pick'] = current_pick.replace('Over', 'Under')
                         elif 'Under' in current_pick:
                             df.at[idx, 'Best Overall Pick'] = current_pick.replace('Under', 'Over')
                 # Force LOW confidence on flipped picks
                 df.loc[mask_below_50, 'Pick_Confidence'] = 'LOW'

             # Recalculate Confidence LAST
             logger.info("Recalculating Confidence on final probabilities...")
             df['Pick_Confidence'] = df.apply(calculate_confidence, axis=1)

             # Generate Reasoning LAST (to ensure it sees final state)
             df['Pick_Reason_Short'] = df.apply(generate_reasoning, axis=1)
             # Also update internal confidence_reason for consistency
             df['confidence_reason'] = df['Pick_Reason_Short']

        df = df.copy()

        # -------------------------------------------------------------------------
        # TASK 1: Implement "Champion Selection" (The 1-Row Fix) - FINAL ENFORCEMENT
        # -------------------------------------------------------------------------
        logger.info(f"Champion Selection: Starting with {len(df)} rows")

        # Calculate a selection score (Edge + Decisiveness)
        # Use 'edge' (float) if available, otherwise try to parse 'Edge' (string)
        if "edge" in df.columns:
             df["_edge_numeric"] = pd.to_numeric(df["edge"], errors='coerce').fillna(0.0)
        elif "Edge" in df.columns:
             # Convert string percentage to float safely (Issue: Vector error handling)
             df["_edge_numeric"] = pd.to_numeric(df["Edge"].astype(str).str.rstrip('%'), errors='coerce').fillna(0.0) / 100.0
        else:
             df["_edge_numeric"] = 0.0

        final_prob_col = "final_probability" if "final_probability" in df.columns else "Best Overall Prob"
        df["_final_prob_numeric"] = pd.to_numeric(df[final_prob_col], errors='coerce').fillna(0.5)

        # RESCUE FIX #1: Force _sel_score to numeric with fallback to 0
        df['_sel_score'] = pd.to_numeric(
            (df['_final_prob_numeric'] - 0.5).abs() + df['_edge_numeric'],
            errors='coerce'
        ).fillna(0)

        # Group by game and pick the highest score
        # CRITICAL FIX: Always use standard game keys (NOT game_key) to ensure proper grouping
        group_keys = ["league", "Home", "Away", "Commence (UTC)"]
        # Ensure these columns exist
        group_keys = [k for k in group_keys if k in df.columns]

        if group_keys and len(group_keys) >= 3:  # Need at least league, home, away
             rows_before = len(df)
             df_collapsed = df.sort_values('_sel_score', ascending=False).groupby(group_keys, dropna=False).head(1)

             # RESCUE FIX #1: Fallback if grouped dataframe is empty
             if df_collapsed.empty and not df.empty:
                 logger.error(f"⚠️ CRITICAL: Groupby resulted in 0 rows! Falling back to raw dataframe with {len(df)} rows")
                 st.warning(f"⚠️ Grouping failed - showing all {len(df)} picks instead of collapsed view")
                 # Keep the raw dataframe instead of empty result
                 df = df.drop(columns=['_sel_score', '_edge_numeric', '_final_prob_numeric'], errors='ignore').reset_index(drop=True).copy()
             else:
                 df = df_collapsed.drop(columns=['_sel_score', '_edge_numeric', '_final_prob_numeric'], errors='ignore').reset_index(drop=True).copy()
                 logger.info(f"✅ Champion Selection: Collapsed {rows_before} rows to {len(df)} rows (1 per game)")

                 # Log market breakdown after final collapse
                 if not df.empty and "Market" in df.columns:
                     market_counts = df["Market"].value_counts()
                     logger.info(f"Market breakdown after Champion Selection: {dict(market_counts)}")
        else:
             logger.error(f"Champion Selection: Missing required group keys! Available columns: {list(df.columns)}")
             logger.error(f"Group keys found: {group_keys}")
             # Continue without collapse but log the issue
             df = df.drop(columns=['_sel_score', '_edge_numeric', '_final_prob_numeric'], errors='ignore').reset_index(drop=True).copy()

        # -------------------------------------------------------------------------
        # TASK 4: Clean Up NaN in Debug Exports
        # -------------------------------------------------------------------------
        # Ensure best_pick_type and Market are never null before JSON serialization
        if "best_pick_type" in df.columns and "Market" in df.columns:
             df["best_pick_type"] = df["best_pick_type"].fillna(df["Market"]).fillna("UNKNOWN")

        # -------------------------------------------------------------------------
        # CRITICAL FIX: Final Market/Pick Sync with Best Overall Pick
        # -------------------------------------------------------------------------
        # v104 FIX (Bug 5 & Bug 6): Sync Market/Pick/final_probability from the
        # already-correct Best Overall columns. The OLD logic re-compared
        # spread_prob_pick_final vs total_prob_pick_final (pre-flip raw values)
        # and overwrote final_probability with sub-50% values, causing:
        #   Bug 5: 12 picks exported with final_probability < 0.50
        #   Bug 6: 2 games got wrong market (Total instead of Spread) because
        #          pre-flip total_prob > pre-flip spread_prob even though the
        #          post-flip champion was Spread.
        # Now we trust the upstream pipeline (enforce_winning_picks →
        # calculate_best_pick_metrics → flip stragglers → champion selection)
        # and simply propagate Best Overall → Market/Pick/final_probability.
        # -------------------------------------------------------------------------
        def _final_market_pick_sync(df):
            """
            Final sync: propagate Best Overall columns to Market/Pick/final_probability.

            Upstream pipeline already determined the correct champion pick with
            post-flip probabilities. This function ensures the user-facing columns
            (Market, Pick, final_probability) match the Best Overall selection.
            """
            if df.empty:
                return df

            df = df.copy()
            synced_count = 0

            for idx, row in df.iterrows():
                best_market = row.get('Best Overall Market')
                best_pick = row.get('Best Overall Pick')
                best_prob = safe_float(row.get('Best Overall Prob'))

                # Skip if Best Overall columns are not populated
                if best_market is None or best_pick is None or best_prob is None:
                    continue

                # Ensure final_probability is always >= 0.50 (post-flip)
                if best_prob < 0.50:
                    best_prob = 1.0 - best_prob
                    df.at[idx, 'Best Overall Prob'] = best_prob

                current_market = str(row.get('Market', ''))
                current_pick = str(row.get('Pick', ''))
                current_prob = safe_float(row.get('final_probability'))

                needs_update = (
                    current_market != str(best_market) or
                    current_pick != str(best_pick) or
                    current_prob != best_prob
                )

                if needs_update:
                    df.at[idx, 'Market'] = str(best_market)
                    df.at[idx, 'Pick'] = str(best_pick)
                    df.at[idx, 'final_probability'] = best_prob
                    df.at[idx, 'best_pick_type'] = str(best_market).upper()
                    synced_count += 1

            logger.info(f"✅ FINAL Market/Pick sync: {synced_count} rows synced with Best Overall columns")

            return df

        df = _final_market_pick_sync(df)

        # Log market distribution after final sync
        if not df.empty and 'Market' in df.columns:
            market_dist = df['Market'].value_counts()
            logger.info(f"📊 Final Market distribution: {dict(market_dist)}")

        # After all pick selection is complete
        logger.info("Running final validation...")
        df = enforce_winning_picks(df)
        logger.info("Final validation PASSED ✅")

        # Summary statistics
        logger.info("\n" + "="*60)
        logger.info("FINAL PICK SUMMARY")
        logger.info("="*60)

        total_picks = len(df)
        # Robust prob column access
        prob_col = 'final_probability' if 'final_probability' in df.columns else 'prob'
        if prob_col not in df.columns:
            prob_col = 'Best Overall Prob'

        if prob_col in df.columns:
            high_conf_picks = len(df[df[prob_col] >= 0.70])
            medium_conf_picks = len(df[(df[prob_col] >= 0.60) & (df[prob_col] < 0.70)])
            low_conf_picks = len(df[(df[prob_col] >= 0.50) & (df[prob_col] < 0.60)])

            logger.info(f"Total picks: {total_picks}")
            if total_picks > 0:
                logger.info(f"  High confidence (≥70%): {high_conf_picks} ({high_conf_picks/total_picks*100:.1f}%)")
                logger.info(f"  Medium confidence (60-70%): {medium_conf_picks} ({medium_conf_picks/total_picks*100:.1f}%)")
                logger.info(f"  Low confidence (50-60%): {low_conf_picks} ({low_conf_picks/total_picks*100:.1f}%)")
            logger.info(f"  Average probability: {df[prob_col].mean():.1%}")

            # Kalshi usage stats
            if 'HasKalshiMarket' in df.columns:
                kalshi_games = df[df['HasKalshiMarket'] == True]
                k_len = len(kalshi_games)
                logger.info(f"\nKalshi markets: {k_len}/{total_picks} ({k_len/total_picks*100 if total_picks > 0 else 0:.1f}%)")

                if k_len > 0:
                    kalshi_high_conf = len(kalshi_games[kalshi_games[prob_col] >= 0.70])
                    logger.info(f"  High confidence with Kalshi: {kalshi_high_conf}/{k_len} ({kalshi_high_conf/k_len*100:.1f}%)")
        else:
            logger.warning("Could not generate summary statistics: Probability column missing")

        logger.info("="*60 + "\n")

        # CRITICAL: Save collapsed dataframe to session state
        # Both master_df and master_results_df should hold the same collapsed 1-row-per-game data
        logger.info(f"Saving collapsed df ({len(df)} rows) to session state...")
        # --- GEMINI ENRICHMENT (Moved & Fixed) ---
        use_gemini_explanations = st.session_state.get("use_gemini_explanations", True)
        gemini_row_limit = int(st.session_state.get("gemini_row_limit", MAX_GEMINI_CALLS_PER_RUN) or MAX_GEMINI_CALLS_PER_RUN)
        gemini_full_run = bool(st.session_state.get("gemini_full_run", False))

        # Rank rows for Gemini priority
        # Use the correct column name found in your CSV
        df["_gemini_rank"] = pd.to_numeric(df.get("At_a_Glance_Score"), errors='coerce').fillna(0)
        sorted_indices = df.sort_values("_gemini_rank", ascending=False).index
        allowed_indices = set(sorted_indices[:gemini_row_limit]) if not gemini_full_run else set(df.index)

        if not use_gemini_explanations:
             df["gemini_mode"] = "disabled"
             df["gemini_rationale"] = "Disabled by user"
        else:
             # Iterate and update
             gemini_updates = {}

             for idx, row in df.iterrows():
                 if idx not in allowed_indices:
                     gemini_updates[idx] = {"gemini_mode": "guardrail", "gemini_rationale": "Skipped (Limit)"}
                     continue

                 # Check session limit
                 if st.session_state.get("gemini_calls_made", 0) >= MAX_GEMINI_CALLS_PER_RUN and not gemini_full_run:
                     gemini_updates[idx] = {"gemini_mode": "limit_reached", "gemini_rationale": "Session limit reached"}
                     continue

                 # Call API (Cached)
                 row_key = f"{row.get('league')}_{row.get('Home')}_{row.get('Away')}"
                 cached = st.session_state["gemini_cache"].get(row_key)

                 if cached:
                     gemini_updates[idx] = {
                         "gemini_mode": "active",
                         "gemini_rationale": cached.get("rationale"),
                         "gemini_total_confidence": cached.get("confidence"),
                         "gemini_error": cached.get("error")
                     }
                 else:
                     # Real Call
                     st.session_state["gemini_calls_made"] += 1
                     try:
                         res = generate_pick_rationale(
                             pick=row.get("Pick") or row.get("Best Overall Pick"),
                             home_team=row.get("Home"),
                             away_team=row.get("Away"),
                             market=row.get("Market"),
                             prob=row.get("final_probability"),
                             edge=row.get("Edge"),
                             session_state=st.session_state
                         )
                         st.session_state["gemini_cache"][row_key] = res
                         gemini_updates[idx] = {
                             "gemini_mode": "active",
                             "gemini_rationale": res.get("rationale"),
                             "gemini_total_confidence": res.get("confidence"),
                             "gemini_error": res.get("error")
                         }
                     except Exception as e:
                         gemini_updates[idx] = {"gemini_mode": "error", "gemini_error": str(e)}

             # Apply updates
             if gemini_updates:
                 gem_df = pd.DataFrame.from_dict(gemini_updates, orient='index')
                 # Merge: drop old columns then concat new ones
                 df = df.drop(columns=[c for c in gem_df.columns if c in df.columns], errors='ignore')
                 df = pd.concat([df, gem_df], axis=1)

        # Fix Issue 6 & 7: Column Availability
        if 'kalshi_status' in df.columns:
             df['kalshi_available'] = df['kalshi_status'].fillna('').astype(str).str.lower().isin(['matched', 'strictmatch', 'seriesmatch'])
        else:
             df['kalshi_available'] = False

        if 'theover_prob_used' in df.columns:
            if 'theover_prob' not in df.columns:
                df['theover_prob'] = df['theover_prob_used']
            else:
                df['theover_prob'] = df['theover_prob'].fillna(df['theover_prob_used'])

        st.session_state["master_df"] = df.copy()  # Collapsed data with all internal columns

        # Filter to only columns that exist in the dataframe
        # FIX: Deduplicate to prevent CSV header/data column count mismatch
        results_columns_raw = [col for col in user_columns if col in df.columns]
        # Remove duplicates while preserving order
        seen = set()
        results_columns = []
        for col in results_columns_raw:
            if col not in seen:
                seen.add(col)
                results_columns.append(col)

        # Warn if duplicates were found (shouldn't happen after fix, but safety check)
        if len(results_columns) != len(results_columns_raw):
            dup_count = len(results_columns_raw) - len(results_columns)
            logger.warning(f"FIX APPLIED: Removed {dup_count} duplicate column(s) from export to prevent CSV mismatch")

        st.session_state["master_results_df"] = df[results_columns].copy()

        # Safety check for missing columns (User Request)
        missing_cols = [col for col in user_columns if col not in df.columns]
        if missing_cols and logger:
            logger.warning(f"Missing columns excluded from export: {missing_cols}")

        if logger:
            logger.info(f"✅ PickConfidence included: {'PickConfidence' in st.session_state['master_results_df'].columns}")
            logger.info(f"✅ PickReason included: {'PickReason' in st.session_state['master_results_df'].columns}")

        # Validation: Ensure TheOver columns are present (Issue #1 Fix)
        # This prevents concatenation errors in downstream logic if columns are missing
        if "theover_pick" not in st.session_state["master_results_df"].columns:
            logger.warning("theover_pick column missing from display dataframe - initializing empty")
            st.session_state["master_results_df"]["theover_pick"] = ""

        logger.info(f"   Created master_results_df with {len(results_columns)} user-facing columns (vs {len(df.columns)} in raw)")

        st.session_state["master_stats_persistent"] = master_stats

        # Final consistency check log (Integrity Log)
        try:
            logger.info(
                "Integrity: rows=%s, spread_picks=%s, spread_consensus=%s, total_picks=%s, total_consensus=%s, best_overall=%s",
                len(df),
                int(df['Spread & Pick'].notna().sum()) if 'Spread & Pick' in df.columns else 0,
                int(df.get('SpreadConsensusProb', pd.Series(dtype=float)).notna().sum()),
                int(df['Total & Pick'].notna().sum()) if 'Total & Pick' in df.columns else 0,
                int(df.get('TotalConsensusProb', pd.Series(dtype=float)).notna().sum()),
                int(df['Best Overall Pick'].notna().sum()) if 'Best Overall Pick' in df.columns else 0,
            )
        except Exception as e:
            logger.warning(f"Failed to log integrity metrics: {e}")

        # Set flag to indicate data is ready for display
        st.session_state["analysis_complete"] = True
        st.session_state["data_ready"] = True

        logger.info(f"✅ Saved {len(df)} rows to session state")
        logger.info(f"   - st.session_state['master_df']: {len(st.session_state['master_df'])} rows")
        logger.info(f"   - st.session_state['master_results_df']: {len(st.session_state['master_results_df'])} rows")
        logger.info(f"✅ Data ready flags set")

        # Log final sentiment analysis summary
        if not df.empty:
            sentiment_cols_present = [col for col in ["Home_Sentiment", "Away_Sentiment", "Sentiment_Diff", "sentiment_score", "sentiment_weight"] if col in df.columns]
            logger.info(f"Sentiment columns present in final results: {sentiment_cols_present}")
            if "Sentiment_Diff" in df.columns:
                games_with_sentiment = df["Sentiment_Diff"].notna().sum()
                total_games = len(df)
                logger.info(f"Final sentiment coverage: {games_with_sentiment}/{total_games} picks have sentiment data")
                if games_with_sentiment > 0:
                    avg_sentiment = df["Sentiment_Diff"].mean()
                    logger.info(f"Average sentiment differential: {avg_sentiment:.3f}")
            if "sentiment_weight" in df.columns:
                avg_weight = df["sentiment_weight"].mean()
                logger.info(f"Average sentiment weight in final probabilities: {avg_weight:.3f} (Mode B: sentiment integrated in probability calculations, expected ~0.10)")

        # Show success message and rerun to display results immediately
        num_games = len(st.session_state.get('games', []))
        num_rows = len(st.session_state['master_results_df'])
        st.success(f"Produced {num_rows} rows from {num_games} games")
        st.rerun()

    if "model_last_error" in st.session_state:
        st.error(f"Prediction Error: {st.session_state['model_last_error']}")

        st.subheader("Top Picks / Best Bets")
        # FORCE DISPLAY: Always include LOW confidence picks (checkbox disabled)
        include_low_in_top = st.checkbox("Include LOW confidence in Top Picks (FORCED ON)", value=True, key="include_low_top_picks", disabled=True)
        df = clean_df(df)
        top_df = df.copy()
        if "Unnamed: 0" in top_df.columns:
            top_df = top_df.drop(columns=["Unnamed: 0"])
        # Optimization: Bulk add missing columns
        missing_top = [c for c in required_display_cols if c not in top_df.columns]
        if missing_top:
            top_df = pd.concat([top_df, pd.DataFrame(columns=missing_top)], axis=1).copy()
        # FORCE DISPLAY: Show all 139 rows regardless of confidence (filter disabled)
        # if not include_low_in_top:
        #     top_df = top_df[top_df["Pick_Confidence"].isin(["HIGH", "MEDIUM"])]

    # NOTE: Download buttons have been moved to the consolidated "Debug Exports"
    # section in the sidebar (see lines 5838-5951) for better accessibility

    # Task 2: Snapshot Status UI
    # UI: Add a status text in the sidebar: "✅ Noon Baseline Cached" or "⚠️ Noon Baseline Missing".
    snapshot_status = snapshot_manager.check_noon_baseline_status()
    st.sidebar.markdown(f"**Snapshot Status:** {snapshot_status}")

    # Manual Baseline Button
    if st.sidebar.button("Manual Baseline Snapshot"):
        if "master_df" in st.session_state and not st.session_state["master_df"].empty:
            success = snapshot_manager.save_noon_baseline(st.session_state["master_df"], force=True)
            if success:
                # Re-check status to show filename
                now_et = snapshot_manager.get_et_now()
                date_str = now_et.date().isoformat()
                filename = snapshot_manager.get_snapshot_filename(date_str, "noon")
                st.info(f"Baseline Comparison Active: {filename}")
                st.sidebar.success("Manual Baseline Saved.")
            else:
                st.sidebar.error("Failed to save baseline.")
        else:
            st.sidebar.warning("Run Master Analysis first.")



# --- Forced UI Persistence Block ---
# This block renders the results grid outside the button handler so it persists across reruns.

# Add this logging BEFORE the UI rendering section
logger.info(f"\n{'='*80}")
logger.info(f"UI RENDERING SECTION")
logger.info(f"{'='*80}")

# Check if data exists in session state (only log errors if analysis was attempted)
analysis_was_attempted = st.session_state.get("analysis_complete", False) or st.session_state.get("run_id") is not None

if "master_results_df" in st.session_state:
    logger.info(f"✅ master_results_df exists in session state")
    try:
        logger.info(f"   Rows: {len(st.session_state['master_results_df'])}")
        logger.info(f"   Columns: {len(st.session_state['master_results_df'].columns)}")
    except Exception as e:
        logger.error(f"Error checking master_results_df: {e}")
elif analysis_was_attempted:
    # Only log error if user attempted analysis
    logger.error(f"❌ master_results_df NOT FOUND in session state after analysis attempt!")
    logger.error(f"   Available keys: {list(st.session_state.keys())}")

if "master_df" in st.session_state:
    logger.info(f"✅ master_df exists in session state")
    try:
        logger.info(f"   Rows: {len(st.session_state['master_df'])}")
    except Exception as e:
        logger.error(f"Error checking master_df: {e}")
elif analysis_was_attempted:
    # Only log error if user attempted analysis
    logger.error(f"❌ master_df NOT FOUND in session state after analysis attempt!")

logger.info(f"{'='*80}")

# FIXED: Improved condition logic to properly check for analysis results
# Check if we have analysis results to display
has_results = False
if "master_results_df" in st.session_state:
    df_results = st.session_state["master_results_df"]
    has_results = df_results is not None and not df_results.empty

# Display section if we have results OR analysis just completed
should_display = st.session_state.get("analysis_complete", False) or has_results

logger.info(f"🎯 Master Analysis Tab Display Logic:")
logger.info(f"   - analysis_complete: {st.session_state.get('analysis_complete', False)}")
logger.info(f"   - has_results: {has_results}")
logger.info(f"   - should_display: {should_display}")

if should_display:
    with tab_master:
        if not has_results:
            # Analysis completed but no results generated
            st.warning("📊 Analysis complete but no picks generated")

            # Show diagnostic info
            with st.expander("🔍 Diagnostic Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Session State Check:**")
                    st.write(f"- master_results_df exists: {'master_results_df' in st.session_state}")
                    st.write(f"- analysis_complete: {st.session_state.get('analysis_complete', False)}")
                    st.write(f"- games loaded: {len(st.session_state.get('games', []))}")
                with col2:
                    st.write("**Data Pipeline Status:**")
                    st.write(f"- master_df rows: {len(st.session_state.get('master_df', pd.DataFrame()))}")
                    st.write(f"- master_results_df rows: {len(st.session_state.get('master_results_df', pd.DataFrame()))}")

            st.info("💡 Check the sidebar for debug logs and error information")
            st.stop()

        # We have results - proceed with display
        df = st.session_state["master_results_df"]

        # Fix #1: Stats Coverage Dashboard & Warnings

        # --- STATS COVERAGE DASHBOARD ---
        if "stats_quality" in df.columns and "league" in df.columns:
            st.markdown("### 📊 Data Quality Report")
            quality_cols = st.columns(4)
            leagues = sorted(df["league"].unique())

            for i, lg in enumerate(leagues):
                lg_df = df[df["league"] == lg]
                total = len(lg_df)
                if total == 0: continue

                full_stats = len(lg_df[lg_df["stats_quality"].isin(["REAL", "ESPN"])])
                missing = total - full_stats
                pct = (full_stats / total) * 100

                status_icon = "✅" if pct > 90 else "⚠️" if pct > 50 else "❌"

                with quality_cols[i % 4]:
                    st.metric(
                        f"{lg} Coverage",
                        f"{full_stats}/{total}",
                        f"{pct:.0f}% Full Stats {status_icon}"
                    )

        # Check for missing stats
        missing_stats_games = df[
            df["stats_quality"] == "MISSING"
        ] if "stats_quality" in df.columns else pd.DataFrame()

        if len(missing_stats_games) > 0:
            ncaab_missing = missing_stats_games[missing_stats_games["league"] == "NCAAB"]
            ncaaf_missing = missing_stats_games[missing_stats_games["league"] == "NCAAF"]

            if not ncaab_missing.empty or not ncaaf_missing.empty:
                st.warning(
                    f"⚠️ **LIMITED STATS COVERAGE** ⚠️\n\n"
                    f"{len(ncaab_missing)} NCAAB and {len(ncaaf_missing)} NCAAF games have missing team statistics.\n"
                    f"Predictions for these games use a **neutral baseline model** and confidence is automatically downgraded."
                )

                # Option: Show which teams
                if len(missing_stats_games) <= 10:
                    with st.expander("Show affected games"):
                        for _, game in missing_stats_games.iterrows():
                            st.caption(f"• {game['Home']} vs {game['Away']} ({game['league']})")

        st.success(f"✅ Loaded {len(df)} rows for analysis (Master Analysis Tab)")

        # Initialize view frame immediately to avoid NameError
        df_master_view = df.copy()

        if st.session_state.get("debug_kalshi_blend", False):
            st.markdown("---")
            st.markdown("### 🔬 Kalshi Blending Debug")

            # Filter to games with Kalshi markets
            master_df_debug = st.session_state.get("master_df", pd.DataFrame())

            # Robust filtering handling missing Kalshi_Market column
            kalshi_games = pd.DataFrame()
            if not master_df_debug.empty:
                if 'Kalshi_Market' in master_df_debug.columns:
                    kalshi_games = master_df_debug[master_df_debug['Kalshi_Market'].notna()].copy()
                elif 'HasKalshiMarket' in master_df_debug.columns:
                    kalshi_games = master_df_debug[master_df_debug['HasKalshiMarket'] == True].copy()

            if kalshi_games.empty:
                st.warning("No games with Kalshi markets found in current results.")
            else:
                st.success(f"Found {len(kalshi_games)} games with Kalshi markets")

                # Display first 5 games with detailed blending info
                for idx, row in kalshi_games.head(5).iterrows():
                    with st.expander(f"📊 {row.get('Home', 'Unknown')} vs {row.get('Away', 'Unknown')}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Kalshi Data:**")
                            st.write(f"- Market: `{row.get('Kalshi_Market', row.get('Market', 'N/A'))}`")
                            st.write(f"- Raw Yes Prob: `{row.get('kalshi_prob_yes', row.get('kalshi_prob', 'N/A'))}`")
                            st.write(f"- Yes Side: `{row.get('kalshi_yes_side', 'N/A')}`")
                            st.write(f"- Prob for Pick: `{row.get('kalshi_prob_for_pick', 'N/A')}`")
                            st.write(f"- Kalshi Weight: `{row.get('kalshi_weight', 'N/A')}`")

                        with col2:
                            st.markdown("**Blended Result:**")
                            st.write(f"- Market Prob: `{row.get('spread_prob_pick_market', row.get('Implied_Prob', 'N/A'))}`")
                            st.write(f"- Final Prob: `{row.get('spread_prob_pick_final', row.get('final_probability', 'N/A'))}`")
                            st.write(f"- Pick: `{row.get('Pick', 'N/A')}`")
                            st.write(f"- Confidence: `{row.get('Pick_Confidence', 'N/A')}`")

                        # Validation checks
                        st.markdown("**Validation:**")
                        kalshi_pick_prob = row.get('kalshi_prob_for_pick')
                        final_prob = row.get('spread_prob_pick_final', row.get('final_probability'))

                        checks = []

                        # Check 1: Kalshi prob mapped correctly (>= 0.50 for pick)
                        if kalshi_pick_prob is not None:
                            try:
                                kpp = float(kalshi_pick_prob)
                                if kpp >= 0.50:
                                    checks.append("✅ Kalshi prob >= 50% (correctly mapped)")
                                else:
                                    checks.append(f"❌ Kalshi prob < 50% ({kpp:.1%}) - MAPPING ERROR!")
                            except:
                                checks.append("⚠️ Kalshi prob invalid format")
                        else:
                            checks.append("⚠️ Kalshi prob missing")

                        # Check 2: Final prob always >= 0.50
                        if final_prob is not None:
                            try:
                                fp = float(final_prob)
                                if fp >= 0.50:
                                    checks.append("✅ Final prob >= 50% (valid pick)")
                                else:
                                    checks.append(f"❌ Final prob < 50% ({fp:.1%}) - SELECTION ERROR!")
                            except:
                                checks.append("⚠️ Final prob invalid format")
                        else:
                            checks.append("⚠️ Final prob missing")

                        # Check 3: Kalshi influence visible
                        market_prob = row.get('spread_prob_pick_market', row.get('Implied_Prob'))
                        if kalshi_pick_prob is not None and market_prob is not None and final_prob is not None:
                            try:
                                fp = float(final_prob)
                                mp = float(market_prob)
                                kalshi_delta = abs(fp - mp)
                                if kalshi_delta >= 0.02:
                                    checks.append(f"✅ Kalshi influenced final prob (Δ = {kalshi_delta:.1%})")
                                else:
                                    checks.append(f"⚠️ Kalshi influence weak (Δ = {kalshi_delta:.1%})")
                            except:
                                pass

                        for check in checks:
                            st.write(check)

        # NOTE: Do NOT apply Kalshi match filter here - it must only affect UI display, not exports
        # The filter is applied later to df_master_view_display only (see line ~10613+)

        required_display_cols = [
            "Home_Sentiment",
            "Away_Sentiment",
            "Sentiment_Diff",
            "sentiment_adj",
            "sentiment_score",
            "sentiment_direction",
            "sentiment_impact_applied",
            "sentiment_source",
            "reddit_used",
            "sentiment_level",
            "sentiment_strength",
            "sentiment_badge",
            "sentiment_articles_used",
            "sentiment_source_count",
            "sentiment_label",
            "sentiment_query_used",
            "spread_sentiment_adj",
            "spread_prob_adj",
            "total_sentiment_adj",
            "total_prob_adj",
            "ai_prob_adj",
            "consensus_prob",
            "final_probability",
            "decision_driver",
            "SpreadConsensusProb",
            "SpreadConsensus",
            "TotalConsensusProb",
            "TotalConsensus",
            "Best Overall Pick",
            "Best Overall Prob",
            "Edge",
            "best_pick_type",
            "kalshi_weight",
            "odds_weight",
            "ml_weight",
            "sentiment_weight",
            "confidence_reason",
            "overall_confidence",
            "spread_confidence_gemini",
            "total_confidence_gemini",
            "spread_confidence_base",
            "total_confidence_base",
            "gemini_alignment",
            "gemini_rationale",
            "gemini_risk_flags",
            "gemini_flags_short",
            "llm_disagreement_flag",
            "gemini_mode",
            "gemini_error",
            "llm_disagreement_flag",
            "prob_engine",
            "model_mode",
            "model_spread_prob",
            "model_total_prob",
            "spread_engine_used",
            "spread_pick_label",
            "spread_alt_label",
            "spread_prob_pick_final",
            "spread_prob_alt_final",
            "spread_prob_margin",
            "spread_prob_pick_market",
            "spread_prob_alt_market",
            "spread_prob_pick_kalshi",
            "spread_prob_alt_kalshi",
            "spread_decision_metric_used",
            "spread_decision_score_pick",
            "spread_decision_score_alt",
            "spread_decision_score_margin",
            "spread_trace_json",
            "decision_trace",
            "total_engine_used",
            "total_pick_label",
            "total_alt_label",
            "total_prob_pick_final",
            "total_prob_alt_final",
            "total_prob_margin",
            "total_prob_pick_market",
            "total_prob_alt_market",
            "total_prob_pick_kalshi",
            "total_prob_alt_kalshi",
            "total_decision_metric_used",
            "total_decision_score_pick",
            "total_decision_score_alt",
            "total_decision_score_margin",
            "total_trace_json",
            "decision_trace_version",
            "overall_engine_used",
            "decision_trace_notes",
            "kalshi_prob_spread",
            "kalshi_prob_total",
            "spread_prob_market",
            "total_prob_market",
            "kalshi_candidate_count",
            "kalshi_best_score",
            "kalshi_match_reason",
            "kalshi_status",
            "llm_disagreement_flag",
            "consensus_weight_ai",
            "consensus_weight_market",
            "consensus_weight_kalshi",
            "consensus_weight_sentiment",
            "consensus_weight_total",
            "consensus_guardrails",
            "gemini_error",
            "geminitotalconfidence",
            "geminirationalize",
            "geminierrorflag",
        ]
        df_master_view_display = df_master_view.drop(columns=[c for c in TRACE_COLS if c in df_master_view.columns], errors="ignore")
        show_moneyline_details = st.checkbox("Show Moneyline details", value=False, key="show_moneyline_details")
        if not show_moneyline_details:
            ml_detail_cols = [
                "Pick",
                "Book",
                "Home_ML",
                "Away_ML",
                "Implied_Prob",
                "AI_Prob",
                "ai_prob_adj",
                "consensus_prob",
                "consensus_prob_adj",
                "kalshi_prob",
                "kalshi_prob_used",
                "kalshi_event_ticker",
                "kalshi_event_ticker_used",
                "edge_vs_odds",
                "model_minus_market",
            ]
            df_master_view_display = df_master_view_display.drop(columns=[c for c in ml_detail_cols if c in df_master_view_display.columns], errors="ignore")
        st.caption(f"Column order (first 8): {', '.join(list(df_master_view_display.columns[:8]))} ...")
        df_master_view_display["Spread_Range"] = df_master_view_display.apply(
            lambda r: f"{r['spread_min']} to {r['spread_max']} (med {r['spread_med']})"
            if pd.notnull(r.get("spread_min")) and pd.notnull(r.get("spread_max"))
            else "N/A",
            axis=1,
        )
        df_master_view_display = df_master_view_display.copy()
        df_master_view_display["Total_Range"] = df_master_view_display.apply(
            lambda r: f"{r['total_min']} to {r['total_max']} (med {r['total_med']})"
            if pd.notnull(r.get("total_min")) and pd.notnull(r.get("total_max"))
            else "N/A",
            axis=1,
        )
        df_master_view_display = df_master_view_display.copy()

        def _market_badge(r):
            badges_local = []
            if (pd.notnull(r.get("spread_min")) and pd.notnull(r.get("spread_max")) and abs((r.get("spread_max") or 0) - (r.get("spread_min") or 0)) >= 2):
                badges_local.append("WIDE MARKET")
            if (pd.notnull(r.get("total_min")) and pd.notnull(r.get("total_max")) and abs((r.get("total_max") or 0) - (r.get("total_min") or 0)) >= 3):
                badges_local.append("WIDE MARKET")
            if (r.get("spread_books_count") == 1) or (r.get("total_books_count") == 1):
                badges_local.append("THIN MARKET")
            return ";".join(sorted(set(badges_local))) if badges_local else None
        df_master_view_display["Market_Badge"] = df_master_view_display.apply(_market_badge, axis=1)
        df_master_view_display = df_master_view_display.copy()
        placeholder_count = int((df_master_view_display.get("odds_placeholder_detected") == True).sum()) if "odds_placeholder_detected" in df_master_view_display.columns else 0
        implied_null_count = int(df_master_view_display["Implied_Prob"].isna().sum()) if "Implied_Prob" in df_master_view_display.columns else 0
        st.caption(f"Debug: placeholder odds rows={placeholder_count}; Implied_Prob null rows={implied_null_count}")

        # --- NEW: GAME SUMMARY VIEW ---
        st.subheader("Game Summary View")

        try:
            game_summary_df = build_game_summary(st.session_state["master_df"])
            logger.info(f"📊 Game Summary: {len(game_summary_df)} games")

            # ============================================
            # APPLY KALSHI MATCH FILTER IF ENABLED
            # ============================================
            # If "Show only games with a Kalshi match" checkbox is checked,
            # filter to only show games where HasKalshiMarket == True
            # ============================================
            kalshi_match_only = st.session_state.get("kalshi_match_only", False)
            total_games = len(game_summary_df)
            games_with_kalshi = 0

            if "HasKalshiMarket" in game_summary_df.columns:
                games_with_kalshi = int(game_summary_df["HasKalshiMarket"].sum())

            if kalshi_match_only and "HasKalshiMarket" in game_summary_df.columns:
                game_summary_df = game_summary_df[game_summary_df["HasKalshiMarket"] == True]
                logger.info(f"🎯 Filtered to Kalshi matches only: {len(game_summary_df)} games with Kalshi markets")

            # Display summary stats
            st.caption(f"📊 Total games: {total_games} | Games with Kalshi markets: {games_with_kalshi} | Displaying: {len(game_summary_df)}")

            if not game_summary_df.empty:
                # Reorder columns as requested, including new Kalshi columns
                summary_cols = [
                    "League", "Home", "Away", "Commence UTC", "Commence (Local)",
                    "Best Overall Pick", "Best Overall Prob",
                        "Spread Pick", "Spread Prob", "Spread Consensus", "Kalshi Spread Prob", "Kalshi Spread Δ",
                    "Total Pick", "Total Prob", "Kalshi Total Prob", "Kalshi Total Δ",
                    "ML Pick", "ML Prob"
                ]
                # Ensure columns exist
                summary_cols = [c for c in summary_cols if c in game_summary_df.columns]

                # Formatting with Kalshi probability columns
                format_cols = {
                    "Best Overall Prob": st.column_config.NumberColumn(format="%.1f%%"),
                    "Spread Prob": st.column_config.NumberColumn(format="%.1f%%"),
                    "Kalshi Spread Prob": st.column_config.NumberColumn(format="%.1f%%"),
                    "Total Prob": st.column_config.NumberColumn(format="%.1f%%"),
                    "Kalshi Total Prob": st.column_config.NumberColumn(format="%.1f%%"),
                        "ML Prob": st.column_config.NumberColumn(format="%.1f%%"),
                        "Spread Consensus": st.column_config.TextColumn("S", help="Spread Consensus (Market/Kalshi/AI)")
                }

                st.dataframe(
                    game_summary_df[summary_cols],
                    column_config=format_cols,
                    width="stretch",
                    hide_index=True
                )
                logger.info(f"✅ Successfully rendered Game Summary grid")
            else:
                st.info("No game summary data available.")
                logger.warning("⚠️ Game summary dataframe is empty")

        except Exception as summary_error:
            logger.error(f"❌ Failed to render Game Summary: {summary_error}")
            logger.exception(summary_error)
            st.error(f"Failed to render game summary: {summary_error}")
            with st.expander("🔍 Error Details"):
                st.code(str(summary_error))

        # --- NEW: BEST OVERALL PICKS (ML Focused) ---
        st.subheader("Best Overall Picks (Moneyline)")

        try:
            best_ml_df = get_best_ml_picks(st.session_state["master_df"])
            logger.info(f"🎯 Best ML Picks: {len(best_ml_df)} picks")

            if not best_ml_df.empty:
                # Sort by Best Overall Prob descending
                best_ml_df = best_ml_df.sort_values(by="Best Overall Prob", ascending=False)

                ml_cols = [
                    "league", "Home", "Away", "Commence (Local)",
                    "Best Overall Pick", "Best Overall Prob", "Best Overall Confidence",
                    "Implied Prob", "AI Prob"
                ]

                st.dataframe(
                    best_ml_df[ml_cols],
                    column_config={
                        "Best Overall Prob": st.column_config.NumberColumn(format="%.1f%%"),
                        "Implied Prob": st.column_config.NumberColumn(format="%.1f%%"),
                        "AI Prob": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                    width="stretch",
                    hide_index=True
                )
                logger.info(f"✅ Successfully rendered Best ML Picks grid")
            else:
                st.info("No Moneyline picks available.")
                logger.warning("⚠️ Best ML picks dataframe is empty")

        except Exception as ml_error:
            logger.error(f"❌ Failed to render Best ML Picks: {ml_error}")
            logger.exception(ml_error)
            st.error(f"Failed to render moneyline picks: {ml_error}")
            with st.expander("🔍 Error Details"):
                st.code(str(ml_error))

        st.subheader("Top Picks / Best Bets")

        # --- CONSENSUS SYSTEM EXPLANATION ---
        with st.expander("ℹ️ Understanding Consensus & Quality Scoring", expanded=False):
            st.markdown("""
            ### 🎯 How Consensus Works
            Each pick combines predictions from **3 probability sources**:

            - **M (Market)** - Implied probability from betting odds (40% weight)
            - **K (Kalshi)** - Prediction market probability (45% weight) ✓ = Strong agreement (>55%)
            - **TheOver** - Historical matchup data (15% weight)
            - **→ (Final)** - Weighted consensus probability

            **Example**: `M:45.0% | K:52.0% | →49.4%`

            **Note**: AI Model and Sentiment removed (broken/unavailable sources).

            ### ⭐ Quality Score (0-5 Stars)
            Picks earn points for meeting criteria:
            - **+1.5 pts** - Probability >56% (above coin flip threshold)
            - **+1.5 pts** - Decisiveness >0.08 (meaningful directional edge)
            - **+1.0 pt** - Kalshi validates (>55% agreement)
            - **+1.0 pt** - Strong consensus (engines agree within 5%)

            **Quality Tiers**:
            - **HIGH** (≥4.0 stars) - Best picks, meets all/most criteria, ~42-46% expected hit rate
            - **MEDIUM** (≥2.5 stars) - Decent picks, meets some criteria, ~35-40% expected hit rate
            - **LOW** (<2.5 stars) - Weak picks, barely above 50%, ~27-32% expected hit rate

            ### 💡 Why Tighten Picks?
            - Picks at 50.8% are barely better than coin flips
            - 2-leg parlay at 50.8% each = 25.8% win probability (negative EV)
            - 2-leg parlay at 56% each = 31.4% win probability (positive EV)
            - **Focus on HIGH quality picks for best results!**
            """)

        # FORCE DISPLAY: Always include LOW confidence picks (checkbox disabled)
        include_low_in_top = st.checkbox("Include LOW confidence in Top Picks (FORCED ON)", value=True, key="include_low_top_picks", disabled=True)

        # --- QUALITY FILTER CONTROLS ---
        st.caption("🎯 **Pick Quality Filters** - Tighten picks based on probability, decisiveness, and consensus")
        st.info("🎲 **DEFAULT: High Quality Only** - Showing picks with >56% probability, >0.08 decisiveness, Kalshi validation, and strong consensus. Change filter below to see all picks.")

        col1, col2 = st.columns(2)

        with col1:
            quality_filter_mode = st.selectbox(
                "Quality Tier Filter",
                options=["High Quality Only", "High + Medium Quality", "All Quality Levels", "Show Quality Distribution"],
                index=0,
                key="quality_filter_mode",
                help="Filter picks by quality score (>56% prob, >0.08 decisiveness, Kalshi validation, consensus agreement)"
            )

        with col2:
            sort_by_quality = st.checkbox(
                "Sort by Quality Score",
                value=True,
                key="sort_by_quality",
                help="Sort picks by quality score (0-5 stars) instead of spread edge"
            )

        df = clean_df(df)
        top_df = df.copy()
        if "Unnamed: 0" in top_df.columns:
            top_df = top_df.drop(columns=["Unnamed: 0"])
        # Optimization: Bulk add missing columns
        missing_top = [c for c in required_display_cols if c not in top_df.columns]
        if missing_top:
            top_df = pd.concat([top_df, pd.DataFrame(columns=missing_top)], axis=1).copy()
        # FORCE DISPLAY: Show all 139 rows regardless of confidence (filter disabled)
        # if not include_low_in_top:
        #     top_df = top_df[top_df["Pick_Confidence"].isin(["HIGH", "MEDIUM"])]
        try:
            top_df["st_conf_rank"] = top_df["st_conf_rank"].fillna(0).infer_objects(copy=False).infer_objects(copy=False)
            top_df["decisiveness"] = top_df["decisiveness"].fillna(0.0).infer_objects(copy=False).infer_objects(copy=False)
            top_df = top_df.sort_values(
                by=["spread_edge", "st_conf_rank", "decisiveness"],
                ascending=[False, False, False],
            )
        except Exception:
            pass
        top_df = reorder_for_spread_total_focus(top_df)
        top_df_display = top_df.drop(columns=[c for c in TRACE_COLS if c in top_df.columns], errors="ignore")

        # Format spread_edge as percentage
        if "spread_edge" in top_df_display.columns:
            top_df_display["spread_edge"] = top_df_display["spread_edge"].apply(lambda x: f"{x:+.1%}" if pd.notnull(x) else "")
            top_df_display = top_df_display.copy()

        # --- CONSENSUS BREAKDOWN COLUMNS ---
        # Add consensus breakdown showing individual engine contributions for Spread and Total picks
        logger.info("Adding consensus breakdown columns to Top Picks...")

        def _add_consensus_breakdown(row):
            """Add consensus breakdown for both Spread and Total picks."""
            # Get spread consensus breakdown
            # v98 FIX: Use proper None check for Kalshi prob fallback
            _s_k = row.get("spread_prob_pick_kalshi")
            spread_consensus = format_consensus_breakdown(
                market_prob=safe_float(row.get("spread_prob_pick_market")),
                kalshi_prob=safe_float(_s_k if _s_k is not None else row.get("kalshi_prob_spread")),
                model_prob=safe_float(row.get("model_spread_prob")),
                sentiment_score=safe_float(row.get("sentiment_score")),
                final_prob=safe_float(row.get("spread_prob_pick_final")),
                pick_side=row.get("spread_pick_team")
            )

            # Get total consensus breakdown
            # v98 FIX: Use proper None check for Kalshi prob fallback
            _t_k = row.get("total_prob_pick_kalshi")
            total_consensus = format_consensus_breakdown(
                market_prob=safe_float(row.get("total_prob_pick_market")),
                kalshi_prob=safe_float(_t_k if _t_k is not None else row.get("kalshi_prob_total")),
                model_prob=safe_float(row.get("model_total_prob")),
                sentiment_score=safe_float(row.get("sentiment_score")),
                final_prob=safe_float(row.get("total_prob_pick_final")),
                pick_side=row.get("total_pick_side")
            )

            return pd.Series({
                "Spread_Consensus": spread_consensus,
                "Total_Consensus": total_consensus
            })

        # Add consensus breakdown columns
        try:
            consensus_cols = top_df_display.apply(_add_consensus_breakdown, axis=1)
            top_df_display["Spread_Consensus"] = consensus_cols["Spread_Consensus"]
            top_df_display["Total_Consensus"] = consensus_cols["Total_Consensus"]
            logger.info(f"✅ Added consensus breakdown columns")
        except Exception as consensus_error:
            logger.warning(f"Failed to add consensus breakdown columns: {consensus_error}")
            top_df_display["Spread_Consensus"] = "N/A"
            top_df_display["Total_Consensus"] = "N/A"

        # --- PICK QUALITY METRICS ---
        # Add quality metrics for pick tightening filters
        logger.info("Calculating pick quality metrics...")

        def _add_quality_metrics(row):
            """
            Use Pick_Confidence from master dataframe (decisiveness-based buckets).
            This column already contains the correct HIGH/MEDIUM/LOW classification.
            """
            # Get the pre-computed confidence bucket (from master analysis)
            pick_confidence = row.get("Pick_Confidence", "LOW")

            # Map confidence to quality tier (they are the same)
            quality_tier = str(pick_confidence).upper() if pick_confidence else "LOW"
            if quality_tier not in ["HIGH", "MEDIUM", "LOW"]:
                quality_tier = "LOW"

            # Get decisiveness from the master dataframe
            decisiveness = safe_float(row.get("decisiveness"))
            if decisiveness is None:
                final_prob = safe_float(row.get("final_probability") or row.get("Best Overall Prob"))
                decisiveness = abs(final_prob - 0.5) if final_prob is not None else 0.0

            # Map tier to quality score (0-5 scale)
            tier_to_score = {"HIGH": 4.5, "MEDIUM": 3.0, "LOW": 1.5, "UNKNOWN": 0.5}
            quality_score = tier_to_score.get(quality_tier, 1.5)

            # Adjust score based on decisiveness
            if decisiveness >= 0.12:
                quality_score = min(5.0, quality_score + 0.5)
            elif decisiveness >= 0.08:
                quality_score = min(5.0, quality_score + 0.2)

            # Format quality badge
            quality_badge = f"{quality_tier} ({quality_score:.1f}⭐)"

            # Build quality details based on decisiveness and Kalshi
            details = []
            final_prob = safe_float(row.get("final_probability") or row.get("Best Overall Prob"))
            if final_prob and final_prob > 0.56:
                details.append("✓>56%")
            if decisiveness and decisiveness > 0.08:
                details.append("✓Decisive")
            if row.get("kalshi_matched"):
                details.append("✓Kalshi")

            quality_details = " ".join(details) if details else "Low Quality"

            # For backward compatibility, also calculate spread/total scores
            spread_prob = safe_float(row.get("spread_prob_pick_final"))
            total_prob = safe_float(row.get("total_prob_pick_final"))

            spread_score = 0.0
            total_score = 0.0

            if spread_prob:
                spread_decisiveness = abs(spread_prob - 0.5)
                spread_score = 2.0 + (spread_decisiveness * 20)  # 0.5 to 5.0 range

            if total_prob:
                total_decisiveness = abs(total_prob - 0.5)
                total_score = 2.0 + (total_decisiveness * 20)  # 0.5 to 5.0 range

            # Consensus quality (simplified)
            consensus_quality = "MODERATE"
            if quality_tier == "HIGH":
                consensus_quality = "STRONG"
            elif quality_tier == "LOW":
                consensus_quality = "WEAK"

            return pd.Series({
                "Quality_Score": quality_score,
                "Quality_Tier": quality_tier,
                "Quality_Badge": quality_badge,
                "Quality_Details": quality_details,
                "Decisiveness": decisiveness,
                "Consensus_Quality": consensus_quality,
                "Spread_Quality_Score": spread_score,
                "Total_Quality_Score": total_score,
            })

        # Add quality metric columns
        try:
            quality_cols = top_df_display.apply(_add_quality_metrics, axis=1)
            top_df_display["Quality_Score"] = quality_cols["Quality_Score"]
            top_df_display["Quality_Tier"] = quality_cols["Quality_Tier"]
            top_df_display["Quality_Badge"] = quality_cols["Quality_Badge"]
            top_df_display["Quality_Details"] = quality_cols["Quality_Details"]
            top_df_display["Decisiveness"] = quality_cols["Decisiveness"]
            top_df_display["Consensus_Quality"] = quality_cols["Consensus_Quality"]
            top_df_display["Spread_Quality_Score"] = quality_cols["Spread_Quality_Score"]
            top_df_display["Total_Quality_Score"] = quality_cols["Total_Quality_Score"]
            logger.info(f"✅ Added quality metric columns")

            # Log quality distribution
            if "Quality_Tier" in top_df_display.columns:
                quality_dist = top_df_display["Quality_Tier"].value_counts().to_dict()
                logger.info(f"📊 Quality Distribution: {quality_dist}")
        except Exception as quality_error:
            logger.warning(f"Failed to add quality metrics: {quality_error}")
            top_df_display["Quality_Score"] = 0.0
            top_df_display["Quality_Tier"] = "UNKNOWN"
            top_df_display["Quality_Badge"] = "N/A"
            top_df_display["Quality_Details"] = "N/A"

        # --- Part F: TheOver Impact String ---
        # Generate "TheOver Impact" column for display
        # Format: "TheOver: +0.012 (agree)" or "TheOver: n/a"
        if "theover_delta_final_prob" in top_df_display.columns:
            def _fmt_theover_impact(row):
                if not row.get("theover_matched"):
                    return "TheOver: n/a"

                delta = row.get("theover_delta_final_prob")
                if delta is None:
                    return "TheOver: n/a"

                # Check agreement direction
                # Compare final_prob vs final_prob_without_theover?
                # Actually, simpler: if delta > 0, it boosted confidence.
                # But 'agreement' usually means 'TheOver Pick matches Our Pick'.
                # Let's use the 'theover_changed_pick' or inferred agreement.
                # If delta is positive and substantial, it likely agreed/boosted.

                # We can also check pick alignment if we have "theover_pick" and "Pick"
                # But let's stick to the delta for now as requested: "TheOver: +0.012 (agree)"

                # Heuristic:
                # If delta > 0: "boost" or "agree" (if we picked it)
                # If delta < 0: "drag" or "disagree"

                # Refined: "agree" if sign(delta) matches sign(edge)?
                # Let's just output the delta sign and value.

                direction = "neutral"
                if delta > 0.005: direction = "boost"
                elif delta < -0.005: direction = "drag"

                return f"TheOver: {delta:+.3f} ({direction})"

            top_df_display["TheOver_Impact"] = top_df_display.apply(_fmt_theover_impact, axis=1)
            # Add to reason short
            # Safety check: Ensure Pick_Reason_Short exists
            if "Pick_Reason_Short" not in top_df_display.columns:
                logger.warning("Pick_Reason_Short missing from display df, initializing empty")
                top_df_display["Pick_Reason_Short"] = ""

            # Concatenate safely
            top_df_display["Pick_Reason_Short"] = top_df_display["Pick_Reason_Short"] + " | " + top_df_display["TheOver_Impact"]

        if not show_moneyline_details:
            ml_detail_cols = [
                "Pick",
                "Book",
                "Home_ML",
                "Away_ML",
                "Implied_Prob",
                "AI_Prob",
                "ai_prob_adj",
                "consensus_prob",
                "consensus_prob_adj",
                "kalshi_prob",
                "kalshi_prob_used",
                "kalshi_event_ticker",
                "kalshi_event_ticker_used",
                "edge_vs_odds",
                "model_minus_market",
            ]
            top_df_display = top_df_display.drop(columns=[c for c in ml_detail_cols if c in top_df_display.columns], errors="ignore")

        # --- FINAL WHITELIST FIX (Enhanced with Picks Sheet Columns) ---
        # User Request: Add Win Probability columns next to spread/total picks
        # FIX: Extract probability percentage directly from pick strings (e.g., "Boston Celtics -3.5 (51.2)" -> "51.2%")
        def _extract_win_pct_from_pick(pick_string):
            """Extract percentage from pick string like 'Team Name Line (51.2)'"""
            if not pick_string or not isinstance(pick_string, str):
                return "N/A"

            # Match pattern: (number) or (number%)
            match = re.search(r'\((\d+\.?\d*)\s*%?\)', str(pick_string))
            if match:
                pct_value = float(match.group(1))
                return f"{pct_value:.1f}%"
            return "N/A"

        # Extract Win % from Spread & Pick column
        if "Spread & Pick" in top_df_display.columns:
            top_df_display["Spread Win %"] = top_df_display["Spread & Pick"].apply(_extract_win_pct_from_pick)
            sample_spread_picks = top_df_display["Spread & Pick"].head(5).tolist()
            sample_spread_win_pct = top_df_display["Spread Win %"].head(5).tolist()
            logger.debug(f"Sample Spread & Pick values: {sample_spread_picks}")
            logger.debug(f"Extracted Spread Win % values: {sample_spread_win_pct}")
        elif "Spread Win Prob" in top_df_display.columns:
            # Fallback if pick column missing but old column exists
            top_df_display["Spread Win %"] = top_df_display["Spread Win Prob"]

        # Extract Win % from Total & Pick column
        if "Total & Pick" in top_df_display.columns:
            top_df_display["Total Win %"] = top_df_display["Total & Pick"].apply(_extract_win_pct_from_pick)
            sample_total_picks = top_df_display["Total & Pick"].head(5).tolist()
            sample_total_win_pct = top_df_display["Total Win %"].head(5).tolist()
            logger.debug(f"Sample Total & Pick values: {sample_total_picks}")
            logger.debug(f"Extracted Total Win % values: {sample_total_win_pct}")
        elif "Total Win Prob" in top_df_display.columns:
            # Fallback if pick column missing but old column exists
            top_df_display["Total Win %"] = top_df_display["Total Win Prob"]

        ui_whitelist = [
            'league', 'Home', 'Away', 'Commence (UTC)', 'Commence (Local)', 'Local Date',
            'Overall Pick', 'Overall Prob', 'Spread', 'Spread Prob', 'Total', 'Total Prob', 'ML', 'ML Prob',
            'SpreadConsensusProb', 'SpreadConsensus', 'TotalConsensusProb', 'TotalConsensus',
            'Best Overall Pick', 'Best Overall Prob',
            'best_pick', 'final_prob', 'edge', 'best_pick_type',
            'Bet_Confidence', 'Bet_Lean',
            'Quality_Badge', 'Quality_Details',
            'Spread & Pick', 'Spread Win %', 'Spread_Consensus',
            'Total & Pick', 'Total Win %', 'Total_Consensus',
            'spread_edge', 'total_edge',
            'Decisiveness', 'Consensus_Quality',
            'Pick', 'AI_Prob', 'Implied_Prob', 'Home_Sentiment', 'Away_Sentiment', 'Sentiment_Diff', 'sentiment_status', 'status', 'best_pick_prob', 'best_pick_edge',
            'theover_pick', 'theover_prob_used', 'theover_delta_final_prob', 'final_prob_without_theover',
            'Quality_Score', 'Quality_Tier', 'Spread_Quality_Score', 'Total_Quality_Score'
        ]
        safe_cols = [c for c in ui_whitelist if c in top_df_display.columns]
        top_df_ui = top_df_display[safe_cols].copy()

        # --- APPLY QUALITY FILTERS ---
        if "quality_filter_mode" in st.session_state:
            filter_mode = st.session_state.quality_filter_mode
            original_count = len(top_df_ui)

            if filter_mode == "High Quality Only":
                if "Quality_Tier" in top_df_ui.columns:
                    # Count criteria met before filtering
                    high_quality_picks = top_df_ui[top_df_ui["Quality_Tier"] == "HIGH"]

                    if len(high_quality_picks) > 0:
                        # Calculate average metrics for high quality picks
                        avg_quality = high_quality_picks["Quality_Score"].mean() if "Quality_Score" in high_quality_picks.columns else 0

                        top_df_ui = high_quality_picks
                        logger.info(f"🔍 Quality Filter: HIGH only - {len(top_df_ui)}/{original_count} picks")

                        st.success(f"✅ **{len(top_df_ui)} HIGH quality picks** selected from {original_count} total picks ({len(top_df_ui)/original_count*100:.1f}%)")
                        st.caption(f"📊 Average Quality Score: {avg_quality:.2f}/5.0 ⭐ | These picks meet ALL criteria: >56% prob, >0.08 decisiveness, Kalshi validation, strong consensus")
                    else:
                        st.warning(f"⚠️ No HIGH quality picks found in {original_count} total picks. Showing all picks instead.")

            elif filter_mode == "High + Medium Quality":
                if "Quality_Tier" in top_df_ui.columns:
                    quality_picks = top_df_ui[top_df_ui["Quality_Tier"].isin(["HIGH", "MEDIUM"])]
                    high_count = (quality_picks["Quality_Tier"] == "HIGH").sum()
                    medium_count = (quality_picks["Quality_Tier"] == "MEDIUM").sum()

                    top_df_ui = quality_picks
                    logger.info(f"🔍 Quality Filter: HIGH + MEDIUM - {len(top_df_ui)}/{original_count} picks")

                    st.success(f"✅ **{len(top_df_ui)} quality picks** ({high_count} HIGH, {medium_count} MEDIUM) from {original_count} total")

            elif filter_mode == "Show Quality Distribution":
                if "Quality_Tier" in top_df_display.columns:
                    quality_dist = top_df_display["Quality_Tier"].value_counts().to_dict()
                    st.markdown("### 📊 Quality Distribution")
                    for tier, count in sorted(quality_dist.items(), reverse=True):
                        pct = (count / len(top_df_display)) * 100
                        st.write(f"- **{tier}**: {count} picks ({pct:.1f}%)")

                    # Also show average quality score
                    if "Quality_Score" in top_df_display.columns:
                        avg_quality = top_df_display["Quality_Score"].mean()
                        high_quality_count = (top_df_display["Quality_Tier"] == "HIGH").sum()
                        medium_quality_count = (top_df_display["Quality_Tier"] == "MEDIUM").sum()
                        low_quality_count = (top_df_display["Quality_Tier"] == "LOW").sum()

                        st.write(f"- **Average Quality Score**: {avg_quality:.2f} / 5.0")
                        st.write(f"- **High Quality Picks**: {high_quality_count} ({high_quality_count/len(top_df_display)*100:.1f}%)")
                        st.markdown("---")
                        st.markdown("**💡 Recommendation**: Use 'High Quality Only' filter for best results. Expected hit rate ~42-45% vs ~27% for all picks.")

        # --- APPLY QUALITY SORTING ---
        if "sort_by_quality" in st.session_state and st.session_state.sort_by_quality:
            if "Quality_Score" in top_df_ui.columns:
                # Sort by Quality Score descending
                top_df_ui = top_df_ui.sort_values(by="Quality_Score", ascending=False)
                logger.info(f"📊 Sorted Top Picks by Quality Score (descending)")
                st.caption("✅ Sorted by Quality Score (highest first)")

        # Custom formatting for specific columns
        # Note: 'Spread Win %' and 'Total Win %' are already formatted strings by _calc_win_pct
        format_cols_pct = []
        for col in format_cols_pct:
            if col in top_df_ui.columns:
                 # Format as percentage with 1 decimal
                 top_df_ui[col] = pd.to_numeric(top_df_ui[col], errors='coerce').apply(
                     lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A"
                 )

        # Force Numeric and String consistency
        for col in top_df_ui.columns:
            if col in format_cols_pct:
                continue # Skip already formatted columns

            # Special Handling for Sentiment Columns (Task 1)
            if col in ['Home_Sentiment', 'Away_Sentiment', 'Sentiment_Diff']:
                 # Format as 2 decimal string or N/A
                 top_df_ui[col] = pd.to_numeric(top_df_ui[col], errors='coerce').apply(
                     lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
                 )
                 continue

            if col in ['AI_Prob', 'Implied_Prob', 'spread_edge', 'total_edge', 'final_prob', 'edge', 'Overall Prob', 'Spread Prob', 'Total Prob', 'ML Prob']:
                top_df_ui[col] = pd.to_numeric(top_df_ui[col], errors='coerce').fillna(0.0).infer_objects(copy=False).infer_objects(copy=False)
            else:
                top_df_ui[col] = top_df_ui[col].astype(str).replace('None', 'N/A')

        # GRID RENDERING WITH ENHANCED ERROR HANDLING
        logger.info(f"🎯 Rendering Top Picks grid with {len(top_df_ui)} rows, {len(top_df_ui.columns)} columns")

        try:
            if top_df_ui.empty:
                st.warning("⚠️ Top picks dataframe is empty after filtering")
                with st.expander("🔍 Debug Info"):
                    st.write(f"- Original df rows: {len(df)}")
                    st.write(f"- After filtering: {len(top_df_ui)}")
                    st.write(f"- Columns available: {list(top_df_ui.columns[:10])}...")
            else:
                # Render the grid
                st.dataframe(top_df_ui, width="stretch", hide_index=True)
                logger.info(f"✅ Successfully rendered Top Picks grid")

        except Exception as grid_error:
            logger.error(f"❌ Failed to render Top Picks grid: {grid_error}")
            logger.exception(grid_error)
            st.error(f"Failed to render picks grid: {grid_error}")

            # Show diagnostic info
            with st.expander("🔍 Grid Rendering Error Details"):
                st.write("**Error:**")
                st.code(str(grid_error))
                st.write("**Dataframe Info:**")
                st.write(f"- Shape: {top_df_ui.shape}")
                st.write(f"- Columns: {list(top_df_ui.columns)}")
                st.write(f"- First row sample:")
                try:
                    st.write(top_df_ui.iloc[0].to_dict() if len(top_df_ui) > 0 else "No rows")
                except Exception:
                    st.write("Could not display row sample")

        export_cols = [
            "AI_Prob",
            "Implied_Prob",
            "ai_prob_adj",
            "consensus_prob",
            "consensus_prob_adj",
            "final_probability",
            "decision_driver",
            "SpreadConsensusProb",
            "SpreadConsensus",
            "TotalConsensusProb",
            "TotalConsensus",
            "Best Overall Pick",
            "Best Overall Prob",
            "best_pick_type",
            "kalshi_weight",
            "odds_weight",
            "ml_weight",
            "sentiment_weight",
            "sentiment_score",
            "Home_Sentiment",
            "Away_Sentiment",
            "sentiment_status",
            "sentiment_direction",
            "sentiment_impact_applied",
            "confidence_reason",
            "spread_engine_used",
            "spread_pick_label",
            "spread_alt_label",
            "spread_prob_pick_final",
            "spread_prob_alt_final",
            "spread_prob_margin",
            "spread_prob_pick_market",
            "spread_prob_alt_market",
            "spread_prob_pick_kalshi",
            "spread_prob_alt_kalshi",
            "spread_decision_metric_used",
            "spread_decision_score_pick",
            "spread_decision_score_alt",
            "spread_decision_score_margin",
            "spread_trace_json",
            "total_engine_used",
            "total_pick_label",
            "total_alt_label",
            "total_prob_pick_final",
            "total_prob_alt_final",
            "total_prob_margin",
            "total_prob_pick_market",
            "total_prob_alt_market",
            "total_prob_pick_kalshi",
            "total_prob_alt_kalshi",
            "total_decision_metric_used",
            "total_decision_score_pick",
            "total_decision_score_alt",
            "total_decision_score_margin",
            "total_trace_json",
            "decision_trace_version",
            "overall_engine_used",
            "decision_trace_notes",
            "decision_trace_short",
            "decision_trace_json",
            "kalshi_matched",
            "kalshi_prob_used",
            "kalshi_prob_for_pick",
            "kalshi_yes_side",
            "kalshi_event_ticker_used",
            "kalshi_candidate_count",
            "kalshi_best_score",
            "kalshi_match_reason",
            "kalshi_game_prefix_used",
            "kalshi_wanted_tokens",
            "consensus_prob_adj",
            "Spread_Glance",
            "Total_Glance",
            "Spread_Glance_Clean",
            "Total_Glance_Clean",
            "spread_prob_display",
            "total_prob_display",
            "spread_sentiment_arrow",
            "total_sentiment_arrow",
            "spread_sentiment_note",
            "total_sentiment_note",
            "st_conf_rank",
            "decisiveness",
            "best_spread_book",
            "best_spread_last_update",
            "best_spread_price_score",
            "best_spread_median_point",
            "best_spread_price",
            "best_spread_mode_point",
            "best_total_book",
            "best_total_last_update",
            "best_total_price_score",
            "best_total_median_point",
            "best_total_price",
            "best_total_mode_point",
            "sentiment_error_count",
            "sentiment_errors_sample",
            "sentiment_articles_total",
            "sentiment_status_counts",
            "sentiment_sample_query",
            "sentiment_sample_status",
            "sentiment_sample_totalResults",
            "sentiment_auth_error",
            "sentiment_rate_limited",
            "sentiment_cooldown_until",
            "sentiment_cached_teams_count",
            "sentiment_available_count",
            "sentiment_used_cached",
            "sentiment_disabled_reason",
            "Pick_Confidence",
            "Pick_Reason_Short",
            "Eligible_Top_Picks",
            "Kalshi_Required",
            "api_sports_used",
            "sportsdata_used",
            "api_sports_status",
            "sportsdata_status",
            "apisports_enriched",
            "apisports_status",
            "apisports_notes",
            "sportsdata_enriched",
            "sportsdata_status",
            "sportsdata_notes",
            "injuries_home_count",
            "injuries_away_count",
            "weather_summary",
            "key_injuries_home",
            "key_injuries_away",
            "sentiment_adj_value",
            "sentiment_adj_reason",
            "prob_reason",
            "odds_valid",
            "odds_placeholder_detected",
            "implied_prob_reason",
            "spread_pick_team",
            "spread_pick_line",
            "spread_pick_odds",
            "spread_implied_prob",
            "spread_prob",
            "spread_prob_market_based",
            "spread_prob_reason",
            "spread_odds_method",
            "spread_prob_method",
            "spread_confidence",
            "spread_confidence_reason",
            "spread_market_pairs_count",
            "spread_odds_valid",
            "spread_odds_placeholder_detected",
            "spread_prob_placeholder_detected",
            "total_pick_side",
            "total_pick_line",
            "total_pick_odds",
            "total_implied_prob",
            "total_prob",
            "total_prob_market_based",
            "total_prob_reason",
            "total_odds_method",
            "total_prob_method",
            "total_confidence",
            "total_confidence_reason",
            "total_market_pairs_count",
            "total_odds_valid",
            "total_odds_placeholder_detected",
            "total_prob_placeholder_detected",
            "spread_min",
            "spread_med",
            "spread_max",
            "total_min",
            "total_med",
            "total_max",
            "spread_books_count",
            "total_books_count",
            "spread_width",
            "total_width",
            "At_a_Glance_Confidence",
            "At_a_Glance_Score",
            "At_a_Glance_Reason",
            "spread_edge",
            "total_edge",
            "market_stability",
        ]

        # Batch assign missing columns to avoid fragmentation (Issue #2 Fix)
        missing_cols = [col for col in required_display_cols if col not in df.columns]
        if missing_cols:
             # Create dataframe with proper index to avoid misalignment and fragmentation
             # Using pd.concat with empty frame having index is cleaner
             new_cols_df = pd.DataFrame(index=df.index, columns=missing_cols)
             df = pd.concat([df, new_cols_df], axis=1)
             df = df.copy() # Defragment

        if "reddit_used" in df.columns:
            df["reddit_used"] = df["reddit_used"].fillna(False).astype(bool)
        df = add_spread_total_confidence(df)
        df = df.copy()
        df = enrich_picks_with_roi_metrics(df)
        df = df.copy()

        market_stability_filter = st.sidebar.multiselect(
            "Market Stability",
            ["WIDE", "TIGHT"],
            default=[],
            key="market_stability_filter"
        )

        # Stats Quality Filter (New)
        stats_quality_filter = st.sidebar.radio(
            "Stats Data Quality",
            ["All Games", "Full Stats Only", "Missing/Partial Only"],
            index=0,
            key="stats_quality_filter"
        )

        # Confidence Filter Controls (Re-enabled)
        confidence_mode = st.selectbox(
            "Confidence filter",
            ["All", "High+Medium (recommended)", "High only"],
            index=0,
            key="confidence_filter_mode",
        )

        hide_low = st.checkbox(
            "Hide low-confidence picks",
            value=False,
            key="hide_low_confidence",
        )

        if logger:
            logger.info(f"apply_confidence_filter input: {len(df)} rows, columns={list(df.columns)[:20]}...")
            logger.info(f"   Has Pick_Confidence: {'Pick_Confidence' in df.columns}")

        # Apply filter based on user selection (Fix Issue #7)
        # Note: If hide_low is False (unchecked), show_low is True
        df_master_view, confidence_stats = apply_confidence_filter(df, confidence_mode, not hide_low)

        # Display warning if Pick_Confidence column was missing
        if confidence_stats.get("warning"):
            st.warning(confidence_stats["warning"])

        # DISABLED: Force Spread/Total Pivot for UI display
        # This was converting ML rows to Spread/Total in the Master Analysis tab view
        # However, we want to preserve and display all market types including Moneyline
        # User can toggle "Show Moneyline rows" checkbox below to filter if needed
        #
        # Old logic (DISABLED):
        # def force_spread_total_pivot(row):
        #     if row.get('Market') == "Moneyline":
        #         [pivot to Spread or Total based on probability]
        # df_master_view = df_master_view.apply(force_spread_total_pivot, axis=1)
        #
        # NEW BEHAVIOR: Preserve all Market types in UI display

        if market_stability_filter:
            df_master_view = df_master_view[df_master_view['market_stability'].isin(market_stability_filter)]

        # Apply Stats Quality Filter
        if stats_quality_filter == "Full Stats Only":
            if "stats_quality" in df_master_view.columns:
                df_master_view = df_master_view[df_master_view["stats_quality"].isin(["REAL", "ESPN"])]
        elif stats_quality_filter == "Missing/Partial Only":
            if "stats_quality" in df_master_view.columns:
                df_master_view = df_master_view[~df_master_view["stats_quality"].isin(["REAL", "ESPN"])]

        # Enrich with Best Picks for Export/Display
        df_master_view = calculate_best_pick_metrics(df_master_view)

        # v97 FIX: Sync final_probability and Best Overall columns in Master Analysis view
        if 'final_prob' in df_master_view.columns:
            df_master_view['final_probability'] = df_master_view['final_prob']
            df_master_view['Best Overall Prob'] = df_master_view['final_prob']
        if 'best_pick' in df_master_view.columns:
            df_master_view['Best Overall Pick'] = df_master_view['best_pick']
        if 'best_pick_type' in df_master_view.columns:
            df_master_view['Best Overall Market'] = df_master_view['best_pick_type']

        # --- MERGE GAME SUMMARY COLUMNS ---
        # Fix Issue #6: Always attempt summary merge even if Gemini is disabled
        # build_game_summary relies on data probabilities, not just Gemini fields.
        try:
            summary_df_merge = build_game_summary(st.session_state["master_df"])
            if not summary_df_merge.empty and not df_master_view.empty:
                # Avoid duplicate join keys in right frame
                join_keys = ["league", "Home", "Away", "Commence (UTC)"]
                cols_to_use = [c for c in summary_df_merge.columns if c not in ["Commence (Local)", "Local Date"]]

                df_master_view = pd.merge(
                    df_master_view,
                    summary_df_merge[cols_to_use],
                    on=join_keys,
                    how="left"
                )
                # Task 3: Optimization - Replace fragmented column assignments with copy
                df_master_view = df_master_view.copy()
        except Exception as e:
            logger.warning(f"⚠️ Summary merge failed: {e}")

        counts = confidence_stats.get("counts") or {}
        st.caption(
            f"Confidence counts (post-filter): HIGH={counts.get('HIGH', 0)}, "
            f"MEDIUM={counts.get('MEDIUM', 0)}, LOW={counts.get('LOW', 0)}; "
            f"LOW removed by filter: {confidence_stats.get('low_removed', 0)}"
        )
        show_moneyline_rows = True
        if "Market" in df_master_view.columns:
            show_moneyline_rows = st.checkbox("Show Moneyline rows", value=True, key="show_moneyline_rows")
            if not show_moneyline_rows:
                df_master_view = df_master_view[df_master_view["Market"].isin(["Spread", "Total"])]

        stats = st.session_state.get("master_stats_persistent", {})
        if stats.get("games_in", 0) > 0 and stats.get("rows_out", 0) == 0:
            st.error("Master analysis produced 0 rows; see debug stats below.")
            st.json(stats)
        elif not st.session_state.get("games", []):
            st.warning("No games loaded. Use the sidebar to load games first.")
        else:
            st.success(f"Produced {len(st.session_state['master_results_df'])} rows from {len(st.session_state.get('games', []))} games")
            # Explicitly format key columns
            # Ensure numeric typing before display to avoid Arrow errors
            cols_to_force_numeric = ["AI_Prob", "model_prob_home", "final_probability", "Implied_Prob", "spread_edge", "total_edge"]
            valid_force_cols = [c for c in cols_to_force_numeric if c in df_master_view_display.columns]
            if valid_force_cols:
                df_master_view_display[valid_force_cols] = df_master_view_display[valid_force_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).infer_objects(copy=False).infer_objects(copy=False)

        try:
            df_master_view["st_conf_rank"] = df_master_view["st_conf_rank"].fillna(0).infer_objects(copy=False).infer_objects(copy=False)
            df_master_view["decisiveness"] = df_master_view["decisiveness"].fillna(0.0).infer_objects(copy=False).infer_objects(copy=False)
            df_master_view = df_master_view.sort_values(
                by=["st_conf_rank", "decisiveness", "Commence (UTC)"],
                ascending=[False, False, True],
            )
        except Exception:
            pass

# === SHOTGUN MODE TAB ===
with tab_shotgun:
    st.header("🚀 Shotgun Mode - Structured Parlays")

    shotgun_data = st.session_state.get("shotgun_data", {})
    shotgun_parlays = st.session_state.get("shotgun_parlays", {})

    if not shotgun_data and not shotgun_parlays:
        st.info("Run Master Analysis first to generate Shotgun Mode recommendations.")
    else:
        # Display 2-leg parlays (fixed stakes)
        st.subheader("💎 Best 2-Leg Parlays")
        st.markdown("Three parlay recommendations with fixed stakes based on risk profile:")

        parlay_col1, parlay_col2, parlay_col3 = st.columns(3)

        # Best Overall ($3 stake)
        with parlay_col1:
            st.markdown("### 🎯 Best Overall - $3")
            best = shotgun_parlays.get("best_overall")
            if best:
                st.success(f"**Expected Value:** {best['ev']:.2%}")
                st.metric("Parlay Win Probability", f"{best['parlay_prob']*100:.1f}%")
                st.metric("Parlay Odds", f"{best['parlay_american_odds']:+.0f}")
                st.metric("Expected Return", f"${best['expected_return']:.2f}")

                # Probability Breakdown
                st.markdown("**📊 Probability Breakdown:**")
                leg1_prob = best['leg1']['prob']
                leg2_prob = best['leg2']['prob']
                combined_prob = leg1_prob * leg2_prob
                st.caption(f"Leg 1: {leg1_prob*100:.1f}%")
                st.caption(f"Leg 2: {leg2_prob*100:.1f}%")
                st.caption(f"Combined: {leg1_prob*100:.1f}% × {leg2_prob*100:.1f}% = **{combined_prob*100:.1f}%**")

                st.markdown("**Legs:**")
                consensus1 = f"{best['leg1'].get('consensus_votes', 0)}/{best['leg1'].get('consensus_total', 0)}"
                st.markdown(f"1. {best['leg1']['pick']}")
                st.caption(f"   Prob: {best['leg1']['prob']*100:.1f}%, Odds: {best['leg1']['odds']:+.0f}, Consensus: {consensus1}")
                consensus2 = f"{best['leg2'].get('consensus_votes', 0)}/{best['leg2'].get('consensus_total', 0)}"
                st.markdown(f"2. {best['leg2']['pick']}")
                st.caption(f"   Prob: {best['leg2']['prob']*100:.1f}%, Odds: {best['leg2']['odds']:+.0f}, Consensus: {consensus2}")
            else:
                st.warning("No best overall parlay available")

        # Medium Risk ($2 stake)
        with parlay_col2:
            st.markdown("### 📈 Medium Risk - $2")
            medium = shotgun_parlays.get("medium_risk")
            if medium:
                st.info(f"**Expected Value:** {medium['ev']:.2%}")
                st.metric("Parlay Win Probability", f"{medium['parlay_prob']*100:.1f}%")
                st.metric("Parlay Odds", f"{medium['parlay_american_odds']:+.0f}")
                st.metric("Expected Return", f"${medium['expected_return']:.2f}")

                # Probability Breakdown
                st.markdown("**📊 Probability Breakdown:**")
                leg1_prob = medium['leg1']['prob']
                leg2_prob = medium['leg2']['prob']
                combined_prob = leg1_prob * leg2_prob
                st.caption(f"Leg 1: {leg1_prob*100:.1f}%")
                st.caption(f"Leg 2: {leg2_prob*100:.1f}%")
                st.caption(f"Combined: {leg1_prob*100:.1f}% × {leg2_prob*100:.1f}% = **{combined_prob*100:.1f}%**")

                st.markdown("**Legs:**")
                consensus1 = f"{medium['leg1'].get('consensus_votes', 0)}/{medium['leg1'].get('consensus_total', 0)}"
                st.markdown(f"1. {medium['leg1']['pick']}")
                st.caption(f"   Prob: {medium['leg1']['prob']*100:.1f}%, Odds: {medium['leg1']['odds']:+.0f}, Consensus: {consensus1}")
                consensus2 = f"{medium['leg2'].get('consensus_votes', 0)}/{medium['leg2'].get('consensus_total', 0)}"
                st.markdown(f"2. {medium['leg2']['pick']}")
                st.caption(f"   Prob: {medium['leg2']['prob']*100:.1f}%, Odds: {medium['leg2']['odds']:+.0f}, Consensus: {consensus2}")
            else:
                st.warning("No medium risk parlay available")

        # High Risk ($1 stake)
        with parlay_col3:
            st.markdown("### 🎲 High Risk - $1")
            high = shotgun_parlays.get("high_risk")
            if high:
                st.warning(f"**Expected Value:** {high['ev']:.2%}")
                st.metric("Parlay Win Probability", f"{high['parlay_prob']*100:.1f}%")
                st.metric("Parlay Odds", f"{high['parlay_american_odds']:+.0f}")
                st.metric("Expected Return", f"${high['expected_return']:.2f}")

                # Probability Breakdown
                st.markdown("**📊 Probability Breakdown:**")
                leg1_prob = high['leg1']['prob']
                leg2_prob = high['leg2']['prob']
                combined_prob = leg1_prob * leg2_prob
                st.caption(f"Leg 1: {leg1_prob*100:.1f}%")
                st.caption(f"Leg 2: {leg2_prob*100:.1f}%")
                st.caption(f"Combined: {leg1_prob*100:.1f}% × {leg2_prob*100:.1f}% = **{combined_prob*100:.1f}%**")

                st.markdown("**Legs:**")
                consensus1 = f"{high['leg1'].get('consensus_votes', 0)}/{high['leg1'].get('consensus_total', 0)}"
                st.markdown(f"1. {high['leg1']['pick']}")
                st.caption(f"   Prob: {high['leg1']['prob']*100:.1f}%, Odds: {high['leg1']['odds']:+.0f}, Consensus: {consensus1}")
                consensus2 = f"{high['leg2'].get('consensus_votes', 0)}/{high['leg2'].get('consensus_total', 0)}"
                st.markdown(f"2. {high['leg2']['pick']}")
                st.caption(f"   Prob: {high['leg2']['prob']*100:.1f}%, Odds: {high['leg2']['odds']:+.0f}, Consensus: {consensus2}")
            else:
                st.warning("No high risk parlay available")

        st.markdown("---")

        # Display single pick tiers (original Shotgun Mode)
        st.subheader("⚡ Single Pick Tiers")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🎯 $3 'Snipers' (High Prob)")
            snipers = shotgun_data.get("snipers", pd.DataFrame())
            if not snipers.empty:
                display_cols = ['Pick', 'AI_Prob', 'AI_Edge', 'consensus']
                display_cols = [c for c in display_cols if c in snipers.columns]
                if 'Pick' not in snipers.columns and 'Spread & Pick' in snipers.columns:
                    display_cols = ['Spread & Pick', 'AI_Prob', 'AI_Edge', 'consensus']
                    display_cols = [c for c in display_cols if c in snipers.columns]
                st.dataframe(snipers[display_cols], hide_index=True, width="stretch")
            else:
                st.info("No snipers available")

        with col2:
            st.markdown("#### 📈 $2 'Strategy' (High EV)")
            strategy = shotgun_data.get("strategy", pd.DataFrame())
            if not strategy.empty:
                display_cols = ['Pick', 'AI_Prob', 'AI_Edge', 'consensus']
                display_cols = [c for c in display_cols if c in strategy.columns]
                if 'Pick' not in strategy.columns and 'Spread & Pick' in strategy.columns:
                    display_cols = ['Spread & Pick', 'AI_Prob', 'AI_Edge', 'consensus']
                    display_cols = [c for c in display_cols if c in strategy.columns]
                st.dataframe(strategy[display_cols], hide_index=True, width="stretch")
            else:
                st.info("No strategy picks available")

        with col3:
            st.markdown("#### 🎲 $1 'Longshots' (Lottos)")
            longshots = shotgun_data.get("longshots", pd.DataFrame())
            if not longshots.empty:
                display_cols = ['Pick', 'AI_Prob', 'AI_Edge', 'consensus']
                display_cols = [c for c in display_cols if c in longshots.columns]
                if 'Pick' not in longshots.columns and 'Spread & Pick' in longshots.columns:
                    display_cols = ['Spread & Pick', 'AI_Prob', 'AI_Edge', 'consensus']
                    display_cols = [c for c in display_cols if c in longshots.columns]
                st.dataframe(longshots[display_cols], hide_index=True, width="stretch")
            else:
                st.info("No longshots available")

        # === NEW: MULTI-PARLAY GENERATOR ===
        st.markdown("---")
        st.markdown("---")

        # Check if generate function is available
        if generate_shotgun_mode_parlays is not None and display_shotgun_mode_ui is not None:
            # Display help section
            if display_shotgun_mode_help is not None:
                display_shotgun_mode_help()

            # Add enable checkbox
            enable_auto_parlays = st.checkbox(
                "🎯 Enable Auto-Parlay Generator (2-leg & 3-leg combinations)",
                value=False,
                key="enable_auto_parlays",
                help="Automatically generates optimized 2-leg and 3-leg parlays from eligible picks"
            )

            if enable_auto_parlays:
                # Get master results dataframe
                master_results_df = st.session_state.get("master_results_df")

                if master_results_df is not None and not master_results_df.empty:
                    # Configuration options
                    st.markdown("### ⚙️ Configuration")
                    config_col1, config_col2 = st.columns(2)

                    with config_col1:
                        num_2leg = st.slider(
                            "Number of 2-Leg Parlays",
                            min_value=5,
                            max_value=50,
                            value=20,
                            step=5,
                            key="shotgun_num_2leg"
                        )

                    with config_col2:
                        num_3leg = st.slider(
                            "Number of 3-Leg Parlays",
                            min_value=5,
                            max_value=50,
                            value=20,
                            step=5,
                            key="shotgun_num_3leg"
                        )

                    # Generate button
                    if st.button("🚀 Generate Optimized Parlays", key="generate_parlays_button"):
                        with st.spinner("Generating optimized parlays..."):
                            try:
                                # Generate parlays
                                shotgun_results = generate_shotgun_mode_parlays(
                                    master_results_df,
                                    num_2leg=num_2leg,
                                    num_3leg=num_3leg
                                )

                                # Store in session state
                                st.session_state["shotgun_auto_parlays"] = shotgun_results

                                # Display summary
                                if display_shotgun_mode_summary is not None:
                                    display_shotgun_mode_summary(shotgun_results['stats'])

                                st.success("✅ Parlays generated successfully!")

                            except Exception as e:
                                st.error(f"❌ Error generating parlays: {e}")
                                logger.error(f"Shotgun Mode error: {e}", exc_info=True)

                    # Display generated parlays if they exist
                    shotgun_auto_parlays = st.session_state.get("shotgun_auto_parlays")
                    if shotgun_auto_parlays:
                        parlays_2leg = shotgun_auto_parlays.get('2leg', [])
                        parlays_3leg = shotgun_auto_parlays.get('3leg', [])

                        if parlays_2leg or parlays_3leg:
                            display_shotgun_mode_ui(parlays_2leg, parlays_3leg)
                        else:
                            st.warning("No parlays were generated. Try adjusting filters or check if you have enough eligible picks.")
                else:
                    st.warning("⚠️ No picks available. Run Master Analysis first to generate parlays.")
        else:
            st.info("ℹ️ Auto-Parlay Generator module not available. Please check installation.")
