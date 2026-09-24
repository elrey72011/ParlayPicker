Warning: truncated output (original token count: 150902)
Total output lines: 11239

from __future__ import annotations

import functools
import re
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Optional
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bankroll_simulator import simulate_bankroll
from core.kelly_optimizer import add_kelly_bet_sizing
from app_core.calibration import generate_calibration_dataset
from core.probability_engine import american_to_prob
from core.schema.base_schema import ensure_base_schema
from core.team_mapper import normalize_team_name, NBA_EXACT_MAP, WNBA_EXACT_MAP, NHL_EXACT_MAP
from app_core.weights_config import (
            TOTAL_UNDER_MIN_WIN_PROB, TOTAL_UNDER_MIN_EV, TOTAL_UNDER_MIN_EDGE,
            NHL_TOTAL_EXTRA_EDGE_PENALTY, MLB_SPREAD_MIN_WIN_PROB,
            MLB_SPREAD_ACTIONABLE_PENALTY, MLB_SPREAD_FINALIST_SCORE_PENALTY,
            MLB_SPREAD_FINALIST_PENALTY_MIN_FAMILY_N,
            MLB_SPREAD_FINALIST_PENALTY_MIN_RATE_GAP,
            WNBA_UNDER_FINALIST_SCORE_PENALTY,
            WNBA_UNDER_FINALIST_PENALTY_MIN_N,
            WNBA_UNDER_FINALIST_PENALTY_MAX_WIN_RATE,
            BEST_AVAILABLE_QUALIFIED_MIN_WIN_PROB,
            BEST_AVAILABLE_QUALIFIED_MIN_EV,
            BEST_AVAILABLE_QUALIFIED_MIN_EDGE,
            NBA_SIDE_ACTIONABLE_BONUS, NBA_OVER_ACTIONABLE_BONUS,
            MLB_OVER_ACTIONABLE_MIN_PROB, MLB_OVER_ACTIONABLE_MIN_EV, MLB_OVER_ACTIONABLE_MIN_EDGE,
            MLB_TOTAL_OVER_ACTIONABLE_PENALTY, MLB_TOTAL_UNDER_ACTIONABLE_PENALTY,
            NBA_TOTAL_OVER_ACTIONABLE_PENALTY, NBA_TOTAL_UNDER_ACTIONABLE_PENALTY,
            NHL_TOTAL_OVER_ACTIONABLE_PENALTY, NHL_TOTAL_UNDER_ACTIONABLE_PENALTY,
            NO_KALSHI_TOTAL_EXTRA_PENALTY, NO_KALSHI_TOTAL_UNDER_EXTRA_PENALTY,
            TOTAL_UNDER_FINALIST_SCORE_PENALTY,
            LEAGUE_MARKET_FAMILY_ACTIONABLE_PENALTIES,
            FALLBACK_HEAVY_TOTAL_EXTRA_PENALTY,
            BEST_PICKS_PROFILE,
            MAX_TOTAL_OVER_ACTIONABLE_SHARE,
            MAX_TOTAL_OVER_ACTIONABLE_COUNT,
            MAX_MLB_TOTAL_OVER_ACTIONABLE_COUNT,
            MAX_TOTAL_UNDER_ACTIONABLE_COUNT,
            MAX_MLB_TOTAL_UNDER_ACTIONABLE_COUNT,
            MAX_TOTAL_OVER_HIGH_VARIANCE_COUNT,
            MAX_MLB_TOTAL_OVER_HIGH_VARIANCE_COUNT,
            TOTAL_OVER_PROB_SHRINK,
            MLB_TOTAL_OVER_PROB_SHRINK,
            MLB_TOTAL_OVER_MIN_PRODUCTION_WIN_PROB,
            MLB_TOTAL_OVER_MIN_PRODUCTION_EV,
            MLB_TOTAL_OVER_MIN_PRODUCTION_EDGE,
            MLB_OVER_CALIBRATED_PROB_CAP,
            DEGRADED_FEATURE_KELLY_MULTIPLIER,
            DEGRADED_FEATURE_MAX_SLATE_EXPOSURE_PCT,
            DEGRADED_FEATURE_MAX_PICK_EXPOSURE_PCT,
    LOCK_UPLOAD_LINES_FOR_MATCHED_ROWS,
    ALLOW_UPLOAD_TOTAL_FALLBACK_ACTIONABLE,
    KALSHI_WEIGHT, MARKET_WEIGHT, ML_MODEL_WEIGHT, THEOVER_WEIGHT, SENTIMENT_WEIGHT,
    FALLBACK_MARKET_WEIGHT, FALLBACK_ML_WEIGHT, FALLBACK_THEOVER_WEIGHT, FALLBACK_SENTIMENT_WEIGHT,
    LOW_LIQUIDITY_KALSHI_WEIGHT, LOW_LIQUIDITY_ML_MODEL_WEIGHT,
    MLB_TOTAL_THEOVER_WEIGHT, MLB_TOTAL_ML_WEIGHT,
    MLB_TOTAL_MARKET_WEIGHT, MLB_TOTAL_KALSHI_WEIGHT,
    MLB_TOTAL_FALLBACK_THEOVER_WEIGHT, MLB_TOTAL_FALLBACK_ML_WEIGHT,
    MLB_TOTAL_FALLBACK_MARKET_WEIGHT,
    NBA_TOTAL_THEOVER_WEIGHT, NBA_TOTAL_ML_WEIGHT,
    NBA_TOTAL_FALLBACK_THEOVER_WEIGHT, NBA_TOTAL_FALLBACK_ML_WEIGHT,
    WNBA_KALSHI_WEIGHT, WNBA_MARKET_WEIGHT, WNBA_ML_MODEL_WEIGHT, WNBA_THEOVER_WEIGHT,
    WNBA_FALLBACK_MARKET_WEIGHT, WNBA_FALLBACK_ML_WEIGHT, WNBA_FALLBACK_THEOVER_WEIGHT,
    NHL_KALSHI_WEIGHT, NHL_ML_MODEL_WEIGHT, NHL_MARKET_WEIGHT, NHL_THEOVER_WEIGHT, NHL_SENTIMENT_WEIGHT,
    KALSHI_DIVERGENCE_THRESHOLD, KALSHI_DIVERGENCE_THRESHOLD_NBA,
    KALSHI_DIVERGENCE_THRESHOLD_MLB, KALSHI_DIVERGENCE_THRESHOLD_NHL,
    MLB_THEOVER_CONFLICT_THRESHOLD, MLB_THEOVER_CONFLICT_PENALTY,
    KALSHI_DIRECTION_CONFIDENCE_WEIGHT,
    KALSHI_DIRECTION_VETO_PENALTY, KALSHI_DIRECTION_VETO_MIN_CONVICTION,
    MLB_THEOVER_FADE_SOURCES, MLB_THEOVER_FADE_SHRINK, THEOVER_FADE_SHRINK_DEFAULT,
    MLB_PUBLIC_BETTING_FADE_SOURCES, MLB_PUBLIC_BETTING_FADE_STRENGTH,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

logger = logging.getLogger(__name__)

# Import odds fetching components
try:
    from app_core.odds_api import TheOddsAPIClient, filter_games_today_only, OddsAPIAuthError
    ODDS_API_AVAILABLE = True
except Exception as e:
    ODDS_API_AVAILABLE = False
    logger.warning(f"Could not import TheOddsAPIClient: {e}")

import os
import streamlit as st

try:
    from thefuzz import fuzz as thefuzz_fuzz
except Exception:  # pragma: no cover - optional dependency fallback
    try:
        from rapidfuzz import fuzz as rapidfuzz_fuzz
    except Exception:  # pragma: no cover
        rapidfuzz_fuzz = None
    thefuzz_fuzz = None

def _get_odds_api_key() -> str:
    """Return the odds API key without requiring a Streamlit secrets file."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("ODDS_API_KEY", "") or "").strip()
    except Exception:
        # Streamlit raises when no secrets.toml exists. Headless runs should
        # degrade to an unauthenticated pipeline instead of crashing here.
        return ""

try:
    from app_core.prediction_engine import PredictionEngine, get_cached_prediction_engine
    ML_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import PredictionEngine: {e}")
    ML_AVAILABLE = False
    PredictionEngine = None

VALID_MARKETS = {"spread_home", "spread_away", "total_over", "total_under", "moneyline_home", "moneyline_away"}
DATE_ALIASES = ["game_date", "game_date_est", "commence_time", "start_time", "time", "date", "event_date"]
LEAGUE_ALIASES = {
    "NCAAM": "NCAAB",
    "NCAA MEN'S BASKETBALL": "NCAAB",
    "NCAA MENS BASKETBALL": "NCAAB",
    "AMERICANFOOTBALL_NCAAF": "NCAAF",
    "NCAA FOOTBALL": "NCAAF",
    "COLLEGE FOOTBALL": "NCAAF",
    "CFB": "NCAAF",
}
_KNOWN_NCAAB_TEAM_TOKENS = {
    "wichita st", "wichita state", "oklahoma st", "oklahoma state", "davidson",
}
_NCAAB_TEAM_KEYWORD_HINTS = {
    "st", "state", "university", "redhawks", "tommies", "cowboys",
}
_NCAAB_LEAGUE_RECOVERY_KEYWORDS = {
    "st", "state", "univ", "university",
    "cowboys", "bulldogs", "redhawks", "tommies", "golden hurricane", "wildcats", "shockers", "unlv",
    "lehigh", "navy", "revolutionaries", "uic", "panthers", "bradley", "dayton", "murray", "saint josephs",
    "valley", "uvu", "george washington", "gw", "billikens",
}
_COLLEGE_SOURCE_HINTS = {"college", "ncaa", "ncaab", "ncaam", "mens basketball", "women\'s basketball"}


# Build stamp emitted on every exported pick. Bump this string with any change that
# should be observable in the export so a deployed app's code version is unambiguous:
# if PIPELINE_BUILD in the export doesn't match the latest value, the running app is
# serving stale code (e.g. a Streamlit deploy that didn't advance to the new commit).
PIPELINE_BUILD = "2026-09-21-nfl-context-v1"

# Best Available must compare standard, reasonably priced markets. A P2P exchange can
# expose alternate run lines (for example +5.5 at -1150) beside the standard MLB +1.5.
# Those quotes are valid exchange outcomes, but they are not comparable candidates for
# a standard spread picker and must never win merely because their hit probability is high.
NOVIG_MLB_SPREAD_OUTLIER_TOL = 0.5
BEST_AVAILABLE_SPREAD_MIN_AMERICAN_ODDS = -400.0


def _is_nonproduction_odds_source(value: object) -> bool:
    """Return True for price sources that may be ranked but never funded."""

    source = "" if value is None else str(value).strip().lower()
    return "fallback_novig" in source or "espn_draftkings_fallback" in source


REQUIRED_BEST_PICK_EXPORT_COLUMNS = [
    "pipeline_build",
    "odds_feed_source",
    "status_metric_basis",
    "selection_probability_used",
    "selection_probability_source",
    "selection_probability_pair_normalized",
    "mlb_spread_finalist_penalty_applied",
    "mlb_spread_finalist_penalty_value",
    "mlb_spread_finalist_penalty_reason",
    "recent_regime_penalty_applied",
    "recent_regime_penalty_value",
    "recent_regime_penalty_reason",
    "recent_regime_bucket",
    "recent_regime_bucket_n",
    "recent_regime_bucket_win_rate",
    "recent_regime_long_win_rate",
    "best_available_value_override_applied",
    "best_available_value_override_from_pick",
    "best_available_value_override_ev_gain",
    "best_available_rank",
    "best_available_family_rank",
    "best_available_score",
    "best_available_runner_up_pick",
    "best_available_runner_up_market_type",
    "best_available_runner_up_score",
    "best_available_score_gap",
    "best_available_candidate_count",
    "best_available_selection_verified",
    "best_available_ranking_verified",
    "final_pick_valid",
    "final_pick_valid_reason",
    "best_available_selection_reason",
    "qualified_pick",
    "qualification_probability",
    "qualification_reason",
    "display_pick",
    "commercial_tier",
    "sellable_as_premium",
    "sellable_as_value_card",
    "controlled_card_recovery",
    "best_available_only",
    "commercial_reason",
    "wager_approved",
    "export_role",
    "wager_instruction",
    "effective_expected_value",
    "effective_edge",
    "effective_win_probability",
    "status_blocker_reason",
    "status_blocker_stage",
    "nba_stats_fetch_status",
    "fallback_summary_by_league",
    "run_health_warning",
    "degraded_feature_subset_flag",
    "degraded_feature_subset_reason",
    "kelly_uncalibrated_probability",
    "kelly_probability_used",
    "kelly_probability_source",
    "actionable_family_counts",
    "totals_only_actionable_flag",
    "viable_side_candidates_count",
    "side_promoted_by_balance_guard_count",
    "side_balance_guard_reason",
    "market_line_used",
    "market_line_source",
    "market_line_source_detail",
    "matched_live_spread_line",
    "matched_live_total_line",
    "upload_spread_line",
    "upload_total_line",
    "base_spread_line",
    "base_total_line",
    "line_consistency_flag",
    "line_consistency_reason",
    "line_provenance_warning",
    "line_event_identity_match_flag",
    "line_event_identity_reason",
    "live_event_match_key",
    "line_candidate_count",
    "selected_live_event_source",
    "export_run_id",
    "pick_id",
    "canonical_pick_key",
    # Signal-transparency + backtest columns. These carry the exact, pick-side-
    # oriented inputs the blend consumed so scripts/fit_blend_weights.py can fit
    # weights from saved exports. Without them the download is a curated subset
    # and the fitting data never reaches the file the user saves.
    "theover_probability",
    "ml_probability",
    "ml_probability_source",
    "ml_target",
    "ml_projection",
    "ml_residual_scale",
    "ml_feature_quality", "ml_unavailable_reason",
    # TheOver WinProbSource tag, surfaced for transparency + as a deploy/version
    # signal: if this column is absent or all-NaN in an export, the running app is
    # not on the build that gates untrusted MLB-total direction sources.
    "win_prob_source",
    "display_probability",
    "blend_in_kalshi",
    "blend_in_market",
    "blend_in_ml",
    "blend_in_theover",
    "blend_tier",
    # One readable string of every signal feeding the blend, each as its own win %
    # (e.g. "Kalshi 35% | Market 46% | ML 74% | TheOver 75%"). Derived from the
    # blend_in_* columns above; absent signals are omitted.
    "signal_breakdown",
    # Kalshi match instrumentation â€” diagnose a systematic over-bias from the export:
    # the Kalshi contract line used, its distance from the pick line, and the raw
    # P(over) before pick-side orientation / proxy decay.
    "kalshi_matched_line",
    "kalshi_line_diff",
    "kalshi_raw_over_prob",
    # Raw Kalshi contract fields â€” confirm strike semantics vs our matched_line, and
    # let the YES bid/ask reveal any de-vig issue, straight from the export.
    "kalshi_market_title",
    "kalshi_floor_strike",
    "kalshi_cap_strike",
    "kalshi_yes_bid",
    "kalshi_yes_ask",
    # Raw per-book spread points + moneyline prices, verbatim from the live feed
    # (e.g. "novig: sp H=-1.5/A=+1.5 ml H=-120/A=+115 | fanduel: ..."). Diagnostic for
    # flipped-orientation cases: when a spread_away sign disagrees with the moneyline
    # favorite, this shows whether the feed delivered the spread wrong-signed/swapped at
    # the source (which the away-mirror derivation cannot repair) vs a parse bug.
    "raw_book_odds_diag",
]


def _normalize_pick_identity_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _build_canonical_pick_key(row: pd.Series) -> str:
    league = _normalize_pick_identity_text(row.get("league", ""))
    home = _normalize_pick_identity_text(row.get("home_team", row.get("Home", "")))
    away = _normalize_pick_identity_text(row.get("away_team", row.get("Away", "")))
    game_date = _normalize_pick_identity_text(row.get("game_date", row.get("Game Date", "")))
    market_type = _normalize_pick_identity_text(row.get("market_type", ""))
    market_family = market_type.split("_")[0] if market_type else ""
    best_pick = _normalize_pick_identity_text(row.get("best_pick", row.get("Best Pick", "")))
    direction = "over" if best_pick.startswith("over ") else "under" if best_pick.startswith("under ") else ""
    line_used = pd.to_numeric(row.get("market_line_used", pd.NA), errors="coerce")
    line_text = "" if pd.isna(line_used) else f"{float(line_used):.4f}"
    line_source = _normalize_pick_identity_text(row.get("market_line_source", ""))
    return "::".join([league, home, away, game_date, market_type, market_family, direction, line_text, best_pick, line_text, line_source])


def ensure_best_pick_export_columns(
    export_df: pd.DataFrame,
    diagnostics_out: dict | None = None,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Guarantee required transparency columns exist on final best-picks export frame."""
    if export_df is None:
        export_df = pd.DataFrame()

    out = export_df.copy()
    # Compact cards retain their public alias and the canonical evidence field.
    if "calibrated_probability" not in out.columns and "WinProbability" in out.columns:
        out["calibrated_probability"] = pd.to_numeric(out["WinProbability"], errors="coerce")
    req_cols = list(required_columns or REQUIRED_BEST_PICK_EXPORT_COLUMNS)
    default_values: dict[str, object] = {
        "odds_feed_source": "",
        "status_metric_basis": "raw",
        "selection_probability_used": pd.NA,
        "selection_probability_source": "calibrated_probability",
        "selection_probability_pair_normalized": False,
        "mlb_spread_finalist_penalty_applied": False,
        "mlb_spread_finalist_penalty_value": 0.0,
        "mlb_spread_finalist_penalty_reason": "not_evaluated",
        "recent_regime_penalty_applied": False,
        "recent_regime_penalty_value": 0.0,
        "recent_regime_penalty_reason": "not_evaluated",
        "recent_regime_bucket": "",
        "recent_regime_bucket_n": 0,
        "recent_regime_bucket_win_rate": pd.NA,
        "recent_regime_long_win_rate": pd.NA,
        "best_available_value_override_applied": False,
        "best_available_value_override_from_pick": "",
        "best_available_value_override_ev_gain": pd.NA,
        "best_available_rank": pd.NA,
        "best_available_family_rank": pd.NA,
        "best_available_score": pd.NA,
        "best_available_runner_up_pick": "",
        "best_available_runner_up_market_type": "",
        "best_available_runner_up_score": pd.NA,
        "best_available_score_gap": pd.NA,
        "best_available_candidate_count": 0,
        "best_available_selection_verified": False,
        "best_available_ranking_verified": False,
        "final_pick_valid": False,
        "final_pick_valid_reason": "not_validated",
        "best_available_selection_reason": "",
        "qualified_pick": False,
        "qualification_probability": pd.NA,
        "qualification_reason": "PASS: wager qualification not evaluated.",
        "display_pick": "",
        "commercial_tier": "Best Available / Pass",
        "sellable_as_premium": False,
        "sellable_as_value_card": False,
        "controlled_card_recovery": False,
        "best_available_only": True,
        "commercial_reason": "Best available only; no production-qualified edge.",
        "effective_expected_value": pd.NA,
        "effective_edge": pd.NA,
        "effective_win_probability": pd.NA,
        "status_blocker_reason": "",
        "status_blocker_stage": "none",
        "nba_stats_fetch_status": "",
        "fallback_summary_by_league": "",
        "run_health_warning": "",
        "degraded_feature_subset_flag": False,
        "degraded_feature_subset_reason": "",
        "actionable_family_counts": "MISSING_COMPUTATION",
        "totals_only_actionable_flag": False,
        "viable_side_candidates_count": -1,
        "side_promoted_by_balance_guard_count": -1,
        "side_balance_guard_reason": "MISSING_COMPUTATION",
        "market_line_used": pd.NA,
        "market_line_source": "",
        "market_line_source_detail": "",
        "matched_live_spread_line": pd.NA,
        "matched_live_total_line": pd.NA,
        "upload_spread_line": pd.NA,
        "upload_total_line": pd.NA,
        "base_spread_line": pd.NA,
        "base_total_line": pd.NA,
        "line_consistency_flag": True,
        "line_consistency_reason": "",
        "line_provenance_warning": "",
        "line_event_identity_match_flag": True,
        "line_event_identity_reason": "",
        "live_event_match_key": "",
        "line_candidate_count": 0,
        "selected_live_event_source": "",
        "export_run_id": "",
        "pick_id": "",
        "canonical_pick_key": "",
    }

    initially_missing_cols = [c for c in req_cols if c not in out.columns]
    for col in initially_missing_cols:
        out[col] = default_values.get(col, pd.NA)
    missing_cols = [c for c in req_cols if c not in out.columns]

    for col in req_cols:
        if col in {"odds_feed_source", "status_blocker_reason", "status_blocker_stage", "nba_stats_fetch_status", "fallback_summary_by_league", "run_health_warning", "degraded_feature_subset_reason", "status_metric_basis", "selection_probability_source", "mlb_spread_finalist_penalty_reason", "recent_regime_penalty_reason", "recent_regime_bucket", "best_available_value_override_from_pick", "market_line_source", "market_line_source_detail", "line_consistency_reason", "line_provenance_warning", "line_event_identity_reason", "live_event_match_key", "selected_live_event_source", "raw_book_odds_diag", "best_available_runner_up_pick", "best_available_runner_up_market_type", "best_available_selection_reason", "qualification_reason", "display_pick", "commercial_tier", "commercial_reason", "final_pick_valid_reason"}:
            out[col] = out[col].fillna(default_values.get(col, "")).astype(str)

    # The public card always answers which candidate ranked first for the game.
    # Wager approval lives in qualified_pick/Bettable/stake fields and must never
    # be encoded by replacing the pick text with an abstention placeholder.
    if "display_pick" in out.columns and "best_pick" in out.columns:
        display = out["display_pick"].fillna("").astype(str).str.strip()
        ranked_pick = out["best_pick"].fillna("").astype(str).str.strip()
        out["display_pick"] = ranked_pick.where(ranked_pick.ne(""), display)

    if "status_blocker_stage" in out.columns:
        out["status_blocker_stage"] = out["status_blocker_stage"].replace({"": "none"})
    for bool_col, default in {
        "best_available_selection_verified": False,
        "best_available_ranking_verified": False,
        "final_pick_valid": False,
        "sellable_as_premium": False,
        "sellable_as_value_card": False,
        "controlled_card_recovery": False,
        "best_available_only": True,
        "qualified_pick": False,
        "selection_probability_pair_normalized": False,
        "mlb_spread_finalist_penalty_applied": False,
        "recent_regime_penalty_applied": False,
        "best_available_value_override_applied": False,
        "degraded_feature_subset_flag": False,
        "totals_only_actionable_flag": False,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
    }.items():
        if bool_col in out.columns:
            # CSV/string-backed inputs must retain explicit failed checks. Python
            # truthiness would turn "False" and "0" into verified/qualified rows.
            values = out[bool_col]
            normalized = values.astype("string").str.strip().str.casefold()
            out[bool_col] = normalized.isin({"true", "1", "1.0", "yes", "y"}).where(
                values.notna(), default
            ).astype(bool)
    for int_col in ("best_available_rank", "best_available_family_rank", "best_available_candidate_count"):
        if int_col in out.columns:
            out[int_col] = pd.to_numeric(out[int_col], errors="coerce").astype("Int64")
    if "viable_side_candidates_count" in out.columns:
        out["viable_side_candidates_count"] = pd.to_numeric(out["viable_side_candidates_count"], errors="coerce").fillna(0).astype(int)
    if "side_promoted_by_balance_guard_count" in out.columns:
        out["side_promoted_by_balance_guard_count"] = pd.to_numeric(out["side_promoted_by_balance_guard_count"], errors="coerce").fillna(0).astype(int)
    if "side_balance_guard_reason" in out.columns:
        out["side_balance_guard_reason"] = out["side_balance_guard_reason"].fillna("MISSING_COMPUTATION").astype(str)
    for numeric_col in {"selection_probability_used", "mlb_spread_finalist_penalty_value", "recent_regime_penalty_value", "recent_regime_bucket_win_rate", "recent_regime_long_win_rate", "best_available_value_override_ev_gain", "qualification_probability", "market_line_used", "matched_live_spread_line", "matched_live_total_line", "upload_spread_line", "upload_total_line", "base_spread_line", "base_total_line"}:
        if numeric_col in out.columns:
            out[numeric_col] = pd.to_numeric(out[numeric_col], errors="coerce")
    if "line_candidate_count" in out.columns:
        out["line_candidate_count"] = pd.to_numeric(out["line_candidate_count"], errors="coerce").fillna(0).astype(int)
    if "recent_regime_bucket_n" in out.columns:
        out["recent_regime_bucket_n"] = pd.to_numeric(
            out["recent_regime_bucket_n"], errors="coerce"
        ).fillna(0).astype(int)
    if "export_run_id" in out.columns:
        out["export_run_id"] = out["export_run_id"].fillna("").astype(str)
    if "pick_id" in out.columns:
        out["pick_id"] = out["pick_id"].fillna("").astype(str)
    if "canonical_pick_key" in out.columns:
        out["canonical_pick_key"] = out["canonical_pick_key"].fillna("").astype(str)
    if "export_run_id" in out.columns and out["export_run_id"].eq("").all():
        out["export_run_id"] = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if "pick_id" in out.columns and out["pick_id"].eq("").any():
        out.loc[out["pick_id"].eq(""), "pick_id"] = out.index.to_series().map(lambda idx: f"pick_{int(idx) + 1:04d}")
    if "canonical_pick_key" in out.columns and out["canonical_pick_key"].eq("").any():
        out.loc[out["canonical_pick_key"].eq(""), "canonical_pick_key"] = out[out["canonical_pick_key"].eq("")].apply(_build_canonical_pick_key, axis=1)

    if diagnostics_out is not None:
        diag_status = str(diagnostics_out.get("nba_stats_fetch_status", "")).strip().lower()
        if "nba_stats_fetch_status" in out.columns:
            row_status = out["nba_stats_fetch_status"].astype(str).str.strip().str.lower()
            valid_nba_statuses = {
                "live",
                "cached",
                "failed",
                "ok",
                "not_started",
                "not_applicable",
            }
            if diag_status in valid_nba_statuses:
                out["nba_stats_fetch_status"] = row_status.mask(
                    ~row_status.isin(valid_nba_statuses),
                    diag_status,
                )
        if "fallback_summary_by_league" in out.columns and not str(diagnostics_out.get("fallback_summary_by_league", "")).strip() == "":
            out["fallback_summary_by_league"] = out["fallback_summary_by_league"].replace("", diagnostics_out.get("fallback_summary_by_league", ""))
        if "run_health_warning" in out.columns and not str(diagnostics_out.get("run_health_warning", "")).strip() == "":
            out["run_health_warning"] = out["run_health_warning"].replace("", diagnostics_out.get("run_health_warning", ""))

    required_ok = all(col in out.columns for col in req_cols)
    if initially_missing_cols:
        logger.warning("best_pick_export_missing_columns=%s", initially_missing_cols)
    line_cols = [
        "market_line_used", "market_line_source", "market_line_source_detail",
        "matched_live_spread_line", "matched_live_total_line", "upload_spread_line",
        "upload_total_line", "base_spread_line", "base_total_line",
        "line_consistency_flag", "line_consistency_reason", "line_provenance_warning",
    ]
    missing_line_cols = [c for c in line_cols if c in req_cols and c not in export_df.columns]
    if missing_line_cols:
        logger.warning("best_pick_export_missing_line_columns=%s", missing_line_cols)
    logger.info("best_pick_export_line_columns_ok=%s", len(missing_line_cols) == 0)
    logger.info("best_pick_export_required_columns_ok=%s", required_ok)

    if diagnostics_out is not None:
        diagnostics_out["best_pick_export_missing_columns"] = missing_cols
        diagnostics_out["best_pick_export_required_columns_ok"] = bool(required_ok)
        diagnostics_out["best_pick_export_missing_line_columns"] = missing_line_cols
        diagnostics_out["best_pick_export_line_columns_ok"] = len(missing_line_cols) == 0

    return out

BEST_PICK_COLUMNS = [
    "pipeline_build",
    "Triple_Filter_Rank", "parlay_rank",
    "league", "home_team", "away_team", "game_date", "game_time_est", "market_type", "candidate_source", "orientation_source", "upload_match_reason", "best_pick", "Kelly_Bet_Size", "Pick_Status", "Status_Reason",
    "calibrated_probability", "expected_value", "edge", "consensus_agreement",
    "best_available_rank", "best_available_family_rank", "best_available_score",
    "best_available_value_override_applied", "best_available_value_override_from_pick",
    "best_available_value_override_ev_gain",
    "best_available_runner_up_pick", "best_available_runner_up_market_type",
    "best_available_runner_up_score", "best_available_score_gap",
    "best_available_candidate_count", "best_available_selection_verified",
    "best_available_ranking_verified", "final_pick_valid", "final_pick_valid_reason",
    "best_available_selection_reason", "qualified_pick", "qualification_probability",
    "qualification_reason", "display_pick", "commercial_tier", "sellable_as_premium",
    "sellable_as_value_card", "controlled_card_recovery",
    "best_available_only", "commercial_reason", "wager_approved", "export_role",
    "wager_instruction",
    "decimal_odds", "matchup_id",
    "odds_american", "odds_source", "odds_feed_source", "market_probability", "ml_probability", "ml_probability_source", "ml_target", "ml_projection", "ml_residual_scale", "ml_feature_quality", "ml_unavailable_reason", "theover_probability", "win_prob_source", "display_probability",
    "kalshi_probability", "kalshi_match_status", "kalshi_match_reason",
    # Kalshi match instrumentation: the contract line actually used, its distance from
    # the pick line, and the raw P(over) before orientation/decay â€” for diagnosing a
    # systematic over-bias straight from the export.
    "kalshi_matched_line", "kalshi_line_diff", "kalshi_raw_over_prob",
    "kalshi_market_title", "kalshi_floor_strike", "kalshi_cap_strike", "kalshi_yes_bid", "kalshi_yes_ask",
    # Exact signal values fed to compute_blended_probability, oriented to the pick
    # side. Persisted so the blend weights can be backtested/fitted from saved
    # exports without having to re-derive orientation (which is ambiguous after
    # the fact). See scripts/fit_blend_weights.py.
    "blend_in_kalshi", "blend_in_market", "blend_in_ml", "blend_in_theover", "blend_tier",
    # Readable per-signal win-% breakdown (Kalshi/Market/ML/TheOver) â€” see REQUIRED_BEST_PICK_EXPORT_COLUMNS.
    "signal_breakdown",
    "gemini_pick", "gemini_confidence", "gemini_flags", "gemini_reviewed",
    "gemini_agreement", "gemini_reviewed_at", "gemini_review_model", "gemini_review_input_hash",
    "gemini_verified_context", "gemini_supporting_evidence", "gemini_missing_information",
    "gemini_explanation", "gemini_risk_notes", "gemini_gate_enabled",
    "gemini_review_status", "gemini_gate_reason", "gemini_approved",
    "gemini_stake_multiplier", "gemini_outage_allowed", "gemini_outage_cap_fraction", "maturity", "used_stale_features", "Pick_Quality", "Conviction_Score",
    "game_already_started_flag",
    "uploaded_spread_line", "uploaded_total_line", "live_spread_line", "live_total_line", "line_source", "line_delta", "upload_market_match",
    "market_line_used", "market_line_source", "market_line_source_detail", "matched_live_spread_line", "matched_live_total_line", "upload_spread_line", "upload_total_line", "base_spread_line", "base_total_line",
    "line_consistency_flag", "line_consistency_reason", "line_provenance_warning", "line_event_identity_match_flag", "line_event_identity_reason", "live_event_match_key", "line_candidate_count", "selected_live_event_source",
    # Moneyline (h2h) prices carried onto every bet row for the spread-orientation
    # guard. Exported so we can see whether the guard actually has its inputs: real
    # values mean the moneyline reached build_best_picks_df (guard can fire); blank
    # on a spread row means there was no h2h to verify orientation against.
    "game_home_ml_price", "game_away_ml_price",
    # Verbatim per-book spread points + moneyline prices (see REQUIRED_BEST_PICK_EXPORT_COLUMNS);
    # listed here so it survives the BEST_PICK_COLUMNS reindex into the export.
    "raw_book_odds_diag",
    "suspicious_data_flag", "suspicious_data_reasons", "status_metric_basis", "best_available_probability", "best_available_probability_source", "best_available_probability_pair_normalized", "best_available_selection_policy",
        "selection_probability_used", "selection_probability_source", "selection_probability_pair_normalized", "mlb_spread_finalist_penalty_applied", "mlb_spread_finalist_penalty_value", "mlb_spread_finalist_penalty_reason", "recent_regime_penalty_applied", "recent_regime_penalty_value", "recent_regime_penalty_reason", "recent_regime_bucket", "recent_regime_bucket_n", "recent_regime_bucket_win_rate", "recent_regime_long_win_rate", "effective_expected_value", "effective_edge", "effective_win_probability",
    "empirical_win_probability", "empirical_edge", "empirical_bucket", "status_blocker_reason", "status_blocker_stage",
    "nba_stats_fetch_status", "nba_stats_fetch_source", "nba_stats_fetch_retries_used", "stats_source_counts", "fallback_summary_by_league", "fallback_heavy_slate_flag", "run_health_warning",
    "degraded_feature_subset_flag", "degraded_feature_subset_reason",
    "actionable_family_counts", "totals_only_actionable_flag", "viable_side_candidates_count", "side_promoted_by_balance_guard_count", "side_balance_guard_reason",
    "production_win_probability", "production_expected_value", "production_edge", "probability_calibration_reason", "production_eligible",
    "raw_kelly_amount", "production_bet_amount", "kelly_cap_reason", "kelly_zero_reason",
    "kelly_uncalibrated_probability", "kelly_probability_used", "kelly_probability_source",
    "export_run_id", "pick_id", "canonical_pick_key",
]

from app_core.prediction_evidence import PROVENANCE_COLUMNS as _EVIDENCE_EXPORT_COLUMNS
REQUIRED_BEST_PICK_EXPORT_COLUMNS = list(dict.fromkeys(REQUIRED_BEST_PICK_EXPORT_COLUMNS + _EVIDENCE_EXPORT_COLUMNS + ["matchup_id"]))
BEST_PICK_COLUMNS = list(dict.fromkeys(BEST_PICK_COLUMNS + _EVIDENCE_EXPORT_COLUMNS))
# Preserve per-row quality evidence through the public recommendation boundary.
_PUBLIC_QUALITY_COLUMNS = ['stats_source', 'stats_resolution_status', 'stats_fallback_reason',
                           'feature_stats_fallback', 'degraded_feature_subset_flag', 'model_status']
REQUIRED_BEST_PICK_EXPORT_COLUMNS = list(dict.fromkeys(REQUIRED_BEST_PICK_EXPORT_COLUMNS + _PUBLIC_QUALITY_COLUMNS))
BEST_PICK_COLUMNS = list(dict.fromkeys(BEST_PICK_COLUMNS + _PUBLIC_QUALITY_COLUMNS))

# Point-in-time NFL context must survive every selection/export boundary.  These
# fields make it possible to reconstruct which completed games and injury report
# affected a pick instead of silently falling back to the sportsbook price.
_NFL_CONTEXT_COLUMNS = [
    "feature_home_games_played", "feature_away_games_played",
    "feature_home_last5_win_pct", "feature_away_last5_win_pct",
    "feature_home_recent_point_margin", "feature_away_recent_point_margin",
    "feature_home_last_game_summary", "feature_away_last_game_summary",
    "feature_home_last_game_date", "feature_away_last_game_date",
    "injuries_home_count", "injuries_away_count",
    "injury_home_impact", "injury_away_impact",
    "injury_home_summary", "injury_away_summary",
    "injury_context_source", "injury_context_status",
    "injury_probability_adjustment",
    "nfl_context_status", "nfl_context_model_used",
]
REQUIRED_BEST_PICK_EXPORT_COLUMNS = list(dict.fromkeys(REQUIRED_BEST_PICK_EXPORT_COLUMNS + _NFL_CONTEXT_COLUMNS))
BEST_PICK_COLUMNS = list(dict.fromkeys(BEST_PICK_COLUMNS + _NFL_CONTEXT_COLUMNS))


CANONICAL_BET_COLUMNS = [
    "league", "home_team", "away_team", "game_date", "game_time_est", "game_key",
    "market_type", "candidate_source", "orientation_source", "upload_match_reason", "spread_line", "total_line",
    "orientation_favorite_side",
    "theover_probability", "win_prob_source", "odds_american", "odds_source", "odds_feed_source", "market_probability",
    "ml_probability", "ml_probability_source", "ml_target", "ml_projection", "ml_residual_scale", "ml_feature_quality", "ml_unavailable_reason", "display_probability", "calibrated_probability", "expected_value", "edge", "best_pick", "used_stale_features", "matchup_id", "Conviction_Score",
    "uploaded_spread_line", "uploaded_total_line", "live_spread_line", "live_total_line", "line_source", "line_delta", "upload_market_match",
    # Carried so a TheOver-feed degradation warning set by _apply_analysis_calculations
    # survives the canonical reindex and reaches the production degraded-run Kelly guard.
    "run_health_warning",
]

_EXPORT_SIGNAL_COLS = {"market_type", "calibrated_probability", "expected_value", "edge"}


def _compute_signal_breakdown(df: pd.DataFrame) -> pd.Series:
    """Readable per-signal win-% string for each row, e.g.
    ``"Kalshi 58% | Market 46% | ML 64% | TheOver 60%"``.

    Each piece is the exact, pick-side-oriented value that signal contributed to
    the blend. We prefer the persisted ``blend_in_*`` inputs but fall back to the
    raw signal columns, because ``blend_in_kalshi`` is stamped early in
    run_analysis_pipeline (before Kalshi is merged onto live-odds bet rows) while
    ``kalshi_probability`` is populated by the time the export is assembled â€” so
    computing the string here, late, keeps the Kalshi piece from being dropped.
    Signals absent for a row are omitted rather than shown as 0%.
    """
    pieces = (
        ("Kalshi", "blend_in_kalshi", "kalshi_probability"),
        ("Market", "blend_in_market", "market_probability"),
        ("ML", "blend_in_ml", "ml_probability"),
        ("TheOver", "blend_in_theover", "theover_probability"),
    )
    breakdown = pd.Series([""] * len(df), index=df.index)
    for label, primary_col, fallback_col in pieces:
        values = pd.Series([pd.NA] * len(df), index=df.index)
        if primary_col in df.columns:
            values = pd.to_numeric(df[primary_col], errors="coerce")
        if fallback_col in df.columns:
            values = values.fillna(pd.to_numeric(df[fallback_col], errors="coerce"))
        piece = values.map(
            lambda v, _l=label: f"{_l} {v * 100:.0f}%" if pd.notna(v) else ""
        )
        sep = pd.Series(
            np.where((breakdown != "") & (piece != ""), " | ", ""),
            index=df.index,
        )
        breakdown = breakdown + sep + piece
    return breakdown


# Cap combos per leg count to prevent combinatorial explosion
_MAX_PARLAY_COMBOS_PER_LEG = 500

MIN_EDGE_THRESHOLD = 0.02
W_ML = 0.5
W_MARKET = 0.3
W_KALSHI = 0.2

_UPLOAD_COLUMN_ALIASES = {
    "hometeam": "home_team",
    "home team": "home_team",
    "home": "home_team",
    "awayteam": "away_team",
    "away team": "away_team",
    "away": "away_team",
    "pickteam": "pick_team",
    "pick team": "pick_team",
    "winprobability": "theover_probability",
    "win probability": "theover_probability",
    "winprobsource": "win_prob_source",
    "win prob source": "win_prob_source",
    "win_prob_source": "win_prob_source",
    "league": "league",
    "sport": "league",
    "game date": "game_date",
    "gamedate": "game_date",
    "game time": "game_time_est",
    "game time (et)": "game_time_est",
    "team 1": "team_1",
    "team1": "team_1",
    "team_1": "team_1",
    "team 2": "team_2",
    "team2": "team_2",
    "team_2": "team_2",
    "match up": "matchup",
    "match_up": "matchup",
    "event_name": "matchup",
    "teams": "matchup",
    "home team name": "home_team",
    "away team name": "away_team",
    "visitor": "away_team",
    "visitor team": "away_team",
    "team one": "team_1",
    "team two": "team_2",
    "market type": "market_type",
    "spread line": "spread_line",
    "total line": "total_line",
    "theover probability": "theover_probability",
    "odds american": "odds_american",
    "american odds": "odds_american",
    "ml probability": "ml_probability",
    "implied prob": "ml_probability",
    "implied_prob": "ml_probability",
    "calibrated probability": "calibrated_probability",
    "expected value": "expected_value",
}


_NULL_TEXT_TOKENS = {"", "none", "null", "nan", "nat", "n/a", "na", "<na>"}


def _clean_text_placeholders(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.where(~s.str.lower().isin(_NULL_TEXT_TOKENS), "")


def _normalize_upload_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )
    return out


def clean_team_name(series: pd.Series) -> pd.Series:
    """
    Sanitizes team names for ultra-strict joining.
    Strips all non-alphanumeric characters and lowercases team names
    to ensure '76ers' and 'Philadelphia 76ers' resolve accurately.
    """
    if series is None or series.empty:
        return series

    typo_map = {
        "sacramento": "sacramento",
        "sacremento": "sacramento",
        "sacramentokings": "sacramento",
        "sacrementokings": "sacramento",
        "sanantonio": "sanantonio",
        "philidelphia": "philadelphia",
        "phildelphia": "philadelphia",
        "newyorkknicks": "newyork",
    }

    cleaned = series.astype("string").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return cleaned.replace(typo_map)


def _first_nonempty_text(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series([""] * len(df), index=df.index, dtype="string")
    for col in candidates:
        if col in df.columns:
            candidate = _clean_text_placeholders(_string_series(df, col))
            out = out.where(out.str.len().gt(0), candidate)
    return out


def _normalize_team_for_known_league(value: object, league: object) -> str:
    """Apply sport-specific identities after the league is known.

    Generic college aliases intentionally normalize Connecticut to UConn. WNBA
    rows must instead retain the Connecticut Sun franchise identity.
    """
    normalized = normalize_team_name(value)
    league_text = "" if league is None or pd.isna(league) else str(league)
    if league_text.strip().upper() == "WNBA" and str(normalized).strip().lower() == "uconn":
        return "Connecticut"
    return normalized


def _restore_known_league_team_identities(df: pd.DataFrame) -> pd.DataFrame:
    """Repair league-specific alias collisions without renormalizing other names.

    The generic mapper intentionally treats Connecticut as college UConn. Once a
    row is known to be WNBA, restore the Connecticut Sun display identity. This
    function is deliberately substitution-only so names such as Boston Celtics
    and synthetic test identities retain their original spelling.
    """
    if df is None or df.empty or "league" not in df.columns:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()
    league = _clean_text_placeholders(_string_series(out, "league")).str.upper()
    for team_column in ("home_team", "away_team"):
        if team_column not in out.columns:
            continue
        team_token = (
            _clean_text_placeholders(_string_series(out, team_column))
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "", regex=True)
        )
        wnba_connecticut = league.eq("WNBA") & team_token.eq("uconn")
        out.loc[wnba_connecticut, team_column] = "Connecticut"
    return out


def _coerce_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    matchup_text = _first_nonempty_text(out, ["matchup", "match_up", "event", "event_name", "teams", "game"])
    if matchup_text.str.len().eq(0).all():
        sep_probe = r"(?i)(?:@|\bvs\b|\bv\b|\bat\b|[-â€”])"
        for col in out.columns:
            if col in {"home_team", "away_team", "team_1", "team_2", "league", "sport", "pick", "pick_team"}:
                continue
            series = _clean_text_placeholders(_string_series(out, col))
            if series.str.contains(sep_probe, regex=True, na=False).any():
                matchup_text = series
                break
    matchup_clean = _clean_text_placeholders(matchup_text)

    away_from_matchup = pd.Series([""] * len(out), index=out.index, dtype="string")
    home_from_matchup = pd.Series([""] * len(out), index=out.index, dtype="string")
    sep_pattern = r"(?i)\s*(?:@|vs\.?|v\.?|at|[-â€”])\s*"
    parts = matchup_clean.str.split(sep_pattern, n=1, expand=True, regex=True)
    if isinstance(parts, pd.DataFrame) and parts.shape[1] >= 2:
        away_from_matchup = _clean_text_placeholders(parts[0])
        home_from_matchup = _clean_text_placeholders(parts[1])

    # Some TheOver exports include Team 1/Team 2 while Home/Away may be blank.
    home_fallback = _first_nonempty_text(out, ["home_team", "team_1", "home"])
    away_fallback = _first_nonempty_text(out, ["away_team", "team_2", "away"])
    home_fallback = home_fallback.where(home_fallback.str.len().gt(0), home_from_matchup)
    away_fallback = away_fallback.where(away_fallback.str.len().gt(0), away_from_matchup)
    league_fallback = _first_nonempty_text(out, ["league", "sport"])
    out["league"] = league_fallback.str.upper().replace(LEAGUE_ALIASES)
    out["home_team"] = pd.Series(
        [
            _normalize_team_for_known_league(team, league)
            for team, league in zip(home_fallback, out["league"])
        ],
        index=out.index,
        dtype="string",
    )
    out["away_team"] = pd.Series(
        [
            _normalize_team_for_known_league(team, league)
            for team, league in zip(away_fallback, out["league"])
        ],
        index=out.index,
        dtype="string",
    )
    return out


def _infer_missing_league_from_base(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty or base_df is None or base_df.empty:
        return out

    out["league"] = _clean_text_placeholders(_string_series(out, "league")).str.upper().replace(LEAGUE_ALIASES)
    out["home_team"] = _clean_text_placeholders(_string_series(out, "home_team")).map(normalize_team_name)
    out["away_team"] = _clean_text_placeholders(_string_series(out, "away_team")).map(normalize_team_name)

    missing_mask = out["league"].str.len().eq(0) & out["home_team"].str.len().gt(0) & out["away_team"].str.len().gt(0)
    if not missing_mask.any():
        return out

    base = base_df.copy()
    base["league"] = _string_series(base, "league").str.upper().replace(LEAGUE_ALIASES)
    base["home_team"] = _string_series(base, "home_team").map(normalize_team_name)
    base["away_team"] = _string_series(base, "away_team").map(normalize_team_name)

    direct = base[["league", "home_team", "away_team"]].drop_duplicates()
    reverse = direct.rename(columns={"home_team": "away_team", "away_team": "home_team"})
    lookup = pd.concat([direct, reverse], ignore_index=True)
    lookup = lookup[lookup["league"].str.len().gt(0)].drop_duplicates(["home_team", "away_team", "league"])
    league_by_match = (
        lookup.groupby(["home_team", "away_team"], as_index=False)["league"]
        .agg(lambda x: sorted(set([v for v in x if isinstance(v, str) and v])) )
    )

    match_fill = out.loc[missing_mask, ["home_team", "away_team"]].merge(
        league_by_match,
        on=["home_team", "away_team"],
        how="left",
    )
    inferred = match_fill["league"].apply(lambda v: v[0] if isinstance(v, list) and len(v) == 1 else "")
    out.loc[missing_mask, "league"] = out.loc[missing_mask, "league"].where(out.loc[missing_mask, "league"].str.len().gt(0), inferred.values)
    return out


def _infer_missing_league_from_team_sets(df: pd.DataFrame, selected_sports: list[str] | None) -> pd.DataFrame:
    """Fill missing league labels from explicit source hints and known team sets."""
    out = df.copy()
    if out.empty:
        return out

    out["league"] = _clean_text_placeholders(_string_series(out, "league")).str.upper().replace(LEAGUE_ALIASES)
    missing_mask = out["league"].str.len().eq(0)
    if not missing_mask.any():
        return out

    # NCAAB and NCAAF share school names and mascots. Source metadata is the only
    # safe discriminator when league is blank, so honor explicit football hints
    # before applying basketball-oriented college-team recovery.
    source_text = pd.Series([""] * len(out), index=out.index, dtype="string")
    for src_col in ["sport", "source", "data_source", "odds_source", "event_name", "matchup", "league_source"]:
        if src_col in out.columns:
            source_text = source_text + " " + _clean_text_placeholders(_string_series(out, src_col)).str.lower()
    ncaaf_source = source_text.str.contains(
        r"americanfootball_ncaaf|\bncaaf\b|\bcfb\b|college football|ncaa football",
        regex=True,
        na=False,
    )
    out.loc[missing_mask & ncaaf_source, "league"] = "NCAAF"
    missing_mask = out["league"].str.len().eq(0)

    # 1. Check NCAAB keyword recovery regex FIRST to prevent college teams from being swallowed by pro city names
    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_LEAGUE_RECOVERY_KEYWORDS)) + r")\b"
    home_text = _clean_text_placeholders(_string_series(out, "home_team")).str.lower()
    away_text = _clean_text_placeholders(_string_series(out, "away_team")).str.lower()
    keyword_mask = home_text.str.contains(keyword_pattern, regex=True, na=False) | away_text.str.contains(keyword_pattern, regex=True, na=False)
    out.loc[missing_mask & keyword_mask, "league"] = "NCAAB"

    # Refresh missing mask after NCAAB assignment
    missing_mask = out["league"].str.len().eq(0)

    # 2. Preserve explicit WNBA franchise identities before global normalization
    # collapses full names to city-only aliases that overlap NBA teams.
    raw_home = _clean_text_placeholders(_string_series(out, "home_team")).str.lower()
    raw_away = _clean_text_placeholders(_string_series(out, "away_team")).str.lower()
    wnba_exact_keys = {str(k).strip().lower() for k in WNBA_EXACT_MAP}
    wnba_mask = missing_mask & (
        raw_home.isin(wnba_exact_keys) | raw_away.isin(wnba_exact_keys)
    )
    out.loc[wnba_mask, "league"] = "WNBA"

    # 3. Precedence Override: Check NBA/NHL exact map
    missing_mask = out["league"].str.len().eq(0)
    nba_teams = {normalize_team_name(v) for v in NBA_EXACT_MAP.values()}
    nhl_teams = {normalize_team_name(v) for v in NHL_EXACT_MAP.values()}

    # We must check against keys of NBA_EXACT_MAP in addition to values.
    nba_exact_keys = {normalize_team_name(k) for k in NBA_EXACT_MAP.keys()}
    nba_full_set = nba_teams.union(nba_exact_keys)

    home = _string_series(out, "home_team").map(normalize_team_name)
    away = _string_series(out, "away_team").map(normalize_team_name)

    # We must NOT override NCAAB assignments that were just made by keyword_mask,
    # so we use the updated missing_mask which excludes rows already assigned to NCAAB.
    nba_mask = missing_mask & (home.isin(nba_full_set) | away.isin(nba_full_set))
    nhl_mask = missing_mask & (home.isin(nhl_teams) | away.isin(nhl_teams))
    out.loc[nba_mask, "league"] = "NBA"
    out.loc[nhl_mask & out["league"].str.len().eq(0), "league"] = "NHL"

    selected = {str(s).upper() for s in (selected_sports or [])}
    has_ncaab = bool(selected.intersection({"NCAAB", "NCAAM", "NCAA MEN'S BASKETBALL", "NCAA MENS BASKETBALL"}))

    # We must not blindly backfill NCAAB if it's already identified as NBA.
    if has_ncaab:
        out.loc[out["league"].str.len().eq(0), "league"] = "NCAAB"

    return out


def _recover_ncaab_league_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Recover missing NCAAB league labels from college-specific team keywords."""
    out = df.copy()
    if out.empty:
        return out

    out["league"] = _clean_text_placeholders(_string_series(out, "league")).astype("string").str.strip().str.lower()
    missing_league = out["league"].str.len().eq(0)
    if not missing_league.any():
        return out

    home_text = _clean_text_placeholders(_string_series(out, "home_team")).str.lower().str.strip()
    away_text = _clean_text_placeholders(_string_series(out, "away_team")).str.lower().str.strip()
    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_LEAGUE_RECOVERY_KEYWORDS)) + r")\b"
    keyword_mask = home_text.str.contains(keyword_pattern, regex=True, na=False) | away_text.str.contains(keyword_pattern, regex=True, na=False)

    out.loc[missing_league & keyword_mask, "league"] = "ncaab"
    return out


def _restore_missing_ncaab_league_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-priority league restoration pass required before Kalshi/ML enrichment.
    Restores league='ncaab' when league is empty/NaN/<NA> and matchup text looks college.
    """
    out = df.copy()
    if out.empty:
        return out

    for col in ["league", "home_team", "away_team"]:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
        out[col] = _clean_text_placeholders(_string_series(out, col)).astype("string").str.strip()

    missing_league = _clean_text_placeholders(_string_series(out, "league")).str.len().eq(0)
    if not missing_league.any():
        return out

    # We must exclude NBA/NHL matches *before* we apply NCAAB regex heuristics,
    # otherwise Golden State might get labeled NCAAB due to the 'state' token.
    nba_teams = {normalize_team_name(v) for v in NBA_EXACT_MAP.values()}
    nba_exact_keys = {normalize_team_name(k) for k in NBA_EXACT_MAP.keys()}
    nba_full_set = nba_teams.union(nba_exact_keys)

    nhl_teams = {normalize_team_name(v) for v in NHL_EXACT_MAP.values()}
    nhl_exact_keys = {normalize_team_name(k) for k in NHL_EXACT_MAP.keys()}
    nhl_full_set = nhl_teams.union(nhl_exact_keys)

    pro_full_set = nba_full_set.union(nhl_full_set)

    home_normalized = _string_series(out, "home_team").map(normalize_team_name)
    away_normalized = _string_series(out, "away_team").map(normalize_team_name)

    # Exclude teams that are specifically mapped to NBA/NHL but could have college namesakes
    # unless they are explicitly accompanied by their pro city token.
    # Note: Indiana and Memphis are mapped to NBA by default in the mapper,
    # but we need to verify they aren't actually college teams based on opponent.

    is_pro_mask = home_normalized.isin(pro_full_set) | away_normalized.isin(pro_full_set)

    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_LEAGUE_RECOVERY_KEYWORDS)) + r")\b"
    home_text = _clean_text_placeholders(_string_series(out, "home_team")).str.lower()
    away_text = _clean_text_placeholders(_string_series(out, "away_team")).str.lower()
    keyword_mask = home_text.str.contains(keyword_pattern, regex=True, na=False) | away_text.str.contains(keyword_pattern, regex=True, na=False)

    # Allow NCAAB keyword mask to override the is_pro_mask. If a team has a college mascot, it's college.
    # Ex: "Saint Louis Billikens" contains "Billikens" which isn't in pro_full_set,
    # but "Saint Louis" is. Because keyword_mask matches, we trust it's NCAAB.
    out.loc[missing_league & keyword_mask, "league"] = "ncaab"
    return out


def _patch_missing_league_for_college_rows(df: pd.DataFrame, selected_sports: list[str] | None = None) -> pd.DataFrame:
    """Backfill missing league labels for college rows before downstream merges."""
    out = df.copy()
    if out.empty:
        return out

    out["league"] = _clean_text_placeholders(_string_series(out, "league")).str.upper().replace(LEAGUE_ALIASES)
    missing_league = out["league"].str.len().eq(0)
    if not missing_league.any():
        return out

    home_raw = _clean_text_placeholders(_string_series(out, "home_team")).str.lower().str.strip()
    away_raw = _clean_text_placeholders(_string_series(out, "away_team")).str.lower().str.strip()
    home = home_raw.map(normalize_team_name).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    away = away_raw.map(normalize_team_name).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)

    teams_mask = home.isin(_KNOWN_NCAAB_TEAM_TOKENS) | away.isin(_KNOWN_NCAAB_TEAM_TOKENS)
    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_TEAM_KEYWORD_HINTS)) + r")\b"
    teams_mask = teams_mask | home_raw.str.contains(keyword_pattern, regex=True, na=False) | away_raw.str.contains(keyword_pattern, regex=True, na=False)

    source_text = pd.Series([""] * len(out), index=out.index, dtype="string")
    for src_col in ["sport", "source", "data_source", "odds_source", "event_name", "matchup", "league_source"]:
        if src_col in out.columns:
            source_text = source_text + " " + _clean_text_placeholders(_string_series(out, src_col)).str.lower()
    college_source_mask = pd.Series(False, index=out.index)
    for hint in _COLLEGE_SOURCE_HINTS:
        college_source_mask = college_source_mask | source_text.str.contains(hint, na=False)

    selected = {str(s).upper() for s in (selected_sports or [])}
    selected_has_college = bool(selected.intersection({"NCAAB", "NCAAM", "NCAA MEN'S BASKETBALL", "NCAA MENS BASKETBALL"}))
    if selected_has_college:
        college_source_mask = college_source_mask | pd.Series(True, index=out.index)

    out.loc[missing_league & (teams_mask | college_source_mask), "league"] = "NCAAB"
    return out


def _preprocess_bet_rows_for_league_bridge(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identity fields and restore missing NCAAB league labels before joins."""
    out = df.copy()
    if out.empty:
        return out

    for col in ["league", "home_team", "away_team"]:
        out[col] = _clean_text_placeholders(_string_series(out, col)).astype("string").str.lower().str.strip()

    # High-priority league restoration: recover blank/NaN/<NA> league values before enrichment.
    missing_league = _clean_text_placeholders(_string_series(out, "league")).str.len().eq(0)
    if not missing_league.any():
        return out

    nba_teams = {normalize_team_name(v).lower() for v in NBA_EXACT_MAP.values()}
    nba_exact_keys = {normalize_team_name(k).lower() for k in NBA_EXACT_MAP.keys()}
    nba_full_set = nba_teams.union(nba_exact_keys)

    nhl_teams = {normalize_team_name(v).lower() for v in NHL_EXACT_MAP.values()}
    nhl_exact_keys = {normalize_team_name(k).lower() for k in NHL_EXACT_MAP.keys()}
    nhl_full_set = nhl_teams.union(nhl_exact_keys)

    pro_full_set = nba_full_set.union(nhl_full_set)

    home_normalized = _string_series(out, "home_team").map(normalize_team_name).str.lower()
    away_normalized = _string_series(out, "away_team").map(normalize_team_name).str.lower()
    is_pro_mask = home_normalized.isin(pro_full_set) | away_normalized.isin(pro_full_set)

    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_LEAGUE_RECOVERY_KEYWORDS)) + r")\b"
    home_text = _clean_text_placeholders(_string_series(out, "home_team")).str.lower().str.strip()
    away_text = _clean_text_placeholders(_string_series(out, "away_team")).str.lower().str.strip()
    team_keyword_mask = home_text.str.contains(keyword_pattern, regex=True, na=False) | away_text.str.contains(keyword_pattern, regex=True, na=False)

    source_text = pd.Series([""] * len(out), index=out.index, dtype="string")
    for src_col in ["sport", "source", "data_source", "odds_source", "event_name", "matchup", "league_source"]:
        if src_col in out.columns:
            source_text = source_text + " " + _clean_text_placeholders(_string_series(out, src_col)).str.lower()
    source_is_college = source_text.str.contains(r"\bncaa\b|\bncaab\b|\bncaam\b|college", regex=True, na=False)

    # Let college hints bypass the pro mask if they have college keywords
    out.loc[missing_league & (team_keyword_mask | source_is_college), "league"] = "ncaab"
    return out


def _normalize_identity_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Normalize key identity columns to pandas StringDtype and stripped text before joins."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = _clean_text_placeholders(_string_series(out, col)).astype("string").str.strip()
    return out


def _enforce_identity_string_dtype(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Force identity columns to pandas StringDtype for safe text operations in Pandas 2.x."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in cols:
        out[col] = _clean_text_placeholders(_string_series(out, col)).astype("string").str.strip()
    return out


def _string_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None:
        return pd.Series(dtype="string")
    if df.empty:
        return pd.Series([default] * len(df), index=df.index, dtype="string")
    if col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("object")
        return series.astype("string").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def _normalize_merge_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """N…120902 tokens truncated…ias = apply_mlb_total_market_debias(calibrated_probability, merged)
    if abs(_prod_debias) > 1e-9:
        merged["mlb_total_market_debias"] = _prod_debias

    merged["theover_probability"] = theover_probability
    merged["model_probability"] = model_probability
    merged["display_probability"] = model_probability.round(3)
    merged["calibrated_probability"] = calibrated_probability

    # Persist the EXACT signal inputs the blend consumed (already oriented to the
    # pick side). These are the values weight-fitting must train on; reconstructing
    # them from the other export columns after the fact is unreliable because the
    # native orientation of kalshi_probability is ambiguous. Tier mirrors the
    # Kalshi>=0.55 split inside compute_blended_probability.
    merged["blend_in_market"] = merged["market_probability"]
    merged["blend_in_kalshi"] = kalshi_probability
    merged["blend_in_ml"] = model_probability
    merged["blend_in_theover"] = theover_blend_input
    merged["blend_tier"] = np.where(
        pd.to_numeric(kalshi_probability, errors="coerce").fillna(0.0) >= 0.55, 1, 2
    )

    # Human-readable breakdown of every separate signal feeding the blend, each
    # oriented to the pick side and shown as its own win %. Recomputed authoritatively
    # in build_best_picks_df once Kalshi is merged onto every row (see helper docstring).
    merged["signal_breakdown"] = _compute_signal_breakdown(merged)
    if "nba_stats_fetch_status" in merged.columns:
        merged["nba_stats_fetch_status"] = _string_series(merged, "nba_stats_fetch_status").replace({"": pd.NA}).fillna(
            str(nba_stats_diag.get("nba_stats_fetch_status", "not_started"))
        )
    else:
        merged["nba_stats_fetch_status"] = nba_stats_diag.get("nba_stats_fetch_status", "not_started")
    if "nba_stats_fetch_source" in merged.columns:
        merged["nba_stats_fetch_source"] = _string_series(merged, "nba_stats_fetch_source").replace({"": pd.NA}).fillna(
            str(nba_stats_diag.get("nba_stats_fetch_source", "none"))
        )
    else:
        merged["nba_stats_fetch_source"] = nba_stats_diag.get("nba_stats_fetch_source", "none")
    if "nba_stats_fetch_retries_used" in merged.columns:
        merged["nba_stats_fetch_retries_used"] = pd.to_numeric(
            merged["nba_stats_fetch_retries_used"], errors="coerce"
        ).fillna(int(nba_stats_diag.get("nba_stats_fetch_retries_used", 0))).astype(int)
    else:
        merged["nba_stats_fetch_retries_used"] = int(nba_stats_diag.get("nba_stats_fetch_retries_used", 0))
    degraded_reason = str(ml_prediction_diag.get("ml_schema_mismatch_reason", "")).strip()
    degraded_flag = "degraded_subset" in degraded_reason
    merged["degraded_feature_subset_flag"] = bool(degraded_flag)
    merged["degraded_feature_subset_reason"] = degraded_reason if degraded_flag else ""
    if "fallback_summary_by_league" not in merged.columns:
        merged["fallback_summary_by_league"] = ""
    if "run_health_warning" not in merged.columns:
        merged["run_health_warning"] = ""
    fallback_summary_series = _string_series(merged, "fallback_summary_by_league")
    if fallback_summary_series.str.strip().eq("").all() and "stats_source" in merged.columns:
        fallback_rows = _string_series(merged, "stats_source").str.lower().isin({"fallback", "failed"})
        if fallback_rows.any():
            summary_by_league = _string_series(merged.loc[fallback_rows], "league").value_counts().to_dict()
            merged["fallback_summary_by_league"] = str({str(k): int(v) for k, v in summary_by_league.items()})
    run_warning_series = _string_series(merged, "run_health_warning")
    if run_warning_series.str.strip().eq("").all():
        fallback_rows = _string_series(merged, "stats_source").str.lower().isin({"fallback", "failed"})
        fallback_heavy = float(fallback_rows.sum()) / float(max(len(merged), 1)) >= 0.25
        if fallback_heavy or degraded_flag:
            merged["run_health_warning"] = (
                "Run health warning: fallback/degraded feature usage is elevated; card confidence may be reduced."
            )

    # Phase 3: NCAAB Statistical Recalibration
    # If is_neutral == True for neutral-site and tournament games, compress margins to prevent false edges on tight spreads.
    # We compress the difference between the calibrated probability and 0.5 (neutral) for NCAAB neutral games.
    if "is_neutral" in merged.columns:
        ncaab_neutral_mask = (merged["league"].str.upper().eq("NCAAB").fillna(False)) & (((merged["is_neutral"].eq(True)).fillna(False)) | (merged["is_neutral"].astype(str).str.lower().eq("true").fillna(False)))
        # Apply a 0.85 variance multiplier compression
        compressed_prob = 0.5 + ((calibrated_probability - 0.5) * 0.85)
        calibrated_probability = calibrated_probability.where(~ncaab_neutral_mask, compressed_prob)
        merged["calibrated_probability"] = calibrated_probability

    # Bypass EV calculation for rows without odds or main lines
    ev = calibrated_probability * (merged["decimal_odds"] - 1) - (1 - calibrated_probability)
    edge = calibrated_probability - merged["market_probability"]

    # Null out EV and edge for missing odds
    missing_odds_mask = merged["odds_american"].isna()
    ev = ev.mask(missing_odds_mask, pd.NA)
    edge = edge.mask(missing_odds_mask, pd.NA)

    # Phase 2: Eradication of Floating-Point Artefacts
    # Cast micro-edges to exact zero.
    edge = pd.to_numeric(edge, errors="coerce")
    ev = pd.to_numeric(ev, errors="coerce")
    edge = edge.round(4)
    ev = ev.round(4)
    zero_mask = edge.abs() < 0.0001
    edge = edge.mask(zero_mask, 0.0)
    ev = ev.mask(zero_mask, 0.0)

    # Phase 3: NHL Statistical Recalibration
    # Apply a fractional discount (0.80) to the Expected Value for NHL Totals and Spreads
    # to account for the bimodal distribution of late-game empty-net scenarios.
    if "league" not in merged.columns:
        merged["league"] = ""
    nhl_totals_mask = (_string_series(merged, "league").str.upper() == "NHL") & (_string_series(merged, "market_type").str.contains("total|spread", case=False, na=False))
    ev = ev.where(~nhl_totals_mask, ev * 0.80)

    merged["expected_value"] = ev
    merged["edge"] = edge

    merged["best_pick"] = merged.apply(_format_best_pick, axis=1)

    # Capture diagnostics before Threshold Filtering so metrics reflect the entire Odds API input
    pre_filter_total_games = int(_canonical_matchup_key(merged).nunique()) if not merged.empty else 0
    pre_filter_total_rows = int(len(merged))

    # Phase 5: Global Threshold Filtering
    # The requirement is that any row returned for display or export must meet strict edge/ev thresholds.
    # To keep all rows in analysis_df (for diagnostics and total_games counting), we do NOT drop here.
    # The Best Picks dataframe builder will use the edge and EV thresholds to filter later.
    if not merged.empty:
        pass

    analysis_df = merged.head(max_rows).copy()
    if not analysis_df.empty and not base_df.empty:
        base_dates = base_df.copy()
        base_dates["league"] = _string_series(base_dates, "league").str.upper().replace(LEAGUE_ALIASES)
        base_dates["home_team"] = _string_series(base_dates, "home_team").map(normalize_team_name)
        base_dates["away_team"] = _string_series(base_dates, "away_team").map(normalize_team_name)
        base_dates["date"] = _game_dates(base_dates)

        base_dates["matchup_key"] = _canonical_matchup_teams_key(base_dates)
        analysis_df["matchup_key"] = _canonical_matchup_teams_key(analysis_df)

        date_fill = analysis_df.merge(
            base_dates[["league", "matchup_key", "date"]].drop_duplicates(["league", "matchup_key"]),
            on=["league", "matchup_key"],
            how="left",
            suffixes=("", "_basefill"),
        )
        date_fill_series = _game_dates(date_fill)
        if "date_basefill" in date_fill.columns:
            date_fill_series = date_fill_series.where(date_fill_series.notna(), pd.to_datetime(date_fill["date_basefill"], errors="coerce", utc=True))
        analysis_df["game_date"] = _game_dates(analysis_df).fillna(date_fill_series)
        analysis_df = analysis_df.drop(columns=["matchup_key"], errors="ignore")

    # Ensure 100% date fill success using fallback if any are still missing
    if not analysis_df.empty:
        analysis_df["game_date"] = analysis_df["game_date"].fillna(_game_date_fallback())

    # Normalize identity merge keys for downstream Kalshi and app-layer merges.
    analysis_df = _normalize_merge_keys(analysis_df, ["league", "home_team", "away_team", "game_date"])

    if "game_key" not in analysis_df.columns:
        analysis_df["game_key"] = _mk_game_key(analysis_df)
    if not analysis_df.empty and "market_type" not in analysis_df.columns:
        raise ValueError("analysis_df missing market_type before best-pick construction")

    # In the refactored flow, we no longer build best_picks_df inside run_analysis_pipeline.
    # Instead, we just return an empty dataframe here, and best_picks_df is built in streamlit_app.py
    # AFTER the full analysis_df has been enriched with Kalshi probabilities.
    best_picks_df = pd.DataFrame(columns=BEST_PICK_COLUMNS)

    base_coverage = float(_game_dates(base_df).notna().mean()) if not base_df.empty else 0.0

    totals_coverage = _theover_upload_coverage(totals_df, "totals")
    spreads_coverage = _theover_upload_coverage(spreads_df, "spreads")

    diagnostics = {
        "candidate_generation_diagnostics": diag_counts,
        "unmatched_live_games": diag_counts.get("unmatched_live_games", []),
        "missing_uploaded_games": diag_counts.get("missing_uploaded_games", []),
        "total_rows": pre_filter_total_rows,
        "rows_with_game_date": int(pd.to_datetime(analysis_df.get("game_date"), errors="coerce", utc=True).notna().sum()) if not analysis_df.empty else 0,
        # Safely sort team names alphabetically to count unique actual physical games (matchups) across all markets
        "total_games": pre_filter_total_games,
        "bet_rows": int(len(analysis_df)),
        "ml_model_loaded": bool(use_ml and ML_AVAILABLE and ml_model_actually_loaded),
        "ml_predictions": int(analysis_df["ml_probability"].notna().sum()) if "ml_probability" in analysis_df.columns else 0,
        "market_specific_ml_predictions": int(ml_prediction_diag.get("market_specific_ml_predictions", 0)),
        "ml_probability_source_counts": _string_series(analysis_df, "ml_probability_source").replace("", "Missing").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "ml_target_counts": _string_series(analysis_df, "ml_target").replace("", "Missing").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "best_picks": int(len(best_picks_df)),
        "kalshi_attempted": 0,
        "kalshi_matches": 0,
        "kalshi_match_rate": 0.0,
        "match_rate": 0.0,
        "theover_totals_games": totals_coverage["file_game_count"],
        "theover_totals_market_games": totals_coverage["market_game_count"],
        "theover_totals_probability_games": totals_coverage["probability_game_count"],
        "theover_spreads_games": spreads_coverage["file_game_count"],
        "theover_spreads_market_games": spreads_coverage["market_game_count"],
        "theover_spreads_probability_games": spreads_coverage["probability_game_count"],
        "date_fill_total_rows": int(date_stats["date_fill_total_rows"]),
        "date_fill_success_rows": int(date_stats["date_fill_success_rows"]),
        "date_fill_success_rate": float(date_stats["date_fill_success_rate"]),
        "missing_game_date_rows": int(date_stats["missing_game_date_rows"]),
        "positive_ev_picks": int((_numeric_series(analysis_df, "expected_value", 0.0) > 0).sum()) if not analysis_df.empty else 0,
        "market_type_counts": _string_series(analysis_df, "market_type").fillna("Missing").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "allowed_market_type_rows": int(_string_series(analysis_df, "market_type").isin(VALID_MARKETS).sum()) if not analysis_df.empty else 0,
        "positive_ev_rows": int((_numeric_series(analysis_df, "expected_value", 0.0) > 0).sum()) if not analysis_df.empty else 0,
        "spread_rows_missing_model_prob": int(((_string_series(analysis_df, "market_type").str.startswith("spread")) & (_numeric_series(analysis_df, "model_probability").isna())).sum()) if not analysis_df.empty else 0,
        "best_pick_nonempty_rows": int(_string_series(best_picks_df, "best_pick").str.strip().str.len().gt(0).sum()) if not best_picks_df.empty else 0,
        "best_picks_count": int(len(best_picks_df)),
        "odds_schedule_loaded": odds_schedule_loaded,
        "odds_source_counts": _string_series(analysis_df, "odds_source").fillna("Missing").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "base_rows_loaded": int(len(base_df)),
        "stale_base_rows_removed": int(stale_base_rows_removed),
        "merge_keys_used": merge_keys,
        "stale_base_schedule": stale,
        "base_date_coverage": base_coverage,
        "has_normalized_bet_rows": not analysis_df.empty,
        "nba_stats_fetch_status": nba_stats_diag["nba_stats_fetch_status"],
        "nba_stats_fetch_source": nba_stats_diag["nba_stats_fetch_source"],
        "nba_stats_fetch_retries_used": nba_stats_diag["nba_stats_fetch_retries_used"],
        "nba_rows_live_stats": nba_stats_diag["nba_rows_live_stats"],
        "nba_rows_cached_stats": nba_stats_diag["nba_rows_cached_stats"],
        "nba_rows_fallback_stats": nba_stats_diag["nba_rows_fallback_stats"],
        "rows_unresolved_team_mapping": nba_stats_diag["rows_unresolved_team_mapping"],
        "rows_excluded_from_ml_unresolved_stats": nba_stats_diag["rows_excluded_from_ml_unresolved_stats"],
        "ml_input_row_count": int(ml_input_diag.get("ml_input_row_count", 0)),
        "ml_feature_eligible_row_count": int(ml_input_diag.get("ml_feature_eligible_row_count", 0)),
        "ml_rows_excluded_count": int(ml_input_diag.get("ml_rows_excluded_count", 0)),
        "ml_zero_variance_feature_count": int(ml_input_diag.get("ml_zero_variance_feature_count", 0)),
        "ml_near_constant_feature_count": int(ml_input_diag.get("ml_near_constant_feature_count", 0)),
        "ml_high_missingness_feature_count": int(ml_input_diag.get("ml_high_missingness_feature_count", 0)),
        "ml_top_missing_features": ml_input_diag.get("ml_top_missing_features", {}),
        "ml_top_feature_nunique": ml_input_diag.get("ml_top_feature_nunique", {}),
        "ml_flatness_root_cause_hint": str(
            ml_prediction_diag.get(
                "ml_flatness_root_cause_hint",
                ml_input_diag.get("ml_flatness_root_cause_hint", "not_computed"),
            )
        ),
        "ml_expected_feature_count": int(ml_prediction_diag.get("ml_expected_feature_count", 0)),
        "ml_actual_feature_count": int(ml_prediction_diag.get("ml_actual_feature_count", 0)),
        "ml_missing_feature_columns": ml_prediction_diag.get("ml_missing_feature_columns", []),
        "ml_extra_feature_columns": ml_prediction_diag.get("ml_extra_feature_columns", []),
        "schema_mismatch_detected": bool(ml_prediction_diag.get("schema_mismatch_detected", False)),
        "degraded_feature_subset_flag": bool("degraded_subset" in str(ml_prediction_diag.get("ml_schema_mismatch_reason", ""))),
        "degraded_feature_subset_reason": str(ml_prediction_diag.get("ml_schema_mismatch_reason", "")),
        "rows_using_league_average_defaults": int(ml_prediction_diag.get("rows_using_league_average_defaults", 0)),
        "rows_with_high_default_feature_share": int(ml_prediction_diag.get("rows_with_high_default_feature_share", 0)),
        "rows_with_duplicate_feature_signature": int(ml_prediction_diag.get("rows_with_duplicate_feature_signature", 0)),
        "top_duplicate_feature_signatures": ml_prediction_diag.get("top_duplicate_feature_signatures", []),
        "raw_prediction_distribution": ml_prediction_diag.get("raw_prediction_distribution", {}),
        "hybrid_fallback_triggered": False,
    }

    default_odds_ratio = float((_numeric_series(analysis_df, "odds_american") == -110).mean()) if not analysis_df.empty else 1.0
    diagnostics["odds_fallback_only"] = bool(default_odds_ratio >= 0.99)
    if diagnostics["odds_fallback_only"] and not analysis_df.empty:
        diagnostics["diagnostic_warning"] = "odds_american mostly fallback -110"
    diagnostics["hybrid_fallback_triggered"] = bool(
        _string_series(analysis_df, "model_status").isin(["Statistical Fallback", "Neutral Fallback"]).any()
    ) if not analysis_df.empty else False

    # Jules: Fix Midnight Flattening by using raw UTC if available
    if not analysis_df.empty:
        # In-progress guard input: an odds-API commence time already in the past at
        # run time means the row's "live" odds are IN-GAME prices, not pre-game
        # lines (1 Jul run: games 1-3 hours underway surfaced a 19.5 MLB "total"
        # and -100000 moneylines). Flag here â€” where the raw timestamp is still
        # available â€” and hard-bench in build_best_picks_df. Only a real parsed
        # timestamp can flag: uploaded rows without a live commence time (date-only
        # midnight game_date) are never flagged.
        if "commence_time_raw" in analysis_df.columns:
            _commence = pd.to_datetime(analysis_df["commence_time_raw"], errors="coerce", utc=True)
            analysis_df["game_already_started_flag"] = (
                _commence.notna() & (_commence <= pd.Timestamp.now(tz="UTC"))
            )
        else:
            analysis_df["game_already_started_flag"] = False

        # Resolve time row by row. A mixed slate may have a raw timestamp for
        # primary-feed games and only a provider-normalized display time for a
        # fallback row; the latter must not be overwritten with a blank.
        analysis_df["game_time_est"] = _coalesce_game_time_est(analysis_df)

        # Final Cleanup
        if "commence_time_raw" in analysis_df:
            analysis_df["game_start_utc"] = analysis_df["commence_time_raw"]
        analysis_df = analysis_df.drop(columns=["commence_time_raw"], errors="ignore")

        # Sync the slate date with actual start time
        # Strip the ' ET' label and use mixed format parsing to handle cases where time is missing
        analysis_df["game_date"] = pd.to_datetime(analysis_df["game_time_est"].astype(str).str.replace(" ET", "", regex=False), format='mixed', errors='coerce').dt.date.fillna(analysis_df["game_date"])

    # Phase 6: Close the Feedback Loop
    # Calculate Conviction_Score based on historical calibration performance
    if not analysis_df.empty:

        def _compute_deterministic_conviction(df):
            """Calculate a robust deterministic fallback Conviction Score safely."""
            try:
                # Safely extract prob
                if 'calibrated_probability' in df.columns:
                    prob_series = df['calibrated_probability']
                elif 'ml_probability' in df.columns:
                    prob_series = df['ml_probability']
                else:
                    prob_series = pd.Series(0.5, index=df.index)

                # Safely extract EV
                if 'expected_value' in df.columns:
                    ev_series = df['expected_value']
                else:
                    ev_series = pd.Series(0.0, index=df.index)

                # Use robust vectorization conversions
                current_prob = pd.to_numeric(prob_series, errors='coerce').fillna(0.5)
                ev = pd.to_numeric(ev_series, errors='coerce').fillna(0.0)

                # Market agreement factor: picks where the model diverges far from the
                # bookmaker price get penalized. The old formula used |prob - 0.5| + EV
                # which simply measured model confidence â€” picks the model was wrongly
                # most confident about received the highest conviction scores.
                if 'market_probability' in df.columns:
                    mkt = pd.to_numeric(df['market_probability'], errors='coerce').fillna(0.5)
                else:
                    mkt = pd.Series(0.5, index=df.index)
                divergence = (current_prob - mkt).abs()
                market_agreement = (1.0 - (divergence / 0.30).clip(0.0, 1.0))
                base = (current_prob - 0.5).abs() + (ev * 2.0).clip(-0.2, 0.2)
                return (0.5 + base * market_agreement).clip(0.01, 0.99)
            except Exception as e:
                logger.warning(f"Failed to compute deterministic conviction: {e}")
                return pd.Series(0.5, index=df.index)

        # 1. Start with the deterministic fallback out of the gate so it never blanks
        analysis_df['Conviction_Score'] = _compute_deterministic_conviction(analysis_df)

        # Load historical outcomes explicitly to ensure full ground truth
        try:
            hist_df = pd.read_csv("data/master_all_sports.csv")
        except Exception:
            hist_df = pd.DataFrame()

        try:
            # We need to ensure we have the required columns for generating calibration dataset
            if 'market_type' not in hist_df.columns and 'best_pick_type' not in hist_df.columns and 'Market' not in hist_df.columns:
                # Add dummy market type if missing to trick generator to run
                hist_df['Market'] = 'SPREAD'

            # Map current slate to SPREAD or TOTAL generically
            analysis_df['generic_market'] = analysis_df['market_type'].astype(str).str.upper().apply(
                lambda x: 'TOTAL' if 'TOTAL' in x else 'SPREAD'
            )

            # Use appropriate probability column
            prob_col = 'final_probability' if 'final_probability' in hist_df.columns else 'calibrated_probability' if 'calibrated_probability' in hist_df.columns else 'ml_probability' if 'ml_probability' in hist_df.columns else None

            if prob_col:
                # Generate metrics
                bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
                calibration_metrics = generate_calibration_dataset(hist_df, probability_col=prob_col, bins=bins)

                if not calibration_metrics.empty:
                    # Bucket current slate predictions
                    analysis_df['prob_bucket'] = pd.cut(
                        pd.to_numeric(analysis_df['calibrated_probability'], errors='coerce').fillna(0.5),
                        bins=bins, right=False, include_lowest=True
                    ).astype(str)

                    # Convert metrics bucket to string for joining
                    calibration_metrics['bucket'] = calibration_metrics['bucket'].astype(str)

                    # Join calibration metrics
                    analysis_df = analysis_df.merge(
                        calibration_metrics[['league', 'market_type', 'bucket', 'empirical_win_rate']],
                        left_on=['league', 'generic_market', 'prob_bucket'],
                        right_on=['league', 'market_type', 'bucket'],
                        how='left',
                        suffixes=('', '_cal')
                    )

                    # Replace missing empirical rates with the previously calculated deterministic Conviction Score
                    # instead of forcing a full 1.0 match (which incorrectly claims 100% conviction for missing bins)
                    empirical_rate = pd.to_numeric(analysis_df['empirical_win_rate'], errors='coerce')
                    current_prob = pd.to_numeric(analysis_df['calibrated_probability'], errors='coerce').fillna(0.5)

                    # If an empirical rate exists for the bin, use it: Conviction = 1.0 - abs(prob - empirical)
                    # If it does not exist, stick with the deterministic base conviction we initialized earlier
                    has_empirical = empirical_rate.notna()
                    if has_empirical.any():
                        analysis_df.loc[has_empirical, 'Conviction_Score'] = 1.0 - (current_prob[has_empirical] - empirical_rate[has_empirical]).abs()

                    # Cleanup
                    drop_cols = ['prob_bucket', 'generic_market', 'market_type_cal', 'bucket', 'empirical_win_rate']
                    analysis_df = analysis_df.drop(columns=[c for c in drop_cols if c in analysis_df.columns], errors='ignore')

            # Final Safety Catch: If anything turned NaN, heal it back via deterministic calculation
            if analysis_df['Conviction_Score'].isna().any():
                deterministic = _compute_deterministic_conviction(analysis_df)
                analysis_df['Conviction_Score'] = analysis_df['Conviction_Score'].fillna(deterministic)

        except Exception as e:
            logger.warning(f"Failed to generate calibration metrics: {e}")

        finally:
             analysis_df = analysis_df.drop(columns=['generic_market'], errors='ignore')

    diagnostics["mlb_receipt_health"] = mlb_receipt_health
    diagnostics["loaded_model_identity"] = loaded_model_identity
    return (analysis_df, best_picks_df, diagnostics)


def generate_parlays(best_picks_df: pd.DataFrame, max_legs: int = 3) -> pd.DataFrame:
    from core.probability_calibration import load_calibration
    from core.smart_parlay_engine import (
        generate_probability_ranked_parlays,
        generate_smart_parlays,
        select_card_unique_parlays,
    )

    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    # Recap-fitted isotonic table (scripts/fit_calibration.py); None when absent,
    # in which case legs use raw effective_win_probability as before.
    production_frame = (
        "Pick_Status" in best_picks_df.columns
        and any(
            column in best_picks_df.columns
            for column in (
                "production_eligible",
                "market_line_source",
                "degraded_feature_subset_flag",
            )
        )
    )
    calibration = load_calibration() if production_frame else None
    parlays_df = generate_smart_parlays(best_picks_df, num_rr_candidates=5, calibration=calibration)
    probability_fallback = parlays_df.empty

    if probability_fallback:
        # A slate can have a valid best pick for every game while correctly
        # funding none of them. Keep the production gate intact, but populate a
        # clearly labeled $0 research board ranked by combined probability.
        parlays_df = generate_probability_ranked_parlays(best_picks_df)
        if parlays_df.empty:
            return parlays_df

    # Accuracy-first product contract: select from highest combined probability
    # downward while allowing each game to appear on the exported card once.
    # This considers the full combination pool before capping, so a repeated
    # top anchor cannot crowd every independent alternative out of the top 10.
    parlays_df = select_card_unique_parlays(parlays_df, max_parlays=10)
    if parlays_df.empty:
        return parlays_df
    parlays_df["parlay_rank"] = range(1, len(parlays_df) + 1)

    # These generators multiply individual leg prices. They do not retrieve an
    # executable sportsbook ticket, so no bankroll can justify funding them.
    parlays_df["kelly_fraction"] = 0.0
    parlays_df["recommended_bet"] = 0.0
    parlays_df["risk_tier"] = "Research"
    parlays_df["ticket_price_verified"] = False
    parlays_df["price_basis"] = "Estimated product of leg prices"
    parlays_df["wager_instruction"] = "RESEARCH ONLY: verify the actual ticket price and joint model before staking."

    def _shared_snapshot_value(columns: tuple[str, ...]) -> object:
        for column in columns:
            if column not in best_picks_df.columns:
                continue
            values = best_picks_df[column].dropna().astype(str).str.strip()
            values = values[~values.str.casefold().isin({"", "nan", "none", "<na>"})]
            unique = values.drop_duplicates()
            if len(unique) == 1:
                return unique.iloc[0]
        return pd.NA

    slate_date = _shared_snapshot_value(
        ("game_date", "slate_date", "date", "Local Date")
    )
    if pd.notna(slate_date):
        slate_date = str(slate_date)[:10]
    parlays_df["slate_date"] = slate_date
    parlays_df["export_run_id"] = _shared_snapshot_value(("export_run_id",))
    parlays_df["pipeline_build"] = _shared_snapshot_value(("pipeline_build",))

    # Persist the day's recommended parlays so they can be graded alongside the
    # slate. Recaps grade single picks only, so the parlay engine has never
    # received realized feedback; this log is the input for that. Last run of the
    # day wins, matching the card actually shown.
    try:
        log_dir = Path("data/parlay_log")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_slate_date = (
            str(slate_date)
            if pd.notna(slate_date) and str(slate_date).strip()
            else pd.Timestamp.now().strftime("%Y-%m-%d")
        )
        logged = parlays_df.copy()
        logged.insert(0, "generated_date", log_slate_date)
        logged.to_csv(log_dir / f"{log_slate_date}.csv", index=False)
    except Exception as e:
        logger.warning(f"Failed to write parlay log: {e}")

    return parlays_df

def optimize_portfolio_allocation(best_picks_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    if best_picks_df is not None and "wager_contract" in best_picks_df.columns:
        from core.live_wager_contract import enforce_frame
        return enforce_frame(best_picks_df)
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    portfolio = best_picks_df.copy()
    portfolio = portfolio[_string_series(portfolio, "best_pick").str.strip().str.len() > 0].copy()
    if portfolio.empty:
        return pd.DataFrame()
    if "Pick_Status" not in portfolio.columns:
        portfolio["Pick_Status"] = ""
    status = _string_series(portfolio, "Pick_Status").str.strip().str.lower()
    line_source = _string_series(portfolio, "market_line_source").str.strip().str.lower()
    line_warning = _string_series(portfolio, "line_provenance_warning").str.strip()
    best_pick_norm = _string_series(portfolio, "best_pick").str.strip().str.lower()
    line_used = pd.to_numeric(portfolio.get("market_line_used", pd.NA), errors="coerce")
    line_consistent = pd.Series(portfolio.get("line_consistency_flag", True), index=portfolio.index).fillna(True).astype(bool)
    event_identity_ok = pd.Series(portfolio.get("line_event_identity_match_flag", True), index=portfolio.index).fillna(True).astype(bool)
    # Production betting is suspended for fallback/degraded model rows. A
    # probability produced after model failure is a display diagnostic, not a
    # bettable forecast. This gate is intentionally conservative: if the
    # pipeline cannot prove the row came from the trained model with a clean
    # feature path, Kelly receives zero.
    model_status = _string_series(portfolio, "model_status").str.strip().str.lower()
    stats_source = _string_series(portfolio, "stats_source").str.strip().str.lower()
    fallback_summary = _string_series(portfolio, "fallback_summary_by_league").str.strip().str.lower()
    health_warning = _string_series(portfolio, "run_health_warning").str.strip().str.lower()
    row_league = _string_series(portfolio, "league").str.strip().str.upper()
    fallback_empty_tokens = {"", "{}", "{ }", "none", "nan", "<na>"}
    fallback_affects_row = pd.Series(False, index=portfolio.index, dtype=bool)
    keyed_summary_pattern = re.compile(r"""(?:['"][^'"]+['"]|[A-Za-z0-9_]+)\s*:""")
    for idx in portfolio.index:
        summary_value = str(fallback_summary.loc[idx]).strip()
        if summary_value in fallback_empty_tokens:
            continue
        league_value = str(row_league.loc[idx]).strip()
        if not league_value:
            fallback_affects_row.loc[idx] = True
            continue
        league_pattern = re.compile(
            rf"""(?:['"])?{re.escape(league_value)}(?:['"])?\s*:""",
            flags=re.IGNORECASE,
        )
        if league_pattern.search(summary_value):
            fallback_affects_row.loc[idx] = True
        elif not keyed_summary_pattern.search(summary_value):
            # Older, unstructured warnings cannot be safely attributed to a
            # different league, so retain the conservative production block.
            fallback_affects_row.loc[idx] = True
    fallback_health_warning = (
        health_warning.str.contains("fallback", regex=False, na=False)
        & fallback_affects_row
    )
    degraded = pd.Series(
        portfolio.get("degraded_feature_subset_flag", False),
        index=portfolio.index,
    ).fillna(False).astype(bool)
    untrusted_model = (
        model_status.isin({"statistical fallback", "neutral fallback", "model failure"})
        | stats_source.isin({"fallback", "failed"})
        | fallback_affects_row
        | fallback_health_warning
        | health_warning.str.contains("degraded|staking suspended", regex=True, na=False)
        | degraded
    )
    from core.market_policy import production_market
    production_market_ok = _string_series(portfolio, "market_type").map(production_market)
    production_eligible = (
        production_market_ok
        & status.eq("actionable")
        & line_source.eq("live")
        & line_warning.eq("")
        & line_used.notna()
        & line_consistent
        & event_identity_ok
        & (~best_pick_norm.str.contains("unresolved", na=False))
        & (~untrusted_model)
    )
    gemini_gate_enabled = pd.Series(
        portfolio.get("gemini_gate_enabled", False), index=portfolio.index
    ).fillna(False).astype(bool)
    gemini_approved = pd.Series(
        portfolio.get("gemini_approved", False), index=portfolio.index
    ).fillna(False).astype(bool)
    from app_core.gemini_bet_gate import gemini_gate_mask
    gemini_gate_ok = gemini_gate_mask(portfolio)
    # Enabling Gemini makes it a real secondary approval gate. It only narrows
    # deterministic eligibility; it can never promote a row on its own.
    production_eligible &= gemini_gate_ok
    portfolio["production_eligible"] = production_eligible

    portfolio["decimal_odds"] = _numeric_series(portfolio, "decimal_odds").fillna(
        _numeric_series(portfolio, "odds_american", -110.0).apply(american_to_decimal)
    )
    # App-generated rows use the same production probability that drove
    # status and EV. Legacy/imported rows without that field retain the prior
    # safety contract: empirical evidence first, otherwise a fitted effective
    # probability, and no stake when an effective value cannot be calibrated.
    production_p = pd.to_numeric(
        portfolio.get("production_win_probability", pd.Series(np.nan, index=portfolio.index)),
        errors="coerce",
    )
    selection_p = pd.to_numeric(
        portfolio.get("selection_probability_used", pd.Series(np.nan, index=portfolio.index)),
        errors="coerce",
    )
    empirical_p = pd.to_numeric(
        portfolio.get("empirical_win_probability", pd.Series(np.nan, index=portfolio.index)),
        errors="coerce",
    )
    controlled_value = pd.Series(
        portfolio.get("controlled_card_recovery", False),
        index=portfolio.index,
    ).fillna(False).astype(bool)
    effective_p = pd.to_numeric(
        portfolio.get("effective_win_probability", pd.Series(np.nan, index=portfolio.index)),
        errors="coerce",
    )
    legacy_calibrated_p = pd.to_numeric(
        portfolio.get("calibrated_probability", pd.Series(np.nan, index=portfolio.index)),
        errors="coerce",
    )
    fitted_p = pd.Series(np.nan, index=portfolio.index, dtype=float)
    try:
        from core.probability_calibration import (
            apply_bucket_calibration,
            apply_calibration,
            load_calibration,
        )
        from core.empirical_tiers import (
            bucket_key,
            bucket_stats_are_fresh,
            load_bucket_stats,
        )

        calibration = load_calibration()
        bucket_stats = load_bucket_stats()
        if not bucket_stats_are_fresh(bucket_stats):
            bucket_stats = None
        if calibration:
            if bucket_stats:
                buckets = [
                    bucket_key(league, market, consensus)
                    for league, market, consensus in zip(
                        portfolio.get("league", pd.Series("", index=portfolio.index)),
                        portfolio.get("market_type", pd.Series("", index=portfolio.index)),
                        portfolio.get("consensus_agreement", pd.Series("", index=portfolio.index)),
                    )
                ]
                fitted_p = pd.to_numeric(
                    apply_bucket_calibration(effective_p, buckets, calibration, bucket_stats),
                    errors="coerce",
                )
            else:
                fitted_p = pd.to_numeric(
                    apply_calibration(effective_p, calibration), errors="coerce"
                )
    except Exception as exc:
        logger.warning(
            "Kelly calibration unavailable; legacy effective-probability rows will not be staked: %s",
            exc,
        )

    p = production_p.copy()
    probability_source = pd.Series(
        "production_win_probability", index=portfolio.index, dtype="object"
    )
    selection_mask = p.isna() & selection_p.notna()
    p.loc[selection_mask] = selection_p.loc[selection_mask]
    probability_source.loc[selection_mask] = "selection_probability_used"
    empirical_mask = p.isna() & empirical_p.notna()
    p.loc[empirical_mask] = empirical_p.loc[empirical_mask]
    probability_source.loc[empirical_mask] = "empirical_win_probability"
    fitted_mask = p.isna() & fitted_p.notna()
    p.loc[fitted_mask] = fitted_p.loc[fitted_mask]
    probability_source.loc[fitted_mask] = "fitted_effective_probability"
    legacy_mask = p.isna() & effective_p.isna() & legacy_calibrated_p.notna()
    p.loc[legacy_mask] = legacy_calibrated_p.loc[legacy_mask]
    probability_source.loc[legacy_mask] = "legacy_calibrated_probability"
    # Empty-card recovery is admitted by a stricter empirical, exact-price
    # gate after the normal production card is empty. On its second portfolio
    # pass, size and validate those marked rows with that same probability.
    # Reusing production_win_probability here silently reapplies the legacy
    # edge basis and can zero a row that the controlled-value gate just
    # promoted (Aug. 5: Yankees -1.5 +141).
    controlled_empirical = controlled_value & empirical_p.notna()
    p.loc[controlled_empirical] = empirical_p.loc[controlled_empirical]
    probability_source.loc[controlled_empirical] = (
        "controlled_value_empirical_price_probability"
    )
    missing_probability = p.isna()
    probability_source.loc[missing_probability] = "missing_fitted_calibration"
    production_eligible = production_eligible & (~missing_probability)
    portfolio["production_eligible"] = production_eligible
    p = p.fillna(0.0).clip(lower=0.0, upper=1.0)

    portfolio["kelly_uncalibrated_probability"] = effective_p.fillna(legacy_calibrated_p)
    portfolio["kelly_probability_source"] = probability_source
    b = (portfolio["decimal_odds"] - 1.0).clip(lower=0.0)
    q = 1.0 - p
    kelly_fraction = pd.Series(0.0, index=portfolio.index, dtype=float)
    valid = b > 0
    kelly_fraction.loc[valid] = (((b.loc[valid] * p.loc[valid]) - q.loc[valid]) / b.loc[valid]).clip(lower=0.0)
    portfolio["kelly_probability_used"] = p
    portfolio["kelly_decimal_odds"] = portfolio["decimal_odds"]
    # Defense in depth: callers may pass a pre-built frame directly to the
    # portfolio allocator. Re-apply the same absolute production gate here so an
    # Actionable label alone can never produce dollars.
    from core.production_gate import evaluate_absolute_production_gate
    portfolio_break_even = (1.0 / portfolio["decimal_odds"]).replace([np.inf, -np.inf], np.nan)
    # Prefer the most production-specific EV, but fall back row by row.  A
    # present-but-sparse production_expected_value column must not hide a valid
    # effective/legacy EV on the same row.
    portfolio_model_ev = pd.to_numeric(
        portfolio.get(
            "production_expected_value",
            pd.Series(np.nan, index=portfolio.index),
        ),
        errors="coerce",
    )
    portfolio_model_ev = portfolio_model_ev.fillna(
        pd.to_numeric(
            portfolio.get(
                "effective_expected_value",
                pd.Series(np.nan, index=portfolio.index),
            ),
            errors="coerce",
        )
    )
    portfolio_model_ev = portfolio_model_ev.fillna(
        pd.to_numeric(
            portfolio.get(
                "expected_value",
                pd.Series(np.nan, index=portfolio.index),
            ),
            errors="coerce",
        )
    )
    portfolio_gate = evaluate_absolute_production_gate(
        p,
        portfolio_break_even,
        portfolio_model_ev,
    )
    # Defense in depth for callers that pass a controlled marker directly:
    # retain the recovery price range and require its larger 3-point margin
    # when consensus Disagrees. Non-controlled production rows are unchanged.
    from app_core.card_recovery import controlled_value_price_gate
    from app_core.weights_config import (
        EMPTY_CARD_RECOVERY_DISAGREES_MIN_ABSOLUTE_EDGE,
        EMPTY_CARD_RECOVERY_MAX_AMERICAN_ODDS,
        EMPTY_CARD_RECOVERY_MIN_ABSOLUTE_EDGE,
        EMPTY_CARD_RECOVERY_MIN_AMERICAN_ODDS,
        EMPTY_CARD_RECOVERY_MIN_PRODUCTION_WIN_PROB,
    )
    controlled_gate = controlled_value_price_gate(
        empirical_p,
        portfolio.get("odds_american", pd.Series(np.nan, index=portfolio.index)),
        portfolio.get("consensus_agreement", pd.Series("", index=portfolio.index)),
        min_absolute_edge=float(EMPTY_CARD_RECOVERY_MIN_ABSOLUTE_EDGE),
        disagrees_min_absolute_edge=float(
            EMPTY_CARD_RECOVERY_DISAGREES_MIN_ABSOLUTE_EDGE
        ),
        min_american_odds=float(EMPTY_CARD_RECOVERY_MIN_AMERICAN_ODDS),
        max_american_odds=float(EMPTY_CARD_RECOVERY_MAX_AMERICAN_ODDS),
    )
    controlled_price_pass = controlled_gate[
        "controlled_value_price_gate_pass"
    ].reindex(portfolio.index).fillna(False)
    controlled_probability_pass = empirical_p.ge(
        float(EMPTY_CARD_RECOVERY_MIN_PRODUCTION_WIN_PROB)
    ).fillna(False)
    portfolio_gate.loc[controlled_value, "production_gate_pass"] &= (
        controlled_price_pass.loc[controlled_value]
        & controlled_probability_pass.loc[controlled_value]
    )
    failed_controlled_probability = controlled_value & ~controlled_probability_pass
    portfolio_gate.loc[
        failed_controlled_probability, "production_gate_reason"
    ] = "controlled value win-probability floor failed"
    failed_controlled_price = (
        controlled_value
        & controlled_probability_pass
        & ~controlled_price_pass
    )
    portfolio_gate.loc[
        failed_controlled_price, "production_gate_reason"
    ] = "controlled value exact-price gate failed"
    portfolio["absolute_production_edge"] = portfolio_gate["absolute_production_edge"]
    portfolio["absolute_production_gate_pass"] = portfolio_gate["production_gate_pass"]
    portfolio["absolute_production_gate_reason"] = portfolio_gate["production_gate_reason"]
    production_eligible = production_eligible & portfolio_gate["production_gate_pass"]
    portfolio["production_eligible"] = production_eligible
    portfolio["kelly_fraction"] = kelly_fraction
    portfolio["raw_kelly_amount"] = float(bankroll) * kelly_fraction
    from app_core.weights_config import (
        PRODUCTION_ABSOLUTE_MAX_PICK_DOLLARS,
        PRODUCTION_ABSOLUTE_MAX_SLATE_DOLLARS,
        PRODUCTION_KELLY_MULTIPLIER,
        PRODUCTION_MAX_PICK_PCT,
        PRODUCTION_MAX_SLATE_PCT,
    )

    portfolio["fractional_kelly_amount"] = portfolio["raw_kelly_amount"] * float(PRODUCTION_KELLY_MULTIPLIER)
    portfolio["recommended_bet"] = portfolio["fractional_kelly_amount"]
    portfolio.loc[~portfolio["production_eligible"], "recommended_bet"] = 0.0

    max_pick = max(
        0.0,
        min(
            float(bankroll) * float(PRODUCTION_MAX_PICK_PCT),
            float(PRODUCTION_ABSOLUTE_MAX_PICK_DOLLARS),
        ),
    )
    max_slate = max(
        0.0,
        min(
            float(bankroll) * float(PRODUCTION_MAX_SLATE_PCT),
            float(PRODUCTION_ABSOLUTE_MAX_SLATE_DOLLARS),
        ),
    )
    portfolio["kelly_cap_reason"] = ""
    portfolio.loc[~portfolio["production_eligible"], "kelly_cap_reason"] = "Non-production row"
    eligible = portfolio["production_eligible"]
    eligible_total = float(portfolio.loc[eligible, "recommended_bet"].sum())
    scale = min(1.0, (max_slate / eligible_total) if eligible_total > 0 else 1.0)
    portfolio["kelly_weight_share"] = 0.0
    if eligible_total > 0:
        portfolio.loc[eligible, "kelly_weight_share"] = portfolio.loc[eligible, "recommended_bet"] / eligible_total
    portfolio["slate_scaled_amount"] = portfolio["recommended_bet"] * scale
    portfolio.loc[(scale < 1.0) & eligible, "kelly_cap_reason"] = "Scaled by slate exposure"
    pre_pick = portfolio["slate_scaled_amount"].copy()
    portfolio["production_bet_amount"] = portfolio["slate_scaled_amount"].clip(upper=max_pick)
    capped = eligible & (pre_pick > max_pick)
    portfolio.loc[capped, "kelly_cap_reason"] = portfolio.loc[capped, "kelly_cap_reason"].replace("", "Capped by pick exposure")
    portfolio["recommended_bet"] = portfolio["production_bet_amount"]
    portfolio["kelly_allocation_method"] = "proportional_fractional_kelly"

    # Non-Actionable Kelly: independent per-status fractions so bet sizes stay
    # meaningful on thin slates (not tied to Actionable total).
    # HV: 0.075x Kelly (30% of Actionable's 0.25), BT: 0.050x (20% of 0.25).
    # Slate-level safety: if non-Actionable total > 30% of combined, scale down.
    from app_core.weights_config import (
        NON_ACTIONABLE_KELLY_SHARE, HIGH_VARIANCE_KELLY_FRACTION,
        BELOW_THRESHOLD_KELLY_FRACTION, NON_ACTIONABLE_MAX_PICK_PCT,
        NON_ACTIONABLE_BELOW_THRESHOLD_MAX_PICK_PCT,
    )
    is_hv = status.eq("high variance/speculative")
    is_bt = status.eq("below threshold")
    na_eligible = is_hv | is_bt

    if float(NON_ACTIONABLE_KELLY_SHARE) <= 0:
        # Non-Actionable staking disabled (16 Jun): confine real stakes to the proven
        # Actionable (Agrees-bucket) tier. High Variance / Below Threshold still surface
        # for visibility but carry NO production stake â€” their production_bet_amount /
        # recommended_bet are already 0 and kelly_cap_reason "Non-production row" from the
        # eligibility pass above. Evidence: those tiers ran sub-break-even over graded
        # history, so staking them (even fractionally) is -EV.
        portfolio.loc[na_eligible, "production_bet_amount"] = 0.0
        portfolio.loc[na_eligible, "recommended_bet"] = 0.0
        portfolio["non_actionable_eligible"] = na_eligible
    else:
        na_kelly_frac = pd.Series(0.0, index=portfolio.index)
        na_valid = (b > 0) & na_eligible
        na_kelly_frac.loc[na_valid] = (
            ((b.loc[na_valid] * p.loc[na_valid]) - q.loc[na_valid]) / b.loc[na_valid]
        ).clip(lower=0.0)

        na_raw = float(bankroll) * na_kelly_frac
        hv_max = float(bankroll) * float(NON_ACTIONABLE_MAX_PICK_PCT)
        bt_max = float(bankroll) * float(NON_ACTIONABLE_BELOW_THRESHOLD_MAX_PICK_PCT)

        na_bet = pd.Series(0.0, index=portfolio.index)
        na_bet.loc[is_hv] = (na_raw.loc[is_hv] * float(HIGH_VARIANCE_KELLY_FRACTION)).clip(upper=hv_max)
        na_bet.loc[is_bt] = (na_raw.loc[is_bt] * float(BELOW_THRESHOLD_KELLY_FRACTION)).clip(upper=bt_max)

        # Slate cap: non-Actionable total must not exceed its share of combined total
        a_total = float(portfolio.loc[eligible, "production_bet_amount"].sum())
        na_total = float(na_bet.loc[na_eligible].sum())
        combined = a_total + na_total
        if combined > 0 and na_total > 0:
            target_na = combined * float(NON_ACTIONABLE_KELLY_SHARE)
            if na_total > target_na:
                na_bet = na_bet * (target_na / na_total)

        portfolio.loc[na_eligible, "production_bet_amount"] = na_bet.loc[na_eligible].round(2)
        portfolio.loc[na_eligible, "recommended_bet"] = na_bet.loc[na_eligible].round(2)
        portfolio.loc[is_hv & na_bet.gt(0), "kelly_cap_reason"] = "High Variance Kelly (0.075x)"
        portfolio.loc[is_bt & na_bet.gt(0), "kelly_cap_reason"] = "Below Threshold Kelly (0.050x)"
        portfolio.loc[na_eligible, "kelly_weight_share"] = (
            na_bet.loc[na_eligible] / combined if combined > 0 else 0.0
        )
        portfolio["non_actionable_eligible"] = na_eligible

    # --- Force-deploy daily stake budget (user-directed) ----------------------
    # Override the fractional-Kelly amounts so the day's card sums to a fixed dollar
    # budget, split Actionable vs viable non-Actionable. See weights_config. Runs last
    # so it supersedes both the Actionable slate/pick caps and the non-Actionable block.
    try:
        from app_core.weights_config import (
            DAILY_STAKE_FORCE_DEPLOY, DAILY_STAKE_BUDGET, ACTIONABLE_STAKE_SHARE,
            FORCE_DEPLOY_NONACTIONABLE_INCLUDE_BELOW_THRESHOLD, FORCE_DEPLOY_MAX_PICK_PCT,
            FORCE_DEPLOY_NONACTIONABLE_CONSENSUS,
        )
    except Exception:
        DAILY_STAKE_FORCE_DEPLOY = False
    if DAILY_STAKE_FORCE_DEPLOY:
        _health = _string_series(portfolio, "run_health_warning").str.lower()
        _staking_suspended = bool(_health.str.contains("staking suspended", na=False).any())
        # Data-integrity gate (same checks as production_eligible, minus the status
        # requirement) so we never force a stake onto an unsafe/unresolved line.
        _data_safe = (
            line_source.eq("live") & line_warning.eq("") & line_used.notna()
            & line_consistent & event_identity_ok
            & (~best_pick_norm.str.contains("unresolved", na=False))
            & (~untrusted_model)
            & gemini_gate_ok
        )
        _act_tier = _data_safe & status.eq("actionable")
        # Non-Actionable staking tier: High Variance only by default. Below Threshold
        # picks failed the thresholds outright, so they carry no forced stake unless
        # explicitly opted in.
        _nonact_status = na_eligible if FORCE_DEPLOY_NONACTIONABLE_INCLUDE_BELOW_THRESHOLD else is_hv
        # Never stake AGAINST Kalshi: gate the non-Actionable tier by consensus so
        # "Disagrees" picks (Kalshi backs the other side) carry no forced stake. See
        # FORCE_DEPLOY_NONACTIONABLE_CONSENSUS in weights_config.
        _consensus = _string_series(portfolio, "consensus_agreement")
        _consensus_ok = _consensus.isin(list(FORCE_DEPLOY_NONACTIONABLE_CONSENSUS))
        _nonact_tier = _data_safe & _nonact_status & _consensus_ok

        # Per-pick concentration cap. Excess above the cap is NOT redistributed, so a
        # tier with too few picks under-deploys instead of dumping the budget onto one.
        _max_pick = float(DAILY_STAKE_BUDGET) * float(FORCE_DEPLOY_MAX_PICK_PCT)

        def _fill_budget(mask: pd.Series, budget: float) -> None:
            idx = portfolio.index[mask.fillna(False)]
            if len(idx) == 0 or budget <= 0:
                return
            w = pd.to_numeric(portfolio.loc[idx, "kelly_fraction"], errors="coerce").fillna(0.0).clip(lower=0.0)
            if float(w.sum()) <= 0:
                return  # never force money onto a row with no positive calibrated edge
            alloc = ((w / float(w.sum())) * float(budget)).clip(upper=_max_pick)
            portfolio.loc[idx, "production_bet_amount"] = alloc.round(2)
            portfolio.loc[idx, "recommended_bet"] = alloc.round(2)

        if not _staking_suspended:
            _fill_budget(_act_tier, float(DAILY_STAKE_BUDGET) * float(ACTIONABLE_STAKE_SHARE))
            _fill_budget(_nonact_tier, float(DAILY_STAKE_BUDGET) * (1.0 - float(ACTIONABLE_STAKE_SHARE)))
            _in_tier = (_act_tier | _nonact_tier).fillna(False)
            portfolio.loc[~_in_tier, "production_bet_amount"] = 0.0
            portfolio.loc[~_in_tier, "recommended_bet"] = 0.0
            portfolio.loc[_act_tier, "kelly_cap_reason"] = "Force-deploy daily budget (Actionable 60%)"
            portfolio.loc[_nonact_tier, "kelly_cap_reason"] = "Force-deploy daily budget (non-Actionable 40%)"
        else:
            # Health-suspended slate: deploy nothing, regardless of force-deploy.
            portfolio["production_bet_amount"] = 0.0
            portfolio["recommended_bet"] = 0.0
            portfolio["kelly_cap_reason"] = "Force-deploy suspended (slate health guard)"
        portfolio["kelly_allocation_method"] = "force_deploy_daily_budget"

    # Absolute value is a final non-bypassable funding requirement. In
    # particular, the optional force-deploy allocator must not reintroduce
    # dollars on a row that the calibrated price gate rejected.
    absolute_pass = portfolio["absolute_production_gate_pass"].fillna(False).astype(bool)
    portfolio.loc[~absolute_pass, "production_bet_amount"] = 0.0
    portfolio.loc[~absolute_pass, "recommended_bet"] = 0.0
    portfolio.loc[~absolute_pass, "production_eligible"] = False
    portfolio.loc[~absolute_pass, "kelly_cap_reason"] = "Absolute production value gate"

    # Final, non-bypassable production ceilings. This runs after every optional
    # allocation mode so neither force-deploy nor a future sizing branch can
    # exceed the bankroll-relative and absolute risk limits.
    pre_hard_cap = pd.to_numeric(portfolio["production_bet_amount"], errors="coerce").fillna(0.0).clip(lower=0.0)
    hard_pick_hits = pre_hard_cap > max_pick
    portfolio["production_bet_amount"] = pre_hard_cap.clip(upper=max_pick)
    if hard_pick_hits.any():
        prior = portfolio.loc[hard_pick_hits, "kelly_cap_reason"].fillna("").astype(str)
        portfolio.loc[hard_pick_hits, "kelly_cap_reason"] = np.where(
            prior.str.len().gt(0),
            prior + "; final pick ceiling",
            "Final pick ceiling",
        )

    hard_total = float(portfolio["production_bet_amount"].sum())
    final_slate_scale = min(1.0, (max_slate / hard_total) if hard_total > 0 else 1.0)
    if final_slate_scale < 1.0:
        positive_before_scale = portfolio["production_bet_amount"] > 0
        portfolio["production_bet_amount"] = portfolio["production_bet_amount"] * final_slate_scale
        prior = portfolio.loc[positive_before_scale, "kelly_cap_reason"].fillna("").astype(str)
        portfolio.loc[positive_before_scale, "kelly_cap_reason"] = np.where(
            prior.str.len().gt(0),
            prior + "; final slate ceiling",
            "Final slate ceiling",
        )

    # Gemini sizing runs last and can only reduce exposure. HIGH keeps the
    # deterministic amount, MEDIUM uses 75%, and every non-approved review is
    # held at $0. Applying this after optional force-deploy logic makes the gate
    # non-bypassable.
    gemini_multiplier = pd.to_numeric(
        pd.Series(
            portfolio.get("gemini_stake_multiplier", 1.0),
            index=portfolio.index,
        ),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    gemini_multiplier = gemini_multiplier.where(gemini_gate_enabled, 1.0)
    pre_gemini_amount = pd.to_numeric(
        portfolio["production_bet_amount"], errors="coerce"
    ).fillna(0.0)
    portfolio["production_bet_amount"] = (
        pre_gemini_amount * gemini_multiplier
    ).where(gemini_gate_ok, 0.0)
    outage_cap = pd.to_numeric(pd.Series(portfolio.get('gemini_outage_cap_fraction',0),index=portfolio.index),errors='coerce').fillna(0).clip(0,.01)
    outage_rows = pd.Series(portfolio.get('gemini_review_status',''),index=portfolio.index).eq('OUTAGE_CAPPED')
    portfolio.loc[outage_rows,'production_bet_amount'] = pd.concat([portfolio['production_bet_amount'], outage_cap * bankroll],axis=1).min(axis=1).loc[outage_rows]
    portfolio.loc[~gemini_gate_ok, "production_eligible"] = False
    portfolio.loc[~gemini_gate_ok, "kelly_cap_reason"] = "Gemini review hold"
    medium_reduction = (
        gemini_gate_enabled
        & gemini_gate_ok
        & gemini_multiplier.lt(1.0)
        & pre_gemini_amount.gt(0.0)
        & ~outage_rows
    )
    if medium_reduction.any():
        prior = portfolio.loc[medium_reduction, "kelly_cap_reason"].fillna("").astype(str)
        portfolio.loc[medium_reduction, "kelly_cap_reason"] = np.where(
            prior.str.len().gt(0),
            prior + "; Gemini MEDIUM 75% multiplier",
            "Gemini MEDIUM 75% multiplier",
        )
    portfolio.loc[outage_rows,"kelly_cap_reason"] = "Gemini outage: reduced straight-only cap; not Gemini approval"
    portfolio["production_bet_amount"] = portfolio["production_bet_amount"].round(2)
    portfolio["recommended_bet"] = portfolio["production_bet_amount"]
    capped = capped | hard_pick_hits
    scale *= final_slate_scale

    positive = portfolio["production_bet_amount"] > 0
    unique_positive = int(portfolio.loc[positive, "production_bet_amount"].round(6).nunique())
    cap_hits = int(capped.sum())
    portfolio["kelly_unique_positive_amount_count"] = unique_positive
    portfolio["kelly_max_pick_cap_hits"] = cap_hits
    portfolio["kelly_slate_scale_factor"] = scale
    portfolio["kelly_total_raw_amount"] = float(portfolio["raw_kelly_amount"].sum())
    portfolio["kelly_total_fractional_amount"] = float(portfolio["fractional_kelly_amount"].sum())
    portfolio["kelly_total_production_amount"] = float(portfolio["production_bet_amount"].sum())
    portfolio["kelly_flattening_detected"] = bool(unique_positive <= 1 and positive.any() and cap_hits >= int(positive.sum()))

    cols = [
        "league", "home_team", "away_team", "best_pick",
        "calibrated_probability", "expected_value", "edge",
        "decimal_odds", "raw_kelly_amount", "production_bet_amount", "recommended_bet", "kelly_cap_reason", "Pick_Status",
        "kelly_uncalibrated_probability", "kelly_probability_used", "kelly_probability_source",
        "kelly_decimal_odds", "kelly_fraction", "fractional_kelly_amount", "kelly_weight_share", "slate_scaled_amount",
        "kelly_allocation_method", "kelly_flattening_detected", "kelly_unique_positive_amount_count", "kelly_total_raw_amount",
        "kelly_total_fractional_amount", "kelly_total_production_amount", "kelly_max_pick_cap_hits", "kelly_slate_scale_factor",
        "market_line_used", "market_line_source", "line_consistency_flag", "line_event_identity_match_flag", "line_provenance_warning",
        "export_run_id", "pick_id", "canonical_pick_key", "production_eligible", "non_actionable_eligible",
        "absolute_production_edge", "absolute_production_gate_pass", "absolute_production_gate_reason",
    ]
    for col in cols:
        if col not in portfolio.columns:
            portfolio[col] = pd.NA
    cols += [key for key in ("gemini_gate_enabled","gemini_approved","gemini_review_status","gemini_outage_allowed","gemini_outage_cap_fraction","gemini_stake_multiplier","maturity") if key in portfolio.columns]
    return portfolio[cols].sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=30, simulations=200)
