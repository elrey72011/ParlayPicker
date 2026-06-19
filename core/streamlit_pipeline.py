from __future__ import annotations

import functools
import re
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any
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
from core.team_mapper import normalize_team_name, NBA_EXACT_MAP, NHL_EXACT_MAP
from app_core.weights_config import (
            TOTAL_UNDER_MIN_WIN_PROB, TOTAL_UNDER_MIN_EV, TOTAL_UNDER_MIN_EDGE,
            NHL_TOTAL_EXTRA_EDGE_PENALTY, MLB_SPREAD_MIN_WIN_PROB,
            MLB_SPREAD_ACTIONABLE_PENALTY, MLB_SPREAD_FINALIST_SCORE_PENALTY,
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
    NHL_KALSHI_WEIGHT, NHL_ML_MODEL_WEIGHT, NHL_MARKET_WEIGHT, NHL_THEOVER_WEIGHT, NHL_SENTIMENT_WEIGHT,
    KALSHI_DIVERGENCE_THRESHOLD, KALSHI_DIVERGENCE_THRESHOLD_NBA,
    KALSHI_DIVERGENCE_THRESHOLD_MLB, KALSHI_DIVERGENCE_THRESHOLD_NHL,
    MLB_THEOVER_CONFLICT_THRESHOLD, MLB_THEOVER_CONFLICT_PENALTY,
    MLB_THEOVER_FADE_SOURCES, MLB_THEOVER_FADE_SHRINK, THEOVER_FADE_SHRINK_DEFAULT,
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
    key = st.secrets.get("ODDS_API_KEY")
    if not key:
        key = os.environ.get("ODDS_API_KEY", "")
    return key

try:
    from app_core.prediction_engine import PredictionEngine, get_cached_prediction_engine
    ML_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import PredictionEngine: {e}")
    ML_AVAILABLE = False
    PredictionEngine = None

VALID_MARKETS = {"spread_home", "spread_away", "total_over", "total_under"}
DATE_ALIASES = ["game_date", "game_date_est", "commence_time", "start_time", "time", "date", "event_date"]
LEAGUE_ALIASES = {"NCAAM": "NCAAB", "NCAA MEN'S BASKETBALL": "NCAAB", "NCAA MENS BASKETBALL": "NCAAB"}
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
PIPELINE_BUILD = "2026-06-19-spread-orient-diag"


REQUIRED_BEST_PICK_EXPORT_COLUMNS = [
    "pipeline_build",
    "status_metric_basis",
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
    # Kalshi match instrumentation — diagnose a systematic over-bias from the export:
    # the Kalshi contract line used, its distance from the pick line, and the raw
    # P(over) before pick-side orientation / proxy decay.
    "kalshi_matched_line",
    "kalshi_line_diff",
    "kalshi_raw_over_prob",
    # Raw Kalshi contract fields — confirm strike semantics vs our matched_line, and
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
    req_cols = list(required_columns or REQUIRED_BEST_PICK_EXPORT_COLUMNS)
    default_values: dict[str, object] = {
        "status_metric_basis": "raw",
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
        if col in {"status_blocker_reason", "status_blocker_stage", "nba_stats_fetch_status", "fallback_summary_by_league", "run_health_warning", "degraded_feature_subset_reason", "status_metric_basis", "market_line_source", "market_line_source_detail", "line_consistency_reason", "line_provenance_warning", "line_event_identity_reason", "live_event_match_key", "selected_live_event_source", "raw_book_odds_diag"}:
            out[col] = out[col].fillna(default_values.get(col, "")).astype(str)

    if "status_blocker_stage" in out.columns:
        out["status_blocker_stage"] = out["status_blocker_stage"].replace({"": "none"})
    if "degraded_feature_subset_flag" in out.columns:
        out["degraded_feature_subset_flag"] = out["degraded_feature_subset_flag"].fillna(False).astype(bool)
    if "totals_only_actionable_flag" in out.columns:
        out["totals_only_actionable_flag"] = out["totals_only_actionable_flag"].fillna(False).astype(bool)
    if "viable_side_candidates_count" in out.columns:
        out["viable_side_candidates_count"] = pd.to_numeric(out["viable_side_candidates_count"], errors="coerce").fillna(0).astype(int)
    if "side_promoted_by_balance_guard_count" in out.columns:
        out["side_promoted_by_balance_guard_count"] = pd.to_numeric(out["side_promoted_by_balance_guard_count"], errors="coerce").fillna(0).astype(int)
    if "side_balance_guard_reason" in out.columns:
        out["side_balance_guard_reason"] = out["side_balance_guard_reason"].fillna("MISSING_COMPUTATION").astype(str)
    for numeric_col in {"market_line_used", "matched_live_spread_line", "matched_live_total_line", "upload_spread_line", "upload_total_line", "base_spread_line", "base_total_line"}:
        if numeric_col in out.columns:
            out[numeric_col] = pd.to_numeric(out[numeric_col], errors="coerce")
    if "line_consistency_flag" in out.columns:
        out["line_consistency_flag"] = out["line_consistency_flag"].fillna(True).astype(bool)
    if "line_event_identity_match_flag" in out.columns:
        out["line_event_identity_match_flag"] = out["line_event_identity_match_flag"].fillna(True).astype(bool)
    if "line_candidate_count" in out.columns:
        out["line_candidate_count"] = pd.to_numeric(out["line_candidate_count"], errors="coerce").fillna(0).astype(int)
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
            if diag_status in {"live", "cached", "failed"}:
                out["nba_stats_fetch_status"] = row_status.mask(~row_status.isin({"live", "cached", "failed"}), diag_status)
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
    "decimal_odds", "matchup_id",
    "odds_american", "odds_source", "market_probability", "ml_probability", "theover_probability", "win_prob_source", "display_probability",
    "kalshi_probability", "kalshi_match_status", "kalshi_match_reason",
    # Kalshi match instrumentation: the contract line actually used, its distance from
    # the pick line, and the raw P(over) before orientation/decay — for diagnosing a
    # systematic over-bias straight from the export.
    "kalshi_matched_line", "kalshi_line_diff", "kalshi_raw_over_prob",
    "kalshi_market_title", "kalshi_floor_strike", "kalshi_cap_strike", "kalshi_yes_bid", "kalshi_yes_ask",
    # Exact signal values fed to compute_blended_probability, oriented to the pick
    # side. Persisted so the blend weights can be backtested/fitted from saved
    # exports without having to re-derive orientation (which is ambiguous after
    # the fact). See scripts/fit_blend_weights.py.
    "blend_in_kalshi", "blend_in_market", "blend_in_ml", "blend_in_theover", "blend_tier",
    # Readable per-signal win-% breakdown (Kalshi/Market/ML/TheOver) — see REQUIRED_BEST_PICK_EXPORT_COLUMNS.
    "signal_breakdown",
    "gemini_explanation", "gemini_risk_notes", "used_stale_features", "Pick_Quality", "Conviction_Score",
    "uploaded_spread_line", "uploaded_total_line", "live_spread_line", "live_total_line", "line_source", "line_delta", "upload_market_match",
    "market_line_used", "market_line_source", "market_line_source_detail", "matched_live_spread_line", "matched_live_total_line", "upload_spread_line", "upload_total_line", "base_spread_line", "base_total_line",
    "line_consistency_flag", "line_consistency_reason", "line_provenance_warning", "line_event_identity_match_flag", "line_event_identity_reason", "live_event_match_key", "line_candidate_count", "selected_live_event_source",
    # Moneyline (h2h) prices carried onto every bet row for the spread-orientation
    # guard. Exported so we can see whether the guard actually has its inputs: real
    # values mean the moneyline reached build_best_picks_df (guard can fire); blank
    # on a spread row means there was no h2h to verify orientation against.
    "game_home_ml_price", "game_away_ml_price",
    "suspicious_data_flag", "suspicious_data_reasons", "status_metric_basis", "effective_expected_value", "effective_edge", "effective_win_probability", "status_blocker_reason", "status_blocker_stage",
    "nba_stats_fetch_status", "nba_stats_fetch_source", "nba_stats_fetch_retries_used", "stats_source_counts", "fallback_summary_by_league", "fallback_heavy_slate_flag", "run_health_warning",
    "degraded_feature_subset_flag", "degraded_feature_subset_reason",
    "actionable_family_counts", "totals_only_actionable_flag", "viable_side_candidates_count", "side_promoted_by_balance_guard_count", "side_balance_guard_reason",
    "production_win_probability", "production_expected_value", "production_edge", "probability_calibration_reason", "production_eligible",
    "raw_kelly_amount", "production_bet_amount", "kelly_cap_reason", "kelly_zero_reason",
    "export_run_id", "pick_id", "canonical_pick_key",
]

CANONICAL_BET_COLUMNS = [
    "league", "home_team", "away_team", "game_date", "game_time_est", "game_key",
    "market_type", "candidate_source", "orientation_source", "upload_match_reason", "spread_line", "total_line",
    "theover_probability", "win_prob_source", "odds_american", "odds_source", "market_probability",
    "ml_probability", "display_probability", "calibrated_probability", "expected_value", "edge", "best_pick", "used_stale_features", "matchup_id", "Conviction_Score",
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
    ``kalshi_probability`` is populated by the time the export is assembled — so
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


def _coerce_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    matchup_text = _first_nonempty_text(out, ["matchup", "match_up", "event", "event_name", "teams", "game"])
    if matchup_text.str.len().eq(0).all():
        sep_probe = r"(?i)(?:@|\bvs\b|\bv\b|\bat\b|[-—])"
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
    sep_pattern = r"(?i)\s*(?:@|vs\.?|v\.?|at|[-—])\s*"
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
    out["home_team"] = home_fallback.map(normalize_team_name)
    out["away_team"] = away_fallback.map(normalize_team_name)
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
    """Fill missing league labels using known pro-team sets, defaulting remaining blanks to NCAAB when selected."""
    out = df.copy()
    if out.empty:
        return out

    out["league"] = _clean_text_placeholders(_string_series(out, "league")).str.upper().replace(LEAGUE_ALIASES)
    missing_mask = out["league"].str.len().eq(0)
    if not missing_mask.any():
        return out

    # 1. Check NCAAB keyword recovery regex FIRST to prevent college teams from being swallowed by pro city names
    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_LEAGUE_RECOVERY_KEYWORDS)) + r")\b"
    home_text = _clean_text_placeholders(_string_series(out, "home_team")).str.lower()
    away_text = _clean_text_placeholders(_string_series(out, "away_team")).str.lower()
    keyword_mask = home_text.str.contains(keyword_pattern, regex=True, na=False) | away_text.str.contains(keyword_pattern, regex=True, na=False)
    out.loc[missing_mask & keyword_mask, "league"] = "NCAAB"

    # Refresh missing mask after NCAAB assignment
    missing_mask = out["league"].str.len().eq(0)

    # 2. Precedence Override: Check NBA/NHL exact map
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
    """Normalize merge keys to reduce cross-source drift and Pandas NA ambiguity."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for key in keys:
        if key not in out.columns:
            continue
        out[key] = _clean_text_placeholders(_string_series(out, key)).astype("string").str.strip()
    return out


def _numeric_series(df: pd.DataFrame, col: str, default: float | int | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if col in df.columns:
        out = pd.to_numeric(df[col], errors="coerce")
    else:
        out = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    if default is not None:
        out = out.fillna(default)
    return out


def _shrink_to_market(model_prob: pd.Series, market_prob: pd.Series, model_weight: float | pd.Series = 0.35) -> pd.Series:
    """Shrink model probabilities toward market-implied probability to reduce overconfidence."""
    model = pd.to_numeric(model_prob, errors="coerce")
    market = pd.to_numeric(market_prob, errors="coerce")
    if isinstance(model_weight, pd.Series):
        mw = pd.to_numeric(model_weight, errors="coerce").reindex(model.index).fillna(0.35).clip(0.05, 0.95)
    else:
        mw = pd.Series(float(model_weight), index=model.index, dtype="float64").clip(0.05, 0.95)
    mk = 1.0 - mw
    shrunk = model * mw + market * mk
    return shrunk.clip(0.02, 0.98)


def _game_dates(df: pd.DataFrame) -> pd.Series:
    """Parse known date columns as UTC-aware datetimes, preserving ISO-8601 timestamps."""
    if df is None or df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    out = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")
    for col in DATE_ALIASES:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            out = out.where(out.notna(), parsed)
    return out




def _format_game_time_est(df: pd.DataFrame) -> pd.Series:
    """Return game times formatted in ET. If only a fallback date exists, return the date."""
    if df is None or df.empty:
        return pd.Series(dtype="string")

    out = pd.Series([""] * len(df), index=df.index, dtype="string")
    raw_time = df.get("game_time_est", pd.Series([""] * len(df))).astype(str).str.strip().replace("nan", "")
    dt_game = pd.to_datetime(df.get("game_date"), errors="coerce", utc=True)

    for idx in df.index:
        t_str = raw_time[idx]
        d_obj = dt_game[idx]

        # 1. If we have a raw time string from the API or merge, use it.
        if t_str:
            out[idx] = t_str
            continue

        # 2. If no time string, but we have a valid UTC date object
        if pd.notna(d_obj):
            # Check if it's a midnight fallback placeholder
            # If the user passed in exactly YYYY-MM-DDT00:00:00Z, we map it to date-only.
            if d_obj.hour == 0 and d_obj.minute == 0 and d_obj.second == 0:
                out[idx] = d_obj.strftime("%Y-%m-%d")
            else:
                # It has a real clock time, convert to ET
                est = d_obj.tz_convert("America/New_York")
                out[idx] = est.strftime("%Y-%m-%d %I:%M %p ET").replace(" 0", " ")

    return out



def _date_join_key(series: pd.Series) -> pd.Series:
    """Return normalized date key robust to timezone formatting mismatches."""

    def _normalize_local(value: Any) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT
        try:
            ts = pd.Timestamp(value)
        except Exception:
            return pd.NaT
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.normalize()

    dt_local = pd.to_datetime(series.apply(_normalize_local), errors="coerce")
    dt_utc = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    return dt_local.where(dt_local.notna(), dt_utc)


def _utc_day_key(series: pd.Series) -> pd.Series:
    """Return ET-floored game-night day key stored as UTC midnight.

    - ISO-8601 UTC timestamps (e.g. 2026-03-18T03:10:00Z) are converted to ET, then floored.
    - Date-only strings (e.g. 2026-03-17) are treated as local slate dates and kept on that day.
    """

    def _to_utc_day(value: Any) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT

        # Preserve date-only inputs as-is to avoid shifting them to the prior ET day.
        if isinstance(value, str) and len(value.strip()) == 10:
            try:
                d = pd.Timestamp(value.strip())
                return pd.Timestamp(year=d.year, month=d.month, day=d.day, tz="UTC")
            except Exception:
                return pd.NaT

        try:
            ts = pd.Timestamp(value)
        except Exception:
            return pd.NaT

        if ts.tzinfo is not None and ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.nanosecond == 0:
            # Date-only values parsed with utc=True should remain on their nominal slate day.
            ts_utc = ts.tz_convert("UTC")
            return pd.Timestamp(year=ts_utc.year, month=ts_utc.month, day=ts_utc.day, tz="UTC")

        if ts.tzinfo is None:
            # Naive timestamps are assumed local ET schedule times.
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("America/New_York")

        floored = ts.floor("D")
        return pd.Timestamp(year=floored.year, month=floored.month, day=floored.day, tz="UTC")

    return pd.to_datetime(series.apply(_to_utc_day), errors="coerce", utc=True)


def _et_day_string(series: pd.Series) -> pd.Series:
    """Return a canonical ET day key (YYYY-MM-DD) for cross-source joins."""
    day_utc = _utc_day_key(series)
    return day_utc.dt.strftime("%Y-%m-%d").astype("string")


def _force_utc_datetime(series: pd.Series) -> pd.Series:
    """Force a datetime series to timezone-aware UTC dtype."""
    return pd.to_datetime(series, errors="coerce", utc=True)
def _game_date_fallback() -> pd.Timestamp:
    """Return today's US/Eastern date, stored as UTC midnight to match parsed date-only strings."""
    from datetime import datetime
    import pytz

    est = pytz.timezone('America/New_York')
    now_est = datetime.now(est)
    return pd.Timestamp(year=now_est.year, month=now_est.month, day=now_est.day, tz="UTC")


def _normalize_upload(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = _normalize_upload_columns(df)
    for src, dst in _UPLOAD_COLUMN_ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    out = _coerce_identity_columns(out)
    out["game_date"] = _utc_day_key(_game_dates(out))
    # Fill any missing dates with fallback
    out["game_date"] = out["game_date"].fillna(_game_date_fallback())

    # Apply _matchup_id logic immediately for deterministic joins
    out["matchup_id"] = _matchup_id(out)

    # Tag rows loaded from a raw export file for priority upstream.
    if "odds_source" not in out.columns:
        out["odds_source"] = "odds_api"

    return out


def _is_pipeline_export(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    return _EXPORT_SIGNAL_COLS.issubset(cols)


def _coerce_export_to_canonical(df: pd.DataFrame, selected_sports: list[str] | None) -> pd.DataFrame:
    out = _normalize_upload_columns(df)
    original_columns = set(out.columns)
    for src, dst in _UPLOAD_COLUMN_ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    out = _coerce_identity_columns(out)
    required_identity = {"league", "home_team", "away_team", "market_type"}
    missing_cols = [c for c in sorted(required_identity) if c not in out.columns or _string_series(out, c).str.len().eq(0).all()]
    if missing_cols:
        logger.warning(
            "Upload normalization missing expected canonical columns: %s (from source columns: %s)",
            ", ".join(missing_cols),
            sorted(original_columns),
        )
    out["game_date"] = _game_dates(out)
    # Fill any missing dates with fallback
    out["game_date"] = out["game_date"].fillna(_game_date_fallback())
    out["market_type"] = _string_series(out, "market_type")
    out["spread_line"] = pd.to_numeric(out.get("spread_line"), errors="coerce")
    out["total_line"] = pd.to_numeric(out.get("total_line"), errors="coerce")
    out["theover_probability"] = pd.to_numeric(out.get("theover_probability"), errors="coerce")
    out["odds_american"] = pd.to_numeric(out.get("odds_american"), errors="coerce")
    out["market_probability"] = out["odds_american"].apply(american_to_prob)
    out["ml_probability"] = pd.to_numeric(out.get("ml_probability"), errors="coerce")
    out["display_probability"] = pd.to_numeric(out.get("display_probability"), errors="coerce")
    out["calibrated_probability"] = pd.to_numeric(out.get("calibrated_probability"), errors="coerce")
    out["expected_value"] = pd.to_numeric(out.get("expected_value"), errors="coerce")
    out["edge"] = pd.to_numeric(out.get("edge"), errors="coerce")
    out["game_key"] = _mk_game_key(out)
    out["best_pick"] = out.apply(_format_best_pick, axis=1)
    out = _apply_analysis_calculations(out)
    # DO NOT filter rows based on league here to prevent losing master slate rows
    # if selected_sports:
    #     selected = {str(s).upper() for s in selected_sports}
    #     out = out[_string_series(out, "league").isin(selected)].copy()
    for col in CANONICAL_BET_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[CANONICAL_BET_COLUMNS]


def _concat_valid_bet_frames(frames: list[pd.DataFrame], expected_columns: list[str]) -> pd.DataFrame:
    valid_frames = [
        frame.copy() for frame in frames
        if frame is not None and isinstance(frame, pd.DataFrame)
        and not frame.empty and not frame.dropna(how="all").empty
    ]
    if not valid_frames:
        return pd.DataFrame(columns=expected_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.concat(valid_frames, ignore_index=True)




def _canonical_matchup_key(df: pd.DataFrame) -> pd.Series:
    """Orientation-insensitive game key (league + sorted teams + date)."""
    league = _string_series(df, "league").str.upper()
    home = clean_team_name(_string_series(df, "home_team").map(normalize_team_name)).str.upper()
    away = clean_team_name(_string_series(df, "away_team").map(normalize_team_name)).str.upper()
    team_a = np.where(home <= away, home, away)
    team_b = np.where(home <= away, away, home)
    team_a = pd.Series(team_a, index=df.index, dtype="string")
    team_b = pd.Series(team_b, index=df.index, dtype="string")
    date = _utc_day_key(_game_dates(df)).dt.strftime("%Y-%m-%d").fillna("")
    return league + "|" + team_a + "|" + team_b + "|" + date


def _canonical_matchup_teams_key(df: pd.DataFrame) -> pd.Series:
    """Orientation-insensitive game key using league + sorted teams only (no date)."""
    league = _string_series(df, "league").str.upper()
    home = clean_team_name(_string_series(df, "home_team").map(normalize_team_name)).str.upper()
    away = clean_team_name(_string_series(df, "away_team").map(normalize_team_name)).str.upper()
    team_a = np.where(home <= away, home, away)
    team_b = np.where(home <= away, away, home)
    team_a = pd.Series(team_a, index=df.index, dtype="string")
    team_b = pd.Series(team_b, index=df.index, dtype="string")
    return league + "|" + team_a + "|" + team_b


def _matchup_id(df: pd.DataFrame) -> pd.Series:
    """Canonical matchup id using sorted normalized team names + ET day (direction-independent)."""
    home = clean_team_name(_string_series(df, "home_team").map(normalize_team_name)).str.upper()
    away = clean_team_name(_string_series(df, "away_team").map(normalize_team_name)).str.upper()
    # Use strict lexicographical sorting to prevent any inverted matchup mismatches
    team_a = np.where(home <= away, home, away)
    team_b = np.where(home <= away, away, home)
    team_a = pd.Series(team_a, index=df.index, dtype="string")
    team_b = pd.Series(team_b, index=df.index, dtype="string")
    date_key = _et_day_string(_game_dates(df)).fillna("")
    return team_a + "|" + team_b + "|" + date_key


def _mk_game_key(df: pd.DataFrame) -> pd.Series:
    return (
        _string_series(df, "league").str.upper()
        + "|"
        + _string_series(df, "home_team").str.upper()
        + "|"
        + _string_series(df, "away_team").str.upper()
    )


def _team_similarity_score(left: str, right: str) -> int:
    l = str(left or "").strip().lower()
    r = str(right or "").strip().lower()
    if not l or not r:
        return 0
    if thefuzz_fuzz is not None:
        return int(thefuzz_fuzz.token_sort_ratio(l, r))
    if 'rapidfuzz_fuzz' in globals() and rapidfuzz_fuzz is not None:
        return int(rapidfuzz_fuzz.token_sort_ratio(l, r))
    return int(round(100 * SequenceMatcher(None, l, r).ratio()))


def _fuzzy_match_schedule_row(row: pd.Series, schedule_df: pd.DataFrame, threshold: int = 85) -> pd.Series:
    league_val = row.get("league")
    home_val = row.get("home_team")
    away_val = row.get("away_team")
    league = str(league_val).upper() if pd.notna(league_val) else ""
    home = str(home_val) if pd.notna(home_val) else ""
    away = str(away_val) if pd.notna(away_val) else ""
    if not league or not home or not away or schedule_df.empty:
        return pd.Series(dtype="object")

    league_pool = schedule_df[schedule_df["league"].eq(league)]
    if league_pool.empty:
        return pd.Series(dtype="object")

    best_idx = None
    best_score = -1
    best_orient = ""
    for idx, cand in league_pool.iterrows():
        direct_home = _team_similarity_score(home, cand.get("home_team", ""))
        direct_away = _team_similarity_score(away, cand.get("away_team", ""))
        direct_min = min(direct_home, direct_away)

        rev_home = _team_similarity_score(home, cand.get("away_team", ""))
        rev_away = _team_similarity_score(away, cand.get("home_team", ""))
        rev_min = min(rev_home, rev_away)

        cand_score = max(direct_min, rev_min)
        if cand_score > best_score:
            best_idx = idx
            best_score = cand_score
            best_orient = "direct" if direct_min >= rev_min else "reverse"

    if best_idx is None or best_score < threshold:
        return pd.Series(dtype="object")

    matched = league_pool.loc[best_idx].copy()
    matched["_fuzzy_score"] = int(best_score)
    matched["_fuzzy_orientation"] = best_orient
    return matched


@functools.lru_cache(maxsize=1)
def load_base_data() -> pd.DataFrame:
    try:
        base_df = pd.read_csv("data/master_all_sports.csv")
    except Exception:
        return pd.DataFrame()
    base_df = ensure_base_schema(base_df)
    base_df.columns = [str(c).strip().lower() for c in base_df.columns]
    base_df["league"] = _string_series(base_df, "league").str.upper().replace(LEAGUE_ALIASES)
    base_df["home_team"] = _string_series(base_df, "home_team").map(normalize_team_name)
    base_df["away_team"] = _string_series(base_df, "away_team").map(normalize_team_name)
    base_df["game_date"] = _game_dates(base_df)
    base_df["game_key"] = _mk_game_key(base_df)
    return base_df


def _drop_stale_stored_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        return df, 0
    out = df.copy()
    game_dates = _game_dates(out)
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tz is None else pd.Timestamp.utcnow()
    stale_mask = game_dates.notna() & game_dates.lt(now_utc)
    stale_count = int(stale_mask.sum())
    if stale_count:
        out = out.loc[~stale_mask].copy()
    return out, stale_count


def _first_existing_numeric(df: pd.DataFrame, candidates: list[str], default: float | int | None = None) -> pd.Series:
    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    for col in candidates:
        if col in df.columns:
            out = out.where(out.notna(), pd.to_numeric(df[col], errors="coerce"))
    if default is not None:
        out = out.fillna(default)
    return out


def american_to_decimal(odds: Any) -> float:
    v = pd.to_numeric(odds, errors="coerce")
    if pd.isna(v):
        # Preserve long-standing -110 default when odds are unavailable.
        return 1.9091
    v = float(v)
    if v == 0.0:
        # Explicit convention for invalid zero-odds placeholder values.
        return 2.0
    if v > 0:
        return 1 + (v / 100.0)
    return 1 + (100.0 / abs(v))


def _format_best_pick(row: pd.Series) -> str:
    def _safe_text(val: Any) -> str:
        return "" if pd.isna(val) else str(val)

    market = _safe_text(row.get("market_type")).strip().lower()
    home_team = _safe_text(row.get("home_team"))
    away_team = _safe_text(row.get("away_team"))

    if market == "spread_home":
        line = pd.to_numeric(row.get("market_line_used"), errors="coerce")
        if pd.isna(line):
            line = pd.to_numeric(row.get("spread_line"), errors="coerce")
        if pd.isna(line):
            line = pd.to_numeric(row.get("spread"), errors="coerce")
        return f"{home_team} {line:+.1f}" if pd.notna(line) else f"{home_team} (No Line)"
    if market == "spread_away":
        line = pd.to_numeric(row.get("market_line_used"), errors="coerce")
        if pd.isna(line):
            line = pd.to_numeric(row.get("spread_line"), errors="coerce")
        if pd.isna(line):
            line = pd.to_numeric(row.get("spread"), errors="coerce")
        return f"{away_team} {line:+.1f}" if pd.notna(line) else f"{away_team} (No Line)"
    if market == "total_over":
        line = pd.to_numeric(row.get("market_line_used"), errors="coerce")
        # try fallback to 'total' if 'total_line' is missing
        if pd.isna(line):
            line = pd.to_numeric(row.get("total_line"), errors="coerce")
        if pd.isna(line):
            line = pd.to_numeric(row.get("total"), errors="coerce")
        return f"Over {line:.1f}" if pd.notna(line) else "Over (No Line)"
    if market == "total_under":
        line = pd.to_numeric(row.get("market_line_used"), errors="coerce")
        if pd.isna(line):
            line = pd.to_numeric(row.get("total_line"), errors="coerce")
        if pd.isna(line):
            line = pd.to_numeric(row.get("total"), errors="coerce")
        return f"Under {line:.1f}" if pd.notna(line) else "Under (No Line)"
    if market == "h2h_home":
        return home_team
    if market == "h2h_away":
        return away_team
    return ""


def compute_blended_probability(
    p_market: pd.Series,
    p_kalshi: pd.Series,
    p_ml: pd.Series,
    p_theover: pd.Series,
    p_sentiment: pd.Series,
    league: pd.Series | None = None,
    market_type: pd.Series | None = None
) -> pd.Series:
    """
    Vectorized blend using two-tier logic defined in weights_config.
    """
    market = pd.to_numeric(p_market, errors="coerce")
    kalshi = pd.to_numeric(p_kalshi, errors="coerce")
    ml = pd.to_numeric(p_ml, errors="coerce")
    # Do NOT fill missing TheOver with 0.5 — a NaN signal must be dropped and its
    # weight redistributed (handled in _blend_row), not injected as a neutral vote
    # that drags every blended estimate toward the midpoint.
    theover = pd.to_numeric(p_theover, errors="coerce")
    sentiment = pd.to_numeric(p_sentiment, errors="coerce").fillna(0.5)
    m_type = pd.Series(market_type).fillna("").astype(str).str.lower()

    def _blend_row(p_mkt, p_kal, p_ml, p_the, p_sen, m_typ, lg):
        if pd.isna(p_mkt):
            p_mkt = p_ml if pd.notna(p_ml) else 0.5

        # Kalshi Probability is already oriented to the pick side before this step
        k_oriented = None
        if pd.notna(p_kal):
            k_oriented = p_kal

        # TheOver outputs exactly 0.5 (WinProbSource=default_0.5) when it has no real
        # prediction. A no-information vote should not consume 30% blend weight — treat
        # it as absent so weight redistributes to signals that have actual information.
        if pd.notna(p_the) and abs(float(p_the) - 0.5) < 1e-9:
            p_the = float('nan')

        # Sentiment is only real when it deviates from neutral (0.5 in probability space).
        # p_sen is already converted: 0.5 + sentiment_diff * 0.5, so 0.5 = no signal.
        has_real_sentiment = pd.notna(p_sen) and abs(p_sen - 0.5) > 0.02

        # Tier 1 vs Tier 2
        if k_oriented is not None and k_oriented >= 0.55:
            # Tier 1
            w_kalshi = KALSHI_WEIGHT
            w_market = MARKET_WEIGHT
            w_ml = ML_MODEL_WEIGHT
            w_the = THEOVER_WEIGHT
            w_sen = SENTIMENT_WEIGHT if has_real_sentiment else 0.0

            # Market Maturity Overrides — league-specific Kalshi reliability
            if lg and lg.upper() == "NHL":
                w_kalshi = NHL_KALSHI_WEIGHT
                w_market = NHL_MARKET_WEIGHT
                w_ml = NHL_ML_MODEL_WEIGHT
                w_the = NHL_THEOVER_WEIGHT
                w_sen = NHL_SENTIMENT_WEIGHT if has_real_sentiment else 0.0
            elif lg and lg.upper() == "MLB":
                w_kalshi = LOW_LIQUIDITY_KALSHI_WEIGHT
                w_ml = LOW_LIQUIDITY_ML_MODEL_WEIGHT
                if "total" in m_typ:
                    # Balanced weighting: TheOver incorporates pitcher data (confirmed May-16);
                    # flat probs on May-13 were caused by team name bugs, not bad data.
                    # Kalshi still leads; TheOver and market share signal equally.
                    w_kalshi = MLB_TOTAL_KALSHI_WEIGHT
                    w_market = MLB_TOTAL_MARKET_WEIGHT
                    w_the = MLB_TOTAL_THEOVER_WEIGHT
                    w_ml = MLB_TOTAL_ML_WEIGHT
            elif lg and lg.upper() == "NBA" and "total" in m_typ:
                # NBA totals: TheOver has pace/defensive-rating context ML lacks.
                w_the = NBA_TOTAL_THEOVER_WEIGHT
                w_ml = NBA_TOTAL_ML_WEIGHT

            # Assemble only the signals that are actually present. A NaN signal
            # (e.g. missing TheOver or ML) has its weight redistributed
            # proportionally across the remaining signals, rather than being
            # injected as a neutral 0.5 vote that silently drags every blended
            # estimate toward the midpoint and disables the conflict penalty.
            p_sen_val = p_sen if has_real_sentiment else 0.0
            signals = [
                (k_oriented, w_kalshi),
                (p_mkt, w_market),
                (p_ml, w_ml),
                (p_the, w_the),
                (p_sen_val, w_sen),
            ]
            present = [(p, w) for p, w in signals if pd.notna(p) and w > 0]
            total_w = sum(w for _, w in present)
            prob = sum(p * (w / total_w) for p, w in present) if total_w > 0 else 0.5
        else:
            # Tier 2 Fallback (Kalshi disagrees or unavailable)
            w_market = FALLBACK_MARKET_WEIGHT
            w_ml = FALLBACK_ML_WEIGHT
            w_the = FALLBACK_THEOVER_WEIGHT
            w_sen = FALLBACK_SENTIMENT_WEIGHT if has_real_sentiment else 0.0
            # For totals, TheOver's contextual signal (pitcher/pace) becomes the
            # dominant source when Kalshi is absent — boost it, cut ML accordingly.
            if lg and "total" in m_typ:
                if lg.upper() == "MLB":
                    w_market = MLB_TOTAL_FALLBACK_MARKET_WEIGHT
                    w_the = MLB_TOTAL_FALLBACK_THEOVER_WEIGHT
                    w_ml = MLB_TOTAL_FALLBACK_ML_WEIGHT
                elif lg.upper() == "NBA":
                    w_the = NBA_TOTAL_FALLBACK_THEOVER_WEIGHT
                    w_ml = NBA_TOTAL_FALLBACK_ML_WEIGHT

            # Same present-signal redistribution as Tier 1 (no Kalshi term in the
            # fallback). Missing signals drop out and their weight is renormalized
            # across what remains.
            p_sen_val = p_sen if has_real_sentiment else 0.0
            signals = [
                (p_mkt, w_market),
                (p_ml, w_ml),
                (p_the, w_the),
                (p_sen_val, w_sen),
            ]
            present = [(p, w) for p, w in signals if pd.notna(p) and w > 0]
            total_w = sum(w for _, w in present)
            prob = sum(p * (w / total_w) for p, w in present) if total_w > 0 else 0.5

        return prob

    lg_series = pd.Series(league).fillna("").astype(str)
    blended = pd.Series([_blend_row(m, k, l, t, s, typ, lg)
                         for m, k, l, t, s, typ, lg in zip(market, kalshi, ml, theover, sentiment, m_type, lg_series)],
                        index=market.index)

    return pd.to_numeric(blended, errors="coerce").clip(0.01, 0.99)


def get_opposing_odds_from_exchange(odds):
    if pd.isna(odds):
        return pd.NA
    odds_val = float(odds)
    if odds_val <= 0:
        return abs(odds_val) - 20.0
    else:
        return -(odds_val + 20.0)

def apply_mlb_total_market_debias(calibrated, df) -> tuple[pd.Series, float]:
    """Market-anchored over-bias correction for MLB totals (13 Jun all-Over card).

    Kalshi+ML sit systematically above the de-vig market on P(over); that gap is
    bias (graded overs ~52%, no edge over the sharp market), so the slate-MEAN gap
    is removed from MLB total P(over) so direction selection rebalances.

    The correction is sign-preserving and FLOORED at the de-vig market, per game:
    an over row that leans above the market has its P(over) pulled DOWN toward the
    market by ``bias`` but never below it; an under row whose P(under) sits below the
    market is pulled UP toward the market by ``bias`` but never above it. A flat
    symmetric ±bias shift (the original form) overshot — it pushed games past the
    sharp market into a model-manufactured UNDER lean, and the EV term in direction
    selection amplified that into a near-unanimous all-Under card (14 Jun: a ~7-under
    / 6-over de-vig market produced a 14/14 Under card). Flooring at the market keeps
    a genuine market-over game on Over and leaves real under reads (already past the
    market) untouched, so the card follows the market's own per-game lean.

    Returns ``(corrected_calibrated, bias)``; bias is 0.0 when not applied. Pure +
    flag-checked + guarded so a single, tested implementation serves BOTH blend
    paths (the Analysis tab and the production best-picks card) — wiring it into
    only one of them was the #1919 bug.
    """
    try:
        from app_core.weights_config import (
            MLB_TOTAL_MARKET_DEBIAS_ENABLED,
            MLB_TOTAL_MARKET_DEBIAS_MAX_SHIFT,
            MLB_TOTAL_MARKET_DEBIAS_EXEMPT_AGREES_OVER,
            MLB_TOTAL_MARKET_DEBIAS_AGREES_KALSHI_MIN,
        )
        if not MLB_TOTAL_MARKET_DEBIAS_ENABLED:
            return calibrated, 0.0
        from core.slate_quality import market_anchored_over_bias

        lg = _string_series(df, "league").str.upper()
        mt = _string_series(df, "market_type").str.lower()
        is_over = lg.eq("MLB") & mt.eq("total_over")
        is_under = lg.eq("MLB") & mt.eq("total_under")
        if not bool(is_over.any()):
            return calibrated, 0.0
        bias = market_anchored_over_bias(
            calibrated[is_over],
            df["market_probability"][is_over],
            max_shift=float(MLB_TOTAL_MARKET_DEBIAS_MAX_SHIFT),
        )
        if abs(bias) <= 1e-9:
            return calibrated, 0.0
        cal = pd.to_numeric(calibrated, errors="coerce").to_numpy(dtype=float)
        mk = pd.to_numeric(df["market_probability"], errors="coerce").to_numpy(dtype=float)
        has_mkt = ~np.isnan(mk)
        # Over rows: shrink a positive over-edge (cal above the market) toward the
        # market by bias, FLOORED at the market — never cross into a manufactured
        # under lean. Rows at/below the market (a genuine under read), or with no
        # market anchor, are left untouched.
        over_corr = np.where(
            has_mkt & (cal > mk),
            np.maximum(cal - bias, np.where(has_mkt, mk, cal)),
            cal,
        )
        # Under rows: the over-bias deflates P(under); raise a below-market under-edge
        # toward the market by bias, CAPPED at the market. Genuine strong-under reads
        # (already above the market) are untouched.
        under_corr = np.where(
            has_mkt & (cal < mk),
            np.minimum(cal + bias, np.where(has_mkt, mk, cal)),
            cal,
        )
        # Exempt Kalshi-backed overs from the over correction: keep their blended
        # value so an independently-corroborated over-consensus is not stripped.
        if MLB_TOTAL_MARKET_DEBIAS_EXEMPT_AGREES_OVER and "kalshi_probability" in df.columns:
            _kalshi_arr = pd.to_numeric(df["kalshi_probability"], errors="coerce").to_numpy(dtype=float)
            _over_agrees_exempt = is_over.to_numpy(dtype=bool) & (
                _kalshi_arr >= float(MLB_TOTAL_MARKET_DEBIAS_AGREES_KALSHI_MIN)
            )
            over_corr = np.where(_over_agrees_exempt, cal, over_corr)
        new_vals = cal.copy()
        new_vals = np.where(is_over.to_numpy(dtype=bool), over_corr, new_vals)
        new_vals = np.where(is_under.to_numpy(dtype=bool), under_corr, new_vals)
        corrected = pd.Series(new_vals, index=calibrated.index).clip(0.05, 0.95)
        return corrected, float(bias)
    except Exception as e:  # never let the correction break the pipeline
        logger.warning(f"MLB total market de-bias skipped: {e}")
        return calibrated, 0.0


def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["odds_american"] = _numeric_series(out, "odds_american", pd.NA)

    # Phase 4: Implementation of Bayesian Shrinkage and Vig Removal
    # De-vig by applying multiplicative normalization for a standard 2-way market.
    # Since Novig is an exchange without standard sportsbook straddles, use its true implied prob.
    # We still perform a simple multiplicative normalization in case of minor bid-ask spread deviations
    implied_prob = out["odds_american"].apply(american_to_prob)

    opposing_implied = out["odds_american"].apply(get_opposing_odds_from_exchange).apply(american_to_prob)
    # Guard the de-vig denominator against 0/NaN (missing odds) and bound the result, as
    # the parallel computation downstream does — a bare division here can emit NaN/inf that
    # silently propagates into EV/edge/Kelly.
    _market_denom = implied_prob + opposing_implied
    out["market_probability"] = (
        implied_prob.divide(_market_denom.where(_market_denom > 0))
    ).clip(0.01, 0.99)

    theover = _numeric_series(out, "theover_probability")
    theover = theover.where(theover <= 1, theover / 100.0)
    ml = _numeric_series(out, "ml_probability")

    # TheOver-feed degradation guard (13 Jun: 5 games shared an identical P(Over)
    # of 0.692 and a column-shift put a text label in the numeric field, biasing
    # every total to the Over). Judge the per-game reads on the total_over rows;
    # if degenerately clustered, DOWN-WEIGHT (do not null) TheOver for the slate's
    # TOTALS by shrinking each read toward neutral 0.50 by THEOVER_DEGRADED_FADE_SHRINK.
    # Nulling outright was too blunt: it discarded legitimate slates where TheOver's
    # model simply rated many games at a common confidence (15 Jun: six of eight reads
    # at hit-rate 0.75 — magnitude 0.25 — folded over/under picks into one cluster and
    # tripped the guard, dropping TheOver entirely and flipping 5 picks, incl. a 95%
    # Over read on MIA@PHI graded to a No Play). Shrinking keeps a damped, game-specific
    # signal in the blend on a false positive while still discounting a truly constant
    # feed. Spreads/H2H are left untouched.
    try:
        from core.slate_quality import theover_feed_degraded
        from app_core.weights_config import THEOVER_DEGRADED_FADE_SHRINK
        _mt_lower = _string_series(out, "market_type").str.lower()
        _over_reads = theover[_mt_lower.eq("total_over")]
        _degraded, _reason = theover_feed_degraded(_over_reads)
        if _degraded:
            _totals_mask = _mt_lower.str.contains("total", na=False)
            _shrunk = 0.5 + (1.0 - float(THEOVER_DEGRADED_FADE_SHRINK)) * (theover - 0.5)
            theover = theover.where(~_totals_mask, _shrunk)
            out["theover_feed_degraded_reason"] = _reason
            existing_warn = _string_series(out, "run_health_warning")
            out["run_health_warning"] = existing_warn.where(
                existing_warn.str.len() > 0, _reason
            )
    except Exception as e:  # never let a guard break the pipeline
        logger.warning(f"TheOver degradation guard skipped: {e}")


    # theover is a legacy column mapping we still ingest
    model_prob = ml.where(ml.notna(), theover)
    out["display_probability"] = model_prob.round(3)
    kalshi_prob = _numeric_series(out, "kalshi_probability") if "kalshi_probability" in out.columns else pd.Series([pd.NA]*len(out), index=out.index)

    calibrated = compute_blended_probability(
        p_market=out["market_probability"],
        p_kalshi=kalshi_prob,
        p_ml=model_prob,
        p_theover=theover,  # Use existing variable
        p_sentiment=_numeric_series(out, "sentiment_diff", default=0.0).apply(lambda x: 0.5 + x * 0.5),
        league=_string_series(out, "league"),
        market_type=_string_series(out, "market_type")
    )

    # Cap calibrated probability for MLB total_over picks — TheOver inflates these
    # because its data is not game-specific (May-13 finding: near-identical ~0.85
    # probability regardless of starting pitcher matchup). Market is now primary signal.
    mlb_over_mask = (
        _string_series(out, "league").str.upper() == "MLB"
    ) & (
        _string_series(out, "market_type").str.lower() == "total_over"
    )
    calibrated = calibrated.where(~mlb_over_mask, calibrated.clip(upper=MLB_OVER_CALIBRATED_PROB_CAP))

    # Market-anchored over-bias correction for MLB totals (Analysis-tab path).
    calibrated, _debias = apply_mlb_total_market_debias(calibrated, out)
    if abs(_debias) > 1e-9:
        out["mlb_total_market_debias"] = _debias

    out["theover_probability"] = theover
    out["ml_probability"] = ml
    out["calibrated_probability"] = calibrated
    out["decimal_odds"] = out["odds_american"].apply(american_to_decimal)

    ev = calibrated * (out["decimal_odds"] - 1) - (1 - calibrated)
    edge = calibrated - out["market_probability"]

    # Null out EV and edge for missing odds
    missing_odds_mask = out["odds_american"].isna()
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
    nhl_totals_mask = (out["league"].str.upper() == "NHL") & (out["market_type"].str.contains("total|spread", case=False, na=False))
    ev = ev.where(~nhl_totals_mask, ev * 0.80)

    out["expected_value"] = ev
    out["edge"] = edge
    out["best_pick"] = out.apply(_format_best_pick, axis=1)
    return out


def _build_spread_rows(normalized: pd.DataFrame) -> list[pd.DataFrame]:
    """Build spread_home and spread_away rows from TheOver export.
    TheOver's 'Line' is the PickTeam's spread line (already signed correctly for them).
    The opposing team gets the negated line.
    """
    line = _first_existing_numeric(normalized, ["line", "spread_line", "spread", "points"])
    prob = _first_existing_numeric(normalized, ["theover_probability", "winprobability", "win_probability", "probability"])

    odds = _first_existing_numeric(normalized, ["odds_american", "american_odds", "odds"], default=pd.NA)

    base_cols = [c for c in ["league", "home_team", "away_team", "game_date", "game_time_est"] if c in normalized.columns]
    base = normalized[base_cols].copy()

    # Determine which team is the pick team
    pick_team = _string_series(normalized, "pick_team")
    home_team = _string_series(normalized, "home_team")
    away_team = _string_series(normalized, "away_team")

    # pick_is_home: True when PickTeam matches HomeTeam
    pick_is_home = pick_team.str.strip().str.lower() == home_team.str.strip().str.lower()

    spread_home = base.copy()
    spread_home["market_type"] = "spread_home"
    spread_home["spread_line"] = line.where(pick_is_home, -line)
    spread_home["total_line"] = pd.NA
    spread_home["theover_probability"] = prob.where(pick_is_home, (1 - prob).where(prob.notna(), pd.NA))
    spread_home["odds_american"] = odds

    spread_away = base.copy()
    spread_away["market_type"] = "spread_away"
    spread_away["spread_line"] = -line.where(pick_is_home, -line)  # negated from home
    spread_away["total_line"] = pd.NA
    spread_away["theover_probability"] = (1 - prob).where(pick_is_home & prob.notna(), prob.where(prob.notna(), pd.NA))
    spread_away["odds_american"] = odds

    return [spread_home, spread_away]


def _build_h2h_rows(normalized: pd.DataFrame) -> list[pd.DataFrame]:
    """Build h2h_home and h2h_away rows from TheOver moneyline export."""
    prob = _first_existing_numeric(normalized, ["theover_probability", "winprobability", "win_probability", "probability"])
    odds = _first_existing_numeric(normalized, ["odds_american", "american_odds", "odds"], default=pd.NA)

    base_cols = [c for c in ["league", "home_team", "away_team", "game_date", "game_time_est"] if c in normalized.columns]
    base = normalized[base_cols].copy()

    pick_team = _string_series(normalized, "pick_team")
    home_team = _string_series(normalized, "home_team")
    pick_is_home = pick_team.str.strip().str.lower() == home_team.str.strip().str.lower()

    h2h_home = base.copy()
    h2h_home["market_type"] = "h2h_home"
    h2h_home["spread_line"] = pd.NA
    h2h_home["total_line"] = pd.NA
    h2h_home["theover_probability"] = prob.where(pick_is_home, (1 - prob).where(prob.notna(), pd.NA))
    h2h_home["odds_american"] = odds

    h2h_away = base.copy()
    h2h_away["market_type"] = "h2h_away"
    h2h_away["spread_line"] = pd.NA
    h2h_away["total_line"] = pd.NA
    h2h_away["theover_probability"] = (1 - prob).where(pick_is_home & prob.notna(), prob.where(prob.notna(), pd.NA))
    h2h_away["odds_american"] = odds

    return [h2h_home, h2h_away]


def _build_total_rows(normalized: pd.DataFrame) -> list[pd.DataFrame]:
    """Expand a raw totals upload into total_over + total_under rows."""
    total_line = _first_existing_numeric(normalized, ["total_line", "total", "line", "points"])
    total_prob = _first_existing_numeric(normalized, ["theover_probability", "winprobability", "win_probability", "probability"])

    total_odds = _first_existing_numeric(normalized, ["odds_american", "american_odds", "odds"], default=pd.NA)

    # TheOver always outputs P(Over wins). Assign directly to over_prob and invert for under_prob.
    # Do NOT re-invert based on pick direction — the M code's flip already handles Over/Under orientation.
    over_prob = total_prob
    under_prob = (1 - total_prob).where(total_prob.notna(), pd.NA)

    # WinProbSource describes how P(Over) was derived; it is the same for both
    # orientations of a game. Carried through so the selection stage can discount
    # low-confidence sources (e.g. model_hit_rate_flipped) for the direction decision.
    if "win_prob_source" in normalized.columns:
        win_prob_source = normalized["win_prob_source"].astype("string")
    else:
        win_prob_source = pd.Series(pd.NA, index=normalized.index, dtype="string")

    base_cols = [c for c in ["league", "home_team", "away_team", "game_date", "game_time_est"] if c in normalized.columns]
    base = normalized[base_cols].copy()

    total_over = base.copy()
    total_over["market_type"] = "total_over"
    total_over["spread_line"] = pd.NA
    total_over["total_line"] = total_line
    total_over["theover_probability"] = over_prob
    total_over["win_prob_source"] = win_prob_source
    total_over["odds_american"] = total_odds

    total_under = base.copy()
    total_under["market_type"] = "total_under"
    total_under["spread_line"] = pd.NA
    total_under["total_line"] = total_line
    total_under["theover_probability"] = under_prob
    total_under["win_prob_source"] = win_prob_source
    total_under["odds_american"] = total_odds

    return [total_over, total_under]


def build_theover_bet_rows(
    spreads_df: pd.DataFrame | None,
    totals_df: pd.DataFrame | None,
    selected_sports: list[str] | None,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    uploads: list[tuple[pd.DataFrame | None, str]] = [
        (spreads_df, "spreads"),
        (totals_df, "totals"),
    ]

    for upload_df, file_type in uploads:
        if upload_df is None or (isinstance(upload_df, pd.DataFrame) and upload_df.empty):
            continue

        if _is_pipeline_export(upload_df):
            logger.info("build_theover_bet_rows: detected pipeline export CSV (%s), passing through directly", file_type)
            pieces.append(_coerce_export_to_canonical(upload_df, selected_sports))
            continue

        normalized = _normalize_upload(upload_df)
        # Drop moneylines from TheOver sides CSV — only spreads and totals are surfaced
        if file_type == "spreads" and "market" in normalized.columns:
            normalized = normalized[~normalized["market"].str.lower().str.strip().eq("moneyline")].copy()
        if normalized.empty:
            continue

        if file_type == "spreads":
            pieces.extend(_build_spread_rows(normalized))
        elif file_type == "totals":
            pieces.extend(_build_total_rows(normalized))
        else:
            has_spread_data = "spread_line" in normalized.columns and normalized["spread_line"].notna().any()
            has_total_data = "total_line" in normalized.columns and normalized["total_line"].notna().any()
            if has_spread_data:
                pieces.extend(_build_spread_rows(normalized))
            if has_total_data:
                pieces.extend(_build_total_rows(normalized))
            if not has_spread_data and not has_total_data:
                pieces.extend(_build_spread_rows(normalized))
                pieces.extend(_build_total_rows(normalized))

    out = _concat_valid_bet_frames(pieces, expected_columns=CANONICAL_BET_COLUMNS)

    if out.empty:
        return pd.DataFrame(columns=CANONICAL_BET_COLUMNS)

    out = _preprocess_bet_rows_for_league_bridge(out)

    if selected_sports and len(selected_sports) == 1:
        inferred_league = str(selected_sports[0]).upper()
        league_series = _clean_text_placeholders(_string_series(out, "league"))
        out["league"] = league_series.where(league_series.str.len().gt(0), inferred_league)

    base_ref = load_base_data()
    out = _infer_missing_league_from_base(out, base_ref)
    out = _patch_missing_league_for_college_rows(out, selected_sports)
    out = _infer_missing_league_from_team_sets(out, selected_sports)
    out = _resolve_team_names_from_base(out, base_ref)
    out = _dedupe_inverted_matchups(out)

    out["spread"] = pd.to_numeric(out.get("spread_line"), errors="coerce")
    out["total"] = pd.to_numeric(out.get("total_line"), errors="coerce")

    if "game_key" not in out.columns:
        out["league"] = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)
        out["home_team"] = _string_series(out, "home_team").map(normalize_team_name)
        out["away_team"] = _string_series(out, "away_team").map(normalize_team_name)
        out["market_type"] = _string_series(out, "market_type")
        out["game_date"] = _utc_day_key(_game_dates(out))
        out["spread_line"] = pd.to_numeric(out.get("spread_line"), errors="coerce")
        out["total_line"] = pd.to_numeric(out.get("total_line"), errors="coerce")
        out["spread"] = pd.to_numeric(out.get("spread_line"), errors="coerce")
        out["total"] = pd.to_numeric(out.get("total_line"), errors="coerce")
        out["theover_probability"] = pd.to_numeric(out.get("theover_probability"), errors="coerce")
        out["odds_american"] = pd.to_numeric(out.get("odds_american"), errors="coerce")
        if "odds_source" not in out.columns:
            out["odds_source"] = pd.NA
        if selected_sports:
            selected = {str(s).upper() for s in selected_sports}
            selected = {LEAGUE_ALIASES.get(s, s) for s in selected}
            # Keep rows with missing league labels to avoid dropping valid NCAAB games before inference.
            league_series = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)
            # DO NOT filter rows based on league here to prevent losing master slate rows
            # out = out[league_series.isin(selected) | league_series.str.len().eq(0)].copy()
        out["game_key"] = _mk_game_key(out)
        out = _apply_analysis_calculations(out)

    for col in CANONICAL_BET_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[CANONICAL_BET_COLUMNS]






TEAM_NAME_OVERRIDES: dict[tuple[str, str], str] = {
    # NBA
    ("NBA", "CHICAGO"): "Chicago Bulls",
    ("NBA", "MEMPHIS"): "Memphis Grizzlies",
    ("NBA", "HOUSTON"): "Houston Rockets",
    ("NBA", "DALLAS"): "Dallas Mavericks",
    ("NBA", "NEW ORLEANS"): "New Orleans Pelicans",
    ("NBA", "PORTLAND"): "Portland Trail Blazers",
    ("NBA", "ATLANTA"): "Atlanta Hawks",
    ("NBA", "BOSTON"): "Boston Celtics",
    ("NBA", "PHOENIX"): "Phoenix Suns",
    ("NBA", "SAN ANTONIO"): "San Antonio Spurs",
    ("NBA", "ORLANDO"): "Orlando Magic",
    ("NBA", "BROOKLYN"): "Brooklyn Nets",
    ("NBA", "GOLDEN STATE"): "Golden State Warriors",
    ("NBA", "WASHINGTON"): "Washington Wizards",
    ("NBA", "DETROIT"): "Detroit Pistons",
    ("NBA", "LOS ANGELES"): "Los Angeles Lakers",
    # NHL
    ("NHL", "DALLAS"): "Dallas Stars",
    ("NHL", "UTAH"): "Utah Hockey Club",
    ("NHL", "DETROIT"): "Detroit Red Wings",
    ("NHL", "CALGARY"): "Calgary Flames",
    ("NHL", "COLORADO"): "Colorado Avalanche",
    ("NHL", "PITTSBURGH"): "Pittsburgh Penguins",
    ("NHL", "NY RANGERS"): "New York Rangers",
    ("NHL", "NEW YORK RANGERS"): "New York Rangers",
    ("NHL", "LOS ANGELES"): "Los Angeles Kings",
    ("NHL", "FLORIDA"): "Florida Panthers",
    ("NHL", "CAROLINA"): "Carolina Hurricanes",
    ("NHL", "TAMPA BAY"): "Tampa Bay Lightning",
    ("NHL", "NEW JERSEY"): "New Jersey Devils",
    ("NHL", "SAN JOSE"): "San Jose Sharks",
    ("NHL", "VEGAS"): "Vegas Golden Knights",
    ("NHL", "BUFFALO"): "Buffalo Sabres",
    ("NHL", "OTTAWA"): "Ottawa Senators",
    ("NHL", "MONTREAL"): "Montreal Canadiens",
    ("NHL", "NASHVILLE"): "Nashville Predators",
    ("NHL", "PHILADELPHIA"): "Philadelphia Flyers",
    ("NHL", "CHICAGO"): "Chicago Blackhawks",
    ("NHL", "COLUMBUS"): "Columbus Blue Jackets",
    ("NHL", "WINNIPEG"): "Winnipeg Jets",
    ("NHL", "MINNESOTA"): "Minnesota Wild",
}


def _apply_team_name_overrides(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    leagues = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)

    def _fix(team: str, league: str) -> str:
        key = (str(league or "").upper(), str(team or "").upper().strip())
        return TEAM_NAME_OVERRIDES.get(key, team)

    out["home_team"] = [_fix(t, lg) for t, lg in zip(_string_series(out, "home_team"), leagues)]
    out["away_team"] = [_fix(t, lg) for t, lg in zip(_string_series(out, "away_team"), leagues)]
    return out
def _resolve_team_names_from_base(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve city-only/alias team names to canonical full names using base schedule by league."""
    if df is None or df.empty or base_df is None or base_df.empty:
        return df

    out = df.copy()
    base = base_df.copy()
    base["league"] = _string_series(base, "league").str.upper().replace(LEAGUE_ALIASES)
    base["home_team"] = _string_series(base, "home_team").map(normalize_team_name)
    base["away_team"] = _string_series(base, "away_team").map(normalize_team_name)

    all_teams = pd.concat([
        base[["league", "home_team"]].rename(columns={"home_team": "team"}),
        base[["league", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).dropna().drop_duplicates()

    alias_map: dict[tuple[str, str], str] = {}
    league_team_pool: dict[str, list[str]] = {}
    for league, group in all_teams.groupby("league"):
        teams = sorted(set(group["team"].astype(str)))
        league_team_pool[str(league)] = teams
        candidate_to_teams: dict[str, set[str]] = {}
        for team in teams:
            parts = team.split()
            candidates = {team.upper()}
            if len(parts) > 1:
                candidates.add(" ".join(parts[:-1]).upper())
            for c in candidates:
                if c and c != "NAN":
                    candidate_to_teams.setdefault(c, set()).add(team)
        for candidate, matches in candidate_to_teams.items():
            if len(matches) == 1:
                alias_map[(str(league), candidate)] = next(iter(matches))

    def _resolve(league: str, team: str) -> str:
        league_norm = str(league or "").upper()
        team_norm = str(team or "").strip()
        key = (league_norm, team_norm.upper())
        direct = alias_map.get(key)
        if direct:
            return direct

        # Fuzzy fallback for typos/partial names when direct alias resolution misses.
        candidates = league_team_pool.get(league_norm, [])
        if not candidates or not team_norm:
            return team
        scored = sorted(
            [(_team_similarity_score(team_norm, cand), cand) for cand in candidates],
            key=lambda x: x[0],
            reverse=True,
        )
        if scored and scored[0][0] >= 90:
            if len(scored) == 1 or scored[0][0] - scored[1][0] >= 5:
                return scored[0][1]
        return team

    out["league"] = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)
    out["home_team"] = _string_series(out, "home_team").map(normalize_team_name)
    out["away_team"] = _string_series(out, "away_team").map(normalize_team_name)
    out["home_team"] = [
        _resolve(lg, tm) for lg, tm in zip(_string_series(out, "league"), _string_series(out, "home_team"))
    ]
    out["away_team"] = [
        _resolve(lg, tm) for lg, tm in zip(_string_series(out, "league"), _string_series(out, "away_team"))
    ]
    out = _apply_team_name_overrides(out)
    return out


def _dedupe_inverted_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated/inverted matchup rows that represent the same market leg."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["_matchup_id_dedupe"] = _matchup_id(out)
    out["_game_date_dedupe"] = _et_day_string(_game_dates(out))
    out["_spread_abs"] = _numeric_series(out, "spread_line").abs().fillna(-9999.0)
    out["_total_line"] = _numeric_series(out, "total_line").fillna(-9999.0)
    out["_market_dedupe"] = _string_series(out, "market_type").str.lower()
    before = len(out)
    out = out.drop_duplicates(
        subset=["_matchup_id_dedupe", "_game_date_dedupe", "_market_dedupe", "_spread_abs", "_total_line"],
        keep="first",
    ).copy()
    dropped = before - len(out)
    if dropped > 0:
        logger.warning("Deduplication removed %s inverted/duplicate matchup rows.", dropped)
    return out.drop(columns=["_matchup_id_dedupe", "_game_date_dedupe", "_spread_abs", "_total_line", "_market_dedupe"])
def _fill_missing_game_dates_from_base(bet_rows_df: pd.DataFrame, base_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = bet_rows_df.copy()

    # Check what is truly missing right before applying any fallback values
    # We look at the raw 'game_date' column before `_game_dates()` which applies fallbacks
    is_missing_before = out.get("game_date", pd.Series([pd.NA] * len(out))).isna()

    out["game_date"] = _game_dates(out)

    missing_before_join = out["game_date"].isna()

    if base_df is not None and not base_df.empty and missing_before_join.any():
        base = base_df.copy()
        base["league"] = _string_series(base, "league").str.upper().replace(LEAGUE_ALIASES)
        base["home_team"] = _string_series(base, "home_team").map(normalize_team_name)
        base["away_team"] = _string_series(base, "away_team").map(normalize_team_name)
        base["game_date"] = _game_dates(base)
        base = base[base["game_date"].notna()].copy()

        schedule = (
            base.sort_values("game_date")
            .drop_duplicates(["league", "home_team", "away_team"], keep="last")
            [[c for c in ["league", "home_team", "away_team", "game_date", "game_time_est"] if c in base.columns]]
        ).copy()

        schedule["matchup_key"] = _canonical_matchup_teams_key(schedule)
        out["matchup_key"] = _canonical_matchup_teams_key(out)

        direct = (
            schedule[["league", "matchup_key", "game_date"]]
            .rename(columns={"game_date": "game_date_base"})
            .drop_duplicates(["league", "matchup_key"])
        )
        out = out.merge(direct, on=["league", "matchup_key"], how="left")
        out["game_date"] = out["game_date"].where(out["game_date"].notna(), out["game_date_base"])
        out = out.drop(columns=["game_date_base", "matchup_key"])

    # Number of rows missing before the join based on the raw pre-fallback state
    missing_count = int(is_missing_before.sum())
    # The count of dates filled successfully after the join
    filled = int((is_missing_before & out["game_date"].notna()).sum())
    missing_after = int(out["game_date"].isna().sum())

    return out, {
        "date_fill_total_rows": missing_count,
        "date_fill_success_rows": filled,
        "date_fill_success_rate": float(filled / max(missing_count, 1)),
        "missing_game_date_rows": missing_after,
    }


def is_postseason_ncaab(df: pd.DataFrame) -> pd.Series:
    """Identify NCAAB games played on or after March 17, 2026."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    league_mask = _string_series(df, "league").str.upper() == "NCAAB"

    date_series = pd.to_datetime(df.get("game_date"), errors="coerce", utc=True)
    postseason_start = pd.Timestamp("2026-03-17", tz="UTC")

    date_mask = date_series >= postseason_start
    return league_mask & date_mask


def is_stale_schedule(base_df: pd.DataFrame, bet_rows_df: pd.DataFrame) -> bool:
    if base_df is None or base_df.empty or bet_rows_df is None or bet_rows_df.empty:
        return False
    base_dates = _game_dates(base_df)
    bet_dates = _game_dates(bet_rows_df)
    if base_dates.notna().sum() == 0 or bet_dates.notna().sum() == 0:
        return False
    return bool((bet_dates.max() - base_dates.max()) > pd.Timedelta(days=7))


def _apply_triple_filter_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Triple-Filter Ranking System to distinguish between high-conviction model predictions and conservative statistical floors."""
    if df is None or df.empty:
        return df

    final_df = df.copy()

    # 1. Edge Calculation (Model vs Market — actual betting edge)
    # Previously used ml_prob - theover_prob (inter-model disagreement), which ranked picks
    # by how much ML diverges from TheOver rather than how much the blended model beats the
    # market. Picks with large ML/TheOver disagreement are NOT systematically better; this
    # was promoting overconfident ML predictions to S/A-Tier while better-calibrated picks
    # with real market edges landed in lower tiers.
    # Default missing columns to a NaN Series (not None/scalar) so the chained
    # .fillna below never hits a numpy scalar — a minimal input lacking
    # market_probability (derived from odds in production) should fall back to a
    # 0.5 coin-flip, not raise. Mirrors the None-safe idiom used elsewhere here.
    _idx = final_df.index
    ml_prob = pd.to_numeric(final_df.get('ml_probability', pd.Series(np.nan, index=_idx)), errors='coerce')
    theover_prob = pd.to_numeric(final_df.get('theover_probability', pd.Series(np.nan, index=_idx)), errors='coerce')
    calibrated_prob = pd.to_numeric(final_df.get('calibrated_probability', pd.Series(np.nan, index=_idx)), errors='coerce').fillna(0.5)
    market_prob = pd.to_numeric(final_df.get('market_probability', pd.Series(np.nan, index=_idx)), errors='coerce').fillna(0.5)
    expected_value = pd.to_numeric(final_df.get('expected_value', pd.Series(0.0, index=_idx)), errors='coerce').fillna(0.0)

    triple_filter_edge = calibrated_prob - market_prob

    # 2. Uniqueness Check (Against the high-precision blacklist)
    BLACKLIST = [0.623034656047821, 0.10671072453260422, 0.48637846, 0.31053704, 0.35, 0.562239, 0.559358, 0.633159]

    if isinstance(ml_prob, pd.Series):
        final_df['is_unique'] = ml_prob.apply(
            lambda x: not any(abs(x - b) < 1e-5 for b in BLACKLIST) if pd.notna(x) else False
        )
    else:
        # ml_prob could be a scalar if dataframe has only 1 row or get returns a single value
        ml_val = float(ml_prob)
        final_df['is_unique'] = not any(abs(ml_val - b) < 1e-5 for b in BLACKLIST) if pd.notna(ml_val) else False

    # 3. Assign Pick_Quality Tiers (S, A, B, C)
    cal_prob_series = final_df.get('calibrated_probability')
    if cal_prob_series is not None:
        calibrated_prob = pd.to_numeric(cal_prob_series, errors='coerce').fillna(0.5)
    else:
        calibrated_prob = pd.Series([0.5] * len(final_df), index=final_df.index)

    tier_scores = []
    for idx in final_df.index:
        is_unique = final_df.at[idx, 'is_unique']

        # Accessing values safely in case of scalars
        edge = triple_filter_edge.at[idx] if isinstance(triple_filter_edge, pd.Series) else triple_filter_edge
        ev = expected_value.at[idx] if isinstance(expected_value, pd.Series) else expected_value
        ml = ml_prob.at[idx] if isinstance(ml_prob, pd.Series) else ml_prob
        theov = theover_prob.at[idx] if isinstance(theover_prob, pd.Series) else theover_prob
        cal_p = calibrated_prob.at[idx] if isinstance(calibrated_prob, pd.Series) else calibrated_prob

        edge = edge if pd.notna(edge) else 0.0
        ml = ml if pd.notna(ml) else 0.5
        theov = theov if pd.notna(theov) else 0.5
        cal_p = cal_p if pd.notna(cal_p) else 0.5

        if is_unique and edge > 0.15 and ev > 0:
            tier_scores.append(1)  # S-Tier
        elif is_unique and edge > 0.08 and ev > 0:
            tier_scores.append(2)  # A-Tier
        elif (is_unique and ml > 0.5 and theov > 0.5) or (not is_unique and ev > 0.10):
            tier_scores.append(3)  # B-Tier
        elif is_unique and ev > 0 and cal_p >= 0.45:
            tier_scores.append(4)  # C-Tier
        else:
            tier_scores.append(5)  # D-Tier

    final_df['tier_score'] = tier_scores

    # Jules: Use this dictionary for labeling and sorting
    TIER_LABELS = {
        1: "S-Tier (The Hammer)",
        2: "A-Tier (High Value)",
        3: "B-Tier (Consensus/Safe Floor)",
        4: "C-Tier (Value)",
        5: "D-Tier (Weak/Negative)"
    }

    # Mapping Logic for sorting
    final_df['Pick_Quality'] = final_df['tier_score'].map(TIER_LABELS)

    # 4. Final Rank Generation
    # Ensure expected_value is numeric and clean of missing values for sorting
    ev_sort = pd.to_numeric(final_df['expected_value'], errors='coerce').fillna(-999)
    final_df['_ev_sort_temp'] = ev_sort

    final_df = final_df.sort_values(by=['tier_score', '_ev_sort_temp'], ascending=[True, False]).reset_index(drop=True)
    final_df = final_df.drop(columns=['_ev_sort_temp'])
    final_df['Triple_Filter_Rank'] = range(1, len(final_df) + 1)

    # Clean up temporary columns
    # final_df = final_df.drop(columns=['is_unique', 'tier_score'], errors='ignore')

    return final_df


def _fade_theover(theover, win_prob_source, fade_sources, shrink):
    """Shrink TheOver P(Over) toward 0.50 for faded WinProbSource rows.

    ``model_hit_rate_flipped`` is a *genuine* TheOver Under pick (P(Over)=1-hit_rate),
    not a fallback — but TheOver's Under model has been cold recently, so we temper its
    influence rather than trust or discard it. ``shrink`` is the fraction of the
    deviation-from-0.50 removed: 1.0 fully neutralizes (-> 0.50), 0.0 leaves it untouched.
    Returns a float ndarray. NaN/absent values pass through unchanged.
    """
    theover = np.asarray(theover, dtype=float)
    if win_prob_source is None or not fade_sources or shrink <= 0:
        return theover
    norm = {str(s).strip().lower() for s in fade_sources}
    src = pd.Series(win_prob_source).astype("string").str.strip().str.lower()
    faded = src.isin(norm).fillna(False).to_numpy()
    out = theover.copy()
    out[faded] = 0.5 + (theover[faded] - 0.5) * (1.0 - float(shrink))
    return out


def _scoped_theover_blend_fade(theover_arr, win_prob_source, league, market_type, index):
    """Fade TheOver's blend input with the MLB-tuned shrink on MLB totals and the
    non-MLB default everywhere else.

    ``MLB_THEOVER_FADE_SHRINK`` is tuned on MLB pitcher-friendly games, but a faded
    WinProbSource (``model_hit_rate_flipped``) can ride on NBA/NHL totals too. Applying
    the MLB shrink frame-wide would silently change those non-MLB calibrated
    probabilities, so MLB totals get ``MLB_THEOVER_FADE_SHRINK`` while every other
    league/market keeps ``THEOVER_FADE_SHRINK_DEFAULT``. Pure + shared by both blend
    call sites (run_analysis_pipeline and the post-Kalshi re-blend) so the scoping is
    unit-testable and cannot drift between them.
    """
    faded_mlb = pd.Series(
        _fade_theover(theover_arr, win_prob_source, MLB_THEOVER_FADE_SOURCES, MLB_THEOVER_FADE_SHRINK),
        index=index,
    )
    faded_default = pd.Series(
        _fade_theover(theover_arr, win_prob_source, MLB_THEOVER_FADE_SOURCES, THEOVER_FADE_SHRINK_DEFAULT),
        index=index,
    )
    is_mlb_total = (
        pd.Series(league, index=index).astype(str).str.upper().eq("MLB")
        & pd.Series(market_type, index=index).astype(str).str.lower().str.contains("total", na=False)
    )
    return faded_default.where(~is_mlb_total, faded_mlb)


def _mlb_total_direction_conflict(
    is_mlb_total: np.ndarray,
    kalshi_probability: np.ndarray,
    theover_probability: np.ndarray,
    theover_source=None,
    fade_sources=None,
    fade_shrink: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve MLB total Over/Under direction by "most-confident source wins".

    ``kalshi_probability`` and ``theover_probability`` are oriented to each row's
    pick direction: a value > 0.50 supports the row's direction, < 0.50 favors the
    opposite side, and a source's directional confidence is its distance from 0.50.
    A source at exactly 0.50 (TheOver's ``default_0.5`` "no read") or NaN is treated
    as having no opinion and never decides direction.

    ``theover_source`` (optional) is the per-row ``win_prob_source`` tag. Rows whose
    source is in ``fade_sources`` (e.g. ``model_hit_rate_flipped``, a genuine but
    recently-cold TheOver Under pick) have their P(Over) shrunk toward 0.50 by
    ``fade_shrink`` before the confidence comparison, so they pull direction less without
    being silenced. shrink=1.0 reproduces the old "drop it" behavior; 0.0 = full trust.

    A row is flagged as the losing direction when the strongest confidence *opposing*
    it exceeds the strongest confidence *supporting* it. For the Over and Under rows
    of the same game the opposing/supporting confidences swap, so exactly the losing
    side is flagged (a confidence tie flags neither and defers to EV/edge). This
    replaces the earlier pair of independent fixed penalties, which cancelled across
    the family whenever Kalshi and TheOver disagreed on direction.

    Returns ``(kalshi_opposes, theover_opposes, direction_conflict)`` boolean arrays,
    where ``direction_conflict`` marks the row whose ``final_family_score`` should be
    penalized so its counterpart wins the family sort.
    """

    def _opinion(prob: np.ndarray):
        prob = np.asarray(prob, dtype=float)
        has = ~np.isnan(prob) & (np.abs(prob - 0.5) > 1e-9)
        conf = np.where(has, np.abs(prob - 0.5), -1.0)
        return conf, has & (prob < 0.5), has & (prob > 0.5)

    is_mlb_total = np.asarray(is_mlb_total, dtype=bool)
    theover_probability = _fade_theover(theover_probability, theover_source, fade_sources, fade_shrink)

    k_conf, k_opp, k_sup = _opinion(kalshi_probability)
    t_conf, t_opp, t_sup = _opinion(theover_probability)

    opp_conf = np.maximum(np.where(k_opp, k_conf, -1.0), np.where(t_opp, t_conf, -1.0))
    sup_conf = np.maximum(np.where(k_sup, k_conf, -1.0), np.where(t_sup, t_conf, -1.0))

    direction_conflict = is_mlb_total & (opp_conf > sup_conf)
    return (k_opp & direction_conflict, t_opp & direction_conflict, direction_conflict)


def _edge_no_stake_demotion(
    *,
    league: str | None,
    market_type: str | None,
    consensus: str | None,
    total_line: float | None,
    status: str | None,
    neutral_no_stake: bool,
    mid_line_no_stake: bool,
    mid_min: float,
    mid_max: float,
):
    """Edge-based no-stake gates for MLB totals (graded 20 May-7 Jun, n=182).

    Returns ``(new_status, reason, blocker_stage)`` when a currently-stakeable MLB total
    falls in a bucket that bled below the -110 breakeven (52.4%), else ``(None, None,
    None)``. Only acts on Actionable / High Variance rows — never promotes. Pure +
    side-effect-free for unit testing (tests/test_edge_no_stake_gates.py).

    Rule 1 — Neutral-consensus totals: Over/Neutral hit 48.2% (n=56), the single largest
      losing cell, while Agrees (61.4%) and Disagrees (63.2%) totals keep their edge.
    Rule 2 — mid-line Overs (``mid_min`` <= line < ``mid_max``, i.e. 8.0-9.5): 46.5%
      (n=43), while the Under on those same lines hit 65.4%.
    """
    if (league or "").strip().upper() != "MLB":
        return (None, None, None)
    if status not in ("Actionable", "High Variance/Speculative"):
        return (None, None, None)
    mt = (market_type or "").strip().lower()
    is_total = "total" in mt

    if neutral_no_stake and is_total and (consensus or "").strip() == "Neutral":
        return (
            "Below Threshold",
            "Below Threshold: Neutral-consensus MLB total — Neutral O/U hit ~48% "
            "(graded n=56), no edge vs -110; only Agrees/Disagrees totals are staked",
            "mlb_total_neutral_no_stake",
        )

    if (mid_line_no_stake and mt == "total_over" and total_line is not None
            and float(mid_min) <= float(total_line) < float(mid_max)):
        return (
            "Below Threshold",
            f"Below Threshold: MLB Over line {float(total_line)} in mid bucket "
            f"[{mid_min}, {mid_max}) — mid-line Overs hit ~46% (graded n=43); "
            f"the Under is the edge side here",
            "mlb_over_mid_line_no_stake",
        )

    return (None, None, None)


def _total_over_concentration_downgrades(
    candidates: pd.DataFrame, *, overall_cap: int, mlb_cap: int, flag_col: str = "_is_mlb_over"
) -> list:
    """Greedy keep of the best-ranked same-direction totals picks under both an overall
    cap and an MLB-specific sub-cap; return the indices of the lowest-ranked excess to
    downgrade.

    ``candidates`` must already be sorted best-first and carry a boolean ``flag_col``
    column (the MLB-specific membership flag). Direction-agnostic: the over guard passes
    ``_is_mlb_over`` and the under guard passes ``_is_mlb_under``. Used by the speculative
    (High Variance) and empirical-overlay concentration guards so the selection logic is
    unit-testable in isolation.
    """
    kept_total = 0
    kept_mlb = 0
    downgrade_idx: list = []
    for idx, is_mlb in candidates[flag_col].items():
        is_mlb = bool(is_mlb)
        overall_ok = kept_total < int(overall_cap)
        mlb_ok = (not is_mlb) or (kept_mlb < int(mlb_cap))
        if overall_ok and mlb_ok:
            kept_total += 1
            if is_mlb:
                kept_mlb += 1
        else:
            downgrade_idx.append(idx)
    return downgrade_idx


def build_best_picks_df(analysis_df: pd.DataFrame, diagnostics_out: dict | None = None) -> pd.DataFrame:
    logger.info(f"BEST PICKS AUDIT: Received analysis_df with {len(analysis_df)} rows")
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)
    if "market_type" not in analysis_df.columns:
        raise ValueError("analysis_df missing market_type before best-pick construction")
    from core.slate_quality import spread_moneyline_orientation_fault

    pool = analysis_df[_string_series(analysis_df, "market_type").isin(list(VALID_MARKETS))].copy()
    logger.info(f"BEST PICKS AUDIT: Rows after VALID_MARKETS filter: {len(pool)}")
    if pool.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)

    pool["expected_value"] = _numeric_series(pool, "expected_value")
    pool["edge"] = _numeric_series(pool, "edge", 0.0)

    # Ensure lines are numeric before applying best_pick formatter
    if "spread_line" in pool.columns:
        pool["spread_line"] = pd.to_numeric(pool["spread_line"], errors="coerce")
    if "total_line" in pool.columns:
        pool["total_line"] = pd.to_numeric(pool["total_line"], errors="coerce")

    pool["best_pick"] = pool.apply(_format_best_pick, axis=1)
    pool["league"] = _clean_text_placeholders(_string_series(pool, "league")).astype("string").str.strip()
    pool["home_team"] = _clean_text_placeholders(_string_series(pool, "home_team")).astype("string").str.strip()
    pool["away_team"] = _clean_text_placeholders(_string_series(pool, "away_team")).astype("string").str.strip()
    pool["game_date"] = _game_dates(pool)

    # Removed strict game_date.notna() requirement to prevent dropping Spread uploads
    pool["has_identity"] = (
        pool["home_team"].str.len().gt(0)
        & pool["away_team"].str.len().gt(0)
    )

    game_key_present = _clean_text_placeholders(_string_series(pool, "game_key")).str.len().gt(0) if "game_key" in pool.columns else pd.Series([False] * len(pool), index=pool.index)
    pool = pool[pool["has_identity"] | game_key_present].copy()
    logger.info(f"BEST PICKS AUDIT: Rows after identity/game_key filter: {len(pool)}")
    if pool.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)

    pool["has_signal_probability"] = _numeric_series(pool, "model_probability").notna() | _numeric_series(pool, "theover_probability").notna() | _numeric_series(pool, "ml_probability").notna()

    # Canonical per-game identity key used for full game coverage selection.
    # Guaranteed 1:1 mapping: one selected pick for each unique game_date/home/away combination.
    dt_utc = _game_dates(pool)
    date_str = pd.Series([""] * len(pool), index=pool.index, dtype="string")
    valid_dt = dt_utc.notna()
    if valid_dt.any():
        date_str.loc[valid_dt] = dt_utc[valid_dt].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    pool["matchup_id"] = (
        date_str
        + "|" + pool["home_team"].str.lower().str.replace(r"\s+", " ", regex=True)
        + "|" + pool["away_team"].str.lower().str.replace(r"\s+", " ", regex=True)
    )

    # Force expected_value to numeric, converting true errors to NaN while preserving negative floats
    pool["expected_value"] = pd.to_numeric(pool["expected_value"], errors="coerce")

    # 2. Assign calibrated probability with a default
    pool["calibrated_probability"] = _numeric_series(pool, "calibrated_probability", 0.5)

    # 1. Add the ranking metadata
    pool = _apply_triple_filter_ranking(pool)
    logger.info(f"BEST PICKS AUDIT: Rows passed through Triple Filter Ranking: {len(pool)}")

    # 2. Assign valid numeric values for EV and Edge sorting
    pool["_ev_numeric"] = pd.to_numeric(pool["expected_value"], errors="coerce")
    pool["_edge_numeric"] = pd.to_numeric(pool["edge"], errors="coerce").fillna(0.0)

    # Ensure market_type is available to distinguish sides vs totals
    pool["_market_family"] = pool["market_type"].astype(str).str.lower().apply(
        lambda x: "total" if "total" in x else "side"
    )

    # 3. Calculate Normalized EV and Normalized Edge (Z-score style) within market families
    # Initialize normalized columns with NaN
    pool["_normalized_ev"] = pd.Series([np.nan] * len(pool), index=pool.index, dtype="float64")
    pool["_normalized_edge"] = pd.Series([np.nan] * len(pool), index=pool.index, dtype="float64")

    # Store diagnostics
    raw_counts = pool["_market_family"].value_counts().to_dict()
    raw_market_type_counts = pool["market_type"].value_counts().to_dict()
    avg_scores = {}

    for family in ["total", "side"]:
        family_mask = pool["_market_family"] == family
        count = family_mask.sum()
        if count > 1:
            # EV Z-score
            mean_ev = pool.loc[family_mask, "_ev_numeric"].mean()
            std_ev = pool.loc[family_mask, "_ev_numeric"].std()
            if std_ev > 0:
                pool.loc[family_mask, "_normalized_ev"] = (pool.loc[family_mask, "_ev_numeric"] - mean_ev) / std_ev
            else:
                pool.loc[family_mask, "_normalized_ev"] = 0.0

            # Edge Z-score
            mean_edge = pool.loc[family_mask, "_edge_numeric"].mean()
            std_edge = pool.loc[family_mask, "_edge_numeric"].std()
            if std_edge > 0:
                pool.loc[family_mask, "_normalized_edge"] = (pool.loc[family_mask, "_edge_numeric"] - mean_edge) / std_edge
            else:
                pool.loc[family_mask, "_normalized_edge"] = 0.0
        elif count == 1:
            pool.loc[family_mask, "_normalized_ev"] = 0.0
            pool.loc[family_mask, "_normalized_edge"] = 0.0

    # Fill NaNs with 0.0 only at the scoring step for safely computing final_family_score
    pool["final_family_score"] = 0.5 * pool["_normalized_ev"].fillna(0.0) + 0.5 * pool["_normalized_edge"].fillna(0.0)

    for family in ["total", "side"]:
        family_mask = pool["_market_family"] == family
        avg_scores[family] = pool.loc[family_mask, "final_family_score"].mean() if family_mask.sum() > 0 else 0.0

    # Small under-family adjustment so totals-under do not dominate cross-family finalists
    # on generic EV/edge momentum alone.
    pool["_under_selection_penalty"] = np.where(
        pool["market_type"].astype(str).str.lower().eq("total_under"),
        float(TOTAL_UNDER_FINALIST_SCORE_PENALTY),
        0.0,
    )
    pool["_mlb_spread_finalist_penalty"] = np.where(
        (pool["league"].astype(str).str.upper().eq("MLB"))
        & (pool["market_type"].astype(str).str.lower().str.contains("spread", na=False)),
        float(MLB_SPREAD_FINALIST_SCORE_PENALTY),
        0.0,
    )
    pool["_family_selection_penalty"] = pool["_under_selection_penalty"] + pool["_mlb_spread_finalist_penalty"]
    pool["final_family_score"] = pool["final_family_score"] - pool["_family_selection_penalty"]
    pool["final_family_score_no_mlb_spread_penalty"] = pool["final_family_score"] + pool["_mlb_spread_finalist_penalty"]

    # MLB total direction resolution — "most-confident source wins".
    #
    # kalshi_probability and theover_probability are pre-oriented to each row's pick
    # direction (P(Over) on Over rows, P(Under) on Under rows). A value < 0.50 means
    # that source favors the OPPOSITE direction; > 0.50 means it supports this row.
    #
    # Previously TheOver and Kalshi each subtracted an independent fixed penalty. When
    # the two sources DISAGREED on direction (e.g. TheOver leans Under, Kalshi leans
    # Over) the penalties hit opposite rows and CANCELLED across the family — the Over
    # row lost MLB_THEOVER_CONFLICT_PENALTY from TheOver and the Under row lost the same
    # from Kalshi — collapsing the decision to raw EV/edge momentum. That is exactly how
    # the 1 Jun slate flipped three winning Overs to losing Unders (TheOver had just
    # published strong Under reads that inflated the Under EV while Kalshi still priced
    # the Over). See app_core/weights_config.py for the opposing May 22/23 history where
    # TheOver's pitcher read was the correct, more-confident side.
    #
    # We now resolve direction by confidence: each source's directional confidence is its
    # distance from 0.50, and a row is penalized only when the MORE-CONFIDENT verdict
    # opposes it. When the two sources agree, they reinforce; when they disagree, the
    # source further from 0.50 dictates direction; a lone present source decides on its
    # own. TheOver emits exactly 0.50 when it has no real pitcher-based read
    # (default_0.5) — that, and NaN, are treated as "no opinion" so they never win.
    #
    # WinProbSource gating: some TheOver sources are not genuine per-game reads. The
    # `model_hit_rate_flipped` source collapses to a near-constant ~0.30 P(Over) across
    # a whole slate (8/10 totals on the 2 Jun upload), which at 0.30 would otherwise
    # carry enough confidence to override Kalshi on every game. We blank TheOver to NaN
    # ("no opinion") for the DIRECTION decision when its source is untrusted, so the
    # direction defers to Kalshi/EV instead of a flat flipped hit-rate. The blend that
    # produced WinProbability is left untouched.
    _is_mlb_total = (
        pool["league"].astype(str).str.upper().eq("MLB")
        & pool["market_type"].astype(str).str.lower().str.contains("total", na=False)
    ).to_numpy()
    _kalshi_arr = (
        pd.to_numeric(pool["kalshi_probability"], errors="coerce").to_numpy(dtype=float)
        if "kalshi_probability" in pool.columns
        else np.full(len(pool), np.nan)
    )
    _theover_arr = (
        pd.to_numeric(pool["theover_probability"], errors="coerce").to_numpy(dtype=float)
        if "theover_probability" in pool.columns
        else np.full(len(pool), np.nan)
    )
    _theover_source = pool["win_prob_source"] if "win_prob_source" in pool.columns else None
    _kalshi_opp, _theover_opp, _direction_conflict = _mlb_total_direction_conflict(
        _is_mlb_total, _kalshi_arr, _theover_arr,
        theover_source=_theover_source,
        fade_sources=MLB_THEOVER_FADE_SOURCES,
        fade_shrink=MLB_THEOVER_FADE_SHRINK,
    )
    pool["_direction_conflict_penalty"] = np.where(
        _direction_conflict, float(MLB_THEOVER_CONFLICT_PENALTY), 0.0
    )
    pool["final_family_score"] = pool["final_family_score"] - pool["_direction_conflict_penalty"]
    # Back-compat: keep the legacy per-source columns populated for debug/transparency,
    # attributing the penalty to whichever source(s) opposed the losing direction.
    pool["_theover_conflict_penalty"] = np.where(
        _direction_conflict & _theover_opp, float(MLB_THEOVER_CONFLICT_PENALTY), 0.0
    )
    pool["_kalshi_direction_conflict_penalty"] = np.where(
        _direction_conflict & _kalshi_opp, float(MLB_THEOVER_CONFLICT_PENALTY), 0.0
    )

    # 4. Sort to prepare for finalist selection within each family per game
    pool = pool.sort_values(
        by=["final_family_score", "tier_score", "_ev_numeric", "_edge_numeric", "calibrated_probability"],
        ascending=[False, True, False, False, False],
        na_position="last"
    )

    # 5. First Stage: Best side finalist vs Best total finalist per game
    # Because we sorted exactly above, drop_duplicates on matchup_id + family gives the true best per family
    finalists = pool.drop_duplicates(subset=["matchup_id", "_market_family"], keep="first").copy()

    finalist_counts = finalists["_market_family"].value_counts().to_dict()
    finalist_market_type_counts = finalists["market_type"].value_counts().to_dict()

    # 6. Second Stage: Compare the two finalists and choose exactly one final winner per game
    preview_rows = []
    final_winner_indices = []
    demoted_by_mlb_spread_finalist_penalty = 0

    # Iterate over each unique game to do the direct comparison
    for matchup, group in finalists.groupby("matchup_id"):
        side_row = group[group["_market_family"] == "side"]
        total_row = group[group["_market_family"] == "total"]

        side_pick = side_row["best_pick"].iloc[0] if not side_row.empty else "None"
        side_ev = side_row["_ev_numeric"].iloc[0] if not side_row.empty else 0.0
        side_edge = side_row["_edge_numeric"].iloc[0] if not side_row.empty else 0.0
        side_score = side_row["final_family_score"].iloc[0] if not side_row.empty else 0.0
        side_market_type = side_row["market_type"].iloc[0] if not side_row.empty else "None"
        side_candidate_source = side_row["candidate_source"].iloc[0] if not side_row.empty and "candidate_source" in side_row.columns else "None"

        total_pick = total_row["best_pick"].iloc[0] if not total_row.empty else "None"
        total_ev = total_row["_ev_numeric"].iloc[0] if not total_row.empty else 0.0
        total_edge = total_row["_edge_numeric"].iloc[0] if not total_row.empty else 0.0
        total_score = total_row["final_family_score"].iloc[0] if not total_row.empty else 0.0
        total_market_type = total_row["market_type"].iloc[0] if not total_row.empty else "None"
        total_candidate_source = total_row["candidate_source"].iloc[0] if not total_row.empty and "candidate_source" in total_row.columns else "None"

        # Sort the finalists for this matchup to find the absolute winner
        group_sorted = group.sort_values(
            by=["final_family_score", "tier_score", "_ev_numeric", "_edge_numeric", "calibrated_probability"],
            ascending=[False, True, False, False, False],
            na_position="last"
        )
        group_sorted_no_mlb_spread_penalty = group.sort_values(
            by=["final_family_score_no_mlb_spread_penalty", "tier_score", "_ev_numeric", "_edge_numeric", "calibrated_probability"],
            ascending=[False, True, False, False, False],
            na_position="last"
        )

        winner_row = group_sorted.iloc[0]
        winner_row_no_mlb_penalty = group_sorted_no_mlb_spread_penalty.iloc[0]
        final_winner_indices.append(winner_row.name)
        if (
            winner_row_no_mlb_penalty.name != winner_row.name
            and str(winner_row_no_mlb_penalty.get("league", "")).upper() == "MLB"
            and "spread" in str(winner_row_no_mlb_penalty.get("market_type", "")).lower()
        ):
            demoted_by_mlb_spread_finalist_penalty += 1

        winner_family = winner_row["_market_family"]
        winner_pick = winner_row["best_pick"]
        winner_reason = winner_row.get("Status_Reason", "N/A (determined later)")
        score_delta = abs(side_score - total_score) if not side_row.empty and not total_row.empty else 0.0

        preview_rows.append({
            "matchup_id": matchup,
            "side_pick": side_pick,
            "side_market_type": side_market_type,
            "side_source": side_candidate_source,
            "side_ev": side_ev,
            "side_edge": side_edge,
            "side_score": side_score,
            "total_pick": total_pick,
            "total_market_type": total_market_type,
            "total_source": total_candidate_source,
            "total_ev": total_ev,
            "total_edge": total_edge,
            "total_score": total_score,
            "winner": winner_pick,
            "winner_family": winner_family,
            "score_delta": score_delta,
            "winner_reason": winner_reason,
            "has_side_finalist": not side_row.empty,
            "has_total_finalist": not total_row.empty,
        })

    preview_df = pd.DataFrame(preview_rows)

    # Select final winner
    best = finalists.loc[final_winner_indices].copy()

    final_counts = best["_market_family"].value_counts().to_dict()
    final_market_type_counts = best["market_type"].value_counts().to_dict()

    # Cleanup temporary columns
    best = best.drop(columns=["_market_family", "_normalized_ev", "_normalized_edge", "final_family_score", "final_family_score_no_mlb_spread_penalty", "_ev_numeric", "_edge_numeric", "_family_selection_penalty", "_under_selection_penalty", "_mlb_spread_finalist_penalty"])

    logger.info(f"BEST PICKS AUDIT: Rows after two-stage finalist comparison: {len(best)} (started with {len(pool)})")

    # Pre-compute consensus agreement before status labelling so overlays can use it
    if "consensus_agreement" not in best.columns:
        best["consensus_agreement"] = "No Kalshi"
    else:
        best["consensus_agreement"] = best["consensus_agreement"].fillna("No Kalshi")

    kalshi_prob = _numeric_series(best, "kalshi_probability") if "kalshi_probability" in best.columns else pd.Series([pd.NA]*len(best), index=best.index)
    is_kalshi_available = ((~pd.isna(kalshi_prob)) & (kalshi_prob > 0.0)).fillna(False).astype(bool)
    best["is_kalshi_available"] = is_kalshi_available

    if is_kalshi_available.any():
        # DIRECTIONAL consensus (16 Jun): does Kalshi back the SAME side as our pick?
        # kalshi_probability is pre-oriented to the pick (P(Under) on Under rows), so
        # >= 0.50 means Kalshi favors our pick and < 0.50 means Kalshi favors the other
        # side. The model's confidence MAGNITUDE relative to Kalshi no longer matters:
        # if both favor the side they AGREE, even when Kalshi is the more confident of
        # the two. (The old rule required the model to LEAD Kalshi by >= 0.03 to "Agree"
        # and tagged everything else "Disagrees"; after the market-trust reweight pulled
        # the model below the confident Kalshi number, that mislabeled every same-side
        # pick as "Disagrees".) A small band around 0.50 is Neutral (Kalshi pick'em).
        # Safety note: this broadens "Agrees", but actual staking stays gated by the
        # realized empirical-bucket bar (>=55% over >=25 graded picks) in the tier
        # overlay, which the daily loop refits — so the directional label routes picks
        # into buckets, it does not by itself loosen the Actionable stake gate.
        _CONSENSUS_NEUTRAL_BAND = 0.02
        agrees_mask = (is_kalshi_available & kalshi_prob.ge(0.50 + _CONSENSUS_NEUTRAL_BAND)).fillna(False).astype(bool)
        disagrees_mask = (is_kalshi_available & kalshi_prob.le(0.50 - _CONSENSUS_NEUTRAL_BAND)).fillna(False).astype(bool)
        best.loc[is_kalshi_available, "consensus_agreement"] = "Neutral"
        best.loc[agrees_mask, "consensus_agreement"] = "Agrees"
        best.loc[disagrees_mask, "consensus_agreement"] = "Disagrees"

    # Recompute the per-signal breakdown here, where Kalshi is merged onto every row.
    # The copy stamped in run_analysis_pipeline runs before the live-odds bet rows get
    # their Kalshi values, so it drops the Kalshi piece; refreshing it now keeps the
    # exported string consistent with the blend_in_*/kalshi_probability columns.
    best["signal_breakdown"] = _compute_signal_breakdown(best)


    # Phase 5: Enforce Thresholds and Pick Status Labelling
    # MIN_EDGE_THRESHOLD of 0.01 for high-liquidity markets.
    # Expected Value Floor of 0.005.

    # Pick_Status logic setup
    if "Pick_Status" not in best.columns:
        best["Pick_Status"] = pd.Series([""] * len(best), index=best.index, dtype="string")

    blocked_by_under_specific_thresholds = 0
    blocked_by_nba_total_penalty = 0
    blocked_by_no_kalshi_total_penalty = 0
    blocked_by_mlb_spread_penalty = 0
    blocked_by_mlb_over_promotion_gate = 0
    promoted_by_nba_side_bonus = 0
    promoted_by_nba_over_bonus = 0
    divergence_rows_preserved = 0
    divergence_rows_blocked_by_viability_floor = 0
    divergence_rows_negative_ev = 0
    divergence_rows_negative_edge = 0
    high_variance_due_only_high_ev = 0
    promoted_high_ev_to_actionable_no_uncertainty = 0
    high_variance_capped_due_to_divergence = 0
    high_variance_capped_due_to_no_kalshi = 0
    high_variance_capped_due_to_suspicious_data = 0
    high_variance_capped_due_to_degraded_subset = 0
    high_variance_capped_due_to_fallback_heavy = 0
    side_balance_promotions = 0

    if "suspicious_data_flag" not in best.columns:
        best["suspicious_data_flag"] = False
    if "suspicious_data_reasons" not in best.columns:
        best["suspicious_data_reasons"] = ""
    if "status_blocker_reason" not in best.columns:
        best["status_blocker_reason"] = ""
    if "status_blocker_stage" not in best.columns:
        best["status_blocker_stage"] = "none"
    if "status_metric_basis" not in best.columns:
        best["status_metric_basis"] = "raw"
    if "effective_expected_value" not in best.columns:
        best["effective_expected_value"] = pd.NA
    if "effective_edge" not in best.columns:
        best["effective_edge"] = pd.NA
    if "effective_win_probability" not in best.columns:
        best["effective_win_probability"] = pd.NA

    # Earned-Actionable relaxation gate for MLB overs (17 Jun): the strict 0.65 prob
    # bar leaves over-heavy slates empty. We allow a lower bar for Agrees overs, but
    # ONLY when the realized MLB:over:Agrees bucket has earned trust (>=55% over >=25
    # graded picks) — the same proof the empirical overlay's Actionable promotion uses.
    # Computed once here so the per-row gate is a cheap lookup. When no graded history
    # exists yet (bucket stats unavailable) we treat the relaxation as available so the
    # card is not permanently empty pre-calibration; once slates are graded the proven
    # condition becomes the binding backstop.
    _mlb_over_agrees_relax_ok = False
    _mlb_over_agrees_bucket_stats_available = False
    try:
        from core.empirical_tiers import (
            load_bucket_stats as _load_bucket_stats,
            smoothed_bucket_rate as _smoothed_bucket_rate,
            ACTIONABLE_MIN_BUCKET_N as _ACT_MIN_N,
            ACTIONABLE_MIN_BUCKET_RATE as _ACT_MIN_RATE,
        )
        _bs = _load_bucket_stats()
        if _bs:
            _mlb_over_agrees_bucket_stats_available = True
            _rate, _n = _smoothed_bucket_rate("MLB:over:Agrees", _bs)
            _mlb_over_agrees_relax_ok = (_n >= _ACT_MIN_N) and (_rate >= _ACT_MIN_RATE)
        else:
            # No graded history yet — let the relaxed bar apply so over-heavy slates
            # can surface Agrees overs; the empirical overlay (once fed) re-gates them.
            _mlb_over_agrees_relax_ok = True
    except Exception:
        _mlb_over_agrees_relax_ok = False

    for idx in best.index:
        status_reason = "Unknown"
        blocker_stage = "none"
        bp = str(best.at[idx, "best_pick"])
        ev = best.at[idx, "expected_value"]
        edge = best.at[idx, "edge"]
        market_type = str(best.at[idx, "market_type"]) if "market_type" in best.columns else ""
        league = str(best.at[idx, "league"]).upper() if "league" in best.columns else ""

        # Probabilities for divergence check
        ml_prob = best.at[idx, "ml_probability"] if "ml_probability" in best.columns else pd.NA
        kalshi_prob = best.at[idx, "kalshi_probability"] if "kalshi_probability" in best.columns else pd.NA

        # Calibrated/Win probability (ensure 0-1)
        win_prob = best.at[idx, "calibrated_probability"] if "calibrated_probability" in best.columns else 0.5
        win_prob = win_prob if pd.notna(win_prob) else 0.5
        effective_ev = ev
        effective_edge = edge
        effective_win_probability = win_prob
        status_metric_basis = "raw"

        # Check fallback indicators
        stale = bool(best.at[idx, "used_stale_features"]) if "used_stale_features" in best.columns and pd.notna(best.at[idx, "used_stale_features"]) else False
        odds_source = str(best.at[idx, "odds_source"]).lower() if "odds_source" in best.columns else ""
        degraded_subset_flag = bool(best.at[idx, "degraded_feature_subset_flag"]) if "degraded_feature_subset_flag" in best.columns and pd.notna(best.at[idx, "degraded_feature_subset_flag"]) else False

        # Additional row-specific fallback signals based on requirement
        is_live_data = bool(best.at[idx, "is_live_data"]) if "is_live_data" in best.columns and pd.notna(best.at[idx, "is_live_data"]) else True

        is_fallback_or_stale = (stale or not is_live_data or "fallback_novig" in odds_source)

        # Determine status (strict precedence)
        is_missing_line = False
        if "(No Line)" in bp:
            is_missing_line = True

        # Also literally check for missing numeric lines
        if "spread" in market_type.lower():
            if "spread_line" not in best.columns or pd.isna(best.at[idx, "spread_line"]):
                is_missing_line = True
        elif "total" in market_type.lower():
            if "total_line" not in best.columns or pd.isna(best.at[idx, "total_line"]):
                is_missing_line = True
        is_nba_extreme_spread = False
        if league == "NBA" and "spread" in market_type.lower():
            # Support spread_line or spread fallback
            line_val = best.at[idx, "spread_line"] if "spread_line" in best.columns else pd.NA
            if pd.isna(line_val) and "spread" in best.columns:
                 line_val = best.at[idx, "spread"]

            if pd.notna(line_val):
                if abs(float(line_val)) > 12.0:
                    is_nba_extreme_spread = True

        from app_core.weights_config import (
            TOTAL_UNDER_MIN_WIN_PROB, TOTAL_UNDER_MIN_EV, TOTAL_UNDER_MIN_EDGE,
            NHL_TOTAL_EXTRA_EDGE_PENALTY, MLB_SPREAD_MIN_WIN_PROB,
            MLB_SPREAD_ACTIONABLE_PENALTY,
            NBA_SIDE_ACTIONABLE_BONUS, NBA_OVER_ACTIONABLE_BONUS,
            MLB_OVER_ACTIONABLE_MIN_PROB, MLB_OVER_ACTIONABLE_MIN_EV, MLB_OVER_ACTIONABLE_MIN_EDGE,
            MLB_OVER_AGREES_ACTIONABLE_MIN_PROB, MLB_OVER_AGREES_ACTIONABLE_MIN_EV, MLB_OVER_AGREES_ACTIONABLE_MIN_EDGE,
            MLB_TOTAL_OVER_ACTIONABLE_PENALTY, MLB_TOTAL_UNDER_ACTIONABLE_PENALTY,
            NBA_TOTAL_OVER_ACTIONABLE_PENALTY, NBA_TOTAL_UNDER_ACTIONABLE_PENALTY,
            NHL_TOTAL_OVER_ACTIONABLE_PENALTY, NHL_TOTAL_UNDER_ACTIONABLE_PENALTY,
            NO_KALSHI_TOTAL_EXTRA_PENALTY, NO_KALSHI_TOTAL_UNDER_EXTRA_PENALTY,
            LEAGUE_MARKET_FAMILY_ACTIONABLE_PENALTIES,
            FALLBACK_HEAVY_TOTAL_EXTRA_PENALTY,
            BEST_PICKS_PROFILE,
            LOCK_UPLOAD_LINES_FOR_MATCHED_ROWS,
            BASELINE_MIN_EV, BASELINE_MIN_EDGE,
            TOTAL_OVER_MIN_EV, TOTAL_OVER_MIN_EDGE,
            TOTAL_MIN_WIN_PROB, NHL_TOTAL_MIN_WIN_PROB,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_PROB,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_EV,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_EDGE,
            DIVERGENCE_HIGH_VARIANCE_MIN_EV, DIVERGENCE_HIGH_VARIANCE_MIN_EDGE, DIVERGENCE_HIGH_VARIANCE_MIN_PROB,
            DIVERGENCE_HIGH_EV_OVERRIDE_MIN_EV, DIVERGENCE_HIGH_EV_OVERRIDE_MIN_EDGE, DIVERGENCE_HIGH_EV_OVERRIDE_MIN_PROB,
            SIDE_MIN_WIN_PROB,
            NEUTRAL_ACTIONABLE_MIN_PROB, NEUTRAL_ACTIONABLE_MIN_EV, NEUTRAL_ACTIONABLE_MIN_EDGE,
            DISAGREES_ACTIONABLE_MIN_PROB, DISAGREES_ACTIONABLE_MIN_EV, DISAGREES_ACTIONABLE_MIN_EDGE,
            MLB_SPREAD_HIGH_EV_OVERRIDE_MIN_EV, MLB_SPREAD_HIGH_EV_OVERRIDE_MIN_EDGE,
            MLB_SPREAD_HIGH_EV_MIN_WIN_PROB,
            MLB_TOTAL_UNDER_MIN_WIN_PROB,
            MLB_HIGH_TOTAL_LINE_THRESHOLD, MLB_HIGH_TOTAL_LINE_OVER_PENALTY,
            MLB_MID_TOTAL_LINE_THRESHOLD, MLB_MID_TOTAL_LINE_OVER_PENALTY,
            MLB_TOTAL_HV_MIN_WIN_PROB,
            NHL_UNDER_ACTIONABLE_CAP,
            TOTAL_ML_CONTRADICTION_OVER_MAX_PROB,
            MLB_OVER_MIN_TOTAL_LINE,
            MLB_TOTAL_NEUTRAL_NO_STAKE,
            MLB_OVER_MID_LINE_NO_STAKE,
        )

        is_kalshi_divergence = False
        is_spread_divergence_override = False
        if pd.notna(ml_prob) and pd.notna(kalshi_prob):
            _div_thresholds = {
                "NBA": KALSHI_DIVERGENCE_THRESHOLD_NBA,
                "MLB": KALSHI_DIVERGENCE_THRESHOLD_MLB,
                "NHL": KALSHI_DIVERGENCE_THRESHOLD_NHL,
            }
            _div_threshold = _div_thresholds.get(str(league).upper(), KALSHI_DIVERGENCE_THRESHOLD)
            if abs(float(ml_prob) - float(kalshi_prob)) > _div_threshold:
                is_kalshi_divergence = True
                if "spread" in market_type.lower() and not pd.isna(ev) and not pd.isna(edge):
                    if win_prob >= SPREAD_DIVERGENCE_OVERRIDE_MIN_PROB and ev >= SPREAD_DIVERGENCE_OVERRIDE_MIN_EV and edge >= SPREAD_DIVERGENCE_OVERRIDE_MIN_EDGE:
                        is_kalshi_divergence = False
                        is_spread_divergence_override = True

        model_status_str = str(best.at[idx, "model_status"]) if "model_status" in best.columns else ""
        is_model_failure = "Fallback" in model_status_str or "Failure" in model_status_str

        suspicious_reasons: list[str] = []
        high_ev_guardrail = (not pd.isna(ev)) and float(ev) > 0.40
        if high_ev_guardrail:
            if is_missing_line:
                suspicious_reasons.append("missing_market_line")
            if is_fallback_or_stale:
                suspicious_reasons.append("stale_or_fallback_odds_source")
            market_status = str(best.at[idx, "market_status"]).strip().lower() if "market_status" in best.columns else ""
            if market_status in {"suspended", "closed", "invalid", "halted"}:
                suspicious_reasons.append(f"market_status={market_status}")
            if "line_source" in best.columns and pd.notna(best.at[idx, "line_source"]):
                line_source = str(best.at[idx, "line_source"]).strip().lower()
                if line_source in {"unknown", "synthetic"}:
                    suspicious_reasons.append("malformed_or_synthetic_line_source")
            if "line_delta" in best.columns and pd.notna(best.at[idx, "line_delta"]):
                try:
                    if abs(float(best.at[idx, "line_delta"])) >= 8.0:
                        suspicious_reasons.append("line_source_mismatch")
                except Exception:
                    pass
            market_prob_val = best.at[idx, "market_probability"] if "market_probability" in best.columns else pd.NA
            if pd.notna(market_prob_val) and pd.notna(win_prob):
                try:
                    if abs(float(market_prob_val) - float(win_prob)) >= 0.25:
                        suspicious_reasons.append("inconsistent_price_probability")
                except Exception:
                    pass
            if "upload_market_match" in best.columns:
                upload_market_match = str(best.at[idx, "upload_market_match"]).strip().lower()
                if upload_market_match in {"false", "0", "mismatch"}:
                    suspicious_reasons.append("upload_line_market_mismatch")

        # Corrupt-odds sanity (UNGATED — runs regardless of EV). A two-way totals or
        # spread market whose de-vigged implied probability sits outside [0.05, 0.95]
        # is almost certainly a bad odds value from the feed: e.g. the 17 Jun Dodgers
        # Over 9.5 came in at +1983 (implied 4.8%), producing a nonsense +850% EV. Real
        # totals/run-line prices live well inside that band, so block the row before
        # the garbage EV reaches the card. Moneyline (heavy favorites/dogs) is exempt.
        corrupt_odds_flag = False
        corrupt_odds_reason = ""
        _mt_lower = market_type.lower()
        if _mt_lower.startswith("total") or _mt_lower.startswith("spread"):
            _mp_val = pd.to_numeric(best.at[idx, "market_probability"], errors="coerce") if "market_probability" in best.columns else pd.NA
            if pd.notna(_mp_val) and (float(_mp_val) < 0.05 or float(_mp_val) > 0.95):
                corrupt_odds_flag = True
                _odds_val = pd.to_numeric(best.at[idx, "odds_american"], errors="coerce") if "odds_american" in best.columns else pd.NA
                corrupt_odds_reason = (
                    f"corrupt odds (implied {float(_mp_val):.0%}"
                    + (f", {int(_odds_val):+d}" if pd.notna(_odds_val) else "")
                    + ") — line/price mismatch from feed"
                )
                suspicious_reasons.append(corrupt_odds_reason)

        # Spread orientation fault (ungated by EV): the spread favorite must be the
        # moneyline favorite. A mismatch means the live feed delivered a flipped
        # home/away spread (14 Jun: Texas shown -1.5/+158 — a favorite line — when
        # Texas was the +1.5 underdog). Block the row rather than ship the wrong side.
        spread_orientation_fault_flag = False
        spread_orientation_fault_reason = ""
        _mt_row = str(best.at[idx, "market_type"]).strip().lower() if "market_type" in best.columns else ""
        if _mt_row in {"spread_home", "spread_away"}:
            _line_for_orient = next(
                (best.at[idx, c] for c in ("market_line_used", "spread_line", "base_spread_line")
                 if c in best.columns and pd.notna(best.at[idx, c])),
                pd.NA,
            )
            _of, _of_reason = spread_moneyline_orientation_fault(
                _mt_row,
                _line_for_orient,
                best.at[idx, "game_home_ml_price"] if "game_home_ml_price" in best.columns else pd.NA,
                best.at[idx, "game_away_ml_price"] if "game_away_ml_price" in best.columns else pd.NA,
            )
            if _of:
                spread_orientation_fault_flag = True
                spread_orientation_fault_reason = _of_reason

        suspicious_data_flag = bool(suspicious_reasons)
        suspicious_reasons_str = "; ".join(dict.fromkeys(suspicious_reasons))
        divergence_viability_pass = (
            pd.notna(ev)
            and pd.notna(edge)
            and float(ev) >= DIVERGENCE_HIGH_VARIANCE_MIN_EV
            and float(edge) >= DIVERGENCE_HIGH_VARIANCE_MIN_EDGE
            and float(win_prob) >= DIVERGENCE_HIGH_VARIANCE_MIN_PROB
        )
        # High-EV override: a strongly +EV, +edge divergent pick whose win prob falls just
        # short of the 0.53 floor is preserved as High Variance/Speculative instead of being
        # dropped to No Play, provided the model still favors the pick (win prob >= 0.50).
        #
        # SCOPE (16 Jun): NOT applied to MLB totals. On the efficient MLB totals market the
        # 13-slate recap study (1-15 Jun, n=171) found model-vs-market divergence is
        # negatively predictive — the staked divergent overs went 33-39% while near-market
        # Below Threshold picks went 54%. Preserving divergent +EV MLB totals into the
        # staked tier therefore adds losing exposure, so they revert to No Play here.
        # Other leagues/markets keep the override.
        _is_mlb_total = (league == "MLB") and ("total" in market_type.lower())
        divergence_high_ev_override = (
            (not _is_mlb_total)
            and pd.notna(ev)
            and pd.notna(edge)
            and float(ev) >= DIVERGENCE_HIGH_EV_OVERRIDE_MIN_EV
            and float(edge) >= DIVERGENCE_HIGH_EV_OVERRIDE_MIN_EDGE
            and float(win_prob) >= DIVERGENCE_HIGH_EV_OVERRIDE_MIN_PROB
        )
        if divergence_high_ev_override:
            divergence_viability_pass = True

        if is_missing_line:
            status = "Missing Line"
            status_reason = "Missing odds or numerical line"
            blocker_stage = "line_integrity_guardrail"
        elif is_nba_extreme_spread:
            status = "No Play"
            status_reason = "NBA spread > 12.0 (Resting/Tanking guardrail)"
            blocker_stage = "line_integrity_guardrail"
        elif spread_orientation_fault_flag:
            status = "No Play"
            status_reason = f"No Play: {spread_orientation_fault_reason}"
            blocker_stage = "spread_orientation_guardrail"
        elif corrupt_odds_flag:
            status = "No Play"
            status_reason = f"No Play: {corrupt_odds_reason}"
            blocker_stage = "corrupt_odds_guardrail"
        elif is_kalshi_divergence:
            if divergence_viability_pass:
                status = "High Variance/Speculative"
                status_reason = "High Variance: capped due to divergence (ML and Kalshi probability diverge by > 20%)"
                if divergence_high_ev_override and float(win_prob) < DIVERGENCE_HIGH_VARIANCE_MIN_PROB:
                    status_reason = (
                        "High Variance: capped due to divergence (ML and Kalshi probability diverge by > 20%); "
                        f"preserved by high-EV override (EV >= {DIVERGENCE_HIGH_EV_OVERRIDE_MIN_EV:.2f}, "
                        f"Edge >= {DIVERGENCE_HIGH_EV_OVERRIDE_MIN_EDGE:.2f}, Win Prob >= {DIVERGENCE_HIGH_EV_OVERRIDE_MIN_PROB:.2f})"
                    )
                blocker_stage = "divergence_guardrail"
                divergence_rows_preserved += 1
                high_variance_capped_due_to_divergence += 1
            else:
                status = "No Play"
                status_reason = (
                    "No Play: divergence override denied by raw viability floor "
                    f"(EV >= {DIVERGENCE_HIGH_VARIANCE_MIN_EV:.2f}, Edge >= {DIVERGENCE_HIGH_VARIANCE_MIN_EDGE:.2f}, "
                    f"Win Prob >= {DIVERGENCE_HIGH_VARIANCE_MIN_PROB:.2f})"
                )
                blocker_stage = "divergence_viability_floor"
                divergence_rows_blocked_by_viability_floor += 1
                if pd.isna(ev) or float(ev) < DIVERGENCE_HIGH_VARIANCE_MIN_EV:
                    divergence_rows_negative_ev += 1
                if pd.isna(edge) or float(edge) < DIVERGENCE_HIGH_VARIANCE_MIN_EDGE:
                    divergence_rows_negative_edge += 1
        elif high_ev_guardrail and suspicious_data_flag:
            status = "No Play"
            status_reason = f"No Play: blocked due to suspicious data ({suspicious_reasons_str})"
            blocker_stage = "suspicious_data_guardrail"
        elif pd.isna(ev) or ev < 0 or win_prob < 0.40:
            status = "No Play"
            status_reason = "Negative EV or Base Win Prob < 40%"
            blocker_stage = "baseline_guardrail"
        elif is_fallback_or_stale or is_model_failure:
            status = "No Play"
            status_reason = "Using stale data or fallback model"
            blocker_stage = "data_fallback_guardrail"
        elif market_type == "total_over" and pd.notna(ml_prob) and float(ml_prob) < TOTAL_ML_CONTRADICTION_OVER_MAX_PROB:
            status = "No Play"
            status_reason = (
                f"No Play: ML probability {float(ml_prob):.1%} extremely low — strong model signal against over"
            )
            blocker_stage = "ml_contradiction_guardrail"
        elif not pd.isna(ev) and not pd.isna(edge):

            consensus_agr = str(best.at[idx, "consensus_agreement"]) if "consensus_agreement" in best.columns else "No Kalshi"

            is_side_market = market_type in {"spread_home", "spread_away"}
            is_total_market = "total" in market_type.lower()

            status_metric_basis = "effective"
            effective_ev = ev
            if is_total_market:
                from app_core.weights_config import FALLBACK_HEAVY_TOTAL_EV_MULTIPLIER
                is_fallback_heavy = diagnostics_out.get("is_fallback_heavy", False) if diagnostics_out else False
                if is_fallback_heavy and ev > 0:
                    effective_ev = ev * FALLBACK_HEAVY_TOTAL_EV_MULTIPLIER

            # Apply the same over-probability shrinkage used in production calibration
            # to the gating metrics BEFORE threshold comparison. Without this, the
            # Actionable gate uses the raw inflated calibrated_probability, promoting
            # overconfident ML picks that the shrinkage was designed to penalize.
            if market_type == "total_over":
                _shrink = float(MLB_TOTAL_OVER_PROB_SHRINK) if league == "MLB" else float(TOTAL_OVER_PROB_SHRINK)
                win_prob = 0.5 + _shrink * (win_prob - 0.5)
                if pd.notna(effective_ev) and effective_ev > 0:
                    effective_ev = effective_ev * _shrink
                if pd.notna(effective_edge):
                    effective_edge = effective_edge * _shrink
                effective_win_probability = win_prob
                status_metric_basis = "shrunk"

            # Determine base thresholds with league + market calibration
            req_prob = SIDE_MIN_WIN_PROB if is_side_market else TOTAL_MIN_WIN_PROB
            req_ev = BASELINE_MIN_EV
            req_edge = BASELINE_MIN_EDGE
            base_req_prob = req_prob
            base_req_ev = req_ev
            base_req_edge = req_edge
            no_kalshi_penalty_applied = False
            mlb_spread_penalty_applied = False
            nba_side_bonus_applied = False
            nba_over_bonus_applied = False
            mlb_over_gate_applied = False
            pre_mlb_over_gate_req_prob = req_prob
            pre_mlb_over_gate_req_ev = req_ev
            pre_mlb_over_gate_req_edge = req_edge

            is_fallback_heavy = diagnostics_out.get("is_fallback_heavy", False) if diagnostics_out else False

            # Apply League + Market specific calibration
            if is_side_market:
                if league == "MLB" and "spread" in market_type.lower():
                    # High-EV underdog override: when the market is significantly
                    # mispricing an MLB spread (EV > 0.20, edge > 0.08), lower the
                    # win_prob floor to MLB_SPREAD_HIGH_EV_MIN_WIN_PROB instead of
                    # MLB_SPREAD_MIN_WIN_PROB. The edge/EV signal outweighs raw prob.
                    if (pd.notna(ev) and float(ev) >= float(MLB_SPREAD_HIGH_EV_OVERRIDE_MIN_EV)
                            and pd.notna(edge) and float(edge) >= float(MLB_SPREAD_HIGH_EV_OVERRIDE_MIN_EDGE)):
                        req_prob = float(MLB_SPREAD_HIGH_EV_MIN_WIN_PROB)
                    else:
                        req_prob = MLB_SPREAD_MIN_WIN_PROB
                    req_ev += MLB_SPREAD_ACTIONABLE_PENALTY
                    req_edge += MLB_SPREAD_ACTIONABLE_PENALTY
                    mlb_spread_penalty_applied = True
                elif league == "NBA":
                    req_ev -= NBA_SIDE_ACTIONABLE_BONUS
                    req_edge -= NBA_SIDE_ACTIONABLE_BONUS
                    nba_side_bonus_applied = True
            elif is_total_market:
                from app_core.weights_config import NBA_TOTAL_MIN_WIN_PROB, NHL_TOTAL_MIN_WIN_PROB_STRICT
                if league == "NHL":
                    req_prob = NHL_TOTAL_MIN_WIN_PROB_STRICT
                    req_edge += NHL_TOTAL_EXTRA_EDGE_PENALTY

                if market_type == "total_over":
                    req_ev = max(req_ev, TOTAL_OVER_MIN_EV)
                    req_edge = max(req_edge, TOTAL_OVER_MIN_EDGE)
                elif market_type == "total_under":
                    req_prob = max(req_prob, TOTAL_UNDER_MIN_WIN_PROB)
                    req_ev = max(req_ev, TOTAL_UNDER_MIN_EV)
                    req_edge = max(req_edge, TOTAL_UNDER_MIN_EDGE)

                if league == "NHL":
                    if market_type == "total_over":
                        req_ev += NHL_TOTAL_OVER_ACTIONABLE_PENALTY
                        req_edge += NHL_TOTAL_OVER_ACTIONABLE_PENALTY
                    elif market_type == "total_under":
                        req_ev += NHL_TOTAL_UNDER_ACTIONABLE_PENALTY
                        req_edge += NHL_TOTAL_UNDER_ACTIONABLE_PENALTY
                elif league == "NBA":
                    req_prob = max(req_prob, NBA_TOTAL_MIN_WIN_PROB)
                    if market_type == "total_over":
                        req_ev += NBA_TOTAL_OVER_ACTIONABLE_PENALTY
                        req_edge += NBA_TOTAL_OVER_ACTIONABLE_PENALTY
                        req_ev -= NBA_OVER_ACTIONABLE_BONUS
                        req_edge -= NBA_OVER_ACTIONABLE_BONUS
                        nba_over_bonus_applied = True
                    elif market_type == "total_under":
                        req_ev += NBA_TOTAL_UNDER_ACTIONABLE_PENALTY
                        req_edge += NBA_TOTAL_UNDER_ACTIONABLE_PENALTY
                elif league == "MLB":
                    if market_type == "total_over":
                        req_ev += MLB_TOTAL_OVER_ACTIONABLE_PENALTY
                        req_edge += MLB_TOTAL_OVER_ACTIONABLE_PENALTY
                    elif market_type == "total_under":
                        req_prob = max(req_prob, MLB_TOTAL_UNDER_MIN_WIN_PROB)
                        req_ev += MLB_TOTAL_UNDER_ACTIONABLE_PENALTY
                        req_edge += MLB_TOTAL_UNDER_ACTIONABLE_PENALTY

                if is_fallback_heavy:
                    req_ev += FALLBACK_HEAVY_TOTAL_EXTRA_PENALTY
                    req_edge += FALLBACK_HEAVY_TOTAL_EXTRA_PENALTY

                if consensus_agr == "No Kalshi":
                    req_ev += NO_KALSHI_TOTAL_EXTRA_PENALTY
                    req_edge += NO_KALSHI_TOTAL_EXTRA_PENALTY
                    no_kalshi_penalty_applied = True
                    if market_type == "total_under":
                        req_ev += NO_KALSHI_TOTAL_UNDER_EXTRA_PENALTY
                        req_edge += NO_KALSHI_TOTAL_UNDER_EXTRA_PENALTY

            # Static empirical penalty hook (league + family), intentionally interpretable
            # and ready for later recap-driven replacement.
            family_key = "side"
            if market_type == "total_over":
                family_key = "over"
            elif market_type == "total_under":
                family_key = "under"
            empirical_penalty = float(LEAGUE_MARKET_FAMILY_ACTIONABLE_PENALTIES.get((league, family_key), 0.0))
            req_ev += empirical_penalty
            req_edge += empirical_penalty

            if league == "MLB" and market_type == "total_over":
                pre_mlb_over_gate_req_prob = req_prob
                pre_mlb_over_gate_req_ev = req_ev
                pre_mlb_over_gate_req_edge = req_edge
                # Agrees overs in a PROVEN bucket earn the relaxed bar; everything
                # else keeps the strict 0.65/0.07/0.04 gate. See weights_config.
                if consensus_agr == "Agrees" and _mlb_over_agrees_relax_ok:
                    _over_min_prob = MLB_OVER_AGREES_ACTIONABLE_MIN_PROB
                    _over_min_ev = MLB_OVER_AGREES_ACTIONABLE_MIN_EV
                    _over_min_edge = MLB_OVER_AGREES_ACTIONABLE_MIN_EDGE
                else:
                    _over_min_prob = MLB_OVER_ACTIONABLE_MIN_PROB
                    _over_min_ev = MLB_OVER_ACTIONABLE_MIN_EV
                    _over_min_edge = MLB_OVER_ACTIONABLE_MIN_EDGE
                req_prob = max(req_prob, _over_min_prob)
                req_ev = max(req_ev, _over_min_ev)
                req_edge = max(req_edge, _over_min_edge)
                # High total line penalty: very high lines (≥11.0) have repeatedly
                # underperformed (COL/ARI Over 11.5 lost on both May-15 and May-16).
                # Mid-range penalty: lines in [9.5, 11.0) also underperform
                # (ARI/COL Over 9.5 went 3 total runs at Actionable on May-21).
                if "total_line" in best.columns:
                    _tl = pd.to_numeric(best.at[idx, "total_line"], errors="coerce")
                    if pd.notna(_tl) and float(_tl) >= MLB_HIGH_TOTAL_LINE_THRESHOLD:
                        req_ev += MLB_HIGH_TOTAL_LINE_OVER_PENALTY
                        req_edge += MLB_HIGH_TOTAL_LINE_OVER_PENALTY
                    elif pd.notna(_tl) and float(_tl) >= MLB_MID_TOTAL_LINE_THRESHOLD:
                        req_ev += MLB_MID_TOTAL_LINE_OVER_PENALTY
                        req_edge += MLB_MID_TOTAL_LINE_OVER_PENALTY
                mlb_over_gate_applied = (
                    req_prob > pre_mlb_over_gate_req_prob
                    or req_ev > pre_mlb_over_gate_req_ev
                    or req_edge > pre_mlb_over_gate_req_edge
                )

            base_pass = (win_prob >= base_req_prob) and (effective_ev >= base_req_ev) and (edge >= base_req_edge)
            final_pass = (win_prob >= req_prob) and (effective_ev >= req_ev) and (edge >= req_edge)
            without_mlb_spread_penalty_pass = final_pass
            without_nba_side_bonus_pass = final_pass
            without_nba_over_bonus_pass = final_pass
            without_mlb_over_gate_pass = final_pass
            if mlb_spread_penalty_applied:
                without_mlb_spread_penalty_pass = (
                    (win_prob >= req_prob)
                    and (effective_ev >= (req_ev - MLB_SPREAD_ACTIONABLE_PENALTY))
                    and (edge >= (req_edge - MLB_SPREAD_ACTIONABLE_PENALTY))
                )
            if nba_side_bonus_applied:
                without_nba_side_bonus_pass = (
                    (win_prob >= req_prob)
                    and (effective_ev >= (req_ev + NBA_SIDE_ACTIONABLE_BONUS))
                    and (edge >= (req_edge + NBA_SIDE_ACTIONABLE_BONUS))
                )
            if nba_over_bonus_applied:
                without_nba_over_bonus_pass = (
                    (win_prob >= req_prob)
                    and (effective_ev >= (req_ev + NBA_OVER_ACTIONABLE_BONUS))
                    and (edge >= (req_edge + NBA_OVER_ACTIONABLE_BONUS))
                )
            if mlb_over_gate_applied:
                without_mlb_over_gate_pass = (
                    (win_prob >= pre_mlb_over_gate_req_prob)
                    and (effective_ev >= pre_mlb_over_gate_req_ev)
                    and (edge >= pre_mlb_over_gate_req_edge)
                )

            blocked_by_mlb_spread_on_metrics = mlb_spread_penalty_applied and without_mlb_spread_penalty_pass and not final_pass
            blocked_by_mlb_over_gate_on_thresholds = mlb_over_gate_applied and without_mlb_over_gate_pass and not final_pass

            if base_pass and not final_pass:
                if market_type == "total_under":
                    blocked_by_under_specific_thresholds += 1
                if league == "NBA" and is_total_market:
                    blocked_by_nba_total_penalty += 1
                if no_kalshi_penalty_applied and is_total_market:
                    blocked_by_no_kalshi_total_penalty += 1
            if blocked_by_mlb_spread_on_metrics:
                blocked_by_mlb_spread_penalty += 1
            if blocked_by_mlb_over_gate_on_thresholds:
                blocked_by_mlb_over_promotion_gate += 1
            if nba_side_bonus_applied and final_pass and not without_nba_side_bonus_pass:
                promoted_by_nba_side_bonus += 1
            if nba_over_bonus_applied and final_pass and not without_nba_over_bonus_pass:
                promoted_by_nba_over_bonus += 1

            if is_side_market and win_prob < req_prob:
                status = "Below Threshold"
                status_reason = f"Fails side minimum Win Probability ({req_prob*100:.1f}%)"
                blocker_stage = "actionable_threshold"
            elif is_total_market and win_prob < req_prob:
                status = "Below Threshold"
                if blocked_by_mlb_over_gate_on_thresholds:
                    status_reason = f"Fails MLB over actionable gate (Prob >= {req_prob*100:.1f}%, Edge >= {req_edge*100:.1f}%, Effective EV >= {req_ev*100:.1f}%)"
                else:
                    status_reason = f"Fails minimum Win Probability for {'NHL ' if league == 'NHL' else 'NBA ' if league == 'NBA' else ''}Totals ({req_prob*100:.1f}%)"
                blocker_stage = "actionable_threshold"
            else:
                if effective_ev < req_ev or edge < req_edge:
                    status = "Below Threshold"
                    blocked_by_fallback_heavy_guardrail = False
                    if blocked_by_mlb_spread_on_metrics:
                        status_reason = f"Fails MLB spread actionable penalty (Edge >= {req_edge*100:.1f}%, Effective EV >= {req_ev*100:.1f}%)"
                    elif blocked_by_mlb_over_gate_on_thresholds:
                        status_reason = f"Fails MLB over actionable gate (Prob >= {req_prob*100:.1f}%, Edge >= {req_edge*100:.1f}%, Effective EV >= {req_ev*100:.1f}%)"
                    elif is_total_market and is_fallback_heavy and ((effective_ev < req_ev and effective_ev >= req_ev - FALLBACK_HEAVY_TOTAL_EXTRA_PENALTY) or (edge < req_edge and edge >= req_edge - FALLBACK_HEAVY_TOTAL_EXTRA_PENALTY)):
                        status_reason = f"Fails due to fallback-heavy totals penalty (Edge >= {req_edge*100:.1f}%, Effective EV >= {req_ev*100:.1f}%)"
                        blocked_by_fallback_heavy_guardrail = True
                    elif is_total_market and market_type == "total_under":
                        status_reason = f"Fails stricter total_under cold-market penalty (Edge >= {req_edge*100:.1f}%, Effective EV >= {req_ev*100:.1f}%)"
                    elif is_total_market and league == "NHL":
                        status_reason = f"Fails NHL total cold-market penalty threshold (Edge >= {req_edge*100:.1f}%, Effective EV >= {req_ev*100:.1f}%)"
                    elif is_total_market and market_type == "total_over":
                        status_reason = f"Fails stricter total_over cold-market penalty (Edge >= {req_edge*100:.1f}%, Effective EV >= {req_ev*100:.1f}%)"
                    else:
                        status_reason = f"Fails minimum Edge ({req_edge*100:.1f}%) or Effective EV ({req_ev*100:.1f}%) thresholds"
                    blocker_stage = "fallback_heavy_guardrail" if blocked_by_fallback_heavy_guardrail else "actionable_threshold"
                else:
                    status = "Actionable"
                    status_reason = "Passed all strict filters"
                    if is_spread_divergence_override:
                        status_reason = "Passed all strict filters (Spread divergence override applied)"

                    if high_ev_guardrail:
                        high_ev_uncertainty_reasons: list[str] = []
                        if is_total_market and is_fallback_heavy:
                            high_ev_uncertainty_reasons.append("fallback-heavy guardrail")
                        if is_total_market and consensus_agr == "No Kalshi":
                            high_ev_uncertainty_reasons.append("No Kalshi")
                        if degraded_subset_flag:
                            high_ev_uncertainty_reasons.append("degraded feature subset")
                        if suspicious_data_flag:
                            high_ev_uncertainty_reasons.append("suspicious data")

                        if high_ev_uncertainty_reasons:
                            status = "High Variance/Speculative"
                            reason_label = ", ".join(dict.fromkeys(high_ev_uncertainty_reasons))
                            status_reason = f"High Variance: strong EV but capped due to {reason_label}"
                            blocker_stage = "variance_uncertainty_guardrail"
                            if "No Kalshi" in high_ev_uncertainty_reasons:
                                high_variance_capped_due_to_no_kalshi += 1
                            if "fallback-heavy guardrail" in high_ev_uncertainty_reasons:
                                high_variance_capped_due_to_fallback_heavy += 1
                            if "degraded feature subset" in high_ev_uncertainty_reasons:
                                high_variance_capped_due_to_degraded_subset += 1
                            if "suspicious data" in high_ev_uncertainty_reasons:
                                high_variance_capped_due_to_suspicious_data += 1
                        else:
                            promoted_high_ev_to_actionable_no_uncertainty += 1
                            status_reason = "Actionable: strong EV/edge and passed all market/league filters"

            # No Kalshi total guard: without prediction-market validation for
            # NBA/NHL totals (thin liquidity sports + volatile lines), cap at High Variance
            # rather than Actionable or Below Threshold.
            if is_total_market and league == "NBA" and consensus_agr == "No Kalshi":
                if status in ("Actionable", "Below Threshold"):
                    status = "High Variance/Speculative"
                    status_reason = "High Variance: NBA total without Kalshi market validation"
                    blocker_stage = "no_kalshi_nba_guardrail"

            if is_total_market and league == "NHL" and consensus_agr == "No Kalshi":
                if status in ("Actionable", "Below Threshold"):
                    status = "High Variance/Speculative"
                    status_reason = "High Variance: NHL total without Kalshi market validation"
                    blocker_stage = "no_kalshi_nhl_guardrail"

            # MLB Under consensus gate — replaces the blanket Actionable cap (removed May-28).
            # Cap was set after May 16-17 (0-4), before TheOver conflict penalty and
            # double-shrink fix. Unders have since outperformed overs (May 22-27 data).
            # Now: only "Agrees" consensus MLB unders can be Actionable; Neutral/Disagrees
            # are capped at High Variance — the same directional signal that predicts wins.
            if league == "MLB" and market_type == "total_under" and status == "Actionable":
                if consensus_agr not in ("Agrees",):
                    status = "High Variance/Speculative"
                    status_reason = (
                        f"High Variance: MLB under consensus '{consensus_agr}' — "
                        f"only 'Agrees' unders qualify for Actionable"
                    )
                    blocker_stage = "mlb_under_consensus_gate"

            # NHL Under Actionable cap — CAR/MTL Under 5.5 went 8 total goals at Actionable
            # on May 21. Same model overconfidence pattern as MLB unders. Cap at High Variance.
            if NHL_UNDER_ACTIONABLE_CAP and league == "NHL" and market_type == "total_under":
                if status == "Actionable":
                    status = "High Variance/Speculative"
                    status_reason = "High Variance: NHL under capped (CAR/MTL Under 5.5 went 8 goals May 21; model overconfident on NHL unders)"
                    blocker_stage = "nhl_under_actionable_cap"

            # MLB total HV floor — May 27: HV/Spec MLB totals 0-6 (BT was 6-2).
            # Require effective_win_probability >= MLB_TOTAL_HV_MIN_WIN_PROB (0.62) for HV/Spec.
            # Picks below this floor become Below Threshold — still visible at minimal sizing.
            if league == "MLB" and is_total_market and status == "High Variance/Speculative":
                if effective_win_probability < MLB_TOTAL_HV_MIN_WIN_PROB:
                    status = "Below Threshold"
                    status_reason = (
                        f"Below Threshold: MLB total effective win prob {effective_win_probability:.1%} "
                        f"below HV floor ({MLB_TOTAL_HV_MIN_WIN_PROB:.0%}); May 27 HV MLB totals 0-6"
                    )
                    blocker_stage = "mlb_total_hv_floor"

            # MLB Under Kalshi direction cap — Kalshi probability is pre-oriented to the pick
            # side (P(Under) for Under rows). When P(Under) < 0.50, Kalshi is saying the Over
            # is more likely than the Under. Historical record: 0-4 on May 31, recurring on
            # May 27-28. Cap these picks at Below Threshold (visible at minimal Kelly sizing)
            # regardless of model/TheOver confidence; Kalshi's direction signal is the most
            # reliable single indicator for MLB totals.
            if (league == "MLB" and market_type == "total_under"
                    and status in ("Actionable", "High Variance/Speculative")
                    and pd.notna(kalshi_prob) and float(kalshi_prob) < 0.50):
                status = "Below Threshold"
                status_reason = (
                    f"Below Threshold: MLB Under — Kalshi P(Under)={float(kalshi_prob):.1%} < 50%; "
                    f"Kalshi prices the Over as more likely (0-4 pattern on such picks)"
                )
                blocker_stage = "mlb_under_kalshi_over_cap"

            # Low total line cap — MLB overs with a line below 8.0 are set on
            # pitcher-friendly matchups where low-scoring shutouts are common.
            # May 20: CHC/MIL Over 6.5 (5 total) and SD/LAD Over 7.5 (4 total) both lost.
            # Surface as High Variance rather than hiding entirely so the pick is visible.
            if league == "MLB" and market_type == "total_over" and "total_line" in best.columns:
                _tl_low = pd.to_numeric(best.at[idx, "total_line"], errors="coerce")
                if pd.notna(_tl_low) and float(_tl_low) < MLB_OVER_MIN_TOTAL_LINE:
                    # Conditioned escape hatch: a strong, Kalshi-aligned sub-8.0 over
                    # keeps its Actionable status instead of being force-demoted.
                    # Backtest-justified and provably excludes the documented losers
                    # (see app_core/low_line_override + scripts/backtest_low_line_over.py).
                    from app_core.low_line_override import low_line_over_override_applies
                    _keep_actionable = low_line_over_override_applies(
                        league=league,
                        market_type=market_type,
                        total_line=float(_tl_low),
                        status=status,
                        consensus=consensus_agr,
                        effective_win_probability=effective_win_probability,
                        effective_edge=effective_edge,
                    )
                    if _keep_actionable:
                        status_reason = (
                            f"Actionable: sub-{MLB_OVER_MIN_TOTAL_LINE} over override — "
                            f"Agrees consensus, effective win {effective_win_probability:.1%}, "
                            f"edge {effective_edge:.1%} (backtest-supported carve-out)"
                        )
                    elif status in ("Actionable", "Below Threshold"):
                        # Consensus-aware low-line demotion. Sub-8.0 overs are NOT a
                        # uniformly weak bucket — graded slates (20 May-5 Jun, n=51) show
                        # Agrees 72.7% and Disagrees 60.0%, but Neutral only 45.0%. The
                        # blanket rule surfaced ALL of them at High Variance (0.075x Kelly),
                        # handing the 45% Neutral bucket the same elevated stake as the
                        # profitable ones. Keep Disagrees/Agrees at High Variance; hold the
                        # weak Neutral low-line overs at Below Threshold instead.
                        if (consensus_agr or "").strip() == "Neutral":
                            status = "Below Threshold"
                            status_reason = (
                                f"Below Threshold: MLB over line {float(_tl_low)} below {MLB_OVER_MIN_TOTAL_LINE} "
                                f"with Neutral consensus — low-line Neutral overs hit ~45% (backtest), "
                                f"held below High Variance"
                            )
                        else:
                            status = "High Variance/Speculative"
                            status_reason = (
                                f"High Variance: MLB over line {float(_tl_low)} below {MLB_OVER_MIN_TOTAL_LINE} "
                                f"— pitcher-friendly game, low-line overs underperform"
                            )
                        blocker_stage = "low_line_over_guardrail"

            # ── Edge-based no-stake gates (graded MLB totals, 20 May-7 Jun, n=182) ──
            # Two buckets bled below the -110 breakeven (52.4%). Hold them out of the
            # production card (Below Threshold = visible, unstaked). Pure helper
            # _edge_no_stake_demotion is unit-tested in tests/test_edge_no_stake_gates.py;
            # see scripts/edge_by_bucket.py for the data.
            _ns_line = pd.to_numeric(best.at[idx, "total_line"], errors="coerce") if "total_line" in best.columns else None
            _ns_status, _ns_reason, _ns_blocker = _edge_no_stake_demotion(
                league=league,
                market_type=market_type,
                consensus=consensus_agr,
                total_line=float(_ns_line) if _ns_line is not None and pd.notna(_ns_line) else None,
                status=status,
                neutral_no_stake=MLB_TOTAL_NEUTRAL_NO_STAKE,
                mid_line_no_stake=MLB_OVER_MID_LINE_NO_STAKE,
                mid_min=float(MLB_OVER_MIN_TOTAL_LINE),
                mid_max=float(MLB_MID_TOTAL_LINE_THRESHOLD),
            )
            if _ns_status is not None:
                status, status_reason, blocker_stage = _ns_status, _ns_reason, _ns_blocker

            # Apply Consensus Overlay Logic
            # STRICT profile: full overlay on all market types.
            # STANDARD profile: apply the Disagrees overlay to side/spread bets only.
            # Spread markets are liquid enough that Kalshi divergence is a reliable
            # signal — if Kalshi is significantly more bullish than the model on a
            # spread and the model's own confidence is still below the Disagrees floor,
            # the pick is genuinely contested and should be flagged High Variance.
            if status == "Actionable" and BEST_PICKS_PROFILE == 'STRICT':
                if consensus_agr == "Neutral":
                    if win_prob < NEUTRAL_ACTIONABLE_MIN_PROB or effective_ev < NEUTRAL_ACTIONABLE_MIN_EV or edge < NEUTRAL_ACTIONABLE_MIN_EDGE:
                        status = "Below Threshold"
                        status_reason = f"Fails stricter Neutral overlay (Prob >= {NEUTRAL_ACTIONABLE_MIN_PROB}, EV >= {NEUTRAL_ACTIONABLE_MIN_EV}, Edge >= {NEUTRAL_ACTIONABLE_MIN_EDGE})"
                        blocker_stage = "consensus_overlay"
                elif consensus_agr == "Disagrees":
                    if win_prob < DISAGREES_ACTIONABLE_MIN_PROB or effective_ev < DISAGREES_ACTIONABLE_MIN_EV or edge < DISAGREES_ACTIONABLE_MIN_EDGE:
                        status = "High Variance/Speculative"
                        status_reason = f"Fails stricter Disagrees overlay (Prob >= {DISAGREES_ACTIONABLE_MIN_PROB}, EV >= {DISAGREES_ACTIONABLE_MIN_EV}, Edge >= {DISAGREES_ACTIONABLE_MIN_EDGE})"
                        blocker_stage = "consensus_overlay"
            elif status == "Actionable" and consensus_agr == "Disagrees" and is_side_market:
                if win_prob < DISAGREES_ACTIONABLE_MIN_PROB or effective_ev < DISAGREES_ACTIONABLE_MIN_EV or edge < DISAGREES_ACTIONABLE_MIN_EDGE:
                    status = "High Variance/Speculative"
                    status_reason = (
                        f"High Variance: Kalshi disagrees on spread — model win prob {win_prob:.1%} below "
                        f"Disagrees floor ({DISAGREES_ACTIONABLE_MIN_PROB:.0%}) or EV/edge insufficient"
                    )
                    blocker_stage = "consensus_overlay_spread"
        else:
            status = "Actionable"
            status_reason = "Passed all strict filters"

        best.at[idx, "Pick_Status"] = status
        best.at[idx, "Status_Reason"] = status_reason
        best.at[idx, "suspicious_data_flag"] = suspicious_data_flag
        best.at[idx, "suspicious_data_reasons"] = suspicious_reasons_str
        best.at[idx, "status_metric_basis"] = status_metric_basis
        best.at[idx, "effective_expected_value"] = effective_ev
        best.at[idx, "effective_edge"] = effective_edge
        best.at[idx, "effective_win_probability"] = effective_win_probability
        best.at[idx, "status_blocker_reason"] = status_reason if status != "Actionable" else ""
        best.at[idx, "status_blocker_stage"] = blocker_stage if status != "Actionable" else "none"

    from app_core.weights_config import SIDE_MIN_WIN_PROB

    # Conservative anti-monoculture guard: only if Actionable is totals-only,
    # and a side is near the weakest existing Actionable row on effective metrics.
    actionable_mask = best["Pick_Status"].astype(str).eq("Actionable")
    actionable_family_counts: dict[str, int] = {}
    totals_only_actionable_flag = False
    viable_side_candidates_count = 0
    side_balance_guard_reason = "Balance guard not evaluated"
    if actionable_mask.any():
        market_type_str = best["market_type"].astype(str).str.lower()
        actionable_side_mask = actionable_mask & market_type_str.str.contains("spread", na=False)
        actionable_total_mask = actionable_mask & market_type_str.str.contains("total", na=False)
        actionable_family_counts = {
            "total": int(actionable_total_mask.sum()),
            "side": int(actionable_side_mask.sum()),
        }
        totals_only_actionable_flag = bool(actionable_total_mask.any() and not actionable_side_mask.any())

        if totals_only_actionable_flag:
            actionable_ev = pd.to_numeric(best.loc[actionable_mask, "effective_expected_value"], errors="coerce")
            actionable_edge = pd.to_numeric(best.loc[actionable_mask, "effective_edge"], errors="coerce")
            actionable_prob = pd.to_numeric(best.loc[actionable_mask, "effective_win_probability"], errors="coerce")
            weakest_actionable_ev = actionable_ev.min(skipna=True)
            weakest_actionable_edge = actionable_edge.min(skipna=True)
            weakest_actionable_prob = actionable_prob.min(skipna=True)

            ev_margin = 0.030
            edge_margin = 0.025
            prob_margin = 0.080

            blocked_stages = {
                "suspicious_data_guardrail",
                "data_fallback_guardrail",
                "line_integrity_guardrail",
                "fallback_heavy_guardrail",
            }

            side_candidate_mask = (
                best["Pick_Status"].astype(str).isin({"High Variance/Speculative", "Below Threshold"})
                & market_type_str.str.contains("spread", na=False)
                & pd.to_numeric(best["effective_expected_value"], errors="coerce").gt(0)
                & pd.to_numeric(best["effective_edge"], errors="coerce").gt(0)
                & pd.to_numeric(best["effective_win_probability"], errors="coerce").ge(SIDE_MIN_WIN_PROB)
                & ~best.get("suspicious_data_flag", pd.Series(False, index=best.index)).fillna(False).astype(bool)
                & ~best.get("status_blocker_stage", pd.Series("", index=best.index)).astype(str).isin(blocked_stages)
                & ~best.get("status_blocker_reason", pd.Series("", index=best.index)).astype(str).str.contains("degraded|fallback-only", case=False, na=False)
                & pd.to_numeric(best["effective_expected_value"], errors="coerce").ge(weakest_actionable_ev - ev_margin)
                & pd.to_numeric(best["effective_edge"], errors="coerce").ge(weakest_actionable_edge - edge_margin)
                & pd.to_numeric(best["effective_win_probability"], errors="coerce").ge(weakest_actionable_prob - prob_margin)
            )
            viable_side_candidates_count = int(side_candidate_mask.sum())

            if viable_side_candidates_count > 0:
                promote_idx = (
                    best.loc[side_candidate_mask]
                    .sort_values(
                        by=["effective_expected_value", "effective_edge", "effective_win_probability"],
                        ascending=[False, False, False],
                    )
                    .index[0]
                )
                best.at[promote_idx, "Pick_Status"] = "Actionable"
                best.at[promote_idx, "Status_Reason"] = (
                    "Actionable: side-balance guard promoted strongest near-threshold side"
                )
                best.at[promote_idx, "status_blocker_reason"] = ""
                best.at[promote_idx, "status_blocker_stage"] = "none"
                side_balance_promotions += 1
                side_balance_guard_reason = "Promoted strongest viable side to avoid totals-only Actionable card"
            else:
                side_balance_guard_reason = "No viable side candidates within margin"
        else:
            side_balance_guard_reason = "Actionable card already contains side and total families"
    else:
        side_balance_guard_reason = "No Actionable rows available for balance guard"

    # Always keep transparency fields row-populated in export.
    best["status_metric_basis"] = best["status_metric_basis"].fillna("raw")
    best["effective_expected_value"] = pd.to_numeric(best["effective_expected_value"], errors="coerce").fillna(
        pd.to_numeric(best.get("expected_value"), errors="coerce")
    )
    best["effective_edge"] = pd.to_numeric(best["effective_edge"], errors="coerce").fillna(
        pd.to_numeric(best.get("edge"), errors="coerce")
    )
    best["effective_win_probability"] = pd.to_numeric(best["effective_win_probability"], errors="coerce").fillna(
        pd.to_numeric(best.get("calibrated_probability"), errors="coerce")
    )
    best["status_blocker_reason"] = best["status_blocker_reason"].fillna("").astype(str)
    best["status_blocker_stage"] = best["status_blocker_stage"].fillna("none").astype(str)

    # Legacy logging/metrics variables for reference
    valid_edge_mask = best["edge"] >= 0.01
    valid_ev_mask = best["expected_value"] >= 0.005

    logger.info(f"BEST PICKS AUDIT: Non-qualifying picks (edge < 0.01 or EV < 0.005) left intact: {(~valid_edge_mask | ~valid_ev_mask).sum()}")

    total_games = int(pool["matchup_id"].nunique(dropna=False))
    logger.info(f"PIPELINE AUDIT: [9/9] Rows surviving into best-picks ranking/export: {len(best)}")
    if len(best) != total_games:
        logger.warning(
            "Best-pick validation mismatch: selected_rows=%s total_games=%s",
            len(best),
            total_games,
        )

    best["calibrated_probability"] = _numeric_series(best, "calibrated_probability", 0.5)
    edge_for_consensus = _numeric_series(best, "edge", 0.0)

    # Sort Phase: Use ordered categorical logic for exact ordering.
    status_order = [
        "Actionable",
        "High Variance/Speculative",
        "Below Threshold",
        "Fallback / Low Confidence",
        "No Play",
        "Missing Line"
    ]
    if "Pick_Status" in best.columns:
        best["Pick_Status"] = best["Pick_Status"].astype(str).str.strip()
        best["Pick_Status"] = pd.Categorical(best["Pick_Status"], categories=status_order, ordered=True)
        best["Pick_Status"] = best["Pick_Status"].fillna("No Play")

    if not best.empty:
        best["expected_value"] = pd.to_numeric(best["expected_value"], errors="coerce")
        best["edge"] = pd.to_numeric(best["edge"], errors="coerce")

        # 1. Final ranking pass: Instead of sequential 1-N globally, we calculate the ranking per bucket
        bucket_dfs = []
        for status in status_order:
            bucket_df = best[best["Pick_Status"] == status].copy()
            if not bucket_df.empty:
                bucket_df = _apply_triple_filter_ranking(bucket_df)
                # Ensure rank is local per bucket 1..N based on tier_score/ev
                bucket_df["Triple_Filter_Rank"] = range(1, len(bucket_df) + 1)
                bucket_dfs.append(bucket_df)

        if bucket_dfs:
            best = pd.concat(bucket_dfs, ignore_index=True)

        # We MUST re-apply the categorical type to Pick_Status in case _apply_triple_filter_ranking lost it
        best["Pick_Status"] = pd.Categorical(best["Pick_Status"], categories=status_order, ordered=True)

        # Sort by:
        # 1) Pick_Status (Categorical ascending so 'Actionable' is first)
        # 2) Triple_Filter_Rank (ascending so 1 is first)
        # 3) expected_value (descending as tie-breaker)
        # 4) edge (descending as tie-breaker)

        # Cast to numeric so that sort logic is proper, missing goes to NaN
        best["_rank_sort"] = pd.to_numeric(best["Triple_Filter_Rank"], errors="coerce")
        best["_ev_sort"] = pd.to_numeric(best["expected_value"], errors="coerce")
        best["_edge_sort"] = pd.to_numeric(best["edge"], errors="coerce")

        best = best.sort_values(
            by=["Pick_Status", "_rank_sort", "_ev_sort", "_edge_sort"],
            ascending=[True, True, False, False],
            na_position="last"
        ).reset_index(drop=True)

        best = best.drop(columns=["_rank_sort", "_ev_sort", "_edge_sort"], errors="ignore")

    # Stamp the running code version onto every pick so the export is self-identifying.
    best["pipeline_build"] = PIPELINE_BUILD

    for col in BEST_PICK_COLUMNS:
        if col not in best.columns:
            best[col] = pd.NA

    # Final Cleanup: Drop temporary columns used for processing
    best = best.drop(columns=["tier_score", "expected_value_sort", "is_unique", "is_kalshi_available", "_status_sort"], errors="ignore")

    # Final guaranteed sort pass immediately before export
    if not best.empty:
        status_order = [
            "Actionable",
            "High Variance/Speculative",
            "Below Threshold",
            "Fallback / Low Confidence",
            "No Play",
            "Missing Line"
        ]
        best["Pick_Status"] = best["Pick_Status"].astype(str).str.strip()
        best["Pick_Status"] = pd.Categorical(best["Pick_Status"], categories=status_order, ordered=True)
        best["Pick_Status"] = best["Pick_Status"].fillna("No Play")
        best["_rank_sort"] = pd.to_numeric(best["Triple_Filter_Rank"], errors="coerce")
        best["_ev_sort"] = pd.to_numeric(best["expected_value"], errors="coerce")
        best["_edge_sort"] = pd.to_numeric(best["edge"], errors="coerce")

        best = best.sort_values(
            by=["Pick_Status", "_rank_sort", "_ev_sort", "_edge_sort"],
            ascending=[True, True, False, False],
            na_position="last"
        ).reset_index(drop=True)

        best = best.drop(columns=["_rank_sort", "_ev_sort", "_edge_sort"], errors="ignore")

    # 3. Assign parlay_rank AFTER the final sort pass so the exported numbers sequentially map 1 to N
    if not best.empty:
        final_actionable_mask = best["Pick_Status"].astype(str).eq("Actionable")
        market_type_str = best["market_type"].astype(str).str.lower()
        final_actionable_side_mask = final_actionable_mask & market_type_str.str.contains("spread", na=False)
        final_actionable_total_mask = final_actionable_mask & market_type_str.str.contains("total", na=False)
        if final_actionable_total_mask.any() and not final_actionable_side_mask.any():
            weakest_ev = pd.to_numeric(best.loc[final_actionable_mask, "effective_expected_value"], errors="coerce").min(skipna=True)
            weakest_edge = pd.to_numeric(best.loc[final_actionable_mask, "effective_edge"], errors="coerce").min(skipna=True)
            weakest_prob = pd.to_numeric(best.loc[final_actionable_mask, "effective_win_probability"], errors="coerce").min(skipna=True)
            post_rank_candidate_mask = (
                best["Pick_Status"].astype(str).isin({"High Variance/Speculative", "Below Threshold"})
                & market_type_str.str.contains("spread", na=False)
                & pd.to_numeric(best["effective_expected_value"], errors="coerce").gt(0)
                & pd.to_numeric(best["effective_edge"], errors="coerce").gt(0)
                & pd.to_numeric(best["effective_win_probability"], errors="coerce").ge(SIDE_MIN_WIN_PROB)
                & ~best.get("suspicious_data_flag", pd.Series(False, index=best.index)).fillna(False).astype(bool)
                & ~best.get("status_blocker_reason", pd.Series("", index=best.index)).astype(str).str.contains("degraded|fallback-only", case=False, na=False)
                & pd.to_numeric(best["effective_expected_value"], errors="coerce").ge(weakest_ev - 0.01)
                & pd.to_numeric(best["effective_edge"], errors="coerce").ge(weakest_edge - 0.01)
                & pd.to_numeric(best["effective_win_probability"], errors="coerce").ge(weakest_prob - 0.05)
            )
            if post_rank_candidate_mask.any():
                promote_idx = (
                    best.loc[post_rank_candidate_mask]
                    .sort_values(by=["effective_expected_value", "effective_edge", "effective_win_probability"], ascending=[False, False, False])
                    .index[0]
                )
                best.at[promote_idx, "Pick_Status"] = "Actionable"
                best.at[promote_idx, "Status_Reason"] = "Actionable: post-rank side-balance guard promoted strongest viable side"
                best.at[promote_idx, "status_blocker_reason"] = ""
                best.at[promote_idx, "status_blocker_stage"] = "none"
                side_balance_promotions += 1
                side_balance_guard_reason = "Promoted strongest viable side to avoid totals-only Actionable card (post-rank)"

        # Recompute final actionable family diagnostics after last promotion stage.
        final_actionable_mask = best["Pick_Status"].astype(str).eq("Actionable")
        final_market_type_str = best["market_type"].astype(str).str.lower()
        final_actionable_side_mask = final_actionable_mask & final_market_type_str.str.contains("spread", na=False)
        final_actionable_total_mask = final_actionable_mask & final_market_type_str.str.contains("total", na=False)
        actionable_family_counts = {
            "total": int(final_actionable_total_mask.sum()),
            "side": int(final_actionable_side_mask.sum()),
        }
        totals_only_actionable_flag = bool(final_actionable_total_mask.any() and not final_actionable_side_mask.any())

        # Ensure row-level export transparency fields are populated with final diagnostics.
        best["actionable_family_counts"] = str(actionable_family_counts)
        best["totals_only_actionable_flag"] = bool(totals_only_actionable_flag)
        best["viable_side_candidates_count"] = int(viable_side_candidates_count)
        best["side_promoted_by_balance_guard_count"] = int(side_balance_promotions)
        best["side_balance_guard_reason"] = str(side_balance_guard_reason)

    best["parlay_rank"] = range(1, len(best) + 1) if not best.empty else pd.Series(dtype=int)

    # Final Validation Logs (Lightweight terminal/application logging)
    if not best.empty:
        logger.info("--- FINAL PIPELINE VALIDATION AUDIT ---")

        # Pick Status Counts
        if "Pick_Status" in best.columns:
            status_counts = best["Pick_Status"].value_counts(dropna=False).to_dict()
            logger.info("--- FINAL STATUS COUNTS ---")
            for status in status_order:
                count = status_counts.get(status, 0)
                if count > 0:
                    logger.info(f"Validated branch: {status} ({count} rows)")
                else:
                    logger.info(f"Branch not exercised: {status} (0 rows)")

        # TheOver coverage audit — the conflict penalty and 0.25 blend weight both
        # depend on theover_probability being populated for MLB totals. If it is NaN,
        # the direction-correction safety net is silently disabled (it defaults to a
        # neutral 0.5 vote and the conflict penalty never fires). Surface the coverage.
        if "market_type" in best.columns and "league" in best.columns:
            _mlb_total_mask = (
                best["league"].astype(str).str.upper().eq("MLB")
                & best["market_type"].astype(str).str.lower().str.contains("total", na=False)
            )
            _mlb_total_count = int(_mlb_total_mask.sum())
            if "theover_probability" in best.columns:
                _to_numeric = pd.to_numeric(best.loc[_mlb_total_mask, "theover_probability"], errors="coerce")
                _theover_populated = int(_to_numeric.notna().sum())
                _theover_nan = _mlb_total_count - _theover_populated
            else:
                _theover_populated = 0
                _theover_nan = _mlb_total_count
            logger.info(
                f"--- THEOVER COVERAGE (MLB totals) --- populated={_theover_populated} "
                f"NaN={_theover_nan} of {_mlb_total_count} rows"
            )
            if _mlb_total_count > 0 and _theover_populated == 0:
                logger.warning(
                    "THEOVER SIGNAL MISSING: 0 of %d MLB total rows have theover_probability. "
                    "Conflict penalty disabled; direction relies on Kalshi/Market/ML only.",
                    _mlb_total_count,
                )
            if diagnostics_out is not None:
                diagnostics_out["mlb_total_theover_populated_count"] = _theover_populated
                diagnostics_out["mlb_total_theover_nan_count"] = _theover_nan
                diagnostics_out["mlb_total_row_count"] = _mlb_total_count

        # Odds Source Counts
        if "odds_source" in best.columns:
            logger.info("--- FINAL ODDS SOURCE CLASSIFICATION ---")

            # Show all sources
            sources = best["odds_source"].dropna().unique()
            for source in sources:
                 count = (best["odds_source"] == source).sum()
                 logger.info(f"odds_source '{source}': {count} total rows")

            # Cross-tabs
            if "Pick_Status" in best.columns:
                logger.info("--- FINAL ODDS SOURCE x PICK STATUS ---")
                for source in sources:
                    source_mask = best["odds_source"] == source
                    counts = best[source_mask]["Pick_Status"].value_counts().to_dict()
                    filtered_counts = {k: v for k, v in counts.items() if v > 0}
                    logger.info(f"odds_source '{source}' -> {filtered_counts}")

    # Final line-fidelity/provenance normalization for export rows.
    if not best.empty:
        raw_live_spread_line = pd.to_numeric(best.get("live_spread_line"), errors="coerce")
        raw_live_total_line = pd.to_numeric(best.get("live_total_line"), errors="coerce")
        best["upload_spread_line"] = pd.to_numeric(best.get("uploaded_spread_line"), errors="coerce")
        best["upload_total_line"] = pd.to_numeric(best.get("uploaded_total_line"), errors="coerce")
        best["base_spread_line"] = pd.to_numeric(best.get("spread_line"), errors="coerce")
        best["base_total_line"] = pd.to_numeric(best.get("total_line"), errors="coerce")

        market_type_norm = best["market_type"].astype(str).str.lower()
        is_spread = market_type_norm.isin({"spread_home", "spread_away"})
        is_total = market_type_norm.isin({"total_over", "total_under"})
        line_source_norm = best.get("line_source", pd.Series([""] * len(best), index=best.index)).astype(str).str.lower()
        live_match = line_source_norm.str.contains("live", na=False)

        has_live_numeric = raw_live_spread_line.notna() | raw_live_total_line.notna()
        trusted_live_match = live_match & has_live_numeric
        best["matched_live_spread_line"] = np.where(trusted_live_match, raw_live_spread_line, np.nan)
        best["matched_live_total_line"] = np.where(trusted_live_match, raw_live_total_line, np.nan)

        best["market_line_source"] = np.where(trusted_live_match, "live", np.where(best["upload_spread_line"].notna() | best["upload_total_line"].notna(), "upload", "base"))
        best["market_line_source_detail"] = np.where(trusted_live_match, line_source_norm.replace("", "live"), np.where(best["market_line_source"] == "upload", "uploaded_line", "base_generated_line"))
        best["selected_live_event_source"] = np.where(trusted_live_match, line_source_norm.replace("", "live"), "")
        norm = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True).str.strip()
        league_key = best.get("league", pd.Series([""] * len(best), index=best.index)).astype(str).str.strip().str.upper()
        home_key = norm(best.get("home_team", pd.Series([""] * len(best), index=best.index)))
        away_key = norm(best.get("away_team", pd.Series([""] * len(best), index=best.index)))
        game_date_key = best.get("game_date", pd.Series([""] * len(best), index=best.index)).astype(str).str.strip()
        commence_key = best.get("commence_time", best.get("game_time_est", pd.Series([""] * len(best), index=best.index))).astype(str).str.strip()
        family_key = np.where(is_total, "total", np.where(is_spread, "spread", "side"))
        strict_event_id = best.get("sportsbook_event_id", best.get("event_id", pd.Series([""] * len(best), index=best.index))).astype(str).str.strip()
        strict_event_key = best.get("sportsbook_event_key", best.get("event_key", pd.Series([""] * len(best), index=best.index))).astype(str).str.strip()
        candidate_key = (
            league_key + "::" + home_key + "::" + away_key + "::" + game_date_key + "::"
            + commence_key + "::" + pd.Series(family_key, index=best.index).astype(str) + "::"
            + market_type_norm.astype(str) + "::" + strict_event_id + "::" + strict_event_key
        )
        candidate_count = best.groupby(candidate_key)["home_team"].transform("size")
        best["live_event_match_key"] = np.where(trusted_live_match, candidate_key, "")
        best["line_candidate_count"] = np.where(trusted_live_match, candidate_count, 0).astype(int)
        best["line_event_identity_match_flag"] = trusted_live_match & candidate_count.eq(1)
        best["line_event_identity_reason"] = np.where(
            ~trusted_live_match, "no_live_candidate_for_row",
            np.where(candidate_count.eq(1), "exact_live_event_identity", "ambiguous_live_event_identity_multiple_candidates")
        )
        ambiguous_identity = trusted_live_match & best.get("orientation_source", pd.Series([""] * len(best), index=best.index)).astype(str).str.contains("fuzzy", case=False, na=False)
        best.loc[ambiguous_identity, "line_event_identity_match_flag"] = False
        best.loc[ambiguous_identity, "line_event_identity_reason"] = "ambiguous_live_event_identity_fuzzy_match"
        best.loc[ambiguous_identity, "line_candidate_count"] = 2

        best["market_line_used"] = pd.NA
        best.loc[is_spread & trusted_live_match, "market_line_used"] = best.loc[is_spread & trusted_live_match, "matched_live_spread_line"]
        best.loc[is_spread & ~live_match & best["upload_spread_line"].notna(), "market_line_used"] = best.loc[is_spread & ~live_match & best["upload_spread_line"].notna(), "upload_spread_line"]
        best.loc[is_spread & best["market_line_used"].isna(), "market_line_used"] = best.loc[is_spread & best["market_line_used"].isna(), "base_spread_line"]
        best.loc[is_total & trusted_live_match, "market_line_used"] = best.loc[is_total & trusted_live_match, "matched_live_total_line"]
        best.loc[is_total & ~live_match & best["upload_total_line"].notna(), "market_line_used"] = best.loc[is_total & ~live_match & best["upload_total_line"].notna(), "upload_total_line"]
        best.loc[is_total & best["market_line_used"].isna(), "market_line_used"] = best.loc[is_total & best["market_line_used"].isna(), "base_total_line"]
        best["market_line_used"] = pd.to_numeric(best["market_line_used"], errors="coerce")

        # Rebuild best_pick from resolved market_line_used to prevent stale/base leak-through.
        best["best_pick"] = best.apply(_format_best_pick, axis=1)

        # Validate line consistency and annotate rows.
        best["line_consistency_flag"] = True
        best["line_consistency_reason"] = ""
        best["line_provenance_warning"] = ""
        expected_pick = best.apply(_format_best_pick, axis=1)
        mismatch_mask = expected_pick.astype(str).str.strip() != best["best_pick"].astype(str).str.strip()
        missing_line_mask = (is_spread | is_total) & best["market_line_used"].isna()
        best.loc[mismatch_mask | missing_line_mask, "line_consistency_flag"] = False
        best.loc[mismatch_mask, "line_consistency_reason"] = "best_pick_text_mismatch_with_market_line_used"
        best.loc[missing_line_mask, "line_consistency_reason"] = best.loc[missing_line_mask, "line_consistency_reason"].replace("", "missing_market_line_used")
        best.loc[~best["line_event_identity_match_flag"], "line_consistency_flag"] = False
        best.loc[~best["line_event_identity_match_flag"], "line_consistency_reason"] = (
            best.loc[~best["line_event_identity_match_flag"], "line_consistency_reason"]
            .replace("", "line_event_identity_mismatch")
        )
        best.loc[(~live_match) & (is_spread | is_total), "line_provenance_warning"] = "Non-live line source used for best_pick"
        best.loc[~best["line_event_identity_match_flag"], "line_provenance_warning"] = (
            best.loc[~best["line_event_identity_match_flag"], "line_provenance_warning"]
            .replace("", "Ambiguous live event identity; manual review required")
        )

        # suspicious delta guardrails by league + market family
        league_norm = best.get("league", pd.Series([""] * len(best), index=best.index)).astype(str).str.upper()
        spread_delta = (best["matched_live_spread_line"] - best["upload_spread_line"]).abs()
        total_delta = (best["matched_live_total_line"] - best["upload_total_line"]).abs()
        suspicious_spread = is_spread & spread_delta.gt(3.0)
        raw_suspicious_total = (
            (league_norm.eq("MLB") & is_total & total_delta.gt(2.0))
            | (league_norm.eq("NHL") & is_total & total_delta.gt(1.5))
            | (league_norm.eq("NBA") & is_total & total_delta.gt(8.0))
        )
        # Plausibility-gated live totals: a live total that deviates materially from the
        # uploaded reference is only treated as suspicious when its OWN value is implausible
        # for the league. The real risk is a bad live read (a mis-scraped number), not a
        # live line that merely disagrees with a (often stale) uploaded reference — the live
        # line is the more current source, so a plausible live total is trusted and used.
        # Only a garbage live value falls through to the reject / upload-fallback path.
        # Ranges mirror the upload-plausibility ranges used in the recovery step below.
        plausible_live_total = (
            (league_norm.eq("MLB") & raw_live_total_line.between(5.5, 13.5, inclusive="both"))
            | (league_norm.eq("NHL") & raw_live_total_line.between(4.5, 8.5, inclusive="both"))
            | (league_norm.eq("NBA") & raw_live_total_line.between(185, 255, inclusive="both"))
            | (league_norm.eq("NCAAB") & raw_live_total_line.between(115, 175, inclusive="both"))
            | (league_norm.eq("NFL") & raw_live_total_line.between(30, 60, inclusive="both"))
            | (league_norm.eq("NCAAF") & raw_live_total_line.between(35, 75, inclusive="both"))
        )
        suspicious_total = raw_suspicious_total & ~plausible_live_total
        if diagnostics_out is not None:
            diagnostics_out["live_total_deviation_count"] = int(raw_suspicious_total.sum())
            diagnostics_out["live_total_trusted_plausible_count"] = int((raw_suspicious_total & plausible_live_total).sum())
        suspicious_line = suspicious_spread | suspicious_total
        best.loc[suspicious_line, "line_consistency_flag"] = False
        best.loc[suspicious_line, "line_consistency_reason"] = best.loc[suspicious_line, "line_consistency_reason"].replace("", "suspicious_live_line_delta")
        best.loc[suspicious_line, "line_provenance_warning"] = best.loc[suspicious_line, "line_provenance_warning"].replace("", "Live line deviates materially from uploaded/base reference")

        # Re-resolve suspicious/provenance rows using stricter identity keys before final status assignment.
        suspicious_or_warned = (
            (~best["line_consistency_flag"]) |
            best["line_consistency_reason"].astype(str).str.contains("suspicious_live_line_delta", na=False) |
            best["line_provenance_warning"].astype(str).str.strip().ne("")
        )
        strict_identity_key = candidate_key
        best["live_event_match_key"] = np.where(suspicious_or_warned, strict_identity_key, best["live_event_match_key"])
        strict_candidate_count = best.groupby(strict_identity_key)["home_team"].transform("size")
        best.loc[suspicious_or_warned, "line_candidate_count"] = strict_candidate_count.loc[suspicious_or_warned].astype(int)
        resolved_unambiguous = suspicious_or_warned & strict_candidate_count.eq(1) & trusted_live_match
        unresolved_suspicious = suspicious_or_warned & ~resolved_unambiguous
        unresolved_total_before_recovery = unresolved_suspicious & is_total

        # For resolved rows, always use the selected live event's direct line values.
        best.loc[resolved_unambiguous & is_spread, "market_line_used"] = best.loc[resolved_unambiguous & is_spread, "matched_live_spread_line"]
        best.loc[resolved_unambiguous & is_total, "market_line_used"] = best.loc[resolved_unambiguous & is_total, "matched_live_total_line"]
        best.loc[resolved_unambiguous, "selected_live_event_source"] = "strict_live_reresolved"
        best.loc[resolved_unambiguous, "line_event_identity_match_flag"] = True
        best.loc[resolved_unambiguous, "line_event_identity_reason"] = "strict_live_event_identity_reresolved"

        # A strict single-candidate re-resolution is treated as clean even when upload/base differ materially.
        best.loc[unresolved_suspicious, "line_event_identity_match_flag"] = False
        best.loc[unresolved_suspicious, "line_event_identity_reason"] = np.where(
            strict_candidate_count.loc[unresolved_suspicious].eq(0),
            "suspicious_live_line_unresolved_no_candidates",
            np.where(
                strict_candidate_count.loc[unresolved_suspicious].gt(1),
                "suspicious_live_line_unresolved_ambiguous_candidates",
                "suspicious_live_line_unresolved_delta_persists",
            ),
        )

        # Hard block unresolved suspicious lines from viable buckets.
        blocked_viable_status = best["Pick_Status"].astype(str).isin({"Actionable", "High Variance/Speculative"})
        best.loc[unresolved_suspicious & blocked_viable_status, "Pick_Status"] = "No Play"
        best.loc[unresolved_suspicious, "status_blocker_stage"] = "line_provenance"
        best.loc[unresolved_suspicious, "status_blocker_reason"] = "Suspicious live line delta could not be resolved"
        best.loc[unresolved_suspicious, "Status_Reason"] = "No Play: suspicious live line delta could not be resolved"
        best.loc[unresolved_suspicious, "market_line_source"] = "rejected_live"
        best.loc[unresolved_suspicious, "market_line_source_detail"] = "suspicious_live_line_rejected"
        best.loc[unresolved_suspicious & is_spread, "matched_live_spread_line"] = np.nan
        best.loc[unresolved_suspicious & is_total, "matched_live_total_line"] = np.nan
        best.loc[unresolved_suspicious, "line_provenance_warning"] = (
            best.loc[unresolved_suspicious, "line_provenance_warning"]
            .replace("", "Suspicious live line rejected; exact event/line unresolved")
        )
        unresolved_label = np.where(
            is_spread,
            best.get("away_team", pd.Series([""] * len(best), index=best.index)).astype(str).str.strip().replace("", "Spread")
            + " line unresolved",
            "Total line unresolved",
        )
        best.loc[unresolved_suspicious, "best_pick"] = pd.Series(unresolved_label, index=best.index).loc[unresolved_suspicious]
        best.loc[unresolved_suspicious, "market_line_used"] = np.nan

        # Conservative fallback: recover unresolved suspicious totals with plausible upload/reference totals.
        upload_total_candidate = pd.to_numeric(
            best.get("uploaded_total_line", best.get("upload_total_line", pd.Series([np.nan] * len(best), index=best.index))),
            errors="coerce",
        )
        upload_total_candidate = upload_total_candidate.fillna(pd.to_numeric(best.get("upload_total_line", pd.Series([np.nan] * len(best), index=best.index)), errors="coerce"))
        plausible_total = pd.Series(False, index=best.index)
        plausible_total = plausible_total | (league_norm.eq("MLB") & upload_total_candidate.between(5.5, 13.5, inclusive="both"))
        plausible_total = plausible_total | (league_norm.eq("NHL") & upload_total_candidate.between(4.5, 8.5, inclusive="both"))
        plausible_total = plausible_total | (league_norm.eq("NBA") & upload_total_candidate.between(185, 255, inclusive="both"))
        plausible_total = plausible_total | (league_norm.eq("NCAAB") & upload_total_candidate.between(115, 175, inclusive="both"))
        plausible_total = plausible_total | (league_norm.eq("NFL") & upload_total_candidate.between(30, 60, inclusive="both"))
        plausible_total = plausible_total | (league_norm.eq("NCAAF") & upload_total_candidate.between(35, 75, inclusive="both"))
        has_team_identity = home_key.ne("") & away_key.ne("")
        upload_recovery_candidate = (
            is_total
            & best["line_consistency_reason"].astype(str).str.contains("suspicious_live_line_delta", na=False)
            & best["market_line_source"].astype(str).isin(["rejected_live", "live", "upload", "base"])
            & has_team_identity
            & upload_total_candidate.notna()
        )
        recover_with_upload_total = upload_recovery_candidate & plausible_total
        rejected_upload_plausibility = upload_recovery_candidate & ~plausible_total

        best.loc[recover_with_upload_total, "market_line_source"] = "upload"
        best.loc[recover_with_upload_total, "market_line_source_detail"] = "upload_total_fallback_after_rejected_live"
        best.loc[recover_with_upload_total, "market_line_used"] = upload_total_candidate.loc[recover_with_upload_total]
        best.loc[recover_with_upload_total, "matched_live_total_line"] = np.nan
        best.loc[recover_with_upload_total, "best_pick"] = best.loc[recover_with_upload_total].apply(_format_best_pick, axis=1)
        best.loc[recover_with_upload_total, "line_consistency_flag"] = True
        best.loc[recover_with_upload_total, "line_consistency_reason"] = "recovered_with_upload_total_after_rejected_live"
        best.loc[recover_with_upload_total, "line_provenance_warning"] = "Live total rejected; using uploaded/reference total"
        best.loc[recover_with_upload_total, "line_event_identity_match_flag"] = False
        best.loc[recover_with_upload_total, "line_event_identity_reason"] = "upload_total_fallback_after_rejected_live"
        best.loc[recover_with_upload_total, "status_blocker_stage"] = "line_provenance"
        best.loc[recover_with_upload_total, "status_blocker_reason"] = "Upload total fallback used after rejected live total"
        best.loc[recover_with_upload_total, "Pick_Status"] = "High Variance/Speculative"
        best.loc[recover_with_upload_total, "Status_Reason"] = "High Variance/Speculative: Upload total fallback used after rejected live total"
        best.loc[recover_with_upload_total, "Kelly_Bet_Size"] = 0.0
        if not ALLOW_UPLOAD_TOTAL_FALLBACK_ACTIONABLE:
            best.loc[recover_with_upload_total, "Pick_Status"] = "High Variance/Speculative"

        if diagnostics_out is not None:
            diagnostics_out["total_unresolved_count_before_upload_recovery"] = int(unresolved_total_before_recovery.sum())
            diagnostics_out["total_upload_recovery_candidate_count"] = int(upload_recovery_candidate.sum())
            diagnostics_out["total_upload_recovered_count"] = int(recover_with_upload_total.sum())
            diagnostics_out["total_upload_recovery_rejected_plausibility_count"] = int(rejected_upload_plausibility.sum())

        # Final hard enforcement: any row that still fails line validation must be rejected.
        recovered_upload_total_mask = best.get("market_line_source_detail", pd.Series([""] * len(best), index=best.index)).astype(str).eq("upload_total_fallback_after_rejected_live")
        final_rejected_line = (
            (~best["line_consistency_flag"])
            | best["line_consistency_reason"].astype(str).str.contains("suspicious_live_line_delta", na=False)
            | best["line_provenance_warning"].astype(str).str.contains("Live line deviates materially", case=False, na=False)
            | (~best["line_event_identity_match_flag"])
        ) & ~recovered_upload_total_mask
        best.loc[final_rejected_line, "Pick_Status"] = "No Play"
        best.loc[final_rejected_line, "Status_Reason"] = "No Play: suspicious live line delta could not be resolved"
        best.loc[final_rejected_line, "status_blocker_stage"] = "line_provenance"
        best.loc[final_rejected_line, "status_blocker_reason"] = "Suspicious live line delta could not be resolved"
        best.loc[final_rejected_line, "market_line_source"] = "rejected_live"
        best.loc[final_rejected_line, "market_line_source_detail"] = "suspicious_live_line_rejected"
        best.loc[final_rejected_line, "market_line_used"] = np.nan
        best.loc[final_rejected_line, "matched_live_spread_line"] = np.nan
        best.loc[final_rejected_line, "matched_live_total_line"] = np.nan
        best.loc[final_rejected_line, "line_event_identity_match_flag"] = False
        best.loc[final_rejected_line, "line_event_identity_reason"] = "suspicious_live_line_rejected_after_validation"
        best.loc[final_rejected_line, "selected_live_event_source"] = "rejected_live"
        rejected_label = np.where(
            is_spread,
            best.get("away_team", pd.Series([""] * len(best), index=best.index)).astype(str).str.strip().replace("", "Spread")
            + " line unresolved",
            "Total line unresolved",
        )
        best.loc[final_rejected_line, "best_pick"] = pd.Series(rejected_label, index=best.index).loc[final_rejected_line]
        if diagnostics_out is not None:
            recovered_mask = best.get("market_line_source_detail", pd.Series([""] * len(best), index=best.index)).astype(str).eq("upload_total_fallback_after_rejected_live")
            diagnostics_out["total_upload_recovered_actionable_count"] = int((recovered_mask & best["Pick_Status"].astype(str).eq("Actionable")).sum())
            diagnostics_out["total_upload_recovered_kelly_positive_count"] = int((recovered_mask & pd.to_numeric(best.get("Kelly_Bet_Size", 0), errors="coerce").fillna(0).gt(0)).sum())
        if (mismatch_mask | missing_line_mask).any():
            logger.warning("Line consistency issues detected rows=%s", int((mismatch_mask | missing_line_mask).sum()))

    # Final value safety guardrail: negative EV/edge rows cannot remain Actionable/High Variance.
    eff_ev = pd.to_numeric(best.get("effective_expected_value", pd.Series([np.nan] * len(best), index=best.index)), errors="coerce")
    eff_edge = pd.to_numeric(best.get("effective_edge", pd.Series([np.nan] * len(best), index=best.index)), errors="coerce")
    base_ev = pd.to_numeric(best.get("expected_value", pd.Series([np.nan] * len(best), index=best.index)), errors="coerce")
    base_edge = pd.to_numeric(best.get("edge", pd.Series([np.nan] * len(best), index=best.index)), errors="coerce")
    negative_ev_mask = eff_ev.lt(0) | base_ev.lt(0)
    negative_edge_mask = eff_edge.lt(0) | base_edge.lt(0)
    negative_value_guardrail_mask = negative_ev_mask | negative_edge_mask
    pre_status = best["Pick_Status"].astype(str).copy()
    downgrade_viable_mask = negative_value_guardrail_mask & pre_status.isin({"Actionable", "High Variance/Speculative"})
    downgraded_from_high_variance_mask = downgrade_viable_mask & pre_status.eq("High Variance/Speculative")
    downgraded_from_actionable_mask = downgrade_viable_mask & pre_status.eq("Actionable")
    best.loc[downgrade_viable_mask, "Pick_Status"] = "No Play"
    best.loc[downgrade_viable_mask, "Status_Reason"] = "No Play: negative EV or negative edge after final validation"
    best.loc[downgrade_viable_mask, "status_blocker_stage"] = "value_guardrail"
    best.loc[downgrade_viable_mask, "status_blocker_reason"] = "Negative EV or edge after final validation"
    best.loc[downgrade_viable_mask, "Kelly_Bet_Size"] = 0.0

    # Production-only probability calibration for totals-over overconfidence.
    best["production_win_probability"] = pd.to_numeric(best.get("effective_win_probability", 0.5), errors="coerce").fillna(0.5)
    best["probability_calibration_reason"] = "none"
    mt_lower = _string_series(best, "market_type").str.lower()
    league_upper = _string_series(best, "league").str.upper()
    is_total_over = mt_lower.eq("total_over")
    is_mlb_total_over = is_total_over & league_upper.eq("MLB")
    total_over_shrink = np.where(is_mlb_total_over, float(MLB_TOTAL_OVER_PROB_SHRINK), float(TOTAL_OVER_PROB_SHRINK))
    # MLB Overs: reset to calibrated_probability before shrinking to avoid double-shrink.
    # The gating stage already applied shrinkage to effective_win_probability; using
    # that as the base here would compound shrink^2 = 0.72x instead of 0.85x.
    if is_mlb_total_over.any():
        calib_base = pd.to_numeric(
            best.get("calibrated_probability", pd.Series([np.nan] * len(best), index=best.index)),
            errors="coerce"
        ).fillna(0.5)
        best.loc[is_mlb_total_over, "production_win_probability"] = calib_base[is_mlb_total_over]
    best.loc[is_total_over, "production_win_probability"] = 0.5 + total_over_shrink[is_total_over] * (best.loc[is_total_over, "production_win_probability"] - 0.5)
    best.loc[is_total_over & ~is_mlb_total_over, "probability_calibration_reason"] = f"total_over_shrink={float(TOTAL_OVER_PROB_SHRINK):.2f}"
    best.loc[is_mlb_total_over, "probability_calibration_reason"] = f"mlb_total_over_shrink={float(MLB_TOTAL_OVER_PROB_SHRINK):.2f}_from_calibrated"

    decimal_odds = pd.to_numeric(best.get("decimal_odds", pd.Series([np.nan] * len(best), index=best.index)), errors="coerce")
    fallback_decimal = 1.0 / pd.to_numeric(best.get("market_probability", pd.Series([np.nan] * len(best), index=best.index)), errors="coerce")
    decimal_odds = decimal_odds.fillna(fallback_decimal).clip(lower=1.01)
    best["production_expected_value"] = (best["production_win_probability"] * decimal_odds) - 1.0
    implied_prob = (1.0 / decimal_odds).replace([np.inf, -np.inf], np.nan)
    best["production_edge"] = best["production_win_probability"] - implied_prob

    # MLB totals-over stricter production gate after shrinkage.
    mlb_over_fail = (
        is_mlb_total_over
        & (
            best["production_win_probability"].lt(float(MLB_TOTAL_OVER_MIN_PRODUCTION_WIN_PROB))
            | best["production_expected_value"].lt(float(MLB_TOTAL_OVER_MIN_PRODUCTION_EV))
            | best["production_edge"].lt(float(MLB_TOTAL_OVER_MIN_PRODUCTION_EDGE))
        )
        & best["Pick_Status"].astype(str).eq("Actionable")
    )
    best.loc[mlb_over_fail, "Pick_Status"] = "High Variance/Speculative"
    best.loc[mlb_over_fail, "production_eligible"] = False
    best.loc[mlb_over_fail, "Kelly_Bet_Size"] = 0.0
    best.loc[mlb_over_fail, "status_blocker_stage"] = "production_market_guard"
    best.loc[mlb_over_fail, "status_blocker_reason"] = "MLB total-over production guard"
    best.loc[mlb_over_fail, "Status_Reason"] = "High Variance: downgraded by MLB total-over production guard"

    # Final production concentration guard: cap total_over and MLB total_over share/count.
    actionable_mask = best["Pick_Status"].astype(str).eq("Actionable")
    actionable_count = int(actionable_mask.sum())
    actionable_type_counts = best.loc[actionable_mask, "market_type"].astype(str).str.lower().value_counts().to_dict()
    actionable_total_over_count = int(actionable_type_counts.get("total_over", 0))
    actionable_mlb_total_over_count = int((actionable_mask & is_mlb_total_over).sum())
    over_share = (actionable_total_over_count / actionable_count) if actionable_count else 0.0
    over_limit = min(
        int(MAX_TOTAL_OVER_ACTIONABLE_COUNT),
        int(np.ceil(actionable_count * float(MAX_TOTAL_OVER_ACTIONABLE_SHARE))) if actionable_count else 0,
    )
    keep_total_over = max(0, min(actionable_total_over_count, over_limit))
    concentration_reasons: list[str] = []
    if actionable_total_over_count > int(MAX_TOTAL_OVER_ACTIONABLE_COUNT):
        concentration_reasons.append("total_over_count_cap")
    if actionable_mlb_total_over_count > int(MAX_MLB_TOTAL_OVER_ACTIONABLE_COUNT):
        concentration_reasons.append("mlb_total_over_count_cap")
    concentration_guard_active = len(concentration_reasons) > 0

    if concentration_guard_active and actionable_total_over_count > keep_total_over:
        total_over_candidates = best[actionable_mask & is_total_over].copy()
        total_over_candidates["_rank_sort"] = pd.to_numeric(total_over_candidates.get("Triple_Filter_Rank"), errors="coerce").fillna(9999)
        total_over_candidates = total_over_candidates.sort_values(
            by=["_rank_sort", "production_expected_value", "production_edge", "production_win_probability"],
            ascending=[True, False, False, False],
            na_position="last",
        )
        keep_idx = set(total_over_candidates.head(keep_total_over).index.tolist())
        downgrade_idx = [idx for idx in total_over_candidates.index if idx not in keep_idx]
        if downgrade_idx:
            best.loc[downgrade_idx, "Pick_Status"] = "High Variance/Speculative"
            best.loc[downgrade_idx, "production_eligible"] = False
            best.loc[downgrade_idx, "Kelly_Bet_Size"] = 0.0
            best.loc[downgrade_idx, "status_blocker_stage"] = "production_concentration_guard"
            best.loc[downgrade_idx, "status_blocker_reason"] = "Total over concentration guard"
            best.loc[downgrade_idx, "Status_Reason"] = "High Variance: downgraded by total-over concentration guard"

    # Speculative concentration guard: the same correlated-Over bleed that hits the
    # Actionable card also hits the High Variance/Speculative surface, which the cap
    # above does not touch (6 Jun: 4 MLB "Over 7.5" plays in HV all lost together while
    # benched Unders won). Cap the number of total_over (and MLB total_over) HV picks and
    # downgrade the lowest-ranked excess to Below Threshold so the speculative card can't
    # collapse onto one league+direction. Greedy keep respects both the overall and the
    # MLB-specific cap, retaining best-ranked picks first.
    hv_total_over_mask = best["Pick_Status"].astype(str).eq("High Variance/Speculative") & is_total_over
    if int(hv_total_over_mask.sum()) > 0:
        hv_candidates = best[hv_total_over_mask].copy()
        hv_candidates["_rank_sort"] = pd.to_numeric(hv_candidates.get("Triple_Filter_Rank"), errors="coerce").fillna(9999)
        hv_candidates["_is_mlb_over"] = is_mlb_total_over.reindex(hv_candidates.index).fillna(False).astype(bool)
        hv_candidates = hv_candidates.sort_values(
            by=["_rank_sort", "effective_expected_value", "effective_edge", "effective_win_probability"],
            ascending=[True, False, False, False],
            na_position="last",
        )
        hv_downgrade_idx = _total_over_concentration_downgrades(
            hv_candidates,
            overall_cap=int(MAX_TOTAL_OVER_HIGH_VARIANCE_COUNT),
            mlb_cap=int(MAX_MLB_TOTAL_OVER_HIGH_VARIANCE_COUNT),
        )
        if hv_downgrade_idx:
            best.loc[hv_downgrade_idx, "Pick_Status"] = "Below Threshold"
            best.loc[hv_downgrade_idx, "production_eligible"] = False
            best.loc[hv_downgrade_idx, "Kelly_Bet_Size"] = 0.0
            best.loc[hv_downgrade_idx, "status_blocker_stage"] = "speculative_concentration_guard"
            best.loc[hv_downgrade_idx, "status_blocker_reason"] = "High Variance total-over concentration guard"
            best.loc[hv_downgrade_idx, "Status_Reason"] = "Below Threshold: downgraded by speculative total-over concentration guard"

    # Empirical tier overlay: final tier pass driven by realized bucket win rates
    # + isotonic-calibrated probability (Jun 5-10: EV/edge tiers hit ~21%, Below
    # Threshold 59% — stake followed inverted tiers). Runs AFTER every guard pass
    # above so safety statuses are final; the existing degraded-feature scaling,
    # non-Actionable Kelly zeroing, and empty-card recovery below all operate on
    # the corrected tiers. Best-effort: any failure leaves legacy tiers in place.
    from app_core.weights_config import EMPIRICAL_TIER_OVERLAY_ENABLED
    if EMPIRICAL_TIER_OVERLAY_ENABLED and not best.empty:
        try:
            from core.empirical_tiers import assign_empirical_tiers, load_bucket_stats
            from core.kelly_optimizer import kelly_fraction
            from core.probability_calibration import load_calibration

            _bucket_stats = load_bucket_stats()
            if _bucket_stats:
                _pre_actionable = int(best["Pick_Status"].astype(str).eq("Actionable").sum())
                best = assign_empirical_tiers(best, _bucket_stats, load_calibration())

                # Re-apply the Actionable total-over concentration caps to the
                # post-overlay card so empirical promotions cannot exceed them.
                _is_total_over = best["market_type"].astype(str).str.lower().eq("total_over")
                _is_mlb_over = _is_total_over & best["league"].astype(str).str.upper().eq("MLB")
                _act_over_mask = best["Pick_Status"].astype(str).eq("Actionable") & _is_total_over
                if int(_act_over_mask.sum()) > 0:
                    _cands = best[_act_over_mask].copy()
                    _cands["_rank_sort"] = -pd.to_numeric(
                        _cands.get("empirical_edge"), errors="coerce"
                    ).fillna(-9.0)
                    _cands["_is_mlb_over"] = _is_mlb_over.reindex(_cands.index).fillna(False).astype(bool)
                    _cands = _cands.sort_values("_rank_sort")
                    _excess = _total_over_concentration_downgrades(
                        _cands,
                        overall_cap=int(MAX_TOTAL_OVER_ACTIONABLE_COUNT),
                        mlb_cap=int(MAX_MLB_TOTAL_OVER_ACTIONABLE_COUNT),
                    )
                    if _excess:
                        best.loc[_excess, "Pick_Status"] = "High Variance/Speculative"
                        best.loc[_excess, "status_blocker_stage"] = "empirical_tier_overlay_concentration"
                        best.loc[_excess, "Status_Reason"] = (
                            "High Variance: empirical promotion capped by total-over concentration guard"
                        )

                # Mirror cap for total_under (12 Jun: 4-5 Actionable Unders busted as a
                # block on a leaguewide-over night). Keep the best-edge Unders up to the
                # cap; drop the lowest-edge excess to High Variance.
                _is_total_under = best["market_type"].astype(str).str.lower().eq("total_under")
                _is_mlb_under = _is_total_under & best["league"].astype(str).str.upper().eq("MLB")
                _act_under_mask = best["Pick_Status"].astype(str).eq("Actionable") & _is_total_under
                if int(_act_under_mask.sum()) > 0:
                    _candsu = best[_act_under_mask].copy()
                    _candsu["_rank_sort"] = -pd.to_numeric(
                        _candsu.get("empirical_edge"), errors="coerce"
                    ).fillna(-9.0)
                    _candsu["_is_mlb_under"] = _is_mlb_under.reindex(_candsu.index).fillna(False).astype(bool)
                    _candsu = _candsu.sort_values("_rank_sort")
                    _excess_u = _total_over_concentration_downgrades(
                        _candsu,
                        overall_cap=int(MAX_TOTAL_UNDER_ACTIONABLE_COUNT),
                        mlb_cap=int(MAX_MLB_TOTAL_UNDER_ACTIONABLE_COUNT),
                        flag_col="_is_mlb_under",
                    )
                    if _excess_u:
                        best.loc[_excess_u, "Pick_Status"] = "High Variance/Speculative"
                        best.loc[_excess_u, "status_blocker_stage"] = "empirical_tier_overlay_concentration"
                        best.loc[_excess_u, "Status_Reason"] = (
                            "High Variance: empirical promotion capped by total-under concentration guard"
                        )

                # Size Kelly for empirically promoted Actionable rows from the
                # empirical probability at the pick's own odds (0.25x fractional
                # Kelly, 4% bankroll cap — the Actionable convention). Demoted
                # rows are zeroed by the non-Actionable Kelly pass below.
                _kelly_now = pd.to_numeric(best.get("Kelly_Bet_Size"), errors="coerce").fillna(0.0)
                _promoted = best["Pick_Status"].astype(str).eq("Actionable") & ~_kelly_now.gt(0)
                for _idx in best.index[_promoted]:
                    _dec = pd.to_numeric(best.at[_idx, "decimal_odds"] if "decimal_odds" in best.columns else None, errors="coerce")
                    if pd.isna(_dec) or _dec <= 1.0:
                        _amer = pd.to_numeric(best.at[_idx, "odds_american"] if "odds_american" in best.columns else None, errors="coerce")
                        if pd.notna(_amer) and _amer != 0:
                            _dec = 1 + _amer / 100.0 if _amer > 0 else 1 + 100.0 / abs(_amer)
                    _p = pd.to_numeric(best.at[_idx, "empirical_win_probability"], errors="coerce")
                    if pd.notna(_dec) and _dec > 1.0 and pd.notna(_p):
                        best.at[_idx, "Kelly_Bet_Size"] = min(0.04, kelly_fraction(float(_p), float(_dec)) * 0.25)

                if diagnostics_out is not None:
                    diagnostics_out["empirical_tier_overlay"] = {
                        "applied": True,
                        "actionable_before": _pre_actionable,
                        "actionable_after": int(best["Pick_Status"].astype(str).eq("Actionable").sum()),
                        "bucket_stats_n": int(_bucket_stats["overall"]["n"]),
                    }
        except Exception as e:
            logger.warning(f"Empirical tier overlay failed; keeping legacy tiers: {e}")

    # Slate direction-balance backstop (13 Jun: a corrupt TheOver feed produced a
    # 14-of-15 all-Over card with a staked Actionable). A near-unanimous totals
    # direction across many games is a near-certain data/orientation fault, not a
    # real read, so suspend big-Kelly staking: demote Actionable totals to High
    # Variance and surface a loud run_health_warning. Conservative by design — it
    # never fabricates the missing side, it just refuses to stake confidently on a
    # slate whose direction signal can't be trusted. Catches ANY cause, not only
    # TheOver. Runs after the overlay so it has the final tiers.
    try:
        from core.slate_quality import slate_direction_imbalanced
        _imbalanced, _imb_reason = slate_direction_imbalanced(
            _string_series(best, "market_type")
        )
        if _imbalanced:
            _is_total = _string_series(best, "market_type").str.lower().str.contains("total", na=False)
            _act_total = best["Pick_Status"].astype(str).eq("Actionable") & _is_total
            best.loc[_act_total, "Pick_Status"] = "High Variance/Speculative"
            best.loc[_act_total, "Kelly_Bet_Size"] = 0.0
            best.loc[_act_total, "production_eligible"] = False
            best.loc[_act_total, "status_blocker_stage"] = "slate_direction_imbalance_guard"
            best.loc[_act_total, "Status_Reason"] = (
                "High Variance: " + _imb_reason
            )
            existing_warn = _string_series(best, "run_health_warning")
            best["run_health_warning"] = existing_warn.where(
                existing_warn.str.len() > 0, _imb_reason
            )
            if diagnostics_out is not None:
                diagnostics_out["slate_direction_imbalance"] = _imb_reason
    except Exception as e:  # never let a guard break the pipeline
        logger.warning(f"Slate direction-balance guard skipped: {e}")

    # Degraded-feature Kelly reduction/caps for production safety.
    degraded_run = bool(
        best.get("degraded_feature_subset_flag", pd.Series([False] * len(best), index=best.index)).fillna(False).astype(bool).any()
        or _string_series(best, "run_health_warning").str.contains("degraded", case=False, na=False).any()
    )
    if degraded_run:
        kelly = pd.to_numeric(best.get("Kelly_Bet_Size", 0.0), errors="coerce").fillna(0.0)
        best["Kelly_Bet_Size"] = (kelly * float(DEGRADED_FEATURE_KELLY_MULTIPLIER)).clip(lower=0.0)
        best["Kelly_Bet_Size"] = best["Kelly_Bet_Size"].clip(upper=float(DEGRADED_FEATURE_MAX_PICK_EXPOSURE_PCT))
        slate_sum = float(best["Kelly_Bet_Size"].sum())
        if slate_sum > float(DEGRADED_FEATURE_MAX_SLATE_EXPOSURE_PCT) and slate_sum > 0:
            best["Kelly_Bet_Size"] = best["Kelly_Bet_Size"] * (float(DEGRADED_FEATURE_MAX_SLATE_EXPOSURE_PCT) / slate_sum)
        best["Kelly_Bet_Size"] = pd.to_numeric(best["Kelly_Bet_Size"], errors="coerce").fillna(0.0)
    best.loc[~best["Pick_Status"].astype(str).eq("Actionable"), "Kelly_Bet_Size"] = 0.0
    best["production_eligible"] = best["Pick_Status"].astype(str).eq("Actionable") & best["Kelly_Bet_Size"].gt(0)

    # Empty Card Recovery: when no Actionable picks survive all gates, surface the
    # best High Variance or Below Threshold candidates under strict production floors.
    from app_core.weights_config import (
        ALLOW_EMPTY_CARD_RECOVERY,
        EMPTY_CARD_RECOVERY_MAX_PICKS,
        EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EV,
        EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EDGE,
        EMPTY_CARD_RECOVERY_MIN_PRODUCTION_WIN_PROB,
        EMPTY_CARD_RECOVERY_EXCLUDE_MARKET_TYPES,
        EMPTY_CARD_RECOVERY_EXCLUDE_SOURCES,
        EMPTY_CARD_RECOVERY_MAX_KELLY_PER_PICK_PCT,
        EMPTY_CARD_RECOVERY_MAX_KELLY_TOTAL_PCT,
        ALLOW_MLB_TOTAL_OVER_EMPTY_CARD_RECOVERY,
        ALLOW_MLB_TOTAL_UNDER_EMPTY_CARD_RECOVERY,
        ALLOW_NHL_TOTAL_UNDER_EMPTY_CARD_RECOVERY,
    )
    if ALLOW_EMPTY_CARD_RECOVERY and not best.empty:
        # Emptiness must be status-based, NOT production_eligible-based. This runs
        # inside build_best_picks_df, BEFORE Kelly is sized downstream in
        # streamlit_app, so production_eligible (Actionable AND Kelly>0) is
        # structurally all-False here — using it made the gate fire unconditionally
        # and backfill a redundant pick even when a legitimate Actionable pick
        # (e.g. a sub-8.0 over carve-out) was present. Matches the streamlit_app
        # recovery's Pick_Status-based emptiness definition.
        from app_core.card_recovery import actionable_card_is_empty
        if actionable_card_is_empty(best):
            is_mlb_total_over = (
                best["league"].astype(str).str.upper().eq("MLB")
                & best["market_type"].astype(str).str.lower().eq("total_over")
            )
            is_mlb_total_under = (
                best["league"].astype(str).str.upper().eq("MLB")
                & best["market_type"].astype(str).str.lower().eq("total_under")
            )
            is_nhl_total_under = (
                best["league"].astype(str).str.upper().eq("NHL")
                & best["market_type"].astype(str).str.lower().eq("total_under")
            )
            recovery_mask = (
                best["Pick_Status"].astype(str).isin({"High Variance/Speculative", "Below Threshold"})
                & best.get("status_metric_basis", pd.Series("", index=best.index)).astype(str).eq("effective")
                & ~best["market_type"].astype(str).str.lower().isin(
                    [m.lower() for m in EMPTY_CARD_RECOVERY_EXCLUDE_MARKET_TYPES]
                )
                & ~best.get("candidate_source", pd.Series("", index=best.index)).astype(str).isin(
                    EMPTY_CARD_RECOVERY_EXCLUDE_SOURCES
                )
                & pd.to_numeric(best.get("effective_expected_value"), errors="coerce").ge(EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EV)
                & pd.to_numeric(best.get("effective_edge"), errors="coerce").ge(EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EDGE)
                & pd.to_numeric(best.get("effective_win_probability"), errors="coerce").ge(EMPTY_CARD_RECOVERY_MIN_PRODUCTION_WIN_PROB)
                & ~best.get("suspicious_data_flag", pd.Series(False, index=best.index)).fillna(False).astype(bool)
            )
            if not ALLOW_MLB_TOTAL_OVER_EMPTY_CARD_RECOVERY:
                recovery_mask = recovery_mask & ~is_mlb_total_over
            if not ALLOW_MLB_TOTAL_UNDER_EMPTY_CARD_RECOVERY:
                recovery_mask = recovery_mask & ~is_mlb_total_under
            if not ALLOW_NHL_TOTAL_UNDER_EMPTY_CARD_RECOVERY:
                recovery_mask = recovery_mask & ~is_nhl_total_under

            recovery_candidates = best[recovery_mask]
            if not recovery_candidates.empty:
                top = recovery_candidates.sort_values(
                    ["effective_expected_value", "effective_edge", "effective_win_probability"],
                    ascending=[False, False, False],
                ).head(EMPTY_CARD_RECOVERY_MAX_PICKS)
                prev_status = best.loc[top.index, "Pick_Status"].astype(str).copy()
                best.loc[top.index, "Pick_Status"] = "Actionable"
                best.loc[top.index, "Status_Reason"] = "Empty card recovery (promoted from: " + prev_status + ")"
                best.loc[top.index, "status_blocker_stage"] = ""
                kelly_vals = pd.to_numeric(best.loc[top.index, "Kelly_Bet_Size"], errors="coerce").fillna(0.0)
                # Picks zeroed by concentration guard or other mechanisms get the cap as a floor
                kelly_vals = kelly_vals.where(kelly_vals.gt(0), EMPTY_CARD_RECOVERY_MAX_KELLY_PER_PICK_PCT)
                best.loc[top.index, "Kelly_Bet_Size"] = kelly_vals.clip(upper=EMPTY_CARD_RECOVERY_MAX_KELLY_PER_PICK_PCT)
                # Cap total slate exposure for recovery picks
                total_recovery_kelly = float(best.loc[top.index, "Kelly_Bet_Size"].sum())
                if total_recovery_kelly > EMPTY_CARD_RECOVERY_MAX_KELLY_TOTAL_PCT and total_recovery_kelly > 0:
                    best.loc[top.index, "Kelly_Bet_Size"] *= EMPTY_CARD_RECOVERY_MAX_KELLY_TOTAL_PCT / total_recovery_kelly
                best["production_eligible"] = best["Pick_Status"].astype(str).eq("Actionable") & best["Kelly_Bet_Size"].gt(0)
                logger.info(f"Empty card recovery: promoted {len(top)} pick(s) — {top['market_type'].tolist()}")
                if diagnostics_out is not None:
                    diagnostics_out["empty_card_recovery_triggered"] = True
                    diagnostics_out["empty_card_recovery_promoted_count"] = len(top)
                    diagnostics_out["empty_card_recovery_market_types"] = top["market_type"].tolist()
                    diagnostics_out["empty_card_recovery_leagues"] = top["league"].tolist() if "league" in top.columns else []

    final_best_df = best[BEST_PICK_COLUMNS].copy()
    final_best_df = ensure_best_pick_export_columns(final_best_df, diagnostics_out=diagnostics_out)

    if diagnostics_out is not None:
        diagnostics_out.setdefault("empty_card_recovery_triggered", False)
        diagnostics_out.setdefault("empty_card_recovery_promoted_count", 0)
        diagnostics_out.setdefault("empty_card_recovery_market_types", [])
        diagnostics_out.setdefault("empty_card_recovery_leagues", [])
        actionable_df = final_best_df[final_best_df["Pick_Status"] == "Actionable"]
        actionable_family_counts = actionable_df["market_type"].astype(str).str.lower().apply(lambda x: "total" if "total" in x else "side").value_counts().to_dict()
        actionable_market_type_counts = actionable_df["market_type"].value_counts().to_dict()
        actionable_counts_by_league = actionable_df["league"].value_counts().to_dict() if "league" in actionable_df.columns else {}

        # Collect additional requested debug metrics
        nhl_totals_actionable = actionable_df[(actionable_df["league"].astype(str).str.upper() == "NHL") & (actionable_df["market_type"].astype(str).str.lower().str.contains("total"))].shape[0]

        # Calculate spans from the logic above by parsing reasons
        spreads_downgraded_by_divergence = final_best_df[
            (final_best_df["Pick_Status"] == "High Variance/Speculative") &
            (final_best_df["Status_Reason"].str.contains("diverge by > 20%", na=False)) &
            (final_best_df["market_type"].astype(str).str.lower().str.contains("spread"))
        ].shape[0]

        spreads_rescued_by_divergence = actionable_df[
            actionable_df["Status_Reason"].str.contains("Spread divergence override applied", na=False)
        ].shape[0]

        # Calculate totals below prob floor
        from app_core.weights_config import NHL_TOTAL_MIN_WIN_PROB, TOTAL_MIN_WIN_PROB
        totals_below_prob_floor = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["market_type"].astype(str).str.lower().str.contains("total")) &
            (final_best_df["Status_Reason"].str.contains("Fails minimum Win Probability", na=False))
        ].shape[0]

        # Calculate new guardrail stats
        blocked_by_cold_market = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["Status_Reason"].str.contains("cold-market penalty", na=False))
        ].shape[0]

        blocked_by_fallback_heavy_totals = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["Status_Reason"].str.contains("fallback-heavy totals penalty", na=False))
        ].shape[0]

        blocked_by_total_under = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["Status_Reason"].str.contains("stricter total_under cold-market penalty", na=False))
        ].shape[0]

        blocked_by_nhl_total = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["Status_Reason"].str.contains("NHL total cold-market penalty threshold", na=False))
        ].shape[0]

        # League + Market calibration metrics
        if "league" in final_best_df.columns and "market_type" in final_best_df.columns:
            final_best_df["league_market"] = final_best_df["league"].astype(str) + " " + final_best_df["market_type"].astype(str)
            actionable_counts_by_league_market = final_best_df[final_best_df["Pick_Status"] == "Actionable"]["league_market"].value_counts().to_dict()
            below_threshold_counts_by_league_market = final_best_df[final_best_df["Pick_Status"] == "Below Threshold"]["league_market"].value_counts().to_dict()
        else:
            actionable_counts_by_league_market = {}
            below_threshold_counts_by_league_market = {}

        # Calculate new metrics
        actionable_counts_by_consensus = actionable_df["consensus_agreement"].value_counts().to_dict() if "consensus_agreement" in actionable_df.columns else {}
        downgraded_by_neutral = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["Status_Reason"].str.contains("Neutral overlay", na=False))
        ].shape[0]
        downgraded_by_disagrees = final_best_df[
            (final_best_df["Pick_Status"] == "High Variance/Speculative") &
            (final_best_df["Status_Reason"].str.contains("Disagrees overlay", na=False))
        ].shape[0]
        side_floor_failures = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["Status_Reason"].str.contains("side minimum Win Probability", na=False))
        ].shape[0]


        diagnostics_out["market_type_counts"] = final_best_df["market_type"].value_counts().to_dict()
        diagnostics_out["actionable_market_type_counts"] = actionable_market_type_counts
        diagnostics_out["actionable_total_over_count"] = int(actionable_market_type_counts.get("total_over", 0))
        diagnostics_out["actionable_total_under_count"] = int(actionable_market_type_counts.get("total_under", 0))
        diagnostics_out["actionable_side_count"] = int(sum(v for k, v in actionable_market_type_counts.items() if "spread" in str(k)))
        diagnostics_out["actionable_mlb_total_over_count"] = int(((actionable_df["league"].astype(str).str.upper() == "MLB") & (actionable_df["market_type"].astype(str).str.lower() == "total_over")).sum())
        diagnostics_out["total_over_concentration_flag"] = bool(concentration_guard_active)
        diagnostics_out["total_over_concentration_reason"] = ",".join(concentration_reasons) if concentration_reasons else "none"
        diagnostics_out["degraded_feature_kelly_guard_active"] = bool(degraded_run)
        diagnostics_out["degraded_feature_kelly_multiplier_applied"] = float(DEGRADED_FEATURE_KELLY_MULTIPLIER) if degraded_run else 1.0
        diagnostics_out["degraded_feature_slate_cap_used"] = float(DEGRADED_FEATURE_MAX_SLATE_EXPOSURE_PCT) if degraded_run else 0.0
        diagnostics_out["degraded_feature_pick_cap_used"] = float(DEGRADED_FEATURE_MAX_PICK_EXPOSURE_PCT) if degraded_run else 0.0

        # Add requested metrics directly to diagnostics_out
        diagnostics_out["actionable_counts_by_league"] = actionable_counts_by_league
        diagnostics_out["actionable_counts_by_market_type"] = actionable_market_type_counts
        diagnostics_out["actionable_counts_by_family"] = actionable_family_counts
        if "league" in actionable_df.columns and "market_type" in actionable_df.columns:
            actionable_league_family = actionable_df.copy()
            actionable_league_family["market_family"] = actionable_league_family["market_type"].astype(str).str.lower().map(
                lambda mt: "over" if mt == "total_over" else ("under" if mt == "total_under" else "side")
            )
            diagnostics_out["actionable_counts_by_league_family"] = (
                actionable_league_family.groupby(["league", "market_family"]).size().to_dict()
            )
        else:
            diagnostics_out["actionable_counts_by_league_family"] = {}
        diagnostics_out["actionable_totals_below_floor"] = totals_below_prob_floor
        diagnostics_out["nhl_totals_actionable"] = nhl_totals_actionable
        diagnostics_out["spreads_downgraded_by_divergence"] = spreads_downgraded_by_divergence
        diagnostics_out["spreads_rescued_by_divergence"] = spreads_rescued_by_divergence

        # Inject new metrics
        diagnostics_out["blocked_by_cold_market"] = blocked_by_cold_market
        diagnostics_out["blocked_by_fallback_heavy_totals"] = blocked_by_fallback_heavy_totals
        diagnostics_out["blocked_by_total_under"] = blocked_by_total_under
        diagnostics_out["blocked_by_nhl_total"] = blocked_by_nhl_total
        diagnostics_out["actionable_counts_by_league_market"] = actionable_counts_by_league_market
        diagnostics_out["below_threshold_counts_by_league_market"] = below_threshold_counts_by_league_market
        diagnostics_out["actionable_counts_by_consensus"] = actionable_counts_by_consensus
        diagnostics_out["downgraded_by_neutral"] = downgraded_by_neutral
        diagnostics_out["downgraded_by_disagrees"] = downgraded_by_disagrees
        diagnostics_out["side_floor_failures"] = side_floor_failures
        diagnostics_out["blocked_by_under_specific_thresholds"] = blocked_by_under_specific_thresholds
        diagnostics_out["blocked_by_nba_total_penalty"] = blocked_by_nba_total_penalty
        diagnostics_out["blocked_by_no_kalshi_total_penalty"] = blocked_by_no_kalshi_total_penalty
        diagnostics_out["blocked_by_mlb_spread_penalty"] = blocked_by_mlb_spread_penalty
        diagnostics_out["blocked_by_mlb_over_promotion_gate"] = blocked_by_mlb_over_promotion_gate
        diagnostics_out["promoted_by_nba_side_bonus"] = promoted_by_nba_side_bonus
        diagnostics_out["promoted_by_nba_over_bonus"] = promoted_by_nba_over_bonus
        diagnostics_out["demoted_by_mlb_spread_finalist_score_penalty"] = demoted_by_mlb_spread_finalist_penalty
        diagnostics_out["blocked_by_suspicious_data"] = int(final_best_df.get("status_blocker_stage", pd.Series([], dtype=str)).astype(str).eq("suspicious_data_guardrail").sum())
        diagnostics_out["suspicious_data_flag_rows"] = int(final_best_df.get("suspicious_data_flag", pd.Series([], dtype=bool)).fillna(False).astype(bool).sum())
        diagnostics_out["divergence_rows_preserved"] = int(divergence_rows_preserved)
        diagnostics_out["divergence_rows_blocked_by_viability_floor"] = int(divergence_rows_blocked_by_viability_floor)
        diagnostics_out["divergence_rows_negative_ev"] = int(divergence_rows_negative_ev)
        diagnostics_out["divergence_rows_negative_edge"] = int(divergence_rows_negative_edge)
        divergence_stage_mask = final_best_df.get("status_blocker_stage", pd.Series([], dtype=str)).astype(str).isin({"divergence_guardrail", "divergence_viability_floor"})
        diagnostics_out["divergence_rows_by_pick_status"] = (
            final_best_df.loc[divergence_stage_mask, "Pick_Status"].value_counts().to_dict()
            if "Pick_Status" in final_best_df.columns
            else {}
        )
        diagnostics_out["high_variance_due_only_high_ev"] = int(high_variance_due_only_high_ev)
        diagnostics_out["promoted_high_ev_to_actionable_no_uncertainty"] = int(promoted_high_ev_to_actionable_no_uncertainty)
        diagnostics_out["high_variance_capped_due_to_divergence"] = int(high_variance_capped_due_to_divergence)
        diagnostics_out["high_variance_capped_due_to_no_kalshi"] = int(high_variance_capped_due_to_no_kalshi)
        diagnostics_out["high_variance_capped_due_to_suspicious_data"] = int(high_variance_capped_due_to_suspicious_data)
        diagnostics_out["high_variance_capped_due_to_degraded_subset"] = int(high_variance_capped_due_to_degraded_subset)
        diagnostics_out["high_variance_capped_due_to_fallback_heavy"] = int(high_variance_capped_due_to_fallback_heavy)
        diagnostics_out["side_balance_promotions"] = int(side_balance_promotions)
        diagnostics_out["actionable_family_counts"] = actionable_family_counts
        diagnostics_out["totals_only_actionable_flag"] = bool(totals_only_actionable_flag)
        diagnostics_out["viable_side_candidates_count"] = int(viable_side_candidates_count)
        diagnostics_out["side_promoted_by_balance_guard_count"] = int(side_balance_promotions)
        diagnostics_out["side_balance_guard_reason"] = side_balance_guard_reason
        diagnostics_out["final_pick_status_counts"] = final_best_df["Pick_Status"].value_counts().to_dict()
        diagnostics_out["negative_ev_final_guardrail_count"] = int(negative_ev_mask.sum())
        diagnostics_out["negative_edge_final_guardrail_count"] = int(negative_edge_mask.sum())
        diagnostics_out["negative_ev_high_variance_downgraded_count"] = int(downgraded_from_high_variance_mask.sum())
        diagnostics_out["negative_ev_actionable_downgraded_count"] = int(downgraded_from_actionable_mask.sum())
        hidden_bad_rows = final_best_df[
            (final_best_df["Pick_Status"] == "High Variance/Speculative")
            & (
                (pd.to_numeric(final_best_df["expected_value"], errors="coerce") <= 0)
                | (pd.to_numeric(final_best_df["edge"], errors="coerce") <= 0)
            )
        ]
        diagnostics_out["high_variance_non_positive_ev_edge_rows"] = hidden_bad_rows[
            ["league", "home_team", "away_team", "market_type", "Pick_Status", "Status_Reason", "expected_value", "edge"]
        ].to_dict("records")
        diagnostics_out["final_actionable_count"] = len(actionable_df)

        current_card_df = final_best_df
        overs_sides_df = final_best_df[~final_best_df["market_type"].astype(str).str.lower().eq("total_under")]
        no_unders_df = final_best_df[~final_best_df["market_type"].astype(str).str.lower().eq("total_under")]
        no_nba_totals_df = final_best_df[~(
            final_best_df["league"].astype(str).str.upper().eq("NBA")
            & final_best_df["market_type"].astype(str).str.lower().str.contains("total", na=False)
        )]
        no_kalshi_totals_df = final_best_df[~(
            final_best_df["consensus_agreement"].astype(str).eq("No Kalshi")
            & final_best_df["market_type"].astype(str).str.lower().str.contains("total", na=False)
        )]
        diagnostics_out["shadow_card_counts"] = {
            "current_card": int(len(current_card_df)),
            "overs_only_plus_sides_card": int(len(overs_sides_df)),
            "no_unders_card": int(len(no_unders_df)),
            "no_nba_totals_card": int(len(no_nba_totals_df)),
            "no_kalshi_totals_card": int(len(no_kalshi_totals_df)),
        }

        if "market_type" in final_best_df.columns:
            totals_mask = final_best_df["market_type"].astype(str).str.contains("total", case=False, na=False)
            actionable_totals_df = final_best_df[(final_best_df["Pick_Status"] == "Actionable") & totals_mask]
            diagnostics_out["actionable_totals_by_league"] = actionable_totals_df["league"].value_counts().to_dict() if "league" in actionable_totals_df.columns else {}

            is_fallback_heavy = diagnostics_out.get("is_fallback_heavy", False) if diagnostics_out else False
            if is_fallback_heavy:
                diagnostics_out["ev_dampener_impact_count"] = len(final_best_df[totals_mask & (final_best_df["expected_value"] > 0)])
            else:
                diagnostics_out["ev_dampener_impact_count"] = 0

            diagnostics_out["totals_rejected_by_new_guardrails"] = int(final_best_df["Status_Reason"].str.contains("Fails minimum Win Probability for.*Totals").sum())

        diagnostics_out["selection_diagnostics"] = {
            "raw_family_counts": raw_counts,
            "raw_market_type_counts": raw_market_type_counts,
            "finalist_family_counts": finalist_counts,
            "finalist_market_type_counts": finalist_market_type_counts,
            "final_family_counts": final_counts,
            "final_market_type_counts": final_market_type_counts,
            "actionable_family_counts": actionable_family_counts,
            "actionable_market_type_counts": actionable_market_type_counts,
            "avg_scores": avg_scores,
            "preview_df": preview_df,
        }

    return final_best_df


def fetch_live_odds_dataframe(sports: list[str] | None = None, date: str | None = None) -> pd.DataFrame:
    from app_core.odds_api import TheOddsAPIClient, filter_games_today_only
    import pandas as pd

    def _sanitize(name):
        # Standardize "St." or "St " to "saint " before stripping
        clean_name = str(name).lower().replace("st. ", "saint ").replace("st ", "saint ")
        return "".join(e for e in clean_name if e.isalnum())

    api_key = _get_odds_api_key()
    if not api_key:
        logger.error("No ODDS_API_KEY found.")
        return pd.DataFrame()

    client = TheOddsAPIClient(api_key=api_key)

    sport_keys = []
    if sports:
        for s in sports:
            s_up = s.upper()
            if s_up in ["NCAAB", "NCAAM", "NCAA MEN'S BASKETBALL"]:
                sport_keys.append("basketball_ncaab")
            elif s_up == "NBA":
                sport_keys.append("basketball_nba")
            elif s_up == "NHL":
                sport_keys.append("icehockey_nhl")
            elif s_up == "MLB":
                sport_keys.append("baseball_mlb_preseason")
                sport_keys.append("baseball_mlb")
    else:
        sport_keys = ["basketball_ncaab", "basketball_nba", "icehockey_nhl", "baseball_mlb"]

    game_dict = {}
    for sk in sport_keys:
        try:
            games = client.get_odds(sk, date=date)
            if not games:
                continue
            if isinstance(games, dict) and "message" in games:
                logger.error(f"Odds API error for {sk}: {games.get('message')}")
                continue

            sport_games = filter_games_today_only(games)
            if not sport_games:
                continue

            for game in sport_games:
                matchup_id = game.get('matchup_id')
                if not matchup_id:
                    continue

                if matchup_id not in game_dict:
                    game_dict[matchup_id] = {
                        'game_id': game.get('id'),
                        'league': game.get('sport_key', '').split('_')[-1].upper(),
                        'raw_home_team': game.get('home_team'),
                        'raw_away_team': game.get('away_team'),
                        'home_team': game.get('home_team'),
                        'away_team': game.get('away_team'),
                        'commence_time': game.get('commence_time'),
                        'commence_time_raw': game.get('commence_time'),
                        'matchup_id': matchup_id,
                    }

                row = game_dict[matchup_id]

                for book in game.get('bookmakers', []):
                    book_key = book.get('key', '')
                    if book_key not in ['novig', 'fanduel', 'draftkings', 'betmgm']:
                        continue

                    for market in book.get('markets', []):
                        if market.get('key') == 'spreads':
                            for o in market.get('outcomes', []):
                                if _sanitize(o.get('name')) == _sanitize(game.get('home_team')):
                                    row[f'{book_key}_home_point'] = o.get('point')
                                    row[f'{book_key}_home_price'] = o.get('price')
                                elif _sanitize(o.get('name')) == _sanitize(game.get('away_team')):
                                    row[f'{book_key}_away_point'] = o.get('point')
                                    row[f'{book_key}_away_price'] = o.get('price')
                        elif market.get('key') == 'totals':
                            for o in market.get('outcomes', []):
                                if str(o.get('name')).lower() == 'over':
                                    row[f'{book_key}_over_point'] = o.get('point')
                                    row[f'{book_key}_over_price'] = o.get('price')
                                elif str(o.get('name')).lower() == 'under':
                                    row[f'{book_key}_under_point'] = o.get('point')
                                    row[f'{book_key}_under_price'] = o.get('price')
                        elif market.get('key') == 'h2h':
                            for o in market.get('outcomes', []):
                                if _sanitize(o.get('name')) == _sanitize(game.get('home_team')):
                                    row[f'{book_key}_h2h_home_price'] = o.get('price')
                                elif _sanitize(o.get('name')) == _sanitize(game.get('away_team')):
                                    row[f'{book_key}_h2h_away_price'] = o.get('price')

        except Exception as e:
            logger.error(f"Network/API failure for {sk}: {e}")
            continue

    if not game_dict:
        return pd.DataFrame()

    return pd.DataFrame(list(game_dict.values()))

def _fmt_odds_token(v):
    """Format a spread point or moneyline price for the raw-odds diagnostic string:
    signed (+1.5, -1.5, -120, +115); '·' for missing values."""
    n = pd.to_numeric(v, errors="coerce")
    if pd.isna(n):
        return "·"
    return f"{float(n):+g}"


def _raw_book_odds_diag(row):
    """One verbatim string of every book's spread points and moneyline prices, e.g.
    ``novig: sp H=-1.5/A=+1.5 ml H=-120/A=+115 | fanduel: ...``. Books with no data are
    omitted. Lets a flipped-orientation case be diagnosed straight from the export."""
    parts = []
    for bk in ("novig", "fanduel", "draftkings", "betmgm"):
        hp, ap = row.get(f"{bk}_home_point"), row.get(f"{bk}_away_point")
        hml, aml = row.get(f"{bk}_h2h_home_price"), row.get(f"{bk}_h2h_away_price")
        if all(pd.isna(pd.to_numeric(v, errors="coerce")) for v in (hp, ap, hml, aml)):
            continue
        parts.append(
            f"{bk}: sp H={_fmt_odds_token(hp)}/A={_fmt_odds_token(ap)} "
            f"ml H={_fmt_odds_token(hml)}/A={_fmt_odds_token(aml)}"
        )
    return " | ".join(parts) if parts else pd.NA


def _derive_spread_away_line(row):
    """Return the away team's run line, robust to the live feed's sign quirks.

    A point spread is an exact mirror: ``away_line == -home_line``. Novig is a
    peer-to-peer exchange that lists BOTH spread outcomes under the home team's
    signed point (e.g. both show -1.5 on a HOU -1.5 market), so ``novig_away_point``
    carries the WRONG sign for the away team and must never be trusted directly
    (this is what produced "Cleveland -1.5" when Cleveland was really +1.5).

    Derivation order, safest first:
      1. Negate the first available home point across books (the home point is
         unambiguously signed for the home team, so -home is the away line).
      2. If no book reports a home point, fall back to a STANDARD book's away
         point — fanduel/draftkings/betmgm sign per-team correctly (unlike the
         Novig P2P convention).
      3. Otherwise NA, so the pick is dropped rather than emitted with a guessed
         sign. (The spread_orientation guardrail remains the final backstop.)
    """
    for bk in ("novig", "fanduel", "draftkings", "betmgm"):
        hp = pd.to_numeric(row.get(f"{bk}_home_point"), errors="coerce")
        if pd.notna(hp):
            return -float(hp)
    for bk in ("fanduel", "draftkings", "betmgm"):
        ap = pd.to_numeric(row.get(f"{bk}_away_point"), errors="coerce")
        if pd.notna(ap):
            return float(ap)
    return pd.NA


def _expand_live_odds_to_bet_rows(live_odds_df: pd.DataFrame, theover_rows: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Expands the wide live_odds_df (1 row per game) into up to 4 market rows per game
    (e.g., spread_home, spread_away, total_over, total_under) and then filters them
    based on the user's uploads in `theover_rows`. If no match, retains all candidates.

    Returns a tuple of (expanded_df, diag_counts).
    """
    if live_odds_df is None or live_odds_df.empty:
        return pd.DataFrame(), {}

    out_rows = []

    # Required identity columns
    id_cols = ["league", "home_team", "away_team", "game_date", "matchup_id", "commence_time_raw"]
    # Check for game_time_est if exists
    if "game_time_est" in live_odds_df.columns:
        id_cols.append("game_time_est")

    has_theover = theover_rows is not None and not theover_rows.empty and "market_type" in theover_rows.columns

    # Diagnostic counters
    diag_counts = {
        "generated": {"spread_home": 0, "spread_away": 0, "total_over": 0, "total_under": 0},
        "filtered": {"spread_home": 0, "spread_away": 0, "total_over": 0, "total_under": 0},
        "games_unmatched": 0,
        "games_matched_exact": 0,
        "games_matched_canonical": 0,
        "games_matched_fuzzy": 0,
        "rows_retained_unmatched": 0,
        "rows_dropped_by_join": 0,
        "upload_matched_rows": 0,
        "upload_matched_drifted_rows": 0,
        "absolute_line_drifts": [],
        "drift_breakdown": {"total_over": 0, "total_under": 0, "spread_home": 0, "spread_away": 0},
    }

    # Normalize dates and canonical keys ahead of the loop for efficiency and robustness
    if has_theover:
        theover_rows_with_canon = theover_rows.copy()
        # Use existing date normalization helper to ensure ET day comparison
        theover_rows_with_canon["_et_day"] = _et_day_string(_game_dates(theover_rows_with_canon))
        theover_rows_with_canon["_canon_key"] = _canonical_matchup_key(theover_rows_with_canon)

        canon_map = {}
        for (canon_key, et_date), group in theover_rows_with_canon.groupby(["_canon_key", "_et_day"]):
            canon_map[(canon_key, et_date)] = group["market_type"].tolist()

        # Group by league and normalized ET date for fuzzy matching
        fuzzy_pool = {}
        for (league, et_date), group in theover_rows_with_canon.groupby([
            theover_rows_with_canon["league"].astype(str).str.upper(),
            theover_rows_with_canon["_et_day"]
        ]):
            fuzzy_pool[(league, et_date)] = group

    # Pre-process live odds dates and canonical keys vectorially
    live_odds_processed = live_odds_df.copy()
    live_odds_processed["_et_day"] = _et_day_string(_game_dates(live_odds_processed))
    live_odds_processed["_canon_key"] = _canonical_matchup_key(live_odds_processed)

    for idx, row in live_odds_processed.iterrows():
        base_dict = {col: row.get(col) for col in id_cols}
        # Carry the game's moneyline (h2h) prices onto every bet row so spread rows
        # can be checked for a flipped home/away orientation downstream (the spread
        # favorite must be the moneyline favorite). First book that priced the h2h.
        for _ml_side in ("home", "away"):
            _ml_price = pd.NA
            for _ml_book in ("novig", "fanduel", "draftkings", "betmgm"):
                _cand_ml = row.get(f"{_ml_book}_h2h_{_ml_side}_price")
                if pd.notna(_cand_ml):
                    _ml_price = _cand_ml
                    break
            base_dict[f"game_{_ml_side}_ml_price"] = _ml_price
        # Verbatim per-book spread points + moneyline prices, so a flipped spread
        # orientation can be diagnosed from the export (source corruption vs parse bug).
        base_dict["raw_book_odds_diag"] = _raw_book_odds_diag(row)
        matchup_id = row.get("matchup_id")
        live_canon_key = row.get("_canon_key")

        league_str = str(row.get("league", "")).upper()
        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        game_date = str(row.get("game_date", ""))
        et_date = str(row.get("_et_day", ""))

        # All leagues use spreads and totals
        candidate_markets = ["spread_home", "spread_away", "total_over", "total_under"]

        for m in candidate_markets:
            diag_counts["generated"][m] += 1

        market_mappings = {
            "spread_home": ("home_price", "home_point"),
            "spread_away": ("away_price", "away_point"),
            "total_over": ("over_price", "over_point"),
            "total_under": ("under_price", "under_point"),
        }

        # Check for matches in the uploaded file
        match_found = False
        orientation_source = "default"
        target_markets = []
        upload_match_reason = ""

        if has_theover and matchup_id:
            # 1. Try exact matchup_id match
            matchup_mask = theover_rows["matchup_id"] == matchup_id
            if matchup_mask.any():
                match_found = True
                orientation_source = "exact_match"
                target_markets = theover_rows.loc[matchup_mask, "market_type"].tolist()
                upload_match_reason = "Matched by exact matchup_id"
            else:
                # 2. Try canonical match (Sorted normalized teams + normalized league + ET Day)
                if (live_canon_key, et_date) in canon_map:
                    match_found = True
                    orientation_source = "canonical_match"
                    target_markets = canon_map[(live_canon_key, et_date)]
                    upload_match_reason = "Matched by canonical matchup key"

                if not match_found:
                    # 3. Try fuzzy match (Same league + ET Day, similarity >= 85)
                    pool = fuzzy_pool.get((league_str, et_date))
                    if pool is not None and not pool.empty:
                        best_score = -1
                        best_match_idx = None

                        # We use _fuzzy_match_schedule_row logic manually to find the best match within the pool
                        # We need to test against all unique match-ups in the pool
                        unique_matchups = pool.drop_duplicates(["home_team", "away_team"])

                        for idx, cand in unique_matchups.iterrows():
                            c_home = str(cand.get("home_team", ""))
                            c_away = str(cand.get("away_team", ""))

                            direct_home = _team_similarity_score(home_team, c_home)
                            direct_away = _team_similarity_score(away_team, c_away)
                            direct_min = min(direct_home, direct_away)

                            rev_home = _team_similarity_score(home_team, c_away)
                            rev_away = _team_similarity_score(away_team, c_home)
                            rev_min = min(rev_home, rev_away)

                            cand_score = max(direct_min, rev_min)
                            if cand_score > best_score:
                                best_score = cand_score
                                best_match_idx = idx

                        if best_score >= 85 and best_match_idx is not None:
                            match_found = True
                            orientation_source = "fuzzy_match"
                            best_match_home = pool.loc[best_match_idx, "home_team"]
                            best_match_away = pool.loc[best_match_idx, "away_team"]
                            matched_rows = pool[(pool["home_team"] == best_match_home) & (pool["away_team"] == best_match_away)]
                            target_markets = matched_rows["market_type"].tolist()
                            upload_match_reason = f"Matched by fuzzy team similarity (score={best_score})"
                        else:
                            upload_match_reason = "No fuzzy match > 85 similarity found"
                    else:
                        upload_match_reason = "Missing from upload (no games for this league/date)"
        elif not has_theover:
             upload_match_reason = "No uploaded theover_rows available"
        else:
             upload_match_reason = "Missing matchup_id for live game"

        if match_found:
            candidate_source = "upload_matched"
            if orientation_source == "exact_match":
                diag_counts["games_matched_exact"] += 1
            elif orientation_source == "canonical_match":
                diag_counts["games_matched_canonical"] += 1
            elif orientation_source == "fuzzy_match":
                diag_counts["games_matched_fuzzy"] += 1
        else:
            candidate_source = "live_unfiltered"
            diag_counts["games_unmatched"] += 1
            logger.info(f"Unmatched game '{home_team} vs {away_team}' ({league_str} {game_date}). Reason: {upload_match_reason}")
            if "unmatched_live_games" not in diag_counts:
                diag_counts["unmatched_live_games"] = []
            diag_counts["unmatched_live_games"].append({
                "league": league_str, "home_team": home_team, "away_team": away_team,
                "game_date": game_date, "reason": upload_match_reason
            })

        # Determine the matched group if match_found
        matched_group = None
        if match_found:
            if orientation_source == "exact_match":
                matched_group = theover_rows[matchup_mask]
            elif orientation_source == "canonical_match":
                matched_group = theover_rows_with_canon[(theover_rows_with_canon["_canon_key"] == live_canon_key) & (theover_rows_with_canon["_et_day"] == et_date)]
            elif orientation_source == "fuzzy_match":
                matched_group = matched_rows


        for market_type in candidate_markets:
            if match_found and market_type not in target_markets:
                diag_counts["rows_dropped_by_join"] += 1
                continue # Skip markets not in the uploaded target set if we found a match

            diag_counts["filtered"][market_type] += 1
            if not match_found:
                diag_counts["rows_retained_unmatched"] += 1

            price_suffix, point_suffix = market_mappings[market_type]
            market_dict = base_dict.copy()
            market_dict["market_type"] = market_type
            market_dict["candidate_source"] = candidate_source
            market_dict["orientation_source"] = orientation_source
            market_dict["upload_match_reason"] = upload_match_reason

            # Map pricing for novig (primary)
            novig_price_col = f"novig_{price_suffix}"
            novig_point_col = f"novig_{point_suffix}" if point_suffix else None

            price_val = pd.to_numeric(row.get(novig_price_col), errors="coerce")
            if pd.isna(price_val):
                market_dict["odds_american"] = -110.0
                market_dict["odds_source"] = "fallback_novig"
            else:
                market_dict["odds_american"] = float(price_val)
                market_dict["odds_source"] = "odds_api"

            # Map pricing for other bookmakers
            for book_key in ["fanduel", "draftkings", "betmgm"]:
                book_price_col = f"{book_key}_{price_suffix}"
                book_price_val = pd.to_numeric(row.get(book_price_col), errors="coerce")
                if pd.notna(book_price_val):
                    market_dict[f"odds_american_{book_key}"] = float(book_price_val)
                else:
                    market_dict[f"odds_american_{book_key}"] = pd.NA

            # Map lines based on market type
            if novig_point_col:
                point_val = pd.to_numeric(row.get(novig_point_col), errors="coerce")
            else:
                point_val = pd.NA

            if market_type.startswith("spread"):
                if market_type == "spread_away":
                    # away_line = -home_line; derived robustly across books because
                    # Novig's away_point carries the home-signed value. See
                    # _derive_spread_away_line for the full rationale.
                    point_val = _derive_spread_away_line(row)
                market_dict["spread_line"] = float(point_val) if pd.notna(point_val) else pd.NA
                market_dict["total_line"] = pd.NA
                market_dict["live_spread_line"] = market_dict["spread_line"]
                market_dict["live_total_line"] = pd.NA
            elif market_type.startswith("total"):
                market_dict["spread_line"] = pd.NA
                market_dict["total_line"] = float(point_val) if pd.notna(point_val) else pd.NA
                market_dict["live_spread_line"] = pd.NA
                market_dict["live_total_line"] = market_dict["total_line"]
            else:
                market_dict["spread_line"] = pd.NA
                market_dict["total_line"] = pd.NA
                market_dict["live_spread_line"] = pd.NA
                market_dict["live_total_line"] = pd.NA

            market_dict["uploaded_spread_line"] = pd.NA
            market_dict["uploaded_total_line"] = pd.NA
            market_dict["line_source"] = "live_odds"
            market_dict["line_delta"] = pd.NA
            market_dict["upload_market_match"] = False

            if match_found and matched_group is not None:
                market_row = matched_group[matched_group["market_type"] == market_type]
                if not market_row.empty:
                    market_dict["upload_market_match"] = True
                    diag_counts["upload_matched_rows"] += 1
                    u_spread = market_row.iloc[0].get("spread_line")
                    u_total = market_row.iloc[0].get("total_line")

                    if market_type.startswith("spread"):
                        market_dict["uploaded_spread_line"] = float(u_spread) if pd.notna(u_spread) else pd.NA
                        if pd.notna(market_dict["live_spread_line"]) and pd.notna(market_dict["uploaded_spread_line"]):
                            delta = market_dict["live_spread_line"] - market_dict["uploaded_spread_line"]
                            market_dict["line_delta"] = delta
                            if delta != 0.0:
                                diag_counts["upload_matched_drifted_rows"] += 1
                                diag_counts["absolute_line_drifts"].append(abs(delta))
                                if market_type in diag_counts["drift_breakdown"]:
                                    diag_counts["drift_breakdown"][market_type] += 1

                        from app_core import weights_config
                        if getattr(weights_config, 'LOCK_UPLOAD_LINES_FOR_MATCHED_ROWS', False) and pd.notna(market_dict["uploaded_spread_line"]):
                            market_dict["spread_line"] = market_dict["uploaded_spread_line"]
                            market_dict["line_source"] = "uploaded_theover"

                    elif market_type.startswith("total"):
                        market_dict["uploaded_total_line"] = float(u_total) if pd.notna(u_total) else pd.NA
                        if pd.notna(market_dict["live_total_line"]) and pd.notna(market_dict["uploaded_total_line"]):
                            delta = market_dict["live_total_line"] - market_dict["uploaded_total_line"]
                            market_dict["line_delta"] = delta
                            if delta != 0.0:
                                diag_counts["upload_matched_drifted_rows"] += 1
                                diag_counts["absolute_line_drifts"].append(abs(delta))
                                if market_type in diag_counts["drift_breakdown"]:
                                    diag_counts["drift_breakdown"][market_type] += 1

                        from app_core import weights_config
                        if getattr(weights_config, 'LOCK_UPLOAD_LINES_FOR_MATCHED_ROWS', False) and pd.notna(market_dict["uploaded_total_line"]):
                            market_dict["total_line"] = market_dict["uploaded_total_line"]
                            market_dict["line_source"] = "uploaded_theover"

            out_rows.append(market_dict)

    # Calculate drift metrics
    drift_mean = sum(diag_counts["absolute_line_drifts"]) / len(diag_counts["absolute_line_drifts"]) if diag_counts["absolute_line_drifts"] else 0.0
    drift_max = max(diag_counts["absolute_line_drifts"]) if diag_counts["absolute_line_drifts"] else 0.0
    diag_counts["drift_mean"] = drift_mean
    diag_counts["drift_max"] = drift_max

    logger.info("=== CANDIDATE GENERATION DIAGNOSTICS ===")
    logger.info(f"Games matched exact: {diag_counts['games_matched_exact']}")
    logger.info(f"Upload Matched Rows: {diag_counts['upload_matched_rows']}")
    logger.info(f"Upload Matched Drifted Rows: {diag_counts['upload_matched_drifted_rows']}")
    logger.info(f"Average Absolute Drift: {drift_mean:.2f}")
    logger.info(f"Max Absolute Drift: {drift_max:.2f}")
    logger.info(f"Drift Breakdown: {diag_counts['drift_breakdown']}")
    logger.info(f"Games matched canonical: {diag_counts['games_matched_canonical']}")
    logger.info(f"Games matched fuzzy: {diag_counts['games_matched_fuzzy']}")
    logger.info(f"Games unmatched (kept all candidates): {diag_counts['games_unmatched']}")
    logger.info(f"Generated candidates: {diag_counts['generated']}")
    logger.info(f"Rows dropped by join: {diag_counts['rows_dropped_by_join']}")
    logger.info(f"Rows retained (unmatched): {diag_counts['rows_retained_unmatched']}")
    logger.info(f"Filtered (final) candidates: {diag_counts['filtered']}")

    # Identify uploaded games that were missed entirely
    if theover_rows is not None and not theover_rows.empty and "matchup_id" in theover_rows.columns:
        uploaded_games = set(theover_rows["matchup_id"].unique())
        matched_games = set()
        for row in out_rows:
            if row.get("candidate_source") == "upload_matched":
                matched_games.add(row.get("matchup_id"))
        missing_uploads = uploaded_games - matched_games
        diag_counts["missing_uploaded_games"] = list(missing_uploads)
        logger.info(f"Missing uploaded games count: {len(missing_uploads)}")
        if missing_uploads:
            logger.info(f"Missing uploaded games IDs: {list(missing_uploads)[:10]}")

    logger.info("========================================")

    expanded_df = pd.DataFrame(out_rows)
    return expanded_df, diag_counts

def _compute_ml_input_flatness_diagnostics(
    enriched_df: pd.DataFrame,
    eligible_mask: pd.Series | None = None,
) -> dict[str, Any]:
    if enriched_df is None or enriched_df.empty:
        return {
            "ml_input_row_count": 0,
            "ml_feature_eligible_row_count": 0,
            "ml_rows_excluded_count": 0,
            "ml_zero_variance_feature_count": 0,
            "ml_near_constant_feature_count": 0,
            "ml_high_missingness_feature_count": 0,
            "ml_top_missing_features": {},
            "ml_top_feature_nunique": {},
            "ml_flatness_root_cause_hint": "no_rows",
        }

    feature_cols = [c for c in enriched_df.columns if c.startswith("feature_")]
    feature_df = enriched_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=enriched_df.index)
    row_count = int(len(enriched_df))
    if eligible_mask is None:
        eligible_mask = pd.Series([True] * row_count, index=enriched_df.index, dtype=bool)
    else:
        eligible_mask = eligible_mask.reindex(enriched_df.index).fillna(False).astype(bool)
    eligible_count = int(eligible_mask.sum())
    excluded_count = int((~eligible_mask).sum())

    if feature_df.empty:
        return {
            "ml_input_row_count": row_count,
            "ml_feature_eligible_row_count": eligible_count,
            "ml_rows_excluded_count": excluded_count,
            "ml_zero_variance_feature_count": 0,
            "ml_near_constant_feature_count": 0,
            "ml_high_missingness_feature_count": 0,
            "ml_top_missing_features": {},
            "ml_top_feature_nunique": {},
            "ml_flatness_root_cause_hint": "no_feature_columns",
        }

    missingness = feature_df.isna().mean()
    nunique = feature_df.nunique(dropna=True)
    zero_variance_count = int((nunique <= 1).sum())
    near_constant_count = int((nunique <= 2).sum())
    high_missing_count = int((missingness >= 0.5).sum())

    top_missing = missingness.sort_values(ascending=False).head(5)
    top_nunique = nunique.sort_values().head(8)

    if eligible_count < 8:
        root_cause = "too_few_ml_eligible_rows"
    elif high_missing_count >= max(3, int(len(feature_df.columns) * 0.25)):
        root_cause = "high_feature_missingness"
    elif zero_variance_count >= max(4, int(len(feature_df.columns) * 0.25)):
        root_cause = "many_zero_variance_features"
    elif near_constant_count >= max(6, int(len(feature_df.columns) * 0.35)):
        root_cause = "many_near_constant_features"
    elif excluded_count > 0 and (excluded_count / max(row_count, 1)) >= 0.25:
        root_cause = "high_stats_exclusion_rate"
    else:
        root_cause = "mixed_or_model_specific"

    return {
        "ml_input_row_count": row_count,
        "ml_feature_eligible_row_count": eligible_count,
        "ml_rows_excluded_count": excluded_count,
        "ml_zero_variance_feature_count": zero_variance_count,
        "ml_near_constant_feature_count": near_constant_count,
        "ml_high_missingness_feature_count": high_missing_count,
        "ml_top_missing_features": {k: round(float(v), 3) for k, v in top_missing.items()},
        "ml_top_feature_nunique": {k: int(v) for k, v in top_nunique.items()},
        "ml_flatness_root_cause_hint": root_cause,
    }


def run_analysis_pipeline(
    sports: list[str] | None = None,
    max_rows: int = 1000,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:

    # 1. Build the enrichment frame (TheOver) BEFORE expanding the Master Slate
    theover_rows = build_theover_bet_rows(spreads_df, totals_df, sports)

    raw_base_df = load_base_data()
    odds_schedule_loaded = not raw_base_df.empty

    # Ensure identity string dtypes on theover_rows
    theover_rows = _infer_missing_league_from_team_sets(theover_rows, sports)
    theover_rows = _restore_missing_ncaab_league_priority(theover_rows)
    theover_rows = _recover_ncaab_league_labels(theover_rows)
    theover_rows = _enforce_identity_string_dtype(theover_rows, ["league", "home_team", "away_team"])
    theover_rows = _preprocess_bet_rows_for_league_bridge(theover_rows)
    theover_rows = _normalize_identity_strings(theover_rows, ["league", "home_team", "away_team"])

    stale = is_stale_schedule(raw_base_df, theover_rows)
    base_df = raw_base_df.copy()

    # Drop stale odds columns to prevent leak into live odds merge
    leak_cols = ['odds_american', 'odds_home', 'odds_away', 'odds_source', 'market_probability', 'implied_home_prob', 'decimal_odds']
    base_df = base_df.drop(columns=[c for c in leak_cols if c in base_df.columns], errors='ignore')

    stale_base_rows_removed = 0

    theover_rows = _enforce_identity_string_dtype(theover_rows, ["league", "home_team", "away_team"])
    theover_rows = _normalize_identity_strings(theover_rows, ["league", "home_team", "away_team"])
    theover_rows["league"] = _string_series(theover_rows, "league").str.upper().replace(LEAGUE_ALIASES)
    theover_rows["home_team"] = _string_series(theover_rows, "home_team").map(normalize_team_name)
    theover_rows["away_team"] = _string_series(theover_rows, "away_team").map(normalize_team_name)
    theover_rows["game_date"] = _et_day_string(_game_dates(theover_rows))
    theover_rows["matchup_id"] = _matchup_id(theover_rows)

    if not theover_rows.empty and not base_df.empty:
        base_dates = base_df.copy()
        base_dates = _normalize_identity_strings(base_dates, ["league", "home_team", "away_team"])
        base_dates["league"] = _string_series(base_dates, "league").str.upper().replace(LEAGUE_ALIASES)
        base_dates["home_team"] = _string_series(base_dates, "home_team").map(normalize_team_name)
        base_dates["away_team"] = _string_series(base_dates, "away_team").map(normalize_team_name)
        base_dates["date"] = _et_day_string(_game_dates(base_dates))
        base_dates["matchup_id"] = _matchup_id(base_dates)

        date_lookup = base_dates[["league", "matchup_id", "date"]].drop_duplicates(["league", "matchup_id"])
        merged_dates = theover_rows.merge(date_lookup, on=["league", "matchup_id"], how="left")
        theover_rows["game_date"] = theover_rows["game_date"].fillna(merged_dates["date"])

    theover_rows, date_stats = _fill_missing_game_dates_from_base(theover_rows, base_df)
    theover_rows = _dedupe_inverted_matchups(theover_rows)

    # 2. Expand TheOdds API into the Master Slate dynamically using theover_rows
    live_odds_df = fetch_live_odds_dataframe(sports)

    if not live_odds_df.empty:
        live_odds_df = _normalize_identity_strings(live_odds_df, ["league", "home_team", "away_team"])
        live_odds_df["league"] = _string_series(live_odds_df, "league").str.upper().replace(LEAGUE_ALIASES)
        live_odds_df["home_team"] = _string_series(live_odds_df, "home_team").map(normalize_team_name)
        live_odds_df["away_team"] = _string_series(live_odds_df, "away_team").map(normalize_team_name)

        fallback_day = _et_day_string(_game_dates(live_odds_df))
        today_et_day = pd.Series([_game_date_fallback().strftime("%Y-%m-%d")] * len(live_odds_df), index=live_odds_df.index, dtype="string")
        live_odds_df["game_date"] = _et_day_string(live_odds_df.get("game_date", pd.Series([pd.NA] * len(live_odds_df), index=live_odds_df.index)))
        live_odds_df["game_date"] = live_odds_df["game_date"].fillna(fallback_day).fillna(today_et_day)
        live_odds_df["matchup_id"] = _matchup_id(live_odds_df)

    master_slate, diag_counts = _expand_live_odds_to_bet_rows(live_odds_df, theover_rows)

    # Graceful fallback if Odds API fails or is empty: use theover_rows directly as the master slate.
    if master_slate.empty and not theover_rows.empty:
        logger.warning("Live odds expansion returned empty. Falling back to uploaded rows as master slate.")
        master_slate = theover_rows.copy()
        if "market_type" not in master_slate.columns:
             master_slate["market_type"] = pd.NA
        if "odds_american" not in master_slate.columns:
             master_slate["odds_american"] = -110.0
             master_slate["odds_source"] = "fallback_novig"

    if master_slate.empty:
        logger.warning("Master slate is empty after odds expansion. Falling back to an empty DataFrame.")
        master_slate = pd.DataFrame(columns=["league", "home_team", "away_team", "game_date", "matchup_id", "market_type", "odds_american", "odds_source"])

    # We removed the second `raw_base_df = load_base_data()` to avoid duplicating the load,
    # but still need `odds_schedule_loaded` since it's used at the very end.

    merge_keys = ["league", "home_team", "away_team", "game_date", "fuzzy_team_match>=85"]

    # Primary ingestion baseline: master_slate (from Odds API) is the master slate frame.
    merged = master_slate.copy()
    logger.info(f"PIPELINE AUDIT: [1/9] Total raw rows loaded into analysis (master_slate initialized): {len(merged)}")

    # 3. Invert the Merge (Odds API is Base, TheOver is Enrichment)
    if not theover_rows.empty and not merged.empty:
        # Standardize both sides of the merge to ET day boundaries before join.
        merged["game_date"] = pd.to_datetime(merged["game_date"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.floor("D")
        theover_rows["game_date"] = pd.to_datetime(theover_rows["game_date"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.floor("D")
        fallback_merge_day = pd.Timestamp.now(tz="America/New_York").floor("D")
        merged["game_date"] = merged["game_date"].fillna(fallback_merge_day)
        theover_rows["game_date"] = theover_rows["game_date"].fillna(fallback_merge_day)

        # Merge theover enrichment columns. win_prob_source rides along with
        # theover_probability so the selection stage can discount untrusted TheOver
        # sources (e.g. model_hit_rate_flipped) for the MLB total direction decision.
        theover_cols_to_merge = ["matchup_id", "market_type", "theover_probability", "ml_probability", "win_prob_source"]
        # Only merge columns that exist
        theover_cols_to_merge = [c for c in theover_cols_to_merge if c in theover_rows.columns]

        merged = merged.merge(
            theover_rows[theover_cols_to_merge].drop_duplicates(["matchup_id", "market_type"]),
            on=["matchup_id", "market_type"],
            how="left"
        )

        # 4. Fuzzy Matching Fallback
        # After the strict join, identify any rows in master_slate where theover_probability is still NaN.
        # Ensure we only try fuzzy match if theover_rows has probability columns
        if "theover_probability" in theover_rows.columns or "ml_probability" in theover_rows.columns:
            needs_fuzzy = pd.Series([False]*len(merged), index=merged.index)
            if "theover_probability" in merged.columns:
                needs_fuzzy = needs_fuzzy | merged["theover_probability"].isna()
            if "ml_probability" in merged.columns:
                needs_fuzzy = needs_fuzzy | merged["ml_probability"].isna()

            if needs_fuzzy.any():
                logger.info(f"Attempting fuzzy match for {needs_fuzzy.sum()} rows missing enrichment from TheOver.")
                theover_schedule = theover_rows.drop_duplicates(["league", "home_team", "away_team", "market_type"])

                for idx in merged.index[needs_fuzzy]:
                    row_market = merged.at[idx, "market_type"]
                    # We need to filter theover_schedule to the same market type
                    market_schedule = theover_schedule[theover_schedule["market_type"] == row_market]

                    if market_schedule.empty:
                        continue

                    match = _fuzzy_match_schedule_row(merged.loc[idx], market_schedule, threshold=65)
                    if match.empty:
                        continue

                    # Patch missing columns
                    if "theover_probability" in merged.columns and pd.isna(merged.at[idx, "theover_probability"]) and pd.notna(match.get("theover_probability")):
                        merged.at[idx, "theover_probability"] = match.get("theover_probability")

                    if "ml_probability" in merged.columns and pd.isna(merged.at[idx, "ml_probability"]) and pd.notna(match.get("ml_probability")):
                        merged.at[idx, "ml_probability"] = match.get("ml_probability")

                    if "win_prob_source" in merged.columns and pd.isna(merged.at[idx, "win_prob_source"]) and pd.notna(match.get("win_prob_source")):
                        merged.at[idx, "win_prob_source"] = match.get("win_prob_source")

        # Propagate any slate-level TheOver-feed degradation warning from the uploaded
        # rows onto the master slate. The column-scoped merge above intentionally carries
        # only the probability/source columns, so without this the run_health_warning set
        # by the degradation guard is lost in the live-odds path and the production-stage
        # degraded-run Kelly reduction never fires. Since we now down-weight (rather than
        # null) a degraded feed, its damped signal still influences direction — so the
        # de-staking safety must travel with it. Preserve any existing warning.
        if "run_health_warning" in theover_rows.columns:
            _theover_warn = _string_series(theover_rows, "run_health_warning")
            _theover_warn = _theover_warn[_theover_warn.str.len() > 0]
            if not _theover_warn.empty:
                _existing_warn = _string_series(merged, "run_health_warning")
                merged["run_health_warning"] = _existing_warn.where(
                    _existing_warn.str.len() > 0, _theover_warn.iloc[0]
                )

    # Ensure identity columns survive master-frame merges.
    if "league" not in merged.columns or _string_series(merged, "league").str.len().eq(0).all():
        if not theover_rows.empty and "league" in theover_rows.columns and "matchup_id" in merged.columns and "game_date" in merged.columns:
            league_lookup = (
                theover_rows[[c for c in ["matchup_id", "game_date", "league"] if c in theover_rows.columns]]
                .dropna(subset=["matchup_id", "game_date"])
                .drop_duplicates(["matchup_id", "game_date"], keep="last")
            )
            if not league_lookup.empty:
                merged = merged.merge(
                    league_lookup.rename(columns={"league": "league_from_bets"}),
                    on=["matchup_id", "game_date"],
                    how="left",
                )
                merged["league"] = _string_series(merged, "league").where(
                    _string_series(merged, "league").str.len().gt(0),
                    _string_series(merged, "league_from_bets"),
                )
                merged = merged.drop(columns=["league_from_bets"], errors="ignore")
    if "league" not in merged.columns:
        merged["league"] = ""
    merged["league"] = _string_series(merged, "league").str.upper().replace(LEAGUE_ALIASES)

    # 5. Eliminate the Fallback Artifacts
    # The clunky np.select code and novig_home_price checks have been completely removed from here
    # since we already mapped odds_american properly during the expand step!

    # Do not set a fallback odds_source
    if "odds_source" not in merged.columns:
        merged["odds_source"] = pd.NA

    uploaded_odds = _numeric_series(merged, "odds_american")
    # Avoid force overwriting odds_source to "uploaded" just because odds are not -110.
    # Preserve true provenance from the live_odds expansion layer (e.g. odds_api).
    # If a row has non-110 odds and no other odds_source, we can assign uploaded_only.
    missing_source = _string_series(merged, "odds_source").isna() | _string_series(merged, "odds_source").eq("")
    merged.loc[uploaded_odds.notna() & (uploaded_odds != -110) & missing_source, "odds_source"] = "uploaded_only"

    if not base_df.empty:
        base_schedule = base_df.copy()
        base_schedule["league"] = _string_series(base_schedule, "league").str.upper().replace(LEAGUE_ALIASES)
        base_schedule["home_team"] = _string_series(base_schedule, "home_team").map(normalize_team_name)
        base_schedule["away_team"] = _string_series(base_schedule, "away_team").map(normalize_team_name)
        base_schedule["date"] = _force_utc_datetime(_game_dates(base_schedule))
        base_schedule["merge_date_utc"] = _et_day_string(base_schedule["date"])
        # Backward-compat safety key: older merge paths referenced game_date_key directly.
        # Keep it aligned with merge_date_utc to prevent KeyError in mixed/stale runtime code paths.
        base_schedule["game_date_key"] = base_schedule["merge_date_utc"]

        base_schedule["home_team_lower"] = clean_team_name(base_schedule["home_team"])
        base_schedule["away_team_lower"] = clean_team_name(base_schedule["away_team"])
        base_schedule["matchup_key"] = _canonical_matchup_teams_key(base_schedule)
        base_schedule["matchup_id"] = _matchup_id(base_schedule)
        base_schedule["date_day"] = _date_join_key(base_schedule["date"])

        base_merge_columns = ["league", "matchup_id", "merge_date_utc"] + [
            col for col in ["date", "game_time_est", "is_neutral"]
            if col in base_schedule.columns
        ]

        merged["home_team_lower"] = clean_team_name(_string_series(merged, "home_team").map(normalize_team_name))
        merged["away_team_lower"] = clean_team_name(_string_series(merged, "away_team").map(normalize_team_name))
        merged["matchup_key"] = _canonical_matchup_teams_key(merged)
        merged["matchup_id"] = _matchup_id(merged)
        merged["merge_date_utc"] = _et_day_string(merged.get("game_date"))

        # Primary join uses explicit UTC day keys and canonical matchup keys (order-insensitive).
        merged = merged.merge(
            base_schedule[base_merge_columns].drop_duplicates(["league", "matchup_id", "merge_date_utc"]),
            on=["league", "matchup_id", "merge_date_utc"],
            how="left",
            suffixes=("", "_base"),
        )

        merged["game_date"] = _game_dates(merged)
        merged["game_date"] = merged["game_date"].fillna(merged["date"])

        if "game_time_est_base" in merged.columns:
            merged["game_time_est"] = _string_series(merged, "game_time_est").where(
                _string_series(merged, "game_time_est").str.len().gt(0),
                _string_series(merged, "game_time_est_base"),
            )
            merged = merged.drop(columns=["game_time_est_base"])

        if "is_neutral_base" in merged.columns:
            merged["is_neutral"] = merged["is_neutral"].fillna(merged["is_neutral_base"]) if "is_neutral" in merged.columns else merged["is_neutral_base"]
            merged = merged.drop(columns=["is_neutral_base"])

        if "merge_date_utc" in merged.columns:
            merged = merged.drop(columns=["merge_date_utc"])

    logger.info(f"Number of live Novig games fetched: {len(live_odds_df)}")

    # We still need to ensure odds_american and odds_source are correctly typed.
    merged["odds_american"] = _numeric_series(merged, "odds_american", pd.NA)

    # Final guardrail: if any rows still lack odds, patch with fallback instead of dropping data.
    missing_odds_mask = merged["odds_american"].isna()
    missing_odds_count = int(missing_odds_mask.sum())
    if missing_odds_count > 0:
        if not live_odds_df.empty and 'raw_home_team' in live_odds_df.columns:
            merged_teams = set(zip(merged['home_team'], merged['away_team']))
            unmapped_live = live_odds_df[~live_odds_df.set_index(['home_team', 'away_team']).index.isin(merged_teams)]
            if not unmapped_live.empty:
                unmapped_homes = unmapped_live['raw_home_team'].unique().tolist()
                unmapped_aways = unmapped_live['raw_away_team'].unique().tolist()
                logger.info(f"Unmapped raw teams from Odds API JSON: Homes={unmapped_homes}, Aways={unmapped_aways}")

        patched_games = merged[missing_odds_mask][['home_team', 'away_team', 'market_type']].to_dict('records')
        logger.warning(f"Warning: Patched {missing_odds_count} rows - odds_american was NaN after Novig line mapping. Applying -110 fallback: {patched_games}")
        merged.loc[missing_odds_mask, "odds_american"] = -110.0
        merged.loc[missing_odds_mask, "odds_source"] = _string_series(merged, "odds_source").where(
            _string_series(merged, "odds_source").str.len().gt(0),
            "fallback_novig"
        )

    merged["decimal_odds"] = merged["odds_american"].apply(american_to_decimal)
    merged["decimal_odds"] = pd.to_numeric(merged["decimal_odds"], errors="coerce").fillna(1.91)

    # Phase 4: Implementation of Bayesian Shrinkage and Vig Removal
    # Calculate True Fair-Value Baseline Probability by removing sportsbook overround (vig).
    implied_prob = merged["odds_american"].apply(american_to_prob)

    # No-vig midpoint method for Novig rows when both sides are available.
    m_type_local = _string_series(merged, "market_type").str.lower()
    implied_back = implied_prob.copy()
    implied_lay = pd.Series([pd.NA] * len(merged), index=merged.index, dtype="Float64")

    novig_away_price = _numeric_series(merged, "novig_away_price")
    novig_home_price = _numeric_series(merged, "novig_home_price")
    novig_under_price = _numeric_series(merged, "novig_under_price")
    novig_over_price = _numeric_series(merged, "novig_over_price")
    novig_h2h_away_price = _numeric_series(merged, "novig_h2h_away_price")
    novig_h2h_home_price = _numeric_series(merged, "novig_h2h_home_price")

    implied_lay = implied_lay.where(~m_type_local.eq("spread_home"), novig_away_price.apply(american_to_prob))
    implied_lay = implied_lay.where(~m_type_local.eq("spread_away"), novig_home_price.apply(american_to_prob))
    implied_lay = implied_lay.where(~m_type_local.eq("total_over"), novig_under_price.apply(american_to_prob))
    implied_lay = implied_lay.where(~m_type_local.eq("total_under"), novig_over_price.apply(american_to_prob))
    implied_lay = implied_lay.where(~m_type_local.eq("h2h_home"), novig_h2h_away_price.apply(american_to_prob))
    implied_lay = implied_lay.where(~m_type_local.eq("h2h_away"), novig_h2h_home_price.apply(american_to_prob))

    novig_midpoint = ((implied_back + implied_lay) / 2.0).clip(0.01, 0.99)

    # Fallback de-vig when midpoint inputs are unavailable.
    opposing_implied = merged["odds_american"].apply(get_opposing_odds_from_exchange).apply(american_to_prob)
    fallback_market_probability = (implied_prob / (implied_prob + opposing_implied)).clip(0.01, 0.99)
    merged["market_probability"] = novig_midpoint.where(novig_midpoint.notna(), fallback_market_probability)

    # Mandatory Sanitization Layer
    logger.info(f"PIPELINE AUDIT: [2/9] Rows before sanitization: {len(merged)}")
    if not merged.empty:
        # Patch pathological/synthetic odds (e.g., -99900)
        valid_odds_mask = merged["odds_american"].isna() | ((merged["odds_american"] >= -10000) & (merged["odds_american"] <= 10000))

        # Patch extreme implied probabilities reflecting suspended markets
        valid_prob_mask = merged["market_probability"].isna() | ((merged["market_probability"] >= 0.05) & (merged["market_probability"] <= 0.95))

        dropped = len(merged) - (valid_odds_mask & valid_prob_mask).sum()
        logger.info(f"PIPELINE AUDIT: [3/9] Rows patched by sanitization: {dropped}")
        # Not dropping, we patch instead.
        logger.info(f"PIPELINE AUDIT: [4/9] Rows dropped or excluded after sanitization: 0")
        if dropped > 0:
            logger.warning(f"Sanitization layer patched {dropped} rows with extreme/synthetic lines instead of dropping.")
            merged.loc[~valid_odds_mask, "odds_american"] = -110.0
            merged.loc[~valid_odds_mask, "odds_source"] = "fallback_novig"
            merged.loc[~valid_prob_mask, "market_probability"] = 0.5238
            merged.loc[~valid_odds_mask | ~valid_prob_mask, "sanitized_value"] = True
        if "sanitized_value" not in merged.columns:
            merged["sanitized_value"] = False

    merged["spread"] = pd.to_numeric(merged.get("spread_line"), errors="coerce")
    merged["total"] = pd.to_numeric(merged.get("total_line"), errors="coerce")
    merged = _enforce_identity_string_dtype(merged, ["league", "home_team", "away_team"])
    merged = _restore_missing_ncaab_league_priority(merged)
    logger.info(f"PIPELINE TRACE: Rows after sanitization & formatting: {len(merged)}")

    # ML Prediction Enrichment [2026-03-08]
    ml_model_actually_loaded = False
    merged["model_status"] = "OK"
    nba_stats_diag = {
        "nba_stats_fetch_status": "not_started",
        "nba_stats_fetch_source": "none",
        "nba_stats_fetch_retries_used": 0,
        "nba_rows_live_stats": 0,
        "nba_rows_cached_stats": 0,
        "nba_rows_fallback_stats": 0,
        "rows_unresolved_team_mapping": 0,
        "rows_excluded_from_ml_unresolved_stats": 0,
    }
    ml_input_diag: dict[str, Any] = {
        "ml_input_row_count": 0,
        "ml_feature_eligible_row_count": 0,
        "ml_rows_excluded_count": 0,
        "ml_zero_variance_feature_count": 0,
        "ml_near_constant_feature_count": 0,
        "ml_high_missingness_feature_count": 0,
        "ml_top_missing_features": {},
        "ml_top_feature_nunique": {},
        "ml_flatness_root_cause_hint": "not_computed",
    }
    ml_prediction_diag: dict[str, Any] = {}
    if use_ml and ML_AVAILABLE and PredictionEngine is not None:
        logger.warning("🔍 ML DEBUG: use_ml=True, attempting predictions...")
        logger.info(f"PIPELINE TRACE: Sending {len(merged)} rows into ML prediction logic.")
        try:
            existing_ml = _numeric_series(merged, "ml_probability")
            non_na_existing = existing_ml.dropna()
            if len(non_na_existing) > 0:
                logger.warning(
                    "⚠️ ML DEBUG: Ignoring %s pre-populated ml_probability values and recomputing from model/features.",
                    len(non_na_existing),
                )

            # Authoritative ML path: when ML is enabled, always recompute all rows.
            # This prevents stale uploaded/base values (including collapsed constants like 0.1906)
            # from leaking into the final ML Prob column.
            needs_prediction = pd.Series([True] * len(merged), index=merged.index)
            merged["ml_probability"] = pd.NA

            logger.info(f"PIPELINE AUDIT: [5/9] Rows eligible for ML (needs_prediction mask): {needs_prediction.sum()}")
            if needs_prediction.any():
                merge_identity_keys = ["league", "home_team", "away_team", "game_date"]
                merged = _normalize_merge_keys(merged, merge_identity_keys)

                if 'decimal_odds' in merged.columns:
                    merged['implied_home_prob'] = merged.get('implied_home_prob', pd.Series(dtype=float))
                    merged['implied_home_prob'] = pd.to_numeric(merged['implied_home_prob'], errors='coerce').fillna(
                        1 / pd.to_numeric(merged['decimal_odds'], errors='coerce')
                    )
                if 'kalshi_probability' in merged.columns:
                    merged['kalshi_prob'] = merged.get('kalshi_prob', merged['kalshi_probability'])

                # We need to enrich the features here to provide live data coverage before hitting prediction fallback
                from app_core.feature_processing import enrich_with_model_features
                api_clients = {}  # Stub for backward compatibility if it expects dict
                enriched_for_prediction = enrich_with_model_features(merged[needs_prediction].copy(), api_clients)
                for col in [
                    "stats_source",
                    "stats_resolution_status",
                    "stats_fallback_reason",
                    "ml_feature_eligible",
                    "nba_stats_fetch_status",
                    "nba_stats_fetch_source",
                    "nba_stats_fetch_retries_used",
                    "fallback_summary_by_league",
                    "fallback_heavy_slate_flag",
                    "run_health_warning",
                    "stats_source_counts",
                ]:
                    if col in enriched_for_prediction.columns:
                        merged.loc[needs_prediction, col] = enriched_for_prediction[col]

                if "League" in enriched_for_prediction.columns:
                    nba_rows = enriched_for_prediction["League"].astype(str).str.upper().eq("NBA")
                    nba_stats_diag["nba_rows_live_stats"] = int((nba_rows & enriched_for_prediction.get("stats_source", pd.Series(index=enriched_for_prediction.index, dtype="object")).astype(str).eq("live")).sum())
                    nba_stats_diag["nba_rows_cached_stats"] = int((nba_rows & enriched_for_prediction.get("stats_source", pd.Series(index=enriched_for_prediction.index, dtype="object")).astype(str).eq("cached")).sum())
                    nba_stats_diag["nba_rows_fallback_stats"] = int((nba_rows & enriched_for_prediction.get("stats_source", pd.Series(index=enriched_for_prediction.index, dtype="object")).astype(str).isin(["fallback", "failed"])).sum())
                    nba_stats_diag["rows_unresolved_team_mapping"] = int(enriched_for_prediction.get("stats_resolution_status", pd.Series(index=enriched_for_prediction.index, dtype="object")).astype(str).eq("unresolved").sum())

                if "nba_stats_fetch_status" in enriched_for_prediction.columns and not enriched_for_prediction.empty:
                    status_series = enriched_for_prediction["nba_stats_fetch_status"].astype(str)
                    if status_series.str.lower().eq("failed").any():
                        nba_stats_diag["nba_stats_fetch_status"] = "failed"
                    elif status_series.str.lower().eq("cached").any():
                        nba_stats_diag["nba_stats_fetch_status"] = "cached"
                    elif status_series.str.lower().eq("live").any():
                        nba_stats_diag["nba_stats_fetch_status"] = "live"
                    else:
                        nba_stats_diag["nba_stats_fetch_status"] = "ok"
                if "nba_stats_fetch_source" in enriched_for_prediction.columns and not enriched_for_prediction.empty:
                    source_series = enriched_for_prediction["nba_stats_fetch_source"].astype(str)
                    if source_series.str.lower().eq("live").any():
                        nba_stats_diag["nba_stats_fetch_source"] = "live"
                    elif source_series.str.lower().eq("cached").any():
                        nba_stats_diag["nba_stats_fetch_source"] = "cached"
                    elif source_series.str.lower().eq("failed").any():
                        nba_stats_diag["nba_stats_fetch_source"] = "failed"
                if "stats_fetch_retries_used" in enriched_for_prediction.columns and not enriched_for_prediction.empty:
                    nba_stats_diag["nba_stats_fetch_retries_used"] = int(
                        pd.to_numeric(enriched_for_prediction["stats_fetch_retries_used"], errors="coerce").fillna(0).max()
                    )
                if "ml_feature_eligible" in enriched_for_prediction.columns:
                    ml_eligible = enriched_for_prediction["ml_feature_eligible"].fillna(True).astype(bool)
                    excluded_count = int((~ml_eligible).sum())
                    nba_stats_diag["rows_excluded_from_ml_unresolved_stats"] = excluded_count
                    needs_prediction = needs_prediction & ml_eligible
                    merged.loc[~ml_eligible, "model_status"] = "Stats Unresolved"
                else:
                    ml_eligible = pd.Series([True] * len(enriched_for_prediction), index=enriched_for_prediction.index, dtype=bool)

                ml_input_diag = _compute_ml_input_flatness_diagnostics(enriched_for_prediction, ml_eligible)
                logger.info(
                    "ML INPUT DIAGNOSTICS: rows=%s eligible=%s excluded=%s zero_var=%s near_const=%s high_missing=%s hint=%s",
                    ml_input_diag["ml_input_row_count"],
                    ml_input_diag["ml_feature_eligible_row_count"],
                    ml_input_diag["ml_rows_excluded_count"],
                    ml_input_diag["ml_zero_variance_feature_count"],
                    ml_input_diag["ml_near_constant_feature_count"],
                    ml_input_diag["ml_high_missingness_feature_count"],
                    ml_input_diag["ml_flatness_root_cause_hint"],
                )

                # DIAGNOSTICS & DEDUPLICATION: check for duplicate columns after feature enrichment
                logger.info(f"Shape after enrich_with_model_features: {enriched_for_prediction.shape}")
                logger.info(f"Dataframe shape after enrichment: {enriched_for_prediction.shape}")
                has_dupes = enriched_for_prediction.columns.duplicated().any()
                logger.info(f"Duplicate columns exist after enrichment: {has_dupes}")
                if has_dupes:
                    dup_cols = enriched_for_prediction.columns[enriched_for_prediction.columns.duplicated()].unique()
                    logger.warning(f"Duplicate columns found after enrichment: {list(dup_cols)}")
                    critical_cols = ['implied_home_prob', 'kalshi_prob', 'market_probability', 'decimal_odds']
                    critical_dupes = [col for col in dup_cols if col in critical_cols]
                    if critical_dupes:
                        logger.warning(f"CRITICAL duplicate columns found after enrichment: {critical_dupes}")

                from app_core.dataframe_utils import collapse_duplicate_columns
                enriched_for_prediction = collapse_duplicate_columns(
                    enriched_for_prediction,
                    critical_cols=['implied_home_prob', 'kalshi_prob', 'market_probability', 'decimal_odds', 'ml_probability']
                )

                if enriched_for_prediction.columns.duplicated().any():
                    remaining_dupes = enriched_for_prediction.columns[enriched_for_prediction.columns.duplicated()].unique()
                    raise RuntimeError(f"Duplicate columns STILL exist in enriched_for_prediction after collapse: {list(remaining_dupes)}")

                # Copy the enriched columns back into merged (or at least provide to predictor)
                # Ensure the predictor runs on the enriched dataframe

                engine = get_cached_prediction_engine()
                ml_model_actually_loaded = not getattr(engine, "use_fallback", True)

                # predict_batch expects a DataFrame, returns List[float]
                logger.info(f"PIPELINE AUDIT: [6/9] Rows actually sent into predict_batch: {len(enriched_for_prediction)}")
                predictions_list = engine.predict_batch(enriched_for_prediction.loc[needs_prediction])
                if hasattr(engine, "_last_metrics") and isinstance(engine._last_metrics, dict):
                    ml_prediction_diag = dict(engine._last_metrics)
                logger.info(f"PIPELINE AUDIT: [7/9] Rows returned from predict_batch: {len(predictions_list)}")

                # Extra safeguard for assignment back to merged
                num_needed = int(needs_prediction.sum())
                if len(predictions_list) != num_needed:
                    error_msg = f"Prediction mismatch: predict_batch returned {len(predictions_list)} predictions, but {num_needed} were needed."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Assign predictions only to rows that needed them
                if "ml_probability" not in merged.columns:
                    merged["ml_probability"] = pd.NA

                merged.loc[needs_prediction, "ml_probability"] = pd.Series(
                    predictions_list,
                    index=merged[needs_prediction].index,
                    dtype="float64"
                )
                logger.info(f"PIPELINE AUDIT: [8/9] Rows successfully merged back into the analysis dataframe: {len(predictions_list)}")

                # Assign used_stale_features flag
                if hasattr(engine, 'last_batch_used_stale_features'):
                    if "used_stale_features" not in merged.columns:
                        merged["used_stale_features"] = pd.Series(False, index=merged.index, dtype=bool)
                    merged.loc[needs_prediction, "used_stale_features"] = pd.Series(
                        engine.last_batch_used_stale_features,
                        index=merged[needs_prediction].index,
                        dtype=bool
                    )

                if hasattr(engine, 'last_batch_used_neutral_fallback') and engine.last_batch_used_neutral_fallback:
                    merged.loc[needs_prediction, "model_status"] = "Neutral Fallback"

                ml_count = merged["ml_probability"].notna().sum()
                ml_unique = _numeric_series(merged, "ml_probability").dropna().nunique()
                logger.warning(
                    f"✅ ML DEBUG: Generated {ml_count} total predictions ({needs_prediction.sum()} new, unique={ml_unique})"
                )
            else:
                logger.warning("✅ ML DEBUG: All rows already have ml_probability")

        except Exception as e:
            logger.error(f"❌ ML prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Graceful fallback unconditionally applied
            fallback_applied = False
            try:
                engine = PredictionEngine()
                engine.use_fallback = True
                fallback_predictions = engine.predict_batch(merged)
                merged["ml_probability"] = pd.Series(fallback_predictions, index=merged.index, dtype="float64")
                merged["model_status"] = "Statistical Fallback"
                fallback_applied = True
                logger.warning("⚠️ ML DEBUG: Applied statistical fallback predictions unconditionally after model failure.")
            except Exception as fallback_err:
                logger.error(f"❌ Statistical fallback prediction failed: {fallback_err}")

            if not fallback_applied:
                if "ml_probability" not in merged.columns:
                    merged["ml_probability"] = pd.NA
                merged["model_status"] = "Model Failure"
    else:
        if "ml_probability" not in merged.columns:
            merged["ml_probability"] = pd.NA

    # If ML is disabled, clear any existing ml_probability values
    if not use_ml:
        if "ml_probability" in merged.columns:
            merged["ml_probability"] = pd.NA
        merged["model_status"] = "Model Disabled"

    merged.loc[_numeric_series(merged, "ml_probability").isna() & _string_series(merged, "model_status").eq("OK"), "model_status"] = "Model Failure"

    theover_probability = _numeric_series(merged, "theover_probability")
    theover_probability = theover_probability.where(theover_probability <= 1, theover_probability / 100.0)

    # Blend-input TheOver: FADE (shrink toward 0.50), don't drop, the genuine-but-cold
    # sources (model_hit_rate_flipped = TheOver's Under pick). Left at full weight, the
    # flipped value inflates the Under's win-prob/EV; fully dropping it discards a real
    # signal. Shrinking it tempers the influence proportionally to MLB_THEOVER_FADE_SHRINK.
    # The raw value is preserved in merged["theover_probability"] for display/backtest.
    _src_col = merged["win_prob_source"] if "win_prob_source" in merged.columns else None
    # MLB-tuned fade shrink applies only to MLB totals; other leagues/markets keep the
    # legacy default so the MLB tuning never silently changes NBA/NHL totals.
    theover_blend_input = _scoped_theover_blend_fade(
        theover_probability.to_numpy(dtype=float),
        _src_col,
        _string_series(merged, "league"),
        _string_series(merged, "market_type"),
        theover_probability.index,
    )

    ml_probability = _numeric_series(merged, "ml_probability")

    market_type = _string_series(merged, "market_type").str.lower()
    spread_model = ml_probability.where(ml_probability.notna(), theover_probability)

    # Non-MLB totals: pre-mix TheOver (60%) + ML (40%) since TheOver adds context ML lacks.
    # MLB totals: use pure ML — TheOver is passed separately to compute_blended_probability
    # and weighted via MLB_TOTAL_THEOVER_WEIGHT. Pre-mixing here caused double-counting:
    # TheOver was effectively ~44% weight instead of the 10% in config.
    is_mlb = _string_series(merged, "league").str.upper() == "MLB"
    is_mlb_total = is_mlb & market_type.str.contains("total", case=False, na=False)

    mixed_total_model = (0.6 * theover_probability) + (0.4 * ml_probability)
    mixed_total_model = mixed_total_model.where(mixed_total_model.notna(), theover_probability.where(theover_probability.notna(), ml_probability))

    # For MLB totals, pass raw ml_probability so TheOver isn't pre-baked into the "ML" input.
    total_model = pd.Series(
        np.where(is_mlb_total, ml_probability, mixed_total_model),
        index=merged.index,
        dtype="float64",
    )
    total_model = total_model.where(total_model.notna(), theover_probability.where(theover_probability.notna(), ml_probability))

    is_side = market_type.str.contains("spread", case=False, na=False)
    model_probability = pd.Series(
        np.where(is_side, spread_model, total_model),
        index=merged.index,
        dtype="float64",
    )

    # Apply lowercase for clean fuzzy matching right before returning
    # merged['home_team'] = merged['home_team'].astype(str).str.lower()
    # merged['away_team'] = merged['away_team'].astype(str).str.lower()

    # --- Metadata Framework & Situational Adjustments ---
    # Live external metadata ingestion (starting goalies, player injuries, live pace)
    # This framework adjusts the raw model probability before blending with the market.

    # Goalie Delta (NHL): Reduce win prob by 6.5% if a secondary goalie is starting, boost opponent by 6.5%.
    # Check for feature_goalie_delta column (1.0 = true)
    is_nhl = merged["league"].str.upper() == "NHL"
    has_goalie_delta = pd.Series([False] * len(merged), index=merged.index)
    if "feature_goalie_delta" in merged.columns:
        has_goalie_delta = pd.to_numeric(merged["feature_goalie_delta"], errors="coerce").fillna(0) == 1.0

    # Apply goalie penalty assuming it applies to the home team's goalie
    # (Downstream data ingestion maps this feature when the home team uses a backup)
    # Home team probability down, Away probability up
    goalie_impact = is_nhl & has_goalie_delta
    model_probability = model_probability.where(~(goalie_impact & (merged["market_type"] == "spread_home")), model_probability - 0.065)
    model_probability = model_probability.where(~(goalie_impact & (merged["market_type"] == "spread_away")), model_probability + 0.065)

    # Pace-Setter (NBA): Inflate "Over" probability when a high-usage star is active.
    # Check for feature_star_active column (1.0 = true)
    is_nba = merged["league"].str.upper() == "NBA"
    has_star_active = pd.Series([False] * len(merged), index=merged.index)
    if "feature_star_active" in merged.columns:
        has_star_active = pd.to_numeric(merged["feature_star_active"], errors="coerce").fillna(0) == 1.0

    star_impact = is_nba & has_star_active
    from app_core.weights_config import NBA_STAR_ACTIVE_TOTAL_OVER_BOOST, NBA_STAR_ACTIVE_TOTAL_UNDER_PENALTY

    star_impact_rows = star_impact.sum()
    if star_impact_rows > 0:
        logger.info(f"Applying NBA star-active totals adjustment (boost={NBA_STAR_ACTIVE_TOTAL_OVER_BOOST}) to {star_impact_rows} games")

    model_probability = model_probability.where(~(star_impact & (merged["market_type"] == "total_over")), model_probability + NBA_STAR_ACTIVE_TOTAL_OVER_BOOST)
    model_probability = model_probability.where(~(star_impact & (merged["market_type"] == "total_under")), model_probability + NBA_STAR_ACTIVE_TOTAL_UNDER_PENALTY)

    # Enrich with external data (injuries, weather) — must run before adjustments below
    try:
        from app_core.external_data_fetcher import enrich_with_external_data
        merged = enrich_with_external_data(merged)
        logger.info(f"External enrichment complete — injuries_home sum={merged.get('injuries_home_count', pd.Series([0])).sum()}, weather flags={merged.get('weather_flag', pd.Series([0.0])).sum()}")
    except Exception as _ext_err:
        logger.warning(f"External data enrichment skipped: {_ext_err}")

    # Injury Impact: adjust model probability based on key player availability.
    from app_core.weights_config import INJURY_PROB_PENALTY_PER_KEY_PLAYER, INJURY_KEY_PLAYER_THRESHOLD
    home_injuries = pd.to_numeric(merged.get("injuries_home_count", 0), errors="coerce").fillna(0)
    away_injuries = pd.to_numeric(merged.get("injuries_away_count", 0), errors="coerce").fillna(0)
    home_injury_penalty = (home_injuries.clip(upper=4) * INJURY_PROB_PENALTY_PER_KEY_PLAYER).where(home_injuries >= INJURY_KEY_PLAYER_THRESHOLD, 0.0)
    away_injury_penalty = (away_injuries.clip(upper=4) * INJURY_PROB_PENALTY_PER_KEY_PLAYER).where(away_injuries >= INJURY_KEY_PLAYER_THRESHOLD, 0.0)
    is_home_side = merged["market_type"].astype(str).eq("spread_home")
    is_away_side = merged["market_type"].astype(str).eq("spread_away")
    model_probability = model_probability.where(~is_home_side, (model_probability - home_injury_penalty + away_injury_penalty).clip(0.01, 0.99))
    model_probability = model_probability.where(~is_away_side, (model_probability - away_injury_penalty + home_injury_penalty).clip(0.01, 0.99))

    # Weather Impact (MLB outdoor games only): reduce total over probability in bad weather.
    from app_core.weights_config import WEATHER_TOTAL_OVER_PENALTY
    if "weather_flag" in merged.columns:
        is_mlb_total_over = (merged["league"].astype(str).str.upper().eq("MLB")) & (merged["market_type"].astype(str).eq("total_over"))
        is_mlb_total_under = (merged["league"].astype(str).str.upper().eq("MLB")) & (merged["market_type"].astype(str).eq("total_under"))
        has_bad_weather = pd.to_numeric(merged["weather_flag"], errors="coerce").fillna(0).eq(1.0)
        model_probability = model_probability.where(~(is_mlb_total_over & has_bad_weather), (model_probability - WEATHER_TOTAL_OVER_PENALTY).clip(0.01, 0.99))
        model_probability = model_probability.where(~(is_mlb_total_under & has_bad_weather), (model_probability + WEATHER_TOTAL_OVER_PENALTY).clip(0.01, 0.99))

    # Tournament Efficiency Decay: Flat 4% reduction to final "Over" probability for all postseason games.
    is_postseason = is_postseason_ncaab(merged)
    model_probability = model_probability.where(~(is_postseason & (merged["market_type"] == "total_over")), model_probability - 0.04)
    model_probability = model_probability.where(~(is_postseason & (merged["market_type"] == "total_under")), model_probability + 0.04)

    # Home Court Dominance: NIT games at campus sites.
    # Placeholder mock: Illinois State and Bradley at home
    nit_campus_home = is_postseason & (merged["home_team"].str.contains("Illinois State", case=False, na=False) | merged["home_team"].str.contains("Bradley", case=False, na=False))
    model_probability = model_probability.where(~(nit_campus_home & (merged["market_type"] == "spread_home")), model_probability + 0.065)
    model_probability = model_probability.where(~(nit_campus_home & (merged["market_type"] == "spread_away")), model_probability - 0.065)

    # Ensure probabilities are bounded
    model_probability = model_probability.clip(0.01, 0.99)
    # --- End Metadata Framework ---

    kalshi_probability = _numeric_series(merged, "kalshi_probability") if "kalshi_probability" in merged.columns else pd.Series([pd.NA]*len(merged), index=merged.index)
    _raw_sentiment = _numeric_series(merged, "sentiment_diff", 0.0) if "sentiment_diff" in merged.columns else pd.Series([0.0]*len(merged), index=merged.index)
    sentiment_prob = (0.5 + _raw_sentiment * 0.5).clip(0.0, 1.0)
    calibrated_probability = compute_blended_probability(
        p_market=merged["market_probability"],
        p_kalshi=kalshi_probability,
        p_ml=model_probability,
        p_theover=theover_blend_input,
        p_sentiment=sentiment_prob,
        league=_string_series(merged, "league"),
        market_type=_string_series(merged, "market_type")
    )

    # Market-anchored over-bias correction for MLB totals — PRODUCTION best-picks
    # path (this is the blend the card is built from; #1919 applied it only to the
    # Analysis-tab blend, so the card never rebalanced). Same shared helper, so the
    # two paths cannot drift. Runs before EV/edge/direction selection below.
    calibrated_probability, _prod_debias = apply_mlb_total_market_debias(calibrated_probability, merged)
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
        "best_picks": int(len(best_picks_df)),
        "kalshi_attempted": 0,
        "kalshi_matches": 0,
        "kalshi_match_rate": 0.0,
        "match_rate": 0.0,
        "theover_totals_games": int(_matchup_id(_coerce_identity_columns(_normalize_upload_columns(totals_df))).nunique()) if totals_df is not None and not totals_df.empty else 0,
        "theover_spreads_games": int(_matchup_id(_coerce_identity_columns(_normalize_upload_columns(spreads_df))).nunique()) if spreads_df is not None and not spreads_df.empty else 0,
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
        # 1. Identify best source for time
        src_col = "commence_time_raw" if "commence_time_raw" in analysis_df.columns else "game_date"

        # 2. Format using a temporary view that preserves the Index
        temp_df = analysis_df[[src_col]].rename(columns={src_col: "game_date"})
        analysis_df["game_time_est"] = _format_game_time_est(temp_df)

        # 3. Final Cleanup
        analysis_df = analysis_df.drop(columns=["commence_time_raw"], errors="ignore")

        # 4. Sync the slate date with actual start time
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
                # which simply measured model confidence — picks the model was wrongly
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

    return (analysis_df, best_picks_df, diagnostics)


def generate_parlays(best_picks_df: pd.DataFrame, max_legs: int = 3) -> pd.DataFrame:
    from core.kelly_optimizer import add_kelly_bet_sizing, apply_simultaneous_kelly
    from core.probability_calibration import load_calibration
    from core.smart_parlay_engine import downweight_correlated_parlay_kelly, generate_smart_parlays

    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    # Recap-fitted isotonic table (scripts/fit_calibration.py); None when absent,
    # in which case legs use raw effective_win_probability as before.
    calibration = load_calibration()
    parlays_df = generate_smart_parlays(best_picks_df, num_rr_candidates=5, calibration=calibration)

    if parlays_df.empty:
        return parlays_df

    # Cap at top 10 per leg count so the UI stays readable.
    # Already sorted by EV desc inside generate_smart_parlays, so head(10) = best 10.
    parlays_df = (
        parlays_df.groupby("legs", group_keys=False)
        .apply(lambda g: g.head(10))
        .reset_index(drop=True)
    )

    parlays_df = add_kelly_bet_sizing(parlays_df, bankroll=1000.0, fraction=0.125)
    # Correlated combos (same-game legs or a same-direction Agrees pair) carry
    # block variance — halve their stake before exposure caps are applied.
    parlays_df = downweight_correlated_parlay_kelly(parlays_df)
    parlays_df = apply_simultaneous_kelly(parlays_df, bankroll=1000.0, max_exposure=0.05)

    # Persist the day's recommended parlays so they can be graded alongside the
    # slate. Recaps grade single picks only, so the parlay engine has never
    # received realized feedback; this log is the input for that. Last run of the
    # day wins, matching the card actually shown.
    try:
        log_dir = Path("data/parlay_log")
        log_dir.mkdir(parents=True, exist_ok=True)
        slate_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        logged = parlays_df.copy()
        logged.insert(0, "generated_date", slate_date)
        logged.to_csv(log_dir / f"{slate_date}.csv", index=False)
    except Exception as e:
        logger.warning(f"Failed to write parlay log: {e}")

    return parlays_df

def optimize_portfolio_allocation(best_picks_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
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
    production_eligible = (
        status.eq("actionable")
        & line_source.eq("live")
        & line_warning.eq("")
        & line_used.notna()
        & line_consistent
        & event_identity_ok
        & (~best_pick_norm.str.contains("unresolved", na=False))
    )
    portfolio["production_eligible"] = production_eligible

    portfolio["decimal_odds"] = _numeric_series(portfolio, "decimal_odds").fillna(
        _numeric_series(portfolio, "odds_american", -110.0).apply(american_to_decimal)
    )
    p = pd.to_numeric(portfolio.get("calibrated_probability", pd.NA), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    b = (portfolio["decimal_odds"] - 1.0).clip(lower=0.0)
    q = 1.0 - p
    kelly_fraction = pd.Series(0.0, index=portfolio.index, dtype=float)
    valid = b > 0
    kelly_fraction.loc[valid] = (((b.loc[valid] * p.loc[valid]) - q.loc[valid]) / b.loc[valid]).clip(lower=0.0)
    portfolio["kelly_probability_used"] = p
    portfolio["kelly_decimal_odds"] = portfolio["decimal_odds"]
    portfolio["kelly_fraction"] = kelly_fraction
    portfolio["raw_kelly_amount"] = float(bankroll) * kelly_fraction
    portfolio["fractional_kelly_amount"] = portfolio["raw_kelly_amount"] * 0.25
    portfolio["recommended_bet"] = portfolio["fractional_kelly_amount"]
    portfolio.loc[~portfolio["production_eligible"], "recommended_bet"] = 0.0

    max_pick = float(bankroll) * 0.04
    max_slate = float(bankroll) * 0.25
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
        # for visibility but carry NO production stake — their production_bet_amount /
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
                w = pd.Series(1.0, index=idx)  # equal-weight when no positive Kelly
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
        "kelly_probability_used", "kelly_decimal_odds", "kelly_fraction", "fractional_kelly_amount", "kelly_weight_share", "slate_scaled_amount",
        "kelly_allocation_method", "kelly_flattening_detected", "kelly_unique_positive_amount_count", "kelly_total_raw_amount",
        "kelly_total_fractional_amount", "kelly_total_production_amount", "kelly_max_pick_cap_hits", "kelly_slate_scale_factor",
        "market_line_used", "market_line_source", "line_consistency_flag", "line_event_identity_match_flag", "line_provenance_warning",
        "export_run_id", "pick_id", "canonical_pick_key", "production_eligible", "non_actionable_eligible",
    ]
    for col in cols:
        if col not in portfolio.columns:
            portfolio[col] = pd.NA
    return portfolio[cols].sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=30, simulations=200)
