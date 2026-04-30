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
    LOCK_UPLOAD_LINES_FOR_MATCHED_ROWS,
    KALSHI_WEIGHT, MARKET_WEIGHT, ML_MODEL_WEIGHT, THEOVER_WEIGHT, SENTIMENT_WEIGHT,
    FALLBACK_MARKET_WEIGHT, FALLBACK_ML_WEIGHT, FALLBACK_THEOVER_WEIGHT, FALLBACK_SENTIMENT_WEIGHT,
    LOW_LIQUIDITY_KALSHI_WEIGHT, LOW_LIQUIDITY_ML_MODEL_WEIGHT
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

VALID_MARKETS = {"spread_home", "spread_away", "total_over", "total_under", "h2h_home", "h2h_away"}
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


REQUIRED_BEST_PICK_EXPORT_COLUMNS = [
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
]


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
    }

    initially_missing_cols = [c for c in req_cols if c not in out.columns]
    for col in initially_missing_cols:
        out[col] = default_values.get(col, pd.NA)
    missing_cols = [c for c in req_cols if c not in out.columns]

    for col in req_cols:
        if col in {"status_blocker_reason", "status_blocker_stage", "nba_stats_fetch_status", "fallback_summary_by_league", "run_health_warning", "degraded_feature_subset_reason", "status_metric_basis", "market_line_source", "market_line_source_detail", "line_consistency_reason", "line_provenance_warning"}:
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
    "Triple_Filter_Rank", "parlay_rank",
    "league", "home_team", "away_team", "game_date", "game_time_est", "market_type", "candidate_source", "orientation_source", "upload_match_reason", "best_pick", "Pick_Status", "Status_Reason",
    "calibrated_probability", "expected_value", "edge", "consensus_agreement",
    "odds_american", "odds_source", "market_probability", "ml_probability", "display_probability",
    "kalshi_probability", "kalshi_match_status", "kalshi_match_reason",
    "gemini_explanation", "gemini_risk_notes", "used_stale_features", "Pick_Quality", "Conviction_Score",
    "uploaded_spread_line", "uploaded_total_line", "live_spread_line", "live_total_line", "line_source", "line_delta", "upload_market_match",
    "market_line_used", "market_line_source", "market_line_source_detail", "matched_live_spread_line", "matched_live_total_line", "upload_spread_line", "upload_total_line", "base_spread_line", "base_total_line",
    "line_consistency_flag", "line_consistency_reason", "line_provenance_warning",
    "suspicious_data_flag", "suspicious_data_reasons", "status_metric_basis", "effective_expected_value", "effective_edge", "effective_win_probability", "status_blocker_reason", "status_blocker_stage",
    "nba_stats_fetch_status", "nba_stats_fetch_source", "nba_stats_fetch_retries_used", "stats_source_counts", "fallback_summary_by_league", "fallback_heavy_slate_flag", "run_health_warning",
    "degraded_feature_subset_flag", "degraded_feature_subset_reason",
    "actionable_family_counts", "totals_only_actionable_flag", "viable_side_candidates_count", "side_promoted_by_balance_guard_count", "side_balance_guard_reason",
]

CANONICAL_BET_COLUMNS = [
    "league", "home_team", "away_team", "game_date", "game_time_est", "game_key",
    "market_type", "candidate_source", "orientation_source", "upload_match_reason", "spread_line", "total_line",
    "theover_probability", "odds_american", "odds_source", "market_probability",
    "ml_probability", "display_probability", "calibrated_probability", "expected_value", "edge", "best_pick", "used_stale_features", "matchup_id", "Conviction_Score",
    "uploaded_spread_line", "uploaded_total_line", "live_spread_line", "live_total_line", "line_source", "line_delta", "upload_market_match",
]

_EXPORT_SIGNAL_COLS = {"market_type", "calibrated_probability", "expected_value", "edge"}

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
    if df is None or df.empty:
        return pd.Series(dtype="string")
    if col in df.columns:
        return df[col].fillna(default).astype("string")
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
    theover = pd.to_numeric(p_theover, errors="coerce").fillna(0.5)
    sentiment = pd.to_numeric(p_sentiment, errors="coerce").fillna(0.5)
    m_type = pd.Series(market_type).fillna("").astype(str).str.lower()

    def _blend_row(p_mkt, p_kal, p_ml, p_the, p_sen, m_typ, lg):
        if pd.isna(p_mkt):
            p_mkt = p_ml if pd.notna(p_ml) else 0.5

        # Kalshi Probability is already oriented to the pick side before this step
        k_oriented = None
        if pd.notna(p_kal):
            k_oriented = p_kal

        # Tier 1 vs Tier 2
        if k_oriented is not None and k_oriented >= 0.55:
            # Tier 1
            w_kalshi = KALSHI_WEIGHT
            w_market = MARKET_WEIGHT
            w_ml = ML_MODEL_WEIGHT
            w_the = THEOVER_WEIGHT
            w_sen = SENTIMENT_WEIGHT

            # Market Maturity Overrides (MLB/NHL)
            if lg and (lg.upper() == "MLB" or lg.upper() == "NHL"):
                w_kalshi = LOW_LIQUIDITY_KALSHI_WEIGHT
                w_ml = LOW_LIQUIDITY_ML_MODEL_WEIGHT

            # Re-normalize if ML is missing
            if pd.isna(p_ml):
                total_w = w_kalshi + w_market + w_the + w_sen
                w_kalshi /= total_w
                w_market /= total_w
                w_the /= total_w
                w_sen /= total_w
                p_ml_val = 0.0
                w_ml = 0.0
            else:
                p_ml_val = p_ml

            prob = (k_oriented * w_kalshi) + (p_mkt * w_market) + (p_ml_val * w_ml) + (p_the * w_the) + (p_sen * w_sen)
        else:
            # Tier 2 Fallback (Kalshi disagrees or unavailable)
            w_market = FALLBACK_MARKET_WEIGHT
            w_ml = FALLBACK_ML_WEIGHT
            w_the = FALLBACK_THEOVER_WEIGHT
            w_sen = FALLBACK_SENTIMENT_WEIGHT

            if pd.isna(p_ml):
                total_w = w_market + w_the + w_sen
                w_market /= total_w
                w_the /= total_w
                w_sen /= total_w
                p_ml_val = 0.0
                w_ml = 0.0
            else:
                p_ml_val = p_ml

            prob = (p_mkt * w_market) + (p_ml_val * w_ml) + (p_the * w_the) + (p_sen * w_sen)

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

def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["odds_american"] = _numeric_series(out, "odds_american", pd.NA)

    # Phase 4: Implementation of Bayesian Shrinkage and Vig Removal
    # De-vig by applying multiplicative normalization for a standard 2-way market.
    # Since Novig is an exchange without standard sportsbook straddles, use its true implied prob.
    # We still perform a simple multiplicative normalization in case of minor bid-ask spread deviations
    implied_prob = out["odds_american"].apply(american_to_prob)

    opposing_implied = out["odds_american"].apply(get_opposing_odds_from_exchange).apply(american_to_prob)
    out["market_probability"] = implied_prob / (implied_prob + opposing_implied)

    theover = _numeric_series(out, "theover_probability")
    theover = theover.where(theover <= 1, theover / 100.0)
    ml = _numeric_series(out, "ml_probability")

    # theover is a legacy column mapping we still ingest
    model_prob = ml.where(ml.notna(), theover)
    out["display_probability"] = model_prob.round(3)
    kalshi_prob = _numeric_series(out, "kalshi_probability") if "kalshi_probability" in out.columns else pd.Series([pd.NA]*len(out), index=out.index)

    calibrated = compute_blended_probability(
        p_market=out["market_probability"],
        p_kalshi=kalshi_prob,
        p_ml=model_prob,
        p_theover=theover,  # Use existing variable
        p_sentiment=_numeric_series(out, "sentiment_diff", default=0.5),
        league=_string_series(out, "league"),
        market_type=_string_series(out, "market_type")
    )

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


def _build_total_rows(normalized: pd.DataFrame) -> list[pd.DataFrame]:
    """Expand a raw totals upload into total_over + total_under rows."""
    total_line = _first_existing_numeric(normalized, ["total_line", "total", "line", "points"])
    total_prob = _first_existing_numeric(normalized, ["theover_probability", "winprobability", "win_probability", "probability"])

    total_odds = _first_existing_numeric(normalized, ["odds_american", "american_odds", "odds"], default=pd.NA)

    pick_text = _string_series(normalized, "pick").str.lower()
    pick_text = pick_text.where(pick_text.str.len().gt(0), _string_series(normalized, "best_pick").str.lower())
    under_selected = pick_text.str.contains("under", na=False)
    over_selected = pick_text.str.contains("over", na=False)

    over_prob = total_prob
    under_prob = (1 - total_prob).where(total_prob.notna(), pd.NA)

    # If uploaded probability is for an explicit UNDER pick, invert the assignment.
    over_prob = over_prob.where(~under_selected, (1 - total_prob).where(total_prob.notna(), pd.NA))
    under_prob = under_prob.where(~under_selected, total_prob)

    # If explicit OVER pick is provided, keep default orientation (prob belongs to OVER).
    over_prob = over_prob.where(~over_selected, total_prob)
    under_prob = under_prob.where(~over_selected, (1 - total_prob).where(total_prob.notna(), pd.NA))

    base_cols = [c for c in ["league", "home_team", "away_team", "game_date", "game_time_est"] if c in normalized.columns]
    base = normalized[base_cols].copy()

    total_over = base.copy()
    total_over["market_type"] = "total_over"
    total_over["spread_line"] = pd.NA
    total_over["total_line"] = total_line
    total_over["theover_probability"] = over_prob
    total_over["odds_american"] = total_odds

    total_under = base.copy()
    total_under["market_type"] = "total_under"
    total_under["spread_line"] = pd.NA
    total_under["total_line"] = total_line
    total_under["theover_probability"] = under_prob
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
        # Drop moneyline rows from spreads file — TheOver exports NHL moneylines in the sides CSV
        if file_type == "spreads" and "market" in normalized.columns:
            normalized = normalized[
                normalized["market"].str.lower().str.strip().ne("moneyline")
            ].copy()
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

    # 1. Edge Calculation (Brain vs Market)
    ml_prob = pd.to_numeric(final_df.get('ml_probability'), errors='coerce')
    theover_prob = pd.to_numeric(final_df.get('theover_probability'), errors='coerce')
    expected_value = pd.to_numeric(final_df.get('expected_value', 0.0), errors='coerce').fillna(0.0)

    triple_filter_edge = ml_prob - theover_prob

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

def build_best_picks_df(analysis_df: pd.DataFrame, diagnostics_out: dict | None = None) -> pd.DataFrame:
    logger.info(f"BEST PICKS AUDIT: Received analysis_df with {len(analysis_df)} rows")
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)
    if "market_type" not in analysis_df.columns:
        raise ValueError("analysis_df missing market_type before best-pick construction")

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
        blended = best["calibrated_probability"]
        gap = blended - kalshi_prob
        agrees_mask = (is_kalshi_available & gap.ge(0.03)).fillna(False).astype(bool)
        disagrees_mask = (is_kalshi_available & gap.le(-0.03)).fillna(False).astype(bool)
        best.loc[is_kalshi_available, "consensus_agreement"] = "Neutral"
        best.loc[agrees_mask, "consensus_agreement"] = "Agrees"
        best.loc[disagrees_mask, "consensus_agreement"] = "Disagrees"


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
        elif "h2h" in market_type.lower():
            if "market_probability" in best.columns and "ml_probability" in best.columns:
                if pd.isna(best.at[idx, "market_probability"]) and pd.isna(best.at[idx, "ml_probability"]):
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
            SIDE_MIN_WIN_PROB,
            NEUTRAL_ACTIONABLE_MIN_PROB, NEUTRAL_ACTIONABLE_MIN_EV, NEUTRAL_ACTIONABLE_MIN_EDGE,
            DISAGREES_ACTIONABLE_MIN_PROB, DISAGREES_ACTIONABLE_MIN_EV, DISAGREES_ACTIONABLE_MIN_EDGE
        )

        is_kalshi_divergence = False
        is_spread_divergence_override = False
        if pd.notna(ml_prob) and pd.notna(kalshi_prob):
            if abs(float(ml_prob) - float(kalshi_prob)) > 0.20:
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

        suspicious_data_flag = bool(suspicious_reasons)
        suspicious_reasons_str = "; ".join(dict.fromkeys(suspicious_reasons))
        divergence_viability_pass = (
            pd.notna(ev)
            and pd.notna(edge)
            and float(ev) >= DIVERGENCE_HIGH_VARIANCE_MIN_EV
            and float(edge) >= DIVERGENCE_HIGH_VARIANCE_MIN_EDGE
            and float(win_prob) >= DIVERGENCE_HIGH_VARIANCE_MIN_PROB
        )

        if is_missing_line:
            status = "Missing Line"
            status_reason = "Missing odds or numerical line"
            blocker_stage = "line_integrity_guardrail"
        elif is_nba_extreme_spread:
            status = "No Play"
            status_reason = "NBA spread > 12.0 (Resting/Tanking guardrail)"
            blocker_stage = "line_integrity_guardrail"
        elif is_kalshi_divergence:
            if divergence_viability_pass:
                status = "High Variance/Speculative"
                status_reason = "High Variance: capped due to divergence (ML and Kalshi probability diverge by > 20%)"
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
        elif not pd.isna(ev) and not pd.isna(edge):

            consensus_agr = str(best.at[idx, "consensus_agreement"]) if "consensus_agreement" in best.columns else "No Kalshi"

            is_side_market = market_type in {"spread_home", "spread_away", "h2h_home", "h2h_away"}
            is_total_market = "total" in market_type.lower()

            status_metric_basis = "effective"
            effective_ev = ev
            if is_total_market:
                from app_core.weights_config import FALLBACK_HEAVY_TOTAL_EV_MULTIPLIER
                is_fallback_heavy = diagnostics_out.get("is_fallback_heavy", False) if diagnostics_out else False
                if is_fallback_heavy and ev > 0:
                    effective_ev = ev * FALLBACK_HEAVY_TOTAL_EV_MULTIPLIER

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
                req_prob = max(req_prob, MLB_OVER_ACTIONABLE_MIN_PROB)
                req_ev = max(req_ev, MLB_OVER_ACTIONABLE_MIN_EV)
                req_edge = max(req_edge, MLB_OVER_ACTIONABLE_MIN_EDGE)
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

            # Apply Consensus Overlay Logic ONLY if profile is STRICT
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
        actionable_side_mask = actionable_mask & market_type_str.str.contains("spread|h2h", na=False)
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

            ev_margin = 0.010
            edge_margin = 0.010
            prob_margin = 0.050

            blocked_stages = {
                "suspicious_data_guardrail",
                "data_fallback_guardrail",
                "line_integrity_guardrail",
                "fallback_heavy_guardrail",
            }

            side_candidate_mask = (
                best["Pick_Status"].astype(str).isin({"High Variance/Speculative", "Below Threshold"})
                & market_type_str.str.contains("spread|h2h", na=False)
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
        final_actionable_side_mask = final_actionable_mask & market_type_str.str.contains("spread|h2h", na=False)
        final_actionable_total_mask = final_actionable_mask & market_type_str.str.contains("total", na=False)
        if final_actionable_total_mask.any() and not final_actionable_side_mask.any():
            weakest_ev = pd.to_numeric(best.loc[final_actionable_mask, "effective_expected_value"], errors="coerce").min(skipna=True)
            weakest_edge = pd.to_numeric(best.loc[final_actionable_mask, "effective_edge"], errors="coerce").min(skipna=True)
            weakest_prob = pd.to_numeric(best.loc[final_actionable_mask, "effective_win_probability"], errors="coerce").min(skipna=True)
            post_rank_candidate_mask = (
                best["Pick_Status"].astype(str).isin({"High Variance/Speculative", "Below Threshold"})
                & market_type_str.str.contains("spread|h2h", na=False)
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
        final_actionable_side_mask = final_actionable_mask & final_market_type_str.str.contains("spread|h2h", na=False)
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
        best["matched_live_spread_line"] = pd.to_numeric(best.get("live_spread_line"), errors="coerce")
        best["matched_live_total_line"] = pd.to_numeric(best.get("live_total_line"), errors="coerce")
        best["upload_spread_line"] = pd.to_numeric(best.get("uploaded_spread_line"), errors="coerce")
        best["upload_total_line"] = pd.to_numeric(best.get("uploaded_total_line"), errors="coerce")
        best["base_spread_line"] = pd.to_numeric(best.get("spread_line"), errors="coerce")
        best["base_total_line"] = pd.to_numeric(best.get("total_line"), errors="coerce")

        market_type_norm = best["market_type"].astype(str).str.lower()
        is_spread = market_type_norm.isin({"spread_home", "spread_away"})
        is_total = market_type_norm.isin({"total_over", "total_under"})
        line_source_norm = best.get("line_source", pd.Series([""] * len(best), index=best.index)).astype(str).str.lower()
        live_match = line_source_norm.str.contains("live", na=False)

        best["market_line_source"] = np.where(live_match, "live", np.where(best["upload_spread_line"].notna() | best["upload_total_line"].notna(), "upload", "base"))
        best["market_line_source_detail"] = np.where(live_match, line_source_norm.replace("", "live"), np.where(best["market_line_source"] == "upload", "uploaded_line", "base_generated_line"))

        best["market_line_used"] = pd.NA
        best.loc[is_spread & live_match, "market_line_used"] = best.loc[is_spread & live_match, "matched_live_spread_line"]
        best.loc[is_spread & ~live_match & best["upload_spread_line"].notna(), "market_line_used"] = best.loc[is_spread & ~live_match & best["upload_spread_line"].notna(), "upload_spread_line"]
        best.loc[is_spread & best["market_line_used"].isna(), "market_line_used"] = best.loc[is_spread & best["market_line_used"].isna(), "base_spread_line"]
        best.loc[is_total & live_match, "market_line_used"] = best.loc[is_total & live_match, "matched_live_total_line"]
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
        best.loc[(~live_match) & (is_spread | is_total), "line_provenance_warning"] = "Non-live line source used for best_pick"
        if (mismatch_mask | missing_line_mask).any():
            logger.warning("Line consistency issues detected rows=%s", int((mismatch_mask | missing_line_mask).sum()))

    final_best_df = best[BEST_PICK_COLUMNS].copy()
    final_best_df = ensure_best_pick_export_columns(final_best_df, diagnostics_out=diagnostics_out)

    if diagnostics_out is not None:
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
        "generated": {"spread_home": 0, "spread_away": 0, "total_over": 0, "total_under": 0, "h2h_home": 0, "h2h_away": 0},
        "filtered": {"spread_home": 0, "spread_away": 0, "total_over": 0, "total_under": 0, "h2h_home": 0, "h2h_away": 0},
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
        matchup_id = row.get("matchup_id")
        live_canon_key = row.get("_canon_key")

        league_str = str(row.get("league", "")).upper()
        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        game_date = str(row.get("game_date", ""))
        et_date = str(row.get("_et_day", ""))

        # Determine which markets to emit for the game
        # NHL gets H2H and Totals. Others get Spreads and Totals.
        if league_str == "NHL":
            candidate_markets = ["h2h_home", "h2h_away", "total_over", "total_under"]
        else:
            candidate_markets = ["spread_home", "spread_away", "total_over", "total_under"]

        for m in candidate_markets:
            diag_counts["generated"][m] += 1

        market_mappings = {
            "spread_home": ("home_price", "home_point"),
            "spread_away": ("away_price", "away_point"),
            "total_over": ("over_price", "over_point"),
            "total_under": ("under_price", "under_point"),
            "h2h_home": ("h2h_home_price", None),
            "h2h_away": ("h2h_away_price", None)
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

        # Merge theover enrichment columns
        theover_cols_to_merge = ["matchup_id", "market_type", "theover_probability", "ml_probability"]
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
    ml_probability = _numeric_series(merged, "ml_probability")

    market_type = _string_series(merged, "market_type").str.lower()
    spread_model = ml_probability.where(ml_probability.notna(), theover_probability)

    total_model = (0.6 * theover_probability) + (0.4 * ml_probability)
    total_model = total_model.where(total_model.notna(), theover_probability.where(theover_probability.notna(), ml_probability))

    # Group Spread and H2H (Moneyline) as 'Sides' for probability logic
    is_side = market_type.str.contains("spread|h2h", case=False, na=False)
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
    sentiment_series = _numeric_series(merged, "sentiment_diff", 0.0).apply(lambda x: 0.5 + (x * 0.5)) if "sentiment_diff" in merged.columns else pd.Series([0.5]*len(merged), index=merged.index)
    calibrated_probability = compute_blended_probability(
        p_market=merged["market_probability"],
        p_kalshi=kalshi_probability,
        p_ml=model_probability,
        p_theover=theover_probability,  # Use existing variable
        p_sentiment=_numeric_series(merged, "sentiment_diff", default=0.5),
        league=_string_series(merged, "league"),
        market_type=_string_series(merged, "market_type")
    )

    merged["theover_probability"] = theover_probability
    merged["model_probability"] = model_probability
    merged["display_probability"] = model_probability.round(3)
    merged["calibrated_probability"] = calibrated_probability
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

                return (0.5 + (current_prob - 0.5).abs() + (ev * 2.0).clip(-0.2, 0.2)).clip(0.01, 0.99)
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
    from core.kelly_optimizer import kelly_fraction, add_kelly_bet_sizing, apply_simultaneous_kelly
    from core.smart_parlay_engine import generate_smart_parlays

    # We now fully delegate to smart_parlay_engine
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    parlays_df = generate_smart_parlays(best_picks_df, num_rr_candidates=5)

    if parlays_df.empty:
        return parlays_df

    # Assume a default $1000 bankroll for Kelly fractional sizing during export if not provided
    parlays_df = add_kelly_bet_sizing(parlays_df, bankroll=1000.0, fraction=0.125) # 1/8th Kelly
    parlays_df = apply_simultaneous_kelly(parlays_df, bankroll=1000.0, max_exposure=0.05)

    return parlays_df

def optimize_portfolio_allocation(best_picks_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    portfolio = best_picks_df.copy()
    portfolio = portfolio[_string_series(portfolio, "best_pick").str.strip().str.len() > 0].copy()
    if portfolio.empty:
        return pd.DataFrame()

    portfolio["decimal_odds"] = _numeric_series(portfolio, "decimal_odds").fillna(
        _numeric_series(portfolio, "odds_american", -110.0).apply(american_to_decimal)
    )
    portfolio = add_kelly_bet_sizing(portfolio, bankroll=bankroll, fraction=0.25)
    if "recommended_bet" not in portfolio.columns:
        portfolio["recommended_bet"] = 0.0

    cols = [
        "league", "home_team", "away_team", "best_pick",
        "calibrated_probability", "expected_value", "edge",
        "decimal_odds", "recommended_bet",
    ]
    for col in cols:
        if col not in portfolio.columns:
            portfolio[col] = pd.NA
    return portfolio[cols].sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=30, simulations=200)
