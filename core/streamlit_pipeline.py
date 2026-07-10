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

VALID_MARKETS = {"spread_home", "spread_away", "total_over", "total_under", "moneyline_home", "moneyline_away"}
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
PIPELINE_BUILD = "2026-07-10e-batter-exposure-cap"


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
    "game_already_started_flag",
    "uploaded_spread_line", "uploaded_total_line", "live_sp…92750 tokens truncated…or TOTAL generically
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
    # Production betting is suspended for fallback/degraded model rows. A
    # probability produced after model failure is a display diagnostic, not a
    # bettable forecast. This gate is intentionally conservative: if the
    # pipeline cannot prove the row came from the trained model with a clean
    # feature path, Kelly receives zero.
    model_status = _string_series(portfolio, "model_status").str.strip().str.lower()
    stats_source = _string_series(portfolio, "stats_source").str.strip().str.lower()
    fallback_summary = _string_series(portfolio, "fallback_summary_by_league").str.strip().str.lower()
    health_warning = _string_series(portfolio, "run_health_warning").str.strip().str.lower()
    degraded = pd.Series(
        portfolio.get("degraded_feature_subset_flag", False),
        index=portfolio.index,
    ).fillna(False).astype(bool)
    untrusted_model = (
        model_status.isin({"statistical fallback", "neutral fallback", "model failure"})
        | stats_source.isin({"fallback", "failed"})
        | fallback_summary.ne("")
        | health_warning.str.contains("fallback|degraded|staking suspended", regex=True, na=False)
        | degraded
    )
    production_eligible = (
        status.eq("actionable")
        & line_source.eq("live")
        & line_warning.eq("")
        & line_used.notna()
        & line_consistent
        & event_identity_ok
        & (~best_pick_norm.str.contains("unresolved", na=False))
        & (~untrusted_model)
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

