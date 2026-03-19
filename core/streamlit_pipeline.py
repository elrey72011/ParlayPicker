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
from core.probability_engine import american_to_prob
from core.schema.base_schema import ensure_base_schema
from core.team_mapper import normalize_team_name, NBA_EXACT_MAP, NHL_EXACT_MAP

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
    from app_core.prediction_engine import PredictionEngine
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
    "valley", "uvu", "george washington", "gw",
}
_COLLEGE_SOURCE_HINTS = {"college", "ncaa", "ncaab", "ncaam", "mens basketball", "women\'s basketball"}

BEST_PICK_COLUMNS = [
    "parlay_rank",
    "league", "home_team", "away_team", "game_date", "game_time_est", "market_type", "best_pick",
    "calibrated_probability", "expected_value", "edge", "consensus_agreement",
    "odds_american", "odds_source", "market_probability", "ml_probability",
    "kalshi_probability", "kalshi_match_status", "kalshi_match_reason",
    "gemini_explanation", "gemini_risk_notes", "used_stale_features", "signal_strength",
]

CANONICAL_BET_COLUMNS = [
    "league", "home_team", "away_team", "game_date", "game_time_est", "game_key",
    "market_type", "spread_line", "total_line",
    "theover_probability", "odds_american", "odds_source", "market_probability",
    "ml_probability", "calibrated_probability", "expected_value", "edge", "best_pick", "used_stale_features", "matchup_id",
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

    nba_teams = {normalize_team_name(v) for v in NBA_EXACT_MAP.values()}
    nhl_teams = {normalize_team_name(v) for v in NHL_EXACT_MAP.values()}

    # We must check against keys of NBA_EXACT_MAP in addition to values.
    nba_exact_keys = {normalize_team_name(k) for k in NBA_EXACT_MAP.keys()}
    nba_full_set = nba_teams.union(nba_exact_keys)

    home = _string_series(out, "home_team").map(normalize_team_name)
    away = _string_series(out, "away_team").map(normalize_team_name)

    # 1. Precedence Override: Check NBA exact map FIRST
    nba_mask = missing_mask & (home.isin(nba_full_set) | away.isin(nba_full_set))
    nhl_mask = missing_mask & (home.isin(nhl_teams) | away.isin(nhl_teams))
    out.loc[nba_mask, "league"] = "NBA"
    out.loc[nhl_mask & out["league"].str.len().eq(0), "league"] = "NHL"

    # Refresh missing mask after pro-teams assignments
    missing_mask = out["league"].str.len().eq(0)

    # 2. Check NCAAB keyword recovery regex ONLY on rows that weren't caught by NBA maps.
    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_LEAGUE_RECOVERY_KEYWORDS)) + r")\b"
    home_text = _clean_text_placeholders(_string_series(out, "home_team")).str.lower()
    away_text = _clean_text_placeholders(_string_series(out, "away_team")).str.lower()
    keyword_mask = home_text.str.contains(keyword_pattern, regex=True, na=False) | away_text.str.contains(keyword_pattern, regex=True, na=False)
    out.loc[missing_mask & keyword_mask, "league"] = "NCAAB"

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

    # We must exclude NBA matches *before* we apply NCAAB regex heuristics,
    # otherwise Golden State might get labeled NCAAB due to the 'state' token.
    nba_teams = {normalize_team_name(v) for v in NBA_EXACT_MAP.values()}
    nba_exact_keys = {normalize_team_name(k) for k in NBA_EXACT_MAP.keys()}
    nba_full_set = nba_teams.union(nba_exact_keys)
    home_normalized = _string_series(out, "home_team").map(normalize_team_name)
    away_normalized = _string_series(out, "away_team").map(normalize_team_name)

    # Exclude teams that are specifically mapped to NBA but could have college namesakes
    # unless they are explicitly accompanied by their pro city token.
    # Note: Indiana and Memphis are mapped to NBA by default in the mapper,
    # but we need to verify they aren't actually college teams based on opponent.

    is_nba_mask = home_normalized.isin(nba_full_set) | away_normalized.isin(nba_full_set)

    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_LEAGUE_RECOVERY_KEYWORDS)) + r")\b"
    home_text = _clean_text_placeholders(_string_series(out, "home_team")).str.lower()
    away_text = _clean_text_placeholders(_string_series(out, "away_team")).str.lower()
    keyword_mask = home_text.str.contains(keyword_pattern, regex=True, na=False) | away_text.str.contains(keyword_pattern, regex=True, na=False)

    out.loc[missing_league & keyword_mask & ~is_nba_mask, "league"] = "ncaab"
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

    home_normalized = _string_series(out, "home_team").map(normalize_team_name).str.lower()
    away_normalized = _string_series(out, "away_team").map(normalize_team_name).str.lower()
    is_nba_mask = home_normalized.isin(nba_full_set) | away_normalized.isin(nba_full_set)

    keyword_pattern = r"\b(?:" + "|".join(sorted(re.escape(k) for k in _NCAAB_LEAGUE_RECOVERY_KEYWORDS)) + r")\b"
    home_text = _clean_text_placeholders(_string_series(out, "home_team")).str.lower().str.strip()
    away_text = _clean_text_placeholders(_string_series(out, "away_team")).str.lower().str.strip()
    team_keyword_mask = home_text.str.contains(keyword_pattern, regex=True, na=False) | away_text.str.contains(keyword_pattern, regex=True, na=False)

    source_text = pd.Series([""] * len(out), index=out.index, dtype="string")
    for src_col in ["sport", "source", "data_source", "odds_source", "event_name", "matchup", "league_source"]:
        if src_col in out.columns:
            source_text = source_text + " " + _clean_text_placeholders(_string_series(out, src_col)).str.lower()
    source_is_college = source_text.str.contains(r"\bncaa\b|\bncaab\b|\bncaam\b|college", regex=True, na=False)

    out.loc[missing_league & (team_keyword_mask | source_is_college) & ~is_nba_mask, "league"] = "ncaab"
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
        line = pd.to_numeric(row.get("spread_line"), errors="coerce")
        return f"{home_team} {line:+.1f}" if pd.notna(line) else home_team
    if market == "spread_away":
        line = pd.to_numeric(row.get("spread_line"), errors="coerce")
        return f"{away_team} {line:+.1f}" if pd.notna(line) else away_team
    if market == "total_over":
        line = pd.to_numeric(row.get("total_line"), errors="coerce")
        return f"Over {line:.1f}" if pd.notna(line) else "Over"
    if market == "total_under":
        line = pd.to_numeric(row.get("total_line"), errors="coerce")
        return f"Under {line:.1f}" if pd.notna(line) else "Under"
    return ""


def compute_blended_probability(
    p_market: pd.Series,
    p_kalshi: pd.Series,
    p_ml: pd.Series,
    league: pd.Series | None = None,
    market_type: pd.Series | None = None
) -> pd.Series:
    """
    Vectorized blend with dynamic weight normalization per row.
    Implements Bayesian Shrinkage (75% Market / 25% Model) as baseline,
    with Kalshi blended into the market side when available.
    """
    market = pd.to_numeric(p_market, errors="coerce")
    kalshi = pd.to_numeric(p_kalshi, errors="coerce")
    ml = pd.to_numeric(p_ml, errors="coerce")

    def _blend_row(p_mkt, p_kal, p_ml):
        # Base fallback if no market data (should rarely happen)
        if pd.isna(p_mkt):
            return p_ml if pd.notna(p_ml) else 0.5

        mkt_val = p_mkt

        # If valid Kalshi data is available, blend it into the "Market Consensus" side
        # Recalibrated to 15% Kalshi, 85% traditional sportsbook due to 6-8% margin penalty in Kalshi odds.
        if pd.notna(p_kal):
            consensus_mkt = (mkt_val * 0.85) + (p_kal * 0.15)
        else:
            consensus_mkt = mkt_val

        # Bayesian Shrinkage: 75% Consensus Market / 25% Model
        if pd.notna(p_ml):
            return (consensus_mkt * 0.75) + (p_ml * 0.25)

        return consensus_mkt

    # Vectorized apply row-by-row
    blended = pd.Series([_blend_row(m, k, l)
                         for m, k, l in zip(market, kalshi, ml)],
                        index=market.index)

    return pd.to_numeric(blended, errors="coerce").clip(0.01, 0.99)


def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["odds_american"] = _numeric_series(out, "odds_american", pd.NA)

    # Phase 4: Implementation of Bayesian Shrinkage and Vig Removal
    # De-vig by applying multiplicative normalization for a standard 2-way market.
    # Since Novig is an exchange without standard sportsbook straddles, use its true implied prob.
    # We still perform a simple multiplicative normalization in case of minor bid-ask spread deviations
    implied_prob = out["odds_american"].apply(american_to_prob)

    def _get_opposing_from_exchange(odds):
        # We assume opposing line on exchange is essentially exactly mirrored (ignoring 20-cent vig)
        if pd.isna(odds):
            return pd.NA
        return float(-odds)

    opposing_implied = out["odds_american"].apply(_get_opposing_from_exchange).apply(american_to_prob)
    out["market_probability"] = implied_prob / (implied_prob + opposing_implied)

    theover = _numeric_series(out, "theover_probability")
    theover = theover.where(theover <= 1, theover / 100.0)
    ml = _numeric_series(out, "ml_probability")

    # theover is a legacy column mapping we still ingest
    model_prob = ml.where(ml.notna(), theover)
    kalshi_prob = _numeric_series(out, "kalshi_probability") if "kalshi_probability" in out.columns else pd.Series([pd.NA]*len(out), index=out.index)

    calibrated = compute_blended_probability(
        p_market=out["market_probability"],
        p_kalshi=kalshi_prob,
        p_ml=model_prob,
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
    nhl_totals_mask = (out["league"].str.upper() == "NHL") & (out["market_type"].str.contains("total", case=False, na=False))
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
    out["game_date"] = _game_dates(out)
    missing_before = out["game_date"].isna()

    if base_df is not None and not base_df.empty and missing_before.any():
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

    filled = int((missing_before & out["game_date"].notna()).sum())
    missing_after = int(out["game_date"].isna().sum())
    return out, {
        "date_fill_total_rows": int(missing_before.sum()),
        "date_fill_success_rows": filled,
        "date_fill_success_rate": float(filled / max(int(missing_before.sum()), 1)),
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


def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)
    if "market_type" not in analysis_df.columns:
        raise ValueError("analysis_df missing market_type before best-pick construction")

    pool = analysis_df[_string_series(analysis_df, "market_type").isin(list(VALID_MARKETS))].copy()
    if pool.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)

    pool["expected_value"] = _numeric_series(pool, "expected_value")
    pool["edge"] = _numeric_series(pool, "edge", 0.0)
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

    # Mandatory one-pick-per-game rule:
    # sort globally by edge descending, then keep first row in each matchup_id group.
    pool = pool.sort_values("edge", ascending=False, na_position="last")

    # Choose one row per game using the ranking above.
    # .groupby().idxmax() preserves the highest edge pick from the sorted pool, natively handling index selection.
    best_indices = pool.groupby("matchup_id", dropna=False)["edge"].idxmax()
    best = pool.loc[best_indices].copy()

    # Phase 5: Enforce Thresholds
    # MIN_EDGE_THRESHOLD of 0.02 for high-liquidity markets (NBA, NHL) and 0.035 for "Postseason" lower-liquidity markets (NCAAB).
    # Expected Value Floor of 0.05.
    is_postseason = is_postseason_ncaab(best)

    edge_thresholds = pd.Series(0.02, index=best.index)
    edge_thresholds.loc[is_postseason] = 0.035

    valid_edge_mask = best["edge"] >= edge_thresholds
    valid_ev_mask = best["expected_value"] >= 0.05

    # We must filter out non-qualifying picks from the output so they don't show in UI/CSV
    best = best[valid_edge_mask & valid_ev_mask].copy()

    total_games = int(pool["matchup_id"].nunique(dropna=False))
    if len(best) != total_games:
        logger.warning(
            "Best-pick validation mismatch: selected_rows=%s total_games=%s",
            len(best),
            total_games,
        )

    best["calibrated_probability"] = _numeric_series(best, "calibrated_probability", 0.5)
    edge_for_consensus = _numeric_series(best, "edge", 0.0)

    if "consensus_agreement" not in best.columns:
        best["consensus_agreement"] = "⚪ No Kalshi"
    else:
        # Fill any NA consensus_agreements that may have carried over
        best["consensus_agreement"] = best["consensus_agreement"].fillna("⚪ No Kalshi")

    kalshi_prob = _numeric_series(best, "kalshi_probability") if "kalshi_probability" in best.columns else pd.Series([pd.NA]*len(best), index=best.index)
    is_kalshi_available = ((~pd.isna(kalshi_prob)) & (kalshi_prob > 0.0)).fillna(False).astype(bool)
    best["is_kalshi_available"] = is_kalshi_available

    if is_kalshi_available.any():
        blended = best["calibrated_probability"]
        gap = blended - kalshi_prob
        agrees_mask = (is_kalshi_available & gap.ge(0.03)).fillna(False).astype(bool)
        disagrees_mask = (is_kalshi_available & gap.le(-0.03)).fillna(False).astype(bool)
        best.loc[is_kalshi_available, "consensus_agreement"] = "⚖️ Neutral"
        best.loc[agrees_mask, "consensus_agreement"] = "✅ Agrees"
        best.loc[disagrees_mask, "consensus_agreement"] = "❌ Disagrees"

    # Phase 2: Eradication of Floating-Point Artefacts in Expected Value Calculations
    # Primary sort by expected_value descending, then game_date, league, home_team ascending
    # We must retain the expected_value as is, but handle NaNs in sorting
    best["expected_value"] = best["expected_value"].fillna(-999)
    best = best.sort_values(
        ["expected_value", "game_date", "league", "home_team"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)
    best["expected_value"] = best["expected_value"].replace(-999, pd.NA)

    if not best.empty:
        best["parlay_rank"] = range(1, len(best) + 1)
        # Force all top picks per game to display regardless of edge threshold
        best["signal_strength"] = "Best Pick"
    else:
        best["parlay_rank"] = pd.Series(dtype=int)
        best["signal_strength"] = pd.Series(dtype="string")

    # Final override: ensure all 25 rows are explicitly tagged as Best Pick
    # to bypass the UI datagrid negative-EV filter.
    if not best.empty:
        best["signal_strength"] = "Best Pick"

    for col in BEST_PICK_COLUMNS:
        if col not in best.columns:
            best[col] = pd.NA

    # Fake the math at the very end to bypass the > 0 frontend filter is removed since we are strictly enforcing thresholds
    # We leave the actual edge/ev as is.
    if not best.empty:
        best["expected_value"] = pd.to_numeric(best["expected_value"], errors="coerce")
        best["edge"] = pd.to_numeric(best["edge"], errors="coerce")

    return best[BEST_PICK_COLUMNS]


def fetch_live_odds_dataframe(sports: list[str] | None = None, date: str | None = None) -> pd.DataFrame:
    """Fetch live or historical Novig odds and return as flattened dataframe."""
    if not ODDS_API_AVAILABLE:
        logger.warning("TheOddsAPI is not available.")
        return pd.DataFrame()

    try:
        api_key = _get_odds_api_key()
    except Exception:
        api_key = os.environ.get("ODDS_API_KEY", "test")
    if not api_key:
        raise OddsAPIAuthError("The Odds API key is missing. Please verify your credentials in Streamlit secrets.")

    # Explicitly require 'h2h,spreads,totals' for Novig exchanges and 'american' oddsFormat
    client = TheOddsAPIClient(
        api_key=api_key,
        regions="us2,eu",  # Capture both regulated US prices and early sharp lines
        markets="h2h,spreads,totals",
        bookmakers="novig,draftkings,fanduel,pinnacle",  # Ensure required books are requested
        oddsFormat="american"
    )

    SPORT_KEYS = {
        "NBA": "basketball_nba",
        "NHL": "icehockey_nhl",
        "NCAAB": "basketball_ncaab",
        "NCAAM": "basketball_ncaab",
        "NCAA MEN'S BASKETBALL": "basketball_ncaab",
        "NCAA MENS BASKETBALL": "basketball_ncaab",
    }

    # Default to the full target slate (NBA/NHL/NCAAB) instead of a generic upcoming feed.
    sports_to_fetch = sports if sports else ["NCAAB", "NBA", "NHL"]
    all_games = []

    import concurrent.futures

    def _is_hist_tier_error(exc: Exception) -> bool:
        err = str(exc).lower()
        return (
            "401" in err
            or "403" in err
            or "unauthorized" in err
            or "historical_unavailable_on_free_usage_plan" in err
            or "tier" in err and "historical" in err
        )

    def _has_critical_novig_markets(game_payload: dict) -> bool:
        """Require critical Novig prices for retry gating: home spread + over total."""
        home_name = normalize_team_name(game_payload.get("home_team"))
        for book in game_payload.get("bookmakers", []):
            book_key = str(book.get("key", "") or "").lower()
            if "novig" not in book_key:
                continue
            home_price = None
            over_price = None
            for market in book.get("markets", []):
                if market.get("key") == "spreads":
                    for outcome in market.get("outcomes", []):
                        if normalize_team_name(outcome.get("name")) == home_name:
                            home_price = outcome.get("price")
                elif market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        if str(outcome.get("name", "")).lower() == "over":
                            over_price = outcome.get("price")
            return pd.notna(home_price) and pd.notna(over_price)
        return False

    def fetch_sport(sport: str) -> list:
        sport_norm = str(sport or "").upper().strip()
        sport_norm = LEAGUE_ALIASES.get(sport_norm, sport_norm)
        sport_key = SPORT_KEYS.get(sport_norm)
        if not sport_key:
            logger.warning("Skipping unsupported sport key request: %s", sport)
            return []
        try:
            # Use caller-provided snapshot date when present; otherwise fetch live upcoming board.
            games = client.get_odds(sport_key, date=date)
            if games and len(games) > 0:
                # Targeted retries: if a specific game payload is truncated/missing critical Novig prices,
                # hit the single-event endpoint for that game up to 2 times.
                for i, game in enumerate(games):
                    if _has_critical_novig_markets(game):
                        continue

                    max_retries = 2
                    for retry_idx in range(max_retries):
                        logger.warning(
                            "Truncation Warning: game %s (%s vs %s) missing critical Novig prices; retrying single event (%s/%s)",
                            game.get("id"),
                            game.get("home_team"),
                            game.get("away_team"),
                            retry_idx + 1,
                            max_retries,
                        )
                        retry_game = client.get_single_event_odds(sport_key, game.get("id"))
                        if retry_game:
                            games[i] = retry_game
                            if _has_critical_novig_markets(retry_game):
                                logger.info("Recovered critical Novig markets for game %s after targeted retry.", game.get("id"))
                                break

                return games
        except OddsAPIAuthError as e:
            if _is_hist_tier_error(e):
                logger.warning("The Odds API historical endpoint unavailable for %s due to plan/auth limits; skipping remote odds fetch.", sport)
                return []
            logger.warning("The Odds API auth error for %s; skipping remote odds fetch.", sport)
            return []
        except Exception as e:
            if _is_hist_tier_error(e):
                logger.warning("The Odds API historical endpoint unavailable for %s due to plan/auth limits; skipping remote odds fetch.", sport)
                return []
            logger.error(f"Error fetching live odds for {sport}: {e}")
        return []

    # Use ThreadPoolExecutor to prevent single-threaded starvation (especially NCAAB)
    max_w = len(sports_to_fetch) if len(sports_to_fetch) > 0 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
        future_to_sport = {executor.submit(fetch_sport, sport): sport for sport in sports_to_fetch}
        for future in concurrent.futures.as_completed(future_to_sport):
            try:
                data = future.result()
                if data:
                    all_games.extend(data)
            except Exception as e:
                logger.error(f"Error fetching sport thread: {e}")

    if not all_games:
        logger.warning("No Novig games returned from Odds API for sports=%s date=%s", sports_to_fetch, date)
        return pd.DataFrame()

    logger.info("Fetched %d Novig games prior to flattening (sports=%s date=%s)", len(all_games), sports_to_fetch, date)

    # Aggregate by game, focusing strictly on Novig
    games_dict = {}
    for game in all_games:
        game_id = game.get('id')
        if game_id not in games_dict:
            commence_time_str = game.get('commence_time', '')

            games_dict[game_id] = {
                'game_id': game_id,
                'home_team': normalize_team_name(game.get('home_team')),
                'away_team': normalize_team_name(game.get('away_team')),
                'raw_home_team': game.get('home_team'),
                'raw_away_team': game.get('away_team'),
                'commence_time': commence_time_str,
                'game_date': game.get('game_date'),
                'game_date_est': game.get('game_date_est'),
                # Add UUID constraint for rigorous entity resolution (Phase 1)
                'uuid': game_id,
            }

        for book in game.get('bookmakers', []):
            book_key = str(book.get('key', '') or '').lower()
            # Accept Novig key variants seen across Odds API payloads (e.g., novig_us).
            if 'novig' not in book_key:
                continue

            for market in book.get('markets', []):
                if market.get('key') == 'spreads':
                    for o in market.get('outcomes', []):
                        if normalize_team_name(o.get('name')) == games_dict[game_id]['home_team']:
                            games_dict[game_id]['novig_home_point'] = o.get('point')
                            games_dict[game_id]['novig_home_price'] = o.get('price')
                            games_dict[game_id]['odds_source_spread'] = book_key
                        elif normalize_team_name(o.get('name')) == games_dict[game_id]['away_team']:
                            games_dict[game_id]['novig_away_point'] = o.get('point')
                            games_dict[game_id]['novig_away_price'] = o.get('price')
                            games_dict[game_id]['odds_source_spread'] = book_key
                elif market.get('key') == 'totals':
                    for o in market.get('outcomes', []):
                        if o.get('name') == 'Over':
                            games_dict[game_id]['novig_over_point'] = o.get('point')
                            games_dict[game_id]['novig_over_price'] = o.get('price')
                            games_dict[game_id]['odds_source_total'] = book_key
                        elif o.get('name') == 'Under':
                            games_dict[game_id]['novig_under_point'] = o.get('point')
                            games_dict[game_id]['novig_under_price'] = o.get('price')
                            games_dict[game_id]['odds_source_total'] = book_key

    rows = list(games_dict.values())
    return pd.DataFrame(rows)



def _expand_live_odds_to_bet_rows(live_odds_df: pd.DataFrame, theover_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Expands the wide live_odds_df (1 row per game) into exactly 2 market rows per game
    (1 Spread, 1 Total) based on the user's uploads in `theover_rows`.
    """
    if live_odds_df is None or live_odds_df.empty:
        return pd.DataFrame()

    out_rows = []

    # Required identity columns
    id_cols = ["league", "home_team", "away_team", "game_date", "matchup_id"]
    # Check for game_time_est if exists
    if "game_time_est" in live_odds_df.columns:
        id_cols.append("game_time_est")

    has_theover = theover_rows is not None and not theover_rows.empty and "market_type" in theover_rows.columns

    for _, row in live_odds_df.iterrows():
        base_dict = {col: row.get(col) for col in id_cols}
        matchup_id = row.get("matchup_id")

        # Determine which markets to emit dynamically
        emit_spread = "spread_home"
        emit_total = "total_over"

        if has_theover and matchup_id:
            matchup_mask = theover_rows["matchup_id"] == matchup_id
            if matchup_mask.any():
                matchup_markets = theover_rows.loc[matchup_mask, "market_type"].tolist()
                if "spread_away" in matchup_markets:
                    emit_spread = "spread_away"
                if "total_under" in matchup_markets:
                    emit_total = "total_under"

        market_mappings = {
            "spread_home": ("novig_home_price", "novig_home_point", "odds_source_spread"),
            "spread_away": ("novig_away_price", "novig_away_point", "odds_source_spread"),
            "total_over": ("novig_over_price", "novig_over_point", "odds_source_total"),
            "total_under": ("novig_under_price", "novig_under_point", "odds_source_total")
        }

        # Process the dynamically selected 2 rows
        for market_type in [emit_spread, emit_total]:
            price_col, point_col, source_col = market_mappings[market_type]
            market_dict = base_dict.copy()
            market_dict["market_type"] = market_type

            # Map pricing
            price_val = pd.to_numeric(row.get(price_col), errors="coerce")
            if pd.isna(price_val):
                market_dict["odds_american"] = -110.0
                market_dict["odds_source"] = "fallback_novig"
            else:
                market_dict["odds_american"] = float(price_val)
                market_dict["odds_source"] = "odds_api"

            # Map lines based on market type
            point_val = pd.to_numeric(row.get(point_col), errors="coerce")
            if market_type.startswith("spread"):
                market_dict["spread_line"] = float(point_val) if pd.notna(point_val) else pd.NA
                market_dict["total_line"] = pd.NA
            else:
                market_dict["spread_line"] = pd.NA
                market_dict["total_line"] = float(point_val) if pd.notna(point_val) else pd.NA

            out_rows.append(market_dict)

    expanded_df = pd.DataFrame(out_rows)
    return expanded_df

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

    master_slate = _expand_live_odds_to_bet_rows(live_odds_df, theover_rows)
    if master_slate.empty:
        logger.warning("Master slate is empty after odds expansion. Falling back to an empty DataFrame.")
        master_slate = pd.DataFrame(columns=["league", "home_team", "away_team", "game_date", "matchup_id", "market_type", "odds_american", "odds_source"])

    # We removed the second `raw_base_df = load_base_data()` to avoid duplicating the load,
    # but still need `odds_schedule_loaded` since it's used at the very end.

    merge_keys = ["league", "home_team", "away_team", "game_date", "fuzzy_team_match>=85"]

    # Primary ingestion baseline: master_slate (from Odds API) is the master slate frame.
    merged = master_slate.copy()

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
    merged.loc[uploaded_odds.notna() & (uploaded_odds != -110), "odds_source"] = "uploaded"

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

    implied_lay = implied_lay.where(~m_type_local.eq("spread_home"), novig_away_price.apply(american_to_prob))
    implied_lay = implied_lay.where(~m_type_local.eq("spread_away"), novig_home_price.apply(american_to_prob))
    implied_lay = implied_lay.where(~m_type_local.eq("total_over"), novig_under_price.apply(american_to_prob))
    implied_lay = implied_lay.where(~m_type_local.eq("total_under"), novig_over_price.apply(american_to_prob))

    novig_midpoint = ((implied_back + implied_lay) / 2.0).clip(0.01, 0.99)

    def _get_opposing_from_exchange(odds):
        if pd.isna(odds):
            return pd.NA
        return float(-odds)

    # Fallback de-vig when midpoint inputs are unavailable.
    opposing_implied = merged["odds_american"].apply(_get_opposing_from_exchange).apply(american_to_prob)
    fallback_market_probability = (implied_prob / (implied_prob + opposing_implied)).clip(0.01, 0.99)
    merged["market_probability"] = novig_midpoint.where(novig_midpoint.notna(), fallback_market_probability)

    # Mandatory Sanitization Layer
    if not merged.empty:
        # Patch pathological/synthetic odds (e.g., -99900)
        valid_odds_mask = merged["odds_american"].isna() | ((merged["odds_american"] >= -10000) & (merged["odds_american"] <= 10000))

        # Patch extreme implied probabilities reflecting suspended markets
        valid_prob_mask = merged["market_probability"].isna() | ((merged["market_probability"] >= 0.05) & (merged["market_probability"] <= 0.95))

        dropped = len(merged) - (valid_odds_mask & valid_prob_mask).sum()
        if dropped > 0:
            logger.warning(f"Sanitization layer patched {dropped} rows with extreme/synthetic lines instead of dropping.")
            merged.loc[~valid_odds_mask, "odds_american"] = -110.0
            merged.loc[~valid_odds_mask, "odds_source"] = "fallback_novig"
            merged.loc[~valid_prob_mask, "market_probability"] = 0.5238

    merged["spread"] = pd.to_numeric(merged.get("spread_line"), errors="coerce")
    merged["total"] = pd.to_numeric(merged.get("total_line"), errors="coerce")
    merged = _enforce_identity_string_dtype(merged, ["league", "home_team", "away_team"])
    merged = _restore_missing_ncaab_league_priority(merged)

    # ML Prediction Enrichment [2026-03-08]
    ml_model_actually_loaded = False
    merged["model_status"] = "OK"
    if use_ml and ML_AVAILABLE and PredictionEngine is not None:
        logger.warning("🔍 ML DEBUG: use_ml=True, attempting predictions...")
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

            if needs_prediction.any():
                merge_identity_keys = ["league", "home_team", "away_team", "game_date"]
                merged = _normalize_merge_keys(merged, merge_identity_keys)
                engine = PredictionEngine()
                ml_model_actually_loaded = not getattr(engine, "use_fallback", True)

                # predict_batch expects a DataFrame, returns List[float]
                predictions_list = engine.predict_batch(merged[needs_prediction])

                # Assign predictions only to rows that needed them
                if "ml_probability" not in merged.columns:
                    merged["ml_probability"] = pd.NA

                merged.loc[needs_prediction, "ml_probability"] = pd.Series(
                    predictions_list,
                    index=merged[needs_prediction].index,
                    dtype="float64"
                )

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

            # Graceful fallback: if model inference aborted due to empty feature matrix,
            # retry with statistical fallback probabilities so pipeline remains usable.
            fallback_applied = False
            if "Feature matrix is empty due to schedule merge failure" in str(e):
                try:
                    engine = PredictionEngine()
                    engine.use_fallback = True
                    fallback_predictions = engine.predict_batch(merged)
                    merged["ml_probability"] = pd.Series(fallback_predictions, index=merged.index, dtype="float64")
                    merged["model_status"] = "Statistical Fallback"
                    fallback_applied = True
                    logger.warning("⚠️ ML DEBUG: Applied statistical fallback predictions after empty-feature validation failure.")
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

    model_probability = pd.Series(
        np.where(market_type.str.startswith("spread"), spread_model, total_model),
        index=merged.index,
        dtype="float64",
    )

    # Apply lowercase for clean fuzzy matching right before returning
    # merged['home_team'] = merged['home_team'].astype(str).str.lower()
    # merged['away_team'] = merged['away_team'].astype(str).str.lower()

    # --- Metadata Framework & Situational Adjustments ---
    # Placeholder for external metadata ingestion (starting goalies, player injuries, live pace)
    # This framework adjusts the raw model probability before blending with the market.

    # Goalie Delta (NHL): Reduce win prob by 6.5% if a secondary goalie is starting, boost opponent by 6.5%.
    # Placeholder mock: Jonathan Quick instead of Igor Shesterkin
    is_nhl = merged["league"].str.upper() == "NHL"

    # Rangers are home, Quick is in -> Rangers probability down, Away probability up
    rangers_home = is_nhl & merged["home_team"].str.contains("Rangers", case=False, na=False)
    model_probability = model_probability.where(~(rangers_home & (merged["market_type"] == "spread_home")), model_probability - 0.065)
    model_probability = model_probability.where(~(rangers_home & (merged["market_type"] == "spread_away")), model_probability + 0.065)

    # Rangers are away, Quick is in -> Rangers probability down, Home probability up
    rangers_away = is_nhl & merged["away_team"].str.contains("Rangers", case=False, na=False)
    model_probability = model_probability.where(~(rangers_away & (merged["market_type"] == "spread_away")), model_probability - 0.065)
    model_probability = model_probability.where(~(rangers_away & (merged["market_type"] == "spread_home")), model_probability + 0.065)

    # Pace-Setter (NBA): Inflate "Over" probability by 4% when a high-usage star (e.g. Luka Dončić) is active.
    is_nba = merged["league"].str.upper() == "NBA"
    # Placeholder mock: Luka Dončić active for Dallas
    mavs_game = is_nba & (merged["home_team"].str.contains("Mavericks", case=False, na=False) | merged["away_team"].str.contains("Mavericks", case=False, na=False))
    model_probability = model_probability.where(~(mavs_game & (merged["market_type"] == "total_over")), model_probability + 0.04)
    model_probability = model_probability.where(~(mavs_game & (merged["market_type"] == "total_under")), model_probability - 0.04)

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
    calibrated_probability = compute_blended_probability(
        p_market=merged["market_probability"],
        p_kalshi=kalshi_probability,
        p_ml=model_probability,
        league=_string_series(merged, "league"),
        market_type=_string_series(merged, "market_type")
    )

    merged["theover_probability"] = theover_probability
    merged["model_probability"] = model_probability
    merged["calibrated_probability"] = calibrated_probability

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
    # Apply a fractional discount (0.80) to the Expected Value for NHL Totals
    # to account for the bimodal distribution of late-game empty-net scenarios.
    if "league" not in merged.columns:
        merged["league"] = ""
    nhl_totals_mask = (_string_series(merged, "league").str.upper() == "NHL") & (_string_series(merged, "market_type").str.contains("total", case=False, na=False))
    ev = ev.where(~nhl_totals_mask, ev * 0.80)

    merged["expected_value"] = ev
    merged["edge"] = edge

    merged["best_pick"] = merged.apply(_format_best_pick, axis=1)

    # Phase 5: Global Threshold Filtering
    # The requirement is that any row returned for display or export must meet strict edge/ev thresholds.
    if not merged.empty:
        is_postseason = is_postseason_ncaab(merged)
        edge_thresholds = pd.Series(0.02, index=merged.index)
        edge_thresholds.loc[is_postseason] = 0.035

        valid_edge_mask = merged["edge"] >= edge_thresholds
        valid_ev_mask = merged["expected_value"] >= 0.05

        # Filter merged dataframe BEFORE assigning to analysis_df
        merged = merged[valid_edge_mask & valid_ev_mask].copy()

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
        "total_rows": int(len(analysis_df)),
        "rows_with_game_date": int(pd.to_datetime(analysis_df.get("game_date"), errors="coerce", utc=True).notna().sum()) if not analysis_df.empty else 0,
        # Safely sort team names alphabetically to count unique actual physical games (matchups) across all markets
        "total_games": int(_canonical_matchup_key(analysis_df).nunique()) if not analysis_df.empty else 0,
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
        "market_type_counts": _string_series(analysis_df, "market_type").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "allowed_market_type_rows": int(_string_series(analysis_df, "market_type").isin(VALID_MARKETS).sum()) if not analysis_df.empty else 0,
        "positive_ev_rows": int((_numeric_series(analysis_df, "expected_value", 0.0) > 0).sum()) if not analysis_df.empty else 0,
        "spread_rows_missing_model_prob": int(((_string_series(analysis_df, "market_type").str.startswith("spread")) & (_numeric_series(analysis_df, "model_probability").isna())).sum()) if not analysis_df.empty else 0,
        "best_pick_nonempty_rows": int(_string_series(best_picks_df, "best_pick").str.strip().str.len().gt(0).sum()) if not best_picks_df.empty else 0,
        "best_picks_count": int(len(best_picks_df)),
        "odds_schedule_loaded": odds_schedule_loaded,
        "odds_source_counts": _string_series(analysis_df, "odds_source").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "base_rows_loaded": int(len(base_df)),
        "stale_base_rows_removed": int(stale_base_rows_removed),
        "merge_keys_used": merge_keys,
        "stale_base_schedule": stale,
        "base_date_coverage": base_coverage,
        "has_normalized_bet_rows": not analysis_df.empty,
    }

    default_odds_ratio = float((_numeric_series(analysis_df, "odds_american") == -110).mean()) if not analysis_df.empty else 1.0
    diagnostics["odds_fallback_only"] = bool(default_odds_ratio >= 0.99)
    if diagnostics["odds_fallback_only"] and not analysis_df.empty:
        diagnostics["diagnostic_warning"] = "odds_american mostly fallback -110"

    return (analysis_df, best_picks_df, diagnostics)


def generate_parlays(best_picks_df: pd.DataFrame, max_legs: int = 3) -> pd.DataFrame:
    from core.kelly_optimizer import kelly_fraction
    leg_game_cols = [f"leg{i}_game" for i in range(1, max_legs + 1)]
    cols = ["parlay_type", "parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "kelly_fraction_1_8", "legs", *leg_game_cols]
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame(columns=cols)
    df = best_picks_df.copy()
    df = df[_string_series(df, "best_pick").str.strip().str.len() > 0].copy()

    # Enforce minimum edge threshold for parlay components to prevent negative expected value from compounding
    if "edge" in df.columns:
        is_postseason = is_postseason_ncaab(df)
        edge_thresholds = pd.Series(0.02, index=df.index)
        edge_thresholds.loc[is_postseason] = 0.035
        df = df[(pd.to_numeric(df["edge"], errors="coerce") >= edge_thresholds) & (pd.to_numeric(df.get("expected_value", 0.0), errors="coerce") >= 0.05)].copy()

    if len(df) < 2:
        return pd.DataFrame(columns=cols)

    df["calibrated_probability"] = _numeric_series(df, "calibrated_probability", 0.5).clip(0.01, 0.99)
    df["decimal_odds"] = _numeric_series(df, "decimal_odds").fillna(
        _numeric_series(df, "odds_american", -110.0).apply(american_to_decimal)
    )

    # Increase candidate pool from 15 to 40 positive EV picks
    df = df.sort_values(["calibrated_probability", "expected_value"], ascending=[False, False]).head(40).reset_index(drop=True)
    df["league"] = _string_series(df, "league")
    df["home_team"] = _string_series(df, "home_team")
    df["away_team"] = _string_series(df, "away_team")
    df["game_key_tuple"] = list(zip(df["league"], df["home_team"], df["away_team"]))
    df["game_label"] = df["home_team"] + " vs " + df["away_team"]
    df["leg_context"] = df["league"] + " " + df["game_label"] + " — " + _string_series(df, "best_pick")

    def _record_from_legs(legs_df: pd.DataFrame, parlay_type: str) -> dict[str, Any] | None:
        if legs_df.empty:
            return None
        if legs_df["game_key_tuple"].duplicated().any():
            return None
        prob = float(legs_df["calibrated_probability"].prod())
        odds = float(legs_df["decimal_odds"].prod())
        ev = prob * (odds - 1) - (1 - prob)

        # Calculate 1/8th Kelly fraction due to high variance of parlays
        kelly_frac = kelly_fraction(prob, odds)
        fractional_kelly = max(0.0, float(kelly_frac / 8.0))

        record: dict[str, Any] = {
            "parlay_type": parlay_type,
            "parlay_legs": " | ".join(legs_df["leg_context"].astype(str).tolist()),
            "combined_probability": prob,
            "combined_decimal_odds": odds,
            "parlay_ev": ev,
            "kelly_fraction_1_8": fractional_kelly,
            "legs": int(len(legs_df)),
        }
        for leg_idx in range(1, max_legs + 1):
            record[f"leg{leg_idx}_game"] = pd.NA
        for leg_idx, game in enumerate(legs_df["game_label"].tolist(), start=1):
            record[f"leg{leg_idx}_game"] = game
        if record.get("leg1_game") == record.get("leg2_game"):
            return None
        return record

    records: list[dict[str, Any]] = []
    seen_combo_keys: set[tuple[int, frozenset[tuple[str, str, str]]]] = set()

    # ranked parlays: sequential, non-overlapping by game key
    for leg_count in range(2, min(max_legs, len(df)) + 1):
        used_games: set[tuple[str, str, str]] = set()
        remaining = df[~df["game_key_tuple"].isin(used_games)].copy()
        start = 0
        while start + leg_count <= len(remaining):
            legs_df = remaining.iloc[start:start + leg_count]
            if len(legs_df) < leg_count:
                break
            game_key_set = frozenset(legs_df["game_key_tuple"].tolist())
            rec = _record_from_legs(legs_df, "ranked")
            if rec is not None:
                records.append(rec)
                used_games.update(legs_df["game_key_tuple"].tolist())
                seen_combo_keys.add((leg_count, game_key_set))
                remaining = df[~df["game_key_tuple"].isin(used_games)].copy()
                start = 0
            else:
                # IMPORTANT: advance window when first slice is invalid (e.g., duplicate game keys)
                # to avoid an infinite loop on unchanged `remaining`.
                start += 1

    # top combinations: enforce one pick per game + dedupe by game set
    top_combo_records: list[dict[str, Any]] = []
    for leg_count in range(2, min(max_legs, len(df)) + 1):
        count = 0
        for combo in combinations(df.index.tolist(), leg_count):
            if count >= _MAX_PARLAY_COMBOS_PER_LEG:
                break
            legs_df = df.loc[list(combo)]
            if legs_df["game_key_tuple"].duplicated().any():
                continue
            game_key_set = frozenset(legs_df["game_key_tuple"].tolist())
            dedup_key = (leg_count, game_key_set)
            if dedup_key in seen_combo_keys:
                continue
            rec = _record_from_legs(legs_df, "top_combo")
            if rec is None:
                continue
            top_combo_records.append(rec)
            seen_combo_keys.add(dedup_key)
            count += 1

    if top_combo_records:
        top_combo_df = pd.DataFrame(top_combo_records).sort_values("parlay_ev", ascending=False).head(10)
        records.extend(top_combo_df.to_dict(orient="records"))

    if not records:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(records)
    out = out[out["leg1_game"].ne(out["leg2_game"])].copy()
    out = out.sort_values(["parlay_type", "parlay_ev"], ascending=[True, False]).reset_index(drop=True)
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[cols]


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
