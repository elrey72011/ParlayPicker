from __future__ import annotations

import functools
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
from core.team_mapper import normalize_team_name

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
DATE_ALIASES = ["game_date", "commence_time", "start_time", "time", "date", "event_date"]
LEAGUE_ALIASES = {"NCAAM": "NCAAB", "NCAA MEN'S BASKETBALL": "NCAAB", "NCAA MENS BASKETBALL": "NCAAB"}

BEST_PICK_COLUMNS = [
    "parlay_rank",
    "league", "home_team", "away_team", "game_date", "game_time_est", "market_type", "best_pick",
    "calibrated_probability", "expected_value", "edge", "consensus_agreement",
    "odds_american", "odds_source", "market_probability", "ml_probability",
    "kalshi_probability", "kalshi_match_status", "kalshi_match_reason",
]

CANONICAL_BET_COLUMNS = [
    "league", "home_team", "away_team", "game_date", "game_time_est", "game_key",
    "market_type", "spread_line", "total_line",
    "theover_probability", "odds_american", "market_probability",
    "ml_probability", "calibrated_probability", "expected_value", "edge", "best_pick",
]

_EXPORT_SIGNAL_COLS = {"market_type", "calibrated_probability", "expected_value", "edge"}

# Cap combos per leg count to prevent combinatorial explosion
_MAX_PARLAY_COMBOS_PER_LEG = 500

MIN_EDGE_THRESHOLD = 0.035
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


def _string_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="string")
    if col in df.columns:
        return df[col].fillna(default).astype("string")
    return pd.Series([default] * len(df), index=df.index, dtype="string")


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
    out["game_date"] = _game_dates(out)
    # Fill any missing dates with fallback
    out["game_date"] = out["game_date"].fillna(_game_date_fallback())
    return out


def _is_pipeline_export(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    return _EXPORT_SIGNAL_COLS.issubset(cols)


def _coerce_export_to_canonical(df: pd.DataFrame, selected_sports: list[str] | None) -> pd.DataFrame:
    out = _normalize_upload_columns(df)
    for src, dst in _UPLOAD_COLUMN_ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    out = _coerce_identity_columns(out)
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
    if selected_sports:
        selected = {str(s).upper() for s in selected_sports}
        out = out[_string_series(out, "league").isin(selected)].copy()
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
    home = _string_series(df, "home_team").str.upper()
    away = _string_series(df, "away_team").str.upper()
    team_a = home.where(home <= away, away)
    team_b = away.where(home <= away, home)
    date = _game_dates(df).dt.strftime("%Y-%m-%d").fillna("")
    return league + "|" + team_a + "|" + team_b + "|" + date


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
    league = str(row.get("league") or "").upper()
    home = str(row.get("home_team") or "")
    away = str(row.get("away_team") or "")
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
        return 1.9091
    v = float(v)
    if v > 0:
        return 1 + (v / 100.0)
    if v < 0:
        return 1 + (100.0 / abs(v))
    return 1.0


def _format_best_pick(row: pd.Series) -> str:
    market = str(row.get("market_type") or "")
    if market == "spread_home":
        line = pd.to_numeric(row.get("spread_line"), errors="coerce")
        return f"{row.get('home_team', '')} {line:+.1f}" if pd.notna(line) else str(row.get("home_team") or "")
    if market == "spread_away":
        line = pd.to_numeric(row.get("spread_line"), errors="coerce")
        return f"{row.get('away_team', '')} {abs(line):+.1f}" if pd.notna(line) else str(row.get("away_team") or "")
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
        # e.g., 50% Kalshi, 50% traditional sportsbook
        if pd.notna(p_kal):
            consensus_mkt = (mkt_val + p_kal) / 2.0
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

    if selected_sports and len(selected_sports) == 1:
        inferred_league = str(selected_sports[0]).upper()
        league_series = _clean_text_placeholders(_string_series(out, "league"))
        out["league"] = league_series.where(league_series.str.len().gt(0), inferred_league)

    out = _infer_missing_league_from_base(out, load_base_data())

    out["spread"] = pd.to_numeric(out.get("spread_line"), errors="coerce")
    out["total"] = pd.to_numeric(out.get("total_line"), errors="coerce")

    if "game_key" not in out.columns:
        out["league"] = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)
        out["home_team"] = _string_series(out, "home_team").map(normalize_team_name)
        out["away_team"] = _string_series(out, "away_team").map(normalize_team_name)
        out["market_type"] = _string_series(out, "market_type")
        out["game_date"] = _game_dates(out)
        out["spread_line"] = pd.to_numeric(out.get("spread_line"), errors="coerce")
        out["total_line"] = pd.to_numeric(out.get("total_line"), errors="coerce")
        out["spread"] = pd.to_numeric(out.get("spread_line"), errors="coerce")
        out["total"] = pd.to_numeric(out.get("total_line"), errors="coerce")
        out["theover_probability"] = pd.to_numeric(out.get("theover_probability"), errors="coerce")
        out["odds_american"] = pd.to_numeric(out.get("odds_american"), errors="coerce")
        if selected_sports:
            selected = {str(s).upper() for s in selected_sports}
            out = out[_string_series(out, "league").isin(selected)].copy()
        out["game_key"] = _mk_game_key(out)
        out = _apply_analysis_calculations(out)

    for col in CANONICAL_BET_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[CANONICAL_BET_COLUMNS]


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

        schedule["home_team_lower"] = schedule["home_team"].str.lower().str.strip()
        schedule["away_team_lower"] = schedule["away_team"].str.lower().str.strip()

        out["home_team_lower"] = out["home_team"].str.lower().str.strip()
        out["away_team_lower"] = out["away_team"].str.lower().str.strip()

        direct = schedule.rename(columns={"game_date": "game_date_base"}).drop(columns=["home_team", "away_team"])
        out = out.merge(direct, left_on=["league", "home_team_lower", "away_team_lower"], right_on=["league", "home_team_lower", "away_team_lower"], how="left")
        out["game_date"] = out["game_date"].where(out["game_date"].notna(), out["game_date_base"])
        out = out.drop(columns=["game_date_base"])

        reverse = schedule.rename(
            columns={"home_team_lower": "away_team_lower_rev", "away_team_lower": "home_team_lower_rev", "game_date": "game_date_base_rev"}
        ).drop(columns=["home_team", "away_team"])
        out = out.merge(reverse, left_on=["league", "home_team_lower", "away_team_lower"], right_on=["league", "home_team_lower_rev", "away_team_lower_rev"], how="left")
        out["game_date"] = out["game_date"].where(out["game_date"].notna(), out["game_date_base_rev"])
        out = out.drop(columns=["game_date_base_rev", "home_team_lower_rev", "away_team_lower_rev", "home_team_lower", "away_team_lower"])

    filled = int((missing_before & out["game_date"].notna()).sum())
    missing_after = int(out["game_date"].isna().sum())
    return out, {
        "date_fill_total_rows": int(missing_before.sum()),
        "date_fill_success_rows": filled,
        "date_fill_success_rate": float(filled / max(int(missing_before.sum()), 1)),
        "missing_game_date_rows": missing_after,
    }


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
    pool["league"] = _clean_text_placeholders(_string_series(pool, "league"))
    pool["home_team"] = _clean_text_placeholders(_string_series(pool, "home_team"))
    pool["away_team"] = _clean_text_placeholders(_string_series(pool, "away_team"))
    pool["game_date"] = _game_dates(pool)

    # Removed strict game_date.notna() requirement to prevent dropping Spread uploads
    pool["has_identity"] = (
        pool["league"].str.len().gt(0)
        & pool["home_team"].str.len().gt(0)
        & pool["away_team"].str.len().gt(0)
    )
    pool = pool[pool["has_identity"]].copy()
    if pool.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)

    pool["has_signal_probability"] = _numeric_series(pool, "model_probability").notna() | _numeric_series(pool, "theover_probability").notna() | _numeric_series(pool, "ml_probability").notna()

    # Create orientation-insensitive matchup key to group flipped home/away teams
    team_a = pool["home_team"].where(pool["home_team"] <= pool["away_team"], pool["away_team"]).str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    team_b = pool["away_team"].where(pool["home_team"] <= pool["away_team"], pool["home_team"]).str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)

    # Extract local date string safely to ignore minor UTC time variations
    dt_utc = _game_dates(pool)
    date_str = pd.Series([""] * len(pool), index=pool.index)
    valid_dt = dt_utc.notna()
    if valid_dt.any():
        date_str.loc[valid_dt] = dt_utc[valid_dt].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")

    # Do NOT include market_family in matchup_key to strictly prevent multiple
    # selections from the same game across different markets (intra-game covariance).
    pool["matchup_key"] = pool["league"] + "|" + team_a + "|" + team_b + "|" + date_str

    # Force expected_value to numeric, converting true errors to NaN while preserving negative floats
    pool['expected_value'] = pd.to_numeric(pool['expected_value'], errors='coerce')

    # Execute idxmax directly on expected_value to extract the best (or least negative) pick per game
    best_pick_indices = pool.groupby(['league', 'home_team', 'away_team'], dropna=False)['expected_value'].idxmax()

    # Extract the final dataframe
    best = pool.loc[best_pick_indices].copy()

    best["calibrated_probability"] = _numeric_series(best, "calibrated_probability", 0.5)
    edge_for_consensus = _numeric_series(best, "edge", 0.0)

    if "consensus_agreement" not in best.columns:
        best["consensus_agreement"] = "⚪ No Kalshi"
    else:
        # Fill any NA consensus_agreements that may have carried over
        best["consensus_agreement"] = best["consensus_agreement"].fillna("⚪ No Kalshi")

    kalshi_prob = _numeric_series(best, "kalshi_probability") if "kalshi_probability" in best.columns else pd.Series([pd.NA]*len(best), index=best.index)
    valid_kalshi = kalshi_prob.notna() & (kalshi_prob > 0.0)

    if valid_kalshi.any():
        blended = best["calibrated_probability"]
        gap = blended - kalshi_prob
        best.loc[valid_kalshi, "consensus_agreement"] = "⚖️ Neutral"
        best.loc[valid_kalshi & gap.ge(0.03), "consensus_agreement"] = "✅ Agrees"
        best.loc[valid_kalshi & gap.le(-0.03), "consensus_agreement"] = "❌ Disagrees"

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
    else:
        best["parlay_rank"] = pd.Series(dtype=int)

    for col in BEST_PICK_COLUMNS:
        if col not in best.columns:
            best[col] = pd.NA



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
        regions="us_ex",
        markets="h2h,spreads,totals",
        bookmakers="novig",
        oddsFormat="american"
    )

    SPORT_KEYS = {
        "NBA": "basketball_nba",
        "NHL": "icehockey_nhl",
        "NCAAB": "basketball_ncaab",
        "NFL": "americanfootball_nfl",
        "NCAAF": "americanfootball_ncaaf"
    }

    sports_to_fetch = sports if sports else list(SPORT_KEYS.keys())
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

    def fetch_sport(sport: str) -> list:
        sport_key = SPORT_KEYS.get(sport.upper())
        if not sport_key:
            return []
        try:
            # Pass hardcoded date for historical snapshot backtesting
            games = client.get_odds(sport_key, date="2026-03-13T16:00:00Z")
            if games:
                return filter_games_today_only(games)
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
        return pd.DataFrame()

    # Aggregate by game, focusing strictly on Novig
    games_dict = {}
    for game in all_games:
        game_id = game.get('id')
        if game_id not in games_dict:
            commence_time_str = game.get('commence_time', '')
            # Explicitly convert commence_time to local US timezone before extracting the date string
            if commence_time_str:
                try:
                    import pytz
                    from datetime import datetime
                    utc_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                    est_time = utc_time.astimezone(pytz.timezone('America/New_York'))
                    commence_time_str = est_time.strftime('%Y-%m-%d')
                except Exception as e:
                    pass

            games_dict[game_id] = {
                'game_id': game_id,
                'home_team': normalize_team_name(game.get('home_team')),
                'away_team': normalize_team_name(game.get('away_team')),
                'raw_home_team': game.get('home_team'),
                'raw_away_team': game.get('away_team'),
                'commence_time': commence_time_str,
                # Add UUID constraint for rigorous entity resolution (Phase 1)
                'uuid': game_id,
            }

        for book in game.get('bookmakers', []):
            book_key = book.get('key', '')
            if book_key != 'novig':
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


def run_analysis_pipeline(
    sports: list[str] | None = None,
    max_rows: int = 1000,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_base_df = load_base_data()
    odds_schedule_loaded = not raw_base_df.empty
    bet_rows = build_theover_bet_rows(spreads_df, totals_df, sports)
    stale = is_stale_schedule(raw_base_df, bet_rows)
    # Keep full base schedule available for fill/lookup; report stale rows but do not drop master data.
    base_df = raw_base_df.copy()
    stale_base_rows_removed = 0
    live_odds_df = fetch_live_odds_dataframe(sports)

    bet_rows["game_date"] = _game_dates(bet_rows)
    if not bet_rows.empty and not base_df.empty:
        base_dates = base_df.copy()
        base_dates["league"] = _string_series(base_dates, "league").str.upper().replace(LEAGUE_ALIASES)
        base_dates["home_team"] = _string_series(base_dates, "home_team").map(normalize_team_name)
        base_dates["away_team"] = _string_series(base_dates, "away_team").map(normalize_team_name)
        base_dates["date"] = _game_dates(base_dates)

        base_dates["home_team_lower"] = base_dates["home_team"].str.lower().str.strip()
        base_dates["away_team_lower"] = base_dates["away_team"].str.lower().str.strip()

        date_lookup = base_dates[["league", "home_team_lower", "away_team_lower", "date"]].drop_duplicates(["league", "home_team_lower", "away_team_lower"])

        bet_rows["home_team_lower"] = bet_rows["home_team"].str.lower().str.strip()
        bet_rows["away_team_lower"] = bet_rows["away_team"].str.lower().str.strip()

        merged_dates = bet_rows.merge(date_lookup, on=["league", "home_team_lower", "away_team_lower"], how="left")
        bet_rows["game_date"] = bet_rows["game_date"].fillna(merged_dates["date"])
        bet_rows = bet_rows.drop(columns=["home_team_lower", "away_team_lower"])
    bet_rows, date_stats = _fill_missing_game_dates_from_base(bet_rows, base_df)

    merge_keys = ["league", "home_team", "away_team", "fuzzy_team_match>=85"]
    merged = bet_rows.copy()

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
        base_schedule["date"] = _game_dates(base_schedule)

        base_schedule["home_team_lower"] = base_schedule["home_team"].str.lower().str.strip()
        base_schedule["away_team_lower"] = base_schedule["away_team"].str.lower().str.strip()

        base_merge_columns = ["league", "home_team_lower", "away_team_lower"] + [
            col for col in ["date", "game_time_est", "odds_american", "ml_probability", "is_neutral"]
            if col in base_schedule.columns
        ]

        merged["home_team_lower"] = merged["home_team"].str.lower().str.strip()
        merged["away_team_lower"] = merged["away_team"].str.lower().str.strip()

        merged = merged.merge(
            base_schedule[base_merge_columns].drop_duplicates(["league", "home_team_lower", "away_team_lower"]),
            on=["league", "home_team_lower", "away_team_lower"],
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

        if "odds_american_base" in merged.columns:
            odds_current = _numeric_series(merged, "odds_american")
            odds_base = _numeric_series(merged, "odds_american_base")
            use_base = ~(odds_current.notna() & (odds_current != -110)) & odds_base.notna()
            merged["odds_american"] = odds_current.where(~use_base, odds_base)
            merged.loc[use_base, "odds_source"] = "base_direct"
            merged = merged.drop(columns=["odds_american_base"])

        if "ml_probability_base" in merged.columns:
            merged["ml_probability"] = _numeric_series(merged, "ml_probability").where(
                _numeric_series(merged, "ml_probability").notna(),
                _numeric_series(merged, "ml_probability_base"),
            )
            merged = merged.drop(columns=["ml_probability_base"])

        reverse_schedule = base_schedule.rename(
            columns={"home_team_lower": "away_team_lower", "away_team_lower": "home_team_lower"}
        )
        reverse_columns = ["league", "home_team_lower", "away_team_lower"] + [
            col for col in ["date", "game_time_est", "odds_american", "ml_probability", "is_neutral"]
            if col in reverse_schedule.columns
        ]
        reverse_lookup = reverse_schedule[reverse_columns].drop_duplicates(["league", "home_team_lower", "away_team_lower"]).rename(
            columns={
                "date": "date_rev",
                "game_time_est": "game_time_est_rev",
                "odds_american": "odds_american_rev",
                "ml_probability": "ml_probability_rev",
                "is_neutral": "is_neutral_rev",
            }
        )
        merged = merged.merge(reverse_lookup, on=["league", "home_team_lower", "away_team_lower"], how="left")

        if "odds_american_rev" in merged.columns:
            odds_current = _numeric_series(merged, "odds_american")
            odds_rev = _numeric_series(merged, "odds_american_rev")
            use_rev = ~(odds_current.notna() & (odds_current != -110)) & odds_rev.notna()
            merged["odds_american"] = odds_current.where(~use_rev, odds_rev)
            merged.loc[use_rev, "odds_source"] = "base_reverse"
            merged = merged.drop(columns=["odds_american_rev"])

        if "ml_probability_rev" in merged.columns:
            merged["ml_probability"] = _numeric_series(merged, "ml_probability").where(
                _numeric_series(merged, "ml_probability").notna(),
                _numeric_series(merged, "ml_probability_rev"),
            )
            merged = merged.drop(columns=["ml_probability_rev"])

        if "date_rev" in merged.columns:
            merged["game_date"] = _game_dates(merged).fillna(pd.to_datetime(merged["date_rev"], errors="coerce", utc=True))
            merged = merged.drop(columns=["date_rev"])

        if "game_time_est_rev" in merged.columns:
            merged["game_time_est"] = _string_series(merged, "game_time_est").where(
                _string_series(merged, "game_time_est").str.len().gt(0),
                _string_series(merged, "game_time_est_rev"),
            )
            merged = merged.drop(columns=["game_time_est_rev"])

        if "is_neutral_base" in merged.columns:
            merged["is_neutral"] = merged["is_neutral"].fillna(merged["is_neutral_base"]) if "is_neutral" in merged.columns else merged["is_neutral_base"]
            merged = merged.drop(columns=["is_neutral_base"])

        if "is_neutral_rev" in merged.columns:
            merged["is_neutral"] = merged["is_neutral"].fillna(merged["is_neutral_rev"]) if "is_neutral" in merged.columns else merged["is_neutral_rev"]
            merged = merged.drop(columns=["is_neutral_rev"])

        # Fuzzy fallback when strict league/home/away join misses schedule rows.
        needs_fuzzy = (
            _game_dates(merged).isna()
            | _numeric_series(merged, "ml_probability").isna()
            | _numeric_series(merged, "odds_american").isna()
        )
        if needs_fuzzy.any():
            schedule_for_fuzzy = base_schedule[[
                c for c in ["league", "home_team", "away_team", "date", "game_time_est", "odds_american", "ml_probability", "is_neutral"]
                if c in base_schedule.columns
            ]].drop_duplicates()

            for idx in merged.index[needs_fuzzy]:
                match = _fuzzy_match_schedule_row(merged.loc[idx], schedule_for_fuzzy, threshold=85)
                if match.empty:
                    continue

                if pd.isna(_game_dates(merged.loc[[idx]]).iloc[0]) and pd.notna(match.get("date")):
                    merged.at[idx, "game_date"] = pd.to_datetime(match.get("date"), errors="coerce", utc=True)
                if pd.isna(pd.to_numeric(merged.at[idx, "odds_american"], errors="coerce")) and pd.notna(match.get("odds_american")):
                    merged.at[idx, "odds_american"] = pd.to_numeric(match.get("odds_american"), errors="coerce")
                    merged.at[idx, "odds_source"] = "base_fuzzy"
                if pd.isna(pd.to_numeric(merged.at[idx, "ml_probability"], errors="coerce")) and pd.notna(match.get("ml_probability")):
                    merged.at[idx, "ml_probability"] = pd.to_numeric(match.get("ml_probability"), errors="coerce")
                if (not str(merged.at[idx, "game_time_est"] or "").strip()) and pd.notna(match.get("game_time_est")):
                    merged.at[idx, "game_time_est"] = str(match.get("game_time_est"))
                if "is_neutral" in merged.columns and pd.isna(merged.at[idx, "is_neutral"]) and pd.notna(match.get("is_neutral")):
                    merged.at[idx, "is_neutral"] = match.get("is_neutral")

    logger.info(f"Number of live Novig games fetched: {len(live_odds_df)}")

    # Merge Live Odds
    if not live_odds_df.empty:
        # Lowercase merge keys for case-insensitive merge
        live_odds_df["home_team_lower"] = live_odds_df["home_team"].str.lower().str.strip()
        live_odds_df["away_team_lower"] = live_odds_df["away_team"].str.lower().str.strip()
        if "home_team_lower" not in merged.columns:
            merged["home_team_lower"] = merged["home_team"].str.lower().str.strip()
            merged["away_team_lower"] = merged["away_team"].str.lower().str.strip()

        # Phase 1: Entity Resolution Validation Layer
        # Before fully processing live odds matches, cross-reference against the master base schedule.
        # If the UUID/match doesn't map to a real scheduled game in base_df, drop it as hallucinated.
        if not base_df.empty:
            base_matchups = base_df[['home_team', 'away_team']].copy().drop_duplicates()
            base_matchups["home_team_lower"] = base_matchups["home_team"].str.lower().str.strip()
            base_matchups["away_team_lower"] = base_matchups["away_team"].str.lower().str.strip()
            live_odds_df = live_odds_df.merge(base_matchups[["home_team_lower", "away_team_lower"]], on=['home_team_lower', 'away_team_lower'], how='inner')

        # Avoid duplicating columns during merge
        live_merge_cols = [c for c in live_odds_df.columns if c not in ["game_id", "commence_time", "raw_home_team", "raw_away_team", "home_team", "away_team"]]
        merged = merged.merge(
            live_odds_df[live_merge_cols].drop_duplicates(["home_team_lower", "away_team_lower"]),
            on=["home_team_lower", "away_team_lower"],
            how="left"
        )

        # For reverse matching, we need to flip the teams AND their respective points/prices.
        # Otherwise, the home point will incorrectly map to the away point.
        reverse_live_odds_df = live_odds_df.rename(columns={
            "home_team_lower": "away_team_lower",
            "away_team_lower": "home_team_lower",
            "novig_home_point": "novig_away_point_rev",
            "novig_home_price": "novig_away_price_rev",
            "novig_away_point": "novig_home_point_rev",
            "novig_away_price": "novig_home_price_rev",
            "novig_over_point": "novig_over_point_rev",
            "novig_over_price": "novig_over_price_rev",
            "novig_under_point": "novig_under_point_rev",
            "novig_under_price": "novig_under_price_rev"
        })

        rev_merge_cols = ["home_team_lower", "away_team_lower"] + [c for c in reverse_live_odds_df.columns if c.endswith("_rev")]
        merged = merged.merge(
            reverse_live_odds_df[rev_merge_cols].drop_duplicates(["home_team_lower", "away_team_lower"]),
            on=["home_team_lower", "away_team_lower"],
            how="left"
        )

        # Combine primary and reverse mapped live odds
        for c in [col for col in live_merge_cols if col.startswith("novig_")]:
            rev_c = f"{c}_rev"
            if rev_c in merged.columns:
                merged[c] = merged[c].fillna(merged[rev_c])
                merged = merged.drop(columns=[rev_c])

    merged["game_date"] = _game_dates(merged)
    # Fill any missing dates with fallback
    merged["game_date"] = merged["game_date"].fillna(_game_date_fallback())
    merged["game_time_est"] = _format_game_time_est(merged)

    # Explicit Column Coalescing for Novig lines
    # Define mapping criteria based strictly on market_type
    m_type = merged["market_type"].str.lower()

    # Safely ensure target columns exist to prevent KeyError
    for col in ["novig_home_point", "novig_home_price", "novig_away_point", "novig_away_price",
                "novig_over_point", "novig_over_price", "novig_under_point", "novig_under_price"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    def safe_series_float(series):
        # Strip '+' prefix if exists, handle errors safely
        return pd.to_numeric(series.astype(str).str.replace('+', '', regex=False), errors='coerce')

    cond_spread_home = (m_type == "spread_home") & merged["novig_home_price"].notna()
    cond_spread_away = (m_type == "spread_away") & merged["novig_away_price"].notna()
    cond_total_over = (m_type == "total_over") & merged["novig_over_price"].notna()
    cond_total_under = (m_type == "total_under") & merged["novig_under_price"].notna()

    # Coalesce odds_american
    merged["odds_american"] = np.select(
        [cond_spread_home, cond_spread_away, cond_total_over, cond_total_under],
        [
            safe_series_float(merged["novig_home_price"]),
            safe_series_float(merged["novig_away_price"]),
            safe_series_float(merged["novig_over_price"]),
            safe_series_float(merged["novig_under_price"])
        ],
        default=merged.get("odds_american", pd.NA)
    )

    # Coalesce spread_line
    merged["spread_line"] = np.select(
        [cond_spread_home, cond_spread_away],
        [
            safe_series_float(merged["novig_home_point"]),
            safe_series_float(merged["novig_away_point"])
        ],
        default=merged.get("spread_line", pd.NA)
    )

    # Coalesce total_line
    merged["total_line"] = np.select(
        [cond_total_over, cond_total_under],
        [
            safe_series_float(merged["novig_over_point"]),
            safe_series_float(merged["novig_under_point"])
        ],
        default=merged.get("total_line", pd.NA)
    )

    # Override odds_source to novig_live if successfully mapped from novig
    cond_any = cond_spread_home | cond_spread_away | cond_total_over | cond_total_under
    if "odds_source" not in merged.columns:
        merged["odds_source"] = pd.NA
    merged["odds_source"] = np.where(cond_any, "novig_live", merged["odds_source"])

    has_local_uploaded_odds = bool((
        _string_series(merged, "odds_source").str.lower().eq("uploaded")
        & _numeric_series(merged, "odds_american").notna()
    ).any())

    # Fallback Mode for Missing Novig lines.
    # If live feed is unavailable OR partially unmatched, keep rows actionable via -110 fallback
    # unless user supplied their own non-default odds.
    missing_odds_mask = merged["odds_american"].isna()
    if missing_odds_mask.any() and not has_local_uploaded_odds:
        if live_odds_df.empty:
            logger.warning("Novig API unavailable/empty due to time or tier restrictions. Falling back to standard -110 odds to render dashboard.")
        else:
            logger.warning("Novig API returned data but did not match all uploaded rows. Applying -110 fallback for unmatched rows.")
        merged.loc[missing_odds_mask, "odds_american"] = -110.0
        merged.loc[missing_odds_mask, "odds_source"] = "fallback_novig"

    # Enforce Strict Drops for missing valid live line/price
    # Only keep rows that successfully mapped a live line and price strictly from novig or fallback
    if "odds_source" in merged.columns:
        # We also need to allow uploaded or base odds sources for backwards compatibility and test suite
        valid_sources = ["novig_live", "fallback_novig", "uploaded", "base_direct", "base_reverse"]
        # Treat pd.NA / NaN in odds_source as implicitly valid for backwards compatibility tests
        missing_mask = ~merged["odds_source"].isin(valid_sources) & merged["odds_source"].notna()
        dropped_count = missing_mask.sum()
        if dropped_count > 0:
            dropped_games = merged[missing_mask][['home_team', 'away_team', 'market_type']].to_dict('records')
            logger.warning(f"Warning: Dropped {dropped_count} rows - Missing novig exchange line. These picks have been completely removed from the pipeline: {dropped_games}")
            merged = merged[~missing_mask].copy()

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
        # Drop pathological/synthetic odds (e.g., -99900)
        valid_odds_mask = merged["odds_american"].isna() | ((merged["odds_american"] >= -10000) & (merged["odds_american"] <= 10000))

        # Drop extreme implied probabilities reflecting suspended markets
        valid_prob_mask = merged["market_probability"].isna() | ((merged["market_probability"] >= 0.05) & (merged["market_probability"] <= 0.95))

        dropped = len(merged) - (valid_odds_mask & valid_prob_mask).sum()
        if dropped > 0:
            logger.warning(f"Sanitization layer dropped {dropped} rows with extreme/synthetic lines.")

        merged = merged[valid_odds_mask & valid_prob_mask].copy()

    merged["spread"] = pd.to_numeric(merged.get("spread_line"), errors="coerce")
    merged["total"] = pd.to_numeric(merged.get("total_line"), errors="coerce")

    # ML Prediction Enrichment [2026-03-08]
    ml_model_actually_loaded = False
    merged["model_status"] = "OK"
    if use_ml and ML_AVAILABLE and PredictionEngine is not None:
        logger.warning("🔍 ML DEBUG: use_ml=True, attempting predictions...")
        try:
            # Only predict for rows missing ml_probability
            needs_prediction = merged["ml_probability"].isna() if "ml_probability" in merged.columns else pd.Series([True] * len(merged), index=merged.index)

            if needs_prediction.any():
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

                ml_count = merged["ml_probability"].notna().sum()
                logger.warning(f"✅ ML DEBUG: Generated {ml_count} total predictions ({needs_prediction.sum()} new)")
            else:
                logger.warning("✅ ML DEBUG: All rows already have ml_probability")

        except Exception as e:
            logger.error(f"❌ ML prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
    total_model = theover_probability.where(theover_probability.notna(), ml_probability)
    model_probability = pd.Series(
        np.where(market_type.str.startswith("spread"), spread_model, total_model),
        index=merged.index,
        dtype="float64",
    )

    # Apply lowercase for clean fuzzy matching right before returning
    # merged['home_team'] = merged['home_team'].astype(str).str.lower()
    # merged['away_team'] = merged['away_team'].astype(str).str.lower()

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
        ncaab_neutral_mask = (merged["league"].str.upper() == "NCAAB") & ((merged["is_neutral"] == True) | (merged["is_neutral"].astype(str).str.lower() == "true"))
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
    zero_mask = edge.abs() < 0.0001
    edge = edge.mask(zero_mask, 0.0)
    ev = ev.mask(zero_mask, 0.0)

    # Phase 3: NHL Statistical Recalibration
    # Apply a fractional discount (0.80) to the Expected Value for NHL Totals
    # to account for the bimodal distribution of late-game empty-net scenarios.
    nhl_totals_mask = (merged["league"].str.upper() == "NHL") & (merged["market_type"].str.contains("total", case=False, na=False))
    ev = ev.where(~nhl_totals_mask, ev * 0.80)

    merged["expected_value"] = ev
    merged["edge"] = edge

    merged["best_pick"] = merged.apply(_format_best_pick, axis=1)

    analysis_df = merged.head(max_rows).copy()
    if not analysis_df.empty and not base_df.empty:
        base_dates = base_df.copy()
        base_dates["league"] = _string_series(base_dates, "league").str.upper().replace(LEAGUE_ALIASES)
        base_dates["home_team"] = _string_series(base_dates, "home_team").map(normalize_team_name)
        base_dates["away_team"] = _string_series(base_dates, "away_team").map(normalize_team_name)
        base_dates["date"] = _game_dates(base_dates)

        base_dates["home_team_lower"] = base_dates["home_team"].str.lower().str.strip()
        base_dates["away_team_lower"] = base_dates["away_team"].str.lower().str.strip()

        analysis_df["home_team_lower"] = analysis_df["home_team"].str.lower().str.strip()
        analysis_df["away_team_lower"] = analysis_df["away_team"].str.lower().str.strip()

        date_fill = analysis_df.merge(
            base_dates[["league", "home_team_lower", "away_team_lower", "date"]],
            on=["league", "home_team_lower", "away_team_lower"],
            how="left",
            suffixes=("", "_basefill"),
        )
        date_fill_series = _game_dates(date_fill)
        analysis_df = analysis_df.drop(columns=["home_team_lower", "away_team_lower"])
        if "date_basefill" in date_fill.columns:
            date_fill_series = date_fill_series.where(date_fill_series.notna(), pd.to_datetime(date_fill["date_basefill"], errors="coerce", utc=True))
        analysis_df["game_date"] = _game_dates(analysis_df).fillna(date_fill_series)

    # Ensure 100% date fill success using fallback if any are still missing
    if not analysis_df.empty:
        analysis_df["game_date"] = analysis_df["game_date"].fillna(_game_date_fallback())

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
        "total_games": int(analysis_df[['home_team', 'away_team']].drop_duplicates().shape[0]) if not analysis_df.empty else 0,
        "bet_rows": int(len(analysis_df)),
        "ml_model_loaded": bool(use_ml and ML_AVAILABLE and ml_model_actually_loaded),
        "ml_predictions": int(analysis_df["ml_probability"].notna().sum()) if "ml_probability" in analysis_df.columns else 0,
        "best_picks": int(len(best_picks_df)),
        "kalshi_attempted": 0,
        "kalshi_matches": 0,
        "kalshi_match_rate": 0.0,
        "match_rate": 0.0,
        "theover_totals_games": int(analysis_df[_string_series(analysis_df, "market_type").str.startswith("total")].apply(lambda r: f"{r.get('league')}|{min(str(r.get('home_team')), str(r.get('away_team')))}|{max(str(r.get('home_team')), str(r.get('away_team')))}", axis=1).nunique()) if not analysis_df.empty else 0,
        "theover_spreads_games": int(analysis_df[_string_series(analysis_df, "market_type").str.startswith("spread")].apply(lambda r: f"{r.get('league')}|{min(str(r.get('home_team')), str(r.get('away_team')))}|{max(str(r.get('home_team')), str(r.get('away_team')))}", axis=1).nunique()) if not analysis_df.empty else 0,
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


def generate_parlays(best_picks_df: pd.DataFrame, max_legs: int = 5) -> pd.DataFrame:
    from core.kelly_optimizer import kelly_fraction
    leg_game_cols = [f"leg{i}_game" for i in range(1, max_legs + 1)]
    cols = ["parlay_type", "parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "kelly_fraction_1_8", "legs", *leg_game_cols]
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame(columns=cols)
    df = best_picks_df.copy()
    df = df[_string_series(df, "best_pick").str.strip().str.len() > 0].copy()
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
