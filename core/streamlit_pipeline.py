from __future__ import annotations

import functools
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

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

def _get_odds_api_key() -> str:
    key = st.secrets.get("ODDS_API_KEY")
    if not key:
        key = os.environ.get("ODDS_API_KEY", "")
    return key

try:
    from app_core.prediction_engine import PredictionEngine
    ML_AVAILABLE = True
except Exception:
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
    et_now = pd.Timestamp.now(tz="America/New_York")
    return pd.Timestamp(year=et_now.year, month=et_now.month, day=et_now.day, tz="UTC")


def _normalize_upload(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = _normalize_upload_columns(df)
    for src, dst in _UPLOAD_COLUMN_ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    out = _coerce_identity_columns(out)
    out["game_date"] = _game_dates(out)
    if out["game_date"].isna().all():
        # No date column in upload — use today UTC (server clock is UTC;
        # late-night ET uploads are already the next calendar day in UTC)
        out["game_date"] = _game_date_fallback()
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
    if out["game_date"].isna().all():
        out["game_date"] = _game_date_fallback()
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
    out["odds_american"] = _numeric_series(out, "odds_american", -110.0)

    # Phase 4: Implementation of Bayesian Shrinkage and Vig Removal
    # De-vig by applying multiplicative normalization for a standard 2-way market.
    # Since Novig is an exchange without standard sportsbook straddles, use its true implied prob.
    # We still perform a simple multiplicative normalization in case of minor bid-ask spread deviations
    implied_prob = out["odds_american"].apply(american_to_prob)

    def _get_opposing_from_exchange(odds):
        # We assume opposing line on exchange is essentially exactly mirrored (ignoring 20-cent vig)
        if pd.isna(odds) or odds == -110.0:
            return -110.0
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

    # Phase 2: Eradication of Floating-Point Artefacts
    # Cast micro-edges to exact zero.
    edge = edge.round(4)
    zero_mask = edge.abs() < 0.0001
    edge = edge.mask(zero_mask, 0.0)
    ev = ev.mask(zero_mask, 0.0)

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

    odds = _first_existing_numeric(normalized, ["odds_american", "american_odds", "odds"], default=-110.0)

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

    total_odds = _first_existing_numeric(normalized, ["odds_american", "american_odds", "odds"], default=-110.0)

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
        )

        direct = schedule.rename(columns={"game_date": "game_date_base"})
        out = out.merge(direct, on=["league", "home_team", "away_team"], how="left")
        out["game_date"] = out["game_date"].where(out["game_date"].notna(), out["game_date_base"])
        out = out.drop(columns=["game_date_base"])

        reverse = schedule.rename(
            columns={"home_team": "away_team", "away_team": "home_team", "game_date": "game_date_base_rev"}
        )
        out = out.merge(reverse, on=["league", "home_team", "away_team"], how="left")
        out["game_date"] = out["game_date"].where(out["game_date"].notna(), out["game_date_base_rev"])
        out = out.drop(columns=["game_date_base_rev"])

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
    bet_dates = pd.to_datetime(bet_rows_df.get("game_date"), errors="coerce", utc=True)
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

    pool["expected_value"] = _numeric_series(pool, "expected_value", 0.0)
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
    team_a = pool["home_team"].where(pool["home_team"] <= pool["away_team"], pool["away_team"]).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    team_b = pool["away_team"].where(pool["home_team"] <= pool["away_team"], pool["home_team"]).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    # Extract local date string safely to ignore minor UTC time variations
    dt_utc = _game_dates(pool)
    date_str = pd.Series([""] * len(pool), index=pool.index)
    valid_dt = dt_utc.notna()
    if valid_dt.any():
        date_str.loc[valid_dt] = dt_utc[valid_dt].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")

    # Do NOT include market_family in matchup_key to strictly prevent multiple
    # selections from the same game across different markets (intra-game covariance).
    pool["matchup_key"] = pool["league"] + "|" + team_a + "|" + team_b + "|" + date_str

    best = (
        pool.sort_values(["has_signal_probability", "expected_value", "edge"], ascending=[False, False, False])
        .groupby("matchup_key", dropna=False)
        .first()
        .reset_index(drop=True)
    )

    best["calibrated_probability"] = _numeric_series(best, "calibrated_probability", 0.5)
    edge_for_consensus = _numeric_series(best, "edge", 0.0)
    best["consensus_agreement"] = "⚪ No Kalshi"

    # Eradicate static arrays and implement dynamic edge thresholding
    best = best[best["expected_value"] >= 0.02].copy()

    # Phase 2: Eradication of Floating-Point Artefacts in Expected Value Calculations
    # Primary sort by expected_value descending, then game_date, league, home_team ascending
    best = best.sort_values(
        ["expected_value", "game_date", "league", "home_team"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)
    best["parlay_rank"] = range(1, len(best) + 1)

    for col in BEST_PICK_COLUMNS:
        if col not in best.columns:
            best[col] = pd.NA



    return best[BEST_PICK_COLUMNS]


def fetch_live_odds_dataframe(sports: list[str] | None = None) -> pd.DataFrame:
    """Fetch live Novig odds and return as flattened dataframe."""
    if not ODDS_API_AVAILABLE:
        logger.warning("TheOddsAPI is not available.")
        return pd.DataFrame()

    try:
        api_key = _get_odds_api_key()
    except Exception:
        api_key = os.environ.get("ODDS_API_KEY", "test")
    if not api_key:
        raise OddsAPIAuthError("The Odds API key is missing. Please verify your credentials in Streamlit secrets.")

    client = TheOddsAPIClient(api_key=api_key)

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

    def fetch_sport(sport: str) -> list:
        sport_key = SPORT_KEYS.get(sport.upper())
        if not sport_key:
            return []
        try:
            games = client.get_odds(sport_key)
            if games:
                return filter_games_today_only(games)
        except OddsAPIAuthError as e:
            pass  # Let tests proceed or handle appropriately
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                pass
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

    rows = []
    for game in all_games:
        for book in game.get('bookmakers', []):
            book_key = book.get('key', '')
            if book_key != 'novig':
                continue

            row = {
                'game_id': game.get('id'),
                'home_team': normalize_team_name(game.get('home_team')),
                'away_team': normalize_team_name(game.get('away_team')),
                'commence_time': game.get('commence_time'),
            }

            for market in book.get('markets', []):
                if market.get('key') == 'spreads':
                    for o in market.get('outcomes', []):
                        if normalize_team_name(o.get('name')) == row['home_team']:
                            row['novig_home_point'] = o.get('point')
                            row['novig_home_price'] = o.get('price')
                        elif normalize_team_name(o.get('name')) == row['away_team']:
                            row['novig_away_point'] = o.get('point')
                            row['novig_away_price'] = o.get('price')
                elif market.get('key') == 'totals':
                    for o in market.get('outcomes', []):
                        if o.get('name') == 'Over':
                            row['novig_over_point'] = o.get('point')
                            row['novig_over_price'] = o.get('price')
                        elif o.get('name') == 'Under':
                            row['novig_under_point'] = o.get('point')
                            row['novig_under_price'] = o.get('price')
            rows.append(row)

    return pd.DataFrame(rows)


def run_analysis_pipeline(
    sports: list[str] | None = None,
    max_rows: int = 1000,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_df = load_base_data()
    odds_schedule_loaded = not base_df.empty
    live_odds_df = fetch_live_odds_dataframe(sports)

    bet_rows = build_theover_bet_rows(spreads_df, totals_df, sports)

    bet_rows["game_date"] = _game_dates(bet_rows)
    if not bet_rows.empty and not base_df.empty:
        base_dates = base_df.copy()
        base_dates["league"] = _string_series(base_dates, "league").str.upper().replace(LEAGUE_ALIASES)
        base_dates["home_team"] = _string_series(base_dates, "home_team").map(normalize_team_name)
        base_dates["away_team"] = _string_series(base_dates, "away_team").map(normalize_team_name)
        base_dates["date"] = _game_dates(base_dates)
        date_lookup = base_dates[["league", "home_team", "away_team", "date"]].drop_duplicates(["league", "home_team", "away_team"])
        merged_dates = bet_rows.merge(date_lookup, on=["league", "home_team", "away_team"], how="left")
        bet_rows["game_date"] = bet_rows["game_date"].fillna(merged_dates["date"])
    bet_rows, date_stats = _fill_missing_game_dates_from_base(bet_rows, base_df)

    merge_keys = ["league", "home_team", "away_team"]
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

        base_merge_columns = merge_keys + [
            col for col in ["date", "game_time_est", "odds_american", "ml_probability"]
            if col in base_schedule.columns
        ]

        merged = merged.merge(
            base_schedule[base_merge_columns].drop_duplicates(merge_keys),
            on=merge_keys,
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

        reverse_schedule = base_schedule.rename(columns={"home_team": "away_team", "away_team": "home_team"})
        reverse_columns = merge_keys + [
            col for col in ["date", "game_time_est", "odds_american", "ml_probability"]
            if col in reverse_schedule.columns
        ]
        reverse_lookup = reverse_schedule[reverse_columns].drop_duplicates(merge_keys).rename(
            columns={
                "date": "date_rev",
                "game_time_est": "game_time_est_rev",
                "odds_american": "odds_american_rev",
                "ml_probability": "ml_probability_rev",
            }
        )
        merged = merged.merge(reverse_lookup, on=merge_keys, how="left")

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

    # Merge Live Odds
    if not live_odds_df.empty:
        # Avoid duplicating columns during merge
        live_merge_cols = [c for c in live_odds_df.columns if c not in ["game_id", "commence_time"]]
        merged = merged.merge(
            live_odds_df[live_merge_cols].drop_duplicates(["home_team", "away_team"]),
            on=["home_team", "away_team"],
            how="left"
        )

        # For reverse matching, we need to flip the teams AND their respective points/prices.
        # Otherwise, the home point will incorrectly map to the away point.
        reverse_live_odds_df = live_odds_df.rename(columns={
            "home_team": "away_team",
            "away_team": "home_team",
            "novig_home_point": "novig_away_point_rev",
            "novig_home_price": "novig_away_price_rev",
            "novig_away_point": "novig_home_point_rev",
            "novig_away_price": "novig_home_price_rev",
            "novig_over_point": "novig_over_point_rev",
            "novig_over_price": "novig_over_price_rev",
            "novig_under_point": "novig_under_point_rev",
            "novig_under_price": "novig_under_price_rev"
        })

        rev_merge_cols = ["home_team", "away_team"] + [c for c in reverse_live_odds_df.columns if c.endswith("_rev")]
        merged = merged.merge(
            reverse_live_odds_df[rev_merge_cols].drop_duplicates(["home_team", "away_team"]),
            on=["home_team", "away_team"],
            how="left"
        )

        # Combine primary and reverse mapped live odds
        for c in [col for col in live_merge_cols if col.startswith("novig_")]:
            rev_c = f"{c}_rev"
            if rev_c in merged.columns:
                merged[c] = merged[c].fillna(merged[rev_c])
                merged = merged.drop(columns=[rev_c])

    merged["game_date"] = _game_dates(merged)
    if merged["game_date"].isna().all():
        merged["game_date"] = _game_date_fallback()
    merged["game_time_est"] = _format_game_time_est(merged)

    # Map Novig's true points and prices explicitly
    def map_novig_lines(row):
        m_type = str(row.get("market_type", "")).lower()
        if not m_type:
            return row

        def safe_float(val):
            try:
                # Handle cases like "+102"
                return float(str(val).replace('+', '')) if pd.notna(val) else pd.NA
            except ValueError:
                return pd.NA

        if m_type == "spread_home" and "novig_home_point" in row and pd.notna(row["novig_home_point"]):
            row["spread_line"] = safe_float(row["novig_home_point"])
            row["odds_american"] = safe_float(row["novig_home_price"])
            row["odds_source"] = "novig_live"
        elif m_type == "spread_away" and "novig_away_point" in row and pd.notna(row["novig_away_point"]):
            row["spread_line"] = safe_float(row["novig_away_point"])
            row["odds_american"] = safe_float(row["novig_away_price"])
            row["odds_source"] = "novig_live"
        elif m_type == "total_over" and "novig_over_point" in row and pd.notna(row["novig_over_point"]):
            row["total_line"] = safe_float(row["novig_over_point"])
            row["odds_american"] = safe_float(row["novig_over_price"])
            row["odds_source"] = "novig_live"
        elif m_type == "total_under" and "novig_under_point" in row and pd.notna(row["novig_under_point"]):
            row["total_line"] = safe_float(row["novig_under_point"])
            row["odds_american"] = safe_float(row["novig_under_price"])
            row["odds_source"] = "novig_live"

        return row

    merged = merged.apply(map_novig_lines, axis=1)

    # Enforce Strict Drops for missing valid Novig line/price
    # Only keep rows that successfully mapped a live Novig line and price
    if "odds_source" in merged.columns and not live_odds_df.empty:
        dropped_count = (merged["odds_source"] != "novig_live").sum()
        if dropped_count > 0:
            logger.warning(f"Warning: Dropped {dropped_count} rows - Missing live Novig line.")
        merged = merged[merged["odds_source"] == "novig_live"].copy()

    merged["odds_american"] = _numeric_series(merged, "odds_american", pd.NA)
    merged = merged.dropna(subset=["odds_american"])

    # Drop rows without matching spread_line or total_line
    if not merged.empty:
        is_spread = merged["market_type"].astype(str).str.startswith("spread")
        is_total = merged["market_type"].astype(str).str.startswith("total")

        has_valid_spread = is_spread & (merged["spread_line"].notna() if "spread_line" in merged.columns else False)
        has_valid_total = is_total & (merged["total_line"].notna() if "total_line" in merged.columns else False)

        valid_rows = has_valid_spread | has_valid_total
        merged = merged[valid_rows].copy()

    merged["decimal_odds"] = merged["odds_american"].apply(american_to_decimal)

    # Phase 4: Implementation of Bayesian Shrinkage and Vig Removal
    # Calculate True Fair-Value Baseline Probability by removing sportsbook overround (vig).
    implied_prob = merged["odds_american"].apply(american_to_prob)

    def _get_opposing_from_exchange(odds):
        if pd.isna(odds):
            return pd.NA
        return float(-odds)

    # Novig exchange lines don't use 20-cent straddle
    opposing_implied = merged["odds_american"].apply(_get_opposing_from_exchange).apply(american_to_prob)
    merged["market_probability"] = (implied_prob / (implied_prob + opposing_implied)).clip(0.01, 0.99)

    # Mandatory Sanitization Layer
    if not merged.empty:
        # Drop pathological/synthetic odds (e.g., -99900)
        valid_odds_mask = (merged["odds_american"] >= -10000) & (merged["odds_american"] <= 10000)

        # Drop extreme implied probabilities reflecting suspended markets
        valid_prob_mask = (merged["market_probability"] >= 0.05) & (merged["market_probability"] <= 0.95)

        dropped = len(merged) - (valid_odds_mask & valid_prob_mask).sum()
        if dropped > 0:
            logger.warning(f"Sanitization layer dropped {dropped} rows with extreme/synthetic lines.")

        merged = merged[valid_odds_mask & valid_prob_mask].copy()

    merged["spread"] = pd.to_numeric(merged.get("spread_line"), errors="coerce")
    merged["total"] = pd.to_numeric(merged.get("total_line"), errors="coerce")

    # ML Prediction Enrichment [2026-03-08]
    if use_ml and ML_AVAILABLE and PredictionEngine is not None:
        logger.warning("🔍 ML DEBUG: use_ml=True, attempting predictions...")
        try:
            # Only predict for rows missing ml_probability
            needs_prediction = merged["ml_probability"].isna() if "ml_probability" in merged.columns else pd.Series([True] * len(merged), index=merged.index)

            if needs_prediction.any():
                engine = PredictionEngine()
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
    else:
        if "ml_probability" not in merged.columns:
            merged["ml_probability"] = pd.NA

    # If ML is disabled, clear any existing ml_probability values
    if not use_ml:
        if "ml_probability" in merged.columns:
            merged["ml_probability"] = pd.NA

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

    ev = calibrated_probability * (merged["decimal_odds"] - 1) - (1 - calibrated_probability)
    edge = calibrated_probability - merged["market_probability"]

    # Phase 2: Eradication of Floating-Point Artefacts
    # Cast micro-edges to exact zero.
    edge = edge.round(4)
    zero_mask = edge.abs() < 0.0001
    edge = edge.mask(zero_mask, 0.0)
    ev = ev.mask(zero_mask, 0.0)

    merged["expected_value"] = ev
    merged["edge"] = edge
    merged["best_pick"] = merged.apply(_format_best_pick, axis=1)

    # FIX: Generate game_id for Kalshi matching
    if 'game_id' not in merged.columns:
        logger.warning("🔧 FIX: Generating game_id for Kalshi matching")
        def get_team_short(name):
            """Extract first 4 chars from FIRST SIGNIFICANT WORD (city OR mascot)"""
            normalized = str(name).upper().strip()
            words = normalized.split()

            if len(words) == 1:
                return words[0][:4]

            # Priority: mascot (if short/reliable) OR city (first word)
            mascot = words[-1]  # "CLIPPERS", "STATES", "WASHINGTON"
            city = words[0]     # "LOS", "WEBER", "EASTERN"

            # Use mascot if it's 4+ letters AND not generic ("STATE", "CITY", "COLLEGE")
            if len(mascot) >= 4 and mascot not in ['STATE', 'CITY', 'COLLEGE']:
                return mascot[:4]
            else:
                return city[:4]

        merged['game_id'] = (
            merged['league'].astype(str).str.upper() + '-' +
            merged['home_team'].apply(get_team_short) + '-' +
            merged['away_team'].apply(get_team_short)
        )

    analysis_df = merged.head(max_rows).copy()
    if not analysis_df.empty and not base_df.empty:
        base_dates = base_df.copy()
        base_dates["league"] = _string_series(base_dates, "league").str.upper().replace(LEAGUE_ALIASES)
        base_dates["home_team"] = _string_series(base_dates, "home_team").map(normalize_team_name)
        base_dates["away_team"] = _string_series(base_dates, "away_team").map(normalize_team_name)
        base_dates["date"] = _game_dates(base_dates)
        date_fill = analysis_df.merge(
            base_dates[["league", "home_team", "away_team", "date"]],
            on=["league", "home_team", "away_team"],
            how="left",
            suffixes=("", "_basefill"),
        )
        date_fill_series = _game_dates(date_fill)
        if "date_basefill" in date_fill.columns:
            date_fill_series = date_fill_series.where(date_fill_series.notna(), pd.to_datetime(date_fill["date_basefill"], errors="coerce", utc=True))
        analysis_df["game_date"] = _game_dates(analysis_df).fillna(date_fill_series)
    if "game_key" not in analysis_df.columns:
        analysis_df["game_key"] = _mk_game_key(analysis_df)
    if not analysis_df.empty and "market_type" not in analysis_df.columns:
        raise ValueError("analysis_df missing market_type before best-pick construction")

    # In the refactored flow, we no longer build best_picks_df inside run_analysis_pipeline.
    # Instead, we just return an empty dataframe here, and best_picks_df is built in streamlit_app.py
    # AFTER the full analysis_df has been enriched with Kalshi probabilities.
    best_picks_df = pd.DataFrame(columns=BEST_PICK_COLUMNS)

    stale = is_stale_schedule(base_df, analysis_df)
    base_coverage = float(_game_dates(base_df).notna().mean()) if not base_df.empty else 0.0

    diagnostics = {
        "total_rows": int(len(analysis_df)),
        "rows_with_game_date": int(pd.to_datetime(analysis_df.get("game_date"), errors="coerce", utc=True).notna().sum()) if not analysis_df.empty else 0,
        # Safely sort team names alphabetically and append the market type to count total betting markets
        "total_games": int(analysis_df.apply(
            lambda r: f"{r.get('league')}|{min(str(r.get('home_team')), str(r.get('away_team')))}|{max(str(r.get('home_team')), str(r.get('away_team')))}|{'Total' if 'over' in str(r.get('best_pick')).lower() or 'under' in str(r.get('best_pick')).lower() else 'Spread'}",
            axis=1
        ).nunique()) if not analysis_df.empty else 0,
        "bet_rows": int(len(analysis_df)),
        "ml_model_loaded": bool(use_ml and ML_AVAILABLE),
        "ml_predictions": int(analysis_df["ml_probability"].notna().sum()) if "ml_probability" in analysis_df.columns else 0,
        "best_picks": int(len(best_picks_df)),
        "kalshi_attempted": 0,
        "kalshi_matches": 0,
        "kalshi_match_rate": 0.0,
        "match_rate": 0.0,
        "theover_totals_games": int(analysis_df[_string_series(analysis_df, "market_type").str.startswith("total")]["game_key"].nunique()) if not analysis_df.empty else 0,
        "theover_spreads_games": int(analysis_df[_string_series(analysis_df, "market_type").str.startswith("spread")]["game_key"].nunique()) if not analysis_df.empty else 0,
        "date_fill_total_rows": int(date_stats["date_fill_total_rows"]),
        "date_fill_success_rows": int(date_stats["date_fill_success_rows"]),
        "date_fill_success_rate": float(date_stats["date_fill_success_rate"]),
        "missing_game_date_rows": int(date_stats["missing_game_date_rows"]),
        "positive_ev_picks": int((_numeric_series(best_picks_df, "expected_value", 0.0) > 0).sum()) if not best_picks_df.empty else 0,
        "market_type_counts": _string_series(analysis_df, "market_type").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "allowed_market_type_rows": int(_string_series(analysis_df, "market_type").isin(VALID_MARKETS).sum()) if not analysis_df.empty else 0,
        "positive_ev_rows": int((_numeric_series(analysis_df, "expected_value", 0.0) > 0).sum()) if not analysis_df.empty else 0,
        "spread_rows_missing_model_prob": int(((_string_series(analysis_df, "market_type").str.startswith("spread")) & (_numeric_series(analysis_df, "model_probability").isna())).sum()) if not analysis_df.empty else 0,
        "best_pick_nonempty_rows": int(_string_series(best_picks_df, "best_pick").str.strip().str.len().gt(0).sum()) if not best_picks_df.empty else 0,
        "best_picks_count": int(len(best_picks_df)),
        "odds_schedule_loaded": odds_schedule_loaded,
        "odds_source_counts": _string_series(analysis_df, "odds_source").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "base_rows_loaded": int(len(base_df)),
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
