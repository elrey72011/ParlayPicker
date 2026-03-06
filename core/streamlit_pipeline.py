from __future__ import annotations

import logging
from itertools import combinations
import os
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.bankroll_simulator import simulate_bankroll
from core.kelly_optimizer import add_kelly_bet_sizing
from core.probability_calibration import calibrate_probabilities
from core.probability_engine import american_to_prob, normalize_probability_components, remove_vig
from core.schema.base_schema import ensure_base_schema
from core.team_mapper import normalize_team_name
from app_core.kalshi_integrator import match_kalshi_markets

logger = logging.getLogger(__name__)


try:
    from complete_workflow_implementation import run_ml_predictions
except Exception:  # pragma: no cover
    run_ml_predictions = None


MERGE_KEYS = ["league", "home_team", "away_team", "game_date"]
MODEL_PATH = "models/sports_model_latest.joblib"
SPORT_ALIASES = {
    "NBA": "NBA",
    "NHL": "NHL",
    "NCAAM": "NCAAB",
    "NCAA MEN'S BASKETBALL": "NCAAB",
    "NCAA MENS BASKETBALL": "NCAAB",
    "NCAA BASKETBALL": "NCAAB",
    "COLLEGE BASKETBALL": "NCAAB",
}
BEST_PICK_COLUMNS = [
    "league",
    "home_team",
    "away_team",
    "game_date",
    "best_pick",
    "calibrated_probability",
    "expected_value",
    "edge",
    "odds_american",
    "market_probability",
    "ml_probability",
    "kalshi_probability",
    "kalshi_match_status",
    "kalshi_event_ticker",
]

THEOVER_COLUMN_ALIASES = {
    "league": ["league"],
    "home_team": ["home_team", "hometeam", "home", "home team", "team_home"],
    "away_team": ["away_team", "awayteam", "away", "away team", "team_away"],
    "game_date": ["game_date", "date", "commence_time", "time", "start_time"],
    "market": ["market", "bet_type", "market_type", "wager_type", "pick_type"],
    "pick": ["pick", "selection", "side", "o/u", "over_under"],
    "pickteam": ["pickteam", "pick_team", "team", "selection_team"],
    "line": ["line", "spread", "spread_line", "total", "total_line", "points", "number"],
    "winprobability": ["winprobability", "probability", "win_prob", "win_probability"],
}


def normalize_theover_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip().str.lower()

    rename_map: dict[str, str] = {}
    for canonical, aliases in THEOVER_COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.strip().lower()
            if key in normalized.columns and canonical not in normalized.columns:
                rename_map[key] = canonical
                break
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce")
    if "line" in normalized.columns:
        normalized["line"] = pd.to_numeric(normalized["line"], errors="coerce")
    if "winprobability" in normalized.columns:
        normalized["winprobability"] = pd.to_numeric(normalized["winprobability"], errors="coerce")

    return _normalize_key_columns(normalized)


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    base_keys = [k for k in ["league", "home_team", "away_team"] if k in left.columns and k in right.columns]
    if "game_date" not in left.columns or "game_date" not in right.columns:
        return base_keys

    left_dates = pd.to_datetime(left["game_date"], errors="coerce")
    right_dates = pd.to_datetime(right["game_date"], errors="coerce")
    left_coverage = left_dates.notna().mean() if len(left_dates) else 0.0
    right_coverage = right_dates.notna().mean() if len(right_dates) else 0.0

    if left_coverage < 0.5 or right_coverage < 0.5:
        return base_keys

    overlap = set(left_dates.dropna().dt.date.unique()) & set(right_dates.dropna().dt.date.unique())
    if not overlap:
        return base_keys

    return base_keys + ["game_date"]


def infer_market_type_and_lines(row: pd.Series) -> pd.Series:
    pick = str(row.get("pick") or "").lower()
    market = str(row.get("market") or "").lower()
    line = pd.to_numeric(row.get("line"), errors="coerce")
    pickteam = normalize_team_name(row.get("pickteam"))
    home_team = normalize_team_name(row.get("home_team"))

    market_type = ""
    spread_line = pd.NA
    total_line = pd.NA

    if "over" in pick or "over" in market:
        market_type = "total_over"
        total_line = line
    elif "under" in pick or "under" in market:
        market_type = "total_under"
        total_line = line
    else:
        market_type = "spread_home" if pickteam and pickteam == home_team else "spread_away"
        spread_line = line

    return pd.Series({"market_type": market_type, "spread_line": spread_line, "total_line": total_line})


def _infer_market_type(row: pd.Series) -> str:
    allowed_market_types = {
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    }

    existing_market_type = str(row.get("market_type") or "").strip().lower()
    if existing_market_type in allowed_market_types:
        return existing_market_type

    market_hint = " ".join(
        [
            str(row.get("market") or ""),
            str(row.get("bet_type") or ""),
            str(row.get("wager_type") or ""),
            str(row.get("pick_type") or ""),
            str(row.get("pick") or ""),
            str(row.get("side") or ""),
            str(row.get("over_under") or ""),
            str(row.get("selection") or ""),
        ]
    ).lower()

    spread_candidates = [row.get("spread"), row.get("spread_line"), row.get("line")]
    total_candidates = [row.get("total"), row.get("total_line"), row.get("total_points"), row.get("points")]
    spread_val = pd.Series(spread_candidates).apply(pd.to_numeric, errors="coerce").dropna()
    total_val = pd.Series(total_candidates).apply(pd.to_numeric, errors="coerce").dropna()
    spread_num = spread_val.iloc[0] if not spread_val.empty else np.nan
    total_num = total_val.iloc[0] if not total_val.empty else np.nan

    pick_team = str(row.get("team") or row.get("selection") or row.get("pick") or "").strip().lower()
    home_team = str(row.get("home_team") or "").strip().lower()
    away_team = str(row.get("away_team") or "").strip().lower()
    is_home_pick = bool(row.get("is_home_pick", False))
    if pick_team and home_team:
        is_home_pick = pick_team == home_team
    elif pick_team and away_team:
        is_home_pick = pick_team != away_team

    has_over_under_text = any(token in market_hint for token in ["over", "under", "o/u", "ou"])
    has_total_text = any(token in market_hint for token in ["total", "over", "under", "o/u", "points"])
    has_spread_text = any(token in market_hint for token in ["spread", "ats", "handicap"])

    if "under" in market_hint and (has_total_text or pd.notna(total_num)):
        return "total_under"
    if "over" in market_hint and (has_total_text or pd.notna(total_num)):
        return "total_over"

    if has_spread_text or (pd.notna(spread_num) and not has_over_under_text):
        return "spread_home" if is_home_pick else "spread_away"

    if pd.notna(total_num) and has_over_under_text:
        return "total_under" if "under" in market_hint else "total_over"

    if has_total_text or pd.notna(total_num):
        return "total_over"

    if pd.notna(spread_num):
        return "spread_home" if is_home_pick else "spread_away"

    return "unknown"


def format_pick(row: pd.Series) -> str:
    def _format_signed_spread(value: float, invert_sign: bool = False) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return ""
        if invert_sign:
            numeric = -numeric
        return f"{numeric:+.1f}"

    def _format_total(value: float) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return ""
        return f"{numeric:.1f}"

    if row["market_type"] == "spread_home":
        spread_display = _format_signed_spread(row.get("spread_line", row.get("spread")))
        return f"{row['home_team']} {spread_display}".strip()

    if row["market_type"] == "spread_away":
        spread_display = _format_signed_spread(row.get("spread_line", row.get("spread")), invert_sign=True)
        return f"{row['away_team']} {spread_display}".strip()

    if row["market_type"] == "total_over":
        return f"Over {_format_total(row.get('total_line', row.get('total')))}".strip()

    if row["market_type"] == "total_under":
        return f"Under {_format_total(row.get('total_line', row.get('total')))}".strip()

    return ""


def _build_best_picks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "market_type" not in df.columns:
        df["market_type"] = ""

    if "spread" not in df.columns:
        df["spread"] = pd.to_numeric(df.get("spread_line"), errors="coerce")
    else:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")

    if "total" not in df.columns:
        df["total"] = pd.to_numeric(df.get("total_line"), errors="coerce")
    else:
        df["total"] = pd.to_numeric(df["total"], errors="coerce")

    df["inferred_market_type"] = df.apply(_infer_market_type, axis=1)
    df["market_type"] = df["inferred_market_type"]
    df["source_has_spread"] = df[[c for c in ["spread", "spread_line", "line"] if c in df.columns]].apply(
        lambda row: row.apply(pd.to_numeric, errors="coerce").notna().any(), axis=1
    ) if any(c in df.columns for c in ["spread", "spread_line", "line"]) else False
    df["source_has_total"] = df[[c for c in ["total", "total_line", "total_points", "points"] if c in df.columns]].apply(
        lambda row: row.apply(pd.to_numeric, errors="coerce").notna().any(), axis=1
    ) if any(c in df.columns for c in ["total", "total_line", "total_points", "points"]) else False

    logger.info(
        "Best-pick inference debug: inferred_market_type_counts=%s source_has_spread=%s source_has_total=%s",
        df["inferred_market_type"].value_counts(dropna=False).to_dict(),
        int(pd.to_numeric(df["source_has_spread"], errors="coerce").fillna(False).astype(bool).sum()),
        int(pd.to_numeric(df["source_has_total"], errors="coerce").fillna(False).astype(bool).sum()),
    )

    allowed_market_types = {"spread_home", "spread_away", "total_over", "total_under"}
    df = df[df["market_type"].isin(allowed_market_types)].copy()
    if df.empty:
        best_picks = pd.DataFrame(columns=BEST_PICK_COLUMNS)
        return best_picks

    group_keys = ["league", "home_team", "away_team", "game_date"]
    available_group_keys = [k for k in group_keys if k in df.columns]
    if not available_group_keys:
        available_group_keys = ["home_team", "away_team"]

    if "expected_value" not in df.columns:
        df["expected_value"] = pd.NA
    if "edge" not in df.columns:
        df["edge"] = pd.NA

    df["expected_value"] = pd.to_numeric(df["expected_value"], errors="coerce")
    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")

    best_picks = (
        df.sort_values(["expected_value", "edge"], ascending=[False, False])
        .groupby(available_group_keys)
        .first()
        .reset_index()
    )
    best_picks["best_pick"] = best_picks.apply(format_pick, axis=1)
    best_picks = best_picks[best_picks["best_pick"].astype(str).str.len() > 0].copy()

    for col in BEST_PICK_COLUMNS:
        if col not in best_picks.columns:
            best_picks[col] = pd.NA

    return best_picks[BEST_PICK_COLUMNS]


def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Build one spread/total best-pick row per game from a raw analysis dataframe."""
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)
    return _build_best_picks(analysis_df)


def _build_kalshi_df(base_df: pd.DataFrame) -> pd.DataFrame | None:
    if base_df is None or base_df.empty:
        return None

    keys = [k for k in ["league", "home_team", "away_team"] if k in base_df.columns]
    if len(keys) < 3:
        logger.info("Kalshi skipped: missing merge keys. available=%s", keys)
        return None
    if "game_date" in base_df.columns:
        keys = keys + ["game_date"]

    games = normalize_merge_keys(_normalize_key_columns(base_df[keys + ["market_type", "spread_line", "total_line"] if "market_type" in base_df.columns else keys].drop_duplicates().copy()))
    if games is None or games.empty:
        return None

    kalshi_df = match_kalshi_markets(games)
    if kalshi_df is None or kalshi_df.empty:
        return None
    return normalize_merge_keys(_normalize_key_columns(kalshi_df))


def normalize_merge_keys(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df

    df = df.copy()

    if "league" in df.columns:
        df["league"] = df["league"].astype(str)

    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].astype(str).str.strip().str.lower()

    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].astype(str).str.strip().str.lower()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    return df


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("ML model not found, using market probabilities.")
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print("Model load failed:", e)
        return None


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["home_team", "away_team", "team"]:
        if col in df.columns:
            df.loc[:, col] = df[col].apply(normalize_team_name)
    return df


def _normalize_league_value(value: str | object) -> str:
    if pd.isna(value):
        return ""
    normalized = str(value).strip().upper()
    return SPORT_ALIASES.get(normalized, normalized)


def _normalize_sports_filter(sports: Iterable[str] | None) -> list[str]:
    if not sports:
        return []
    return [_normalize_league_value(sport) for sport in sports]


def _normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {"sport": "league", "date": "game_date", "commence_time": "game_date"}
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "game_date" not in df.columns:
        df["game_date"] = pd.NaT

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    if "league" not in df.columns:
        df["league"] = ""
    df["league"] = df["league"].apply(_normalize_league_value)
    return _normalize_teams(df)


def _infer_uploaded_league_row(row: pd.Series, selected_sports: list[str] | None = None) -> str:
    current = _normalize_league_value(row.get("league"))
    if current:
        return current

    selected_set = set(selected_sports or [])
    context_text = " ".join(
        [
            str(row.get("market") or ""),
            str(row.get("bet_type") or ""),
            str(row.get("wager_type") or ""),
            str(row.get("pick_type") or ""),
            str(row.get("source") or ""),
            str(row.get("source_file") or ""),
            str(row.get("filename") or ""),
        ]
    ).upper()

    if "NHL" in context_text and (not selected_set or "NHL" in selected_set):
        return "NHL"
    if any(token in context_text for token in ["NCAAB", "NCAAM", "COLLEGE BASKETBALL", "NCAA"]) and (
        not selected_set or "NCAAB" in selected_set
    ):
        return "NCAAB"
    if "NBA" in context_text and (not selected_set or "NBA" in selected_set):
        return "NBA"

    if len(selected_set) == 1:
        return next(iter(selected_set))

    return ""


def _enrich_uploaded_league(df: pd.DataFrame, selected_sports: list[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    enriched = df.copy()
    if "league" not in enriched.columns:
        enriched["league"] = ""
    enriched["league"] = enriched.apply(lambda row: _infer_uploaded_league_row(row, selected_sports), axis=1)
    enriched["league"] = enriched["league"].apply(_normalize_league_value)
    return enriched




def _is_stale_base_data(base_df: pd.DataFrame, theover_df: pd.DataFrame) -> bool:
    if "game_date" not in base_df.columns or "game_date" not in theover_df.columns:
        return False
    base_dates = pd.to_datetime(base_df["game_date"], errors="coerce").dropna()
    over_dates = pd.to_datetime(theover_df["game_date"], errors="coerce").dropna()
    if base_dates.empty or over_dates.empty:
        return False
    delta_days = abs((base_dates.median() - over_dates.median()).days)
    return delta_days > 14


def build_theover_bet_rows(spreads_df: pd.DataFrame | None, totals_df: pd.DataFrame | None, selected_sports: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in (spreads_df, totals_df):
        if src is not None and not src.empty:
            norm = normalize_theover_df(src)
            norm = _enrich_uploaded_league(norm, selected_sports)
            frames.append(norm)

    if not frames:
        return pd.DataFrame()

    bets = pd.concat(frames, ignore_index=True)
    inferred = bets.apply(infer_market_type_and_lines, axis=1)
    bets = pd.concat([bets, inferred], axis=1)
    bets["theover_probability"] = pd.to_numeric(bets.get("winprobability"), errors="coerce").clip(lower=0.0, upper=1.0)
    bets["matchup"] = bets.get("away_team", "").astype(str) + " @ " + bets.get("home_team", "").astype(str)
    return bets

def _resolve_american_odds(row: pd.Series) -> float:
    for col in ["odds_american", "home_odds", "odds"]:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                continue
    return -110.0


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame | None, suffix: str) -> pd.DataFrame:
    if right is None or right.empty:
        return left

    left = normalize_merge_keys(_normalize_key_columns(left))
    right = normalize_merge_keys(_normalize_key_columns(right))

    keys = choose_merge_keys(left, right)
    if not keys:
        return left

    right = right.drop_duplicates(subset=keys)

    merged = left.merge(
        right,
        on=keys,
        how="left",
        suffixes=("", suffix),
    )

    merged["debug_merge_keys"] = ", ".join(keys)
    return merged


def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = ensure_base_schema(df)

    df["odds_american"] = df.apply(_resolve_american_odds, axis=1)

    # 1) Market probability from each row's odds
    df["market_prob"] = pd.to_numeric(df["odds_american"], errors="coerce").apply(american_to_prob)

    # 2) Remove vig when both sides are available
    if {"home_odds", "away_odds"}.issubset(df.columns):
        home_prob = pd.to_numeric(df["home_odds"], errors="coerce").apply(lambda x: american_to_prob(x) if pd.notna(x) else pd.NA)
        away_prob = pd.to_numeric(df["away_odds"], errors="coerce").apply(lambda x: american_to_prob(x) if pd.notna(x) else pd.NA)
        no_vig = pd.DataFrame(
            [remove_vig(h, a) if pd.notna(h) and pd.notna(a) else (pd.NA, pd.NA) for h, a in zip(home_prob, away_prob)]
        )

        if "is_home_pick" in df.columns:
            is_home_pick = df["is_home_pick"].fillna(False).astype(bool)
            df["market_prob"] = no_vig[1].where(is_home_pick, no_vig[0]).fillna(df["market_prob"])
        elif "team" in df.columns and {"home_team", "away_team"}.issubset(df.columns):
            is_home_pick = df["team"].astype(str).str.lower() == df["home_team"].astype(str).str.lower()
            df["market_prob"] = no_vig[1].where(is_home_pick, no_vig[0]).fillna(df["market_prob"])
        else:
            df["market_prob"] = no_vig[0].fillna(df["market_prob"])

    # 3) ML probability: use model when available, fallback to market probability + noise
    model = load_model()
    model_loaded = model is not None
    df["model_probability"] = pd.to_numeric(df.get("theover_probability", df["market_prob"]), errors="coerce").fillna(pd.to_numeric(df["market_prob"], errors="coerce"))
    if model is not None:
        try:
            if isinstance(model, dict) and {"model", "feature_names"}.issubset(model):
                estimator = model["model"]
                feature_names = model["feature_names"]
            else:
                estimator = model
                feature_names = []

            if feature_names and all(f in df.columns for f in feature_names):
                X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                df["model_probability"] = estimator.predict_proba(X)[:, 1]
            elif hasattr(estimator, "predict_proba"):
                numeric_df = df.select_dtypes(include=["number"]).fillna(0.0)
                if not numeric_df.empty:
                    df["model_probability"] = estimator.predict_proba(numeric_df)[:, 1]
                else:
                    model_loaded = False
        except Exception as exc:
            model_loaded = False
            logger.warning("ML model unavailable for predict_proba; falling back to market_probability: %s", exc)

    if not model_loaded:
        market_prob = pd.to_numeric(df["market_prob"], errors="coerce").fillna(0.5238)
        df["model_probability"] = pd.to_numeric(df.get("theover_probability"), errors="coerce").fillna(market_prob).clip(0.01, 0.99)

    df["ml_prob"] = pd.to_numeric(df["model_probability"], errors="coerce").fillna(df["market_prob"])
    df["ai_prob"] = pd.to_numeric(df.get("ai_probability", pd.NA), errors="coerce")

    # 4) Weighted consensus probability
    df = normalize_probability_components(df)
    df["market_probability"] = df["market_prob"].clip(lower=0.0, upper=1.0)
    df["ml_probability"] = df["ml_prob"].clip(lower=0.0, upper=1.0)
    df["ai_probability"] = df["ai_prob"].clip(lower=0.0, upper=1.0)

    # 5) Probability calibration, EV calculation, and edge
    df["decimal_odds"] = pd.to_numeric(df["odds_american"], errors="coerce").apply(american_to_decimal)
    df = calibrate_probabilities(df)
    prob_for_ev = pd.to_numeric(df.get("calibrated_probability"), errors="coerce").fillna(df["model_probability"])
    df["expected_value"] = (
        prob_for_ev * (df["decimal_odds"] - 1)
        - (1 - prob_for_ev)
    )
    df["edge"] = prob_for_ev - df["market_probability"]
    if "debug_merge_keys" not in df.columns:
        df["debug_merge_keys"] = ", ".join([k for k in MERGE_KEYS if k in df.columns])
    df["debug_model_loaded"] = bool(model_loaded)

    if "team" not in df.columns:
        df["team"] = df.get("away_team", "")

    df = df.sort_values("edge", ascending=False).reset_index(drop=True)

    # 9) Debug output for verification in logs
    debug_cols = [
        "home_team",
        "away_team",
        "odds_american",
        "market_probability",
        "ml_probability",
        "calibrated_probability",
        "consensus_prob",
        "expected_value",
        "edge",
    ]
    available_debug_cols = [c for c in debug_cols if c in df.columns]
    if available_debug_cols:
        logger.info("Analysis probability debug sample:\n%s", df[available_debug_cols].head(25).to_string(index=False))

    return df


@st.cache_data(ttl=300)
def load_base_data() -> pd.DataFrame:
    df = pd.read_csv("data/master_all_sports.csv")
    return _normalize_key_columns(df)


@st.cache_data(ttl=180)
def run_analysis_pipeline(
    sports: Iterable[str],
    max_rows: int,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base_df = load_base_data().copy()
    if "league" in base_df.columns:
        base_df["league"] = base_df["league"].apply(_normalize_league_value)

    selected_sports = _normalize_sports_filter(sports)
    if selected_sports and "league" in base_df.columns:
        base_df = base_df[base_df["league"].isin(selected_sports)].copy()

    theover_bets_df = build_theover_bet_rows(spreads_df, totals_df, selected_sports)
    if selected_sports and "league" in theover_bets_df.columns:
        theover_bets_df = theover_bets_df[theover_bets_df["league"].isin(selected_sports)].copy()

    use_theover_as_base = theover_bets_df is not None and not theover_bets_df.empty
    if use_theover_as_base:
        filtered = theover_bets_df.head(max_rows).copy()
    else:
        filtered = base_df.head(max_rows).copy()

    if use_theover_as_base and not base_df.empty:
        if _is_stale_base_data(base_df, theover_bets_df) and "game_date" in base_df.columns:
            base_df = base_df.copy()
            base_df["game_date"] = pd.NaT
        filtered = _safe_merge(filtered, base_df, "_base")

    if use_ml and run_ml_predictions and not filtered.empty and not use_theover_as_base:
        ml_df = _normalize_key_columns(run_ml_predictions(filtered))
        filtered = _safe_merge(filtered, ml_df, "_ml")

    kalshi_df = _build_kalshi_df(filtered)
    filtered = _safe_merge(filtered, kalshi_df, "_kalshi")

    analyzed = _apply_analysis_calculations(filtered)
    if analyzed.empty:
        return analyzed
    return analyzed


def generate_parlays(analysis_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"]
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=columns)

    required = {"best_pick", "calibrated_probability", "expected_value"}
    if not required.issubset(analysis_df.columns):
        return pd.DataFrame(columns=columns)

    candidate_bets = analysis_df.copy()
    if "market_type" in candidate_bets.columns:
        candidate_bets = candidate_bets[
            candidate_bets["market_type"].isin({"spread_home", "spread_away", "total_over", "total_under"})
        ]

    candidate_bets = candidate_bets[candidate_bets["best_pick"].astype(str).str.strip().str.len() > 0].copy()
    candidate_bets["calibrated_probability"] = pd.to_numeric(candidate_bets["calibrated_probability"], errors="coerce")
    candidate_bets["expected_value"] = pd.to_numeric(candidate_bets["expected_value"], errors="coerce")
    if "decimal_odds" in candidate_bets.columns:
        candidate_bets["decimal_odds"] = pd.to_numeric(candidate_bets["decimal_odds"], errors="coerce")
    else:
        candidate_bets["decimal_odds"] = pd.to_numeric(candidate_bets.get("odds_american"), errors="coerce").apply(
            lambda x: american_to_decimal(x) if pd.notna(x) else pd.NA
        )

    rank_column = "edge" if "edge" in candidate_bets.columns else "expected_value"
    candidate_bets[rank_column] = pd.to_numeric(candidate_bets[rank_column], errors="coerce")
    candidate_bets = candidate_bets.dropna(subset=["calibrated_probability", "decimal_odds", "expected_value", rank_column])
    candidate_bets = candidate_bets[candidate_bets["expected_value"] > 0]
    candidate_bets = candidate_bets.nlargest(20, rank_column)

    records: list[dict[str, float | int | str]] = []
    for leg_count in (2, 3, 4, 5):
        if len(candidate_bets) < leg_count:
            continue

        for combo in combinations(candidate_bets.index, leg_count):
            legs = candidate_bets.loc[list(combo)]
            combined_probability = float(legs["calibrated_probability"].prod())
            combined_decimal_odds = float(legs["decimal_odds"].prod())
            parlay_ev = (combined_probability * (combined_decimal_odds - 1)) - (1 - combined_probability)
            if parlay_ev <= 0:
                continue

            leg_labels = [str(row["best_pick"]).strip() for _, row in legs.iterrows()]
            records.append(
                {
                    "parlay_legs": " | ".join(leg_labels),
                    "combined_probability": combined_probability,
                    "combined_decimal_odds": combined_decimal_odds,
                    "parlay_ev": parlay_ev,
                    "legs": leg_count,
                }
            )

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records)[columns].sort_values("parlay_ev", ascending=False).reset_index(drop=True)


def build_realtime_edges(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    edge_cols = [
        c
        for c in [
            "league",
            "home_team",
            "away_team",
            "calibrated_probability",
            "market_probability",
            "decimal_odds",
            "expected_value",
            "edge",
        ]
        if c in analysis_df.columns
    ]
    if edge_cols:
        edges_df = analysis_df[edge_cols]
        if "edge" in edges_df.columns:
            return edges_df.sort_values("edge", ascending=False).head(25)
        if "expected_value" in edges_df.columns:
            return edges_df.sort_values("expected_value", ascending=False).head(25)
        return edges_df.head(25)

    return analysis_df.head(25)


def optimize_portfolio_allocation(analysis_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    edges = build_realtime_edges(analysis_df)
    if edges.empty:
        return edges

    portfolio = add_kelly_bet_sizing(edges, bankroll=bankroll, fraction=0.25)
    recommended_total = portfolio["recommended_bet"].sum() if "recommended_bet" in portfolio.columns else 0.0
    if recommended_total > 0:
        portfolio["allocation_pct"] = ((portfolio["recommended_bet"] / recommended_total) * 100).round(2)
    else:
        portfolio["allocation_pct"] = 0.0
    return portfolio.sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=1000, simulations=1000)
