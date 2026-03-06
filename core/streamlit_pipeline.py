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
from app_core.kalshi_integrator import enrich_with_kalshi_markets

logger = logging.getLogger(__name__)


try:
    from complete_workflow_implementation import run_ml_predictions
except Exception:  # pragma: no cover
    run_ml_predictions = None


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
        is_home_pick = bool(pickteam and home_team and pickteam == home_team)
        market_type = "spread_home" if is_home_pick else "spread_away"
        spread_line = line if is_home_pick else (-line if pd.notna(line) else pd.NA)

    return pd.Series({"market_type": market_type, "spread_line": spread_line, "total_line": total_line})


def infer_market_type_from_row(row: pd.Series) -> str:
    """Infer canonical market type for an analysis row."""
    allowed_market_types = {
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
        "moneyline_home",
        "moneyline_away",
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
            str(row.get("best_pick") or ""),
            str(row.get("team") or ""),
        ]
    ).lower()

    spread_candidates = [row.get("spread"), row.get("spread_line"), row.get("line")]
    total_candidates = [row.get("total"), row.get("total_line"), row.get("total_points"), row.get("points")]
    spread_val = pd.Series(spread_candidates).apply(pd.to_numeric, errors="coerce").dropna()
    total_val = pd.Series(total_candidates).apply(pd.to_numeric, errors="coerce").dropna()
    spread_num = spread_val.iloc[0] if not spread_val.empty else np.nan
    total_num = total_val.iloc[0] if not total_val.empty else np.nan

    pick_team = normalize_team_name(row.get("team") or row.get("selection") or row.get("pick") or row.get("pickteam"))
    home_team = normalize_team_name(row.get("home_team"))
    away_team = normalize_team_name(row.get("away_team"))

    is_home_pick = bool(row.get("is_home_pick", False))
    if pick_team and home_team and pick_team == home_team:
        is_home_pick = True
    elif pick_team and away_team and pick_team == away_team:
        is_home_pick = False

    has_moneyline_text = any(token in market_hint for token in ["moneyline", "ml", "to win", "winner"])
    has_total_text = any(token in market_hint for token in ["total", "over", "under", "o/u", "points"])
    has_spread_text = any(token in market_hint for token in ["spread", "ats", "handicap"])

    if "under" in market_hint and (has_total_text or pd.notna(total_num)):
        return "total_under"
    if "over" in market_hint and (has_total_text or pd.notna(total_num)):
        return "total_over"

    if has_spread_text or pd.notna(spread_num):
        return "spread_home" if is_home_pick else "spread_away"

    if has_moneyline_text:
        return "moneyline_home" if is_home_pick else "moneyline_away"

    # Fallback when team is explicit but market family is not.
    if pick_team and (pick_team == home_team or pick_team == away_team):
        return "moneyline_home" if pick_team == home_team else "moneyline_away"

    return "unknown"


def _infer_market_type(row: pd.Series) -> str:
    return infer_market_type_from_row(row)


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
        spread_display = _format_signed_spread(row.get("spread_line", row.get("spread")))
        return f"{row['away_team']} {spread_display}".strip()

    if row["market_type"] == "total_over":
        return f"Over {_format_total(row.get('total_line', row.get('total')))}".strip()

    if row["market_type"] == "total_under":
        return f"Under {_format_total(row.get('total_line', row.get('total')))}".strip()

    return ""


def _ensure_best_pick_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    resolved = df.copy()
    if "market_type" not in resolved.columns:
        resolved["market_type"] = resolved.apply(_infer_market_type, axis=1)
    else:
        market_type_series = resolved["market_type"].astype(str).str.strip().str.lower()
        missing_market_type = market_type_series.eq("") | market_type_series.eq("nan")
        if missing_market_type.any():
            resolved.loc[missing_market_type, "market_type"] = resolved.loc[missing_market_type].apply(_infer_market_type, axis=1)

    if "spread_line" not in resolved.columns:
        resolved["spread_line"] = pd.to_numeric(resolved.get("spread"), errors="coerce")
    else:
        resolved["spread_line"] = pd.to_numeric(resolved["spread_line"], errors="coerce")
        fallback_spread = pd.to_numeric(resolved.get("spread"), errors="coerce")
        resolved["spread_line"] = resolved["spread_line"].where(resolved["spread_line"].notna(), fallback_spread)

    if "total_line" not in resolved.columns:
        resolved["total_line"] = pd.to_numeric(resolved.get("total"), errors="coerce")
    else:
        resolved["total_line"] = pd.to_numeric(resolved["total_line"], errors="coerce")
        fallback_total = pd.to_numeric(resolved.get("total"), errors="coerce")
        resolved["total_line"] = resolved["total_line"].where(resolved["total_line"].notna(), fallback_total)

    generated_best_pick = resolved.apply(format_pick, axis=1)
    if "best_pick" in resolved.columns:
        best_pick_series = resolved["best_pick"].fillna("").astype(str).str.strip()
        resolved["best_pick"] = best_pick_series.where(best_pick_series.str.len() > 0, generated_best_pick)
    else:
        resolved["best_pick"] = generated_best_pick

    resolved["best_pick"] = resolved["best_pick"].fillna("").astype(str).str.strip()
    return resolved


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




def is_stale_schedule(df: pd.DataFrame, now_utc: pd.Timestamp, max_age_days: int = 14) -> bool:
    if df is None or df.empty or "game_date" not in df.columns:
        return False
    max_game_date = pd.to_datetime(df["game_date"], errors="coerce", utc=True).dropna()
    if max_game_date.empty:
        return False
    return max_game_date.max() < (now_utc - pd.Timedelta(days=max_age_days))


def build_theover_bet_rows(spreads_df: pd.DataFrame | None, totals_df: pd.DataFrame | None, selected_sports: list[str]) -> pd.DataFrame:
    required_cols = [
        "league", "game_date", "home_team", "away_team", "market_type", "spread_line", "total_line",
        "theover_probability", "odds_american", "market", "pick", "pickteam", "line",
    ]

    def _prep(df: pd.DataFrame | None, default_market: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=required_cols)
        out = normalize_theover_df(df)
        out = _enrich_uploaded_league(out, selected_sports)
        if "market" in out.columns:
            out["market"] = out["market"].fillna(default_market)
        else:
            out["market"] = default_market
        out["pick"] = out["pick"] if "pick" in out.columns else ""
        out["pickteam"] = out["pickteam"] if "pickteam" in out.columns else ""
        out["line"] = pd.to_numeric(out["line"] if "line" in out.columns else pd.Series([pd.NA] * len(out), index=out.index), errors="coerce")
        out["winprobability"] = pd.to_numeric(out.get("winprobability"), errors="coerce")
        if out["winprobability"].dropna().gt(1.0).mean() > 0.8:
            out["winprobability"] = out["winprobability"] / 100.0
        out["theover_probability"] = out["winprobability"].clip(0.0, 1.0)
        odds_series = out["odds_american"] if "odds_american" in out.columns else pd.Series([-110.0] * len(out), index=out.index)
        out["odds_american"] = pd.to_numeric(odds_series, errors="coerce").fillna(-110.0)
        out["spread_line"] = pd.NA
        out["total_line"] = pd.NA
        out["market_type"] = ""

        if default_market == "spread":
            pickteam_norm = out["pickteam"].apply(normalize_team_name)
            home_norm = out["home_team"].apply(normalize_team_name)
            away_norm = out["away_team"].apply(normalize_team_name)
            home_pick = pickteam_norm.eq(home_norm)
            away_pick = pickteam_norm.eq(away_norm)
            out.loc[home_pick, "spread_line"] = out.loc[home_pick, "line"]
            out.loc[away_pick, "spread_line"] = -out.loc[away_pick, "line"]
            spread_num = pd.to_numeric(out["spread_line"], errors="coerce")
            out.loc[spread_num < 0, "market_type"] = "spread_home"
            out.loc[spread_num > 0, "market_type"] = "spread_away"
        else:
            pick_s = out["pick"].astype(str).str.lower()
            out["total_line"] = out["line"]
            out.loc[pick_s.str.contains("over"), "market_type"] = "total_over"
            out.loc[pick_s.str.contains("under"), "market_type"] = "total_under"

        for col in required_cols:
            if col not in out.columns:
                out[col] = pd.NA
        return out[required_cols]

    spread_rows = _prep(spreads_df, "spread")
    total_rows = _prep(totals_df, "total")
    bets = pd.concat([spread_rows, total_rows], ignore_index=True)
    if bets.empty:
        return bets
    bets = _normalize_key_columns(bets)
    if selected_sports and "league" in bets.columns:
        bets = bets[bets["league"].isin(selected_sports)].copy()
    bets = bets[bets["market_type"].isin({"spread_home", "spread_away", "total_over", "total_under"})].copy()
    bets["matchup"] = bets["away_team"].astype(str) + " @ " + bets["home_team"].astype(str)
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
        df["debug_merge_keys"] = ", ".join([k for k in ["league", "home_team", "away_team", "game_date"] if k in df.columns])
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str | None]]:
    now_utc = pd.Timestamp.utcnow()
    base_df = load_base_data().copy()
    if "league" in base_df.columns:
        base_df["league"] = base_df["league"].apply(_normalize_league_value)

    selected_sports = _normalize_sports_filter(sports)
    if selected_sports and "league" in base_df.columns:
        base_df = base_df[base_df["league"].isin(selected_sports)].copy()

    theover_bets_df = build_theover_bet_rows(spreads_df, totals_df, selected_sports)
    base_stale = is_stale_schedule(base_df, now_utc)

    if not theover_bets_df.empty:
        filtered = theover_bets_df.head(max_rows).copy()
        if not base_df.empty and not base_stale:
            enrich_cols = [c for c in base_df.columns if c not in filtered.columns]
            if enrich_cols:
                filtered = _safe_merge(filtered, base_df[[*choose_merge_keys(filtered, base_df), *enrich_cols]], "_base")
    else:
        filtered = base_df.head(max_rows).copy()

    if use_ml and run_ml_predictions and not filtered.empty and theover_bets_df.empty:
        ml_df = _normalize_key_columns(run_ml_predictions(filtered))
        filtered = _safe_merge(filtered, ml_df, "_ml")

    analyzed = _apply_analysis_calculations(filtered)
    if analyzed.empty:
        diagnostics = {
            "total_rows": 0,
            "bet_rows": 0,
            "best_picks_rows": 0,
            "base_max_date": None,
            "theover_max_date": None,
            "kalshi_attempted": 0,
            "kalshi_matched": 0,
            "bet_row_coverage": 0.0,
            "base_schedule_stale": base_stale,
        }
        return analyzed, pd.DataFrame(columns=BEST_PICK_COLUMNS), diagnostics

    market_types = analyzed["market_type"].astype(str) if "market_type" in analyzed.columns else pd.Series(["" for _ in range(len(analyzed))], index=analyzed.index)
    analyzed = analyzed[market_types.isin({"spread_home", "spread_away", "total_over", "total_under"})].copy()
    prob_priority = pd.to_numeric(analyzed.get("theover_probability"), errors="coerce")
    prob_priority = prob_priority.fillna(pd.to_numeric(analyzed.get("model_probability"), errors="coerce"))
    prob_priority = prob_priority.fillna(pd.to_numeric(analyzed.get("market_probability"), errors="coerce"))
    analyzed["calibrated_probability"] = prob_priority.clip(0.0, 1.0)
    analyzed["odds_american"] = pd.to_numeric(analyzed.get("odds_american"), errors="coerce").fillna(-110.0)
    analyzed["decimal_odds"] = analyzed["odds_american"].apply(american_to_decimal)
    analyzed["expected_value"] = analyzed["calibrated_probability"] * (analyzed["decimal_odds"] - 1) - (1 - analyzed["calibrated_probability"])
    analyzed["edge"] = analyzed["calibrated_probability"] - pd.to_numeric(analyzed.get("market_probability"), errors="coerce").fillna(0.5)

    best_picks_df = _build_best_picks(analyzed)
    diagnostics = {
        "total_rows": int(len(filtered)),
        "bet_rows": int(len(analyzed)),
        "best_picks_rows": int(len(best_picks_df)),
        "base_max_date": pd.to_datetime(base_df.get("game_date"), errors="coerce", utc=True).max() if not base_df.empty else None,
        "theover_max_date": pd.to_datetime(theover_bets_df.get("game_date"), errors="coerce", utc=True).max() if not theover_bets_df.empty else None,
        "kalshi_attempted": 0,
        "kalshi_matched": 0,
        "bet_row_coverage": float(len(analyzed) / max(len(filtered), 1)),
        "base_schedule_stale": base_stale,
    }
    return analyzed, best_picks_df, diagnostics


def generate_parlays(analysis_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"]
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=columns)

    required = {"best_pick", "calibrated_probability", "expected_value"}
    if not required.issubset(analysis_df.columns):
        return pd.DataFrame(columns=columns)

    candidate_bets = analysis_df.copy()
    if "best_pick" in candidate_bets.columns:
        candidate_bets["team"] = candidate_bets["best_pick"]
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

    analysis_df = _ensure_best_pick_column(analysis_df)

    edge_cols = [
        c
        for c in [
            "league",
            "home_team",
            "away_team",
            "best_pick",
            "market_type",
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
    analysis_df = _ensure_best_pick_column(analysis_df)
    edges = build_realtime_edges(analysis_df)
    if edges.empty:
        return edges

    if "decimal_odds" not in edges.columns:
        if "odds_american" in edges.columns:
            edges["decimal_odds"] = pd.to_numeric(edges["odds_american"], errors="coerce").apply(american_to_decimal)
        else:
            edges["decimal_odds"] = 1.9091
    else:
        decimal_odds = pd.to_numeric(edges["decimal_odds"], errors="coerce")
        if "odds_american" in edges.columns:
            fallback_odds = pd.to_numeric(edges["odds_american"], errors="coerce").apply(american_to_decimal)
            edges["decimal_odds"] = decimal_odds.fillna(fallback_odds).fillna(1.9091)
        else:
            edges["decimal_odds"] = decimal_odds.fillna(1.9091)

    portfolio = add_kelly_bet_sizing(edges, bankroll=bankroll, fraction=0.25)
    portfolio = _ensure_best_pick_column(portfolio)

    if "decimal_odds" not in portfolio.columns:
        portfolio["decimal_odds"] = pd.to_numeric(portfolio.get("odds_american"), errors="coerce").apply(american_to_decimal) if "odds_american" in portfolio.columns else 1.9091
    else:
        decimal_odds = pd.to_numeric(portfolio["decimal_odds"], errors="coerce")
        if "odds_american" in portfolio.columns:
            fallback_odds = pd.to_numeric(portfolio["odds_american"], errors="coerce").apply(american_to_decimal)
            portfolio["decimal_odds"] = decimal_odds.fillna(fallback_odds).fillna(1.9091)
        else:
            portfolio["decimal_odds"] = decimal_odds.fillna(1.9091)

    required_columns = [
        "league",
        "away_team",
        "home_team",
        "best_pick",
        "calibrated_probability",
        "expected_value",
        "edge",
        "decimal_odds",
        "recommended_bet",
    ]
    for col in required_columns:
        if col not in portfolio.columns:
            portfolio[col] = pd.NA

    portfolio_columns = [
        "league",
        "away_team",
        "home_team",
        "best_pick",
        "market_type",
        "calibrated_probability",
        "expected_value",
        "edge",
        "decimal_odds",
        "market_probability",
        "ml_probability",
        "recommended_bet",
    ]
    available_columns = [c for c in portfolio_columns if c in portfolio.columns]
    trailing_columns = [c for c in portfolio.columns if c not in available_columns]
    portfolio = portfolio[available_columns + trailing_columns]

    if "best_pick" in portfolio.columns:
        portfolio = portfolio[portfolio["best_pick"].astype(str).str.strip().str.len() > 0].copy()

    recommended_total = portfolio["recommended_bet"].sum() if "recommended_bet" in portfolio.columns else 0.0
    if recommended_total > 0:
        portfolio["allocation_pct"] = ((portfolio["recommended_bet"] / recommended_total) * 100).round(2)
    else:
        portfolio["allocation_pct"] = 0.0
    return portfolio.sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=1000, simulations=1000)

# ---- v2 canonical TheOver + best pick pipeline overrides ----
from core.theover_loader import normalize_theover_df as loader_normalize_theover_df

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
]


def _empty_best_picks_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BEST_PICK_COLUMNS)


def normalize_theover_df(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
    return loader_normalize_theover_df(df)


def _numeric_series(df: pd.DataFrame, col: str, default: float | int | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    if default is not None:
        s = s.fillna(default)
    return s


def _string_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="string")
    if col in df.columns:
        return df[col].fillna(default).astype("string")
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def _coerce_utc_ts(value) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True)


def _with_game_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["game_key"] = pd.Series(dtype="string")
        return df
    out = df.copy()
    date_s = pd.to_datetime(_string_series(out, "game_date"), errors="coerce", utc=True)
    date_token = date_s.dt.strftime("%Y-%m-%d").fillna("")
    league = _string_series(out, "league").str.upper().str.strip()
    away = _string_series(out, "away_team").str.lower().str.strip()
    home = _string_series(out, "home_team").str.lower().str.strip()
    out["game_key"] = np.where(
        date_token != "",
        league + "|" + date_token + "|" + away + "|" + home,
        league + "|" + away + "|" + home,
    )
    return out


def is_stale_schedule(base_df: pd.DataFrame, theover_df: pd.DataFrame, now: pd.Timestamp, max_age_days: int = 14) -> bool:  # type: ignore[override]
    now_ts = _coerce_utc_ts(now)
    if pd.isna(now_ts):
        return False

    base_dates = pd.to_datetime(_string_series(base_df, "game_date"), errors="coerce", utc=True)
    theover_dates = pd.to_datetime(_string_series(theover_df, "game_date"), errors="coerce", utc=True)
    base_max = base_dates.max() if base_dates.notna().any() else pd.NaT
    upload_max = theover_dates.max() if theover_dates.notna().any() else pd.NaT

    stale_by_age = pd.notna(base_max) and base_max < (now_ts - pd.Timedelta(days=max_age_days))
    stale_vs_upload = pd.notna(base_max) and pd.notna(upload_max) and base_max < (upload_max - pd.Timedelta(days=1))
    missing_base = pd.isna(base_max) and pd.notna(upload_max)
    return bool(stale_by_age or stale_vs_upload or missing_base)


def build_theover_bet_rows(spreads_df: pd.DataFrame | None, totals_df: pd.DataFrame | None, selected_sports: list[str]) -> pd.DataFrame:  # type: ignore[override]
    cols = [
        "league", "home_team", "away_team", "game_date", "game_key", "market_type", "spread_line", "total_line",
        "theover_probability", "odds_american", "market_probability", "expected_value", "edge", "best_pick", "line", "pick", "pickteam",
    ]
    selected_norm = {_normalize_league_value(s) for s in (selected_sports or [])}

    def _prep_common(df: pd.DataFrame | None) -> pd.DataFrame:
        norm = normalize_theover_df(df)
        if norm.empty:
            return pd.DataFrame()
        out = _normalize_key_columns(norm.copy())
        out = _enrich_uploaded_league(out, list(selected_norm))
        out["league"] = _string_series(out, "league").apply(_normalize_league_value)
        out["line"] = _numeric_series(out, "line")
        out["pick"] = _string_series(out, "pick")
        out["pickteam"] = _string_series(out, "pickteam")
        out["game_date"] = pd.to_datetime(_string_series(out, "game_date"), errors="coerce", utc=True)
        wp = _numeric_series(out, "winprobability")
        out["theover_probability"] = wp.clip(0.0, 1.0)
        out["odds_american"] = _numeric_series(out, "odds_american", -110.0)
        return out

    spreads = _prep_common(spreads_df)
    if not spreads.empty:
        pickteam_norm = spreads["pickteam"].apply(normalize_team_name)
        home_norm = _string_series(spreads, "home_team").apply(normalize_team_name)
        away_norm = _string_series(spreads, "away_team").apply(normalize_team_name)
        is_home = pickteam_norm.eq(home_norm)
        is_away = pickteam_norm.eq(away_norm)
        spreads = spreads[spreads["line"].notna() & (is_home | is_away)].copy()
        spreads["spread_line"] = np.where(is_home.loc[spreads.index], spreads["line"], -spreads["line"])
        spreads["market_type"] = np.where(spreads["spread_line"] < 0, "spread_home", "spread_away")
        spreads["total_line"] = pd.NA

    totals = _prep_common(totals_df)
    if not totals.empty:
        pick_s = totals["pick"].str.lower()
        totals = totals[totals["line"].notna() & pick_s.str.contains("over|under", regex=True, na=False)].copy()
        pick_s = totals["pick"].str.lower()
        totals["market_type"] = np.where(pick_s.str.contains("over", na=False), "total_over", "total_under")
        totals["total_line"] = totals["line"]
        totals["spread_line"] = pd.NA

    bets = pd.concat([spreads, totals], ignore_index=True)
    if bets.empty:
        return pd.DataFrame(columns=cols)
    bets = _with_game_key(bets)
    if selected_norm:
        bets = bets[bets["league"].isin(selected_norm)].copy()
    bets["market_probability"] = _numeric_series(bets, "odds_american", -110.0).apply(american_to_prob)
    bets["expected_value"] = _numeric_series(bets, "expected_value")
    bets["edge"] = _numeric_series(bets, "edge")
    bets["best_pick"] = ""
    for c in cols:
        if c not in bets.columns:
            bets[c] = pd.NA
    return bets[cols]


def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
    if analysis_df is None or analysis_df.empty:
        return _empty_best_picks_df()

    df = analysis_df.copy()
    allowed = {"spread_home", "spread_away", "total_over", "total_under"}

    if "market_type" not in df.columns:
        df["market_type"] = df.apply(_infer_market_type, axis=1)
    else:
        mt = _string_series(df, "market_type", "").str.lower().str.strip()
        missing = mt.eq("") | mt.eq("nan")
        if missing.any():
            df.loc[missing, "market_type"] = df.loc[missing].apply(_infer_market_type, axis=1)

    market_type_s = _string_series(df, "market_type", "").str.lower().str.strip()
    df = df[market_type_s.isin(list(allowed))].copy()
    if df.empty:
        return _empty_best_picks_df()

    df["expected_value"] = _numeric_series(df, "expected_value")
    df["edge"] = _numeric_series(df, "edge")
    df["calibrated_probability"] = _numeric_series(df, "calibrated_probability")
    df["market_probability"] = _numeric_series(df, "market_probability")
    df["ml_probability"] = _numeric_series(df, "ml_probability")
    df["odds_american"] = _numeric_series(df, "odds_american", -110.0)
    df["decimal_odds"] = _numeric_series(df, "decimal_odds")
    df["game_date"] = pd.to_datetime(_string_series(df, "game_date"), errors="coerce", utc=True)

    df = df.sort_values(["expected_value", "edge"], ascending=[False, False])
    grp = [c for c in ["league", "home_team", "away_team", "game_date"] if c in df.columns]
    if not grp:
        return _empty_best_picks_df()

    best = df.groupby(grp, dropna=False).first().reset_index()
    best = _ensure_best_pick_column(best)
    best_pick_s = _string_series(best, "best_pick", "").str.strip()
    best = best[best_pick_s.str.len() > 0].copy()
    if best.empty:
        return _empty_best_picks_df()

    for c in BEST_PICK_COLUMNS:
        if c not in best.columns:
            best[c] = pd.NA
    return best[BEST_PICK_COLUMNS]


@st.cache_data(ttl=180)
def run_analysis_pipeline(  # type: ignore[override]
    sports: Iterable[str],
    max_rows: int,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str | None]]:
    now_utc = _coerce_utc_ts(pd.Timestamp.utcnow())
    selected_sports = _normalize_sports_filter(sports)
    base_df = load_base_data().copy()
    if "game_date" in base_df.columns:
        base_df["game_date"] = pd.to_datetime(_string_series(base_df, "game_date"), errors="coerce", utc=True)
    if selected_sports and "league" in base_df.columns:
        base_df = base_df[base_df["league"].isin(selected_sports)].copy()

    spreads_norm = normalize_theover_df(spreads_df)
    totals_norm = normalize_theover_df(totals_df)
    spreads_games_df = _with_game_key(_normalize_key_columns(spreads_norm.copy())) if not spreads_norm.empty else pd.DataFrame(columns=["game_key"])
    totals_games_df = _with_game_key(_normalize_key_columns(totals_norm.copy())) if not totals_norm.empty else pd.DataFrame(columns=["game_key"])

    bet_rows_df = build_theover_bet_rows(spreads_df, totals_df, selected_sports)
    has_normalized_bet_rows = not bet_rows_df.empty
    base_stale = is_stale_schedule(base_df, bet_rows_df, now_utc) if has_normalized_bet_rows else False
    base_dates = pd.to_datetime(_string_series(base_df, "game_date"), errors="coerce", utc=True)
    upload_dates = pd.to_datetime(_string_series(bet_rows_df, "game_date"), errors="coerce", utc=True)
    base_max_date = base_dates.max() if base_dates.notna().any() else pd.NaT
    theover_max_date = upload_dates.max() if upload_dates.notna().any() else pd.NaT

    merge_keys_used: list[str] = []
    if not bet_rows_df.empty:
        analysis_input = bet_rows_df.head(max_rows).copy()
        if (not base_stale) and (not base_df.empty):
            merge_keys = choose_merge_keys(analysis_input, base_df)
            merge_keys_used = list(merge_keys)
            enrich_cols = [c for c in base_df.columns if c not in analysis_input.columns and c not in merge_keys]
            if merge_keys and enrich_cols:
                analysis_input = analysis_input.merge(base_df[merge_keys + enrich_cols], on=merge_keys, how="left")
    else:
        analysis_input = base_df.head(max_rows).copy()

    analyzed = _apply_analysis_calculations(analysis_input)

    if not analyzed.empty:
        if "market_type" not in analyzed.columns:
            analyzed["market_type"] = analyzed.apply(infer_market_type_from_row, axis=1)
        else:
            mt = _string_series(analyzed, "market_type", "").str.lower().str.strip()
            missing_mt = mt.eq("") | mt.eq("nan")
            if missing_mt.any():
                analyzed.loc[missing_mt, "market_type"] = analyzed.loc[missing_mt].apply(infer_market_type_from_row, axis=1)

    if analyzed.empty:
        empty = _empty_best_picks_df()
        di = {
            "total_games": 0,
            "bet_rows": 0,
            "best_picks": 0,
            "theover_spreads_rows": int(len(spreads_norm)),
            "theover_totals_rows": int(len(totals_norm)),
            "theover_spreads_games": int(spreads_games_df["game_key"].nunique() if "game_key" in spreads_games_df else 0),
            "theover_totals_games": int(totals_games_df["game_key"].nunique() if "game_key" in totals_games_df else 0),
            "theover_spreads_bet_games": 0,
            "theover_totals_bet_games": 0,
            "market_type_counts": {},
            "base_max_date": None if pd.isna(base_max_date) else str(base_max_date),
            "theover_max_date": None if pd.isna(theover_max_date) else str(theover_max_date),
            "bet_rows_max_date": None if pd.isna(theover_max_date) else str(theover_max_date),
            "allowed_market_type_rows": 0,
            "positive_ev_rows": 0,
            "best_pick_nonempty_rows": 0,
            "now_utc": None if pd.isna(now_utc) else str(now_utc),
            "base_stale": bool(base_stale),
            "has_normalized_bet_rows": bool(has_normalized_bet_rows),
            "analysis_source": "uploads" if has_normalized_bet_rows else "base_schedule",
            "merge_keys_used": merge_keys_used,
            "kalshi_matches": 0,
            "kalshi_match_rate": 0.0,
        }
        return analyzed, empty, di

    analyzed["model_probability"] = _numeric_series(analyzed, "theover_probability").fillna(
        _numeric_series(analyzed, "model_probability")
    ).fillna(_numeric_series(analyzed, "market_probability")).fillna(0.5).clip(0.01, 0.99)
    analyzed = calibrate_probabilities(analyzed)
    analyzed["calibrated_probability"] = _numeric_series(analyzed, "calibrated_probability").fillna(analyzed["model_probability"]).clip(0.01, 0.99)
    analyzed["odds_american"] = _numeric_series(analyzed, "odds_american", -110.0)
    analyzed["market_probability"] = _numeric_series(analyzed, "market_probability").fillna(analyzed["odds_american"].apply(american_to_prob))
    analyzed["decimal_odds"] = analyzed["odds_american"].apply(american_to_decimal)
    analyzed["expected_value"] = analyzed["calibrated_probability"] * (analyzed["decimal_odds"] - 1) - (1 - analyzed["calibrated_probability"])
    analyzed["edge"] = analyzed["calibrated_probability"] - analyzed["market_probability"]

    try:
        analyzed = enrich_with_kalshi_markets(analyzed)
    except Exception as exc:
        logger.warning("Kalshi enrichment failed in pipeline: %s", exc)
        for c, d in {
            "kalshi_probability": pd.NA,
            "kalshi_market_title": pd.NA,
            "kalshi_event_ticker": pd.NA,
            "kalshi_market_ticker": pd.NA,
            "kalshi_line": pd.NA,
            "kalshi_match_status": "api_error",
            "kalshi_match_reason": "pipeline_exception",
            "kalshi_tried_tickers": "",
        }.items():
            if c not in analyzed.columns:
                analyzed[c] = d

    best_picks_df = build_best_picks_df(analyzed)

    market_type_s = _string_series(analyzed, "market_type", "").str.lower().str.strip()
    allowed_mask = market_type_s.isin(["spread_home", "spread_away", "total_over", "total_under"])
    positive_ev_rows = int(_numeric_series(analyzed, "expected_value").gt(0).sum())
    best_pick_nonempty_rows = 0
    if not best_picks_df.empty:
        best_pick_nonempty_rows = int(_string_series(best_picks_df, "best_pick", "").str.strip().str.len().gt(0).sum())

    spread_games = 0
    total_games = 0
    if not bet_rows_df.empty and "game_key" in bet_rows_df.columns:
        bet_market = _string_series(bet_rows_df, "market_type", "").str.lower()
        spread_games = int(bet_rows_df[bet_market.str.startswith("spread", na=False)]["game_key"].nunique())
        total_games = int(bet_rows_df[bet_market.str.startswith("total", na=False)]["game_key"].nunique())

    di = {
        "total_games": int(analyzed[["league", "home_team", "away_team", "game_date"]].drop_duplicates().shape[0]) if set(["league", "home_team", "away_team", "game_date"]).issubset(analyzed.columns) else int(len(analyzed)),
        "bet_rows": int(len(analyzed)),
        "best_picks": int(len(best_picks_df)),
        "theover_spreads_rows": int(len(spreads_norm)),
        "theover_totals_rows": int(len(totals_norm)),
        "theover_spreads_games": int(spreads_games_df["game_key"].nunique() if "game_key" in spreads_games_df else 0),
        "theover_totals_games": int(totals_games_df["game_key"].nunique() if "game_key" in totals_games_df else 0),
        "theover_spreads_bet_games": spread_games,
        "theover_totals_bet_games": total_games,
        "market_type_counts": market_type_s.value_counts(dropna=False).to_dict(),
        "base_max_date": None if pd.isna(base_max_date) else str(base_max_date),
        "theover_max_date": None if pd.isna(theover_max_date) else str(theover_max_date),
        "bet_rows_max_date": None if pd.isna(theover_max_date) else str(theover_max_date),
        "allowed_market_type_rows": int(allowed_mask.sum()),
        "positive_ev_rows": positive_ev_rows,
        "best_pick_nonempty_rows": best_pick_nonempty_rows,
        "now_utc": None if pd.isna(now_utc) else str(now_utc),
        "base_stale": bool(base_stale),
        "has_normalized_bet_rows": bool(has_normalized_bet_rows),
        "analysis_source": "uploads" if has_normalized_bet_rows else "base_schedule",
        "merge_keys_used": merge_keys_used,
        "kalshi_matches": int(_string_series(analyzed, "kalshi_match_status", "").str.lower().eq("matched").sum()),
        "kalshi_match_rate": float(_string_series(analyzed, "kalshi_match_status", "").str.lower().eq("matched").sum() / max(len(analyzed), 1)),
    }
    return analyzed, best_picks_df, di
