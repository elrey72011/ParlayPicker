from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
import streamlit as st

from core.ev_engine import calculate_ev
from core.parlay_engine import generate_parlays as generate_parlay_candidates
from core.probability_engine import american_to_prob, normalize_probability_components, remove_vig
from core.schema.base_schema import ensure_base_schema
from core.team_mapper import normalize_team_name

logger = logging.getLogger(__name__)


try:
    from complete_workflow_implementation import run_ml_predictions
except Exception:  # pragma: no cover
    run_ml_predictions = None


MERGE_KEYS = ["league", "home_team", "away_team", "game_date"]


def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["home_team", "away_team", "team"]:
        if col in df.columns:
            df.loc[:, col] = df[col].apply(normalize_team_name)
    return df


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
    return _normalize_teams(df)


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
    right = _normalize_key_columns(right)
    shared_keys = [k for k in MERGE_KEYS if k in left.columns and k in right.columns]
    if not shared_keys:
        return left
    return left.merge(right, on=shared_keys, how="left", suffixes=("", suffix))


def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = ensure_base_schema(df)

    df["odds_american"] = df.apply(_resolve_american_odds, axis=1)

    df["market_prob"] = df["odds_american"].apply(american_to_prob)
    if {"home_odds", "away_odds"}.issubset(df.columns):
        home_prob = pd.to_numeric(df["home_odds"], errors="coerce").apply(lambda x: american_to_prob(x) if pd.notna(x) else pd.NA)
        away_prob = pd.to_numeric(df["away_odds"], errors="coerce").apply(lambda x: american_to_prob(x) if pd.notna(x) else pd.NA)
        no_vig = pd.DataFrame([remove_vig(h, a) if pd.notna(h) and pd.notna(a) else (pd.NA, pd.NA) for h, a in zip(home_prob, away_prob)])
        df["market_prob"] = no_vig[0].fillna(df["market_prob"])

    df["ml_prob"] = pd.to_numeric(df.get("ml_probability", 0.5), errors="coerce")
    df["ai_prob"] = pd.to_numeric(df.get("ai_probability", 0.5), errors="coerce")
    df["kalshi_prob"] = pd.to_numeric(df.get("kalshi_probability", pd.NA), errors="coerce")

    df = normalize_probability_components(df)
    df["market_probability"] = df["market_prob"]

    df["expected_value"] = calculate_ev(df["consensus_prob"], df["odds_american"])

    if "team" not in df.columns:
        df["team"] = df.get("away_team", "")

    df["best_pick"] = df.apply(
        lambda r: f"{r['away_team']} vs {r['home_team']}" if r["expected_value"] > 0 else "No Edge",
        axis=1,
    )
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

    if sports and "league" in base_df.columns:
        filtered = base_df[base_df["league"].isin(list(sports))].copy()
    else:
        filtered = base_df.copy()

    filtered = filtered.head(max_rows)

    if use_ml and run_ml_predictions and not filtered.empty:
        ml_df = _normalize_key_columns(run_ml_predictions(filtered))
        filtered = _safe_merge(filtered, ml_df, "_ml")

    merged_theover = None
    if spreads_df is not None and not spreads_df.empty:
        merged_theover = _normalize_key_columns(spreads_df)
    if totals_df is not None and not totals_df.empty:
        totals_norm = _normalize_key_columns(totals_df)
        merged_theover = pd.concat([merged_theover, totals_norm], ignore_index=True) if merged_theover is not None else totals_norm

    filtered = _safe_merge(filtered, merged_theover, "_theover")

    return _apply_analysis_calculations(filtered)


def generate_parlays(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()
    return generate_parlay_candidates(analysis_df)


def build_realtime_edges(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    edge_cols = [c for c in ["league", "home_team", "away_team", "consensus_prob", "expected_value"] if c in analysis_df.columns]
    if edge_cols:
        return analysis_df[edge_cols].sort_values("expected_value", ascending=False).head(25)

    return analysis_df.head(25)


def optimize_portfolio_allocation(analysis_df: pd.DataFrame) -> pd.DataFrame:
    edges = build_realtime_edges(analysis_df)
    if edges.empty:
        return edges

    portfolio = edges.copy()
    ev_abs = portfolio["expected_value"].abs().replace(0, 1)
    portfolio["allocation_pct"] = ((ev_abs / ev_abs.sum()) * 100).round(2)
    return portfolio
