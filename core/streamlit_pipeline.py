from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
import streamlit as st

from core.ev_engine import calculate_ev
from core.probability_engine import american_odds_to_probability
from core.schema.base_schema import ensure_base_schema
from core.team_normalizer import normalize_team

logger = logging.getLogger(__name__)

try:
    from complete_workflow_implementation import (
        build_optimal_parlays,
        compute_best_bets,
        run_master_analysis,
        run_ml_predictions,
    )
except Exception:  # pragma: no cover
    build_optimal_parlays = None
    compute_best_bets = None
    run_master_analysis = None
    run_ml_predictions = None




def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df.loc[:, col] = df[col].apply(normalize_team)
    return df


def _resolve_american_odds(row: pd.Series) -> float:
    for col in ["odds_american", "home_odds", "odds"]:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                continue
    return -110.0


def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = ensure_base_schema(df)

    df["ai_probability"] = pd.to_numeric(df["ai_probability"], errors="coerce").fillna(0.5)
    df["ml_probability"] = pd.to_numeric(df["ml_probability"], errors="coerce").fillna(0.5)

    df["odds_american"] = df.apply(_resolve_american_odds, axis=1)
    df["market_probability"] = df["odds_american"].apply(american_odds_to_probability)

    df["consensus_prob"] = (
        df["ai_probability"].fillna(0.5)
        + df["ml_probability"].fillna(0.5)
        + df["market_probability"].fillna(0.5)
    ) / 3

    df["expected_value"] = df.apply(
        lambda row: calculate_ev(float(row["consensus_prob"]), float(row["odds_american"])),
        axis=1,
    )

    if "spread_edge" not in df.columns:
        df["spread_edge"] = 0.0
    if "total_edge" not in df.columns:
        df["total_edge"] = 0.0

    df["best_pick"] = df.apply(determine_best_pick, axis=1)
    return df


def determine_best_pick(row: pd.Series) -> str:
    if row["expected_value"] <= 0:
        return "No Edge"

    return f"{row['away_team']} vs {row['home_team']}"

@st.cache_data(ttl=300)
def load_base_data() -> pd.DataFrame:
    df = pd.read_csv("data/master_all_sports.csv")
    return _normalize_teams(df)


def detect_league_column(df: pd.DataFrame) -> str | None:
    possible_names = ["league", "sport", "sport_key", "league_name", "league_id"]

    for col in possible_names:
        if col in df.columns:
            return col

    return None


@st.cache_data(ttl=180)
def run_analysis_pipeline(
    sports: Iterable[str],
    max_rows: int,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base_df = load_base_data()
    base_df.columns = base_df.columns.str.strip().str.lower()

    required_columns = ["home_team", "away_team"]
    for col in required_columns:
        if col not in base_df.columns:
            raise ValueError(f"Required column missing: {col}")

    league_col = detect_league_column(base_df)
    logger.info("Available columns: %s", list(base_df.columns))
    logger.info("Detected league column: %s", league_col)

    if sports and league_col:
        filtered = base_df[base_df[league_col].isin(list(sports))].copy()
    else:
        if sports and not league_col:
            logger.warning("League column not found. Skipping sports filter.")
        filtered = base_df.copy()

    filtered = _normalize_teams(filtered.head(max_rows))

    if spreads_df is not None:
        filtered["theover_spreads_uploaded"] = True
        filtered["theover_spreads_rows"] = len(spreads_df)
    else:
        filtered["theover_spreads_uploaded"] = False

    if totals_df is not None:
        filtered["theover_totals_uploaded"] = True
        filtered["theover_totals_rows"] = len(totals_df)
    else:
        filtered["theover_totals_uploaded"] = False

    if use_ml and run_ml_predictions and run_master_analysis and not filtered.empty:
        ml_df = run_ml_predictions(filtered)
        analyzed = run_master_analysis(filtered, ml_df, filtered)
        if analyzed is None or analyzed.empty:
            return _apply_analysis_calculations(filtered)

        if "ml_probability_x" in analyzed.columns:
            analyzed["ml_probability"] = analyzed["ml_probability_x"]
        if "ml_probability_y" in analyzed.columns:
            analyzed["ml_probability"] = analyzed["ml_probability_y"]
        analyzed.drop(
            columns=["ml_probability_x", "ml_probability_y"],
            errors="ignore",
            inplace=True,
        )
        return _apply_analysis_calculations(_normalize_teams(analyzed))
    return _apply_analysis_calculations(filtered)


def generate_parlays(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    parlay_candidates = analysis_df.attrs.get("parlay_candidates")
    if isinstance(parlay_candidates, pd.DataFrame) and not parlay_candidates.empty:
        return parlay_candidates

    if "expected_value" in analysis_df.columns:
        parlay_candidates = analysis_df[analysis_df["expected_value"] > 0.02].copy()
        if not parlay_candidates.empty:
            parlay_candidates.sort_values("expected_value", ascending=False, inplace=True)
            return parlay_candidates

    if compute_best_bets and build_optimal_parlays:
        best_bets = compute_best_bets(analysis_df)
        parlays = build_optimal_parlays(best_bets, parlay_sizes=[2, 3], max_per_size=8, check_correlation=True)
        rows = []
        for size, entries in parlays.items():
            for entry in entries:
                rows.append(
                    {
                        "parlay_size": size,
                        "probability": entry.get("probability"),
                        "odds": entry.get("odds"),
                        "expected_value": entry.get("expected_value"),
                    }
                )
        return pd.DataFrame(rows)

    return analysis_df.head(10)


# Backward compatible alias
generate_parlays_table = generate_parlays


def build_realtime_edges(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    edge_cols = [c for c in ["league", "Home", "Away", "consensus_prob", "expected_value"] if c in analysis_df.columns]
    if edge_cols:
        return analysis_df[edge_cols].sort_values(edge_cols[-1], ascending=False).head(25)

    return analysis_df.head(25)


def optimize_portfolio_allocation(analysis_df: pd.DataFrame) -> pd.DataFrame:
    edges = build_realtime_edges(analysis_df)
    if edges.empty:
        return edges

    portfolio = edges.copy()
    if "expected_value" in portfolio.columns:
        ev_abs = portfolio["expected_value"].abs().replace(0, 1)
        portfolio["allocation_pct"] = ((ev_abs / ev_abs.sum()) * 100).round(2)
    else:
        portfolio["allocation_pct"] = round(100 / max(len(portfolio), 1), 2)
    return portfolio
