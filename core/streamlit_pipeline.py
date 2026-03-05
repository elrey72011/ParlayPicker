from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

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


@st.cache_data(ttl=300)
def load_base_data() -> pd.DataFrame:
    df = pd.read_csv("data/master_all_sports.csv")
    return df


@st.cache_data(ttl=180)
def run_analysis_pipeline(sports: Iterable[str], max_rows: int) -> pd.DataFrame:
    base_df = load_base_data()
    filtered = base_df[base_df["league"].isin(list(sports))].copy() if sports else base_df.copy()
    filtered = filtered.head(max_rows)

    if run_ml_predictions and run_master_analysis and not filtered.empty:
        ml_df = run_ml_predictions(filtered)
        return run_master_analysis(filtered, ml_df, filtered)
    return filtered


def generate_parlays_table(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

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
